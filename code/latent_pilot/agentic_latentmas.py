"""Agentic LatentMAS: text ReAct for tool work, latent for communication.

Uses the official LatentMAS repo's ModelWrapper for both text generation
and latent thought generation. Only the inter-agent communication step
differs between conditions.

Architecture (matches user's decentralized protocol):
  Round 1: Each agent does text ReAct (tool calls) — IDENTICAL for both
  Communication:
    TEXT: agents share text responses as prompt text
    LATENT: agents do m latent steps → share KV working memory
  Round 2+: Each agent does text ReAct conditioned on shared info
  Final: synthesizer produces answer

Requires:
  - LatentMAS repo cloned at $REPO_DIR (for ModelWrapper, latent generation)
  - QAMPARI data at $QAMPARI_DIR
  - vllm installed (LatentMAS dependency)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add LatentMAS repo to path
REPO_DIR = os.environ.get("LATENTMAS_REPO",
    str(Path(f"/atlas2/u/{os.environ['USER']}/mas_project/LatentMAS")))
sys.path.insert(0, REPO_DIR)

# Add mas-energy code to path
CODE_DIR = os.environ.get("MAS_ENERGY_CODE",
    str(Path(f"/atlas2/u/{os.environ['USER']}/mas_project/mas-energy/code")))
sys.path.insert(0, CODE_DIR)

# The LatentMAS repo's ModelWrapper was never actually instantiated by this
# module — only used as a type annotation. Runtime uses SimpleModelWrapper
# (an inline class in main()) which only needs torch + transformers.
# Try importing ModelWrapper for type-correctness; fall back to a sentinel
# string if the LatentMAS repo isn't on PYTHONPATH (e.g. running in
# mas-energy env, which has SGLang + working torch but no LatentMAS install).
try:
    from models import ModelWrapper  # from LatentMAS repo
except ImportError:
    ModelWrapper = "ModelWrapper"  # forward-ref sentinel; never instantiated

# Benchmark imports — loaded dynamically based on --benchmark flag
def load_benchmark(name, data_dir=None):
    """Load benchmark class and evaluation function by name."""
    if name == "qampari":
        from benchmarks_qampari import QampariBenchmark, evaluate_qampari
        bench = QampariBenchmark(data_dir=data_dir)
        def evaluate(task, answer):
            result = evaluate_qampari(task["answer_list"], answer)
            return result["recall_substr"], result["f1"], result
        return bench, evaluate, "question_text"
    elif name == "fanoutqa":
        from benchmarks_fanoutqa import FanOutQABenchmark, evaluate_answer
        bench = FanOutQABenchmark()
        def evaluate(task, answer):
            is_correct, loose_acc = evaluate_answer(task, answer)
            return loose_acc, loose_acc, {"correct": is_correct, "loose_accuracy": loose_acc}
        return bench, evaluate, "question"
    else:
        raise ValueError(f"Unknown benchmark: {name}. Choose from: qampari, fanoutqa")


# Prompts copied verbatim from mas-energy/code/prompts.py (main study) to
# ensure exact parity. TOOL_FORMAT_INSTRUCTION is appended to system prompts
# because HF transformers doesn't do native tool calling like SGLang; the
# model needs to be told the emission format. Format spec only — no
# directive language that would diverge from the main study prompts.
SAS_PROMPT = (
    "You are a helpful assistant with access to tools. Use the provided "
    "tools to complete the user's task. When you have gathered enough "
    "information or completed the required actions, provide your final "
    "answer directly."
)

DEBATE_AGENT_PROMPT = (
    "You are a helpful assistant with access to tools, participating in a "
    "collaborative problem-solving process. Use tools to investigate the "
    "task thoroughly. Provide your reasoning and final answer clearly."
)

DEBATE_SYNTHESIZER_PROMPT = (
    "You are a synthesis agent. Given multiple agents' responses after "
    "debate, synthesize the best final answer based on all agents' work."
)


def build_sas_prompt(question: str) -> list[dict]:
    """Build chat messages for the single-agent (SAS) condition. Uses the
    main study's SAS_PROMPT, with tool-format spec appended since HF
    transformers requires explicit instruction on how to emit tool calls.
    """
    return [
        {"role": "system", "content": f"{SAS_PROMPT}\n\n{TOOL_FORMAT_INSTRUCTION}"},
        {"role": "user", "content": question},
    ]


def build_react_prompt(question: str, tool_instruction: str = None) -> list[dict]:
    """Build chat messages for a debate agent (Phase 1/2 of decentralized
    or latent topologies). Uses the main study's DEBATE_AGENT_PROMPT.

    NOTE 2026-06-04: history of this function:
    - Originally appended QAMPARI-style TOOL_FORMAT_INSTRUCTION
      ("BM25 passages from the corpus") to system prompt unconditionally.
    - 2026-06-03 morning: removed the instruction entirely, hoping the
      chat template's native tool schema would suffice. Wrong: HF
      transformers has no `tool_choice="auto"` equivalent, so the model
      defaulted to answering from parametric knowledge instead of
      searching. text MAS F1 stayed at 0.506 (no improvement).
    - 2026-06-04: re-added the instruction, but with benchmark-agnostic
      wording (TOOL_FORMAT_INSTRUCTION below). The chat template injects
      the tool *schema*, this instruction provides the prose nudge that
      SGLang's tool_choice param provides in the main study. Combined,
      they should match SGLang's prompt-the-model-to-use-tools behavior.
    The `tool_instruction` arg is preserved for call-site compat but ignored.
    """
    return [
        {"role": "system",
         "content": f"{DEBATE_AGENT_PROMPT}\n\n{TOOL_FORMAT_INSTRUCTION}"},
        {"role": "user", "content": question},
    ]


class _CachingExecutor:
    """Per-task memoization wrapper for tool execution.

    In MAS runs, multiple parallel agents often issue near-identical queries
    (QAMPARI pilot measured 72.7% exact-duplicate rate). Caching tool
    results avoids redundant retrieval work AND, because the identical tool
    result text will now be appended verbatim to each agent's context,
    lets SGLang/vLLM prefix caching reuse more of the encoded KV. HF
    transformers doesn't auto-cache prefixes, but the tool-execution cost
    itself is reduced; the bigger savings are whatever serving engine uses.
    """
    def __init__(self, inner):
        self._inner = inner
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, name, args):
        key = self._key(name, args)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = self._inner(name, args)
        self._cache[key] = result
        return result

    @staticmethod
    def _key(name, args):
        if isinstance(args, dict):
            q = args.get("query", "")
            if isinstance(q, str):
                return f"{name}|{q.strip().lower()}"
        return f"{name}|{repr(args)}"


def generate_text(model_wrapper, messages, temperature: float = 0.0,
                  max_new_tokens: int = 4096,
                  constrain_list: bool = False) -> str:
    """Plain chat generation with no ReAct/tool loop. For synthesis steps
    that should integrate prior agent findings rather than issue more
    tool calls.

    If constrain_list=True, generation stops as soon as a paragraph-break
    token pattern is emitted. This is a lightweight approximation of
    grammar-constrained decoding: it doesn't forbid prose tokens per se,
    but halts decoding after the model completes its first list/sentence
    cluster. On QAMPARI-style tasks this typically truncates verbose
    explanatory prose that follows the entity list, cutting decode energy
    without affecting the list content itself.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_tok)
        if _tid is not None and _tid != tokenizer.unk_token_id and _tid not in eos_ids:
            eos_ids.append(_tid)
    gen_kwargs = dict(
        input_ids=enc.input_ids.to(dev),
        attention_mask=enc.attention_mask.to(dev),
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=0.95 if temperature > 0 else 1.0,
        eos_token_id=eos_ids if eos_ids else None,
        pad_token_id=(tokenizer.pad_token_id
                      if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
    )
    if constrain_list:
        # Stop at paragraph-break / prose-introduction markers. The model
        # is instructed to output a list; once it's done emitting the list
        # and moves to prose explanation ("\n\nNote:", "\n\nExplanation:",
        # "\n\nThese are..."), halt. Short strings match across Qwen3 tokens.
        gen_kwargs["stop_strings"] = ["\n\n", ". These", ". Note", ". Explanation"]
        gen_kwargs["tokenizer"] = tokenizer  # required for stop_strings
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    response = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    import re as _re
    response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>", 1)[0]
    return response.strip()


# Format spec only — no directive ("MUST use"/"Do NOT"/"multiple searches")
# that would diverge from main study's prompts. The main study's prompts
# handle behavior; tool format is just a technical necessity for HF.
# Tool-format instruction (NEW 2026-06-04, replaces QAMPARI-specific version).
# Why this exists: the main study uses SGLang with `tool_choice="auto"` on every
# chat.completions.create() call, which actively prompts the model to either
# emit a tool call OR a final answer. HF transformers has no equivalent
# parameter — apply_chat_template(tools=...) only injects the tool schema
# *passively*. Without explicit prose telling the model "you should call
# these tools", Qwen3.5-9B reliably defaults to answering from parametric
# knowledge instead of searching, producing F1 ~= chance on FanOutQA.
#
# The instruction is benchmark-agnostic — it references "the tools shown
# above" (which the chat template will have injected), not specific tool
# names or descriptions, so it works equally for FanOutQA Wikipedia search,
# QAMPARI BM25 corpus, BrowseComp dense index, etc.
TOOL_FORMAT_INSTRUCTION = (
    "Your training data may be outdated or incomplete. For factual questions, "
    "always investigate using the tools shown above before answering. "
    "Emit tool calls as <tool_call>{...}</tool_call> blocks containing JSON "
    "with the tool name and arguments. Use multiple tool calls if needed to "
    "gather complete evidence. Only after receiving tool results should you "
    "synthesize a final answer."
)

# Backwards-compatibility alias; existing call sites pass this as a second
# arg to build_react_prompt, which now uses it again (with the corrected
# benchmark-agnostic wording above).
TOOL_INSTRUCTION = TOOL_FORMAT_INSTRUCTION


def detect_tool_call(text: str):
    """Detect tool calls in Qwen3 output. Supports three formats:

    1. JSON format (Qwen3 chat template default):
       <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>

    2. XML format (qwen3_coder parser / SGLang):
       <tool_call><function=search><parameter=query>...</parameter></function></tool_call>

    3. Python-style fallback:
       search("query text")
    """
    import re
    # Native-tool-calling assistant turns carry content=None when they only
    # emit tool_calls (per OpenAI message format). _extract_tool_summary
    # iterates over messages and feeds each .content to this function, so
    # we have to handle None / non-string up front.
    if not isinstance(text, str) or not text:
        return None
    # Strip <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)

    # Format 1: JSON inside <tool_call> tags
    tc_json = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if tc_json:
        try:
            obj = json.loads(tc_json.group(1))
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return (obj.get("name", ""), args)
        except json.JSONDecodeError:
            pass

    # Format 1b: Partial JSON match (incomplete closing tag)
    tc_partial = re.search(r'<tool_call>\s*\{[^}]*"name"\s*:\s*"(\w+)"[^}]*"query"\s*:\s*"([^"]+)"', text)
    if tc_partial:
        return (tc_partial.group(1), {"query": tc_partial.group(2)})

    # Format 2: XML format (qwen3_coder / SGLang style)
    tc_xml = re.search(r'<tool_call>\s*<function=(\w+)>(.*?)</function>', text, re.DOTALL)
    if tc_xml:
        func_name = tc_xml.group(1)
        params_text = tc_xml.group(2)
        params = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', params_text, re.DOTALL):
            params[pm.group(1)] = pm.group(2).strip()
        return (func_name, params)

    # Format 2b: Partial XML (function tag without closing)
    tc_xml_partial = re.search(r'<function=(\w+)>.*?<parameter=(\w+)>(.*?)</parameter>', text, re.DOTALL)
    if tc_xml_partial:
        return (tc_xml_partial.group(1), {tc_xml_partial.group(2): tc_xml_partial.group(3).strip()})

    # Format 3: Python-style fallback
    py = re.search(r'search\s*\(\s*(?:query\s*=\s*)?["\']([^"\']+)["\']', text)
    if py:
        return ("search", {"query": py.group(1)})

    return None


# ────────────────────────────────────────────────────────────────────
# Message-level soft truncation — ported byte-for-byte from main study's
# `llm._truncate_messages`. Drops oldest tool_calling rounds when prompt
# exceeds budget; preserves system + first user + most recent messages.
# Identical strategy to what SGLang sees in the main study so the pilot's
# Phase 2 inputs are structurally equivalent at all input sizes.
# ────────────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 3  # matches main study llm.py:41


def _estimate_msg_tokens(messages):
    """Rough token count, matches main study llm._estimate_tokens exactly."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content) // _CHARS_PER_TOKEN + 4
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", "")) // _CHARS_PER_TOKEN + 10
    return total


def _soft_truncate_messages(messages, max_context_tokens=98304, max_generation=4096):
    """Port of main study's llm._truncate_messages. Reserves 5% safety margin
    beyond max_generation; effective prompt budget = int(ctx * 0.95) - gen.

    With defaults (ctx=98304, gen=4096): budget = 89292 — identical to main
    study's effective input limit on its SGLang server.

    Preserves: system message, first user message, most recent messages.
    Drops oldest assistant(tool_calls) + tool result groups from the middle.
    """
    available = int(max_context_tokens * 0.95) - max_generation
    if _estimate_msg_tokens(messages) <= available:
        return messages

    # Parse into segments: tool_round groups and other singletons.
    segments = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            segments.append(("tool_round", group))
            i = j
        else:
            segments.append(("other", [msg]))
            i += 1

    prefix_count = min(2, len(segments))
    prefix_segs = segments[:prefix_count]
    rest_segs = list(segments[prefix_count:])

    while rest_segs:
        flat = [m for _, seg_msgs in prefix_segs + rest_segs for m in seg_msgs]
        if _estimate_msg_tokens(flat) <= available:
            break
        # Drop the oldest tool_round from rest
        dropped = False
        for idx, (seg_type, _) in enumerate(rest_segs):
            if seg_type == "tool_round":
                rest_segs.pop(idx)
                dropped = True
                break
        if not dropped:
            rest_segs.pop(0)

    return [m for _, seg_msgs in prefix_segs + rest_segs for m in seg_msgs]


def text_react_loop(model_wrapper: ModelWrapper, messages: list[dict],
                    executor, max_steps: int = 5, temperature: float = 0.5,
                    skip_final_response: bool = False,
                    capture_kv: bool = False,
                    tools: list[dict] | None = None):
    """Run a text ReAct loop using LatentMAS's ModelWrapper.

    Mutates `messages` in place (appends tool calls, results, responses).

    Args:
        tools: if provided, use the model's native chat template with tools=
            and parse <tool_call>...</tool_call> blocks from the response.
            Tool results are appended as proper {role: tool, ...} messages.
            This matches the main study's `llm.react_loop` protocol.
            If None, falls back to the legacy regex-on-text + fake user-role
            "Tool result: ... Continue." pattern (kept for backward compat
            with non-tool callers like the synthesis pass).

    Returns:
      - str (response text) when capture_kv=False
      - (str, kv_cache) when capture_kv=True — kv_cache from the last
        generate call, containing the full conversation. Eliminates the
        need for build_working_memory to re-encode from scratch.

    skip_final_response: if True, return "" after all tool calls are done
    instead of generating the final text response. Used for intermediate
    agents in latent mode — their "response" is latent steps, not text.
    """
    tokenizer = model_wrapper.tokenizer
    last_kv = None
    use_native = tools is not None
    valid_tool_names = ({t["function"]["name"] for t in tools}
                        if use_native else set())
    # Native-tool parsing (matches main study and SGLang qwen3_coder parser).
    if use_native:
        from latent_pilot.native_react import (
            _parse_tool_calls, _content_outside_tool_calls,
        )

    # Truncation cap for ReAct prompt — matched EXACTLY to the main study:
    # SGLang launches with --context-length 98304, and llm._truncate_messages
    # reserves 95% of context minus MAX_TOKENS (=4096) → effective prompt cap
    # of 89292 tokens. This makes the pilot text_react_loop's prompt budget
    # identical to the main study's react_loop. (Earlier hard limit of 4096
    # was the dominant cause of the pilot's 0.07 F1 underperformance vs main
    # study on FanOutQA debate rounds where Phase 2 inputs exceed 4k tokens.)
    REACT_PROMPT_TOKEN_CAP = 89292

    def _tokenize_left_trunc(prompt_str):
        """Tokenize and left-truncate (drop OLDEST tokens) so the generation
        prompt at the end of the chat template is always preserved. Default
        HF truncation drops from the right, which would silently chop the
        generation prompt — never what we want for ReAct."""
        enc = tokenizer(prompt_str, return_tensors="pt", truncation=False)
        if enc.input_ids.size(1) > REACT_PROMPT_TOKEN_CAP:
            enc["input_ids"] = enc.input_ids[:, -REACT_PROMPT_TOKEN_CAP:]
            enc["attention_mask"] = enc.attention_mask[:, -REACT_PROMPT_TOKEN_CAP:]
        return enc

    for step in range(max_steps):
        tmpl_kwargs = dict(tokenize=False, add_generation_prompt=True,
                           enable_thinking=False)
        if use_native:
            tmpl_kwargs["tools"] = tools
        # Spec-matched truncation: apply message-level soft truncation FIRST
        # (mirrors main study's `_truncate_messages` call before chat()).
        # This preserves system + first user + most recent messages by dropping
        # oldest tool_call rounds — identical strategy to main study.
        # The token-level _tokenize_left_trunc is a safety net for the residual.
        messages = _soft_truncate_messages(messages,
                                           max_context_tokens=98304,
                                           max_generation=4096)
        prompt = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
        enc = _tokenize_left_trunc(prompt)
        input_ids = enc.input_ids.to(model_wrapper.model.device)

        # Build EOS token list. For Qwen3.5 with native thinking-mode prefix,
        # the model's normal eos_token isn't enough — it can emit <|im_end|>
        # without setting eos_token_id and run on past it. Catch both.
        eos_ids = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
                eos_ids.append(tid)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=enc.attention_mask.to(model_wrapper.model.device),
            max_new_tokens=4096,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95 if temperature > 0 else 1.0,
            eos_token_id=eos_ids if eos_ids else None,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )
        if capture_kv:
            gen_kwargs["return_dict_in_generate"] = True

        with torch.no_grad():
            out = model_wrapper.model.generate(**gen_kwargs)

        if capture_kv:
            last_kv = out.past_key_values
            token_ids = out.sequences[0]
            response = tokenizer.decode(token_ids[input_ids.shape[1]:], skip_special_tokens=True)
        else:
            response = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

        # Strip <think>...</think> blocks from the visible response (Qwen3.5
        # with thinking mode emits these; we want the post-thinking content).
        # Also defensively trim anything past the first <|im_end|> just in case.
        import re as _re
        response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL)
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>", 1)[0]
        response = response.strip()

        if use_native:
            # Native path: parse <tool_call>...</tool_call> blocks, append
            # proper {role: tool, ...} messages. Matches main study format.
            tool_calls = _parse_tool_calls(response)
            real_calls = [tc for tc in tool_calls
                          if tc["name"] in valid_tool_names]
            content_part = _content_outside_tool_calls(response)
            if not real_calls:
                if skip_final_response:
                    if capture_kv:
                        return "", last_kv
                    return ""
                final = response.strip() if not tool_calls else content_part
                messages.append({"role": "assistant", "content": final})
                if capture_kv:
                    return final, last_kv
                return final
            assistant_turn = {
                "role": "assistant",
                "content": content_part or None,
                "tool_calls": [
                    {
                        "id": f"call_{step}_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (json.dumps(tc["arguments"])
                                          if isinstance(tc["arguments"], dict)
                                          else str(tc["arguments"])),
                        },
                    }
                    for i, tc in enumerate(real_calls)
                ],
            }
            messages.append(assistant_turn)
            for i, tc in enumerate(real_calls):
                args = tc["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                try:
                    result = executor(tc["name"], args)
                except Exception as e:
                    result = f"Error executing {tc['name']}: {e}"
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{step}_{i}",
                    "name": tc["name"],
                    "content": result,
                })
        else:
            # Legacy regex-parse path. Kept for callers that don't pass tools.
            tool = detect_tool_call(response)
            if tool is None:
                if skip_final_response:
                    if capture_kv:
                        return "", last_kv
                    return ""
                messages.append({"role": "assistant", "content": response})
                if capture_kv:
                    return response, last_kv
                return response

            tool_name, tool_args = tool
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"query": tool_args}

            result = executor(tool_name, tool_args)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue."})

    if skip_final_response:
        if capture_kv:
            return "", last_kv
        return ""

    # Max steps — final generation without tools
    prompt = tokenizer.apply_chat_template(
        messages + [{"role": "user", "content": "Provide your final answer now as a comma-separated list."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)

    gen_kwargs = dict(
        input_ids=enc.input_ids.to(model_wrapper.model.device),
        attention_mask=enc.attention_mask.to(model_wrapper.model.device),
        max_new_tokens=4096, do_sample=False,
        eos_token_id=eos_ids if eos_ids else None,
        pad_token_id=(tokenizer.pad_token_id
                      if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
    )
    if capture_kv:
        gen_kwargs["return_dict_in_generate"] = True

    with torch.no_grad():
        out = model_wrapper.model.generate(**gen_kwargs)

    if capture_kv:
        response = tokenizer.decode(out.sequences[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    else:
        response = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    import re as _re
    response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>", 1)[0]
    response = response.strip()
    if capture_kv:
        return response, out.past_key_values
    return response


_WE_ALIGNMENT_CACHE = None


def _get_we_alignment(model):
    """Lazily compute (and cache module-wide) the W_e alignment matrix
    mapping final hidden states to token-embedding space. Required for
    latent-thought loopback to produce coherent continuations: hidden states
    and token embeddings live in different vector spaces, so raw loopback
    without alignment causes the model to decode gibberish (tested: zero-
    shot raw CoCoNuT produced Python code instead of answers).

    Uses compute_alignment from latentmas_baseline (CPU computation, moves
    result to model device) to avoid OOM on large vocabularies (Qwen3.5 has
    248K vocab → 3.79 GiB alignment matrix).
    """
    global _WE_ALIGNMENT_CACHE
    if _WE_ALIGNMENT_CACHE is None:
        from latent_pilot.latentmas_baseline import compute_alignment
        print("  Computing W_e alignment matrix (one-time, CPU→GPU)...")
        _WE_ALIGNMENT_CACHE = compute_alignment(model)
    return _WE_ALIGNMENT_CACHE


def latent_react_loop(model_wrapper: ModelWrapper, messages: list[dict],
                      executor, max_steps: int = 5, m_latent: int = 20,
                      temperature: float = 0.5, skip_final_response: bool = False):
    """ReAct loop where each step runs m latent forward passes with W_e-
    aligned hidden-state loopback before decoding, letting the model "think"
    silently before emitting a tool_call or final answer.

    Mechanism per step:
      1. Encode full message context → KV cache + h_t (last hidden state)
      2. m_latent loopback passes: e_next = apply_alignment(h_t, W_e) projects
         h_t into embedding space, fed back as inputs_embeds
      3. Manual autoregressive decode from the accumulated cache

    Savings vs text_react_loop: m pre-decode forwards replace ~m tokens of
    "thinking" text the model would otherwise decode. Net savings depend on
    whether the model emits shorter responses after latent thinking (zero-
    shot behavior — measured empirically).

    W_e alignment is essential: without it, raw h_t loopback produces
    incoherent outputs (verified in earlier run: F1=0 on Qwen3-8B).
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device
    # Set of stop tokens. Qwen3.5's eos_token_id is sometimes <|endoftext|>
    # while the chat-template turn boundary is <|im_end|>; manual decode
    # loops below stop only if tok_id is in this set, catching either.
    eos_token_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_tok)
        if _tid is not None and _tid != tokenizer.unk_token_id:
            eos_token_ids.add(_tid)
    eos = tokenizer.eos_token_id  # back-compat for the variable name
    MAX_RESPONSE_TOKENS = 4096

    for step in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
        input_ids = enc.input_ids.to(dev)

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True,
                        output_hidden_states=True, return_dict=True)
        kv_cache = out.past_key_values
        h_t = out.hidden_states[-1][:, -1:, :]

        # m latent steps with W_e alignment
        from latent_pilot.latentmas_baseline import apply_alignment
        W_e, target_norm = _get_we_alignment(model)
        for _ in range(m_latent):
            e_next = apply_alignment(h_t, W_e, target_norm)
            with torch.no_grad():
                step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                                 use_cache=True, output_hidden_states=True,
                                 return_dict=True)
            kv_cache = step_out.past_key_values
            h_t = step_out.hidden_states[-1][:, -1:, :]

        # Manual decode from accumulated cache. Seed first token from h_t's logits.
        logits = model.lm_head(h_t)[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        generated = [next_token[0, 0].item()]

        for _ in range(MAX_RESPONSE_TOKENS - 1):
            with torch.no_grad():
                step_out = model(input_ids=next_token, past_key_values=kv_cache,
                                 use_cache=True, return_dict=True)
            kv_cache = step_out.past_key_values
            logits = step_out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            tok_id = next_token[0, 0].item()
            if tok_id in eos_token_ids:
                break
            generated.append(tok_id)

        response = tokenizer.decode(generated, skip_special_tokens=True)

        tool = detect_tool_call(response)
        if tool is None:
            if skip_final_response:
                return ""
            messages.append({"role": "assistant", "content": response})
            return response

        tool_name, tool_args = tool
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"query": tool_args}

        result = executor(tool_name, tool_args)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue."})

    if skip_final_response:
        return ""

    # Max steps exhausted: force a final answer (text decode, no latent)
    messages.append({"role": "user",
                     "content": "Provide your final answer now as a comma-separated list."})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
    with torch.no_grad():
        gen = model.generate(
            input_ids=enc.input_ids.to(dev),
            attention_mask=enc.attention_mask.to(dev),
            max_new_tokens=4096, do_sample=False,
        )
    return tokenizer.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def build_working_memory(model_wrapper: ModelWrapper, messages: list[dict],
                         m_latent: int, prior_kv=None) -> object:
    """Build KV working memory from an agent's conversation history,
    optionally continuing from a prior agent's accumulated KV cache.

    If prior_kv is provided, this agent's conversation is forwarded as a
    CONTINUATION of the prior cache (positions are contiguous). The result
    accumulates ALL prior agents' work + this agent's work + latent steps.

    If prior_kv is None, encodes this agent's conversation from scratch.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
    input_ids = enc.input_ids.to(dev)

    if prior_kv is not None:
        # Continue from prior agent's KV — positions are contiguous
        prefix_len = _kv_seq_len(prior_kv)
        cache_pos = torch.arange(prefix_len, prefix_len + input_ids.shape[1],
                                 device=dev, dtype=torch.long)
        attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]),
                               device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       past_key_values=prior_kv, cache_position=cache_pos,
                       use_cache=True, output_hidden_states=True, return_dict=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True,
                        output_hidden_states=True, return_dict=True)

    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    # Latent steps using our proven W_e alignment (72-86% agreement in pilot).
    # Share the module-level alignment cache (_WE_ALIGNMENT_CACHE) so we
    # only pay the CPU computation once, whether the first caller is
    # build_working_memory, latent_react_loop, or run_cocunut_synthesis.
    if m_latent > 0:
        from latent_pilot.latentmas_baseline import apply_alignment
        W_e, target_norm = _get_we_alignment(model)
        for _ in range(m_latent):
            e_next = apply_alignment(h_t, W_e, target_norm)
            with torch.no_grad():
                step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                                use_cache=True, output_hidden_states=True, return_dict=True)
            kv_cache = step_out.past_key_values
            h_t = step_out.hidden_states[-1][:, -1:, :]
        # Free intermediate tensors to reduce VRAM pressure on large-vocab
        # models (Qwen3.5-9B 248K vocab). Without this, sequential KV
        # accumulation across 3 agents OOMs on 24G A5000.
        del h_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kv_cache


def _extract_tool_summary(messages):
    """Extract compact tool call summary from messages (matches production)."""
    calls = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content", "")
            tool = detect_tool_call(content)
            if tool:
                name, args = tool
                args_str = str(args)[:60]
                calls.append(f"{name}({args_str})")
    return " -> ".join(calls) if calls else "No tool calls"


_STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "for", "nor", "so", "yet",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "of", "to", "in", "on", "at", "by",
    "with", "from", "as", "that", "this", "these", "those", "it",
    "its", "their", "there", "which", "who", "whom", "whose", "what",
    "when", "where", "why", "how", "all", "any", "each", "some",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "your", "not", "no", "can",
    "will", "would", "could", "should", "may", "might", "must",
])


def _compute_semantic_agreement(responses):
    """Average pairwise Jaccard similarity over normalized response content.

    Decides list-vs-prose mode PER AGENT RESPONSE based on comma count, then
    normalizes consistently:
      - list mode (≥2 commas): extract comma-separated entity strings, filter
        obvious prose fragments. Works for QAMPARI.
      - prose mode (<2 commas): extract content-word tokens (≥4 chars, not
        stopwords). Works for FanOutQA/BrowseComp.

    All responses within a task use the SAME mode (majority vote on commas)
    so Jaccard compares like-to-like. Pure string ops; cheap to compute.
    """
    import re

    # Decide mode: if majority of non-empty responses have ≥2 commas, use list mode
    non_empty = [r for r in responses if r]
    if not non_empty:
        return 0.0
    list_mode_votes = sum(1 for r in non_empty if r.count(",") >= 2)
    use_list_mode = list_mode_votes >= (len(non_empty) + 1) // 2

    def normalize_list(text):
        items = re.split(r'[,;\n]', text)
        cleaned = set()
        for item in items:
            item = re.sub(r'[*_#`]', '', item).lower().strip()
            item = item.strip('.!?:;-()[]{}')
            # Stricter prose filter: reject items with many spaces (phrases)
            # unless they look like proper-noun entities ("anton webern")
            if 1 <= len(item) <= 100 and item.count(" ") <= 3:
                if not item.startswith(('the ', 'a ', 'an ', 'and ', 'but ', 'however ')):
                    cleaned.add(item)
        return cleaned

    def normalize_prose(text):
        text = re.sub(r'[*_#`]', '', text).lower()
        tokens = re.findall(r"[a-z][a-z0-9']{3,}", text)
        return set(t for t in tokens if t not in _STOPWORDS)

    normalize = normalize_list if use_list_mode else normalize_prose
    sets = [normalize(r) for r in responses]
    if len(sets) < 2 or any(not s for s in sets):
        return 0.0
    jaccs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            union = sets[i] | sets[j]
            jaccs.append(len(inter) / len(union) if union else 0.0)
    return sum(jaccs) / len(jaccs) if jaccs else 0.0


def _extract_tool_queries(messages):
    """Return structured list of (tool_name, args_dict) from assistant messages.
    Used for measuring cross-agent tool-call duplicate rates."""
    calls = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content", "")
            tool = detect_tool_call(content)
            if tool:
                name, args = tool
                calls.append({"name": name, "args": args if isinstance(args, dict) else {"raw": str(args)}})
    return calls


def _format_debate_prompt(trajectories, exclude_idx):
    """Format debate prompt showing other agents' work (matches production)."""
    parts = []
    for i, traj in enumerate(trajectories):
        if i == exclude_idx or traj is None:
            continue
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool calls: {traj.get('tool_summary', 'None')}\n"
            f"  Final answer: {traj.get('final_response', 'No response')}\n"
        )
    other_text = "\n".join(parts)
    return (
        f"These are the approaches and results from other agents:\n\n"
        f"{other_text}\n\n"
        f"Review their tool-calling strategies and results. "
        f"If you believe your approach was correct, reaffirm it. "
        f"If you see a better approach or errors in your work, "
        f"make additional tool calls to correct or verify. "
        f"Provide your updated final answer."
    )


def _format_synthesis(question, trajectories):
    """Format final synthesis prompt. Matches main study's
    format_debate_synthesis in mas-energy/code/prompts.py exactly."""
    parts = [f"Task: {question}\n\nFinal agent responses after debate:\n"]
    for i, traj in enumerate(trajectories):
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool calls: {traj.get('tool_summary', 'None')}\n"
            f"  Answer: {traj.get('final_response', 'No response')}\n"
        )
    parts.append("\nSynthesize the best final answer based on all agents' work.")
    return "\n".join(parts)


def run_text_parallel_decentralized(model_wrapper, question, executor,
                                     n_agents=3, max_react_steps=5, n_rounds=1,
                                     metadata_out=None,
                                     enable_agreement_skip=False,
                                     agreement_threshold=0.5,
                                     enable_tool_cache=False,
                                     enable_grammar=False,
                                     tools=None):
    """Text decentralized matching the production topology:
    Phase 1: parallel independent search
    Phase 2: all-to-all text debate (R rounds)
    Phase 3: synthesis
    """
    # Optional tool-result cache: wraps executor so identical queries from
    # different agents reuse cached results (MAS duplicate rate is ~73% on
    # QAMPARI per our measurement).
    if enable_tool_cache:
        executor = _CachingExecutor(executor)

    # Phase 1: Independent parallel ReAct
    trajectories = []
    for agent_idx in range(n_agents):
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)
        response = text_react_loop(model_wrapper, msgs, executor,
                                   max_steps=max_react_steps, temperature=0.5,
                                   tools=tools)
        trajectories.append({
            "final_response": response,
            "tool_summary": _extract_tool_summary(msgs),
            "messages": msgs,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Snapshot Phase 1 tool calls before debate extends message histories
    phase1_tool_calls = [_extract_tool_queries(t["messages"]) for t in trajectories]

    # Agreement gate: if enabled and agents already agree, skip debate.
    # Measures whether debate is adding value on THIS task; records decision
    # in metadata so we can post-hoc verify gating correctness vs always-debate.
    phase1_responses = [t["final_response"] for t in trajectories]
    agreement_score = _compute_semantic_agreement(phase1_responses)
    did_skip_debate = (enable_agreement_skip
                      and agreement_score >= agreement_threshold)

    # Phase 2: All-to-all debate rounds (skipped if agents agreed)
    if not did_skip_debate:
        for round_idx in range(n_rounds):
            new_trajectories = []
            for agent_idx in range(n_agents):
                debate_msg = _format_debate_prompt(trajectories, exclude_idx=agent_idx)
                msgs = trajectories[agent_idx]["messages"] + [
                    {"role": "user", "content": debate_msg}
                ]
                response = text_react_loop(model_wrapper, msgs, executor,
                                           max_steps=max_react_steps, temperature=0.5,
                                           tools=tools)
                new_trajectories.append({
                    "final_response": response,
                    "tool_summary": _extract_tool_summary(msgs),
                    "messages": msgs,
                })
            trajectories = new_trajectories

    # Phase 3: Synthesis — plain generate, no tool loop. Previously wrapped
    # the synth prompt with a "use tools" system prompt via build_react_prompt,
    # which caused the synthesizer to waste its only step on tool calls
    # instead of integrating debate findings. This matched the main study's
    # DEBATE_SYNTHESIZER_PROMPT pattern in topologies.py.
    synth_msgs = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": _format_synthesis(question, trajectories)},
    ]
    # Grammar-constrained synthesis: stop_strings cut the decode when the
    # model transitions from list output to prose. Lightweight but effective
    # on verbose models (Qwen3.5 especially).
    synth_ans = generate_text(model_wrapper, synth_msgs, temperature=0.0,
                              constrain_list=enable_grammar)

    # Surface per-agent tool-call structure for duplicate-rate analysis +
    # agreement-gate decision for post-hoc validation of the gate.
    if metadata_out is not None:
        phase_all = [_extract_tool_queries(t["messages"]) for t in trajectories]
        metadata_out["phase1_tool_calls"] = phase1_tool_calls
        metadata_out["phase2_tool_calls"] = [
            all_c[len(p1):] for all_c, p1 in zip(phase_all, phase1_tool_calls)
        ]
        metadata_out["phase1_agreement"] = agreement_score
        metadata_out["skipped_debate"] = did_skip_debate
        if isinstance(executor, _CachingExecutor):
            metadata_out["tool_cache_hits"] = executor.hits
            metadata_out["tool_cache_misses"] = executor.misses
    return synth_ans


def run_latent_parallel_decentralized(model_wrapper, question, executor,
                                       m_latent=40, n_agents=3, max_react_steps=5,
                                       n_rounds=1, tools=None):
    """Latent decentralized: parallel search + sequential latent debate.

    Phase 1 (PARALLEL, identical to text): all agents search independently.
    Phase 2 (SEQUENTIAL LATENT): agents take turns, each building on
      accumulated KV working memory from prior agents.
    Phase 3: final agent synthesizes from accumulated working memory.

    This is the natural latent topology:
    - Phase 1 exploits QAMPARI's parallelism (different agents find different entities)
    - Phase 2 uses sequential KV sharing (natural for latent, no concatenation needed)
    - Each agent's KV encodes ALL prior agents' accumulated findings
    """
    # Phase 1: Independent parallel ReAct (SAME as text condition)
    agent_msgs_list = []
    agent_responses = []
    for agent_idx in range(n_agents):
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)
        response = text_react_loop(model_wrapper, msgs, executor,
                                   max_steps=max_react_steps, temperature=0.5,
                                   tools=tools)
        agent_msgs_list.append(msgs)
        agent_responses.append(response)

    # Phase 2: Sequential latent accumulation
    # Each agent's Phase 1 conversation is forwarded as a CONTINUATION
    # of the prior accumulated KV. Positions are contiguous:
    #   [Agent 0 conv + latent | Agent 1 conv + latent | Agent 2 conv + latent]
    # Each agent's KV encodes ALL prior agents' work.
    working_memory = None
    for agent_idx in range(n_agents - 1):  # last agent is synthesizer
        # Build on prior working memory (None for first agent)
        working_memory = build_working_memory(
            model_wrapper, agent_msgs_list[agent_idx],
            m_latent=m_latent, prior_kv=working_memory,
        )

    # Phase 3: Final synthesis conditioned on accumulated working memory.
    # System prompt aligned to DEBATE_SYNTHESIZER_PROMPT for fair paired
    # comparison with text MAS and other latent modes. The synthesizer
    # sees prior agents' context via accumulated KV prefix (that's the
    # kv_share mechanism), not via explicit trajectory text in the user
    # message — this is the structural distinction being tested.
    synth_msgs = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": question},
    ]
    synth_response = text_react_with_memory(
        model_wrapper, synth_msgs, executor, working_memory,
        max_steps=1, temperature=0.0,
    )
    return synth_response


def run_cocunut_synthesis(model_wrapper, messages, m_latent,
                          max_new_tokens=4096, temperature=0.0):
    """Latent synthesis via W_e-aligned hidden-state loopback.

    NAMING NOTE: the name "cocunut" refers to the *decoding mechanism*
    (hidden-state loopback) introduced in Hao et al. 2024 — NOT the full
    CoCoNuT trained system. Our use is zero-shot: we apply the mechanism
    at inference time to a pre-trained model that was never fine-tuned
    for continuous-space reasoning. The paper's mixed results even with
    full training suggest the zero-shot application is strictly harder.

    After encoding the prompt, project the final hidden state into embedding
    space via W_e alignment and feed it back as the next input embedding
    for m_latent steps. Then autoregressive-decode the final answer from
    the accumulated cache.

    W_e alignment is essential. Hidden states and token embeddings live in
    different vector spaces, so raw loopback without alignment produces
    incoherent outputs (verified: zero-shot raw CoCoNuT decoded Python code
    instead of answers on Qwen3-8B QAMPARI). The alignment matrix is
    computed once via compute_alignment from latentmas_baseline (CPU-side
    to avoid OOM on large vocabularies) and cached module-wide.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
    input_ids = enc.input_ids.to(dev)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    from latent_pilot.latentmas_baseline import apply_alignment
    W_e, target_norm = _get_we_alignment(model)
    for _ in range(m_latent):
        e_next = apply_alignment(h_t, W_e, target_norm)
        with torch.no_grad():
            step = model(inputs_embeds=e_next, past_key_values=kv_cache,
                         use_cache=True, output_hidden_states=True, return_dict=True)
        kv_cache = step.past_key_values
        h_t = step.hidden_states[-1][:, -1:, :]

    # Manual autoregressive decode from the accumulated cache. Seed the first
    # output token from h_t's logits, then continue normally.
    logits = model.lm_head(h_t)[:, -1, :]
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
    else:
        next_token = logits.argmax(dim=-1, keepdim=True)
    generated = [next_token[0, 0].item()]
    # Set of stop tokens. Qwen3.5's eos_token_id is sometimes <|endoftext|>
    # while the chat-template turn boundary is <|im_end|>; manual decode
    # loops below stop only if tok_id is in this set, catching either.
    eos_token_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_tok)
        if _tid is not None and _tid != tokenizer.unk_token_id:
            eos_token_ids.add(_tid)
    eos = tokenizer.eos_token_id  # back-compat for the variable name

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            step = model(input_ids=next_token, past_key_values=kv_cache,
                         use_cache=True, return_dict=True)
        kv_cache = step.past_key_values
        logits = step.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        tok_id = next_token[0, 0].item()
        if tok_id in eos_token_ids:
            break
        generated.append(tok_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_cocunut_parallel_decentralized(model_wrapper, question, executor,
                                        m_latent=40, n_agents=3, max_react_steps=5,
                                        n_rounds=1):
    """CoCoNuT-based parallel decentralized MAS.

    Architecture-agnostic analog of run_latent_parallel_decentralized for
    models whose hybrid attention breaks KV sharing (Qwen3.5 hybrid Gated
    DeltaNet).

    Phase 1 (PARALLEL text, identical to text condition): agents search.
    Phase 2 (TEXT debate summaries, compact): each agent sees short summaries
      of others' final answers + tool summaries and can revise via one round
      of additional ReAct. This handoff is text but deliberately compact.
    Phase 3 (COCOUNUT SYNTHESIS): synthesizer encodes the compact debate
      transcript, runs m_latent hidden-state loopback steps, then decodes
      the final answer. This is where the decode-cost reduction happens:
      m_latent forward passes without sample/detokenize, vs the otherwise-
      decoded synthesis tokens.
    """
    # Phase 1: parallel independent text ReAct (same as text baseline)
    trajectories = []
    for agent_idx in range(n_agents):
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)
        response = text_react_loop(model_wrapper, msgs, executor,
                                   max_steps=max_react_steps, temperature=0.5)
        trajectories.append({
            "final_response": response,
            "tool_summary": _extract_tool_summary(msgs),
            "messages": msgs,
        })

    # Phase 2: compact text debate (same payload as text condition but no
    # extra ReAct — debate is collapsed into the synthesis prompt directly,
    # because the synthesis step does the reasoning latently).
    # n_rounds is kept as a parameter for API symmetry but not used here;
    # debate happens in synthesis via CoCoNuT.
    _ = n_rounds

    # Phase 3: CoCoNuT synthesis from compact debate transcript.
    # Aligned to DEBATE_SYNTHESIZER_PROMPT for paired comparison consistency
    # across all three latent modes and the text MAS baseline.
    synth_msgs = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": _format_synthesis(question, trajectories)},
    ]
    return run_cocunut_synthesis(
        model_wrapper, synth_msgs, m_latent=m_latent, temperature=0.0,
    )


def _free_vram():
    """Free cached allocator memory between heavy phases to reduce OOM risk
    on 24 GB A5000s. No-op on non-CUDA; cheap on CUDA (~ms)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_phase1_latent_parallel_decentralized(model_wrapper, question, executor,
                                              m_latent=20, n_agents=3,
                                              max_react_steps=10, n_rounds=1):
    """Fully-latent parallel decentralized MAS (all three phases latent).

    Architecture-agnostic attack on both the main decode-cost driver (intra-
    agent reasoning) AND the inter-agent cross-talk (debate). No KV sharing,
    no W_e alignment — just latent_react_loop (CoCoNuT thinking between tool
    calls) swapped in wherever text decoding would dominate.

    Phase 1 (PARALLEL latent-ReAct, N agents independent):
      Each agent runs latent_react_loop. m CoCoNuT forwards replace decoded
      reasoning between tool_calls. Tool calls and final answers stay text
      (executor needs strings; answers are the output).

    Phase 2 (LATENT DEBATE, R rounds):
      Each round, each agent receives a compact text payload describing the
      OTHER agents' tool_summaries + final answers (prefill is 308x cheaper
      than decode per token, so the text payload is cheap). Agent then runs
      latent_react_loop again: it can issue additional tool calls if it
      thinks others missed entities, or just decode a revised answer after
      m latent thinking steps.

    Phase 3 (COCOUNUT synthesis):
      Synthesizer encodes the post-debate trajectories and decodes the final
      comma-separated entity list after m hidden-state loopback steps.

    Transfer to Qwen3.5 hybrid attention: architecture-agnostic at the
    interface (no KV surgery, no alignment). Empirically tests whether raw
    hidden-state loopback produces coherent continuations through Gated
    DeltaNet layers.
    """
    # Phase 1: parallel independent latent-ReAct
    trajectories = []
    for agent_idx in range(n_agents):
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)
        response = latent_react_loop(
            model_wrapper, msgs, executor,
            max_steps=max_react_steps, m_latent=m_latent, temperature=0.5,
        )
        trajectories.append({
            "final_response": response,
            "tool_summary": _extract_tool_summary(msgs),
            "messages": msgs,
        })
        _free_vram()

    # Phase 2: latent debate rounds (all-to-all; agents refine with latent
    # thinking after seeing others' compact findings).
    for round_idx in range(n_rounds):
        new_trajectories = []
        for agent_idx in range(n_agents):
            debate_msg = _format_debate_prompt(trajectories, exclude_idx=agent_idx)
            msgs = trajectories[agent_idx]["messages"] + [
                {"role": "user", "content": debate_msg}
            ]
            response = latent_react_loop(
                model_wrapper, msgs, executor,
                max_steps=max_react_steps, m_latent=m_latent, temperature=0.5,
            )
            new_trajectories.append({
                "final_response": response,
                "tool_summary": _extract_tool_summary(msgs),
                "messages": msgs,
            })
            _free_vram()
        trajectories = new_trajectories

    # Phase 3: CoCoNuT synthesis over post-debate trajectories
    synth_msgs = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": _format_synthesis(question, trajectories)},
    ]
    result = run_cocunut_synthesis(
        model_wrapper, synth_msgs, m_latent=m_latent, temperature=0.0,
    )
    _free_vram()
    return result


def run_latent_decentralized(model_wrapper, question, executor,
                             m_latent=40, n_agents=3, max_react_steps=5):
    """Latent decentralized: text ReAct for work, latent for communication.

    Each agent does text ReAct (tool calls in text — identical to text
    condition). The ONLY difference: instead of sharing the text response,
    the agent's FULL conversation KV cache (including tool trajectory) is
    shared as working memory, with optional m latent steps on top.

    The text response generated during ReAct is NOT shared with other
    agents. Only the KV cache is shared.
    """
    working_memory = None

    for agent_idx in range(n_agents):
        is_intermediate = agent_idx < n_agents - 1
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)

        if working_memory is not None:
            response = text_react_with_memory(
                model_wrapper, msgs, executor, working_memory,
                max_steps=max_react_steps,
                temperature=0.5 if is_intermediate else 0.0,
                skip_final_response=is_intermediate,
            )
        else:
            # Intermediate agents: skip_final_response=True so the ~300-token
            # text response is NOT decoded. Latent steps replace it entirely.
            # Final agent: generates text response (the answer).
            response = text_react_loop(
                model_wrapper, msgs, executor,
                max_steps=max_react_steps,
                temperature=0.5 if is_intermediate else 0.0,
                skip_final_response=is_intermediate,
            )

        if is_intermediate:
            # Latent summary replaces text response. msgs contains tool work
            # only (no final response decode happened).
            working_memory = build_working_memory(
                model_wrapper, msgs, m_latent=m_latent,
            )

    return response


def _latent_steps_on_kv(model_wrapper, kv_cache, m_latent):
    """Do m latent steps directly on an existing KV cache. No re-encode."""
    model = model_wrapper.model
    # Get h_t from the last position — need a forward pass on a dummy token
    # to extract the hidden state. Actually, we need output_hidden_states
    # which generate() doesn't give us. So do one forward pass on a padding
    # token to get h_t, then latent steps from there.
    # Simpler: re-derive h_t by forwarding a single newline token.
    tokenizer = model_wrapper.tokenizer
    dev = next(model.parameters()).device
    dummy_ids = tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

    seq_len = _kv_seq_len(kv_cache)
    with torch.no_grad():
        out = model(input_ids=dummy_ids, past_key_values=kv_cache,
                    use_cache=True, output_hidden_states=True, return_dict=True)
    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    from latent_pilot.latentmas_baseline import apply_alignment, compute_alignment
    if not hasattr(_latent_steps_on_kv, '_we_cache'):
        print("  Computing W_e alignment matrix...")
        _latent_steps_on_kv._we_cache = compute_alignment(model)
    W_e, target_norm = _latent_steps_on_kv._we_cache

    for _ in range(m_latent):
        e_next = apply_alignment(h_t, W_e, target_norm)
        with torch.no_grad():
            step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                            use_cache=True, output_hidden_states=True, return_dict=True)
        kv_cache = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]

    return kv_cache


def run_latent_decentralized_no_reencode(model_wrapper, question, executor,
                                          m_latent=40, n_agents=3, max_react_steps=5):
    """Latent decentralized WITHOUT re-encoding.

    Same as run_latent_decentralized but captures the KV cache directly
    from model.generate() via return_dict_in_generate=True. Eliminates
    the ~500-1000 token re-encode in build_working_memory.

    Ablation: compare energy vs the re-encode version to quantify
    the re-encode overhead.
    """
    working_memory = None

    for agent_idx in range(n_agents):
        is_intermediate = agent_idx < n_agents - 1
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)

        if working_memory is not None:
            response = text_react_with_memory(
                model_wrapper, msgs, executor, working_memory,
                max_steps=max_react_steps,
                temperature=0.5 if is_intermediate else 0.0,
                skip_final_response=is_intermediate,
            )
            # text_react_with_memory doesn't support capture_kv yet,
            # so fall back to re-encode for agent 2+ receiving memory
            if is_intermediate:
                working_memory = build_working_memory(
                    model_wrapper, msgs, m_latent=m_latent,
                )
        else:
            # First agent: capture KV directly from generate (no re-encode)
            result = text_react_loop(
                model_wrapper, msgs, executor,
                max_steps=max_react_steps,
                temperature=0.5 if is_intermediate else 0.0,
                skip_final_response=is_intermediate,
                capture_kv=is_intermediate,
            )
            if is_intermediate:
                response, kv_cache = result
                working_memory = _latent_steps_on_kv(model_wrapper, kv_cache, m_latent)
            else:
                response = result

    return response


def text_react_with_memory(model_wrapper, messages, executor, past_kv,
                           max_steps=5, temperature=0.5,
                           skip_final_response=False):
    """Text ReAct loop where the FIRST generation is conditioned on prior
    agent's KV working memory, and subsequent steps use normal generation.

    Step 1: model.generate with past_key_values (working memory influence)
    Step 2+: normal model.generate without past_kv (working memory's influence
             is captured in step 1's response, which is in the conversation)

    skip_final_response: if True, return "" after tool calls complete instead
    of generating final text. For intermediate latent agents.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device

    for step in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
        input_ids = enc.input_ids.to(dev)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=enc.attention_mask.to(dev),
            max_new_tokens=4096,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95 if temperature > 0 else 1.0,
        )

        # Only use working memory on the first step
        if step == 0 and past_kv is not None:
            prefix_len = _kv_seq_len(past_kv)
            attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]),
                                   device=dev, dtype=torch.long)
            gen_kwargs["attention_mask"] = attn_mask
            gen_kwargs["past_key_values"] = copy.deepcopy(past_kv)

        with torch.no_grad():
            gen_out = model.generate(**gen_kwargs)
        response = tokenizer.decode(gen_out[0, input_ids.shape[1]:], skip_special_tokens=True)

        tool = detect_tool_call(response)
        if tool is None:
            if skip_final_response:
                return ""
            messages.append({"role": "assistant", "content": response})
            return response

        tool_name, tool_args = tool
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"query": tool_args}
        result = executor(tool_name, tool_args)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue."})

    if skip_final_response:
        return ""

    # Final answer
    messages.append({"role": "user", "content": "Provide your final answer as a comma-separated list."})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=89292)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids.to(dev),
                            attention_mask=enc.attention_mask.to(dev),
                            max_new_tokens=4096, do_sample=False)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def _kv_seq_len(pkv) -> int:
    if hasattr(pkv, "get_seq_length"):
        try:
            n = pkv.get_seq_length()
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    if hasattr(pkv, "layers") and pkv.layers:
        for layer in pkv.layers:
            for attr in ("keys", "key_states", "key_cache"):
                kt = getattr(layer, attr, None)
                if kt is not None:
                    return int(kt.shape[-2])
    if hasattr(pkv, "key_cache") and pkv.key_cache:
        for k in pkv.key_cache:
            if k is not None and hasattr(k, "shape"):
                return int(k.shape[-2])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--m-latent", type=int, default=40)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=1,
                    help="Debate rounds for decentralized topology. Main study "
                         "uses 2 (config.py DECENTRALIZED_ROUNDS); pilot historically "
                         "hardcoded 1 which capped text-MAS strength.")
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--benchmark", type=str, default="qampari", choices=["qampari", "fanoutqa"])
    ap.add_argument("--data-dir", type=str, default=None, help="Benchmark data dir (QAMPARI only)")
    ap.add_argument("--no-reencode", action="store_true",
                    help="Skip re-encode in build_working_memory; capture KV from generate() directly")
    ap.add_argument("--latent-mode", type=str, default="kv_share",
                    choices=["kv_share", "cocunut", "phase1_latent"],
                    help="kv_share: LatentMAS-style KV sharing (softmax attention only). "
                         "cocunut: architecture-agnostic synthesis-only latent (hybrid OK). "
                         "phase1_latent: latent-ReAct in Phase 1 (architecture-agnostic; "
                         "attacks the main decode-cost driver).")
    ap.add_argument("--enable-agreement-skip", action="store_true",
                    help="Skip Phase 2 debate when agents' Phase 1 responses agree "
                         "semantically above --agreement-threshold.")
    ap.add_argument("--agreement-threshold", type=float, default=0.5,
                    help="Jaccard threshold (0-1) above which debate is skipped.")
    ap.add_argument("--enable-tool-cache", action="store_true",
                    help="Per-task cache on tool_name+query so duplicate "
                         "cross-agent queries reuse results.")
    ap.add_argument("--enable-grammar", action="store_true",
                    help="Grammar-constrained synthesis via stop_strings — halts "
                         "decode at paragraph-break markers to cut verbose prose tails.")
    ap.add_argument("--lora-adapter", type=str, default=None,
                    help="Path to a LoRA adapter directory to attach to the base "
                         "model (e.g., output of coconut_train.py or "
                         "concision_train.py). Enables evaluating trained "
                         "adapters through the existing pipeline.")
    args = ap.parse_args()

    # Output setup
    out_dir = Path(f"/atlas2/u/{os.environ['USER']}/mas_project/mas-energy/results/latent_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    reencode_tag = "_noreencode" if args.no_reencode else ""
    mode_tag = f"_{args.latent_mode}" if args.latent_mode != "kv_share" else ""
    agree_tag = "_agreeskip" if args.enable_agreement_skip else ""
    cache_tag = "_toolcache" if args.enable_tool_cache else ""
    grammar_tag = "_grammar" if args.enable_grammar else ""
    # Include model name so concurrent runs (same bench, same m, different model)
    # don't share a filename and clobber each other via unlink+append.
    model_tag = args.model_name.replace("/", "_").replace(".", "")
    tag = f"{model_tag}_{args.benchmark}_agentic_k{args.max_react_steps}_m{args.m_latent}{reencode_tag}{mode_tag}{agree_tag}{cache_tag}{grammar_tag}"
    jsonl_path = out_dir / f"eval_{tag}.jsonl"
    summary_path = out_dir / f"eval_{tag}_summary.json"
    # Preserve any prior run at this path as a timestamped .bak so re-running
    # the same config doesn't silently destroy earlier ablation data (e.g.,
    # agreement-gate runs overwriting the kvshare m_latent ablation's m=0 file).
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in (jsonl_path, summary_path):
        if p.exists():
            backup = p.with_name(p.name + f".{ts}.bak")
            p.rename(backup)
            print(f"  Preserved prior {p.name} as {backup.name}")

    # Load model directly via transformers, bypassing LatentMAS's ModelWrapper.
    # ModelWrapper eagerly builds a W_e realignment matrix on init (OOMs on
    # Qwen3.5-9B at 3.79 GiB on top of the loaded model). Phase 1 latent /
    # CoCoNuT paths don't use W_e alignment — they're raw hidden-state
    # loopback — so we don't need any realignment matrix. The kv_share path
    # uses compute_alignment from latentmas_baseline (CPU-safe) when called.
    print(f"Loading {args.model_name} directly via transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class SimpleModelWrapper:
        def __init__(self, model_name, device="cuda"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            self.model.eval()

    mw = SimpleModelWrapper(args.model_name, device="cuda")
    print(f"Model loaded: {args.model_name} on {mw.model.device}")

    # Optional LoRA adapter: attach a trained adapter (e.g., from coconut_train
    # or concision_train) to evaluate it through the same pipeline. After
    # attaching, mw.model is a PeftModel; forward pass + generate still work.
    if args.lora_adapter:
        from peft import PeftModel
        print(f"Attaching LoRA adapter from {args.lora_adapter}...")
        mw.model = PeftModel.from_pretrained(mw.model, args.lora_adapter)
        mw.model.eval()
        print(f"  Adapter attached. Base model: {args.model_name}")

    # Load benchmark
    bench, evaluate_fn, question_key = load_benchmark(args.benchmark, data_dir=args.data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} {args.benchmark} tasks")

    # Energy measurement
    from energy import EnergyMonitor
    em = EnergyMonitor()
    idle_w = em.measure_idle(duration=5)
    print(f"Idle GPU power: {idle_w:.1f} W")

    agg = {k: 0.0 for k in ["single_f1", "text_f1", "latent_f1",
                              "single_acc", "text_acc", "latent_acc",
                              "single_energy", "text_energy", "latent_energy", "n"]}
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task[question_key]
        executor, cleanup = bench.make_executor(task)

        # Per-task, per-condition seeding for reproducibility across runs.
        # Previously text MAS showed 42-139% energy variance across runs
        # because temperature=0.5 agents produced different token streams.
        # Seeding torch randomness per (task, condition) fixes this so
        # paired comparisons are deterministic.
        def _seed_for(condition_idx):
            torch.manual_seed(args.seed * 10000 + i * 10 + condition_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed * 10000 + i * 10 + condition_idx)

        try:
            # Single agent baseline (energy-measured) — uses SAS_PROMPT
            # (not DEBATE_AGENT_PROMPT) to match main study's run_sas.
            _seed_for(0)
            em.start()
            msgs = build_sas_prompt(question)
            single_ans = text_react_loop(mw, msgs, executor,
                                         max_steps=args.max_react_steps, temperature=0.0)
            single_rec = em.stop(metadata={"condition": "single"})
            single_acc, single_f1_val, single_eval = evaluate_fn(task, single_ans)

            # Text parallel-decentralized MAS (energy-measured)
            _seed_for(1)
            em.start()
            text_meta = {}
            text_ans = run_text_parallel_decentralized(
                mw, question, executor,
                n_agents=args.n_agents,
                max_react_steps=args.max_react_steps,
                n_rounds=args.n_rounds,
                metadata_out=text_meta,
                enable_agreement_skip=args.enable_agreement_skip,
                agreement_threshold=args.agreement_threshold,
                enable_tool_cache=args.enable_tool_cache,
                enable_grammar=args.enable_grammar,
                tools=bench.get_tools() if hasattr(bench, "get_tools") else None,
            )
            text_rec = em.stop(metadata={"condition": "text_mas"})
            text_acc, text_f1_val, text_eval = evaluate_fn(task, text_ans)

            # Latent parallel-decentralized MAS (energy-measured)
            _seed_for(2)
            em.start()
            if args.latent_mode == "phase1_latent":
                latent_ans = run_phase1_latent_parallel_decentralized(
                    mw, question, executor,
                    m_latent=args.m_latent, n_agents=args.n_agents,
                    max_react_steps=args.max_react_steps, n_rounds=args.n_rounds,
                )
            elif args.latent_mode == "cocunut":
                latent_ans = run_cocunut_parallel_decentralized(
                    mw, question, executor,
                    m_latent=args.m_latent, n_agents=args.n_agents,
                    max_react_steps=args.max_react_steps, n_rounds=args.n_rounds,
                )
            else:
                latent_ans = run_latent_parallel_decentralized(
                    mw, question, executor,
                    m_latent=args.m_latent, n_agents=args.n_agents,
                    max_react_steps=args.max_react_steps, n_rounds=args.n_rounds,
                    tools=bench.get_tools() if hasattr(bench, "get_tools") else None,
                )
            latent_rec = em.stop(metadata={"condition": f"latent_mas_{args.latent_mode}"})
            latent_acc, latent_f1_val, latent_eval = evaluate_fn(task, latent_ans)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            cleanup()
            continue

        cleanup()

        single_j = single_rec.get("gpu_dynamic_energy_joules", 0)
        text_j = text_rec.get("gpu_dynamic_energy_joules", 0)
        latent_j = latent_rec.get("gpu_dynamic_energy_joules", 0)

        if i == 0:
            print(f"\n--- First task ---")
            print(f"  Q: {question[:80]}...")
            print(f"  Single: {single_ans[:120]}... acc={single_acc:.3f} E={single_j:.1f}J")
            print(f"  Text: {text_ans[:120]}... acc={text_acc:.3f} E={text_j:.1f}J")
            print(f"  Latent: {latent_ans[:120]}... acc={latent_acc:.3f} E={latent_j:.1f}J")
            print()

        task_id = task.get("qid", task.get("id", i))
        rec = {"i": i, "task_id": task_id,
               "single_loose_accuracy": single_acc, "text_loose_accuracy": text_acc,
               "latent_loose_accuracy": latent_acc,
               "single_f1": single_f1_val, "text_f1": text_f1_val,
               "latent_f1": latent_f1_val,
               "single_energy_j": single_j, "text_energy_j": text_j,
               "latent_energy_j": latent_j,
               # Final answers per condition (added 2026-05-16 for F1-collapse
               # diagnosis; previously only metrics were captured, which made
               # it impossible to tell *what* the latent agent emitted when
               # accuracy dropped vs text MAS).
               "single_ans": single_ans,
               "text_ans": text_ans,
               "latent_ans": latent_ans,
               "question": question,
               # Per-agent tool-call structure for duplicate-rate analysis.
               # Present only when text MAS ran (metadata_out was populated).
               "text_phase1_tool_calls": text_meta.get("phase1_tool_calls"),
               "text_phase2_tool_calls": text_meta.get("phase2_tool_calls"),
               # Agreement-gate decisions for post-hoc analysis.
               "text_phase1_agreement": text_meta.get("phase1_agreement"),
               "text_skipped_debate": text_meta.get("skipped_debate"),
               # Tool-cache hit/miss counts (only populated if cache was enabled)
               "tool_cache_hits": text_meta.get("tool_cache_hits"),
               "tool_cache_misses": text_meta.get("tool_cache_misses")}
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        agg["single_f1"] += single_f1_val
        agg["text_f1"] += text_f1_val
        agg["latent_f1"] += latent_f1_val
        agg["single_acc"] += single_acc
        agg["text_acc"] += text_acc
        agg["latent_acc"] += latent_acc
        agg["single_energy"] += single_j
        agg["text_energy"] += text_j
        agg["latent_energy"] += latent_j
        agg["n"] += 1

        if (i + 1) % 5 == 0:
            n = agg["n"]
            print(f"[{i+1:3d}/{len(tasks)}] "
                  f"single={agg['single_acc']/n:.3f}({agg['single_energy']/n:.0f}J) "
                  f"text={agg['text_acc']/n:.3f}({agg['text_energy']/n:.0f}J) "
                  f"latent={agg['latent_acc']/n:.3f}({agg['latent_energy']/n:.0f}J)")

    if agg["n"] == 0:
        print("No tasks completed")
        return

    n = agg["n"]
    summary = {
        "model": args.model_name, "benchmark": "qampari_agentic_latentmas",
        "n_tasks": int(n), "n_agents": args.n_agents,
        "max_react_steps": args.max_react_steps, "m_latent": args.m_latent,
        "max_new_tokens": 4096,
        "idle_power_w": idle_w,
        # loose_accuracy (recall_substr) — matches main experiment format
        "single_loose_accuracy": agg["single_acc"] / n,
        "text_loose_accuracy": agg["text_acc"] / n,
        "latent_loose_accuracy": agg["latent_acc"] / n,
        # F1 for reference
        "single_f1": agg["single_f1"] / n,
        "text_f1": agg["text_f1"] / n,
        "latent_f1": agg["latent_f1"] / n,
        # Energy
        "single_energy_j_mean": agg["single_energy"] / n,
        "text_energy_j_mean": agg["text_energy"] / n,
        "latent_energy_j_mean": agg["latent_energy"] / n,
        "latent_vs_text_energy_pct": ((agg["latent_energy"] - agg["text_energy"])
                                      / max(agg["text_energy"], 0.01)) * 100,
        "elapsed_sec": time.time() - t0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  {'Condition':12s} {'Accuracy':>10s} {'F1':>8s} {'Energy':>10s}")
    print(f"  {'Single':12s} {summary['single_loose_accuracy']:10.3f} {summary['single_f1']:8.3f} {summary['single_energy_j_mean']:10.0f} J")
    print(f"  {'Text MAS':12s} {summary['text_loose_accuracy']:10.3f} {summary['text_f1']:8.3f} {summary['text_energy_j_mean']:10.0f} J")
    print(f"  {'Latent MAS':12s} {summary['latent_loose_accuracy']:10.3f} {summary['latent_f1']:8.3f} {summary['latent_energy_j_mean']:10.0f} J")
    print(f"  Latent vs Text energy: {summary['latent_vs_text_energy_pct']:+.1f}%")
    baselines = {"qampari": "0.183", "fanoutqa": "0.577"}
    print(f"  Main experiment SAS k=10 baseline: loose_accuracy={baselines.get(args.benchmark, '?')} "
          f"(Qwen3.5-9B, SGLang)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
