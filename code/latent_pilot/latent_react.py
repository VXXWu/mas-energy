"""Latent ReAct: interleaved latent thinking + text tool calling.

Within a single agent's execution, alternates between:
  - THINK phase: m latent thought steps (cheap, no text decode)
  - ACT phase: decode text tokens until tool call detected
  - OBSERVE phase: prefill tool result tokens into context
  - Repeat for k steps, then decode final answer

The KV cache accumulates both latent and text positions seamlessly.
Attention doesn't distinguish between them — both are valid KV entries.

For inter-agent communication: after an agent completes its Latent ReAct
loop, its full KV cache (containing text + latent positions) is shared
as working memory with the next agent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import torch
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.latentmas_baseline import (  # noqa: E402
    QWEN3_MODEL_ID,
    _cache_seq_len,
    apply_alignment,
    compute_alignment,
    load_qwen3,
)


def find_lm_head(model) -> torch.nn.Module:
    for path in [("lm_head",), ("model", "lm_head")]:
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate lm_head")


def find_embed_tokens(model) -> torch.nn.Module:
    for path in [("model", "embed_tokens"), ("model", "model", "embed_tokens")]:
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate embed_tokens")


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from generated text.

    Qwen3 may enter thinking mode spontaneously during the ACT phase.
    Tool calls appear AFTER the thinking block. Stripping <think> blocks
    lets detect_tool_call find tool calls that follow reasoning.
    Also handles unclosed <think> blocks (thinking that hit the token limit).
    """
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove unclosed <think>... (thinking that was truncated)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def detect_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a tool call from generated text. Supports:
      - Qwen3 native: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
      - Python-style: search("query") or search(query="...")
      - ReAct-style: Action: search / Action Input: query

    Strips <think> blocks first so tool calls after reasoning are found.
    Returns (tool_name, {arg_name: value}) or None.
    """
    import json as _json

    # Strip thinking blocks that may precede the tool call
    text = strip_think_blocks(text)

    # Qwen3 native format (primary): <tool_call>{"name": ..., "arguments": ...}</tool_call>
    tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if tc_match:
        try:
            obj = _json.loads(tc_match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = _json.loads(args)
            return (name, args)
        except (_json.JSONDecodeError, TypeError):
            pass

    # Qwen3 partial (tool_call tag opened but not closed yet — detect early)
    tc_partial = re.search(r'<tool_call>\s*\{[^}]*"name"\s*:\s*"(\w+)"[^}]*"query"\s*:\s*"([^"]+)"', text)
    if tc_partial:
        return (tc_partial.group(1), {"query": tc_partial.group(2)})

    # Python-style: search("query")
    py_match = re.search(r'search\s*\(\s*(?:query\s*=\s*)?["\']([^"\']+)["\']', text)
    if py_match:
        return ("search", {"query": py_match.group(1)})

    # ReAct-style: Action: search / Action Input: query
    action_match = re.search(r'Action:\s*search\s*\n\s*Action Input:\s*(.+?)(?:\n|$)', text)
    if action_match:
        return ("search", {"query": action_match.group(1).strip().strip('"\'')})

    return None


def latent_react_loop(
    model, tokenizer, input_ids: torch.Tensor,
    W_e: torch.Tensor, target_norm: float,
    executor: Callable[[str, dict], str],
    m_think: int = 15,
    max_react_steps: int = 10,
    max_tokens_per_act: int = 150,
    max_answer_tokens: int = 256,
    past_kv=None,
) -> tuple[str, object, dict]:
    """Run a Latent ReAct loop: think(latent) → act(text) → observe(text).

    Args:
        model: the LLM
        tokenizer: tokenizer
        input_ids: initial prompt token IDs [1, seq]
        W_e, target_norm: alignment matrix and norm for latent thoughts
        executor: tool execution function (tool_name, args) → result_text
        m_think: latent thought steps per think phase
        max_react_steps: maximum tool-calling rounds
        max_tokens_per_act: max tokens to decode when generating tool call / answer
        max_answer_tokens: max tokens for final answer
        past_kv: optional working memory from a prior agent

    Returns:
        (final_answer_text, kv_cache, stats_dict)
    """
    dev = input_ids.device
    lm_head = find_lm_head(model)
    embed_mod = find_embed_tokens(model)

    stats = {"latent_steps": 0, "text_tokens": 0, "tool_calls": 0, "react_steps": 0}

    # Initial forward pass on input tokens
    if past_kv is not None:
        prefix_len = _cache_seq_len(past_kv)
        cache_position = torch.arange(prefix_len, prefix_len + input_ids.shape[1],
                                       device=dev, dtype=torch.long)
        attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        past_key_values=past_kv, cache_position=cache_position,
                        use_cache=True, output_hidden_states=True, return_dict=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True,
                        output_hidden_states=True, return_dict=True)

    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]
    stats["text_tokens"] += input_ids.shape[1]

    for step in range(max_react_steps):
        stats["react_steps"] += 1

        # ── ACT phase FIRST: decode text until tool call or answer ──
        # Let the model decide to search or answer BEFORE latent thinking.
        # Latent thoughts process search results, not replace the search.
        generated_tokens = []
        generated_text = ""
        found_tool = False

        for _ in range(max_tokens_per_act):
            with torch.no_grad():
                logits = lm_head(h_t.to(lm_head.weight.dtype))  # [1, 1, vocab]
            next_token_id = logits[:, -1, :].argmax(dim=-1)  # [1]
            token_text = tokenizer.decode(next_token_id, skip_special_tokens=False)
            generated_tokens.append(next_token_id.item())
            generated_text += token_text
            stats["text_tokens"] += 1

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Check for tool call
            tool_call = detect_tool_call(generated_text)
            if tool_call is not None:
                found_tool = True
                break

            # Continue decode: embed the token and forward
            with torch.no_grad():
                token_embed = embed_mod(next_token_id.unsqueeze(0))  # [1, 1, d]
                step_out = model(inputs_embeds=token_embed, past_key_values=kv_cache,
                                use_cache=True, output_hidden_states=True, return_dict=True)
            kv_cache = step_out.past_key_values
            h_t = step_out.hidden_states[-1][:, -1:, :]

        if not found_tool:
            # No tool call found — this is the final answer
            # Continue decoding until EOS or max_answer_tokens
            remaining = max_answer_tokens - len(generated_tokens)
            for _ in range(remaining):
                with torch.no_grad():
                    logits = lm_head(h_t.to(lm_head.weight.dtype))
                next_token_id = logits[:, -1, :].argmax(dim=-1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                generated_tokens.append(next_token_id.item())
                generated_text += tokenizer.decode(next_token_id, skip_special_tokens=False)
                stats["text_tokens"] += 1
                with torch.no_grad():
                    token_embed = embed_mod(next_token_id.unsqueeze(0))
                    step_out = model(inputs_embeds=token_embed, past_key_values=kv_cache,
                                    use_cache=True, output_hidden_states=True, return_dict=True)
                kv_cache = step_out.past_key_values
                h_t = step_out.hidden_states[-1][:, -1:, :]

            return generated_text, kv_cache, stats

        # ── Forward the final un-cached ACT token before observing ──
        # The ACT loop forwards each token EXCEPT the last one (where the
        # tool call was detected and we broke). Forward that last token so
        # the model "remembers" its tool call in the KV cache.
        if generated_tokens:
            last_tok = torch.tensor([[generated_tokens[-1]]], device=dev, dtype=torch.long)
            with torch.no_grad():
                last_embed = embed_mod(last_tok)
                last_out = model(inputs_embeds=last_embed, past_key_values=kv_cache,
                                use_cache=True, output_hidden_states=True, return_dict=True)
            kv_cache = last_out.past_key_values
            h_t = last_out.hidden_states[-1][:, -1:, :]

        # ── OBSERVE phase: execute tool and prefill result ──
        tool_name, tool_args = tool_call
        stats["tool_calls"] += 1
        result_text = executor(tool_name, tool_args)

        # Format observation using Qwen3's expected tool_response tags
        obs_text = f"\n<tool_response>\n{result_text}\n</tool_response>\n"
        obs_ids = tokenizer(obs_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)

        obs_prefix_len = _cache_seq_len(kv_cache)
        obs_cache_pos = torch.arange(
            obs_prefix_len, obs_prefix_len + obs_ids.shape[1], device=dev, dtype=torch.long
        )
        obs_attn_mask = torch.ones(
            (1, obs_prefix_len + obs_ids.shape[1]), device=dev, dtype=torch.long
        )

        with torch.no_grad():
            obs_out = model(input_ids=obs_ids, past_key_values=kv_cache,
                           attention_mask=obs_attn_mask, cache_position=obs_cache_pos,
                           use_cache=True, output_hidden_states=True, return_dict=True)
        kv_cache = obs_out.past_key_values
        h_t = obs_out.hidden_states[-1][:, -1:, :]
        stats["text_tokens"] += obs_ids.shape[1]

        # ── THINK phase: process the observation in latent space ──
        # Latent thoughts AFTER tool results let the model reason about
        # evidence without consuming decode tokens. This is where the
        # energy savings come from.
        for _ in range(m_think):
            e_next = apply_alignment(h_t, W_e, target_norm)
            with torch.no_grad():
                step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                                use_cache=True, output_hidden_states=True, return_dict=True)
            kv_cache = step_out.past_key_values
            h_t = step_out.hidden_states[-1][:, -1:, :]
            stats["latent_steps"] += 1

    # Exhausted react steps — decode whatever we have
    return generated_text, kv_cache, stats
