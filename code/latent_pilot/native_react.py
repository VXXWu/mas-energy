"""Native-tool-calling ReAct loop for HF transformers.

Replaces `agentic_latentmas.text_react_loop`'s raw-text + regex-parse approach
with the model's actual chat template + tools (the same template SGLang's
qwen3_coder parser wraps). The wire format the model sees is identical
between this loop and the main study's `llm.react_loop` path; the only
substantive difference is that SGLang can constrain decoding to the tool
format while HF cannot, so a small fraction of HF tool calls may be
slightly malformed.

This is path (c) from the 2026-06-02 LATENT_HANDOFF correction: both the
text and latent variants of the pilot now use the same tool-calling
protocol the model was trained on, removing the pilot's regex parser as a
confound vs the main study.

Usage:
    from latent_pilot.native_react import native_react_loop
    response = native_react_loop(model_wrapper, messages, tools, executor,
                                 max_steps=10, temperature=0.5)
"""
from __future__ import annotations

import json
import re
from typing import Any

import torch


# Qwen3 chat template wraps tool calls in <tool_call>...</tool_call> blocks
# containing JSON. The same format SGLang's qwen3_coder parser extracts.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_calls(response: str) -> list[dict]:
    """Extract all <tool_call>...</tool_call> JSON blocks from a response.

    Returns a list of dicts with 'name' and 'arguments' keys (arguments
    may be dict or str depending on what the model emitted).
    """
    calls = []
    for m in TOOL_CALL_RE.finditer(response):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if name:
            calls.append({"name": name, "arguments": args})
    return calls


def _content_outside_tool_calls(response: str) -> str:
    """Return the response text with any tool_call blocks removed.

    Used to capture any natural-language content the model wrote alongside
    its tool calls (the equivalent of msg.content in the OpenAI API).
    """
    return TOOL_CALL_RE.sub("", response).strip()


def native_react_loop(model_wrapper, messages: list[dict], tools: list[dict],
                      executor, max_steps: int = 10, temperature: float = 0.5,
                      max_new_tokens: int = 4096,
                      max_prompt_tokens: int = 4096) -> str:
    """Run a ReAct loop using the model's native chat template + tools.

    Mutates `messages` in place (appends assistant turns, tool turns, etc.).

    Args:
        model_wrapper: object with `.tokenizer` (HF tokenizer with
            apply_chat_template supporting tools=) and `.model` (HF causal LM).
        messages: list of {role, content} dicts. May already include a
            system prompt; will be extended in-place with assistant turns
            (with optional tool_calls) and tool result messages.
        tools: list of OpenAI function-calling schemas (one per tool).
        executor: callable(name: str, args: dict) -> str | dict, returns
            the tool result. Synchronous.
        max_steps: max tool-calling rounds before forcing a final response.
        temperature: sampling temperature (0 = greedy, >0 = sample with top-p).
        max_new_tokens: per-step decode cap.
        max_prompt_tokens: hard truncation cap for the prompt (matches the
            existing pilot's 4096 setting; keep aligned).

    Returns:
        The model's final text response (string). On error or overflow,
        returns whatever content was last emitted, never None.
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    device = model.device
    valid_tool_names = {t["function"]["name"] for t in tools}

    for step in range(max_steps):
        # Build prompt using the model's native chat template WITH tools.
        # This is the same wire format SGLang sends; the model sees
        # <|im_start|>system\n...\n<|tool_definitions>...</tool_definitions>\n...
        # (exact format depends on the chat template) — the format it was
        # trained to expect.
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_prompt_tokens)
        input_ids = enc.input_ids.to(device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=enc.attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        response = tokenizer.decode(out[0, input_ids.shape[1]:],
                                    skip_special_tokens=False)
        # Strip end-of-sequence markers that survive skip_special_tokens=False
        for stop_token in ("<|im_end|>", tokenizer.eos_token or ""):
            if stop_token and stop_token in response:
                response = response.split(stop_token, 1)[0]

        tool_calls = _parse_tool_calls(response)
        content_part = _content_outside_tool_calls(response)

        # No tool calls → model produced a final answer
        if not tool_calls:
            messages.append({"role": "assistant", "content": response.strip()})
            return response.strip()

        # Filter out hallucinated tool names (matches main study's react_loop
        # behavior — treat as termination if ALL emitted tool names are bogus).
        real_calls = [tc for tc in tool_calls if tc["name"] in valid_tool_names]
        if not real_calls:
            messages.append({"role": "assistant", "content": content_part})
            return content_part

        # Append the assistant turn with tool_calls (proper OpenAI format).
        # Each tool_call gets a synthetic id (Qwen3 doesn't emit ids; we
        # generate "call_{step}_{i}" for parity with SGLang's behavior).
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

        # Execute each tool and append a proper {role: tool} result message
        # (NOT a fake {role: user, content: "Tool result: ..."} like the old
        # text_react_loop did — that was the off-distribution pattern that
        # caused the main study comparison gap).
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

    # max_steps exhausted → force a no-tools final response
    prompt = tokenizer.apply_chat_template(
        messages,
        # Don't pass tools= here; we want the model to give a final answer.
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids.to(device),
            attention_mask=enc.attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    final = tokenizer.decode(out[0, enc.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": final})
    return final
