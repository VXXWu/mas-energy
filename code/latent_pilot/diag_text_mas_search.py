"""Diagnostic: trace what happens when text MAS runs ONE FanOutQA task.

Question: with the spec-matched prompt (no TOOL_FORMAT_INSTRUCTION) and the
native tool calling path, does the model actually emit <tool_call> blocks?

Run on cluster with:
  conda activate mas-energy-pilot
  cd /atlas2/u/$USER/mas_project
  PYTHONPATH=mas-energy/code python -m latent_pilot.diag_text_mas_search

Prints:
  - The exact chat-template-rendered prompt the model sees (system + user + tools)
  - The model's raw generation output (FULL, not truncated to 500 chars)
  - Whether <tool_call> blocks were detected
  - If yes, the parsed tool calls and result of executing them
  - If no, what content the model produced instead

Conclusions to draw from the output:
  - If the prompt contains the tool schema but the response has NO <tool_call>
    block: model is choosing not to call tools (prompt or training issue).
  - If the prompt has NO tool schema: chat template / tools pass-through is broken.
  - If response has <tool_call> but in a different format than the regex
    expects: parser bug.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Standard pilot path bootstrap (matches eval_hybrid_diffusion.py)
_USER = os.environ["USER"]
sys.path.insert(0, f"/atlas2/u/{_USER}/mas_project/mas-energy/code")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_pilot.agentic_latentmas import (
    build_react_prompt, _parse_tool_calls,
)
from latent_pilot.native_react import _parse_tool_calls as _native_parse_tool_calls
from benchmarks_fanoutqa import FANOUTQA_TOOLS, FanOutQAExecutor, load_tasks


def main():
    print("=" * 80)
    print("Pilot text-MAS search diagnostic")
    print("=" * 80)

    print("\n[1] Loading FanOutQA — first task...")
    tasks = list(load_tasks())[:1]
    task = tasks[0]
    question = task["question"]
    print(f"    Task ID: {task.get('id')}")
    print(f"    Question: {question[:200]}")

    print("\n[2] Building system+user messages via build_react_prompt...")
    msgs = build_react_prompt(question)
    for m in msgs:
        print(f"    [{m['role']}] {m['content'][:300]}")

    print("\n[3] Loading Qwen3.5-9B tokenizer + applying chat template WITH tools...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    rendered = tok.apply_chat_template(
        msgs, tools=FANOUTQA_TOOLS, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    print(f"    Rendered prompt length: {len(rendered)} chars")
    print("    --- FIRST 2500 chars of rendered prompt ---")
    print(rendered[:2500])
    print("    --- (rest truncated) ---")
    # Critical check: does the rendered prompt actually contain the tool schema?
    has_search_tool = '"search"' in rendered or 'name=search' in rendered
    has_tool_marker = "<tools>" in rendered or "tool_call" in rendered or "function" in rendered.lower()
    print(f"\n    ✓ contains 'search' tool name: {has_search_tool}")
    print(f"    ✓ contains tool marker (<tools>/tool_call/function): {has_tool_marker}")
    if not has_search_tool and not has_tool_marker:
        print("    ❌ TOOLS NOT IN RENDERED PROMPT — chat template isn't injecting them.")
        print("       This means the model has no way to know tools exist.")

    print("\n[4] Loading Qwen3.5-9B model (bf16, may take ~1 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda",
    )
    model.eval()

    print("\n[5] Generating response (temperature=0.5, max_new_tokens=2048)...")
    enc = tok(rendered, return_tensors="pt", truncation=False).to(model.device)
    eos_ids = [tok.eos_token_id]
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=2048, do_sample=True,
            temperature=0.5, top_p=0.95,
            eos_token_id=eos_ids,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    response = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"    Response length: {len(response)} chars")
    print("    --- RAW RESPONSE ---")
    print(response)
    print("    --- END RAW RESPONSE ---")

    print("\n[6] Checking if response contains <tool_call> blocks...")
    pilot_calls = _native_parse_tool_calls(response)
    print(f"    Pilot regex parser found {len(pilot_calls)} tool calls")
    for i, tc in enumerate(pilot_calls):
        print(f"      [{i}] name={tc.get('name')}  args={tc.get('arguments')}")

    if not pilot_calls:
        print("\n    ❌ NO TOOL CALLS in model response.")
        print("       The model chose to answer directly from internal knowledge.")
        print("       Possible causes:")
        print("       (a) Tools not in prompt (check section [3] above)")
        print("       (b) Model decided knowledge sufficed — could be specific to question")
        print("       (c) System prompt doesn't push hard enough toward tool use")
        print("       (d) Qwen3.5-9B's training has weak tool-using bias")
        print("\n       Counter-test: re-run with a question requiring lookup the model")
        print("       can't possibly know (e.g., 'What was the closing price of AAPL on")
        print("       2024-05-15?') and see if it still skips tools.")
    else:
        print("\n    ✓ Model emitted tool calls. Executing one to confirm tool integration works...")
        ex = FanOutQAExecutor(question)
        tc = pilot_calls[0]
        result = ex(tc["name"], tc.get("arguments", {}) if isinstance(tc.get("arguments"), dict) else {"query": tc.get("arguments")})
        print(f"    Tool result: {str(result)[:500]}")


if __name__ == "__main__":
    main()
