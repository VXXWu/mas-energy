"""HF-transformers-based decentralized text-MAS transcript generator.

SGLang-free alternative to Stage 0 of the pilot. Produces JSONL records
in the call_records format expected by extract_training_pairs.py:

    {
      "task_id": ..., "topology": "decentralized",
      "loose_accuracy": ..., "correct": ...,
      "call_records": [
        {"agent_id": "debater_0_init", "call_type": "llm",
         "request_messages": [...], "response": "...",
         "prompt_tokens": int, "completion_tokens": int},
        {"agent_id": "debater_0_r0", "call_type": "llm", ...},
        {"agent_id": "synthesizer",  "call_type": "llm", ...},
      ]
    }

Output file name matches `*decentralized*.jsonl` so extract_training_pairs
picks it up without modification.

Why this exists: the SGLang-based Stage 0 of run_pilot_4b.sbatch has been
blocked since mid-May by an sm100-only sgl_kernel binary issue on the
A5000s (CUDA 13 libnvrtc missing, no sm86 binaries shipped in recent
builds). Multiple setup-script repair attempts (sklearn install, multi-
version sgl_kernel probe) have not succeeded as of May 26. This bypass
sidesteps SGLang entirely; the cost is slower per-task throughput, but
the pilot only needs ~30 tasks for pair extraction.

Usage:
    python -m latent_pilot.gen_transcripts_hf \\
        --model-name Qwen/Qwen3-4B \\
        --benchmark fanoutqa \\
        --n-tasks 30 \\
        --max-react-steps 5 \\
        --n-agents 3 \\
        --output-dir mas-energy/results/diffusion_pilot_XYZ/transcripts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

_USER = os.environ["USER"]
REPO_DIR = os.environ.get("LATENTMAS_REPO",
    str(Path(f"/atlas2/u/{_USER}/mas_project/LatentMAS")))
CODE_DIR = os.environ.get("MAS_ENERGY_CODE",
    str(Path(f"/atlas2/u/{_USER}/mas_project/mas-energy/code")))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_record(agent_id, messages, response, tokenizer):
    """One call_records entry. Matches the format extract_training_pairs reads."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tok = len(tokenizer(prompt_text).input_ids)
    completion_tok = len(tokenizer(response or "").input_ids)
    return {
        "agent_id": agent_id,
        "call_type": "llm",
        "request_messages": list(messages),  # snapshot
        "response": response or "",
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
    }


def run_one_task(model_wrapper, executor, question, task, evaluate_fn,
                 n_agents, max_react_steps, n_rounds, tokenizer):
    """Run decentralized debate on one task, accumulating call_records."""
    from latent_pilot.agentic_latentmas import (
        build_react_prompt, text_react_loop, _format_debate_prompt,
        _format_synthesis, generate_text,
        DEBATE_SYNTHESIZER_PROMPT, TOOL_INSTRUCTION,
    )

    call_records = []

    # Phase 1: independent ReAct per agent
    trajectories = []
    for i in range(n_agents):
        msgs = build_react_prompt(question, TOOL_INSTRUCTION)
        # Snapshot the input BEFORE text_react_loop mutates `msgs` in place
        # (text_react_loop appends tool calls and results to msgs).
        msgs_before = list(msgs)
        response = text_react_loop(
            model_wrapper, msgs, executor,
            max_steps=max_react_steps, temperature=0.5,
        )
        call_records.append(make_record(
            f"debater_{i}_init", msgs_before, response, tokenizer,
        ))
        trajectories.append({"final_response": response, "messages": msgs})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Phase 2: debate rounds
    for round_idx in range(n_rounds):
        new_trajectories = []
        for i in range(n_agents):
            debate_msg = _format_debate_prompt(trajectories, exclude_idx=i)
            msgs = trajectories[i]["messages"] + [
                {"role": "user", "content": debate_msg}
            ]
            msgs_before = list(msgs)
            response = text_react_loop(
                model_wrapper, msgs, executor,
                max_steps=max_react_steps, temperature=0.5,
            )
            # agent_id MUST match `_r\d+` for is_debate_round_call() to accept it
            call_records.append(make_record(
                f"debater_{i}_r{round_idx}", msgs_before, response, tokenizer,
            ))
            new_trajectories.append({
                "final_response": response, "messages": msgs,
            })
        trajectories = new_trajectories

    # Phase 3: synthesizer
    synth_msgs = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user",   "content": _format_synthesis(question, trajectories)},
    ]
    synth_ans = generate_text(model_wrapper, synth_msgs, temperature=0.0)
    call_records.append(make_record("synthesizer", synth_msgs, synth_ans,
                                    tokenizer))

    # Eval
    loose, f1, eval_extra = evaluate_fn(task, synth_ans)
    return {
        "task_id": task.get("qid") or task.get("id"),
        "topology": "decentralized",
        "answer": synth_ans,
        "loose_accuracy": loose,
        "correct": eval_extra.get("correct") if isinstance(eval_extra, dict) else None,
        "f1": f1,
        "call_records": call_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-4B")
    ap.add_argument("--benchmark", default="fanoutqa",
                    choices=["qampari", "fanoutqa"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=1)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")
    out_path = args.output_dir / f"{model_tag}_decentralized.jsonl"
    print(f"Output: {out_path}")

    # Use the SAME SimpleModelWrapper pattern as agentic_latentmas.main()
    # (LatentMAS's ModelWrapper eagerly builds a W_e realignment matrix on
    # init, OOMs on large vocabs; we don't need it here).
    print(f"Loading {args.model_name} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class SimpleModelWrapper:
        def __init__(self, model_name, device="cuda"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=True,
            ).to(device)
            self.model.eval()

    mw = SimpleModelWrapper(args.model_name, device="cuda")
    print(f"Model loaded on {mw.model.device}")

    # Benchmark
    from latent_pilot.agentic_latentmas import load_benchmark
    bench, evaluate_fn, question_key = load_benchmark(
        args.benchmark, data_dir=args.data_dir,
    )
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} {args.benchmark} tasks")

    t0 = time.time()
    n_ok = 0
    with open(out_path, "w") as f:
        for ti, task in enumerate(tasks):
            executor, cleanup = bench.make_executor(task)
            question = task[question_key]
            try:
                rec = run_one_task(
                    mw, executor, question, task, evaluate_fn,
                    args.n_agents, args.max_react_steps, args.n_rounds,
                    mw.tokenizer,
                )
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                n_ok += 1
                n_calls = len(rec["call_records"])
                elapsed = time.time() - t0
                eta = elapsed / max(1, ti + 1) * (len(tasks) - ti - 1)
                print(f"  [{ti+1}/{len(tasks)}] {n_calls} calls, "
                      f"loose={rec['loose_accuracy']:.3f} "
                      f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            except Exception as e:
                print(f"  [{ti+1}] FAILED: {e}")
                traceback.print_exc()
            try:
                cleanup()
            except Exception:
                pass

    print(f"\nDone: {n_ok}/{len(tasks)} tasks ({(time.time()-t0)/60:.1f} min)")
    print(f"Transcript: {out_path}")


if __name__ == "__main__":
    main()
