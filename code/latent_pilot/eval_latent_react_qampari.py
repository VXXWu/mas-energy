"""Evaluate Latent ReAct on QAMPARI — agentic tool calling with latent thinking.

Three conditions:
  1. Single agent, standard ReAct (text only, k=10 steps)
  2. Text-debate MAS: 3 agents do ReAct independently, share text answers,
     revise via text
  3. Latent-ReAct MAS: 3 agents do Latent ReAct (latent think + text tools),
     share working memory via KV cache, final agent decodes answer

This is the full hybrid latent-tool topology on an agentic benchmark.

Outputs:
  results/latent_pilot/eval_latent_react_qampari.jsonl
  results/latent_pilot/eval_latent_react_summary.json
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.latentmas_baseline import (  # noqa: E402
    QWEN3_MODEL_ID,
    _cache_seq_len,
    compute_alignment,
    load_qwen3,
)
from latent_pilot.latent_react import latent_react_loop, detect_tool_call  # noqa: E402
from latent_pilot.utils import append_jsonl, results_root, save_json  # noqa: E402
from benchmarks_qampari import QampariBenchmark, evaluate_qampari  # noqa: E402


# Use the same prompts as the production decentralized topology
try:
    from prompts import DEBATE_AGENT_PROMPT, DEBATE_SYNTHESIZER_PROMPT
except ImportError:
    DEBATE_AGENT_PROMPT = ("You are a helpful assistant with access to tools, participating in a "
                           "collaborative problem-solving process. Use tools to investigate the task "
                           "thoroughly. Provide your reasoning and final answer clearly.")
    DEBATE_SYNTHESIZER_PROMPT = ("You are a synthesis agent. Given multiple agents' responses after "
                                 "debate, synthesize the best final answer based on all agents' work.")


def run_single_react(model, tokenizer, question, executor, max_steps=10):
    """Single agent text ReAct via HF transformers.

    Uses DEBATE_AGENT_PROMPT (same as production decentralized) with Qwen3
    native tool-call format. Thinking mode disabled to match production config.
    """
    dev = next(model.parameters()).device
    gen_config = GenerationConfig(max_new_tokens=512, do_sample=False)

    # Build prompt via chat template with enable_thinking=False (matches
    # production config.py extra_body). This prevents the model from entering
    # <think> mode, which would consume the token budget without tool calls.
    messages = [
        {"role": "system", "content": (
            f"{DEBATE_AGENT_PROMPT}\n\n"
            f"You have access to a search tool. To search, output:\n"
            f'<tool_call>\n{{"name": "search", "arguments": {{"query": "your query"}}}}\n</tool_call>'
        )},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    total_tokens = 0

    for step in range(max_steps):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        with torch.no_grad():
            out = model.generate(input_ids=enc.input_ids.to(dev),
                                attention_mask=enc.attention_mask.to(dev),
                                generation_config=gen_config)
        new_text = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        total_tokens += out.shape[1] - enc.input_ids.shape[1]
        prompt += new_text

        tool = detect_tool_call(new_text)
        if tool is None:
            break
        tool_name, tool_args = tool
        result = executor(tool_name, tool_args)
        prompt += f"\n<tool_response>\n{result}\n</tool_response>\n\n"

    answer = new_text if 'new_text' in dir() else ""
    return answer, {"text_tokens": total_tokens, "tool_calls": step + 1}


def run_latent_react_mas(model, tokenizer, question, executor, W_e, target_norm,
                         m_think=15, max_steps=5, n_agents=3):
    """Latent-ReAct MAS: agents do Latent ReAct, share working memory.

    Uses DEBATE_AGENT_PROMPT (same as production decentralized topology).
    """
    dev = next(model.parameters()).device

    working_memory = None
    all_stats = []

    for agent_idx in range(n_agents):
        if agent_idx < n_agents - 1:
            user_msg = f"{question}\n\nSearch for evidence and reason about the question."
        else:
            user_msg = (f"{question}\n\nBased on all prior research, provide the final "
                       f"complete answer as a comma-separated list.")

        messages = [
            {"role": "system", "content": (
                f"{DEBATE_AGENT_PROMPT}\n\n"
                f"You have access to a search tool. To search, output:\n"
                f'<tool_call>\n{{"name": "search", "arguments": {{"query": "your query"}}}}\n</tool_call>'
            )},
            {"role": "user", "content": user_msg},
        ]
        agent_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(agent_prompt, return_tensors="pt", truncation=True, max_length=2048)

        answer, kv_cache, stats = latent_react_loop(
            model, tokenizer, enc.input_ids.to(dev),
            W_e, target_norm, executor,
            m_think=m_think,
            max_react_steps=max_steps,
            past_kv=working_memory,
        )
        all_stats.append(stats)

        if agent_idx < n_agents - 1:
            working_memory = kv_cache
        else:
            combined_stats = {
                "total_latent_steps": sum(s["latent_steps"] for s in all_stats),
                "total_text_tokens": sum(s["text_tokens"] for s in all_stats),
                "total_tool_calls": sum(s["tool_calls"] for s in all_stats),
            }
            return answer, combined_stats

    return "", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--m-think", type=int, default=15)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qampari-data-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"k{args.max_react_steps}_m{args.m_think}"
    jsonl_path = out_dir / f"eval_latent_react_qampari_{run_tag}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    model, tok = load_qwen3()
    print("Computing alignment matrix...")
    W_e, target_norm = compute_alignment(model)

    bench = QampariBenchmark(data_dir=args.qampari_data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} QAMPARI tasks")

    agg = {k: 0 for k in ["single_f1", "latent_f1", "n"]}
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task["question_text"]
        executor, cleanup = bench.make_executor(task)

        try:
            # Single-agent text ReAct (same prompts/tools as production)
            single_answer, single_stats = run_single_react(
                model, tok, question, executor, max_steps=args.max_react_steps)
            single_eval = evaluate_qampari(task["answer_list"], single_answer)

            # Latent-ReAct MAS
            latent_answer, latent_stats = run_latent_react_mas(
                model, tok, question, executor, W_e, target_norm,
                m_think=args.m_think, max_steps=args.max_react_steps,
                n_agents=args.n_agents)
            latent_eval = evaluate_qampari(task["answer_list"], latent_answer)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            cleanup()
            continue

        cleanup()

        if i == 0:
            print(f"\n--- First task diagnostic ---")
            print(f"  Q: {question[:100]}...")
            print(f"  Single: {single_answer[:200]}...")
            print(f"  Single stats: {single_stats}")
            print(f"  Single F1={single_eval['f1']:.3f}")
            print(f"  Latent: {latent_answer[:200]}...")
            print(f"  Latent stats: {latent_stats}")
            print(f"  Latent F1={latent_eval['f1']:.3f}")
            print()

        rec = {
            "i": i, "qid": task["qid"], "question": question[:100],
            "single_answer": single_answer[:300], "single_f1": single_eval["f1"],
            "single_stats": single_stats,
            "latent_answer": latent_answer[:300], "latent_f1": latent_eval["f1"],
            "latent_stats": latent_stats,
        }
        append_jsonl(rec, jsonl_path)

        agg["single_f1"] += single_eval["f1"]
        agg["latent_f1"] += latent_eval["f1"]
        agg["n"] += 1

        if (i + 1) % 5 == 0:
            n = agg["n"]
            print(f"[{i+1:3d}/{len(tasks)}] single={agg['single_f1']/n:.3f} "
                  f"latent={agg['latent_f1']/n:.3f} ({(i+1)/(time.time()-t0):.2f}/s)")

    if agg["n"] == 0:
        print("No tasks completed")
        return

    n = agg["n"]
    summary = {
        "model": QWEN3_MODEL_ID,
        "benchmark": "qampari_latent_react",
        "n_tasks": n,
        "n_agents": args.n_agents,
        "m_think": args.m_think,
        "max_react_steps": args.max_react_steps,
        "single_f1": agg["single_f1"] / n,
        "latent_f1": agg["latent_f1"] / n,
        "delta": (agg["latent_f1"] - agg["single_f1"]) / n,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, out_dir / f"eval_latent_react_summary_{run_tag}.json")

    print(f"\n{'='*60}")
    print("LATENT REACT QAMPARI EVALUATION")
    print(f"  Single ReAct:      F1={summary['single_f1']:.3f}")
    print(f"  Latent-ReAct MAS:  F1={summary['latent_f1']:.3f}")
    print(f"  Delta:             {summary['delta']:+.3f}")


if __name__ == "__main__":
    main()
