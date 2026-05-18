"""Evaluate LatentMAS on QAMPARI (closed-book, no tool calling).

Runs the LatentMAS sequential protocol on Qwen3-8B:
  Agent 1 (planner): m latent thoughts on the question
  Agent 2 (refiner): m latent thoughts conditioned on Agent 1's working memory
  Agent 3 (solver): decodes text answer conditioned on Agent 1+2's working memory

Evaluates using QAMPARI F1 (official string-matching protocol).

This is closed-book (no BM25 search tool) because latent thoughts cannot
express tool calls. The comparison baseline is single-agent closed-book.
Tool-calling integration requires a hybrid protocol (text for tools,
latent for debate) which is a separate engineering task.

Outputs:
  results/latent_pilot/eval_qampari_latent_m{M}.jsonl
  results/latent_pilot/eval_qampari_summary.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.latentmas_baseline import (  # noqa: E402
    QWEN3_MODEL_ID,
    _cache_seq_len,
    apply_alignment,
    compute_alignment,
    load_qwen3,
)
from latent_pilot.utils import append_jsonl, results_root, save_json  # noqa: E402
from benchmarks_qampari import QampariBenchmark, evaluate_qampari  # noqa: E402


def generate_latent_thoughts(model, input_ids, W_e, target_norm, m, past_kv=None):
    """Generate m latent thoughts, optionally conditioned on prior working memory.

    If past_kv is provided, Agent's input tokens are appended after the
    existing cache (position-shifted correctly).
    """
    dev = input_ids.device

    if past_kv is not None:
        prefix_len = _cache_seq_len(past_kv)
        cache_position = torch.arange(
            prefix_len, prefix_len + input_ids.shape[1], device=dev, dtype=torch.long
        )
        attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
    else:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    for step in range(m):
        e_next = apply_alignment(h_t, W_e, target_norm)
        with torch.no_grad():
            step_out = model(
                inputs_embeds=e_next,
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        kv_cache = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]

    return kv_cache, h_t


def decode_text(model, tokenizer, past_kv, prompt_ids, max_new_tokens=256):
    """Decode text answer conditioned on accumulated working memory."""
    dev = prompt_ids.device
    prefix_len = _cache_seq_len(past_kv)
    cache_position = torch.arange(
        prefix_len, prefix_len + prompt_ids.shape[1], device=dev, dtype=torch.long
    )
    attn_mask = torch.ones((1, prefix_len + prompt_ids.shape[1]), device=dev, dtype=torch.long)

    generated_ids = prompt_ids.clone()
    current_kv = past_kv

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated_ids[:, -1:] if generated_ids.shape[1] > prompt_ids.shape[1] else generated_ids,
                attention_mask=torch.ones((1, _cache_seq_len(current_kv) + 1), device=dev, dtype=torch.long),
                past_key_values=current_kv,
                cache_position=torch.tensor([_cache_seq_len(current_kv)], device=dev, dtype=torch.long),
                use_cache=True,
                return_dict=True,
            )
        current_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated_ids[:, prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def run_single_agent_baseline(model, tokenizer, question: str, max_new_tokens=256):
    """Single-agent closed-book baseline: just generate answer directly."""
    prompt = (
        "You are a knowledgeable research assistant. Answer the following question "
        "by listing all relevant entities you know. Separate answers by commas.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids.to(dev),
            attention_mask=enc.attention_mask.to(dev),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = out[:, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def run_latentmas_sequential(
    model, tokenizer, question: str, W_e, target_norm, m: int,
    n_agents: int = 3, max_decode_tokens: int = 256,
) -> str:
    """LatentMAS sequential protocol: N-1 agents do latent thoughts,
    final agent decodes text answer.

    Agent roles:
      Agent 1 (analyst): reasons about the question
      Agent 2 (refiner): refines reasoning conditioned on Agent 1
      ...
      Agent N (solver): decodes the final text answer
    """
    dev = next(model.parameters()).device

    agent_prompts = [
        "You are an analyst. Think carefully about all possible answers to this question.",
        "You are a refiner. Consider the prior analysis and identify any missing answers.",
        "You are a solver. Based on all prior reasoning, list every correct answer. Separate by commas.",
    ]
    if n_agents > len(agent_prompts):
        agent_prompts = agent_prompts[:1] * (n_agents - 1) + agent_prompts[-1:]

    working_memory = None

    for agent_idx in range(n_agents):
        prompt = f"{agent_prompts[agent_idx]}\n\nQuestion: {question}"

        if agent_idx < n_agents - 1:
            prompt += "<think>"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            working_memory, _ = generate_latent_thoughts(
                model, enc.input_ids.to(dev), W_e, target_norm, m,
                past_kv=working_memory,
            )
        else:
            prompt += "\n\nAnswer:"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            answer = decode_text(
                model, tokenizer, working_memory, enc.input_ids.to(dev),
                max_new_tokens=max_decode_tokens,
            )
            return answer

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--m", type=int, default=40, help="latent steps per agent")
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qampari-data-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_qwen3()
    dev = next(model.parameters()).device

    print("Computing alignment matrix...")
    W_e, target_norm = compute_alignment(model)

    print(f"Loading QAMPARI tasks (n={args.n_tasks})...")
    bench = QampariBenchmark(data_dir=args.qampari_data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} tasks")

    jsonl_path = out_dir / f"eval_qampari_latent_m{args.m}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    agg_latent = {"f1_sum": 0, "correct_sum": 0, "n": 0}
    agg_single = {"f1_sum": 0, "correct_sum": 0, "n": 0}
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task["question_text"]

        try:
            # LatentMAS sequential
            latent_answer = run_latentmas_sequential(
                model, tok, question, W_e, target_norm,
                m=args.m, n_agents=args.n_agents,
            )
            latent_eval = evaluate_qampari(task["answer_list"], latent_answer)

            # Single-agent baseline
            single_answer = run_single_agent_baseline(model, tok, question)
            single_eval = evaluate_qampari(task["answer_list"], single_answer)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            continue

        rec = {
            "i": i,
            "qid": task["qid"],
            "question": question,
            "latent_answer": latent_answer[:500],
            "latent_f1": latent_eval["f1"],
            "latent_correct": latent_eval["correct"],
            "latent_recall": latent_eval["recall"],
            "single_answer": single_answer[:500],
            "single_f1": single_eval["f1"],
            "single_correct": single_eval["correct"],
            "single_recall": single_eval["recall"],
        }
        append_jsonl(rec, jsonl_path)

        agg_latent["f1_sum"] += latent_eval["f1"]
        agg_latent["correct_sum"] += int(latent_eval["correct"])
        agg_latent["n"] += 1
        agg_single["f1_sum"] += single_eval["f1"]
        agg_single["correct_sum"] += int(single_eval["correct"])
        agg_single["n"] += 1

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            lf1 = agg_latent["f1_sum"] / agg_latent["n"]
            sf1 = agg_single["f1_sum"] / agg_single["n"]
            print(f"[{i+1:3d}/{len(tasks)}] latent_f1={lf1:.3f} single_f1={sf1:.3f} "
                  f"({(i+1)/elapsed:.2f} tasks/s)")

    if agg_latent["n"] == 0:
        print("No tasks completed")
        return

    n = agg_latent["n"]
    summary = {
        "model": QWEN3_MODEL_ID,
        "benchmark": "qampari_closed_book",
        "n_tasks": n,
        "m_latent_steps": args.m,
        "n_agents": args.n_agents,
        "latent_f1_mean": agg_latent["f1_sum"] / n,
        "latent_accuracy": agg_latent["correct_sum"] / n,
        "single_f1_mean": agg_single["f1_sum"] / n,
        "single_accuracy": agg_single["correct_sum"] / n,
        "f1_delta": (agg_latent["f1_sum"] - agg_single["f1_sum"]) / n,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, out_dir / "eval_qampari_summary.json")

    print(f"\n{'='*60}")
    print("QAMPARI CLOSED-BOOK EVALUATION")
    print(f"  LatentMAS (m={args.m}, {args.n_agents} agents): "
          f"F1={summary['latent_f1_mean']:.3f}, acc={summary['latent_accuracy']:.3f}")
    print(f"  Single agent:                          "
          f"F1={summary['single_f1_mean']:.3f}, acc={summary['single_accuracy']:.3f}")
    print(f"  Delta F1: {summary['f1_delta']:+.3f}")


if __name__ == "__main__":
    main()
