"""LatentMAS evaluation on GSM8K (short-answer math reasoning).

GSM8K is the primary benchmark LatentMAS was validated on. Final answer
is a single number, minimizing sensitivity to extended-generation corruption
from latent working memory.

Three conditions:
  1. Single agent: solve directly
  2. Text debate: 3 agents independently solve, share text, revise
  3. Latent debate: 3 agents, Round 1 text, Round 2 latent thoughts

Evaluation: exact-match on extracted numeric answer.

Outputs:
  results/latent_pilot/eval_gsm8k.jsonl
  results/latent_pilot/eval_gsm8k_summary.json
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.latentmas_baseline import (  # noqa: E402
    QWEN3_MODEL_ID,
    _cache_seq_len,
    apply_alignment,
    compute_alignment,
    load_qwen3,
)
from latent_pilot.utils import append_jsonl, results_root, save_json  # noqa: E402


def extract_number(text: str) -> str | None:
    """Extract the final numeric answer from model output.
    Looks for #### pattern (GSM8K convention) or last number in text."""
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def check_answer(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-3
    except ValueError:
        return predicted.strip() == gold.strip()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.0) -> str:
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if temperature > 0:
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                      do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids.to(dev),
                            attention_mask=enc.attention_mask.to(dev),
                            generation_config=gen_config)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def generate_latent_thoughts(model, tokenizer, prompt, W_e, target_norm, m, past_kv=None):
    dev = next(model.parameters()).device
    enc = tokenizer(prompt + "<think>", return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc.input_ids.to(dev)

    if past_kv is not None:
        prefix_len = _cache_seq_len(past_kv)
        cache_position = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=dev, dtype=torch.long)
        attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, past_key_values=past_kv,
                        cache_position=cache_position, use_cache=True, output_hidden_states=True, return_dict=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)

    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]
    for _ in range(m):
        e_next = apply_alignment(h_t, W_e, target_norm)
        with torch.no_grad():
            step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                            use_cache=True, output_hidden_states=True, return_dict=True)
        kv_cache = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]
    return kv_cache


def decode_from_working_memory(model, tokenizer, prompt, past_kv, max_new_tokens=256):
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(dev)
    prefix_len = _cache_seq_len(past_kv)

    cache_position = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=dev, dtype=torch.long)
    attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, past_key_values=past_kv,
                    cache_position=cache_position, use_cache=True, return_dict=True)
    current_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        pos = torch.tensor([_cache_seq_len(current_kv)], device=dev, dtype=torch.long)
        mask = torch.ones((1, _cache_seq_len(current_kv) + 1), device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=next_token, attention_mask=mask, past_key_values=current_kv,
                        cache_position=pos, use_cache=True, return_dict=True)
        current_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == tokenizer.eos_token_id:
            break

    token_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)


MATH_PROMPT = ("Solve the following math problem step by step. "
               "End your answer with #### followed by the numeric answer.\n\n")


def run_single(model, tok, question):
    prompt = MATH_PROMPT + f"Question: {question}\n\nAnswer:"
    return generate_text(model, tok, prompt)


def run_text_debate(model, tok, question, n_agents=3):
    # Round 1: independent solutions with diversity
    round1 = []
    for a in range(n_agents):
        prompt = f"You are Solver {a+1}. " + MATH_PROMPT + f"Question: {question}\n\nAnswer:"
        round1.append(generate_text(model, tok, prompt, temperature=0.7))

    # Round 2: each agent revises after seeing others
    revised = []
    for a in range(n_agents):
        others = "\n".join(f"Solver {j+1}: {round1[j]}" for j in range(n_agents) if j != a)
        prompt = (f"You are Solver {a+1}. Other solvers gave these answers:\n{others}\n\n"
                  f"Question: {question}\n\n"
                  f"{MATH_PROMPT}Provide the correct answer.\n\nAnswer:")
        revised.append(generate_text(model, tok, prompt))

    return revised[-1]


def run_latent_debate(model, tok, question, W_e, target_norm, m, n_agents=3):
    # Round 1: independent solutions (text, same as text debate)
    round1 = []
    for a in range(n_agents):
        prompt = f"You are Solver {a+1}. " + MATH_PROMPT + f"Question: {question}\n\nAnswer:"
        round1.append(generate_text(model, tok, prompt, temperature=0.7))

    # Round 2: latent debate
    working_memory = None
    for a in range(n_agents):
        agent_prompt = (f"You are Solver {a+1}. Your answer was: {round1[a]}\n\n"
                       f"Question: {question}\n\nReconsider carefully.")

        if a < n_agents - 1:
            working_memory = generate_latent_thoughts(
                model, tok, agent_prompt, W_e, target_norm, m, past_kv=working_memory)
        else:
            solver_prompt = (f"Question: {question}\n\n"
                           f"{MATH_PROMPT}Based on all reasoning, give the final answer.\n\nAnswer:")
            return decode_from_working_memory(model, tok, solver_prompt, working_memory)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=100)
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "eval_gsm8k.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    model, tok = load_qwen3()
    print("Computing alignment matrix...")
    W_e, target_norm = compute_alignment(model)

    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_tasks < len(ds):
        import random
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(ds)), args.n_tasks)
        tasks = [ds[i] for i in indices]
    else:
        tasks = list(ds)
    print(f"Loaded {len(tasks)} tasks")

    agg = {k: 0 for k in ["single_correct", "text_correct", "latent_correct", "n"]}
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task["question"]
        gold_answer = task["answer"].split("####")[-1].strip().replace(",", "")

        try:
            single_out = run_single(model, tok, question)
            single_pred = extract_number(single_out)
            single_ok = check_answer(single_pred, gold_answer)

            text_out = run_text_debate(model, tok, question, n_agents=args.n_agents)
            text_pred = extract_number(text_out)
            text_ok = check_answer(text_pred, gold_answer)

            latent_out = run_latent_debate(model, tok, question, W_e, target_norm,
                                          m=args.m, n_agents=args.n_agents)
            latent_pred = extract_number(latent_out)
            latent_ok = check_answer(latent_pred, gold_answer)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            continue

        if i == 0:
            print(f"\n--- First task diagnostic ---")
            print(f"  Q: {question[:100]}...")
            print(f"  Gold: {gold_answer}")
            print(f"  Single: {single_out[:150]}... → pred={single_pred}, correct={single_ok}")
            print(f"  Text:   {text_out[:150]}... → pred={text_pred}, correct={text_ok}")
            print(f"  Latent: {latent_out[:150]}... → pred={latent_pred}, correct={latent_ok}")
            print()

        rec = {
            "i": i, "question": question[:100], "gold": gold_answer,
            "single_pred": single_pred, "single_correct": single_ok,
            "text_pred": text_pred, "text_correct": text_ok,
            "latent_pred": latent_pred, "latent_correct": latent_ok,
            "latent_output_sample": latent_out[:200],
        }
        append_jsonl(rec, jsonl_path)

        agg["single_correct"] += int(single_ok)
        agg["text_correct"] += int(text_ok)
        agg["latent_correct"] += int(latent_ok)
        agg["n"] += 1

        if (i + 1) % 10 == 0:
            n = agg["n"]
            print(f"[{i+1:3d}/{len(tasks)}] "
                  f"single={agg['single_correct']/n:.3f} "
                  f"text={agg['text_correct']/n:.3f} "
                  f"latent={agg['latent_correct']/n:.3f} "
                  f"({(i+1)/(time.time()-t0):.2f} tasks/s)")

    if agg["n"] == 0:
        print("No tasks completed")
        return

    n = agg["n"]
    summary = {
        "model": QWEN3_MODEL_ID,
        "benchmark": "gsm8k",
        "n_tasks": n,
        "n_agents": args.n_agents,
        "m_latent_steps": args.m,
        "single_accuracy": agg["single_correct"] / n,
        "text_debate_accuracy": agg["text_correct"] / n,
        "latent_debate_accuracy": agg["latent_correct"] / n,
        "latent_vs_text_delta": (agg["latent_correct"] - agg["text_correct"]) / n,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, out_dir / "eval_gsm8k_summary.json")

    print(f"\n{'='*60}")
    print("GSM8K EVALUATION")
    print(f"  Single agent:   {summary['single_accuracy']:.3f}")
    print(f"  Text debate:    {summary['text_debate_accuracy']:.3f}")
    print(f"  Latent debate:  {summary['latent_debate_accuracy']:.3f}")
    print(f"  Latent vs text: {summary['latent_vs_text_delta']:+.3f}")


if __name__ == "__main__":
    main()
