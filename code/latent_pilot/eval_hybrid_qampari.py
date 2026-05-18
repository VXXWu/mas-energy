"""Hybrid latent-tool evaluation on QAMPARI.

Compares text-debate MAS vs latent-debate MAS on QAMPARI with tool access.

Both conditions receive identical pre-retrieved evidence (BM25 search
results), eliminating the need for interactive ReAct tool calling. The
comparison isolates the debate mechanism: text communication vs latent
working memory sharing.

Protocol:
  Round 1 (shared, identical for both conditions):
    - Pre-retrieve top-K passages via BM25 for multiple search queries
    - Provide passages as context to all agents
    - Each agent independently generates an answer list (text decode)

  Round 2 — TEXT debate:
    - Each agent reads all other agents' Round 1 answers as text
    - Each agent generates a revised answer (text decode)
    - Final answer = last agent's revision

  Round 2 — LATENT debate:
    - Each agent does m latent thoughts conditioned on its Round 1 context
    - Working memory (KV cache) accumulated across agents
    - Final agent decodes text answer conditioned on all working memories

Evaluation: QAMPARI F1 (official protocol).

Outputs:
  results/latent_pilot/eval_hybrid_qampari.jsonl
  results/latent_pilot/eval_hybrid_summary.json
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
from benchmarks_qampari import (  # noqa: E402
    QampariBenchmark,
    evaluate_qampari,
)


# ---- Pre-retrieval: generate search queries and retrieve passages ----

def pre_retrieve(task: dict, benchmark: QampariBenchmark, n_queries: int = 5) -> str:
    """Generate search queries from the question and retrieve passages.

    Uses simple heuristic query generation (question substrings) + BM25.
    Returns a formatted context string with all retrieved passages.
    """
    question = task["question_text"]
    executor, cleanup = benchmark.make_executor(task)

    # Generate diverse queries from the question
    queries = [question]  # full question as first query
    words = question.split()
    if len(words) > 5:
        queries.append(" ".join(words[:len(words)//2]))
        queries.append(" ".join(words[len(words)//2:]))
    # Add entity-focused queries (capitalized words)
    entities = [w for w in words if w[0].isupper() and len(w) > 2]
    if entities:
        queries.append(" ".join(entities))
    queries = queries[:n_queries]

    # Retrieve and deduplicate
    seen_texts = set()
    all_passages = []
    for q in queries:
        results = executor("search", {"query": q})
        if isinstance(results, str):
            for line in results.split("\n\n"):
                line = line.strip()
                if line.startswith("[Result"):
                    text = line.split("] ", 1)[-1] if "] " in line else line
                    if text not in seen_texts:
                        seen_texts.add(text)
                        all_passages.append(text)

    cleanup()

    if not all_passages:
        return "(No relevant passages found.)"

    context = "Retrieved evidence:\n"
    for i, p in enumerate(all_passages[:15], 1):  # cap at 15 passages
        context += f"\n[{i}] {p}\n"
    return context


# ---- Text decode helper ----

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                   temperature: float = 0.0) -> str:
    """Generate text from a prompt string.

    temperature=0.0 → greedy (deterministic, for single-agent baseline and solver).
    temperature>0   → sampling (for Round 1 agents, diversity is critical for debate).

    Uses GenerationConfig to pass sampling parameters (transformers 4.57+
    ignores temperature/top_p as direct kwargs to model.generate).
    """
    from transformers import GenerationConfig
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)

    if temperature > 0:
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )
    else:
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids.to(dev),
            attention_mask=enc.attention_mask.to(dev),
            generation_config=gen_config,
        )
    new_tokens = out[:, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


# ---- Latent thought helpers ----

def generate_latent_thoughts_from_prompt(model, tokenizer, prompt, W_e, target_norm, m, past_kv=None):
    """Tokenize prompt, run latent thoughts, return working memory."""
    dev = next(model.parameters()).device
    enc = tokenizer(prompt + "<think>", return_tensors="pt", truncation=True, max_length=3072)
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

    for step in range(m):
        e_next = apply_alignment(h_t, W_e, target_norm)
        with torch.no_grad():
            step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                            use_cache=True, output_hidden_states=True, return_dict=True)
        kv_cache = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]

    return kv_cache


def decode_from_working_memory(model, tokenizer, prompt, past_kv, max_new_tokens=256):
    """Decode text answer conditioned on accumulated working memory."""
    dev = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(dev)
    prefix_len = _cache_seq_len(past_kv)

    # First forward: process the solver prompt with working memory prefix
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


# ---- MAS protocols ----

def run_text_debate(model, tokenizer, question: str, context: str, n_agents: int = 3) -> str:
    """Text-debate MAS: Round 1 independent answers, Round 2 text-based debate."""
    base_prompt = f"Question: {question}\n\n{context}\n\nList all correct answers, separated by commas.\n\nAnswer:"

    # Round 1: independent answers (temperature=0.7 for diversity, matching config.py DEBATE_TEMP)
    round1_answers = []
    for a in range(n_agents):
        prompt = f"You are Agent {a+1}. {base_prompt}"
        answer = generate_text(model, tokenizer, prompt, temperature=0.7)
        round1_answers.append(answer)

    # Round 2: each agent reads all others, generates revised answer
    revised_answers = []
    for a in range(n_agents):
        others = "\n".join(f"Agent {j+1} said: {round1_answers[j]}"
                          for j in range(n_agents) if j != a)
        prompt = (f"You are Agent {a+1}. Other agents provided these answers:\n{others}\n\n"
                  f"Question: {question}\n\n{context}\n\n"
                  f"Considering all answers, provide the complete list. Separate by commas.\n\nAnswer:")
        revised = generate_text(model, tokenizer, prompt)
        revised_answers.append(revised)

    return revised_answers[-1]  # last agent's revision


def run_latent_debate(model, tokenizer, question: str, context: str,
                      W_e, target_norm, m: int, n_agents: int = 3) -> str:
    """Latent-debate MAS: Round 1 independent answers (text), Round 2 latent debate."""
    base_prompt = f"Question: {question}\n\n{context}\n\nList all correct answers, separated by commas."

    # Round 1: independent answers (text, temperature=0.7 for diversity, same as text-debate)
    round1_answers = []
    for a in range(n_agents):
        prompt = f"You are Agent {a+1}. {base_prompt}\n\nAnswer:"
        answer = generate_text(model, tokenizer, prompt, temperature=0.7)
        round1_answers.append(answer)

    # Round 2: latent debate — each agent contributes latent thoughts, final decodes
    working_memory = None
    for a in range(n_agents):
        agent_context = (f"You are Agent {a+1}. Your initial answer was: {round1_answers[a]}\n\n"
                        f"Question: {question}\n\n{context}\n\n"
                        f"Refine your answer considering all perspectives.")

        if a < n_agents - 1:
            working_memory = generate_latent_thoughts_from_prompt(
                model, tokenizer, agent_context, W_e, target_norm, m,
                past_kv=working_memory,
            )
        else:
            # Solver gets the question and context explicitly (not just working memory)
            # so it knows WHAT to answer, with working memory providing refined reasoning
            solver_prompt = (f"You are Agent {a+1} (solver). Your initial answer was: {round1_answers[a]}\n\n"
                           f"Question: {question}\n\n{context}\n\n"
                           f"Based on all prior reasoning, provide the final complete "
                           f"list of answers. Separate by commas.\n\nAnswer:")
            answer = decode_from_working_memory(
                model, tokenizer, solver_prompt, working_memory,
            )
            return answer

    return ""


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--m", type=int, default=40)
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

    jsonl_path = out_dir / "eval_hybrid_qampari.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    agg = {
        "text_f1": 0, "text_correct": 0,
        "latent_f1": 0, "latent_correct": 0,
        "single_f1": 0, "single_correct": 0,
        "n": 0,
    }
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task["question_text"]

        try:
            # Pre-retrieve passages (shared across all conditions)
            context = pre_retrieve(task, bench)

            # Single-agent baseline
            single_prompt = (f"Question: {question}\n\n{context}\n\n"
                           f"List all correct answers, separated by commas.\n\nAnswer:")
            single_answer = generate_text(model, tok, single_prompt)
            single_eval = evaluate_qampari(task["answer_list"], single_answer)

            # Text-debate MAS (3 agents, R=2)
            text_answer = run_text_debate(model, tok, question, context, n_agents=args.n_agents)
            text_eval = evaluate_qampari(task["answer_list"], text_answer)

            # Latent-debate MAS (3 agents, m=40 latent steps in Round 2)
            latent_answer = run_latent_debate(model, tok, question, context,
                                             W_e, target_norm, m=args.m, n_agents=args.n_agents)
            latent_eval = evaluate_qampari(task["answer_list"], latent_answer)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            continue

        # First-task diagnostic
        if i == 0:
            print(f"\n--- First task diagnostic ---")
            print(f"  Question: {question[:100]}...")
            print(f"  Context length: {len(context)} chars")
            print(f"  Single answer: {single_answer[:200]}")
            print(f"  Text debate answer: {text_answer[:200]}")
            print(f"  Latent debate answer: {latent_answer[:200]}")
            print(f"  Single F1={single_eval['f1']:.3f}, Text F1={text_eval['f1']:.3f}, "
                  f"Latent F1={latent_eval['f1']:.3f}")
            print(f"---\n")

        rec = {
            "i": i, "qid": task["qid"], "question": question,
            "single_answer": single_answer[:300], "single_f1": single_eval["f1"],
            "text_answer": text_answer[:300], "text_f1": text_eval["f1"],
            "latent_answer": latent_answer[:300], "latent_f1": latent_eval["f1"],
        }
        append_jsonl(rec, jsonl_path)

        agg["single_f1"] += single_eval["f1"]
        agg["single_correct"] += int(single_eval["correct"])
        agg["text_f1"] += text_eval["f1"]
        agg["text_correct"] += int(text_eval["correct"])
        agg["latent_f1"] += latent_eval["f1"]
        agg["latent_correct"] += int(latent_eval["correct"])
        agg["n"] += 1

        if (i + 1) % 5 == 0:
            n = agg["n"]
            elapsed = time.time() - t0
            print(f"[{i+1:3d}/{len(tasks)}] "
                  f"single={agg['single_f1']/n:.3f} "
                  f"text_debate={agg['text_f1']/n:.3f} "
                  f"latent_debate={agg['latent_f1']/n:.3f} "
                  f"({(i+1)/elapsed:.2f} tasks/s)")

    if agg["n"] == 0:
        print("No tasks completed")
        return

    n = agg["n"]
    summary = {
        "model": QWEN3_MODEL_ID,
        "benchmark": "qampari_with_retrieval",
        "n_tasks": n,
        "n_agents": args.n_agents,
        "m_latent_steps": args.m,
        "single_agent": {"f1": agg["single_f1"] / n, "accuracy": agg["single_correct"] / n},
        "text_debate": {"f1": agg["text_f1"] / n, "accuracy": agg["text_correct"] / n},
        "latent_debate": {"f1": agg["latent_f1"] / n, "accuracy": agg["latent_correct"] / n},
        "latent_vs_text_f1_delta": (agg["latent_f1"] - agg["text_f1"]) / n,
        "latent_vs_single_f1_delta": (agg["latent_f1"] - agg["single_f1"]) / n,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, out_dir / "eval_hybrid_summary.json")

    print(f"\n{'='*60}")
    print("QAMPARI HYBRID EVALUATION")
    print(f"  Single agent:   F1={summary['single_agent']['f1']:.3f}")
    print(f"  Text debate:    F1={summary['text_debate']['f1']:.3f}")
    print(f"  Latent debate:  F1={summary['latent_debate']['f1']:.3f}")
    print(f"  Latent vs text: {summary['latent_vs_text_f1_delta']:+.3f}")


if __name__ == "__main__":
    main()
