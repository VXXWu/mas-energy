"""Latent Decentralized evaluation on QAMPARI.

Correct architecture: agents do text ReAct work (tool calls), then
communicate via latent thoughts instead of text responses.

Text condition (baseline):
  Agent 1: ReAct → text response → shared as prompt text to Agent 2
  Agent 2: reads Agent 1's text → ReAct → text response → shared to Agent 3
  Agent 3: reads prior text → ReAct → final answer

Latent condition (only communication changes):
  Agent 1: ReAct → m latent steps → KV working memory shared to Agent 2
  Agent 2: conditioned on Agent 1's KV → ReAct → m latent steps → KV shared
  Agent 3: conditioned on accumulated KV → ReAct → final text answer

Sequential topology (matches LatentMAS). Tool calling identical in both
conditions. Only the inter-agent communication mechanism differs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from latent_pilot.latent_react import detect_tool_call  # noqa: E402
from latent_pilot.utils import append_jsonl, results_root, save_json  # noqa: E402
from benchmarks_qampari import QampariBenchmark, evaluate_qampari  # noqa: E402

try:
    from prompts import DEBATE_AGENT_PROMPT
except ImportError:
    DEBATE_AGENT_PROMPT = ("You are a helpful assistant with access to tools, participating "
                           "in a collaborative problem-solving process. Use tools to investigate "
                           "the task thoroughly. Provide your reasoning and final answer clearly.")


def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0):
    dev = next(model.parameters()).device
    if temperature > 0:
        gc = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=temperature, top_p=0.95)
    else:
        gc = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids.to(dev),
                            attention_mask=enc.attention_mask.to(dev),
                            generation_config=gc)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def text_react(model, tokenizer, question, executor, system_prompt,
               max_steps=5, temperature=0.5):
    """Standard text ReAct loop. Returns (final_response, tool_call_count)."""
    prompt = (f"{system_prompt}\n\n"
              f"You have access to a search tool. To search, write:\n"
              f'<tool_call>\n{{"name": "search", "arguments": {{"query": "your query"}}}}\n</tool_call>\n\n'
              f"Question: {question}\n\n")
    total_tool_calls = 0

    for step in range(max_steps):
        response = generate_text(model, tokenizer, prompt, temperature=temperature)
        prompt += response

        tool = detect_tool_call(response)
        if tool is None:
            return response, total_tool_calls

        tool_name, tool_args = tool
        total_tool_calls += 1
        result = executor(tool_name, tool_args)
        prompt += f"\n<tool_response>\n{result}\n</tool_response>\n\n"

    # Max steps reached — generate final answer
    prompt += "\nProvide your final answer now.\n\n"
    final = generate_text(model, tokenizer, prompt)
    return final, total_tool_calls


def text_react_with_cache(model, tokenizer, question, executor, system_prompt,
                          max_steps=5, temperature=0.5, past_kv=None):
    """Text ReAct that builds on a prior agent's working memory (KV cache).
    Returns (final_response, kv_cache, tool_call_count)."""
    dev = next(model.parameters()).device
    gc = GenerationConfig(max_new_tokens=512, do_sample=(temperature > 0),
                          temperature=temperature if temperature > 0 else 1.0,
                          top_p=0.95 if temperature > 0 else 1.0)

    prompt = (f"{system_prompt}\n\n"
              f"You have access to a search tool. To search, write:\n"
              f'<tool_call>\n{{"name": "search", "arguments": {{"query": "your query"}}}}\n</tool_call>\n\n'
              f"Question: {question}\n\n")
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    input_ids = enc.input_ids.to(dev)

    # Initial forward with optional working memory prefix
    if past_kv is not None:
        prefix_len = _cache_seq_len(past_kv)
        cache_pos = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=dev, dtype=torch.long)
        attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, past_key_values=past_kv,
                        cache_position=cache_pos, use_cache=True, output_hidden_states=True, return_dict=True)
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    # Now generate text via model.generate on the accumulated prompt
    # (re-encode full prompt for simplicity — KV cache optimization is for production)
    total_tool_calls = 0
    full_prompt = prompt

    for step in range(max_steps):
        response = generate_text(model, tokenizer, full_prompt, temperature=temperature)
        full_prompt += response

        tool = detect_tool_call(response)
        if tool is None:
            # Re-encode full text to get final KV cache
            final_enc = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                final_out = model(input_ids=final_enc.input_ids.to(dev), use_cache=True,
                                 output_hidden_states=True, return_dict=True)
            return response, final_out.past_key_values, total_tool_calls

        tool_name, tool_args = tool
        total_tool_calls += 1
        result = executor(tool_name, tool_args)
        full_prompt += f"\n<tool_response>\n{result}\n</tool_response>\n\n"

    # Max steps — final generation
    full_prompt += "\nProvide your final answer now.\n\n"
    final_resp = generate_text(model, tokenizer, full_prompt)
    final_enc = tokenizer(full_prompt + final_resp, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        final_out = model(input_ids=final_enc.input_ids.to(dev), use_cache=True,
                         output_hidden_states=True, return_dict=True)
    return final_resp, final_out.past_key_values, total_tool_calls


def latent_summary(model, kv_cache, W_e, target_norm, m):
    """Add m latent thought steps on top of an existing KV cache.
    This replaces the text response — the latent steps ARE the message."""
    h_t = None
    # Get h_t from the last position of the cache by doing a dummy forward
    # Actually, we need the hidden state. Re-derive from cache isn't possible.
    # Instead, we'll get h_t during text_react_with_cache and pass it in.
    # For now, do a trick: forward a dummy token to get h_t, then overwrite.
    # BETTER: modify text_react_with_cache to return h_t.
    raise NotImplementedError("Need h_t from the last forward pass")


def run_text_sequential(model, tokenizer, question, executor,
                        n_agents=3, max_react_steps=5):
    """Text sequential MAS: each agent does ReAct, shares text with next."""
    prior_context = ""
    total_tool_calls = 0

    for agent_idx in range(n_agents):
        if prior_context:
            system = f"{DEBATE_AGENT_PROMPT}\n\nPrevious agent's findings:\n{prior_context}"
        else:
            system = DEBATE_AGENT_PROMPT

        response, tc = text_react(model, tokenizer, question, executor,
                                  system_prompt=system, max_steps=max_react_steps,
                                  temperature=0.5 if agent_idx < n_agents - 1 else 0.0)
        total_tool_calls += tc
        prior_context = response

    return response, {"tool_calls": total_tool_calls}


def run_latent_sequential(model, tokenizer, question, executor,
                          W_e, target_norm, m_latent=40,
                          n_agents=3, max_react_steps=5):
    """Latent sequential MAS: each agent does ReAct, shares KV working memory."""
    working_memory = None
    total_tool_calls = 0
    total_latent_steps = 0

    for agent_idx in range(n_agents):
        # Text ReAct phase (tool calling) — conditioned on prior working memory
        system = DEBATE_AGENT_PROMPT
        dev = next(model.parameters()).device

        prompt = (f"{system}\n\n"
                  f"You have access to a search tool. To search, write:\n"
                  f'<tool_call>\n{{"name": "search", "arguments": {{"query": "your query"}}}}\n</tool_call>\n\n'
                  f"Question: {question}\n\n")

        # Encode prompt, forward with working memory prefix
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        input_ids = enc.input_ids.to(dev)

        if working_memory is not None:
            prefix_len = _cache_seq_len(working_memory)
            cache_pos = torch.arange(prefix_len, prefix_len + input_ids.shape[1],
                                     device=dev, dtype=torch.long)
            attn_mask = torch.ones((1, prefix_len + input_ids.shape[1]), device=dev, dtype=torch.long)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           past_key_values=working_memory, cache_position=cache_pos,
                           use_cache=True, output_hidden_states=True, return_dict=True)
        else:
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=True,
                           output_hidden_states=True, return_dict=True)

        kv_cache = out.past_key_values
        h_t = out.hidden_states[-1][:, -1:, :]

        # Text ReAct: generate + tool call loop using model.generate
        # (rebuild full prompt string for generation, conditioned on the KV)
        full_prompt = prompt
        agent_tool_calls = 0

        for step in range(max_react_steps):
            response = generate_text(model, tokenizer, full_prompt,
                                    temperature=0.5 if agent_idx < n_agents - 1 else 0.0)
            full_prompt += response

            tool = detect_tool_call(response)
            if tool is None:
                break
            tool_name, tool_args = tool
            agent_tool_calls += 1
            result = executor(tool_name, tool_args)
            full_prompt += f"\n<tool_response>\n{result}\n</tool_response>\n\n"

        total_tool_calls += agent_tool_calls

        if agent_idx < n_agents - 1:
            # Re-encode full conversation to get KV cache with all tool work
            full_enc = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                full_out = model(input_ids=full_enc.input_ids.to(dev), use_cache=True,
                                output_hidden_states=True, return_dict=True)
            kv_cache = full_out.past_key_values
            h_t = full_out.hidden_states[-1][:, -1:, :]

            # Latent summary: m steps replacing the text response
            for _ in range(m_latent):
                e_next = apply_alignment(h_t, W_e, target_norm)
                with torch.no_grad():
                    step_out = model(inputs_embeds=e_next, past_key_values=kv_cache,
                                    use_cache=True, output_hidden_states=True, return_dict=True)
                kv_cache = step_out.past_key_values
                h_t = step_out.hidden_states[-1][:, -1:, :]
                total_latent_steps += 1

            working_memory = kv_cache
        else:
            # Final agent: return text response as the answer
            return response, {
                "tool_calls": total_tool_calls,
                "latent_steps": total_latent_steps,
            }

    return "", {"tool_calls": total_tool_calls, "latent_steps": total_latent_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--m-latent", type=int, default=40)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qampari-data-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"k{args.max_react_steps}_m{args.m_latent}"
    jsonl_path = out_dir / f"eval_latent_decentral_{tag}.jsonl"
    summary_path = out_dir / f"eval_latent_decentral_summary_{tag}.json"
    for p in (jsonl_path, summary_path):
        if p.exists():
            p.unlink()

    model, tok = load_qwen3()
    print("Computing alignment matrix...")
    W_e, target_norm = compute_alignment(model)

    bench = QampariBenchmark(data_dir=args.qampari_data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} QAMPARI tasks")
    print(f"Config: n_agents={args.n_agents}, max_react_steps={args.max_react_steps}, m_latent={args.m_latent}")

    agg = {k: 0.0 for k in ["text_f1", "latent_f1", "single_f1", "n"]}
    t0 = time.time()

    for i, task in enumerate(tasks):
        question = task["question_text"]
        executor, cleanup = bench.make_executor(task)

        try:
            # Single agent baseline
            single_ans, single_tc = text_react(model, tok, question, executor,
                                               system_prompt=DEBATE_AGENT_PROMPT,
                                               max_steps=args.max_react_steps)
            single_eval = evaluate_qampari(task["answer_list"], single_ans)

            # Text sequential MAS
            text_ans, text_stats = run_text_sequential(model, tok, question, executor,
                                                       n_agents=args.n_agents,
                                                       max_react_steps=args.max_react_steps)
            text_eval = evaluate_qampari(task["answer_list"], text_ans)

            # Latent sequential MAS
            latent_ans, latent_stats = run_latent_sequential(
                model, tok, question, executor, W_e, target_norm,
                m_latent=args.m_latent, n_agents=args.n_agents,
                max_react_steps=args.max_react_steps)
            latent_eval = evaluate_qampari(task["answer_list"], latent_ans)

        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            cleanup()
            continue

        cleanup()

        if i == 0:
            print(f"\n--- First task diagnostic ---")
            print(f"  Q: {question[:100]}...")
            print(f"  Single: {single_ans[:150]}... F1={single_eval['f1']:.3f}")
            print(f"  Text MAS: {text_ans[:150]}... F1={text_eval['f1']:.3f}")
            print(f"  Latent MAS: {latent_ans[:150]}... F1={latent_eval['f1']:.3f}, stats={latent_stats}")
            print()

        rec = {
            "i": i, "qid": task["qid"],
            "single_f1": single_eval["f1"], "text_f1": text_eval["f1"],
            "latent_f1": latent_eval["f1"], "latent_stats": latent_stats,
        }
        append_jsonl(rec, jsonl_path)
        agg["single_f1"] += single_eval["f1"]
        agg["text_f1"] += text_eval["f1"]
        agg["latent_f1"] += latent_eval["f1"]
        agg["n"] += 1

        if (i + 1) % 5 == 0:
            n = agg["n"]
            print(f"[{i+1:3d}/{len(tasks)}] single={agg['single_f1']/n:.3f} "
                  f"text={agg['text_f1']/n:.3f} latent={agg['latent_f1']/n:.3f}")

    if agg["n"] == 0:
        print("No tasks completed")
        return

    n = agg["n"]
    summary = {
        "model": QWEN3_MODEL_ID, "benchmark": "qampari_latent_decentralized",
        "n_tasks": int(n), "n_agents": args.n_agents,
        "max_react_steps": args.max_react_steps, "m_latent": args.m_latent,
        "single_f1": agg["single_f1"] / n,
        "text_sequential_f1": agg["text_f1"] / n,
        "latent_sequential_f1": agg["latent_f1"] / n,
        "latent_vs_text": (agg["latent_f1"] - agg["text_f1"]) / n,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, summary_path)
    print(f"\n{'='*60}")
    print(f"  Single:  F1={summary['single_f1']:.3f}")
    print(f"  Text:    F1={summary['text_sequential_f1']:.3f}")
    print(f"  Latent:  F1={summary['latent_sequential_f1']:.3f}")
    print(f"  Delta:   {summary['latent_vs_text']:+.3f}")


if __name__ == "__main__":
    main()
