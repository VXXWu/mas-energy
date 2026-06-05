"""Generate training pairs for the diffusion bridge using actual latent
chain-of-thought as source (Phase 2 data path).

The default trainer (`train_diffusion_bridge.py`) uses re-encoded text as
the source representation. That matches the training distribution but
doesn't capture the inference-time setup where the sender does latent CoT
without decoding. This script generates Phase 2 pairs whose source IS the
m latent-CoT hidden states, saved as .pt tensors alongside the text target.

Pair record format (JSONL, one line per pair):
    {
      "task_id": ...,
      "benchmark": ...,
      "agent_id": ...,
      "source_pt": "relative/path/to/source_NNNNN.pt",
      "target_response": <text the receiver would naturally see>,
      "target_completion_tokens": <int>,
      "question": <task question>,
      "m_latent": <int>, "layer_indices": [int, ...],
    }

Each .pt file holds: {"source": (1, m, L, d) tensor, "task_vec": (1, d)
tensor}. The trainer's BridgePairs subclass can lazily torch.load these.

Usage:
    python -m latent_pilot.generate_latent_pairs \\
        --model-name Qwen/Qwen3-4B \\
        --benchmark fanoutqa \\
        --n-tasks 30 \\
        --m-latent 20 \\
        --layer-indices 12 24 32 \\
        --output-dir mas-energy/results/latent_pilot/phase2_pairs_4b_fanoutqa
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.latentmas_baseline import compute_alignment, apply_alignment  # noqa: E402


def load_benchmark(name: str, data_dir: str = None):
    """Mirror agentic_latentmas.load_benchmark with a question-text getter."""
    if name == "qampari":
        from benchmarks_qampari import QampariBenchmark
        bench = QampariBenchmark(data_dir=data_dir)
        return bench, "question_text"
    if name == "fanoutqa":
        from benchmarks_fanoutqa import FanOutQABenchmark
        bench = FanOutQABenchmark()
        return bench, "question"
    raise ValueError(f"Unknown benchmark: {name}")


@torch.no_grad()
def run_latent_thoughts(model, input_ids, m_latent, layer_indices,
                        W_e, target_norm):
    """Forward the prompt, then run m latent CoT steps capturing multi-layer
    hidden states at each new latent position.

    Returns: (source: (1, m, L, d), task_vec: (1, d))
        - source[:, t, l, :] = hidden state at latent position t, layer
          layer_indices[l]
        - task_vec = last-layer hidden state at the FINAL prompt position
          (before any latent steps), to use as the bridge's task condition
    """
    out = model(input_ids=input_ids, use_cache=True,
                output_hidden_states=True, return_dict=True)
    kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]               # (1, 1, d)
    task_vec = h_t.squeeze(1)                            # (1, d)

    per_step_layers = []
    for _ in range(m_latent):
        e_next = apply_alignment(h_t, W_e, target_norm)
        step_out = model(inputs_embeds=e_next, past_key_values=kv,
                         use_cache=True, output_hidden_states=True,
                         return_dict=True)
        kv = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]
        layer_slices = [step_out.hidden_states[li][:, -1:, :]
                        for li in layer_indices]
        per_step_layers.append(torch.stack(layer_slices, dim=2))  # (1,1,L,d)

    source = torch.cat(per_step_layers, dim=1)           # (1, m, L, d)
    return source, task_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-4B")
    ap.add_argument("--benchmark", required=True, choices=["qampari", "fanoutqa"])
    ap.add_argument("--data-dir", default=None, help="Benchmark data dir (QAMPARI)")
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--m-latent", type=int, default=20)
    ap.add_argument("--layer-indices", type=int, nargs="+",
                    default=None, help="Source layers (1-indexed); default = "
                                       "last + middle + early auto-pick")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-target-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--text-pairs-path", type=Path, default=None,
                    help="Optional: existing text-pair JSONL to pull target_response "
                         "from (matched on task_id). If absent, target_response is "
                         "the agent's own greedy decode from the prompt position.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = args.output_dir / "tensors"
    tensors_dir.mkdir(exist_ok=True)
    jsonl_path = args.output_dir / "pairs.jsonl"

    torch.manual_seed(args.seed)

    # ---- Backbone ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading backbone: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    layer_indices = args.layer_indices
    if not layer_indices:
        # default: 3 layers spread across mid-to-late
        from latent_pilot.diffusion_bridge import default_layer_indices
        layer_indices = default_layer_indices(n_layers, 3)
    print(f"  Source layers (1-indexed): {layer_indices}")

    # ---- W_e for latent CoT loop ----
    print("Computing W_e ...")
    W_e, target_norm = compute_alignment(model)
    W_e = W_e.to(dev, dtype=dtype)

    # ---- Optional text-pair lookup for target_response ----
    text_targets = {}
    if args.text_pairs_path and args.text_pairs_path.exists():
        with open(args.text_pairs_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    tid = r.get("task_id")
                    if tid and r.get("target_response"):
                        text_targets.setdefault(tid, []).append(r["target_response"])
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded text targets for {len(text_targets)} tasks")

    # ---- Benchmark ----
    bench, question_key = load_benchmark(args.benchmark, args.data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"  Loaded {len(tasks)} {args.benchmark} tasks")

    # ---- Generation loop ----
    n_written = 0
    t0 = time.time()
    with open(jsonl_path, "w") as out_f:
        for ti, task in enumerate(tasks):
            question = task.get(question_key) or task.get("question") or ""
            tid = task.get("qid") or task.get("id") or f"task_{ti}"

            # Build a simple agent-style prompt (no chat template — we want
            # the raw sender's "what I would say" continuation hidden states).
            # Using apply_chat_template would also work; here we use a minimal
            # instruction prefix that matches what a decentralized debater
            # typically sees on its FIRST round.
            sys_text = ("You are a helpful research agent. Given the question "
                        "below, think step-by-step and produce your answer.")
            prompt = f"{sys_text}\n\nQuestion: {question}\n\nAnswer:"
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=args.max_prompt_tokens).input_ids.to(dev)

            try:
                source, task_vec = run_latent_thoughts(
                    model, ids, args.m_latent, layer_indices, W_e, target_norm,
                )
            except Exception as e:
                print(f"  [{ti}] latent CoT failed: {e}")
                continue

            # Save tensor file (CPU-side, half precision to keep size down)
            pt_name = f"source_{ti:05d}.pt"
            pt_path = tensors_dir / pt_name
            torch.save({
                "source": source.detach().cpu().to(torch.bfloat16),
                "task_vec": task_vec.detach().cpu().to(torch.bfloat16),
                "layer_indices": layer_indices,
            }, pt_path)

            # Resolve target_response: prefer matched text-pair, else greedy decode
            target_response = None
            if tid in text_targets and text_targets[tid]:
                target_response = text_targets[tid][0]
            else:
                # Greedy decode from current latent state for ~max_target_tokens
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=ids,
                        max_new_tokens=args.max_target_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                target_response = tok.decode(
                    gen[0, ids.size(1):], skip_special_tokens=True,
                )

            tgt_tokens = len(tok(target_response).input_ids)
            rec = {
                "task_id": tid,
                "benchmark": args.benchmark,
                "agent_id": "sender_phase2",
                "source_pt": str(pt_path.relative_to(args.output_dir)),
                "target_response": target_response,
                "target_completion_tokens": tgt_tokens,
                "question": question,
                "m_latent": args.m_latent,
                "layer_indices": layer_indices,
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_written += 1
            if (ti + 1) % 5 == 0:
                el = time.time() - t0
                print(f"  [{ti+1}/{len(tasks)}] wrote {n_written}, "
                      f"elapsed {el/60:.1f} min")

    el = time.time() - t0
    print(f"\nDone. Wrote {n_written} pairs to {jsonl_path}")
    print(f"Tensor dir: {tensors_dir} ({sum(p.stat().st_size for p in tensors_dir.iterdir())/1e6:.1f} MB)")
    print(f"Elapsed: {el/60:.1f} min")


if __name__ == "__main__":
    main()
