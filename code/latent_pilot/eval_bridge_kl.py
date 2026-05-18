"""End-to-end KL comparison: W_e baseline vs diffusion-bridge sampling.

Loads a trained DiffusionBridge checkpoint and computes end-to-end KL of the
receiver's next-token distribution under three conditions:

    K=0        — W_e baseline only (warm-start path; should match LatentMAS)
    K=K_eval   — diffusion bridge with K denoising steps (e.g. K=20)
    text       — receiver's distribution given the text peer input (the
                 teacher; KL of teacher vs teacher = 0 by construction; this
                 is the reference for the other two)

Reports KL(text || K=0) and KL(text || K=K_eval) on a held-out pair set.
The decision rule from LATENT_HANDOFF.md (2026-05 resumption):
    K=K_eval KL drops below K=0 KL → diffusion bridge is doing useful work.
    K=K_eval KL ≈ K=0 KL → diffusion isn't earning its complexity.

Usage:
    python -m latent_pilot.eval_bridge_kl \\
        --bridge-ckpt $RUN_DIR/bridge_epoch1.pt \\
        --pairs-path $PAIRS \\
        --output $RUN_DIR/kl_comparison.json \\
        --K-values 0 5 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.diffusion_bridge import (  # noqa: E402
    BridgeConfig, DiffusionBridge,
)
from latent_pilot.latentmas_baseline import compute_alignment  # noqa: E402
from latent_pilot.train_diffusion_bridge import (  # noqa: E402
    BridgePairs, collate, task_split, eval_endtoend_kl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-4B")
    ap.add_argument("--bridge-ckpt", required=True, type=Path)
    ap.add_argument("--pairs-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--K-values", nargs="+", type=int, default=[0, 5, 20])
    ap.add_argument("--max-eval-batches", type=int, default=16,
                    help="How many held-out batches of size 1 to evaluate")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Backbone ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading backbone: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    device = next(backbone.parameters()).device
    embed_layer = backbone.get_input_embeddings()

    # ---- W_e ----
    print("Computing W_e (one-time)...")
    W_e, target_norm = compute_alignment(backbone)
    W_e = W_e.to(device, dtype=dtype)

    # ---- Bridge ----
    print(f"Loading bridge: {args.bridge_ckpt}")
    state = torch.load(args.bridge_ckpt, map_location="cpu", weights_only=False)
    cfg = BridgeConfig(**state["config"])
    bridge = DiffusionBridge(cfg).to(device).to(dtype)
    bridge.attach_w_e(W_e, target_norm)
    bridge.load_state_dict(state["bridge_state_dict"], strict=False)
    bridge.eval()
    layer_indices = state.get("layer_indices",
        list(range(backbone.config.num_hidden_layers - cfg.n_source_layers + 1,
                   backbone.config.num_hidden_layers + 1)))
    print(f"  cfg: k={cfg.k_soft_prompts}, layers={layer_indices}, "
          f"n_blocks={cfg.n_blocks}")

    # ---- Held-out pairs ----
    print(f"Loading pairs: {args.pairs_path}")
    full_ds = BridgePairs(args.pairs_path, tok)
    print(f"  total: {len(full_ds)}")
    _, eval_recs = task_split(full_ds.records, args.eval_frac, args.seed)
    eval_ds = BridgePairs.__new__(BridgePairs)
    eval_ds.tok = tok
    eval_ds.records = eval_recs
    eval_ds.max_peer_tokens = full_ds.max_peer_tokens
    eval_ds.max_question_tokens = full_ds.max_question_tokens
    eval_ds.max_target_tokens = full_ds.max_target_tokens
    print(f"  eval: {len(eval_ds)}")

    from torch.utils.data import DataLoader
    eval_loader = DataLoader(
        eval_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    # ---- Sweep K ----
    m_positions = state.get("m_positions",
                            cfg.k_soft_prompts * 2)  # heuristic fallback
    results = {
        "model_name": args.model_name,
        "bridge_ckpt": str(args.bridge_ckpt),
        "bridge_config": cfg.to_dict(),
        "layer_indices": layer_indices,
        "n_eval_batches": min(args.max_eval_batches, len(eval_loader)),
        "K_values": args.K_values,
        "kl_per_K": {},
    }

    for K in args.K_values:
        print(f"  Sampling at K={K} ...")
        kl = eval_endtoend_kl(
            backbone, embed_layer, bridge, eval_loader, layer_indices,
            m_positions, cfg.k_soft_prompts, device, K_sample=K,
            max_batches=args.max_eval_batches,
        )
        print(f"  K={K:3d}  KL = {kl:.4f} nats/token")
        results["kl_per_K"][str(K)] = kl

    # Pretty print summary table
    print()
    print("=== KL Comparison ===")
    print(f"{'K':>5} {'KL (nats/token)':>18}")
    base = results["kl_per_K"][str(args.K_values[0])]
    for K in args.K_values:
        kl = results["kl_per_K"][str(K)]
        delta = kl - base
        print(f"{K:>5} {kl:>18.4f}  Δ vs K={args.K_values[0]} = {delta:+.4f}")

    # Decision rule reminder
    print()
    print("Decision rule (LATENT_HANDOFF.md, 2026-05 resumption):")
    print("  K=20 KL < 0.55         → bridge has real headroom; full eval next")
    print("  K=20 KL in [0.55, 0.60] → ambiguous; try MLP adapter or more layers")
    print("  K=20 KL ≥ 0.60         → bridge isn't earning its complexity")
    print(f"  Prior linear-encoder ceiling for reference: ~0.65 nats")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
