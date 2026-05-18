"""Test 2c: Hidden-state injection with layer sweep.

Tests whether extracting from an EARLIER layer (not the last) produces
hidden states that are more compatible with layer-0 input processing.

Rationale: Test 2b showed last-layer hidden states (layer 32) fail at 0-11%
agreement. Last-layer states are maximally transformed away from the input
embedding manifold. Earlier layers may carry enough semantic information
while remaining closer to the embedding space.

Sweeps extraction layers: [4, 8, 16, 24, 32 (last)].
Uses pool compression (best performer from Test 2b) at K=64.

Outputs:
  results/latent_pilot/test2c_layer{L}_k{K}.jsonl
  results/latent_pilot/test2c_summary.json  (combined across layers)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.model_probe import load_model  # noqa: E402
from latent_pilot.utils import (  # noqa: E402
    append_jsonl,
    load_existing_traces,
    results_root,
    save_json,
    top1_agreement,
)


# ---- Pair generation (same as test2b) ----

def sample_pairs(n: int, seed: int = 42) -> list[dict]:
    recs = load_existing_traces("qampari", n=n, seed=seed)
    recs += load_existing_traces("workbench", n=n, seed=seed + 1)
    pairs = []
    for rec in recs:
        msgs = rec.get("messages")
        resp = rec.get("response")
        if not (msgs and resp):
            continue
        if isinstance(msgs, list):
            system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        else:
            system, user = "", str(msgs)
        a_text = resp if isinstance(resp, str) else str(resp)
        pairs.append({
            "system": system, "user": user, "a_text": a_text[:2000],
            "b_hint": "Considering the prior agent's response, reason step-by-step and give your answer.",
        })
        if len(pairs) >= n:
            break
    if len(pairs) < n:
        pairs.extend(_synthetic_pairs(n - len(pairs), seed=seed))
    return pairs[:n]


def _synthetic_pairs(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    topics = [
        "multi-agent AI energy efficiency", "dense vs sparse model architectures",
        "retrieval-augmented generation tradeoffs", "chain-of-thought reasoning validity",
        "scaling laws for language models", "quantization impact on task performance",
        "inter-agent communication overhead", "evaluation benchmark reliability",
        "environmental cost of AI training", "emergent capabilities in large models",
    ]
    a_texts = [
        (
            "After careful analysis, the primary finding is that current approaches "
            "significantly underestimate the true computational cost. The token-based "
            "metrics commonly used in the literature fail to account for the hardware-"
            "level asymmetry between different operation types. Specifically, decode "
            "operations consume substantially more energy per token than prefill "
            "operations, yet cost analyses treat them identically. This systematic "
            "underestimation grows worse with multi-agent topologies because coordination "
            "messages are disproportionately decode-heavy. Our measurements show the "
            "actual energy overhead ranges from 7x to 49x higher than what token counts "
            "would predict, depending on the topology and benchmark used."
        ),
        (
            "The experimental results suggest a more nuanced picture than previously "
            "assumed. While the overall trend confirms that adding agents increases "
            "total resource consumption, the relationship between agent count and task "
            "accuracy is highly benchmark-dependent. On parallelizable tasks like list-"
            "answer QA, multi-agent systems show genuine accuracy improvements that "
            "justify the additional cost. On sequential tasks, however, the coordination "
            "overhead provides no accuracy benefit and strictly wastes resources. The "
            "critical insight is that the decision to use multi-agent systems should be "
            "guided by task structure analysis, not applied uniformly across all problem "
            "types. Furthermore, the energy cost profile differs markedly from the token "
            "cost profile due to hardware-level decode/prefill asymmetry."
        ),
    ]
    pairs = []
    for i in range(n):
        pairs.append({
            "system": f"You are a research assistant analyzing {rng.choice(topics)}.",
            "user": f"Evaluate the following claim about {rng.choice(topics)}.",
            "a_text": rng.choice(a_texts),
            "b_hint": "Considering the prior agent's response, reason step-by-step and give your answer.",
        })
    return pairs


# ---- Hidden state extraction at a specific layer ----

def extract_hidden_at_layer(probed, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run forward with output_hidden_states=True, return hidden states from
    the specified layer index. Layer 0 = embedding output, layer N = last
    decoder layer output (before final norm).

    Returns: [1, seq, hidden_dim] in float32.
    """
    dev = probed.device
    with torch.no_grad():
        out = probed.model(
            input_ids=input_ids.to(dev),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        for attr in ("decoder_hidden_states", "language_model_hidden_states"):
            hs = getattr(out, attr, None)
            if hs is not None:
                break
    if hs is None:
        raise RuntimeError(f"No hidden_states in output. Keys: {list(out.keys())}")

    # hs is a tuple of (n_layers + 1) tensors: [embedding_output, layer_0, ..., layer_N]
    # layer_idx=0 corresponds to hs[1] (after first decoder layer)
    # layer_idx=-1 or layer_idx=n_layers-1 corresponds to hs[-1]
    actual_idx = layer_idx + 1 if layer_idx >= 0 else layer_idx
    if actual_idx >= len(hs):
        print(f"WARNING: requested layer {layer_idx} but only {len(hs)-1} layers exist; using last")
        actual_idx = -1
    return hs[actual_idx].float()


def compress_pool(hidden: torch.Tensor, k: int) -> torch.Tensor:
    """Mean-pool [1, L, D] into [1, K, D] chunks."""
    L = hidden.shape[1]
    if L <= k:
        return hidden
    usable = L - (L % k)
    chunks = hidden[:, :usable, :].reshape(1, k, usable // k, -1)
    return chunks.mean(dim=2)


# ---- Forward with soft prefix (same as test2b, inputs_embeds approach) ----

def _find_embed_module(probed) -> torch.nn.Module:
    for path in [
        ("model", "model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
        ("model", "embed_tokens"),
    ]:
        obj = probed.model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate embed_tokens module")


def forward_with_soft_prefix(
    probed, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    prefix_embeds: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with prefix_embeds prepended via inputs_embeds."""
    dev = probed.device
    prefix_embeds = prefix_embeds.to(dev, dtype=torch.bfloat16)
    input_ids = input_ids.to(dev)
    K = prefix_embeds.shape[1]
    new_len = input_ids.shape[1]
    total_len = K + new_len

    embed_mod = _find_embed_module(probed)
    with torch.no_grad():
        token_embeds = embed_mod(input_ids)
    combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    if attention_mask is not None:
        prefix_mask = torch.ones((1, K), device=dev, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask.to(dev)], dim=1)
    else:
        full_mask = torch.ones((1, total_len), device=dev, dtype=torch.long)

    position_ids = torch.arange(0, total_len, device=dev, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = probed.model(
            inputs_embeds=combined_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

    if getattr(out, "logits", None) is not None:
        return out.logits[:, -1, :].float().cpu()

    hidden = out.last_hidden_state[:, -1:, :]
    lm_head = None
    for path in (("lm_head",), ("language_model", "lm_head"), ("model", "lm_head")):
        obj = probed.model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            lm_head = obj
            break
    if lm_head is None:
        raise RuntimeError("No lm_head found")
    with torch.no_grad():
        return lm_head(hidden).squeeze(1).float().cpu()


# ---- Main: sweep extraction layers ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 16, 24, 31],
                    help="decoder layer indices to extract from (0-indexed, 31=last)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    probed = load_model()
    pairs = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} debate-turn pairs; K={args.k}")
    print(f"Layer sweep: {args.layers}")

    all_results = {}

    for extract_layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Extracting from layer {extract_layer}")
        print(f"{'='*60}")

        jsonl_path = out_dir / f"test2c_layer{extract_layer}_k{args.k}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

        agg = {"n": 0, "agree_sum": 0.0, "kl_sum": 0.0}
        t0 = time.time()

        for i, pair in enumerate(pairs):
            try:
                # Agent A: extract hidden states at the target layer
                a_full = pair["system"] + "\n" + pair["user"] + "\n" + pair["a_text"]
                a_enc = probed.tokenizer(a_full, return_tensors="pt", truncation=True, max_length=4096)
                a_hidden = extract_hidden_at_layer(probed, a_enc.input_ids, extract_layer)
                z = compress_pool(a_hidden, args.k)

                # B_text: baseline
                b_text_prompt = (
                    pair["system"] + "\n"
                    + "Prior agent said:\n" + pair["a_text"] + "\n"
                    + pair["b_hint"]
                )
                bt_enc = probed.tokenizer(b_text_prompt, return_tensors="pt", truncation=True, max_length=4096)
                with torch.no_grad():
                    bt_out = probed.model(
                        input_ids=bt_enc.input_ids.to(probed.device),
                        attention_mask=bt_enc.attention_mask.to(probed.device),
                        use_cache=False,
                        return_dict=True,
                    )
                logits_text = bt_out.logits[:, -1, :].float().cpu()

                # B_latent: soft-prefix injection
                bl_prompt = pair["system"] + "\n" + pair["b_hint"]
                bl_enc = probed.tokenizer(bl_prompt, return_tensors="pt", truncation=True, max_length=4096)
                logits_latent = forward_with_soft_prefix(
                    probed, bl_enc.input_ids, bl_enc.attention_mask, z,
                )

                agree = top1_agreement(logits_text.unsqueeze(1), logits_latent.unsqueeze(1))
                kl = F.kl_div(
                    F.log_softmax(logits_latent, dim=-1),
                    F.softmax(logits_text, dim=-1),
                    reduction="batchmean",
                ).item()

            except Exception as e:
                print(f"[layer={extract_layer}, pair={i}] error: {e}")
                continue

            rec = {
                "i": i, "extract_layer": extract_layer, "k": args.k,
                "first_token_agreement": agree, "kl_div": kl,
            }
            append_jsonl(rec, jsonl_path)
            agg["n"] += 1
            agg["agree_sum"] += agree
            agg["kl_sum"] += kl

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:4d}/{len(pairs)}] agree={agg['agree_sum']/agg['n']:.3f} "
                      f"kl={agg['kl_sum']/agg['n']:.3f} ({(i+1)/elapsed:.2f}/s)")

        if agg["n"] > 0:
            agree_mean = agg["agree_sum"] / agg["n"]
            kl_mean = agg["kl_sum"] / agg["n"]
        else:
            agree_mean, kl_mean = 0.0, float("inf")

        all_results[extract_layer] = {
            "n_pairs": agg["n"],
            "extract_layer": extract_layer,
            "k": args.k,
            "method": "pool",
            "first_token_agreement_mean": agree_mean,
            "kl_div_mean": kl_mean,
        }
        print(f"  Layer {extract_layer}: agree={agree_mean:.3f} kl={kl_mean:.3f}")

    # Combined summary
    best_layer = max(all_results, key=lambda l: all_results[l]["first_token_agreement_mean"])
    best = all_results[best_layer]

    if best["first_token_agreement_mean"] > 0.60:
        verdict = f"PASS: layer {best_layer} extraction works (agree={best['first_token_agreement_mean']:.3f})"
    elif best["first_token_agreement_mean"] > 0.20:
        verdict = f"MARGINAL: layer {best_layer} is best (agree={best['first_token_agreement_mean']:.3f}); learned compressor may push it over"
    else:
        verdict = f"FAIL: no extraction layer works; best is layer {best_layer} (agree={best['first_token_agreement_mean']:.3f})"

    summary = {
        "per_layer": all_results,
        "best_layer": best_layer,
        "best_agreement": best["first_token_agreement_mean"],
        "verdict": verdict,
    }
    save_json(summary, out_dir / "test2c_summary.json")

    print(f"\n{'='*60}")
    print("TEST 2c RESULT")
    print(f"  Best layer: {best_layer} (agree={best['first_token_agreement_mean']:.3f})")
    print(f"  Verdict: {verdict}")
    for layer in sorted(all_results):
        r = all_results[layer]
        print(f"  Layer {layer:2d}: agree={r['first_token_agreement_mean']:.3f}  kl={r['kl_div_mean']:.3f}")


if __name__ == "__main__":
    main()
