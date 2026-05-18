"""Test 2b: Hidden-state soft-embedding injection (Option 2 pilot).

Unlike Test 2 (KV injection into 8 attention layers only), this test
extracts Agent A's last-layer hidden states and feeds them as input
embeddings to Agent B. All 32 layers -- including the 24 GatedDeltaNet
layers -- process the injected signal. No 8-layer bottleneck.

Mechanism:
  1. Run Agent A forward; extract last-layer hidden states H_A at A's
     output token positions: shape [1, L, 4096].
  2. Compress H_A to K vectors via selection or pooling:
     Z = H_A[:, -K:, :]  (last-K selection) or mean-pool to K chunks.
  3. Hook Agent B's embedding layer to prepend Z before B's own token
     embeddings. Agent B's full forward (all 32 layers) processes Z first.
  4. Compare B's next-token logits to the B_text baseline.

Why this might work despite feeding "post-processed" vectors into layer 0:
  - LatentMAS (Zhang et al. 2025) showed this works on Qwen3 (full attn)
  - The model's residual stream means each layer adds to a running sum;
    high-level features from layer 32 are still partially interpretable
    by layer 0's residual-stream processing
  - Empirically, same-model hidden states lie in a compatible manifold

Why this might fail:
  - 32 layers of re-processing could amplify noise in the recycled vectors
  - GatedDeltaNet layers' recurrent state may not initialize properly from
    high-level representations (they expect low-level token features)

Decision thresholds (same as Test 2):
  agreement > 0.60  → Option 2 viable, proceed to full implementation
  agreement < 0.20  → hidden-state injection doesn't work either
  in between        → partial signal, learned compressor may help

Outputs:
  results/latent_pilot/test2b_hidden_k{K}.jsonl
  results/latent_pilot/test2b_summary_k{K}.json
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


# ---- Reuse synthetic pair generation from test2_injection ----

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
        print(f"Only {len(pairs)} trace pairs found; generating {n - len(pairs)} synthetic pairs")
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


# ---- Hidden state extraction ----

def extract_last_hidden_states(probed, input_ids: torch.Tensor) -> torch.Tensor:
    """Run forward, return last-layer hidden states. Shape: [1, seq, hidden_dim].

    For Qwen3_5ForConditionalGeneration, the output may expose hidden_states
    via output_hidden_states=True, or we hook the final layernorm.
    """
    dev = probed.device
    with torch.no_grad():
        out = probed.model(
            input_ids=input_ids.to(dev),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    # output_hidden_states=True returns a tuple of hidden states per layer.
    # Last element is the output of the final layer (before lm_head).
    hs = getattr(out, "hidden_states", None)
    if hs is not None and len(hs) > 0:
        return hs[-1].float()  # [batch, seq, hidden_dim]

    # Fallback: some VLMs put hidden_states under a different attr
    for attr in ("decoder_hidden_states", "language_model_hidden_states"):
        hs = getattr(out, attr, None)
        if hs is not None and len(hs) > 0:
            return hs[-1].float()

    raise RuntimeError(
        f"Model output has no hidden_states. Available keys: {list(out.keys())}. "
        f"Ensure the model class supports output_hidden_states=True."
    )


def compress_hidden_states(
    hidden: torch.Tensor, k: int, method: str = "last_k"
) -> torch.Tensor:
    """Compress [1, L, D] hidden states to [1, K, D].

    Methods:
      last_k:  take the last K positions (assumes conclusion is at end)
      stride:  take every L//K-th position (uniform coverage)
      pool:    chunk into K groups, mean-pool each (smoothed compression)
    """
    L = hidden.shape[1]
    if L <= k:
        return hidden

    if method == "last_k":
        return hidden[:, -k:, :]
    elif method == "stride":
        step = max(1, L // k)
        indices = list(range(0, L, step))[:k]
        return hidden[:, indices, :]
    elif method == "pool":
        chunks = hidden[:, :L - (L % k), :].reshape(1, k, L // k, -1)
        return chunks.mean(dim=2)
    else:
        raise ValueError(f"Unknown compression method: {method}")


# ---- Embedding injection via hook ----

def _find_embed_module(probed) -> torch.nn.Module:
    """Locate the token embedding module (embed_tokens)."""
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
    """Forward pass where `prefix_embeds` [1, K, D] are prepended to the
    token embeddings before entering the decoder stack. All 32 layers
    process the prefix.

    Uses `inputs_embeds` (not `input_ids`) to avoid shape mismatch between
    input_ids and position_ids when the prefix is prepended. The embedding
    lookup is done manually, then concatenated with prefix_embeds.
    """
    dev = probed.device
    prefix_embeds = prefix_embeds.to(dev, dtype=torch.bfloat16)
    input_ids = input_ids.to(dev)
    K = prefix_embeds.shape[1]
    new_len = input_ids.shape[1]
    total_len = K + new_len

    # Manually embed input_ids, then prepend prefix
    embed_mod = _find_embed_module(probed)
    with torch.no_grad():
        token_embeds = embed_mod(input_ids)  # [1, new_len, D]
    combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [1, total_len, D]

    # Attention mask covering prefix + new tokens
    if attention_mask is not None:
        prefix_mask = torch.ones((1, K), device=dev, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask.to(dev)], dim=1)
    else:
        full_mask = torch.ones((1, total_len), device=dev, dtype=torch.long)

    # Position IDs: [0, 1, ..., total_len-1]
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

    # Fallback: apply lm_head manually
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--k", type=int, default=64, help="number of compressed hidden-state vectors")
    ap.add_argument("--method", type=str, default="last_k",
                    choices=["last_k", "stride", "pool"],
                    help="compression method for hidden states")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"test2b_hidden_k{args.k}_{args.method}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    probed = load_model()
    pairs = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} debate-turn pairs; K={args.k}, method={args.method}")

    agg = {"n": 0, "agree_sum": 0.0, "kl_sum": 0.0}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        try:
            # --- Agent A: extract hidden states ---
            a_full = pair["system"] + "\n" + pair["user"] + "\n" + pair["a_text"]
            a_enc = probed.tokenizer(a_full, return_tensors="pt", truncation=True, max_length=4096)
            a_hidden = extract_last_hidden_states(probed, a_enc.input_ids)
            z = compress_hidden_states(a_hidden, args.k, method=args.method)

            # --- B_text: baseline (reads A's text as tokens) ---
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

            # --- B_latent: soft-embedding injection ---
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
            print(f"[{i}] error: {e}")
            continue

        rec = {
            "i": i, "k": args.k, "method": args.method,
            "first_token_agreement": agree, "kl_div": kl,
        }
        append_jsonl(rec, jsonl_path)
        agg["n"] += 1
        agg["agree_sum"] += agree
        agg["kl_sum"] += kl

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1:4d}/{len(pairs)}] agree={agg['agree_sum']/agg['n']:.3f} "
                  f"kl={agg['kl_sum']/agg['n']:.3f} ({(i+1)/elapsed:.2f}/s)")

    if agg["n"] == 0:
        print("No pairs produced results")
        return

    agree_mean = agg["agree_sum"] / agg["n"]
    kl_mean = agg["kl_sum"] / agg["n"]
    if agree_mean > 0.60:
        verdict = "PASS: hidden-state injection transmits useful signal"
    elif agree_mean < 0.20:
        verdict = "FAIL: hidden-state injection doesn't work; need learned compressor"
    else:
        verdict = "MARGINAL: partial signal; learned compressor likely needed"

    summary = {
        "n_pairs": agg["n"],
        "k": args.k,
        "method": args.method,
        "first_token_agreement_mean": agree_mean,
        "kl_div_mean": kl_mean,
        "verdict": verdict,
    }
    save_json(summary, out_dir / f"test2b_summary_k{args.k}_{args.method}.json")
    print("\n" + "=" * 60)
    print("TEST 2b RESULT")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
