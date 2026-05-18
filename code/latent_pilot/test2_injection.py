"""Test 2: KV-injection functional equivalence.

Question: when Agent B reads Agent A's output as injected KV entries vs.
reading it as text, does Agent B produce equivalent next-token
distributions?

Mechanism (resolved against Qwen3_5's transformers implementation):
  - Qwen3_5 uses a unified `DynamicCache` shared across both layer types.
  - Softmax attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31) store
    per-position `.keys` / `.values` in `cache.layers[i]`. These ARE
    sliceable in the sequence dim.
  - GatedDeltaNet layers (all other indices) store fixed-size recurrent
    state (`conv_states`, `recurrent_states`) which is NOT per-position.
    Cannot be compressed to K; either passed through verbatim or reset.
  - `take_last_k_cache(..., reset_deltanet=True)` keeps attention-layer
    KV sliced to last K positions, resets DeltaNet state so Agent B
    computes its own from its own input.

Procedure (per pair):
  1. Run Agent A forward with its input; capture (text output, full
     past_key_values).
  2. For B_text:
       prefix_ids = tokenize([B_system, A_text, B_hint])
       logits_text = forward(prefix_ids).logits[:, -1, :]
  3. For B_latent:
       prefix_ids = tokenize([B_system, B_hint])
       injected_cache = take_last_K(A_cache, K=64)
       logits_latent = forward(prefix_ids, past_key_values=injected_cache).logits[:, -1, :]
  4. Compare logits_text vs. logits_latent: top-1 agreement, KL divergence.

Outputs:
  results/latent_pilot/test2_injection.jsonl
  results/latent_pilot/test2_summary.json
"""
from __future__ import annotations

import argparse
import copy
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


def sample_pairs(n: int, seed: int = 42) -> list[dict]:
    """Build debate-turn pairs for injection testing.

    Tries existing traces first. If none are available (SAVE_TRANSCRIPTS
    was not enabled for most runs), falls back to synthetic pairs where
    Agent A's "output" is a pre-written argument and Agent B must respond.
    """
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
            "system": system,
            "user": user,
            "a_text": a_text[:2000],
            "b_hint": "Considering the prior agent's response, reason step-by-step and give your answer.",
        })
        if len(pairs) >= n:
            break

    if len(pairs) < n:
        print(f"Only {len(pairs)} trace pairs found; generating {n - len(pairs)} synthetic pairs")
        pairs.extend(_synthetic_pairs(n - len(pairs), seed=seed))

    return pairs[:n]


def _synthetic_pairs(n: int, seed: int = 42) -> list[dict]:
    """Generate synthetic debate pairs with substantial Agent A text."""
    rng = random.Random(seed)
    topics = [
        "multi-agent AI energy efficiency",
        "dense vs sparse model architectures",
        "retrieval-augmented generation tradeoffs",
        "chain-of-thought reasoning validity",
        "scaling laws for language models",
        "quantization impact on task performance",
        "inter-agent communication overhead",
        "evaluation benchmark reliability",
        "environmental cost of AI training",
        "emergent capabilities in large models",
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
        topic = rng.choice(topics)
        a_text = rng.choice(a_texts)
        pairs.append({
            "system": f"You are a research assistant analyzing {topic}.",
            "user": f"Evaluate the following claim about {topic}.",
            "a_text": a_text,
            "b_hint": "Considering the prior agent's response, reason step-by-step and give your answer.",
        })
    return pairs


def take_last_k_cache(past_key_values, k: int, layer_indices: list[int], reset_deltanet: bool = True):
    """Slice the last K positions out of the softmax-attention layers of
    a populated Qwen3_5DynamicCache, returning a modified cache suitable
    for injection into another forward pass.

    Qwen3_5DynamicCache exposes four parallel lists (len = n_layers = 32):
      .key_cache[i]   — [batch, n_kv_heads=4, seq, head_dim=256] for
                         softmax layers; None for GatedDeltaNet layers.
      .value_cache[i] — same layout as key_cache.
      .conv_states[i] — GatedDeltaNet conv1d running state (fixed size).
      .recurrent_states[i] — GatedDeltaNet recurrent state (fixed size).

    `reset_deltanet=True` clears conv_states and recurrent_states so the
    receiving agent starts DeltaNet processing from scratch on its own
    inputs. This isolates the KV-injection signal through the 8 softmax
    attention layers only.

    Mutates in place AND returns. Copy first if reuse is needed.
    """
    kc = past_key_values.key_cache
    vc = past_key_values.value_cache
    attn_set = set(layer_indices)

    for i in range(len(kc)):
        if i in attn_set:
            if kc[i] is not None and kc[i].shape[-2] > k:
                kc[i] = kc[i][:, :, -k:, :].contiguous()
                vc[i] = vc[i][:, :, -k:, :].contiguous()
        elif reset_deltanet:
            # GatedDeltaNet layer: clear recurrent state so Agent B
            # recomputes its own DeltaNet state from its own input.
            cs = getattr(past_key_values, "conv_states", None)
            rs = getattr(past_key_values, "recurrent_states", None)
            if cs is not None and i < len(cs):
                cs[i] = None
            if rs is not None and i < len(rs):
                rs[i] = None

    return past_key_values


def forward_with_cache(probed, input_ids, attention_mask, past_key_values):
    """Forward pass with an injected DynamicCache as past_key_values.

    Modern HF transformers compute new-token positions via `cache_position`
    when a populated cache is passed. We derive it from the cache's
    reported sequence length. For Qwen3_5's mRoPE with
    partial_rotary_factor=0.25, correct position assignment matters — the
    injected prefix occupies positions [0, injected_len) and new tokens
    occupy [injected_len, injected_len + L).
    """
    injected_len = _cache_seq_len(past_key_values)
    new_len = input_ids.shape[1]

    # cache_position tells the model which absolute positions the NEW tokens occupy
    cache_position = torch.arange(
        injected_len, injected_len + new_len, device=input_ids.device, dtype=torch.long
    )

    # Full-sequence attention mask: injected prefix (all 1s, the injected tokens
    # are "real" from the model's perspective) + original attention mask.
    full_mask = None
    if attention_mask is not None:
        prefix_mask = torch.ones(
            (input_ids.shape[0], injected_len),
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    with torch.no_grad():
        out = probed.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,  # must be True to read from past_key_values in some HF versions
            return_dict=True,
        )
    # Fall back to lm_head application if .logits missing (AutoModel fallback).
    if getattr(out, "logits", None) is not None:
        return out.logits[:, -1, :].float().cpu()
    hidden = out.last_hidden_state[:, -1:, :]
    lm_head = getattr(probed.model, "lm_head", None)
    if lm_head is None:
        for path in (("language_model", "lm_head"), ("model", "lm_head")):
            obj = probed.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None: break
            if obj is not None:
                lm_head = obj; break
    if lm_head is None:
        raise RuntimeError("No lm_head available to produce logits")
    with torch.no_grad():
        return lm_head(hidden).squeeze(1).float().cpu()


def _cache_seq_len(pkv) -> int:
    """Extract sequence length from a Qwen3_5DynamicCache.

    Prefers get_seq_length() (present on this cache class per the report).
    Falls back to scanning key_cache for the first non-None entry and
    reading its seq dim (dim -2).
    """
    if hasattr(pkv, "get_seq_length"):
        try:
            n = pkv.get_seq_length()
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    kc = getattr(pkv, "key_cache", None)
    if kc:
        for k in kc:
            if k is not None and hasattr(k, "shape"):
                return int(k.shape[-2])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--k", type=int, default=64, help="latent message length (KV slots)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"test2_injection_k{args.k}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    probed = load_model()
    pairs = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} debate-turn pairs; K={args.k}")

    attn_layer_ids = probed.attn_layer_indices()
    if not attn_layer_ids:
        print("ERROR: no softmax attention layers discovered")
        return

    agg = {"n": 0, "agree_sum": 0.0, "kl_sum": 0.0}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        # --- Generate / obtain Agent A's KV cache for its output tokens ---
        a_full = pair["system"] + "\n" + pair["user"] + "\n" + pair["a_text"]
        a_enc = probed.tokenizer(a_full, return_tensors="pt", truncation=True, max_length=4096)
        a_ids = a_enc.input_ids.to(probed.device)
        with torch.no_grad():
            a_out = probed.model(input_ids=a_ids, use_cache=True, return_dict=True)
        a_cache = a_out.past_key_values

        # --- B_text: prefix contains A's text verbatim ---
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

        # --- B_latent: A's last K KV entries injected as prefix ---
        # Deepcopy because take_last_k_cache mutates in place and we
        # must not corrupt a_cache (which came from A's forward pass).
        injected = take_last_k_cache(copy.deepcopy(a_cache), args.k, attn_layer_ids)

        bl_prompt = pair["system"] + "\n" + pair["b_hint"]
        bl_enc = probed.tokenizer(bl_prompt, return_tensors="pt", truncation=True, max_length=4096)
        logits_latent = forward_with_cache(
            probed,
            bl_enc.input_ids.to(probed.device),
            bl_enc.attention_mask.to(probed.device),
            injected,
        )

        agree = top1_agreement(logits_text.unsqueeze(1), logits_latent.unsqueeze(1))
        kl = F.kl_div(
            F.log_softmax(logits_latent, dim=-1),
            F.softmax(logits_text, dim=-1),
            reduction="batchmean",
        ).item()

        rec = {"i": i, "k": args.k, "first_token_agreement": agree, "kl_div": kl}
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
        verdict = "PASS: injection transmits useful signal; proceed to Tests 3-5"
    elif agree_mean < 0.20:
        verdict = "FAIL: injection broken on this architecture; abandon Option 1"
    else:
        verdict = "MARGINAL: consider learned projection on top of KV"

    summary = {
        "n_pairs": agg["n"],
        "k": args.k,
        "first_token_agreement_mean": agree_mean,
        "kl_div_mean": kl_mean,
        "verdict": verdict,
    }
    save_json(summary, out_dir / f"test2_summary_k{args.k}.json")
    print("\n" + "=" * 60)
    print("TEST 2 RESULT")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
