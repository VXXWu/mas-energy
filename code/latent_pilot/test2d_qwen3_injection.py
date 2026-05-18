"""Test 2d: KV-injection on Qwen3-8B (pure softmax attention).

Qwen3-8B has 36 standard attention layers with KV cache at every layer.
No DeltaNet, no hybrid attention. KV injection should work here because
all layers participate — validated by LatentMAS on this model family.

This test answers: does training-free KV injection work on a pure-attention
model? If yes, the paper story is "latent communication works on standard
architectures; hybrid architectures need a learned adapter." If no,
something deeper is wrong with the injection methodology.

Uses the same protocol as Test 2 (KV injection) but:
  - Loads Qwen3-8B instead of Qwen3.5-9B
  - Injects KV into ALL 36 layers (no DeltaNet to skip)
  - No cache state reset needed

Outputs:
  results/latent_pilot/test2d_qwen3_k{K}.jsonl
  results/latent_pilot/test2d_qwen3_summary_k{K}.json
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.utils import (  # noqa: E402
    append_jsonl,
    load_existing_traces,
    results_root,
    save_json,
    top1_agreement,
)


QWEN3_MODEL_ID = "Qwen/Qwen3-8B"


# ---- Model loading (simpler than model_probe — no hybrid complexity) ----

def load_qwen3(dtype: str = "bfloat16"):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    print(f"Loading {QWEN3_MODEL_ID} (dtype={dtype})...")
    tok = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    config = AutoConfig.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    print(f"Loaded: {n_layers} layers, pure softmax attention")
    return model, tok, n_layers


# ---- Pair generation ----

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


# ---- KV cache manipulation (simplified: all layers are attention) ----

def _get_kv_lists(pkv):
    """Extract key/value lists from a DynamicCache, handling three API variants:
      1. .key_cache / .value_cache lists (Qwen3_5DynamicCache)
      2. .layers[i] with .key_states / .value_states attrs (transformers 4.57+ DynamicCache)
      3. __getitem__ returning (key, value) tuples (older DynamicCache)
    Returns (keys_list, values_list) where each is a list of tensors per layer.
    """
    # Variant 1: direct parallel lists
    if hasattr(pkv, "key_cache") and pkv.key_cache is not None:
        return pkv.key_cache, pkv.value_cache

    # Variant 2: .layers API (transformers 4.57+)
    if hasattr(pkv, "layers") and pkv.layers is not None and len(pkv.layers) > 0:
        keys = []
        vals = []
        for layer in pkv.layers:
            # Try common attr names for per-layer KV storage
            k = None
            v = None
            for k_attr in ("key_states", "keys", "key_cache", "key"):
                k = getattr(layer, k_attr, None)
                if k is not None:
                    break
            for v_attr in ("value_states", "values", "value_cache", "value"):
                v = getattr(layer, v_attr, None)
                if v is not None:
                    break
            keys.append(k)
            vals.append(v)
        if any(k is not None for k in keys):
            return keys, vals
        # .layers exists but no recognized KV attrs — dump first layer for debug
        first = pkv.layers[0]
        layer_attrs = [a for a in dir(first) if not a.startswith("_")]
        raise RuntimeError(
            f"Cache .layers[0] has no recognized KV attrs. "
            f"Class: {first.__class__.__name__}, attrs: {layer_attrs[:20]}"
        )

    # Variant 3: __getitem__ (older DynamicCache)
    if hasattr(pkv, "__len__") and hasattr(pkv, "__getitem__"):
        n = len(pkv)
        keys = []
        vals = []
        for i in range(n):
            try:
                layer = pkv[i]
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    keys.append(layer[0])
                    vals.append(layer[1])
                else:
                    keys.append(None)
                    vals.append(None)
            except (IndexError, KeyError, TypeError):
                keys.append(None)
                vals.append(None)
        return keys, vals

    raise RuntimeError(
        f"Cache {pkv.__class__.__name__} has no recognized KV access API. "
        f"Attrs: {[a for a in dir(pkv) if not a.startswith('_')][:20]}"
    )


def _rope_correction(keys: torch.Tensor, offset: int, rope_theta: float = 10000000.0) -> torch.Tensor:
    """Apply a constant RoPE position offset to pre-rotated key tensors.

    When slicing the last K entries from a cache, keys have RoPE baked in
    at positions [L-K, L-K+1, ..., L-1]. To re-index them as [0, 1, ..., K-1],
    apply rotation by offset = -(L-K) to every key.

    Uses Qwen3's half-split RoPE: first half and second half of head_dim
    are the (cos, sin) pair, NOT interleaved.

    keys: [batch, n_heads, seq, head_dim]
    offset: position shift (negative to shift left)
    """
    head_dim = keys.shape[-1]
    device = keys.device
    dtype = keys.dtype
    half = head_dim // 2

    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = offset * freqs
    cos_a = torch.cos(angles).to(dtype)
    sin_a = torch.sin(angles).to(dtype)

    k1 = keys[..., :half]
    k2 = keys[..., half:]
    new_k1 = k1 * cos_a - k2 * sin_a
    new_k2 = k1 * sin_a + k2 * cos_a
    return torch.cat([new_k1, new_k2], dim=-1)


def take_last_k_cache(past_key_values, k: int, rope_correct: bool = True) -> object:
    """Slice all layers' KV to last K positions with RoPE position correction.

    After slicing, keys have RoPE baked at positions [L-K..L-1]. With
    rope_correct=True, re-rotates them to positions [0..K-1] so the
    receiving model sees correct relative positions. Values are NOT rotated
    (RoPE only applies to keys and queries).

    Mutates in place and returns.
    """
    # Determine original seq_len BEFORE slicing (for RoPE offset calculation)
    orig_len = _cache_seq_len(past_key_values)
    offset = -(orig_len - k) if rope_correct else 0

    # Variant 1: .key_cache parallel lists (Qwen3_5DynamicCache)
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache is not None:
        kc = past_key_values.key_cache
        vc = past_key_values.value_cache
        for i in range(len(kc)):
            if kc[i] is not None and kc[i].shape[-2] > k:
                sliced_k = kc[i][:, :, -k:, :].contiguous()
                kc[i] = _rope_correction(sliced_k, offset) if offset != 0 else sliced_k
                vc[i] = vc[i][:, :, -k:, :].contiguous()
        return past_key_values

    # Variant 2: .layers API (transformers 4.57+ DynamicCache)
    if hasattr(past_key_values, "layers") and past_key_values.layers:
        for layer in past_key_values.layers:
            for k_attr in ("key_states", "keys", "key_cache", "key"):
                kt = getattr(layer, k_attr, None)
                if kt is not None and kt.shape[-2] > k:
                    sliced_k = kt[:, :, -k:, :].contiguous()
                    setattr(layer, k_attr, _rope_correction(sliced_k, offset) if offset != 0 else sliced_k)
                    break
            for v_attr in ("value_states", "values", "value_cache", "value"):
                vt = getattr(layer, v_attr, None)
                if vt is not None and vt.shape[-2] > k:
                    setattr(layer, v_attr, vt[:, :, -k:, :].contiguous())
                    break
        return past_key_values

    # Variant 3: rebuild from extracted lists
    keys, vals = _get_kv_lists(past_key_values)
    from transformers import DynamicCache
    new_cache = DynamicCache()
    for i in range(len(keys)):
        if keys[i] is not None:
            sliced_k = keys[i][:, :, -k:, :].contiguous() if keys[i].shape[-2] > k else keys[i]
            vi = vals[i][:, :, -k:, :].contiguous() if vals[i].shape[-2] > k else vals[i]
            new_cache.update(_rope_correction(sliced_k, offset) if offset != 0 else sliced_k, vi, i)
    return new_cache


def _cache_seq_len(pkv) -> int:
    """Get sequence length from actual tensor shapes, NOT get_seq_length().
    After take_last_k_cache mutates tensors via setattr, get_seq_length()
    may return the stale pre-mutation length, causing shape mismatches
    in forward_with_cache.
    """
    try:
        keys, _ = _get_kv_lists(pkv)
        for k in keys:
            if k is not None and hasattr(k, "shape"):
                return int(k.shape[-2])
    except Exception:
        pass
    if hasattr(pkv, "get_seq_length"):
        try:
            n = pkv.get_seq_length()
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    return 0


_forward_accepts_cache_position = None  # determined on first call

def forward_with_cache(model, input_ids, attention_mask, past_key_values):
    """Forward with injected KV cache. Tries cache_position first; falls
    back to position_ids if the model doesn't accept cache_position."""
    global _forward_accepts_cache_position
    dev = next(model.parameters()).device
    injected_len = _cache_seq_len(past_key_values)
    new_len = input_ids.shape[1]

    full_mask = None
    if attention_mask is not None:
        prefix_mask = torch.ones((1, injected_len), device=dev, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask.to(dev)], dim=1)

    base_kwargs = dict(
        input_ids=input_ids.to(dev),
        attention_mask=full_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    if _forward_accepts_cache_position is not False:
        cache_position = torch.arange(
            injected_len, injected_len + new_len, device=dev, dtype=torch.long
        )
        try:
            with torch.no_grad():
                out = model(**base_kwargs, cache_position=cache_position)
            _forward_accepts_cache_position = True
            return out.logits[:, -1, :].float().cpu()
        except TypeError:
            print("  cache_position not accepted; falling back to position_ids")
            _forward_accepts_cache_position = False

    # Fallback: use position_ids offset by injected prefix length
    position_ids = torch.arange(
        injected_len, injected_len + new_len, device=dev, dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        out = model(**base_kwargs, position_ids=position_ids)
    return out.logits[:, -1, :].float().cpu()


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    k_label = "full" if args.k == 0 else str(args.k)
    jsonl_path = out_dir / f"test2d_qwen3_k{k_label}.jsonl"
    summary_path = out_dir / f"test2d_qwen3_summary_k{k_label}.json"
    if jsonl_path.exists():
        jsonl_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    model, tok, n_layers = load_qwen3()
    dev = next(model.parameters()).device

    pairs = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} pairs; K={args.k}, model={QWEN3_MODEL_ID}, {n_layers} layers (all attention)")

    agg = {"n": 0, "agree_sum": 0.0, "kl_sum": 0.0}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        try:
            # Agent A: forward with cache capture
            a_full = pair["system"] + "\n" + pair["user"] + "\n" + pair["a_text"]
            a_enc = tok(a_full, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                a_out = model(input_ids=a_enc.input_ids.to(dev), use_cache=True, return_dict=True)
            a_cache = a_out.past_key_values

            # First-pair diagnostic: print cache structure so failures are debuggable
            if i == 0:
                print(f"Cache class: {a_cache.__class__.__name__}")
                print(f"  len: {len(a_cache) if hasattr(a_cache, '__len__') else 'N/A'}")
                print(f"  has key_cache: {hasattr(a_cache, 'key_cache')}")
                print(f"  has layers: {hasattr(a_cache, 'layers')}")
                if hasattr(a_cache, 'get_seq_length'):
                    print(f"  get_seq_length: {a_cache.get_seq_length()}")
                if hasattr(a_cache, 'layers') and a_cache.layers:
                    layer0 = a_cache.layers[0]
                    l0_attrs = [a for a in dir(layer0) if not a.startswith("_")]
                    print(f"  layers[0] class: {layer0.__class__.__name__}")
                    print(f"  layers[0] attrs: {l0_attrs[:25]}")
                    for attr in ("key_states", "keys", "key_cache", "key"):
                        val = getattr(layer0, attr, None)
                        if val is not None:
                            print(f"  layers[0].{attr} shape: {val.shape}, dtype: {val.dtype}")
                            break
                    else:
                        print(f"  layers[0]: no recognized key attr found")
                print(f"  cache attrs: {[a for a in dir(a_cache) if not a.startswith('_')][:20]}")

            # B_text: baseline
            b_text_prompt = (
                pair["system"] + "\n"
                + "Prior agent said:\n" + pair["a_text"] + "\n"
                + pair["b_hint"]
            )
            bt_enc = tok(b_text_prompt, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                bt_out = model(
                    input_ids=bt_enc.input_ids.to(dev),
                    attention_mask=bt_enc.attention_mask.to(dev),
                    use_cache=False,
                    return_dict=True,
                )
            logits_text = bt_out.logits[:, -1, :].float().cpu()

            # B_latent: inject A's KV cache as prefix.
            # k=0 means use full cache (no slicing) — the LatentMAS protocol.
            # k>0 slices to last K positions (compression, may break RoPE).
            injected = copy.deepcopy(a_cache)
            if args.k > 0:
                injected = take_last_k_cache(injected, args.k)
            if i == 0:
                print(f"  Injected cache seq_len: {_cache_seq_len(injected)} (k={'full' if args.k == 0 else args.k})")
            bl_prompt = pair["system"] + "\n" + pair["b_hint"]
            bl_enc = tok(bl_prompt, return_tensors="pt", truncation=True, max_length=4096)
            logits_latent = forward_with_cache(
                model, bl_enc.input_ids, bl_enc.attention_mask, injected,
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
        verdict = "PASS: KV injection works on pure-attention Qwen3-8B"
    elif agree_mean < 0.20:
        verdict = "FAIL: KV injection fails even on pure attention; methodology issue"
    else:
        verdict = "MARGINAL: partial signal on pure attention"

    summary = {
        "model": QWEN3_MODEL_ID,
        "n_layers": n_layers,
        "architecture": "pure_softmax_attention",
        "n_pairs": agg["n"],
        "k": args.k,
        "first_token_agreement_mean": agree_mean,
        "kl_div_mean": kl_mean,
        "verdict": verdict,
    }
    save_json(summary, summary_path)
    print("\n" + "=" * 60)
    print("TEST 2d RESULT")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
