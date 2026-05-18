"""LatentMAS baseline implementation on Qwen3-8B.

Implements the core mechanism from Zhang et al. (arXiv:2511.20639):
  1. Compute W_e = pseudo-inverse of token embedding matrix (one-time)
  2. Latent thought generation: h_t -> W_e projection -> forward -> h_{t+1}
  3. Working memory transfer: share KV cache between agents
  4. Only the final agent decodes text

Evaluation: compare latent-debate Agent B output to text-debate Agent B
output across m ∈ {10, 20, 40, 80} latent thought steps.

Outputs:
  results/latent_pilot/latentmas_m{M}.jsonl
  results/latent_pilot/latentmas_summary.json
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.utils import (  # noqa: E402
    append_jsonl,
    load_existing_traces,
    results_root,
    save_json,
    top1_agreement,
)

QWEN3_MODEL_ID = "Qwen/Qwen3-8B"  # default, overridable via --model-name


def load_qwen3(model_id: str = None, dtype: str = "bfloat16"):
    model_id = model_id or QWEN3_MODEL_ID
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    print(f"Loading {model_id} (dtype={dtype})...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def find_weight_matrix(model, kind: str) -> torch.Tensor:
    """Locate embedding (W_in) or LM head (W_out) weight matrix."""
    if kind == "embed":
        paths = [("model", "embed_tokens"), ("model", "model", "embed_tokens"), ("transformer", "wte")]
    elif kind == "lm_head":
        paths = [("lm_head",), ("model", "lm_head")]
    else:
        raise ValueError(f"Unknown weight kind: {kind}")
    for path in paths:
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "weight"):
            return obj.weight.data
    raise RuntimeError(f"Could not locate {kind} weight matrix")


def compute_alignment(model, reg: float = 1e-4) -> tuple[torch.Tensor, float]:
    """Compute the LatentMAS alignment matrix W_e and target embedding norm.

    Following LatentMAS's _build_latent_realign_matrix():
      gram = W_out^T @ W_out + reg * I
      W_e = solve(gram, W_out^T @ W_in)

    Where W_out is the LM head weight [V, d] and W_in is the token embedding
    weight [V, d]. For tied embeddings they're identical; for untied (Qwen3-8B)
    they differ and BOTH must be used.

    Also computes target_norm = mean L2 norm of input embeddings, used to
    renormalize projected vectors (prevents representation drift).

    Returns: (W_e [d, d], target_norm scalar)
    """
    W_in = find_weight_matrix(model, "embed")   # [V, d]
    W_out = find_weight_matrix(model, "lm_head")  # [V, d]
    d = W_in.shape[1]

    # Compute on CPU to avoid GPU OOM (V×d float32 tensors can be ~4GB each
    # for large vocab models like Qwen3.5-9B with V=248K)
    W_in_f = W_in.float().cpu()
    W_out_f = W_out.float().cpu()

    gram = W_out_f.T @ W_out_f + reg * torch.eye(d, dtype=torch.float32)
    rhs = W_out_f.T @ W_in_f  # [d, d]
    W_e = torch.linalg.solve(gram, rhs)  # [d, d]

    target_norm = W_in_f.norm(dim=1).mean().item()

    print(f"  W_in shape: {W_in.shape}, W_out shape: {W_out.shape}")
    print(f"  W_e shape: {W_e.shape}, target_norm: {target_norm:.4f}")
    return W_e.to(W_in.dtype), target_norm


def apply_alignment(h_t: torch.Tensor, W_e: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Project hidden state to embedding space and renormalize.

    Following LatentMAS's _apply_latent_realignment():
      1. e = h_t @ W_e (in float32)
      2. Normalize e to target_norm (match input embedding distribution)
    """
    h_float = h_t.float()
    W_e_float = W_e.float().to(h_t.device)
    e = h_float @ W_e_float
    # Renormalize to target embedding norm
    e_norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    e = e * (target_norm / e_norm)
    return e.to(h_t.dtype)


def generate_latent_thoughts(
    model, input_ids: torch.Tensor, W_e: torch.Tensor,
    target_norm: float, m: int,
) -> tuple[object, torch.Tensor]:
    """Generate m latent thought steps, returning (full KV cache, final hidden state).

    Protocol (matches LatentMAS's generate_latent_batch):
      1. Forward on input_ids with use_cache=True, output_hidden_states=True
      2. Extract h_t = last-layer hidden state at final position
      3. For each step i in [1, m]:
         a. e_i = apply_alignment(h_t, W_e, target_norm)
         b. Forward with inputs_embeds=e_i, past_key_values=cache
         c. h_t = new last-layer hidden state
      4. Return (full KV cache, h_t)
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    kv_cache = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]  # [1, 1, d]

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


_fwd_accepts_cache_position = None

def generate_text_with_cache(
    model, tokenizer, prefix_cache, prompt_ids: torch.Tensor,
    max_new_tokens: int = 1,
) -> torch.Tensor:
    """Given a KV cache prefix (from another agent's latent thoughts),
    run Agent B's prompt and return logits at the final position.

    Tries cache_position first; falls back to position_ids if the model
    doesn't accept cache_position.
    """
    global _fwd_accepts_cache_position
    dev = prompt_ids.device
    prefix_len = _cache_seq_len(prefix_cache)
    new_len = prompt_ids.shape[1]

    attn_mask = torch.ones((1, prefix_len + new_len), device=dev, dtype=torch.long)
    base_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=attn_mask,
        past_key_values=prefix_cache,
        use_cache=True,
        return_dict=True,
    )

    if _fwd_accepts_cache_position is not False:
        cache_position = torch.arange(prefix_len, prefix_len + new_len, device=dev, dtype=torch.long)
        try:
            with torch.no_grad():
                out = model(**base_kwargs, cache_position=cache_position)
            _fwd_accepts_cache_position = True
            return out.logits[:, -1, :].float().cpu()
        except TypeError:
            print("  cache_position not accepted; falling back to position_ids")
            _fwd_accepts_cache_position = False

    position_ids = torch.arange(prefix_len, prefix_len + new_len, device=dev, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model(**base_kwargs, position_ids=position_ids)
    return out.logits[:, -1, :].float().cpu()


def _cache_seq_len(pkv) -> int:
    if hasattr(pkv, "get_seq_length"):
        try:
            n = pkv.get_seq_length()
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    # Fallback: check layers
    if hasattr(pkv, "layers") and pkv.layers:
        for layer in pkv.layers:
            for attr in ("keys", "key_states", "key_cache"):
                kt = getattr(layer, attr, None)
                if kt is not None:
                    return int(kt.shape[-2])
    return 0


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


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--m-values", type=int, nargs="+", default=[10, 20, 40, 80],
                    help="latent thought step counts to sweep")
    ap.add_argument("--model-name", type=str, default=None,
                    help="Model to test (default: Qwen/Qwen3-8B)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_id = args.model_name or QWEN3_MODEL_ID
    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_qwen3(model_id=model_id)
    dev = next(model.parameters()).device

    # Compute alignment matrix (one-time)
    print("Computing alignment matrix W_e...")
    t0 = time.time()
    W_e, target_norm = compute_alignment(model)
    print(f"  Computed in {time.time()-t0:.1f}s")

    pairs = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"Built {len(pairs)} pairs; m_values={args.m_values}")

    all_results = {}

    for m in args.m_values:
        print(f"\n{'='*60}")
        print(f"Latent thought steps: m={m}")
        print(f"{'='*60}")

        jsonl_path = out_dir / f"latentmas_m{m}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

        agg = {"n": 0, "agree_sum": 0.0, "kl_sum": 0.0}
        t0 = time.time()

        for i, pair in enumerate(pairs):
            try:
                # --- Agent A: generate latent thoughts ---
                # LatentMAS uses <think> tag to trigger reasoning mode
                a_input = pair["system"] + "\n" + pair["user"] + "<think>"
                a_enc = tok(a_input, return_tensors="pt", truncation=True, max_length=2048)
                a_ids = a_enc.input_ids.to(dev)

                a_cache, a_final_h = generate_latent_thoughts(model, a_ids, W_e, target_norm, m)

                if i == 0:
                    a_cache_len = _cache_seq_len(a_cache)
                    a_input_len = a_ids.shape[1]
                    print(f"  Agent A: input={a_input_len} tokens + {m} latent steps = {a_cache_len} cache entries")
                    print(f"  Final hidden state norm: {a_final_h.float().norm().item():.2f}")

                # --- Agent B (latent): read A's working memory, produce logits ---
                b_prompt = pair["system"] + "\n" + pair["b_hint"]
                b_enc = tok(b_prompt, return_tensors="pt", truncation=True, max_length=2048)
                b_ids = b_enc.input_ids.to(dev)

                logits_latent = generate_text_with_cache(model, tok, a_cache, b_ids)

                # --- Agent B (text baseline): read A's text, produce logits ---
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

                # --- Compare ---
                agree = top1_agreement(logits_text.unsqueeze(1), logits_latent.unsqueeze(1))
                kl = F.kl_div(
                    F.log_softmax(logits_latent, dim=-1),
                    F.softmax(logits_text, dim=-1),
                    reduction="batchmean",
                ).item()

            except Exception as e:
                print(f"  [{i}] error: {e}")
                import traceback; traceback.print_exc()
                continue

            rec = {"i": i, "m": m, "first_token_agreement": agree, "kl_div": kl}
            append_jsonl(rec, jsonl_path)
            agg["n"] += 1
            agg["agree_sum"] += agree
            agg["kl_sum"] += kl

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:4d}/{len(pairs)}] agree={agg['agree_sum']/agg['n']:.3f} "
                      f"kl={agg['kl_sum']/agg['n']:.3f} ({(i+1)/elapsed:.2f}/s)")

        if agg["n"] > 0:
            agree_mean = agg["agree_sum"] / agg["n"]
            kl_mean = agg["kl_sum"] / agg["n"]
        else:
            agree_mean, kl_mean = 0.0, float("inf")

        all_results[m] = {
            "m": m,
            "n_pairs": agg["n"],
            "first_token_agreement_mean": agree_mean,
            "kl_div_mean": kl_mean,
        }
        print(f"  m={m}: agree={agree_mean:.3f} kl={kl_mean:.3f}")

    # Combined summary
    best_m = max(all_results, key=lambda m: all_results[m]["first_token_agreement_mean"])
    best = all_results[best_m]

    summary = {
        "model": model_id,
        "per_m": all_results,
        "best_m": best_m,
        "best_agreement": best["first_token_agreement_mean"],
    }
    save_json(summary, out_dir / "latentmas_summary.json")

    print(f"\n{'='*60}")
    print("LATENTMAS BASELINE RESULTS")
    print(f"  Model: {QWEN3_MODEL_ID}")
    for m in sorted(all_results):
        r = all_results[m]
        print(f"  m={m:3d}: agree={r['first_token_agreement_mean']:.3f}  kl={r['kl_div_mean']:.3f}")
    print(f"  Best: m={best_m} (agree={best['first_token_agreement_mean']:.3f})")


if __name__ == "__main__":
    main()
