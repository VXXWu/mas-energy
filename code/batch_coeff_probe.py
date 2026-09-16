#!/usr/bin/env python3
"""Batch-size energy-coefficient probe (2026-07-17).

Measures how the energy model's coefficients scale with serving batch size:
    E_dyn = a(B)*P + b(B)*C + kv(B)*SumKV
Purpose: the Fig 8 coefficients were fit under the serial (B=1) protocol,
where the completion/prompt asymmetry b/a ~ 336x vs API pricing's 3-5x.
Batched decode amortizes weight reads (b falls ~1/B until the roofline);
prefill is compute-bound (a ~flat); KV reads are per-sequence (kv should not
amortize). This probe measures all three so the cost-vs-energy analysis can
rescale existing per-cell (P, C, SumKV) traces to a realistic regime.

Standalone: talks to a running SGLang server, uses NVML directly. Radix
caching must be OFF on the server, and every prompt is uuid-salted anyway.
Homogeneous batches (all B requests share P, C) fired concurrently; energy
attribution is the NVML counter delta over the window divided by tokens.

Output: one JSONL row per (cell, rep) + a per-B least-squares fit summary.
"""
import argparse
import concurrent.futures
import json
import os
import random
import time
import uuid

import numpy as np
import pynvml
from openai import OpenAI

WORDS = ("energy scaling agent topology debate orchestrate search patch "
         "context token batch prefill decode cache joule watt measure").split()

B_LIST = [int(x) for x in os.environ.get("BATCH_COEFF_B_LIST", "1,8,32,64").split(",")]
# (family, P_target, C_target); actual tokens come from response usage.
PREFILL_CELLS = [("prefill", p, 16) for p in (512, 2048, 8192, 32768)]
DECODE_CELLS = [("decode", 64, c) for c in (256, 1024)]
KV_CELLS = [("kvread", p, 256) for p in (4096, 16384, 40960)]
# kvsep (2026-07-20): the original three families are ~perfectly collinear in
# kv_proxy vs (P, C) — R2 >= 0.999995 within each — so kv was structurally
# unidentifiable from them. Identification needs C VARIED at FIXED LARGE P:
# kv*C*(P + C/2) then scales with P while b*C does not. A P x C grid gives
# the fit curvature in both directions. Fits the A5000 80K pool at B=1-4;
# the A6000 pool carries it to B=8-16 (B=32 x P=16K needs ~640K -> skipped
# by the pool guard; that corner needs 80GB-class hardware).
KVSEP_CELLS = [("kvsep", p, c) for p in (8192, 16384) for c in (512, 2048, 4096)]
# decsmall (2026-07-22): tiny-P decode pair for same-card b(B) at high B on
# 24GB cards — the C=1024 decode cell exceeds the ~23K token pool at B>=48
# (pool-limited admission staggering, not clean concurrency); these fit to
# B=64 (64*(20+256) ~ 18K) and give b via the C-pair slope at matched tiny P.
DECSMALL_CELLS = [("decsmall", 16, c) for c in (64, 256)]
# longdecode (2026-07-23): fixed moderate prompt, C swept to reasoning/CoT
# scale. kv-energy = kv*C*(P + C/2) is QUADRATIC in C (each generated token
# re-reads the whole context), so at long C the KV-read term overtakes prefill
# (crossover C > a/kv ≈ 2000). Directly makes kv the dominant energy term and
# exposes the energy∝C² vs price∝C mispricing for long generation. Opt-in
# (BATCH_COEFF_LONGDECODE=1) — these cells generate up to 32K tokens (~14 min
# each at B=1) so they must not run inside the standard family sweep. Fits the
# A5000 at B=1-2 (P+C ≤ 41K < 49152 ctx; ~80K pool at low B).
LONGDECODE_CELLS = [("longdecode", 8192, c)
                    for c in (1024, 2048, 4096, 8192, 16384, 32768)]
CELLS = PREFILL_CELLS + DECODE_CELLS + KV_CELLS + KVSEP_CELLS + DECSMALL_CELLS
if os.environ.get("BATCH_COEFF_LONGDECODE", "0") == "1":
    CELLS = LONGDECODE_CELLS   # dedicated long-generation run, nothing else
MAX_CONC_TOKENS = int(os.environ.get("BATCH_COEFF_MAX_CONC_TOKENS", 450_000))  # pool guard: skip cells with B*(P+C) above (A6000 default; a5000 ~80K)
N_REPS = 5
MAX_B = int(os.environ.get("BATCH_COEFF_MAX_B", 1_000_000))  # skip B above server slot cap (wave contamination)
N_WARMUP = 5


def make_prompt(n_tokens):
    # ~1 token/word filler + uuid salt; actual count read from usage.
    rng = random.Random()
    body = " ".join(rng.choice(WORDS) for _ in range(max(n_tokens - 16, 8)))
    return f"[{uuid.uuid4()}] Repeat nothing. Filler follows: {body}"


def gpu_energy_mj(h):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)


def measure_idle(h, seconds=10):
    samples = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        samples.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.2)
    return float(np.mean(samples))


def one_request(client, model, prompt, max_tokens):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"ignore_eos": True},
        )
        return r.usage.prompt_tokens, r.usage.completion_tokens
    except Exception as e:                     # 400 over-cap, timeouts: skip, don't kill
        print(f"    request failed: {str(e)[:140]}", flush=True)
        return None


def run_cell(client, model, h, idle_w, B, p_tgt, c_tgt):
    prompts = [make_prompt(p_tgt) for _ in range(B)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=B) as ex:
        t0 = time.time()
        e0 = gpu_energy_mj(h)
        futs = [ex.submit(one_request, client, model, pr, c_tgt)
                for pr in prompts]
        usages = [f.result() for f in futs]
        e1 = gpu_energy_mj(h)
        t1 = time.time()
    n_failed = sum(1 for u in usages if u is None)
    usages = [u for u in usages if u is not None]
    wall = t1 - t0
    e_total = (e1 - e0) / 1000.0
    e_dyn = e_total - idle_w * wall
    if not usages:
        return {"B": B, "p_target": p_tgt, "c_target": c_tgt, "error": "all_requests_failed",
                "n_failed": n_failed, "wall_s": round(wall, 2)}
    P = sum(u[0] for u in usages)
    C = sum(u[1] for u in usages)
    # Per-sequence KV-token-reads proxy: sum over decode steps of ctx length.
    kv = sum(u[1] * (u[0] + u[1] / 2.0) for u in usages)
    return {"B": B, "p_target": p_tgt, "c_target": c_tgt, "n_failed": n_failed,
            "prompt_tokens": P, "completion_tokens": C, "kv_proxy": kv,
            "energy_total_j": round(e_total, 2),
            "energy_dynamic_j": round(e_dyn, 2),
            "wall_s": round(wall, 2), "idle_power_w": round(idle_w, 1)}


def fit_per_b(rows):
    print("\n=== per-B least-squares fit: E_dyn = a*P + b*C + kv*KVproxy ===")
    print(f"{'B':>4} {'a (J/ptok)':>12} {'b (J/ctok)':>12} {'kv':>12} "
          f"{'b/a':>8} {'n':>4}")
    for B in sorted({r["B"] for r in rows}):
        rs = [r for r in rows if r["B"] == B
              and r.get("energy_dynamic_j", 0) > 0]
        if len(rs) < 4:
            print(f"{B:>4}  (too few rows: {len(rs)})")
            continue
        X = np.array([[r["prompt_tokens"], r["completion_tokens"],
                       r["kv_proxy"]] for r in rs])
        y = np.array([r["energy_dynamic_j"] for r in rs])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, kv = coef
        ratio = b / a if a > 0 else float("nan")
        print(f"{B:>4} {a:>12.5f} {b:>12.5f} {kv:>12.3e} {ratio:>8.1f} "
              f"{len(rs):>4}")
    print("Compare b/a with API price ratios (3-5x) to read off the")
    print("implied provider batch size; kv(B) flat vs falling decides the")
    print("structural cost!=energy claim.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-url", default="http://localhost:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-index", type=int,
                    default=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                                .split(",")[0] or 0))
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
    client = OpenAI(base_url=args.sglang_url, api_key="EMPTY", timeout=7200)

    print(f"GPU: {pynvml.nvmlDeviceGetName(h)}")
    # Discover real per-request/total token caps from the server (the pool, not the
    # context flag, binds on small cards; hybrid mamba slots shrink it further).
    import requests as _rq
    max_total = MAX_CONC_TOKENS
    serving_meta = {}
    try:
        info = _rq.get(args.sglang_url.rsplit("/v1", 1)[0] + "/get_server_info", timeout=10).json()
        for k in ("max_total_num_tokens", "max_total_tokens"):
            if k in info:
                max_total = int(info[k]); break
        # Record graph/slot config per row: decode batches above cuda-graph-max-bs
        # run EAGER (kernel-launch overhead ~2x b for the hybrid model) — rows
        # from the two modes must be distinguishable (2026-07-21 A5000 lesson).
        for k in ("cuda_graph_max_bs", "max_running_requests"):
            if k in info:
                serving_meta[k] = info[k]
        print(f"server max_total_num_tokens: {max_total}; serving: {serving_meta}")
    except Exception as e:
        print(f"server info unavailable ({e}); using env/default cap {max_total}")
    max_req = int(0.95 * max_total)
    for _ in range(N_WARMUP):
        one_request(client, args.model, make_prompt(256), 32)
    idle_w = measure_idle(h)
    print(f"Idle power: {idle_w:.1f} W")

    # Resume: skip (family, B, P, C, rep) already in the output file.
    done = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            try:
                d = json.loads(line)
                done.add((d["family"], d["B"], d["p_target"],
                          d["c_target"], d["rep"]))
            except Exception:
                continue

    rows = []
    with open(args.output, "a") as out:
        for family, p_tgt, c_tgt in CELLS:
            for B in B_LIST:
                if B > MAX_B:
                    print(f"SKIP (B > slot cap {MAX_B}): {family} B={B}")
                    continue
                est_p = int(1.25 * p_tgt)          # word->token overshoot margin
                if est_p + c_tgt > max_req:
                    print(f"SKIP (per-request cap {max_req}): {family} B={B} "
                          f"P={p_tgt} C={c_tgt}")
                    continue
                if B * (est_p + c_tgt) > min(MAX_CONC_TOKENS, max_total):
                    print(f"SKIP (pool budget {min(MAX_CONC_TOKENS, max_total)}): {family} B={B} "
                          f"P={p_tgt} C={c_tgt}")
                    continue
                n_skipped = sum(1 for rep in range(N_REPS)
                                if (family, B, p_tgt, c_tgt, rep) in done)
                if n_skipped:
                    print(f"RESUME-SKIP: {family} B={B} P={p_tgt} C={c_tgt} "
                          f"({n_skipped}/{N_REPS} reps already in output)")
                for rep in range(N_REPS):
                    key = (family, B, p_tgt, c_tgt, rep)
                    if key in done:
                        continue
                    r = run_cell(client, args.model, h, idle_w,
                                 B, p_tgt, c_tgt)
                    r.update({"family": family, "rep": rep,
                              "timestamp": time.time(), **serving_meta})
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    rows.append(r)
                    print(f"{family} B={B} P={p_tgt} C={c_tgt} rep={rep}: "
                          f"{r['energy_dynamic_j']} J, {r['wall_s']} s")
                # Refresh idle estimate periodically (thermal drift).
                if random.random() < 0.1:
                    idle_w = measure_idle(h, 5)

    for line in open(args.output):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    # De-dup (rows appended twice for fresh cells)
    seen, uniq = set(), []
    for r in rows:
        k = (r["family"], r["B"], r["p_target"], r["c_target"], r["rep"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    fit_per_b(uniq)


if __name__ == "__main__":
    main()
