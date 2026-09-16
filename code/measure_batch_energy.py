#!/usr/bin/env python3
"""Batched-serving energy grid: measure J/completion-token at (concurrency B, context P).

Answers the bracket-middle question (diagnostics G-cost): our serial protocol measures
the B=1 end; the amortization model predicts J/decode-token(B, ctx) = c/B + d·ctx.
This driver measures that surface directly via steady-state window metering — per-call
NVML attribution is impossible under concurrency, so we meter aggregate energy over a
fixed window and divide by tokens completed in the window.

Design:
  - Synthetic workload: each request = fresh random-word prompt of ~P tokens (unique
    per request so RadixAttention gets ~zero hits and prefill is honestly uncached),
    completion capped at C_GEN tokens (temperature 0.7, long-form instruction so the
    model reliably generates to the cap; completion-heavy so decode dominates).
  - B worker threads each loop send-receive against the same SGLang server.
  - Steady state: warmup WARMUP_S, then a window of WINDOW_S; energy = NVML counter
    delta over the window; token counts = sum of usage fields of requests FINISHING
    inside the window (boundary error ~ request_time/window; window is sized >> one
    request). Idle power measured at start (10 s quiet) for dynamic subtraction;
    raw and dynamic both recorded.
  - Feasibility cap: B·(P + C_GEN + margin) ≤ TOKEN_BUDGET (KV-pool limit on 24 GB).

Output: one JSON line per grid point → $OUTPUT_DIR/batch_grid.jsonl
Usage: python measure_batch_energy.py --sglang-url http://localhost:PORT/v1 \
          --model-path Qwen/Qwen3.5-9B --output-dir RESULTS_DIR
"""
import argparse, json, os, random, threading, time

import pynvml
from openai import OpenAI

C_GEN = 512
WARMUP_S = 45
WINDOW_S = 180
TOKEN_BUDGET = int(os.environ.get("BATCH_GRID_TOKEN_BUDGET", 90_000))
CTX_GRID = [1000, 2000, 4000, 8000, 16000, 32000]
B_GRID = [1, 2, 4, 8, 16, 32, 64]

VOCAB = ("time year people way day man thing woman life child world school state family "
         "student group country problem hand part place case week company system program "
         "question work government number night point home water room mother area money "
         "story fact month lot right study book eye job word business issue side kind head "
         "house service friend father power hour game line end member law car city name").split()


def make_prompt(n_tokens):
    # ~1 common word ≈ 1.3 tokens; unique shuffle per request defeats prefix caching
    words = [random.choice(VOCAB) for _ in range(max(int(n_tokens / 1.3), 20))]
    return ("Write a long, detailed, meandering story that weaves together every one of "
            "the following words. Do not stop early; keep writing until cut off:\n"
            + " ".join(words))


def worker(client, model, ctx, cgen, stop, results, lock):
    while not stop.is_set():
        t_sub = time.monotonic()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": make_prompt(ctx)}],
                max_tokens=cgen, temperature=0.7,
            )
            with lock:
                results.append((time.monotonic(), time.monotonic() - t_sub,
                                r.usage.prompt_tokens, r.usage.completion_tokens))
        except Exception as e:
            with lock:
                results.append((time.monotonic(), None, None, str(e)[:120]))
            time.sleep(2)


def read_mj(h):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "batch_grid.jsonl")

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    client = OpenAI(base_url=args.sglang_url, api_key="none", timeout=600)

    # idle baseline (server loaded, quiet)
    time.sleep(10)
    e0, t0 = read_mj(h), time.monotonic()
    time.sleep(10)
    idle_w = (read_mj(h) - e0) / 1000.0 / (time.monotonic() - t0)
    print(f"idle power (loaded, quiet): {idle_w:.1f} W", flush=True)

    done = set()
    if os.path.exists(out_path):                      # resume: skip completed points
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["ctx"], r["B"], r.get("c_gen", C_GEN)))
            except Exception:
                pass

    # pass 1: main grid at C_GEN. pass 2: short-completion subset at the same
    # (ctx, B) — varies P/C at fixed ctx to separate the prefill slope (b·ctx/C)
    # from the KV slope (d·ctx), which are otherwise collinear in ctx.
    points = [(ctx, B, C_GEN) for ctx in CTX_GRID for B in B_GRID]
    points += [(ctx, B, 128) for ctx in (2000, 8000, 16000) for B in (1, 4, 8)]
    for ctx, B, cgen in points:
            if B * (ctx + cgen + 256) > TOKEN_BUDGET:
                continue
            if (ctx, B, cgen) in done:
                print(f"skip ctx={ctx} B={B} cgen={cgen} (done)", flush=True)
                continue
            print(f"=== ctx={ctx} B={B} cgen={cgen} ===", flush=True)
            stop, lock, results = threading.Event(), threading.Lock(), []
            threads = [threading.Thread(target=worker, args=(client, args.model_path, ctx, cgen, stop, results, lock),
                                        daemon=True) for _ in range(B)]
            for th in threads:
                th.start()
            time.sleep(WARMUP_S)
            e_start, t_start = read_mj(h), time.monotonic()
            time.sleep(WINDOW_S)
            e_end, t_end = read_mj(h), time.monotonic()
            stop.set()
            time.sleep(1)
            wall = t_end - t_start
            with lock:
                inwin = [(dur, p, c) for (tt, dur, p, c) in results
                         if t_start <= tt <= t_end and isinstance(c, int)]
                errs = sum(1 for (_, _, p, c) in results if not isinstance(c, int))
            P_w, C_w = sum(p for _, p, _ in inwin), sum(c for _, _, c in inwin)
            # Little's law: time-integrated concurrency of completed requests. Near the
            # KV-pool cap SGLang queues excess streams, so B_eff < nominal B; the
            # amortization analysis must use B_eff, not B.
            b_eff = sum(dur for dur, _, _ in inwin) / wall if inwin else 0.0
            E_raw = (e_end - e_start) / 1000.0
            E_dyn = max(0.0, E_raw - idle_w * wall)
            rec = dict(ctx=ctx, B=B, c_gen=cgen, B_eff=round(b_eff, 2), window_s=wall,
                       n_completions=len(inwin), n_errors=errs,
                       prompt_tokens=P_w, completion_tokens=C_w,
                       energy_raw_j=E_raw, energy_dyn_j=E_dyn, idle_w=idle_w,
                       avg_power_w=E_raw / wall,
                       j_per_completion_token_dyn=(E_dyn / C_w) if C_w else None,
                       completion_tok_per_s=C_w / wall)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if not inwin:
                print("WARNING: no completions in window — check server/feasibility", flush=True)

    print("grid complete", flush=True)


if __name__ == "__main__":
    main()
