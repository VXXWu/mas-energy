"""E2-MAS mechanism ablation analysis.

Reads the 5 jsonls produced by run_e2mas_combined_qwen3.sbatch and
computes paired per-task metrics to show the incremental contribution
of each mechanism (agreement gate, tool cache, grammar constraint) and
whether they compound when stacked.

Input files (all on same 30 QAMPARI tasks, kv_share m=0):
  (1) eval_<model>_qampari_agentic_k10_m0.jsonl                                    — baseline
  (2) eval_<model>_qampari_agentic_k10_m0_agreeskip.jsonl                          — agreement only
  (3) eval_<model>_qampari_agentic_k10_m0_toolcache.jsonl                          — cache only
  (4) eval_<model>_qampari_agentic_k10_m0_grammar.jsonl                            — grammar only
  (5) eval_<model>_qampari_agentic_k10_m0_agreeskip_toolcache_grammar.jsonl        — all three

Reports for each condition (on the common task intersection):
  loose_acc / F1 / text_energy (mean, vs baseline delta)
  latent_energy (kv_share m=0 — same mechanism, sanity check parity)
  per-mechanism activation stats (skip rate, cache hit rate)
"""
import json
import statistics
from pathlib import Path

RESULTS = Path("mas-energy/results/latent_pilot")


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def agg(rows, field):
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    return statistics.mean(vals) if vals else None


def main():
    model_tag = "Qwen_Qwen3-8B"
    bench = "qampari"
    m = 0
    base_stem = f"eval_{model_tag}_{bench}_agentic_k10_m{m}"

    configs = [
        ("baseline",        f"{base_stem}.jsonl"),
        ("agree-skip",      f"{base_stem}_agreeskip.jsonl"),
        ("tool-cache",      f"{base_stem}_toolcache.jsonl"),
        ("grammar",         f"{base_stem}_grammar.jsonl"),
        ("all-three",       f"{base_stem}_agreeskip_toolcache_grammar.jsonl"),
    ]

    loaded = {}
    for label, fname in configs:
        rows = load(fname)
        if rows is None:
            print(f"MISSING: {fname}")
            loaded[label] = None
        else:
            loaded[label] = {r["task_id"]: r for r in rows}

    # Task IDs present in ALL loaded configs
    non_none = [d for d in loaded.values() if d is not None]
    if not non_none:
        print("No jsonls loaded.")
        return
    common = set(non_none[0].keys())
    for d in non_none[1:]:
        common &= set(d.keys())
    common = sorted(common)
    print(f"Common tasks across {sum(1 for d in non_none)} loaded configs: {len(common)}")

    if not common:
        print("No paired tasks; aborting.")
        return

    # Table of metrics
    print(f"\n{'Config':<14} {'n':>3} {'loose_acc':>10} {'F1':>7} {'Text E (J)':>12} {'vs base':>9} {'Lat E (J)':>11}")
    print("-" * 78)
    base_rows = None
    for label, fname in configs:
        d = loaded[label]
        if d is None:
            print(f"{label:<14}     (missing)")
            continue
        rows = [d[t] for t in common]
        la = agg(rows, "text_loose_accuracy")
        f1 = agg(rows, "text_f1")
        te = agg(rows, "text_energy_j")
        le = agg(rows, "latent_energy_j")
        if label == "baseline":
            base_te = te
            base_la = la
            pct = ""
        else:
            pct = f"{(te - base_te) / max(base_te, 1) * 100:+.1f}%" if base_te else ""
        print(f"{label:<14} {len(rows):>3d} {la:>10.3f} {f1:>7.3f} "
              f"{te:>12.0f} {pct:>9} {le:>11.0f}")

    # Per-mechanism activation stats
    print("\nMechanism activation stats:")
    for label, fname in configs:
        d = loaded[label]
        if d is None or label == "baseline":
            continue
        rows = [d[t] for t in common]
        parts = []
        if any(r.get("text_skipped_debate") is not None for r in rows):
            skipped = sum(1 for r in rows if r.get("text_skipped_debate"))
            parts.append(f"agreement gate: skipped {skipped}/{len(rows)} tasks")
        if any(r.get("tool_cache_hits") is not None for r in rows):
            hits = sum(r.get("tool_cache_hits", 0) for r in rows)
            misses = sum(r.get("tool_cache_misses", 0) for r in rows)
            tot = hits + misses
            parts.append(f"tool cache: {hits}/{tot} hits ({hits/max(tot,1):.0%})")
        if parts:
            print(f"  {label:<14} {' | '.join(parts)}")

    # Pairwise per-task Δ energy and Δ accuracy vs baseline
    print("\nPaired per-task deltas vs baseline (mean ± stdev, n = common tasks):")
    if loaded["baseline"] is None:
        print("  (baseline missing, cannot compute deltas)")
    else:
        for label, fname in configs[1:]:
            d = loaded[label]
            if d is None:
                continue
            te_d = []
            la_d = []
            for t in common:
                te_d.append(d[t]["text_energy_j"] - loaded["baseline"][t]["text_energy_j"])
                la_d.append(d[t]["text_loose_accuracy"] - loaded["baseline"][t]["text_loose_accuracy"])
            mean_te = statistics.mean(te_d)
            std_te = statistics.stdev(te_d) if len(te_d) > 1 else 0
            mean_la = statistics.mean(la_d)
            std_la = statistics.stdev(la_d) if len(la_d) > 1 else 0
            print(f"  {label:<14}  Δenergy = {mean_te:+.0f} ± {std_te:.0f} J   "
                  f"Δloose_acc = {mean_la:+.3f} ± {std_la:.3f}")

    # Compounding check: does the all-three combination approximately equal
    # the sum of individual mechanism savings?
    if all(loaded[c] is not None for c in ("baseline", "agree-skip", "tool-cache", "grammar", "all-three")):
        delta_sum = 0
        for label in ("agree-skip", "tool-cache", "grammar"):
            d = loaded[label]
            per_task = [d[t]["text_energy_j"] - loaded["baseline"][t]["text_energy_j"]
                        for t in common]
            delta_sum += statistics.mean(per_task)
        all_delta = statistics.mean([loaded["all-three"][t]["text_energy_j"]
                                      - loaded["baseline"][t]["text_energy_j"]
                                      for t in common])
        print(f"\nCompounding check:")
        print(f"  Sum of individual mechanism Δenergy:   {delta_sum:+.0f} J")
        print(f"  Combined all-three Δenergy:            {all_delta:+.0f} J")
        ratio = all_delta / max(abs(delta_sum), 1)
        if ratio > 0.9:
            print(f"  → mechanisms compound ~additively (ratio {ratio:.2f}).")
        elif ratio > 0.5:
            print(f"  → mechanisms partially compound (ratio {ratio:.2f}) — some overlap.")
        else:
            print(f"  → mechanisms heavily overlap (ratio {ratio:.2f}) — redundant.")


if __name__ == "__main__":
    main()
