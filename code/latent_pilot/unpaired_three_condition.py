"""Unpaired three-condition analysis when per-task records have been lost.

Combines aggregate stats from a saved summary.json (older text + latent_we)
with current per-task records (diffusion). Computes unpaired delta CIs by
treating the saved 95% CIs as ±1.96·SE of the per-condition mean and
propagating SEs as: SE_delta = sqrt(SE_A² + SE_B²).

This is strictly weaker than the paired bootstrap on intersect(task_ids), but
it's what's possible without re-running text+latent_we. Use only as a
recovery interim until the paired data is restored.

Usage:
  python -m latent_pilot.unpaired_three_condition \
      --aggregates .../analysis/summary.json \
      --records    .../end_to_end_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

CONDITIONS = ["text", "latent_we", "latent_diffusion"]
COND_LABEL = {"text": "Text MAS", "latent_we": "W_e hybrid",
              "latent_diffusion": "Diffusion hybrid"}


def se_from_ci(ci_lo, ci_hi):
    """Approximate SE of the mean from a 95% bootstrap CI: SE = width / 3.92."""
    return (ci_hi - ci_lo) / (2 * 1.96)


def per_condition_from_records(records, cond):
    rs = [r for r in records if r.get("condition") == cond and not r.get("error")]
    if not rs:
        return None
    f1 = np.array([r.get("f1", 0.0) or 0.0 for r in rs], dtype=float)
    energy = np.array([r["energy"].get("gpu_dynamic_energy_joules", float("nan"))
                       for r in rs], dtype=float)
    f1 = f1[~np.isnan(f1)]
    energy = energy[~np.isnan(energy)]
    rng = np.random.default_rng(0)
    def boot(arr, n=2000):
        idx = rng.integers(0, len(arr), size=(n, len(arr)))
        return float(arr.mean()), float(np.quantile(arr[idx].mean(axis=1), 0.025)), \
               float(np.quantile(arr[idx].mean(axis=1), 0.975))
    return {
        "condition": cond,
        "label": COND_LABEL[cond],
        "n_tasks": len(rs),
        "mean_f1": boot(f1),
        "mean_energy_J": boot(energy),
    }


def per_condition_from_summary(summary_obj, cond):
    for c in summary_obj["conditions"]:
        if c["condition"] == cond:
            return {
                "condition": cond,
                "label": COND_LABEL[cond],
                "n_tasks": c["n_tasks"],
                "mean_f1": tuple(c["mean_f1"]),
                "mean_energy_J": tuple(c["mean_energy_J"]),
            }
    return None


def unpaired_delta(stat_a, stat_b, key):
    """Welch-style delta of two independent means using bootstrap-CI-implied SEs."""
    m_a, lo_a, hi_a = stat_a[key]
    m_b, lo_b, hi_b = stat_b[key]
    se_a = se_from_ci(lo_a, hi_a)
    se_b = se_from_ci(lo_b, hi_b)
    se_d = math.sqrt(se_a * se_a + se_b * se_b)
    m_d = m_b - m_a
    return {
        "mean_delta": m_d,
        "ci_lo": m_d - 1.96 * se_d,
        "ci_hi": m_d + 1.96 * se_d,
    }


def interpret_delta(metric, d, lower_is_better=False):
    lo, hi, m = d["ci_lo"], d["ci_hi"], d["mean_delta"]
    if lo < 0 < hi:
        return (f"{metric}: indistinguishable — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}] crosses 0", 0)
    if lower_is_better:
        if hi < 0:
            return (f"{metric}: B is lower — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", 1)
        return (f"{metric}: B is higher — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", -1)
    if lo > 0:
        return (f"{metric}: B is higher — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", 1)
    return (f"{metric}: B is lower — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", -1)


def fmt_ci(triple, digits=3):
    m, lo, hi = triple
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregates", required=True, type=Path,
                    help="Prior analysis/summary.json with text + latent_we aggregates")
    ap.add_argument("--records", required=True, type=Path,
                    help="Current end_to_end_results.jsonl (contains diffusion records)")
    args = ap.parse_args()

    summary = json.load(open(args.aggregates))
    records = []
    with open(args.records) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "condition" in r:
                records.append(r)

    stats = {}
    text = per_condition_from_summary(summary, "text")
    we = per_condition_from_summary(summary, "latent_we")
    if text: stats["text"] = text
    if we: stats["latent_we"] = we
    df = per_condition_from_records(records, "latent_diffusion")
    if df: stats["latent_diffusion"] = df

    print()
    print("UNPAIRED three-condition reconstruction")
    print("=" * 130)
    print("Note: text + latent_we from saved aggregates (paired data lost). "
          "Diffusion from current per-task records. CIs are bootstrap on the means; deltas use")
    print("Welch-style SE propagation, NOT a paired bootstrap. Re-run on cluster restores paired analysis.")
    print()
    print(f"{'condition':>20}  {'n':>5}  {'F1 (95% CI on mean)':>28}  {'Energy J (95% CI)':>26}")
    print("-" * 130)
    for c in CONDITIONS:
        if c not in stats: continue
        s = stats[c]
        print(f"{s['label']:>20}  {s['n_tasks']:>5}  "
              f"{fmt_ci(s['mean_f1']):>28}  "
              f"{fmt_ci(s['mean_energy_J'], digits=1):>26}")
    print()

    pairs = [("text", "latent_we"), ("text", "latent_diffusion"),
             ("latent_we", "latent_diffusion")]
    print("Unpaired deltas (B - A) with Welch-propagated 95% CI:")
    print("-" * 130)
    verdicts = {}
    for a, b in pairs:
        if a not in stats or b not in stats: continue
        df1 = unpaired_delta(stats[a], stats[b], "mean_f1")
        de = unpaired_delta(stats[a], stats[b], "mean_energy_J")
        print(f"  {COND_LABEL[b]:>17} vs {COND_LABEL[a]:>17}  "
              f"ΔF1={df1['mean_delta']:+.3f} [{df1['ci_lo']:+.3f}, {df1['ci_hi']:+.3f}]  "
              f"ΔEnergy={de['mean_delta']:+.0f}J [{de['ci_lo']:+.0f}, {de['ci_hi']:+.0f}]")
        f1_v, f1_s = interpret_delta("F1", df1)
        en_v, en_s = interpret_delta("Energy", de, lower_is_better=True)
        verdicts[(a, b)] = (f1_v, f1_s, en_v, en_s)

    print()
    print("Auto-interpretation (CI-based):")
    print("=" * 130)
    for (a, b), (f1_v, f1_s, en_v, en_s) in verdicts.items():
        print(f"  {COND_LABEL[b]} vs {COND_LABEL[a]}:")
        print(f"    {f1_v}")
        print(f"    {en_v}")

    # Headline
    print()
    print("Headline:")
    if ("text", "latent_diffusion") in verdicts:
        f1_v, f1_s, en_v, en_s = verdicts[("text", "latent_diffusion")]
        if f1_s == 0 and en_s == 1:
            print("  Diffusion hybrid matches text F1 at LOWER energy — energy-efficient substitution succeeds.")
        elif f1_s == 1 and en_s == 1:
            print("  Diffusion hybrid IMPROVES F1 AND lowers energy — strict Pareto improvement.")
        elif f1_s == -1 and en_s == 1:
            print("  Diffusion hybrid trades F1 for energy savings.")
        elif f1_s == 0 and en_s == 0:
            print("  Diffusion hybrid matches text MAS on both F1 and energy — no measurable effect.")
        elif f1_s == 0 and en_s == -1:
            print("  Diffusion hybrid is DOMINATED by text MAS — F1 indistinguishable while energy is HIGHER.")
        elif f1_s == -1 and en_s == -1:
            print("  Diffusion hybrid is strictly DOMINATED by text MAS — worse F1 AND higher energy.")
        elif f1_s == 1 and en_s == -1:
            print("  Diffusion hybrid trades higher energy for higher F1 — accuracy-cost frontier shift.")
        elif f1_s == 1 and en_s == 0:
            print("  Diffusion hybrid IMPROVES F1 at indistinguishable energy — accuracy gain free.")
        elif f1_s == -1 and en_s == 0:
            print("  Diffusion hybrid LOSES F1 at indistinguishable energy — bridge harms downstream.")

    if ("latent_we", "latent_diffusion") in verdicts:
        f1_v, f1_s, _, _ = verdicts[("latent_we", "latent_diffusion")]
        if f1_s == 1:
            print("  Diffusion beats W_e closed-form on F1 — KL advantage transfers downstream.")
        elif f1_s == 0:
            print("  Diffusion ≈ W_e closed-form on F1 — bridge's KL gain does NOT translate to accuracy "
                  "(consistent with channel-muting).")
        else:
            print("  Diffusion underperforms W_e closed-form on F1 — learned bridge harms vs alignment alone.")

    if ("text", "latent_we") in verdicts:
        f1_v, f1_s, _, _ = verdicts[("text", "latent_we")]
        if f1_s == 0:
            print("  W_e hybrid matches text MAS on F1 — latent-channel substitution preserves accuracy.")
        elif f1_s == 1:
            print("  W_e hybrid IMPROVES on text MAS F1 — latent channel adds signal.")
        else:
            print("  W_e hybrid degrades text MAS F1.")


if __name__ == "__main__":
    main()
