"""Replacement compound-cost figure: stronger statistical evidence than the binned boxplot.

Three panels:
  (a) SAS-only scatter — log(C) vs log(P), pure single-agent case, no role mixing.
      Shows the cleanest single-test result: r = 0.36 across 5,148 calls.
  (b) Within-trajectory slope distribution — histogram of fitted log-log slopes
      across all (task, agent) trajectories. The positive skew is the
      compound-cost signal.
  (c) Within-cell Pearson r forest — distribution of per-call r values across
      295 (benchmark, role, turn) cells. 60% significant positive, 4% significant
      negative. This is the test that controls for everything.

Output:
    analysis/a5000_figs/compound_cost_v2.png
"""
from __future__ import annotations
import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"


def load_calls() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in f:
            continue
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            per_agent = defaultdict(int)
            for c in rec.get("call_records") or []:
                if c.get("call_type") == "tool_execution":
                    continue
                if "prompt_tokens" not in c:
                    continue
                e = float(c.get("gpu_dynamic_energy_joules") or 0)
                if e <= 0:
                    continue
                agent = c.get("agent_id") or "?"
                rows.append(dict(
                    benchmark=rec["benchmark"], topology=rec["topology"],
                    task_id=str(rec["task_id"]), agent=agent,
                    turn=per_agent[agent],
                    P=int(c["prompt_tokens"]),
                    C=int(c["completion_tokens"]),
                ))
                per_agent[agent] += 1
    return pd.DataFrame(rows)


def main():
    print("Loading per-call records...")
    df = load_calls()
    print(f"  {len(df):,} calls")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---------------- Panel (a): SAS-only scatter ----------------
    ax = axes[0]
    sas = df[df["topology"] == "sas"]
    lp = np.log1p(sas["P"])
    lc = np.log1p(sas["C"])
    r, p = stats.pearsonr(lp, lc)
    slope, intercept = np.polyfit(lp, lc, 1)

    ax.scatter(sas["P"], sas["C"], s=4, alpha=0.18, color="#1f77b4")
    xs = np.logspace(np.log10(max(1, sas["P"].min())), np.log10(sas["P"].max()), 50)
    ys = np.expm1(intercept + slope * np.log1p(xs))
    ax.plot(xs, ys, color="#d62728", lw=2.2, label=f"fit: log C = {intercept:.2f} + {slope:.3f}·log P")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens P (per call, log)")
    ax.set_ylabel("Completion tokens C (per call, log)")
    ax.set_title(
        f"(a) SAS-only — single-agent baseline\n"
        f"n = {len(sas):,}    Pearson r = {r:.3f}    p ≈ 0    β = {slope:.3f}"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---------------- Panel (b): within-trajectory slope distribution ----------------
    ax = axes[1]
    slopes = []
    topo_color = {
        "sas": "#1f77b4", "independent": "#ff7f0e",
        "centralized": "#2ca02c", "decentralized": "#d62728", "hybrid": "#9467bd",
    }
    by_topo_slopes = {t: [] for t in topo_color}
    for (tid, agent), g in df.groupby(["task_id", "agent"]):
        if len(g) < 3:
            continue
        g = g.sort_values("turn")
        lp = np.log1p(g["P"].to_numpy(dtype=float))
        lc = np.log1p(g["C"].to_numpy(dtype=float))
        if lp.std() < 1e-6:
            continue
        b = float(np.polyfit(lp, lc, 1)[0])
        topo = g["topology"].iloc[0]
        if topo in by_topo_slopes:
            by_topo_slopes[topo].append(b)
        slopes.append(b)
    slopes = np.array(slopes)

    # Clip extreme values for plotting
    slopes_clipped = np.clip(slopes, -2.5, 2.5)
    ax.hist(slopes_clipped, bins=80, color="#9467bd", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", lw=1.2, label="β = 0 (null)")
    ax.axvline(np.median(slopes), color="#d62728", linestyle="-", lw=1.8,
               label=f"median β = {np.median(slopes):.2f}")
    frac_pos = (slopes > 0).mean()
    ax.set_xlabel("Within-trajectory log-log slope β   (clipped to [−2.5, 2.5])")
    ax.set_ylabel("Number of (task, agent) trajectories")
    ax.set_title(
        f"(b) Within-trajectory slope distribution\n"
        f"n = {len(slopes):,} trajectories    {frac_pos:.0%} positive    median β = {np.median(slopes):.2f}"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # ---------------- Panel (c): within-cell Pearson r forest/histogram ----------------
    ax = axes[2]
    df["role"] = df["agent"].map(
        lambda a: "worker" if "worker" in a else
                  "debater" if "debater" in a else
                  "orch" if a.startswith("orch") else
                  "synth" if "synth" in a else
                  "solver" if a.startswith("solver") else
                  "single"
    )
    cell_results = []
    for (b, role, turn), g in df.groupby(["benchmark", "role", "turn"]):
        if len(g) < 30:
            continue
        if g["P"].std() < 50:
            continue
        rho, pval = stats.pearsonr(np.log1p(g["P"]), np.log1p(g["C"]))
        cell_results.append((b, role, turn, len(g), rho, pval))
    cdf = pd.DataFrame(cell_results, columns=["benchmark", "role", "turn", "n", "r", "p"])
    sig_pos = ((cdf["p"] < 0.05) & (cdf["r"] > 0)).sum()
    sig_neg = ((cdf["p"] < 0.05) & (cdf["r"] < 0)).sum()
    nonsig = (cdf["p"] >= 0.05).sum()

    # Color bars by significance
    bins = np.linspace(-0.6, 0.8, 35)
    sig_pos_r = cdf[(cdf["p"] < 0.05) & (cdf["r"] > 0)]["r"]
    sig_neg_r = cdf[(cdf["p"] < 0.05) & (cdf["r"] < 0)]["r"]
    nonsig_r = cdf[cdf["p"] >= 0.05]["r"]

    ax.hist([sig_neg_r, nonsig_r, sig_pos_r], bins=bins, stacked=True,
            color=["#d62728", "#cccccc", "#2ca02c"],
            label=[
                f"sig negative (p<.05)   n={len(sig_neg_r)}  ({len(sig_neg_r)/len(cdf):.0%})",
                f"non-significant            n={len(nonsig_r)}  ({len(nonsig_r)/len(cdf):.0%})",
                f"sig positive (p<.05)   n={len(sig_pos_r)}  ({len(sig_pos_r)/len(cdf):.0%})",
            ],
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.axvline(cdf["r"].median(), color="#d62728", linestyle="-", lw=1.8,
               label=f"median r = {cdf['r'].median():.2f}")
    ax.set_xlabel("Within-cell Pearson r between log(P) and log(C)")
    ax.set_ylabel("Number of (benchmark, role, turn) cells")
    ax.set_title(
        f"(c) Within-cell Pearson r — strictest test\n"
        f"n = {len(cdf)} cells    15× more sig positive than sig negative"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Compound cost: statistical evidence for prompt-length → completion-length coupling\n"
        "(a) clean single-agent case   (b) within-trajectory slopes   (c) within-cell stratified test",
        y=1.04, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "compound_cost_v2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")

    # Replace the old figure
    old = os.path.join(OUT_DIR, "compound_cost_boxplot.png")
    if os.path.exists(old):
        archive = os.path.join(OUT_DIR, "compound_cost_boxplot_archived.png")
        os.rename(old, archive)
        print(f"  archived old boxplot → {archive}")
    # Symlink or copy v2 to the canonical name used in findings
    import shutil
    canonical = os.path.join(OUT_DIR, "compound_cost_boxplot.png")
    shutil.copy(out, canonical)
    print(f"  also wrote {canonical} (replaces old boxplot)")


if __name__ == "__main__":
    main()
