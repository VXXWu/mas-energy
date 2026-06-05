"""5-panel per-topology compound-cost figure with LINEAR y-axis (completion tokens).

X-axis: prompt tokens P, log scale (P spans 0–80K so log is necessary).
Y-axis: completion tokens C, LINEAR scale (so the C ≈ 30 tool-call preamble
cluster is visible at its actual position, and the truncation cap at C=4096
sits at the top).

The Pearson r and log-log slope β are still computed in log-log space (the
canonical elasticity), and reported in the title — the change is purely visual.

Output:
    analysis/a5000_figs/compound_cost_per_topology_linear.png
"""
from __future__ import annotations
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

TOPO_ORDER = ["sas", "independent", "centralized", "decentralized", "hybrid"]
TOPO_LABEL = {
    "sas": "SAS",
    "independent": "Independent",
    "centralized": "Centralized",
    "decentralized": "Decentralized",
    "hybrid": "Hybrid",
}
TOPO_COLOR = {
    "sas":            "#1f77b4",
    "independent":    "#ff7f0e",
    "centralized":    "#2ca02c",
    "decentralized":  "#d62728",
    "hybrid":         "#9467bd",
}


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
            for c in rec.get("call_records") or []:
                if c.get("call_type") == "tool_execution":
                    continue
                if "prompt_tokens" not in c or "completion_tokens" not in c:
                    continue
                e = float(c.get("gpu_dynamic_energy_joules") or 0)
                if e <= 0:
                    continue
                rows.append(dict(
                    topology=rec.get("topology", "?"),
                    P=int(c["prompt_tokens"]),
                    C=int(c["completion_tokens"]),
                ))
    return pd.DataFrame(rows)


def main():
    print("Loading per-call records...")
    df = load_calls()
    print(f"  {len(df):,} calls")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharey=True)

    # Use a common y-limit so panels are comparable. Cap at 1500 to keep the
    # bulk of the distribution visible (the 4096 truncation tail and the rare
    # outliers above 1500 would otherwise compress everything to the bottom).
    y_top = 1500

    for ax, topo in zip(axes, TOPO_ORDER):
        sub = df[df["topology"] == topo]
        if len(sub) < 50:
            ax.set_visible(False)
            continue
        # Statistics still computed in log-log for canonical elasticity reporting
        lp = np.log1p(sub["P"])
        lc = np.log1p(sub["C"])
        r, p = stats.pearsonr(lp, lc)
        slope, intercept = np.polyfit(lp, lc, 1)

        P = sub["P"].to_numpy(dtype=float)
        C = sub["C"].to_numpy(dtype=float)
        mask = (P > 0) & (C > 0) & (C <= y_top)
        # Hexbin: log x, linear y
        hb = ax.hexbin(
            P[mask], C[mask],
            gridsize=(60, 50),
            xscale="log", yscale="linear",
            cmap="viridis", bins="log", mincnt=1,
        )
        # Overlay log-log fit projected onto linear y axis
        xs = np.logspace(np.log10(max(1, sub["P"].min())),
                         np.log10(sub["P"].max()), 80)
        ys = np.expm1(intercept + slope * np.log1p(xs))
        ax.plot(xs, ys, color="#d62728", lw=2.0,
                label=f"log-log fit β = {slope:.3f}")

        ax.set_xscale("log")
        ax.set_xlabel("Prompt tokens P (per call, log)")
        if ax is axes[0]:
            ax.set_ylabel("Completion tokens C (per call, linear)")
        # Reference markers for the most prominent C clusters
        ax.axhline(30, color="white", lw=0.7, linestyle=":", alpha=0.7)
        ax.text(P[mask].max() * 0.7, 32, "C≈30 tool-call preamble cluster",
                fontsize=7, color="white")
        # P-value text
        if p < 1e-300:
            p_str = "p < 1e-300"
        elif p < 1e-3:
            p_str = f"p = {p:.1e}"
        else:
            p_str = f"p = {p:.3f}"
        n_in_view = mask.sum()
        n_clipped = (C > y_top).sum()
        ax.set_title(
            f"{TOPO_LABEL[topo]}\n"
            f"n = {len(sub):,}    r = {r:.3f}    {p_str}\n"
            f"({n_clipped:,} calls > {y_top} clipped from view)",
            fontsize=10,
        )
        ax.set_ylim(0, y_top)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Compound cost per topology — log P, LINEAR C\n"
        "Hexbin density (log color scale). White dotted line at C=30 marks the "
        "tool-call preamble cluster.",
        y=1.05, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "compound_cost_per_topology_linear.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
