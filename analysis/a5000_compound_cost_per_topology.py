"""5-panel per-topology version of the compound-cost scatter:
log(C) vs log(P) with fit line + Pearson r + slope β, faceted by topology.

Output: analysis/a5000_figs/compound_cost_per_topology.png
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

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    for ax, topo in zip(axes, TOPO_ORDER):
        sub = df[df["topology"] == topo]
        if len(sub) < 50:
            ax.set_visible(False)
            continue
        lp = np.log1p(sub["P"])
        lc = np.log1p(sub["C"])
        r, p = stats.pearsonr(lp, lc)
        slope, intercept = np.polyfit(lp, lc, 1)

        color = TOPO_COLOR[topo]
        # Use hexbin density rather than alpha-blended scatter to avoid the
        # saturation-edge artifact that creates an apparent horizontal line
        # at C ≈ 200-300 in dense regions.
        P = sub["P"].to_numpy(dtype=float)
        C = sub["C"].to_numpy(dtype=float)
        mask = (P > 0) & (C > 0)
        ax.hexbin(P[mask], C[mask], gridsize=45,
                  xscale="log", yscale="log",
                  cmap="viridis", bins="log", mincnt=1, alpha=0.95)
        xs = np.logspace(np.log10(max(1, sub["P"].min())), np.log10(sub["P"].max()), 60)
        ys = np.expm1(intercept + slope * np.log1p(xs))
        ax.plot(xs, ys, color="#d62728", lw=2.2,
                label=f"fit β = {slope:.3f}")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Prompt tokens P (per call, log)")
        if ax is axes[0]:
            ax.set_ylabel("Completion tokens C (per call, log)")
        # Format p-value text
        if p < 1e-300:
            p_str = "p < 1e-300"
        elif p < 1e-3:
            p_str = f"p = {p:.1e}"
        else:
            p_str = f"p = {p:.3f}"
        ax.set_title(
            f"{TOPO_LABEL[topo]}\n"
            f"n = {len(sub):,}    r = {r:.3f}    {p_str}",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Compound cost per topology — log(C) vs log(P) per call\n"
        "Pearson r and log-log slope β reported per topology",
        y=1.04, fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "compound_cost_per_topology.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
