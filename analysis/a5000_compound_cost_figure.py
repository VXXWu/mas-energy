"""Compound-cost figure: prompt length → completion length → energy.

Shows the headline mechanism in three panels:
  (a) C vs P scatter on log-log, with per-topology fit lines (the elasticity)
  (b) E vs P stacked: prefill (b·P) vs decode (c·C(P)) contribution
      → makes the dominance of decode visible
  (c) Energy vs (P, C) heatmap or contour, with measured points overlaid

Outputs:
    analysis/a5000_figs/compound_cost_figure.png
    analysis/a5000_figs/compound_cost_slopes.csv
"""
from __future__ import annotations
import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

# Energy regression coefficients (from project canon)
A_INT = -84.0
B_PR = 0.018
C_DE = 5.54

TOPO_COLOR = {
    "sas":            "#1f77b4",
    "independent":    "#ff7f0e",
    "centralized":    "#2ca02c",
    "decentralized":  "#d62728",
    "hybrid":         "#9467bd",
}


def load_calls() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        for line in open(path):
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
                P = int(c["prompt_tokens"])
                C = int(c["completion_tokens"])
                if e <= 0:
                    continue
                rows.append(dict(
                    benchmark=rec.get("benchmark", "?"),
                    topology=rec.get("topology", "?"),
                    P=P, C=C, E=e,
                ))
    return pd.DataFrame(rows)


def fit_loglog(x, y):
    """Return (slope, intercept) of log(y+1) on log(x+1)."""
    lx = np.log1p(np.asarray(x, dtype=float))
    ly = np.log1p(np.asarray(y, dtype=float))
    if lx.std() < 1e-6:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def main():
    df = load_calls()
    print(f"Loaded {len(df)} per-call records")

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.4])

    # ---------- Panel (a): C vs P with per-topology fit ----------
    ax = fig.add_subplot(gs[0, 0])
    slopes = []
    for topo in sorted(df["topology"].unique()):
        sub = df[df["topology"] == topo]
        if len(sub) < 50:
            continue
        col = TOPO_COLOR.get(topo, "gray")
        ax.scatter(sub["P"], sub["C"], s=3, alpha=0.12, color=col)
        slope, intercept = fit_loglog(sub["P"], sub["C"])
        if not np.isnan(slope):
            xs = np.linspace(np.log1p(sub["P"].min()), np.log1p(sub["P"].max()), 50)
            ys = intercept + slope * xs
            ax.plot(np.expm1(xs), np.expm1(ys), color=col, lw=2,
                    label=f"{topo}  β={slope:.2f}")
            slopes.append(dict(topology=topo, n=len(sub),
                               slope=slope, intercept=intercept))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens P (log)")
    ax.set_ylabel("Completion tokens C (log)")
    ax.set_title("(a) Compound cost: C grows with P\nlog-log slope β ≈ elasticity of completion length")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    pd.DataFrame(slopes).to_csv(os.path.join(OUT_DIR, "compound_cost_slopes.csv"), index=False)

    # ---------- Panel (b): prefill vs decode contribution to E ----------
    ax = fig.add_subplot(gs[0, 1])
    # Bin by P; in each bin compute mean prefill cost (b·P) and mean decode cost (c·C)
    bins = np.logspace(np.log10(max(50, df["P"].min())), np.log10(df["P"].max()), 25)
    df = df.assign(P_bin=pd.cut(df["P"], bins))
    grp = df.groupby("P_bin", observed=True).agg(
        P_mid=("P", "mean"), C_mean=("C", "mean"), E_mean=("E", "mean"))
    grp["E_prefill"] = B_PR * grp["P_mid"]
    grp["E_decode"] = C_DE * grp["C_mean"]
    width = grp["P_mid"] * 0.18
    ax.bar(grp["P_mid"], grp["E_prefill"], width=width, color="#1f77b4",
           label="Prefill: b·P")
    ax.bar(grp["P_mid"], grp["E_decode"], width=width, bottom=grp["E_prefill"],
           color="#d62728", label="Decode: c·C(P)")
    # Overlay measured E
    ax.plot(grp["P_mid"], grp["E_mean"], "ko-", lw=1.5, ms=4, label="Measured E")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens P (log, binned)")
    ax.set_ylabel("Energy per call (J, log)")
    ax.set_title("(b) Decode dominates total energy\nstacked: b·P (prefill) + c·C̄(P) (induced decode)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ---------- Panel (c): heatmap E(P, C) with overlaid points ----------
    ax = fig.add_subplot(gs[0, 2])
    P_grid = np.logspace(2, np.log10(df["P"].quantile(0.99)), 60)
    C_grid = np.logspace(0.5, np.log10(df["C"].quantile(0.99) + 1), 60)
    PG, CG = np.meshgrid(P_grid, C_grid)
    EG = A_INT + B_PR * PG + C_DE * CG
    im = ax.pcolormesh(PG, CG, EG / 1000, shading="auto", cmap="viridis",
                       norm=plt.matplotlib.colors.LogNorm(vmin=0.05, vmax=EG.max() / 1000))
    fig.colorbar(im, ax=ax, label="Predicted energy per call (kJ)")
    # Overlay measured points
    sub = df.sample(min(3000, len(df)), random_state=0)
    ax.scatter(sub["P"], sub["C"], s=4, color="white", alpha=0.5, edgecolor="black", linewidth=0.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prompt tokens P (log)")
    ax.set_ylabel("Completion tokens C (log)")
    ax.set_title("(c) Energy heatmap E = a + b·P + c·C\nmeasured calls overlaid (white)")

    fig.suptitle(
        "Compound cost — prompt length → completion length → energy\n"
        f"(a) elasticity β   (b) prefill vs decode share   (c) energy surface  |  c/b = {C_DE/B_PR:.0f}×",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "compound_cost_figure.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
