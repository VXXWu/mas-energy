"""Dual-row Pareto figure for the A5000 dataset.

Top row: tokens vs accuracy (one panel per benchmark)
Bottom row: energy vs accuracy (same)

A point that is on the frontier in one row but not the other is highlighted
with a red circle — the "frontier shift" that proves tokens != joules.

Outputs:
    analysis/a5000_figs/pareto_dual.png
    analysis/a5000_figs/pareto_dual.csv      (per-config means used)
"""
from __future__ import annotations
import os
import json
import glob
import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"
OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)

TOPO_STYLE = {
    "sas":            {"color": "#1f77b4", "marker": "o", "label": "SAS"},
    "independent":    {"color": "#ff7f0e", "marker": "s", "label": "Independent"},
    "centralized":    {"color": "#2ca02c", "marker": "^", "label": "Centralized R=2"},
    "centralized_R1": {"color": "#98df8a", "marker": "v", "label": "Centralized R=1"},
    "decentralized":  {"color": "#d62728", "marker": "D", "label": "Decentralized R=2"},
    "hybrid":         {"color": "#9467bd", "marker": "P", "label": "Hybrid R=2"},
    "hybrid_R1":      {"color": "#c5b0d5", "marker": "X", "label": "Hybrid R=1"},
}
BENCH_ORDER = ["qampari", "fanoutqa", "browsecomp_plus", "workbench", "swebench"]
BENCH_LABEL = {
    "qampari": "QAMPARI", "fanoutqa": "FanOutQA",
    "browsecomp_plus": "BrowseComp+", "workbench": "WorkBench",
    "swebench": "SWE-bench",
}

CONFIG_RE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        m = CONFIG_RE.match(os.path.basename(path))
        if not m:
            continue
        topo, k, r = m.group(1), int(m.group(2)), m.group(3)
        config_key = f"{topo}_R1" if r == "1" else topo
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("error"):
                continue
            e = d.get("gpu_dynamic_energy_joules", 0) or 0
            P = d.get("total_prompt_tokens", 0) or 0
            C = d.get("total_completion_tokens", 0) or 0
            pred = -84 + 0.018 * P + 5.54 * C
            if pred > 0 and e / pred < 0.1:
                continue  # filter known-bad
            acc = (
                float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                else (1.0 if d.get("correct") else 0.0)
            )
            rows.append(dict(
                benchmark=d.get("benchmark", "?"),
                config_key=config_key, k=k,
                task_id=d.get("task_id"), rep=d.get("rep", 0),
                acc=acc, energy_J=e,
                tokens=int(P) + int(C),
            ))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["benchmark", "config_key", "k", "task_id", "rep"]).copy()
    df = df.groupby(["benchmark", "config_key", "k"]).head(50)
    return (
        df.groupby(["benchmark", "config_key", "k"], as_index=False)
          .agg(n=("task_id", "count"),
               acc_pct=("acc", lambda s: 100 * s.mean()),
               energy_kJ=("energy_J", lambda s: s.mean() / 1000),
               tokens=("tokens", "mean"))
    )


def pareto_mask(xy: np.ndarray) -> np.ndarray:
    """Min x, max y."""
    n = len(xy)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (xy[j, 0] <= xy[i, 0] and xy[j, 1] >= xy[i, 1]
                    and (xy[j, 0] < xy[i, 0] or xy[j, 1] > xy[i, 1])):
                keep[i] = False
                break
    return keep


def plot(agg: pd.DataFrame, out: str) -> None:
    benches = [b for b in BENCH_ORDER if b in agg["benchmark"].unique()]
    n = len(benches)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.6), squeeze=False)

    for col, bench in enumerate(benches):
        sub = agg[agg["benchmark"] == bench].copy()
        if sub.empty:
            for r in (0, 1):
                axes[r, col].set_visible(False)
            continue

        # Token frontier
        xy_tok = sub[["tokens", "acc_pct"]].to_numpy(dtype=float)
        tok_front = pareto_mask(xy_tok)
        # Energy frontier
        xy_eng = sub[["energy_kJ", "acc_pct"]].to_numpy(dtype=float)
        eng_front = pareto_mask(xy_eng)
        # "Frontier shift": configs on one frontier but not the other
        shift = (tok_front != eng_front)

        for row_idx, (xcol, xlabel, frontier_mask) in enumerate([
            ("tokens", "Total tokens per task", tok_front),
            ("energy_kJ", "GPU dynamic energy per task (kJ)", eng_front),
        ]):
            ax = axes[row_idx, col]
            for ck in sub["config_key"].unique():
                style = TOPO_STYLE.get(ck, {"color": "gray", "marker": "x", "label": ck})
                pts = sub[sub["config_key"] == ck]
                ax.scatter(pts[xcol], pts["acc_pct"],
                           color=style["color"], marker=style["marker"],
                           s=70, alpha=0.85, edgecolor="black", linewidth=0.4,
                           label=style["label"])
                if ck == "sas":
                    for _, rr in pts.iterrows():
                        ax.annotate(f"k={int(rr['k'])}", (rr[xcol], rr["acc_pct"]),
                                    xytext=(4, 4), textcoords="offset points",
                                    fontsize=7, color=style["color"])
            # Frontier line
            front = sub[frontier_mask].sort_values(xcol)
            ax.plot(front[xcol], front["acc_pct"],
                    color="black", linestyle="--", linewidth=1.2, zorder=0)
            # Highlight shifted points
            shifted = sub[shift]
            ax.scatter(shifted[xcol], shifted["acc_pct"],
                       facecolors="none", edgecolors="red", s=240,
                       linewidth=1.6, zorder=5)

            ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=9)
            if col == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=9)
            if row_idx == 0:
                ax.set_title(BENCH_LABEL.get(bench, bench), fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    # One legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(*zip(*uniq), loc="lower center", ncol=min(len(uniq), 7),
               bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=True)

    fig.suptitle(
        "A5000 Pareto Frontiers — Token cost (top row) vs Energy cost (bottom row)\n"
        "Red circles = configurations whose Pareto-frontier status differs between token and energy axes",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    df = load()
    print(f"Loaded {len(df)} task-level records")
    agg = aggregate(df)
    agg.to_csv(os.path.join(OUT_DIR, "pareto_dual.csv"), index=False)
    plot(agg, os.path.join(OUT_DIR, "pareto_dual.png"))


if __name__ == "__main__":
    main()
