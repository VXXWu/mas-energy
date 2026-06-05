"""Pareto frontier + energy amplification analysis on the a5000 dataset.

Inputs:  mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl
Outputs: analysis/a5000_figs/
  - pareto_<benchmark>.png       (energy vs accuracy, points colored by topology)
  - pareto_grid.png              (all benchmarks faceted)
  - amplification_grid.png       (energy / SAS_energy at matched k, by topology × benchmark)
  - summary_table.csv            (per-config means: acc, energy, wall, calls, tokens)

Run from repo root:
    python analysis/a5000_pareto_amplification.py
"""
from __future__ import annotations
import os
import json
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"
OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Color/marker for topologies
TOPO_STYLE = {
    "sas":            {"color": "#1f77b4", "marker": "o", "label": "SAS"},
    "independent":    {"color": "#ff7f0e", "marker": "s", "label": "Independent"},
    "centralized":    {"color": "#2ca02c", "marker": "^", "label": "Centralized R=2"},
    "centralized_R1": {"color": "#98df8a", "marker": "v", "label": "Centralized R=1"},
    "decentralized":  {"color": "#d62728", "marker": "D", "label": "Decentralized R=2"},
    "hybrid":         {"color": "#9467bd", "marker": "P", "label": "Hybrid R=2"},
    "hybrid_R1":      {"color": "#c5b0d5", "marker": "X", "label": "Hybrid R=1"},
}

# Benchmark display order
BENCH_ORDER = ["qampari", "fanoutqa", "browsecomp_plus", "workbench", "swebench"]
BENCH_LABEL = {
    "qampari": "QAMPARI",
    "fanoutqa": "FanOutQA",
    "browsecomp_plus": "BrowseComp+",
    "workbench": "WorkBench",
    "swebench": "SWE-bench",
}

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

CONFIG_RE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")


def load_records() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        fname = os.path.basename(path)
        m = CONFIG_RE.match(fname)
        if not m:
            continue
        topo, k, r = m.group(1), int(m.group(2)), m.group(3)
        rounds_override = int(r) if r else None
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("error"):
                continue
            if "gpu_dynamic_energy_joules" not in d:
                continue
            # Filter records with implausibly low energy (broken measurement,
            # legacy records from early runs). Whole-task energy must be >100J;
            # any real run is at least one inference call ≈ 200-500J minimum.
            if float(d.get("gpu_dynamic_energy_joules", 0.0)) < 100.0:
                continue
            rows.append(
                dict(
                    benchmark=d.get("benchmark", "?"),
                    topology=topo,
                    rounds_override=rounds_override,
                    k=k,
                    config_key=f"{topo}{'_R1' if rounds_override == 1 else ''}",
                    task_id=d.get("task_id", "?"),
                    rep=d.get("rep", 0),
                    accuracy=float(
                        d["loose_accuracy"]
                        if d.get("loose_accuracy") is not None
                        else (1.0 if d.get("correct") else 0.0)
                    ),
                    energy_J=float(d.get("gpu_dynamic_energy_joules", 0.0)),
                    wall_s=float(d.get("total_wall_seconds", 0.0)),
                    n_llm_calls=int(d.get("n_llm_calls", 0) or 0),
                    P=int(d.get("total_prompt_tokens", 0) or 0),
                    C=int(d.get("total_completion_tokens", 0) or 0),
                )
            )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, config_key, k) means, capped at first 50 records per config to harmonize."""
    df = df.copy()
    df = df.sort_values(["benchmark", "config_key", "k", "task_id", "rep"])
    df = df.groupby(["benchmark", "config_key", "k"]).head(50)
    agg = (
        df.groupby(["benchmark", "config_key", "k"], as_index=False)
          .agg(
              n=("task_id", "count"),
              acc=("accuracy", "mean"),
              acc_sd=("accuracy", "std"),
              energy_J=("energy_J", "mean"),
              wall_s=("wall_s", "mean"),
              calls=("n_llm_calls", "mean"),
              P=("P", "mean"),
              C=("C", "mean"),
          )
    )
    agg["acc_pct"] = agg["acc"] * 100
    agg["energy_kJ"] = agg["energy_J"] / 1000
    return agg


# ----------------------------------------------------------------------
# Pareto helpers
# ----------------------------------------------------------------------


def pareto_front(points_xy: np.ndarray) -> np.ndarray:
    """Return mask of Pareto-optimal points minimizing x and maximizing y."""
    n = len(points_xy)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower x and higher y (or equal with strict in one)
            if (points_xy[j, 0] <= points_xy[i, 0] and points_xy[j, 1] >= points_xy[i, 1]
                    and (points_xy[j, 0] < points_xy[i, 0] or points_xy[j, 1] > points_xy[i, 1])):
                is_pareto[i] = False
                break
    return is_pareto


# ----------------------------------------------------------------------
# Figure: Pareto grid
# ----------------------------------------------------------------------


def plot_pareto_grid(agg: pd.DataFrame, out_path: str) -> None:
    benches = [b for b in BENCH_ORDER if b in agg["benchmark"].unique()]
    n = len(benches)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, bench in zip(axes, benches):
        sub = agg[agg["benchmark"] == bench].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        # Plot every (config_key, k) point
        for config_key in sub["config_key"].unique():
            style = TOPO_STYLE.get(config_key, {"color": "gray", "marker": "x", "label": config_key})
            pts = sub[sub["config_key"] == config_key]
            ax.scatter(
                pts["energy_kJ"], pts["acc_pct"],
                color=style["color"], marker=style["marker"], s=60, alpha=0.85,
                label=style["label"], edgecolor="black", linewidth=0.4,
            )
            # k labels (only for SAS to avoid clutter; others have many)
            if config_key == "sas":
                for _, r in pts.iterrows():
                    ax.annotate(f"k={int(r['k'])}", (r["energy_kJ"], r["acc_pct"]),
                                xytext=(4, 4), textcoords="offset points", fontsize=7, color="#1f77b4")

        # Pareto frontier (minimize energy, maximize acc)
        xy = sub[["energy_kJ", "acc_pct"]].to_numpy()
        mask = pareto_front(xy)
        front = sub[mask].sort_values("energy_kJ")
        ax.plot(front["energy_kJ"], front["acc_pct"],
                color="black", linestyle="--", linewidth=1.2, zorder=0, label="Pareto frontier")

        ax.set_xscale("log")
        ax.set_xlabel("GPU dynamic energy per task (kJ, log)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(BENCH_LABEL.get(bench, bench))
        ax.grid(True, alpha=0.3)

    for ax in axes[len(benches):]:
        ax.set_visible(False)

    # One legend at the figure level
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    fig.legend(*zip(*uniq), loc="lower center", ncol=min(len(uniq), 4),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True)
    fig.suptitle("Energy-vs-accuracy Pareto frontiers — A5000, Qwen3.5-9B", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# Figure: amplification grid (energy ratio vs SAS at matched k)
# ----------------------------------------------------------------------


def amplification_table(agg: pd.DataFrame) -> pd.DataFrame:
    """For each (benchmark, k), compute energy_topo / energy_SAS."""
    out = []
    for (bench, k), g in agg.groupby(["benchmark", "k"]):
        sas = g[g["config_key"] == "sas"]
        if sas.empty:
            continue
        sas_E = float(sas["energy_J"].iloc[0])
        sas_acc = float(sas["acc_pct"].iloc[0])
        for _, r in g.iterrows():
            if r["config_key"] == "sas":
                continue
            out.append(
                dict(
                    benchmark=bench,
                    k=int(k),
                    config_key=r["config_key"],
                    energy_ratio=r["energy_J"] / sas_E if sas_E > 0 else np.nan,
                    acc_delta_pct=r["acc_pct"] - sas_acc,
                    energy_kJ=r["energy_kJ"],
                    sas_energy_kJ=sas_E / 1000,
                )
            )
    return pd.DataFrame(out)


def plot_amplification_grid(amp: pd.DataFrame, out_path: str) -> None:
    benches = [b for b in BENCH_ORDER if b in amp["benchmark"].unique()]
    n = len(benches)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    config_order = ["independent", "centralized", "centralized_R1", "decentralized", "hybrid", "hybrid_R1"]

    for ax, bench in zip(axes, benches):
        sub = amp[amp["benchmark"] == bench].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        # Mean ratio per config (averaged across k values)
        means = (
            sub.groupby("config_key")["energy_ratio"].agg(["mean", "std", "count"]).reset_index()
        )
        means = means[means["config_key"].isin(config_order)]
        means["order"] = means["config_key"].map(lambda c: config_order.index(c) if c in config_order else 99)
        means = means.sort_values("order")
        if means.empty:
            ax.set_visible(False)
            continue
        colors = [TOPO_STYLE.get(c, {"color": "gray"})["color"] for c in means["config_key"]]
        labels = [TOPO_STYLE.get(c, {"label": c})["label"] for c in means["config_key"]]
        ax.bar(range(len(means)), means["mean"], color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(range(len(means)), means["mean"],
                    yerr=means["std"].fillna(0),
                    fmt="none", color="black", capsize=3, linewidth=0.8)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(BENCH_LABEL.get(bench, bench))
        ax.set_ylabel("Energy ratio vs SAS")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[len(benches):]:
        ax.set_visible(False)
    fig.suptitle("Energy amplification: MAS energy / SAS energy (matched k, mean across k)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:
    print("Loading records...")
    df = load_records()
    print(f"  {len(df)} task-level records, {df['benchmark'].nunique()} benchmarks, "
          f"{df['config_key'].nunique()} configs")

    agg = aggregate(df)
    agg.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    print(f"  wrote summary_table.csv ({len(agg)} (bench, config, k) cells)")

    print("\n=== Per-benchmark config counts ===")
    print(agg.groupby("benchmark")["config_key"].nunique().to_string())

    print("\nPareto grid...")
    plot_pareto_grid(agg, os.path.join(OUT_DIR, "pareto_grid.png"))

    print("Amplification table...")
    amp = amplification_table(agg)
    amp.to_csv(os.path.join(OUT_DIR, "amplification_table.csv"), index=False)
    print(amp.groupby(["benchmark", "config_key"])["energy_ratio"].mean().round(2).to_string())

    print("\nAmplification grid...")
    plot_amplification_grid(amp, os.path.join(OUT_DIR, "amplification_grid.png"))

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
