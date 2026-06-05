"""Two clean figures for the paper:

  Figure 1 (asymmetry):
    Left  — per-task P_total distribution grouped by benchmark (color = topology)
    Right — per-task C_total distribution grouped by topology (color = benchmark)
    Goal: show visually that prompt tokens cluster by benchmark while
          completion tokens cluster by topology.

  Figure 2 (compound cost, clean):
    Per-call C distribution as boxplots, binned by per-call P range.
    Pooled across topologies. Shows C stepping up monotonically with P,
    cleanly demonstrating the verbosity-induced-by-context effect.

Outputs:
    analysis/a5000_figs/asymmetry_P_vs_C.png
    analysis/a5000_figs/compound_cost_boxplot.png
"""
from __future__ import annotations
import os
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

BENCH_ORDER = ["qampari", "fanoutqa", "browsecomp_plus", "workbench", "swebench"]
BENCH_LABEL = {
    "qampari": "QAMPARI", "fanoutqa": "FanOutQA",
    "browsecomp_plus": "BrowseComp+", "workbench": "WorkBench",
    "swebench": "SWE-bench",
}

TOPO_ORDER = ["sas", "independent", "centralized_R1", "centralized",
              "decentralized", "hybrid_R1", "hybrid"]
TOPO_LABEL = {
    "sas": "SAS",
    "independent": "Independent",
    "centralized_R1": "Centralized R=1",
    "centralized": "Centralized R=2",
    "decentralized": "Decentralized",
    "hybrid_R1": "Hybrid R=1",
    "hybrid": "Hybrid R=2",
}
TOPO_COLOR = {
    "sas":            "#1f77b4",
    "independent":    "#ff7f0e",
    "centralized_R1": "#98df8a",
    "centralized":    "#2ca02c",
    "decentralized":  "#d62728",
    "hybrid_R1":      "#c5b0d5",
    "hybrid":         "#9467bd",
}
BENCH_COLOR = {
    "qampari":         "#1f77b4",
    "fanoutqa":        "#ff7f0e",
    "browsecomp_plus": "#2ca02c",
    "workbench":       "#d62728",
    "swebench":        "#9467bd",
}


def load_task_records() -> pd.DataFrame:
    rows = []
    cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        m = cre.match(os.path.basename(path))
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
                continue
            rows.append(dict(
                benchmark=d.get("benchmark", "?"),
                topology=topo,
                config_key=config_key,
                k=k,
                P=int(P), C=int(C),
            ))
    return pd.DataFrame(rows)


def load_call_records() -> pd.DataFrame:
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
                if e <= 0:
                    continue
                rows.append(dict(
                    benchmark=rec.get("benchmark", "?"),
                    topology=rec.get("topology", "?"),
                    P=int(c["prompt_tokens"]),
                    C=int(c["completion_tokens"]),
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figure 1: asymmetry — P clusters by benchmark, C clusters by topology
# ----------------------------------------------------------------------

def figure_asymmetry(df: pd.DataFrame, out: str) -> None:
    """Two-panel: pooled boxplots showing P clusters by benchmark, C clusters by topology.
    Plus a bottom panel with a variance-decomposition bar chart making the asymmetry
    quantitative."""
    benches = [b for b in BENCH_ORDER if b in df["benchmark"].unique()]
    configs = [c for c in TOPO_ORDER if c in df["config_key"].unique()]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0])

    # ---- Panel A: P pooled by benchmark ----
    ax = fig.add_subplot(gs[0, 0])
    data_P = [df[df["benchmark"] == b]["P"].to_numpy() for b in benches]
    bp = ax.boxplot(data_P, positions=range(len(benches)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, b in zip(bp["boxes"], benches):
        patch.set_facecolor(BENCH_COLOR[b])
        patch.set_alpha(0.75)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([BENCH_LABEL[b] for b in benches], fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("Prompt tokens per task (log)")
    ax.set_title("Prompt cost (P) — pooled across topologies")
    ax.grid(True, axis="y", alpha=0.3)

    # ---- Panel B: C pooled by topology ----
    ax = fig.add_subplot(gs[0, 1])
    data_C = [df[df["config_key"] == c]["C"].to_numpy() for c in configs]
    bp = ax.boxplot(data_C, positions=range(len(configs)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], configs):
        patch.set_facecolor(TOPO_COLOR[c])
        patch.set_alpha(0.75)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([TOPO_LABEL[c] for c in configs], rotation=20, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Completion tokens per task (log)")
    ax.set_title("Completion cost (C) — pooled across benchmarks")
    ax.grid(True, axis="y", alpha=0.3)

    # ---- Bottom panel: variance decomposition ----
    # Two-way ANOVA-style: how much of the variance in log(P) and log(C) is
    # explained by benchmark vs topology vs interaction vs residual?
    def variance_breakdown(series_log, group_b, group_t):
        total_var = series_log.var()
        b_means = series_log.groupby(group_b).transform("mean")
        t_means = series_log.groupby(group_t).transform("mean")
        grand_mean = series_log.mean()
        ss_b = ((b_means - grand_mean) ** 2).mean()
        ss_t = ((t_means - grand_mean) ** 2).mean()
        ss_total = ((series_log - grand_mean) ** 2).mean()
        return ss_b / ss_total, ss_t / ss_total

    df = df.copy()
    df["log_P"] = np.log1p(df["P"])
    df["log_C"] = np.log1p(df["C"])
    p_by_bench, p_by_topo = variance_breakdown(df["log_P"], df["benchmark"], df["config_key"])
    c_by_bench, c_by_topo = variance_breakdown(df["log_C"], df["benchmark"], df["config_key"])

    ax = fig.add_subplot(gs[1, :])
    x = np.arange(2)
    w = 0.35
    bars1 = ax.bar(x - w/2, [p_by_bench, c_by_bench], w,
                   label="Variance explained by BENCHMARK", color="#2ca02c", edgecolor="black")
    bars2 = ax.bar(x + w/2, [p_by_topo, c_by_topo], w,
                   label="Variance explained by TOPOLOGY", color="#d62728", edgecolor="black")
    for bars, vals in [(bars1, [p_by_bench, c_by_bench]),
                       (bars2, [p_by_topo, c_by_topo])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["log(P) — prompt tokens", "log(C) — completion tokens"], fontsize=11)
    ax.set_ylabel("Fraction of variance explained")
    ax.set_ylim(0, max(p_by_bench, p_by_topo, c_by_bench, c_by_topo) * 1.25)
    ax.set_title("Variance decomposition: which factor drives each cost component?")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotation
    ax.text(0.5, -0.2,
            f"Prompt P:  benchmark explains {p_by_bench:.0%}, topology explains {p_by_topo:.0%}  →  "
            f"P is {p_by_bench / max(p_by_topo, 0.001):.1f}× more benchmark-driven\n"
            f"Completion C:  benchmark explains {c_by_bench:.0%}, topology explains {c_by_topo:.0%}  →  "
            f"C is {c_by_topo / max(c_by_bench, 0.001):.1f}× more topology-driven",
            ha="center", va="top", fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="gray"))

    fig.suptitle("P (prompt) and C (completion) have different drivers", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")
    print(f"  variance-explained: log(P) bench={p_by_bench:.2%}/topo={p_by_topo:.2%}   log(C) bench={c_by_bench:.2%}/topo={c_by_topo:.2%}")


# ----------------------------------------------------------------------
# Figure 2: compound cost as a clean boxplot
# ----------------------------------------------------------------------

def figure_compound_boxplot(call_df: pd.DataFrame, out: str) -> None:
    """Per-topology faceted boxplots: C distribution per P bin, shown for each topology
    separately. Pooling across topologies washes out the trend because role types have
    different P/C joint distributions; faceting recovers the within-topology elasticity."""
    bins = [0, 500, 2000, 5000, 15000, 50000, 200000]
    bin_labels = ["<0.5k", "0.5–2k", "2–5k", "5–15k", "15–50k", "50k+"]

    topos = ["sas", "independent", "centralized", "decentralized", "hybrid"]
    topo_label = {"sas": "SAS", "independent": "Independent", "centralized": "Centralized",
                  "decentralized": "Decentralized", "hybrid": "Hybrid"}

    fig, axes = plt.subplots(1, len(topos), figsize=(4 * len(topos), 5), sharey=True)

    for ax, topo in zip(axes, topos):
        sub = call_df[call_df["topology"] == topo].copy()
        if len(sub) < 100:
            ax.set_visible(False)
            continue
        sub["P_bin"] = pd.cut(sub["P"], bins=bins, labels=bin_labels, right=False)
        grouped = sub.groupby("P_bin", observed=True)["C"]
        data = [grouped.get_group(b).to_numpy() for b in bin_labels if b in grouped.groups]
        used = [b for b in bin_labels if b in grouped.groups]
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            data, positions=range(len(used)), widths=0.6,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=TOPO_COLOR.get(topo, "gray"), alpha=0.65),
            medianprops=dict(color="black", linewidth=1.4),
        )
        medians = [np.median(d) for d in data]
        ax.plot(range(len(used)), medians, color="#d62728", marker="o",
                lw=1.8, markersize=6, zorder=5)

        # Fit log-log slope of bin medians
        bin_centers = []
        for b in used:
            idx = bin_labels.index(b)
            lo, hi = bins[idx], bins[idx + 1]
            bin_centers.append(np.sqrt(max(lo, 1) * hi))
        lp = np.log(np.array(bin_centers))
        lc = np.log(np.array(medians) + 1)
        if len(lp) >= 3:
            slope, intercept = np.polyfit(lp, lc, 1)
        else:
            slope = float("nan")

        ax.set_xticks(range(len(used)))
        ax.set_xticklabels(used, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{topo_label[topo]}\nβ = {slope:.2f}", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Completion tokens per call (log)")
        ax.set_xlabel("Prompt tokens per call (binned)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Compound cost: completion length increases with prompt length within each topology\n"
        "(per-call boxplots, faceted by topology — β is log-log slope of bin medians)",
        y=1.03, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    print("Loading task records...")
    task_df = load_task_records()
    print(f"  {len(task_df)} tasks across {task_df['benchmark'].nunique()} benchmarks × {task_df['config_key'].nunique()} topologies")

    print("Generating Figure 1 (asymmetry)...")
    figure_asymmetry(task_df, os.path.join(OUT_DIR, "asymmetry_P_vs_C.png"))

    print("Loading per-call records...")
    call_df = load_call_records()
    print(f"  {len(call_df):,} per-call records")

    print("Generating Figure 2 (compound cost boxplot)...")
    figure_compound_boxplot(call_df, os.path.join(OUT_DIR, "compound_cost_boxplot.png"))

    print("Done.")


if __name__ == "__main__":
    main()
