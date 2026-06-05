"""Generate all scaling figures (k-sweep + R-sweep + M-sweep) for complete benchmarks.

All three sweeps share the anchor point (k=10, R=2, M=3):
  - k-sweep:  varies k at (R=2, M=3)  for centralized/decentralized (defaults for SAS/Indep)
  - R-sweep:  varies R at (k=10, M=3) for centralized/decentralized
  - M-sweep:  varies M at (k=10, R=2) for centralized/decentralized

Default: linear scaling on accuracy/wall axes, log on energy axis.

Usage:
    python analysis/generate_scaling_figures.py
"""
import json, glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accuracy_helper import record_accuracy

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
DEFAULT_R = {"centralized": 5, "decentralized": 2}
DEFAULT_M = 3  # config default agent count
# For k-sweep, which R to prefer per topology (centralized now standardized to R=2)
KSWEEP_R = {"sas": None, "independent": None, "centralized": 2, "decentralized": 2}
RSWEEP_K = 10  # all benchmarks: R-sweep done at k=10
MSWEEP_K = 10  # M-sweep done at k=10
MSWEEP_R = 2   # M-sweep done at R=2

TOPO_STYLE = {
    "sas":            {"color": "#1f77b4", "marker": "o", "label": "SAS"},
    "independent":    {"color": "#ff7f0e", "marker": "s", "label": "Independent (M=3)"},
    "centralized":    {"color": "#2ca02c", "marker": "^", "label": "Centralized (M=3)"},
    "decentralized":  {"color": "#d62728", "marker": "D", "label": "Decentralized (M=3)"},
}

BENCH = {
    "qampari":   ("a5000_qampari_v4",   "QAMPARI"),
    "workbench": ("a5000_workbench_v2",  "WorkBench"),
    "fanoutqa":  ("a5000_fanoutqa_v4",   "FanOutQA"),
    "browsecomp":("a5000_browsecomp_pilot","BrowseComp+"),
    "swebench":  ("a5000_swebench",      "SWE-bench"),
    "math":      ("a5000_math_pilot",    "MATH (Level 5)"),
}


def load_data(bench_dir):
    cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")
    rows = []
    for f in sorted(glob.glob(f"mas-energy/results/{bench_dir}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = cre.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo == "hybrid": continue
        R_actual = int(r_str) if r_str else DEFAULT_R.get(topo, 2)
        M_actual = int(m_str) if m_str else DEFAULT_M
        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            e = d.get("gpu_dynamic_energy_joules", 0) or 0
            P = d.get("total_prompt_tokens", 0) or 0
            C = d.get("total_completion_tokens", 0) or 0
            pred = -84 + 0.018 * P + 5.54 * C
            if pred > 0 and e / pred < 0.1: continue
            acc, ok = record_accuracy(d)
            if not ok: continue
            wall = d.get("total_wall_seconds", 0) or 0
            rows.append(dict(topology=topo, k=k, R=R_actual, R_explicit=r_str,
                             M=M_actual, M_explicit=m_str,
                             acc=acc, energy_J=e, wall_s=wall))
    return pd.DataFrame(rows)


def aggregate(df, cols):
    return (df.groupby(cols, as_index=False)
              .agg(n=("acc", "count"),
                   acc_pct=("acc", lambda s: 100 * s.mean()),
                   acc_se=("acc", lambda s: 100 * s.std() / max(1, len(s)**0.5)),
                   energy_kJ=("energy_J", lambda s: s.mean() / 1000),
                   energy_se=("energy_J", lambda s: s.std() / 1000 / max(1, len(s)**0.5)),
                   wall_s=("wall_s", "mean"),
                   wall_se=("wall_s", lambda s: s.std() / max(1, len(s)**0.5))))


def annotate_frontier(ax, sub, col, color):
    for _, r in sub.iterrows():
        ax.annotate(f"{int(r[col])}", (r["energy_kJ"], r["acc_pct"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def plot_kscan(bench_key, bench_label, agg):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # (a) Accuracy vs k
    ax = axes[0]
    for topo in ["sas", "independent", "centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg[agg["topology"] == topo].sort_values("k")
        if sub.empty: continue
        ax.errorbar(sub["k"], sub["acc_pct"], yerr=sub["acc_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2, markersize=8, capsize=3)
    ax.set_xlabel("k (max react steps per agent)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(a) Accuracy vs k", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (b) Energy vs k
    ax = axes[1]
    fits = []
    for topo in ["sas", "independent", "centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg[agg["topology"] == topo].sort_values("k")
        if sub.empty: continue
        # Always render the points so single-k pilots still show topology spread
        ax.errorbar(sub["k"], sub["energy_kJ"], yerr=sub["energy_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2, markersize=8, capsize=3)
        # Fit the power law only when there are enough points
        if len(sub) >= 3:
            lk = np.log(sub["k"].to_numpy(float))
            le = np.log(sub["energy_kJ"].to_numpy(float).clip(1e-6))
            sl, ic, r, p, se = stats.linregress(lk, le)
            fits.append((topo, sl, r**2))
            ks_fit = np.linspace(sub["k"].min(), sub["k"].max(), 100)
            ax.plot(ks_fit, np.exp(ic) * ks_fit**sl,
                    color=s["color"], linestyle="--", lw=1.2, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("k (max react steps per agent)", fontsize=11)
    ax.set_ylabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_title("(b) Energy vs k — E~k^β", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    for t, sl, r2 in fits:
        idx = next(i for i, l in enumerate(labels) if t in l.lower())
        labels[idx] += f"  β={sl:.2f}"
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Wall-clock time vs k
    ax = axes[2]
    for topo in ["sas", "independent", "centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg[agg["topology"] == topo].sort_values("k")
        if sub.empty: continue
        ax.errorbar(sub["k"], sub["wall_s"] / 60, yerr=sub["wall_se"] / 60 * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2, markersize=8, capsize=3)
    ax.set_xlabel("k (max react steps per agent)", fontsize=11)
    ax.set_ylabel("Wall-clock time per task (min)", fontsize=11)
    ax.set_title("(c) Wall time vs k (serial)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (d) Accuracy vs Energy frontier
    ax = axes[3]
    for topo in ["sas", "independent", "centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg[agg["topology"] == topo].sort_values("energy_kJ")
        if sub.empty: continue
        ax.plot(sub["energy_kJ"], sub["acc_pct"],
                color=s["color"], marker=s["marker"], label=s["label"],
                lw=2, markersize=8)
        annotate_frontier(ax, sub, "k", s["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(d) Accuracy-Energy frontier (labels=k)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{bench_label} — k-scaling (M=3, R=2 for centralized/decentralized)",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/{bench_key}_kscan_scaling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return fits


def plot_rscan(bench_key, bench_label, agg_r):
    if agg_r.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # (a) Accuracy vs R
    ax = axes[0]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_r[agg_r["topology"] == topo].sort_values("R")
        if sub.empty: continue
        ax.errorbar(sub["R"], sub["acc_pct"], yerr=sub["acc_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
        for _, r in sub.iterrows():
            ax.annotate(f"R={int(r['R'])}", (r["R"], r["acc_pct"]),
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=s["color"], ha="center")
    ax.set_xlabel("R (debate / orchestration rounds)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"(a) Accuracy vs R (k={RSWEEP_K}, M=3)", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Energy vs R
    ax = axes[1]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_r[agg_r["topology"] == topo].sort_values("R")
        if len(sub) < 3: continue
        ax.errorbar(sub["R"], sub["energy_kJ"], yerr=sub["energy_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
        lr = np.log(sub["R"].to_numpy(float))
        le = np.log(sub["energy_kJ"].to_numpy(float).clip(1e-6))
        sl, ic, r, p, se = stats.linregress(lr, le)
        rs_fit = np.linspace(sub["R"].min(), sub["R"].max(), 30)
        ax.plot(rs_fit, np.exp(ic) * rs_fit**sl,
                color=s["color"], linestyle="--", lw=1.5, alpha=0.6,
                label=f"  β={sl:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("R (rounds)", fontsize=11)
    ax.set_ylabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_title("(b) Energy vs R", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Wall-clock time vs R
    ax = axes[2]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_r[agg_r["topology"] == topo].sort_values("R")
        if sub.empty: continue
        ax.errorbar(sub["R"], sub["wall_s"] / 60, yerr=sub["wall_se"] / 60 * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
    ax.set_xlabel("R (rounds)", fontsize=11)
    ax.set_ylabel("Wall-clock time per task (min)", fontsize=11)
    ax.set_title("(c) Wall time vs R (serial)", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (d) Accuracy vs Energy frontier
    ax = axes[3]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_r[agg_r["topology"] == topo].sort_values("energy_kJ")
        if sub.empty: continue
        ax.plot(sub["energy_kJ"], sub["acc_pct"],
                color=s["color"], marker=s["marker"], label=s["label"],
                lw=2.5, markersize=10)
        annotate_frontier(ax, sub, "R", s["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(d) Accuracy-Energy frontier (labels=R)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{bench_label} — R-scaling: Centralized & Decentralized (k={RSWEEP_K}, M=3)",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/{bench_key}_rscan_scaling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_mscan(bench_key, bench_label, agg_m):
    if agg_m.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    M_max = float(agg_m["M"].max())
    use_log_x = M_max > 8  # wide-range benchmarks (browsecomp, swebench) → log x

    # (a) Accuracy vs M
    ax = axes[0]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_m[agg_m["topology"] == topo].sort_values("M")
        if sub.empty: continue
        ax.errorbar(sub["M"], sub["acc_pct"], yerr=sub["acc_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
    if use_log_x: ax.set_xscale("log")
    ax.set_xlabel("M (agent count)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"(a) Accuracy vs M (k={MSWEEP_K}, R={MSWEEP_R})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Energy vs M — power-law fit
    ax = axes[1]
    fits = []
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_m[agg_m["topology"] == topo].sort_values("M")
        if sub.empty: continue
        ax.errorbar(sub["M"], sub["energy_kJ"], yerr=sub["energy_se"] * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
        if len(sub) >= 3:
            lm = np.log(sub["M"].to_numpy(float))
            le = np.log(sub["energy_kJ"].to_numpy(float).clip(1e-6))
            sl, ic, r, p, se = stats.linregress(lm, le)
            fits.append((topo, sl, r**2))
            ms_fit = np.linspace(sub["M"].min(), sub["M"].max(), 50)
            ax.plot(ms_fit, np.exp(ic) * ms_fit**sl,
                    color=s["color"], linestyle="--", lw=1.2, alpha=0.6)
    ax.set_yscale("log")
    if use_log_x: ax.set_xscale("log")
    ax.set_xlabel("M (agent count)", fontsize=11)
    ax.set_ylabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_title("(b) Energy vs M — E~M^β", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    for t, sl, r2 in fits:
        idx = next((i for i, l in enumerate(labels) if t in l.lower()), None)
        if idx is not None:
            labels[idx] += f"  β={sl:.2f}"
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Wall-clock time vs M
    ax = axes[2]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_m[agg_m["topology"] == topo].sort_values("M")
        if sub.empty: continue
        ax.errorbar(sub["M"], sub["wall_s"] / 60, yerr=sub["wall_se"] / 60 * 1.96,
                    color=s["color"], marker=s["marker"], label=s["label"],
                    lw=2.5, markersize=10, capsize=4)
    if use_log_x: ax.set_xscale("log")
    ax.set_xlabel("M (agent count)", fontsize=11)
    ax.set_ylabel("Wall-clock time per task (min)", fontsize=11)
    ax.set_title("(c) Wall time vs M (serial)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (d) Accuracy-Energy frontier (labels = M)
    ax = axes[3]
    for topo in ["centralized", "decentralized"]:
        s = TOPO_STYLE[topo]
        sub = agg_m[agg_m["topology"] == topo].sort_values("energy_kJ")
        if sub.empty: continue
        ax.plot(sub["energy_kJ"], sub["acc_pct"],
                color=s["color"], marker=s["marker"], label=s["label"],
                lw=2.5, markersize=10)
        annotate_frontier(ax, sub, "M", s["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(d) Accuracy-Energy frontier (labels=M)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{bench_label} — M-scaling: Centralized & Decentralized (k={MSWEEP_K}, R={MSWEEP_R})",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/{bench_key}_mscan_scaling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return fits


MAIN_4 = ["fanoutqa", "workbench", "browsecomp", "swebench"]


def plot_combined_4bench_pareto(per_bench_aggs):
    """4 rows (benchmarks) x 3 cols (k/R/M-scan) grid.

    Same structure as plot_combined_4bench but axes swapped:
      x = energy per task (kJ, linear)
      y = accuracy (%)
    Each panel shows the (energy, accuracy) trajectory each topology traces
    as the swept axis (k, R, or M) grows. Points annotated with axis value
    at the endpoints; lines connect consecutive sweep values per topology.
    Effectively shows the per-axis "Pareto-like" curves for each benchmark.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharey=False, sharex=False)
    col_titles = ["k-scan (R=2, M=3)",
                  f"R-scan (k={RSWEEP_K}, M=3)",
                  f"M-scan (k={MSWEEP_K}, R={MSWEEP_R})"]
    col_xkey = ["k", "R", "M"]

    for row, bench_key in enumerate(MAIN_4):
        bench_label = BENCH[bench_key][1]
        aggs = per_bench_aggs.get(bench_key, {})
        for col, axis_key in enumerate(["k", "r", "m"]):
            ax = axes[row, col]
            agg = aggs.get(axis_key)
            if agg is None or agg.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            topos = (["sas", "independent", "centralized", "decentralized"]
                     if axis_key == "k" else ["centralized", "decentralized"])
            xk = col_xkey[col]
            # Plot each topology as a trajectory — same styling as plot_combined_4bench
            for topo in topos:
                s = TOPO_STYLE[topo]
                sub = agg[agg["topology"] == topo].sort_values(xk)
                if sub.empty: continue
                ax.errorbar(sub["energy_kJ"], sub["acc_pct"],
                            yerr=sub["acc_se"] * 1.96,
                            color=s["color"], marker=s["marker"], label=s["label"],
                            lw=2, markersize=7, capsize=3)
                # Annotate the endpoint with the sweep value
                last = sub.iloc[-1]
                ax.annotate(f"{xk}={int(last[xk])}", (last["energy_kJ"], last["acc_pct"]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=s["color"], va="center")
            ax.set_xscale("log")  # energy spans wide ranges across topologies — log x makes trajectories readable
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
                ax.legend(fontsize=8, loc="best")
            if row == 3:
                ax.set_xlabel("Energy per task (kJ, log)", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{bench_label}\nAccuracy (%)", fontsize=11, fontweight="bold")

    fig.suptitle("Per-axis (k, R, M) trajectories in (energy, accuracy) space — 4 main benchmarks "
                 "(anchor: k=10, R=2, M=3)",
                 y=1.00, fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/combined_kRM_pareto_4bench.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_combined_4bench(per_bench_aggs, metric="acc"):
    """4 rows (benchmarks) x 3 cols (k-scan, R-scan, M-scan) grid.

    metric: "acc" → accuracy (linear y) or "energy" → energy in kJ (log y).
    per_bench_aggs: dict {bench_key: {"k": agg_k, "r": agg_r, "m": agg_m}}
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharey=False)
    col_titles = ["k-scan (R=2, M=3)",
                  f"R-scan (k={RSWEEP_K}, M=3)",
                  f"M-scan (k={MSWEEP_K}, R={MSWEEP_R})"]
    col_xlabels = ["k (max react steps)", "R (rounds)", "M (agent count)"]
    col_xkey = ["k", "R", "M"]
    # Default R-axis ticks {1..5}; SWE-bench gets extended ticks {1..10} since
    # the rhighext sbatch landed R=7, R=10 cells. Other benchmarks only have R={1..5}.
    col_xticks_default = [None, [1, 2, 3, 4, 5], None]
    col_xticks_swebench = [None, [1, 2, 3, 4, 5, 7, 10], None]

    if metric == "acc":
        ycol, yerr_col, ylabel_short = "acc_pct", "acc_se", "Accuracy (%)"
        ylog = False
        suptitle = "Accuracy"
        out_name = "combined_kRM_scaling_4bench"
    elif metric == "energy":
        ycol, yerr_col, ylabel_short = "energy_kJ", "energy_se", "Energy per task (kJ)"
        ylog = False
        suptitle = "Energy"
        out_name = "combined_kRM_energy_4bench"
    else:
        raise ValueError(f"unknown metric: {metric}")

    for row, bench_key in enumerate(MAIN_4):
        bench_label = BENCH[bench_key][1]
        aggs = per_bench_aggs.get(bench_key, {})
        col_xticks = col_xticks_swebench if bench_key == "swebench" else col_xticks_default
        for col, axis_key in enumerate(["k", "r", "m"]):
            ax = axes[row, col]
            agg = aggs.get(axis_key)
            if agg is None or agg.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            topos = (["sas", "independent", "centralized", "decentralized"]
                     if axis_key == "k" else ["centralized", "decentralized"])
            xk = col_xkey[col]
            for topo in topos:
                s = TOPO_STYLE[topo]
                sub = agg[agg["topology"] == topo].sort_values(xk)
                if sub.empty: continue
                ax.errorbar(sub[xk], sub[ycol], yerr=sub[yerr_col] * 1.96,
                            color=s["color"], marker=s["marker"], label=s["label"],
                            lw=2, markersize=7, capsize=3)
            # Linear y-axis throughout — log scale produced confusing intra-decade
            # minor tick labels on the prose benchmark rows (data spans single decade).
            if col_xticks[col] is not None: ax.set_xticks(col_xticks[col])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
            if row == 3:
                ax.set_xlabel(col_xlabels[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{bench_label}\n{ylabel_short}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Combined (k, R, M) scaling — {suptitle} — 4 main benchmarks "
                 "(anchor: k=10, R=2, M=3)",
                 y=1.00, fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/{out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    per_bench_aggs = {}  # collected for combined figure
    for bench_key, (bench_dir, bench_label) in BENCH.items():
        df = load_data(bench_dir)
        if df.empty:
            print(f"  {bench_label}: no data, skipping")
            continue
        per_bench_aggs[bench_key] = {}

        # k-sweep: centralized standardized to R=2; prefer _R2 files where available,
        # exclude no-suffix centralized entirely (those are old R=5 runs).
        # Decentralized no-suffix = R=2 (default). SAS/Independent have no R.
        has_R2 = set()
        for (topo, k), g in df.groupby(["topology", "k"]):
            if (g["R_explicit"] == "2").any():
                has_R2.add((topo, k))
        def _keep_k(row):
            topo = row["topology"]
            target = KSWEEP_R.get(topo)
            if target is None:
                return pd.isna(row["R_explicit"])
            # For centralized + decentralized: prefer _R2 files when they exist
            # (disambiguates from legacy R=5 unsuffixed runs in qampari/fanoutqa).
            # Fall back to unsuffixed (which uses config-default R=2) for benchmarks
            # like MATH that never produced legacy R=5 files.
            if (topo, row["k"]) in has_R2:
                return row["R_explicit"] == "2"
            return pd.isna(row["R_explicit"])
        df_k = df[df.apply(_keep_k, axis=1)].copy()
        if not df_k.empty:
            agg_k = aggregate(df_k, ["topology", "k"])
            # k=200 SWE-bench truncation removed 2026-06-04: under real-eval,
            # SAS keeps climbing past k=200 (peaks ~35% at k=200, partial coverage
            # at k=400/800), so suppressing those hides meaningful headroom.
            per_bench_aggs[bench_key]["k"] = agg_k
            # Threshold of 4 lets single-k pilots (e.g. MATH at k=10 only) render
            # in the same 4-panel format. β-fit panel auto-skips topos with <3
            # points so it gracefully degenerates to a topology comparison.
            if len(agg_k) >= 4:
                fits = plot_kscan(bench_key, bench_label, agg_k)
                print(f"  {bench_label} k-sweep exponents:")
                for t, sl, r2 in fits:
                    print(f"    {t:<15} β={sl:.3f}  R²={r2:.3f}")

        # R-sweep (k=RSWEEP_K, cent + decent, M=3 default).
        # Filter to k=RSWEEP_K and require R_explicit so we exclude unsuffixed
        # legacy R=5 cent files (which would double-count with explicit _R5).
        # Accept either unsuffixed M (default M=3) OR explicit "_M3" suffix
        # (the rhighext sbatch writes _R{N}_M3 for high-R cells like R=7, R=10).
        df_r = df[(df["topology"].isin(["centralized", "decentralized"])) &
                  (df["k"] == RSWEEP_K) &
                  df["R_explicit"].notna() &
                  (df["M_explicit"].isna() | (df["M_explicit"] == "3"))].copy()
        if not df_r.empty:
            agg_r = aggregate(df_r, ["topology", "R"])
            per_bench_aggs[bench_key]["r"] = agg_r
            if len(agg_r) >= 4:
                plot_rscan(bench_key, bench_label, agg_r)

        # M-sweep (k=MSWEEP_K, R=MSWEEP_R, cent + decent).
        # Unsuffixed-M files at this (k, R) are treated as the default M=3 point.
        df_m = df[(df["topology"].isin(["centralized", "decentralized"])) &
                  (df["k"] == MSWEEP_K) &
                  (df["R"] == MSWEEP_R)].copy()
        if not df_m.empty:
            agg_m = aggregate(df_m, ["topology", "M"])
            per_bench_aggs[bench_key]["m"] = agg_m
            if len(agg_m) >= 4:
                m_fits = plot_mscan(bench_key, bench_label, agg_m)
                if m_fits:
                    print(f"  {bench_label} M-sweep exponents:")
                    for t, sl, r2 in m_fits:
                        print(f"    {t:<15} β={sl:.3f}  R²={r2:.3f}")

    # Combined 4x3 grids across 4 main benchmarks (accuracy + energy)
    plot_combined_4bench(per_bench_aggs, metric="acc")
    plot_combined_4bench(per_bench_aggs, metric="energy")
    plot_combined_4bench_pareto(per_bench_aggs)


if __name__ == "__main__":
    main()
