"""Combined cross-benchmark k-scaling and R-scaling figures.

Replaces the older finding3_kscan_2x4.png / finding5_rscan_2x4.png which
covered only 4 prose benchmarks. New versions include swebench (and math
where applicable). swebench is the visible iteration-required outlier and
should appear alongside prose benchmarks for direct comparison.

Outputs:
  figures/finding3_kscan_combined.png   — 2 rows × 6 cols, all 6 benches
  figures/finding5_rscan_combined.png   — 2 rows × 4 cols, prose with R-scan data
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_R = {"centralized": 5, "decentralized": 2}
KSWEEP_R  = {"sas": None, "independent": None, "centralized": 2, "decentralized": 2}

TOPO_STYLE = {
    "sas":            {"color": "#1f77b4", "marker": "o", "label": "SAS"},
    "independent":    {"color": "#ff7f0e", "marker": "s", "label": "Independent (M=3)"},
    "centralized":    {"color": "#2ca02c", "marker": "^", "label": "Centralized R=2"},
    "decentralized":  {"color": "#d62728", "marker": "D", "label": "Decentralized R=2"},
}

BENCH_K = [
    ("qampari",         "a5000_qampari_v4",      "QAMPARI"),
    ("fanoutqa",        "a5000_fanoutqa_v4",     "FanOutQA"),
    ("workbench",       "a5000_workbench_v2",    "WorkBench"),
    ("browsecomp",      "a5000_browsecomp_pilot","BrowseComp+"),
    ("swebench",        "a5000_swebench",        "SWE-bench"),
]
BENCH_R = [
    ("qampari",         "a5000_qampari_v4",      "QAMPARI"),
    ("fanoutqa",        "a5000_fanoutqa_v4",     "FanOutQA"),
    ("workbench",       "a5000_workbench_v2",    "WorkBench"),
    ("browsecomp",      "a5000_browsecomp_pilot","BrowseComp+"),
]

# M-scan covers 5 benchmarks with usable data. Qampari M-ablation completed
# 2026-05-06 (all 6 cells at n=100); math has no M-ablation and is excluded.
BENCH_M = [
    ("workbench",  "a5000_workbench_v2",     "WorkBench"),
    ("fanoutqa",   "a5000_fanoutqa_v4",      "FanOutQA"),
    ("browsecomp", "a5000_browsecomp_pilot", "BrowseComp+"),
    ("qampari",    "a5000_qampari_v4",       "QAMPARI"),
    ("swebench",   "a5000_swebench",         "SWE-bench"),
]

cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z_]+)_k(\d+)(?:_R(\d+))?\.jsonl$")


def load_data(bench_dir, k_max=None):
    """Load all task-runs from the benchmark dir. If k_max is specified,
    drop rows with k > k_max (used to keep the combined figure readable
    when one benchmark — swebench — has very-high-k SAS sweeps that would
    otherwise dominate the x-axis range)."""
    rows = []
    for f in sorted(glob.glob(f"mas-energy/results/{bench_dir}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = cre.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str = m.group(1), int(m.group(2)), m.group(3)
        if topo not in TOPO_STYLE: continue
        if k_max is not None and k > k_max: continue
        R_actual = int(r_str) if r_str else DEFAULT_R.get(topo, 2)
        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            e = d.get("gpu_dynamic_energy_joules", 0) or 0
            acc = (float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                   else (1.0 if d.get("correct") else 0.0))
            rows.append(dict(topology=topo, k=k, R=R_actual,
                             R_explicit=r_str, acc=acc, energy_J=e))
    return rows


def aggregate_kscan(rows):
    """Filter to KSWEEP_R per topology, aggregate by (topology, k)."""
    has_R2 = set()
    by_topo_k = defaultdict(list)
    for r in rows:
        by_topo_k[(r["topology"], r["k"])].append(r)
        if r["R_explicit"] == "2":
            has_R2.add((r["topology"], r["k"]))

    keep = []
    for r in rows:
        topo = r["topology"]
        target = KSWEEP_R.get(topo)
        if target is None:
            if r["R_explicit"] is None:
                keep.append(r)
        else:
            if (topo, r["k"]) in has_R2:
                if r["R_explicit"] == "2":
                    keep.append(r)
            else:
                if r["R_explicit"] is None:
                    keep.append(r)

    agg = defaultdict(lambda: dict(accs=[], energies=[]))
    for r in keep:
        agg[(r["topology"], r["k"])]["accs"].append(r["acc"])
        agg[(r["topology"], r["k"])]["energies"].append(r["energy_J"])
    return [
        dict(topology=t, k=k, n=len(d["accs"]),
             acc=100*np.mean(d["accs"]), acc_se=100*np.std(d["accs"])/max(1,len(d["accs"])**0.5),
             energy_kJ=np.mean(d["energies"])/1000,
             energy_se=np.std(d["energies"])/1000/max(1,len(d["energies"])**0.5))
        for (t, k), d in sorted(agg.items())
    ]


def aggregate_rscan(rows):
    """Filter to centralized + decentralized at k=10 across R∈[1,5]."""
    keep = [r for r in rows
            if r["topology"] in ("centralized", "decentralized")
            and r["k"] == 10 and r["R_explicit"] is not None]
    agg = defaultdict(lambda: dict(accs=[], energies=[]))
    for r in keep:
        agg[(r["topology"], r["R"])]["accs"].append(r["acc"])
        agg[(r["topology"], r["R"])]["energies"].append(r["energy_J"])
    return [
        dict(topology=t, R=R, n=len(d["accs"]),
             acc=100*np.mean(d["accs"]), acc_se=100*np.std(d["accs"])/max(1,len(d["accs"])**0.5),
             energy_kJ=np.mean(d["energies"])/1000,
             energy_se=np.std(d["energies"])/1000/max(1,len(d["energies"])**0.5))
        for (t, R), d in sorted(agg.items())
    ]


def plot_axis(ax, agg, x_field, ylab, title, log_y=False, x_label=""):
    has_data = False
    for topo, style in TOPO_STYLE.items():
        sub = [a for a in agg if a["topology"] == topo]
        if not sub: continue
        sub = sorted(sub, key=lambda r: r[x_field])
        xs = [r[x_field] for r in sub]
        ys = [r["acc"] if "acc" in ylab.lower() or "accuracy" in ylab.lower() else r["energy_kJ"] for r in sub]
        ses = [r["acc_se"] if "acc" in ylab.lower() or "accuracy" in ylab.lower() else r["energy_se"] for r in sub]
        ax.errorbar(xs, ys, yerr=[1.96*s for s in ses],
                    color=style["color"], marker=style["marker"], label=style["label"],
                    lw=1.8, markersize=6, capsize=2)
        has_data = True
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    return has_data


def make_kscan_combined():
    """Combined k-scan figure across benchmarks. Cap swebench at k=50 so its
    very-high-k SAS sweep (k=100,200,400,800) doesn't blow out the x-axis
    range and obscure the other benchmarks. The full swebench sweep is
    plotted separately by make_swebench_full_kscan()."""
    nb = len(BENCH_K)
    fig, axes = plt.subplots(2, nb, figsize=(4.2 * nb, 8.0), sharex=False)
    for col, (key, sub, label) in enumerate(BENCH_K):
        k_cap = 50 if key == 'swebench' else None
        rows = load_data(sub, k_max=k_cap)
        agg = aggregate_kscan(rows)
        # row 0: accuracy vs k
        ax_acc = axes[0, col]
        plot_axis(ax_acc, agg, "k", "Accuracy (%)",
                  label, x_label="k (max ReAct steps)")
        # row 1: energy vs k (log y to handle wide swebench range)
        ax_e = axes[1, col]
        plot_axis(ax_e, agg, "k", "Energy per task (kJ)",
                  label, log_y=True, x_label="k (max ReAct steps)")
        # legend only on first column
        if col == 0:
            ax_acc.legend(fontsize=8, loc="lower right")

    fig.suptitle("k-scaling across benchmarks: accuracy and energy",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = f"{OUT_DIR}/finding3_kscan_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def make_rscan_combined():
    nb = len(BENCH_R)
    fig, axes = plt.subplots(2, nb, figsize=(4.2 * nb, 8.0))
    for col, (key, sub, label) in enumerate(BENCH_R):
        rows = load_data(sub)
        agg = aggregate_rscan(rows)
        ax_acc = axes[0, col]
        plot_axis(ax_acc, agg, "R", "Accuracy (%)",
                  label, x_label="R (debate / orchestration rounds)")
        ax_e = axes[1, col]
        plot_axis(ax_e, agg, "R", "Energy per task (kJ)",
                  label, x_label="R (debate / orchestration rounds)")
        if col == 0:
            ax_acc.legend(fontsize=8, loc="best")

    fig.suptitle("R-scaling across benchmarks (k=10, M=3): accuracy and energy",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = f"{OUT_DIR}/finding5_rscan_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def make_swebench_full_kscan():
    """Dedicated swebench k-scan covering very-high-k SAS sweeps (k up to 800)
    to demonstrate that SAS itself saturates given enough iteration budget.
    This complements the iteration-required claim: SWE-bench accuracy keeps
    climbing with MAS iteration (β_E ≈ 1.09 from finding #4), but single-
    agent SAS does eventually saturate too — extra iteration alone hits a
    ceiling without the parallelism MAS provides."""
    rows = load_data('a5000_swebench', k_max=None)
    if not rows:
        print("  no swebench data"); return
    agg = aggregate_kscan(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_acc, ax_e = axes

    for topo, style in TOPO_STYLE.items():
        sub = sorted([a for a in agg if a['topology'] == topo], key=lambda r: r['k'])
        if not sub: continue
        xs = [r['k'] for r in sub]
        ax_acc.errorbar(xs, [r['acc'] for r in sub],
                        yerr=[1.96 * r['acc_se'] for r in sub],
                        color=style['color'], marker=style['marker'],
                        label=style['label'], lw=1.8, markersize=7, capsize=3)
        ax_e.errorbar(xs, [r['energy_kJ'] for r in sub],
                      yerr=[1.96 * r['energy_se'] for r in sub],
                      color=style['color'], marker=style['marker'],
                      label=style['label'], lw=1.8, markersize=7, capsize=3)

    ax_acc.set_xscale('log'); ax_acc.set_xlabel('k (max ReAct steps, log scale)', fontsize=11)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
    ax_acc.set_title('Accuracy vs k (full range, including high-k SAS sweep)',
                     fontsize=11, fontweight='bold')
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3, which='both')

    ax_e.set_xscale('log'); ax_e.set_yscale('log')
    ax_e.set_xlabel('k (max ReAct steps, log scale)', fontsize=11)
    ax_e.set_ylabel('Energy per task (kJ, log scale)', fontsize=11)
    ax_e.set_title('Energy vs k (full range)', fontsize=11, fontweight='bold')
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.3, which='both')

    fig.suptitle('SWE-bench full k-scan: SAS saturates at very high k '
                 '(parallelism is the additional lever MAS provides)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = f"{OUT_DIR}/swebench_full_kscan.png"
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved {out}")
    plt.close(fig)


def aggregate_mscan(rows_m_explicit, rows_M3_baseline):
    """M-scan covers M ∈ {2,4,5} explicitly + M=3 from the canonical k=10 R=2
    main-study baseline. Aggregates by (topology, M)."""
    agg = defaultdict(lambda: dict(accs=[], energies=[]))
    for r in rows_m_explicit:
        if r["topology"] not in ("centralized", "decentralized"): continue
        agg[(r["topology"], r["M"])]["accs"].append(r["acc"])
        agg[(r["topology"], r["M"])]["energies"].append(r["energy_J"])
    for r in rows_M3_baseline:
        if r["topology"] not in ("centralized", "decentralized"): continue
        agg[(r["topology"], 3)]["accs"].append(r["acc"])
        agg[(r["topology"], 3)]["energies"].append(r["energy_J"])
    return [
        dict(topology=t, M=M, n=len(d["accs"]),
             acc=100*np.mean(d["accs"]), acc_se=100*np.std(d["accs"])/max(1, len(d["accs"])**0.5),
             energy_kJ=np.mean(d["energies"])/1000,
             energy_se=np.std(d["energies"])/1000/max(1, len(d["energies"])**0.5))
        for (t, M), d in sorted(agg.items())
    ]


# Filename pattern for M-explicit cells: ..._k10_R2_M<M>.jsonl
mre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z_]+)_k10_R2_M(\d+)\.jsonl$")


def load_m_data(bench_dir):
    """Load M-explicit cells (M=2,4,5) plus the M=3 canonical baseline.
    M=3 baseline preferred from <topo>_k10_R2.jsonl, falls back to
    <topo>_k10.jsonl (some benches like swebench use the no-R-suffix
    form for the canonical R=2 main-study baseline)."""
    rows_m = []
    M3_by_topo = {}  # topo -> list of rows; preferred files override later
    for f in sorted(glob.glob(f"mas-energy/results/{bench_dir}/Qwen_Qwen3.5-9B_*.jsonl")):
        bn = os.path.basename(f)
        m = mre.match(bn)
        if m:
            topo, M = m.group(1), int(m.group(2))
            for line in open(f):
                try: d = json.loads(line)
                except: continue
                if d.get("error"): continue
                e = d.get("gpu_dynamic_energy_joules", 0) or 0
                acc = (float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                       else (1.0 if d.get("correct") else 0.0))
                rows_m.append(dict(topology=topo, M=M, acc=acc, energy_J=e))
            continue
        # M=3 baseline candidates, preferring _R2 suffix when present
        m3_r2 = re.match(r"Qwen_Qwen3\.5-9B_([a-z_]+)_k10_R2\.jsonl$", bn)
        m3_no = re.match(r"Qwen_Qwen3\.5-9B_([a-z_]+)_k10\.jsonl$", bn)
        if m3_r2:
            topo, prefer = m3_r2.group(1), True
        elif m3_no:
            topo, prefer = m3_no.group(1), False
        else:
            continue
        rows = []
        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            e = d.get("gpu_dynamic_energy_joules", 0) or 0
            acc = (float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                   else (1.0 if d.get("correct") else 0.0))
            rows.append(dict(topology=topo, acc=acc, energy_J=e))
        if not rows:
            continue
        # Preferred (_R2) wins. Fallback (_k10) only if preferred not seen yet.
        if prefer or topo not in M3_by_topo:
            M3_by_topo[topo] = rows
    rows_M3 = [r for rs in M3_by_topo.values() for r in rs]
    return rows_m, rows_M3


def make_mscan_combined():
    """Combined M-scan figure across the 4 benchmarks with usable data,
    matching style of finding3_kscan_combined.png. Top row = accuracy %,
    bottom row = energy kJ (log scale). Both topologies with 95% CI bars."""
    nb = len(BENCH_M)
    fig, axes = plt.subplots(2, nb, figsize=(4.2 * nb, 8.0), sharex=False)
    Ms_full = {2, 3, 4, 5}
    pending_summary = []  # list of "Bench Decent M=4,5"

    for col, (key, sub, label) in enumerate(BENCH_M):
        rows_m, rows_M3 = load_m_data(sub)
        agg = aggregate_mscan(rows_m, rows_M3)
        ax_acc = axes[0, col]
        plot_axis(ax_acc, agg, "M", "Accuracy (%)",
                  label, x_label="M (number of agents)")
        ax_e = axes[1, col]
        plot_axis(ax_e, agg, "M", "Energy per task (kJ)",
                  label, log_y=True, x_label="M (number of agents)")
        if col == 0:
            ax_acc.legend(fontsize=8, loc="lower right")

        # Flag pending cells visually so the figure honestly conveys gaps.
        for topo, color in (("centralized", "#2ca02c"), ("decentralized", "#d62728")):
            present = {r["M"] for r in agg if r["topology"] == topo}
            missing = sorted(Ms_full - present)
            if missing:
                pending_summary.append(f"{label} {topo[:5]} M={','.join(str(m) for m in missing)}")
                ax_acc.text(0.02, 0.02 + 0.08 * (1 if topo == "decentralized" else 0),
                            f"{topo[:5].capitalize()}: M={','.join(str(m) for m in missing)} pending",
                            transform=ax_acc.transAxes, fontsize=7.5, color=color,
                            style='italic', alpha=0.85)

    title_main = "M-scaling across benchmarks (k=10, R=2): accuracy and energy"
    if pending_summary:
        title_sub = "Pending cells (jobs queued/running): " + "; ".join(pending_summary)
        fig.suptitle(title_main + "\n" + title_sub, fontsize=12, fontweight="bold", y=1.005)
    else:
        fig.suptitle(title_main, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = f"{OUT_DIR}/finding_mscan_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    if pending_summary:
        print("  pending cells flagged on figure: " + "; ".join(pending_summary))
    plt.close(fig)


if __name__ == "__main__":
    make_kscan_combined()
    make_rscan_combined()
    make_swebench_full_kscan()
    make_mscan_combined()
