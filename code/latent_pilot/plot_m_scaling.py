"""M-scaling figures: per-benchmark M-scan and overlay with k-scan saturation
curves.

Produces:
  figures/frontier/fig_m_scaling_per_bench.png   — 2x3 acc-vs-M panels
  figures/frontier/fig_m_scaling_energy_acc.png  — M cells on per-topology
                                                    saturation curves (test
                                                    of "M is on the same axis
                                                    as k" claim)

Coverage as of this run is incomplete on qampari (Cent M={2,4,5}, Decent M=2)
and swebench (Decent M={4,5}). math has no M-ablation. These cells render as
gaps in the figure; legend documents which.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path('mas-energy/results')
DIRS = {
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'qampari':         'a5000_qampari_v4',
    'workbench':       'a5000_workbench_v2',
    'swebench':        'a5000_swebench',
    'math':            'a5000_math_pilot',
}

BENCH_LABEL = {
    'workbench':       'WorkBench',
    'fanoutqa':        'FanOutQA',
    'browsecomp_plus': 'BrowseComp+',
    'qampari':         'QAMPARI',
    'swebench':        'SWE-bench',
    'math':            'MATH-L5',
}

TOPO_STYLE = {
    'centralized':   dict(color='#2ca02c', marker='P', label='Centralized'),
    'decentralized': dict(color='#d62728', marker='D', label='Decentralized'),
}


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return float(bool(r.get('correct')))


def load_cells():
    """Return {(bench, topo, M, axis): {n, acc, E_kJ}} where axis is 'M' or 'k'."""
    cells = {}
    for bench, sub in DIRS.items():
        d = ROOT / sub
        if not d.exists():
            continue

        # M-variants
        for path in sorted(d.glob('Qwen_Qwen3.5-9B_*_k10_R2_M*.jsonl')):
            m = re.match(r'Qwen_Qwen3\.5-9B_(centralized|decentralized)_k10_R2_M(\d+)\.jsonl', path.name)
            if not m:
                continue
            topo, M = m.group(1), int(m.group(2))
            rows = [json.loads(l) for l in path.open()]
            if len(rows) < 30:
                continue
            cells[(bench, topo, M, 'M')] = dict(
                n=len(rows),
                acc=mean(acc(r) for r in rows),
                E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
            )

        # M=3 baseline = canonical k=10 R=2 main-study file (no _M suffix)
        for path in sorted(d.glob('Qwen_Qwen3.5-9B_*_k10_R2.jsonl')):
            m = re.match(r'Qwen_Qwen3\.5-9B_(centralized|decentralized)_k10_R2\.jsonl', path.name)
            if not m:
                continue
            topo = m.group(1)
            rows = [json.loads(l) for l in path.open()]
            if len(rows) < 50:
                continue
            cells[(bench, topo, 3, 'M')] = dict(
                n=len(rows),
                acc=mean(acc(r) for r in rows),
                E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
            )

        # k-scan cells (varying k, M=3 default)
        for path in sorted(d.glob('Qwen_Qwen3.5-9B_*_k*.jsonl')):
            m = re.match(r'Qwen_Qwen3\.5-9B_(centralized|decentralized)_k(\d+)(?:_R2)?\.jsonl', path.name)
            if not m:
                continue
            topo, k = m.group(1), int(m.group(2))
            rows = [json.loads(l) for l in path.open()]
            if len(rows) < 50:
                continue
            cells[(bench, topo, k, 'k')] = dict(
                n=len(rows),
                acc=mean(acc(r) for r in rows),
                E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
                k=k,
            )

    return cells


def plot_per_bench(cells, out):
    """2x3 grid: per-benchmark acc-vs-M and E-vs-M, both topologies."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True)
    axes = axes.flatten()
    Ms = [2, 3, 4, 5]

    for i, bench in enumerate(['workbench', 'fanoutqa', 'browsecomp_plus',
                               'qampari', 'swebench', 'math']):
        ax = axes[i]
        ax2 = ax.twinx()
        any_data = False
        for topo in ('centralized', 'decentralized'):
            s = TOPO_STYLE[topo]
            xs, ys, es, ns = [], [], [], []
            for M in Ms:
                c = cells.get((bench, topo, M, 'M'))
                if c is None:
                    continue
                xs.append(M)
                ys.append(c['acc'])
                es.append(c['E_kJ'])
                ns.append(c['n'])

            if xs:
                any_data = True
                # Mark partial cells with open markers
                for x, y, e, n in zip(xs, ys, es, ns):
                    facecolor = s['color'] if n >= 100 else 'none'
                    ax.scatter([x], [y], s=110, color=facecolor, marker=s['marker'],
                               edgecolor=s['color'], linewidth=1.4, alpha=0.95, zorder=5)
                    ax2.scatter([x], [e], s=70, color=s['color'], marker=s['marker'],
                                edgecolor='gray', linewidth=0.5, alpha=0.35, zorder=3)
                ax.plot(xs, ys, color=s['color'], linewidth=1.5, alpha=0.7, zorder=4,
                        label=f'{s["label"]} (acc)')
                ax2.plot(xs, es, color=s['color'], linewidth=1.0, alpha=0.35,
                         linestyle=':', zorder=2)

        ax.set_title(BENCH_LABEL[bench], fontsize=11, fontweight='bold')
        ax.set_xticks(Ms)
        ax.set_ylabel('Accuracy', color='black', fontsize=9)
        ax2.set_ylabel('Energy (kJ)', color='gray', fontsize=9)
        ax2.tick_params(axis='y', colors='gray')
        ax.grid(True, alpha=0.3)
        if not any_data:
            ax.text(0.5, 0.5, 'no M-ablation data', ha='center', va='center',
                    transform=ax.transAxes, color='gray', style='italic')
        if i >= 3:
            ax.set_xlabel('M (number of agents)', fontsize=9)
        if i == 0:
            ax.legend(loc='lower right', fontsize=8.5)

    fig.suptitle(
        'M-scaling per benchmark — accuracy (filled) and energy (dotted) vs M  |  k=10, R=2\n'
        'Open markers = partial n<100. Energy axis (right) shows ~linear M scaling everywhere.',
        fontsize=11, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'  saved {out}')
    plt.close()


def plot_energy_acc_overlay(cells, out):
    """Overlay M-scan cells onto per-topology k-scan saturation curves to test
    whether M cells sit on the same compute axis as k cells."""
    from scipy.optimize import curve_fit

    # SAS-k=10 baseline per benchmark for ΔAcc differencing.
    sas_k10 = {}
    for bench, sub in DIRS.items():
        f = ROOT / sub / 'Qwen_Qwen3.5-9B_sas_k10.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        if rows:
            sas_k10[bench] = mean(acc(r) for r in rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()

    for i, bench in enumerate(['workbench', 'fanoutqa', 'browsecomp_plus',
                               'qampari', 'swebench', 'math']):
        ax = axes[i]
        any_data = False

        for topo in ('centralized', 'decentralized'):
            s = TOPO_STYLE[topo]

            # k-scan points (varying k, M=3 default)
            k_pts = sorted([(c['E_kJ'], c['acc'], c.get('k'))
                            for (b, t, _, axis), c in cells.items()
                            if b == bench and t == topo and axis == 'k'])

            # M-scan points (varying M, k=10 fixed)
            m_pts = sorted([(c['E_kJ'], c['acc'], M)
                            for (b, t, M, axis), c in cells.items()
                            if b == bench and t == topo and axis == 'M'])

            if not k_pts and not m_pts:
                continue
            any_data = True

            # ΔAcc relative to SAS-k=10 baseline
            base = sas_k10.get(bench, 0)

            if k_pts:
                xs = [p[0] for p in k_pts]
                ys = [p[1] - base for p in k_pts]
                ax.scatter(xs, ys, s=70, color=s['color'], marker=s['marker'],
                           edgecolor='black', linewidth=0.5, alpha=0.5, zorder=3,
                           label=f'{s["label"]} k-scan ({len(k_pts)})')
                # Fit saturation curve in k-space, plot against E
                if len(k_pts) >= 4:
                    Es = np.array(xs)
                    accs = np.array([p[1] for p in k_pts])
                    def sat(E, ceiling, drop, alpha):
                        return ceiling - drop * np.power(np.maximum(E, 0.01), -alpha)
                    try:
                        popt, _ = curve_fit(sat, Es, accs, p0=[1.0, 5.0, 0.3], maxfev=5000)
                        E_grid = np.logspace(np.log10(Es.min()), np.log10(Es.max()), 100)
                        ax.plot(E_grid, sat(E_grid, *popt) - base, color=s['color'],
                                linewidth=1.5, alpha=0.6, zorder=2)
                    except Exception:
                        pass

            if m_pts:
                xs = [p[0] for p in m_pts]
                ys = [p[1] - base for p in m_pts]
                Ms_lab = [p[2] for p in m_pts]
                ax.scatter(xs, ys, s=130, color=s['color'], marker=s['marker'],
                           edgecolor='black', linewidth=1.4, alpha=0.95, zorder=5,
                           label=f'{s["label"]} M-scan ({len(m_pts)})')
                # Annotate each M-point with its M value
                for x, y, M in zip(xs, ys, Ms_lab):
                    ax.annotate(f'M={M}', xy=(x, y), xytext=(5, 4),
                                textcoords='offset points', fontsize=7.5,
                                color=s['color'], fontweight='bold')

        ax.set_xscale('log')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_title(BENCH_LABEL[bench], fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔAcc vs SAS k=10', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        if i >= 3:
            ax.set_xlabel('Energy per task (kJ, log)', fontsize=9)
        if any_data and i == 0:
            ax.legend(loc='lower right', fontsize=8)
        if not any_data:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray', style='italic')

    fig.suptitle(
        'M-scaling cells overlaid on k-scan saturation curves\n'
        'Big bold markers = M-scan (k=10 fixed). Small faint = k-scan (M=3 fixed). '
        'If M cells lie on the k saturation curve, M and k are the same compute axis.',
        fontsize=11, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'  saved {out}')
    plt.close()


def main():
    cells = load_cells()
    print(f'Loaded {len(cells)} cells.')
    Path('figures/frontier').mkdir(parents=True, exist_ok=True)
    plot_per_bench(cells, 'figures/frontier/fig_m_scaling_per_bench.png')
    plot_energy_acc_overlay(cells, 'figures/frontier/fig_m_scaling_energy_acc.png')


if __name__ == '__main__':
    main()
