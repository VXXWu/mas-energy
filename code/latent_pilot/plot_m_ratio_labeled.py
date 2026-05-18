"""M-ablation analog of fig_unified_scaling_energy_ratio_labeled.png.

For each (topology, M) cell at k=10, R=2: compute energy ratio relative to
SAS k=10 baseline (per-bench geomean) and ΔAcc vs SAS k=10 (per-bench
differenced mean). Plot on a shared (E_ratio, ΔAcc) axis with each point
labeled with its M value, colored/shaped by topology, log-x scale.

Output: figures/frontier/fig_m_scaling_energy_ratio_labeled.png

Cells with n < 100 are open markers (excluded from the per-topology fit).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean
from math import log10, log

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path('mas-energy/results')
DIRS = {
    'workbench':       'a5000_workbench_v2',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'qampari':         'a5000_qampari_v4',
    'swebench':        'a5000_swebench',
}

TOPO_STYLE = {
    'centralized':   dict(color='#2ca02c', marker='P', label='Centralized'),
    'decentralized': dict(color='#d62728', marker='D', label='Decentralized'),
}

# Within-topology M-marker offsets so labels don't overlap when curves cross.
TOPO_OFFSET = {
    'centralized':   (5, 8),
    'decentralized': (5, -10),
}


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return float(bool(r.get('correct')))


def load_per_bench_anchors():
    """SAS k=10 baseline per benchmark — for E_ratio normalization and ΔAcc."""
    anchors = {}
    for bench, sub in DIRS.items():
        f = ROOT / sub / 'Qwen_Qwen3.5-9B_sas_k10.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        if not rows:
            continue
        anchors[bench] = dict(
            acc=mean(acc(r) for r in rows),
            E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
        )
    return anchors


def load_m_cells(anchors):
    """{(topo, M): {acc, E_ratio, n_total}} aggregated across benches.

    For each (topo, M, bench), compute per-bench acc and E_kJ.
    Then ΔAcc = mean across benches of (acc[bench] - sas[bench]).
    E_ratio = geomean across benches of (E_kJ[bench] / sas_E_kJ[bench])."""
    per_cell = defaultdict(dict)  # (topo, M) -> {bench: {acc, E_kJ, n}}

    for bench, sub in DIRS.items():
        d = ROOT / sub
        if not d.exists():
            continue
        # M-explicit cells
        for path in sorted(d.glob('Qwen_Qwen3.5-9B_*_k10_R2_M*.jsonl')):
            m = re.match(r'Qwen_Qwen3\.5-9B_(centralized|decentralized)_k10_R2_M(\d+)\.jsonl', path.name)
            if not m:
                continue
            topo, M = m.group(1), int(m.group(2))
            rows = [json.loads(l) for l in path.open()]
            if len(rows) < 30:
                continue
            per_cell[(topo, M)][bench] = dict(
                n=len(rows),
                acc=mean(acc(r) for r in rows),
                E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
            )
        # M=3 baseline = canonical _k10_R2 (preferred) or _k10
        for fname in ('Qwen_Qwen3.5-9B_centralized_k10_R2.jsonl',
                      'Qwen_Qwen3.5-9B_decentralized_k10_R2.jsonl',
                      'Qwen_Qwen3.5-9B_centralized_k10.jsonl',
                      'Qwen_Qwen3.5-9B_decentralized_k10.jsonl'):
            f = d / fname
            if not f.exists():
                continue
            topo = 'decentralized' if 'decentralized' in fname else 'centralized'
            if bench in per_cell.get((topo, 3), {}):
                continue  # prefer R2 over no-suffix
            rows = [json.loads(l) for l in f.open()]
            if len(rows) < 30:
                continue
            per_cell[(topo, 3)][bench] = dict(
                n=len(rows),
                acc=mean(acc(r) for r in rows),
                E_kJ=mean((r.get('gpu_dynamic_energy_joules') or 0) for r in rows) / 1000,
            )

    # Aggregate to (E_ratio_geomean, ΔAcc_mean)
    cells = []
    for (topo, M), per_bench in per_cell.items():
        bench_overlap = [b for b in per_bench if b in anchors]
        if len(bench_overlap) < 2:
            continue
        d_acc = mean(per_bench[b]['acc'] - anchors[b]['acc'] for b in bench_overlap)
        log_ratios = [log(per_bench[b]['E_kJ'] / anchors[b]['E_kJ'])
                      for b in bench_overlap if anchors[b]['E_kJ'] > 0]
        E_ratio = float(np.exp(np.mean(log_ratios))) if log_ratios else None
        n_total = sum(per_bench[b]['n'] for b in bench_overlap)
        cells.append(dict(
            topo=topo, M=M,
            E_ratio=E_ratio, delta_acc=d_acc,
            n_total=n_total, n_benches=len(bench_overlap),
            min_n=min(per_bench[b]['n'] for b in bench_overlap),
        ))
    return cells


def main():
    anchors = load_per_bench_anchors()
    cells = load_m_cells(anchors)
    print(f'Loaded {len(cells)} (topo, M) cells across {len(anchors)} benches.')

    print(f'\n{"topo":<14} {"M":>2}  {"n_min":>5}  {"benches":>2}  {"E_ratio":>8}  {"ΔAcc":>7}')
    for c in sorted(cells, key=lambda c: (c['topo'], c['M'])):
        print(f'  {c["topo"]:<14} {c["M"]:>2}  {c["min_n"]:>5}  {c["n_benches"]:>2}  '
              f'{c["E_ratio"]:>8.3f}  {c["delta_acc"]:>+7.3f}')

    # Per-topology log-linear fit. Threshold min_n >= 50 (M=3 main-study
    # baselines are at n=50). Also require n_benches == max_benches across
    # cells in the fit, so we don't mix cells averaged over different
    # benchmark subsets (high-M cells only ran on 2/5 benches and have
    # systematically higher ΔAcc due to subset selection).
    fits = {}
    max_benches = max(c['n_benches'] for c in cells) if cells else 0
    print(f'\nLog-linear fit on M-cells (min_n>=50, n_benches=={max_benches}):')
    print(f'{"topo":<14} {"n":>3} {"slope":>8} {"intcpt":>8} {"R²":>6}')
    for topo in ('centralized', 'decentralized'):
        pts = [c for c in cells if c['topo'] == topo
               and c['min_n'] >= 50 and c['n_benches'] == max_benches]
        if len(pts) < 3:
            continue
        xs = np.log10([c['E_ratio'] for c in pts])
        ys = np.array([c['delta_acc'] for c in pts])
        s, b = np.polyfit(xs, ys, 1)
        pred = s * xs + b
        r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum() if ys.var() > 0 else float('nan')
        fits[topo] = (s, b, r2, pts)
        print(f'{topo:<14} {len(pts):>3d} {s:>+8.3f} {b:>+8.3f} {r2:>6.3f}')

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for topo in ('centralized', 'decentralized'):
        s = TOPO_STYLE[topo]
        topo_cells = sorted([c for c in cells if c['topo'] == topo], key=lambda c: c['M'])
        # Full-data (min_n >= 100) cells: filled markers
        full = [c for c in topo_cells if c['min_n'] >= 100]
        partial = [c for c in topo_cells if c['min_n'] < 100]
        if full:
            ax.scatter([c['E_ratio'] for c in full],
                       [c['delta_acc'] for c in full],
                       s=130, color=s['color'], marker=s['marker'],
                       edgecolor='black', linewidth=1.4, alpha=0.95, zorder=5,
                       label=f"{s['label']}  ({len(full)} full)")
        if partial:
            ax.scatter([c['E_ratio'] for c in partial],
                       [c['delta_acc'] for c in partial],
                       s=130, facecolors='none', edgecolor=s['color'],
                       marker=s['marker'], linewidth=1.6, alpha=0.85, zorder=4,
                       label=f"{s['label']}  ({len(partial)} partial n<100)")

        # Connect M-points in order with a dashed line for visual flow
        if len(topo_cells) >= 2:
            ax.plot([c['E_ratio'] for c in topo_cells],
                    [c['delta_acc'] for c in topo_cells],
                    color=s['color'], linewidth=1.4, alpha=0.45, linestyle='-',
                    zorder=3)

        # Log-linear fit overlay
        if topo in fits:
            s_fit, b_fit, r2, pts = fits[topo]
            x_lo = min(c['E_ratio'] for c in pts)
            x_hi = max(c['E_ratio'] for c in pts)
            x_grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 50)
            y_grid = s_fit * np.log10(x_grid) + b_fit
            ax.plot(x_grid, y_grid, color=s['color'], linewidth=1.6, alpha=0.6,
                    linestyle='--', zorder=2)

        # Annotate every point with its M value
        dx, dy = TOPO_OFFSET[topo]
        for c in topo_cells:
            ax.annotate(f'M={c["M"]}', xy=(c['E_ratio'], c['delta_acc']),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=8.5, color=s['color'], fontweight='bold',
                        zorder=7)

    # Reference lines
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Energy ratio  (per-bench geomean of E / E_SAS_k=10, log scale)', fontsize=11)
    ax.set_ylabel('Mean ΔAccuracy vs SAS k=10  (per-bench differenced)', fontsize=11)

    title = ('M-scaling on the unified energy-ratio axis  (k=10, R=2)\n'
             '4 benches: WorkBench, FanOutQA, BrowseComp+, SWE-bench  |  '
             'Open markers = at least one bench n<100')
    if fits:
        slopes = '  '.join(f'{topo} α={s:+.2f}/dec R²={r2:.2f}'
                          for topo, (s, _, r2, _) in fits.items())
        title += f'\nPer-topology log-linear fit: {slopes}'
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

    out = Path('figures/frontier/fig_m_scaling_energy_ratio_labeled.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
