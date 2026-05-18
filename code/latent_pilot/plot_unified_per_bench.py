"""Per-benchmark k-trajectories overlaid on the unified scaling law.

The unified scaling law (fig_unified_scaling) averages across benchmarks, which
hides a tension: per-benchmark accuracy saturates with k (each benchmark has
its own plateau), yet across benchmarks the law collapses to one curve.

This figure visualizes the reconciliation: each benchmark's k-trajectory is
a *sub-curve* covering a different segment of the global law. Easy benchmarks
(math, qampari) saturate quickly at low energy / low accuracy; harder ones
(swebench) extend further. Stitched together, they form the unified curve.

Output: figures/fig_unified_per_bench.png
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


MAIN = {
    'qampari':         'a5000_qampari_v4',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'workbench':       'a5000_workbench_v2',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'math':            'a5000_math_pilot',
    'swebench':        'a5000_swebench',
}
PHASE_A = 'a5000_phase_a_ablation'
PHASE_B2 = 'a5000_phase_b2_terse'

ABLATION_TOPOS = {
    'decentralized_answer_only':       PHASE_A,
    'decentralized_empty':             PHASE_A,
    'decentralized_empty_silent':      PHASE_A,
    'decentralized_minimal':           PHASE_A,
    'decentralized_minimal_empty':     PHASE_A,
    'decentralized_terse':             PHASE_B2,
    'independent_share':               PHASE_B2,
    'independent_share_minimal':       PHASE_B2,
}

EXCLUDED = {'centralized', 'hybrid', 'specialist'}
def is_excluded(topo):
    if topo in EXCLUDED: return True
    if 'truncate' in topo: return True
    return False

OPERATIONAL_K_MIN = 5

cre_main = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl$")

BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}
# Distinct colors for benchmarks
BENCH_COLOR = {
    'qampari':         '#e41a1c',
    'fanoutqa':        '#377eb8',
    'math':            '#4daf4a',
    'workbench':       '#984ea3',
    'browsecomp_plus': '#ff7f00',
    'swebench':        '#a65628',
}

TOPO_MARKER = {
    'sas':                  ('o', 60),
    'independent':          ('s', 60),
    'decentralized':        ('D', 60),
    # ablations: filled triangles
}


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def acc(r):
    if r.get('loose_accuracy') is not None: return float(r['loose_accuracy'])
    return 1.0 if r.get('correct') else 0.0


def gather_per_bench_cells():
    """Return dict bench -> list of (topo, k, R, energy, accuracy) cells."""
    out = defaultdict(list)
    for bench, sub in MAIN.items():
        p = Path('mas-energy/results') / sub
        if not p.exists(): continue
        for f in p.glob('Qwen_Qwen3.5-9B_*.jsonl'):
            if 'preclean' in f.name or 'bak' in f.name: continue
            m = cre_main.match(f.name)
            if not m: continue
            topo, k, r = m.group(1), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)
            if is_excluded(topo): continue
            rows = load(f)
            if len(rows) < 25: continue
            e = mean(row.get('gpu_dynamic_energy_joules', 0) or 0 for row in rows)
            a = mean(acc(row) for row in rows)
            out[bench].append(dict(topo=topo, k=k, R=r, energy=e, accuracy=a, family='main'))

    # Channel-muting ablations (k=10 R=2 each)
    for topo, root in ABLATION_TOPOS.items():
        for bench in MAIN:
            p = Path('mas-energy/results') / root / bench
            if not p.exists(): continue
            files = sorted(p.glob(f'*_{topo}_k*.jsonl'))
            if not files: continue
            rows = load(files[0])
            if len(rows) < 25: continue
            e = mean(row.get('gpu_dynamic_energy_joules', 0) or 0 for row in rows)
            a = mean(acc(row) for row in rows)
            out[bench].append(dict(topo=topo, k=10, R=2, energy=e, accuracy=a, family='ablation'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_unified_per_bench.png')
    args = ap.parse_args()

    per_bench = gather_per_bench_cells()
    benches = list(MAIN.keys())

    # Compute SAS k=10 per benchmark for ΔAcc reference
    sas_k10_per_bench = {}
    for bench, cells in per_bench.items():
        sas_k10 = next((c for c in cells if c['topo']=='sas' and c['k']==10 and c['R'] is None), None)
        if sas_k10:
            sas_k10_per_bench[bench] = sas_k10['accuracy']

    # ─── Single-panel version: all benchmarks overlaid on same axes ───
    fig, ax = plt.subplots(figsize=(12.5, 8))

    # Plot per-benchmark k-trajectories for SAS / Indep / Decent (varying k)
    for bench in benches:
        cells = per_bench.get(bench, [])
        if bench not in sas_k10_per_bench: continue
        sas_a = sas_k10_per_bench[bench]
        color = BENCH_COLOR[bench]
        # For each topology, gather points across k (R=2 default for cent/decent or no-R)
        for topo in ['sas', 'independent', 'decentralized']:
            sub = [c for c in cells if c['topo']==topo
                   and (c['R'] is None or c['R']==2)
                   and c['k'] >= OPERATIONAL_K_MIN]
            if not sub: continue
            sub.sort(key=lambda c: c['k'])
            xs = [c['energy'] for c in sub]
            ys = [c['accuracy'] - sas_a for c in sub]
            marker, _ = TOPO_MARKER.get(topo, ('o', 60))
            ax.plot(xs, ys, '-', color=color, linewidth=1.2, alpha=0.5, zorder=2)
            ax.scatter(xs, ys, s=55, color=color, marker=marker,
                       edgecolor='black', linewidth=0.6, alpha=0.85, zorder=4,
                       label=None)

        # Channel-muting ablations (single k=10 each, plotted as small ^ markers)
        abl_pts = [c for c in cells if c.get('family')=='ablation']
        if abl_pts:
            xs = [c['energy'] for c in abl_pts]
            ys = [c['accuracy'] - sas_a for c in abl_pts]
            ax.scatter(xs, ys, s=40, color=color, marker='^',
                       edgecolor='black', linewidth=0.5, alpha=0.7, zorder=3)

    # ─── Per-benchmark log-linear fits ───
    # Tests whether each benchmark INDIVIDUALLY shows log-linear scaling, or
    # whether the unified law is an averaging artifact across heterogeneous
    # per-benchmark shapes. If per-bench fits are individually tight (R²≥0.7),
    # the unified law is robust at the benchmark level. If they're heterogeneous
    # (mixed slopes, varying R²), the unified law captures average behavior
    # across a mixture of per-benchmark scaling regimes.
    print(f"\n{'benchmark':<18} {'n':>3} {'slope/dec':>10} {'R²':>6}  intercept")
    print('-' * 60)
    per_bench_fits = {}
    for bench in benches:
        if bench not in sas_k10_per_bench: continue
        sas_a = sas_k10_per_bench[bench]
        cells = per_bench.get(bench, [])
        # Restrict to in-regime cells (k≥5 multi-agent OR any SAS k)
        valid = [c for c in cells
                 if not (c['k'] < OPERATIONAL_K_MIN and c['topo'] != 'sas')]
        if len(valid) < 4: continue
        xs_b = np.log10([c['energy'] for c in valid])
        ys_b = np.array([c['accuracy'] - sas_a for c in valid])
        slope_b, intercept_b = np.polyfit(xs_b, ys_b, 1)
        pred_b = slope_b * xs_b + intercept_b
        ss_tot_b = ((ys_b - ys_b.mean())**2).sum()
        ss_res_b = ((ys_b - pred_b)**2).sum()
        r2_b = 1 - ss_res_b/ss_tot_b if ss_tot_b > 0 else float('nan')
        per_bench_fits[bench] = dict(slope=slope_b, intercept=intercept_b, r2=r2_b,
                                      n=len(valid), x_range=(xs_b.min(), xs_b.max()))
        print(f'{bench:<18} {len(valid):>3} {slope_b:>+10.3f} {r2_b:>6.3f}  {intercept_b:+.3f}')

        # Overlay each benchmark's fitted line on the figure
        x_fit_b = np.linspace(xs_b.min(), xs_b.max(), 50)
        y_fit_b = slope_b * x_fit_b + intercept_b
        ax.plot(10**x_fit_b, y_fit_b, '-', color=BENCH_COLOR[bench],
                linewidth=2.0, alpha=0.55, zorder=2.5)

    # Cross-benchmark unified fit (averaged across all in-regime cells)
    all_diffs = []
    for bench, cells in per_bench.items():
        if bench not in sas_k10_per_bench: continue
        sas_a = sas_k10_per_bench[bench]
        for c in cells:
            if c['k'] < OPERATIONAL_K_MIN and c['topo'] != 'sas':
                continue
            all_diffs.append((np.log10(c['energy']), c['accuracy'] - sas_a))
    if all_diffs:
        xs_arr = np.array([d[0] for d in all_diffs])
        ys_arr = np.array([d[1] for d in all_diffs])
        slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
        pred = slope*xs_arr + intercept
        ss_tot = ((ys_arr-ys_arr.mean())**2).sum()
        ss_res = ((ys_arr-pred)**2).sum()
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
        x_fit = np.linspace(xs_arr.min()-0.05, xs_arr.max()+0.05, 100)
        y_fit = slope*x_fit + intercept
        ax.plot(10**x_fit, y_fit, '--', color='#222', linewidth=2.4, alpha=0.85,
                zorder=5,
                label=f'Cross-benchmark unified fit  (slope=+{slope:.3f}/decade, R²={r2:.3f}, n={len(all_diffs)})')
        print(f"\nCross-benchmark fit: slope={slope:.3f}, R²={r2:.3f}, n={len(all_diffs)}")

    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel('Mean energy per task (J, log scale)', fontsize=12)
    ax.set_ylabel('ΔAccuracy vs SAS k=10 (per-benchmark)', fontsize=12)
    ax.set_title('Per-benchmark k-trajectories overlaid on the unified scaling law\n'
                 'Each benchmark saturates with k, but the saturation points lie along the same global curve',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Custom legend: benchmark colors + topology markers
    from matplotlib.lines import Line2D
    bench_legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor=BENCH_COLOR[b],
                           markeredgecolor='black', markersize=9, label=BENCH_LABEL[b])
                    for b in benches if b in sas_k10_per_bench]
    topo_legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=8, label='SAS'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=8, label='Independent'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=8, label='Decentralized'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=7, label='Channel-muting ablations'),
    ]
    fit_legend = [l for l in ax.get_legend_handles_labels()[0]
                  if hasattr(l, 'get_linestyle') and l.get_linestyle() == '--'][:1]

    leg1 = ax.legend(handles=bench_legend, loc='upper left',
                     fontsize=9, framealpha=0.95, title='Benchmark',
                     title_fontsize=10)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=topo_legend, loc='upper left', bbox_to_anchor=(0.0, 0.78),
                     fontsize=9, framealpha=0.95, title='Topology',
                     title_fontsize=10)
    ax.add_artist(leg2)
    if fit_legend:
        ax.legend(handles=fit_legend, loc='lower right', fontsize=9, framealpha=0.95)

    fig.text(0.5, -0.02,
             'Lines connect k-values within each (benchmark, topology) — '
             'within-benchmark accuracy saturates as k grows, but each saturation '
             'plateau lies along the cross-benchmark unified curve.\n'
             'Easy benchmarks plateau early at low energy / low ΔAcc; harder ones '
             'extend further. Stitched together, they form the unified law.',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444')

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
