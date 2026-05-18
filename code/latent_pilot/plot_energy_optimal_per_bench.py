"""Finding #5 figure: energy-optimal topology per benchmark.

Two-panel layout:
  (left)  ΔE % savings of the energy-optimal topology vs full Decent for each benchmark
          (color-coded by which topology was optimal: independent_share vs terse vs IS+minimal)
  (right) ΔAcc CI for the same configuration vs full Decent — confirming accuracy preserved
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


BASE_ROOT = 'mas-energy/results/a5000_latent_transcripts'
PHASE_A   = 'mas-energy/results/a5000_phase_a_ablation'
PHASE_B2  = 'mas-energy/results/a5000_phase_b2_terse'

BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}

# Per-benchmark energy-optimal cell (cheapest topology with ΔAcc CI overlapping zero,
# from project audit; matches finding #5 in the paper).
OPTIMAL = {
    'qampari':         ('independent_share_minimal', PHASE_B2, 'IS + minimal'),
    'fanoutqa':        ('independent_share',          PHASE_B2, 'independent_share'),
    'math':            ('independent_share_minimal', PHASE_B2, 'IS + minimal'),
    'workbench':       ('independent_share_minimal', PHASE_B2, 'IS + minimal'),
    'browsecomp_plus': ('independent_share',          PHASE_B2, 'independent_share'),
    'swebench':        ('decentralized_terse',        PHASE_B2, 'terse'),
}

COLOR = {
    'IS + minimal':       '#a50f15',
    'independent_share':  '#fb6a4a',
    'terse':              '#fdae6b',
}


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def acc(r):
    if r.get('loose_accuracy') is not None: return float(r['loose_accuracy'])
    return 1.0 if r.get('correct') else 0.0


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=42):
    if len(deltas) < 5: return (None, None)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randint(0, n-1)] for _ in range(n))/n for _ in range(n_boot))
    return means[int(alpha/2 * n_boot)], means[int((1-alpha/2) * n_boot)]


def paired(bench, topo, abl_root):
    base_dir = Path(BASE_ROOT) / bench
    abl_dir  = Path(abl_root) / bench
    if not base_dir.exists() or not abl_dir.exists():
        return None
    base_files = list(base_dir.glob('*decentralized_k*.jsonl'))
    abl_files  = list(abl_dir.glob(f'*_{topo}_k*.jsonl'))
    if not base_files or not abl_files: return None
    base = {}
    for f in base_files:
        for r in load(f): base[(r['task_id'], r.get('rep', 0))] = r
    abl = {}
    for f in abl_files:
        for r in load(f): abl[(r['task_id'], r.get('rep', 0))] = r
    keys = sorted(set(base) & set(abl))
    if len(keys) < 5: return None
    deltas_acc = [acc(abl[k]) - acc(base[k]) for k in keys]
    deltas_e   = [(abl[k].get('gpu_dynamic_energy_joules', 0) or 0)
                  - (base[k].get('gpu_dynamic_energy_joules', 0) or 0) for k in keys]
    base_E = mean((base[k].get('gpu_dynamic_energy_joules', 0) or 0) for k in keys)
    abl_E  = mean((abl[k].get('gpu_dynamic_energy_joules', 0) or 0) for k in keys)
    lo, hi = bootstrap_ci(deltas_acc)
    return dict(n=len(keys), delta_acc=mean(deltas_acc),
                ci_lo=lo, ci_hi=hi,
                delta_E=mean(deltas_e), base_E=base_E, abl_E=abl_E,
                pct_E=100*(abl_E - base_E)/base_E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_energy_optimal_per_bench.png')
    args = ap.parse_args()

    benches = list(OPTIMAL.keys())
    rows = []
    for b in benches:
        topo, root, label = OPTIMAL[b]
        r = paired(b, topo, root)
        if r is None:
            print(f"  no data for {b}/{topo}")
            continue
        r['bench'] = b
        r['topo_label'] = label
        rows.append(r)

    rows.sort(key=lambda r: r['pct_E'])  # most negative (largest savings) first

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6),
                                    gridspec_kw=dict(width_ratios=[1.1, 1.0], wspace=0.4))

    y = np.arange(len(rows))[::-1]

    # ─── Left panel: Energy savings (% of Decent baseline) ───
    for i, r in enumerate(rows):
        c = COLOR[r['topo_label']]
        axL.barh(y[i], r['pct_E'], color=c, edgecolor='black', linewidth=0.8,
                 alpha=0.85, height=0.7)
        # Numeric label
        x_lab = r['pct_E'] - 1 if r['pct_E'] < 0 else r['pct_E'] + 1
        ha = 'right' if r['pct_E'] < 0 else 'left'
        axL.text(x_lab, y[i], f'{r["pct_E"]:+.1f}%   (Decent {r["base_E"]/1000:.1f} kJ → {r["abl_E"]/1000:.1f} kJ)',
                 va='center', ha=ha, fontsize=9)
    axL.axvline(0, color='black', linewidth=0.8)
    axL.set_yticks(y)
    axL.set_yticklabels([BENCH_LABEL[r['bench']] for r in rows], fontsize=10)
    axL.set_xlabel('Energy ΔE  (% of Decent baseline)', fontsize=11)
    axL.set_title('(A) Energy savings of energy-optimal topology', fontsize=11.5, fontweight='bold')
    axL.set_xlim(-95, 25)
    axL.grid(True, axis='x', alpha=0.3)
    axL.spines['top'].set_visible(False); axL.spines['right'].set_visible(False)

    # Right panel: ΔAcc CI of the same config
    for i, r in enumerate(rows):
        c = COLOR[r['topo_label']]
        d = r['delta_acc']
        lo, hi = r['ci_lo'], r['ci_hi']
        axR.scatter(d, y[i], s=140, color=c, edgecolor='black', linewidth=1.0,
                    zorder=5)
        if lo is not None:
            axR.errorbar(d, y[i], xerr=[[d-lo], [hi-d]],
                         fmt='none', ecolor='black', linewidth=1.2, capsize=4, zorder=4)
        axR.text(d + 0.005, y[i], f'{d:+.3f}', va='center', ha='left', fontsize=9)

    axR.axvline(0, color='black', linewidth=0.8, alpha=0.7)
    axR.axvspan(-0.05, 0.05, color='lightgray', alpha=0.2, zorder=1)
    axR.text(0, len(rows)-0.4, 'practical-equivalence band',
             ha='center', va='bottom', fontsize=8, style='italic', color='gray')

    axR.set_yticks(y)
    axR.set_yticklabels([])  # share with left panel
    axR.set_xlabel('Paired ΔAcc vs Decent', fontsize=11)
    axR.set_title('(B) Accuracy preserved (CI95)', fontsize=11.5, fontweight='bold')
    axR.set_xlim(-0.12, 0.18)
    axR.grid(True, axis='x', alpha=0.3)
    axR.spines['top'].set_visible(False); axR.spines['right'].set_visible(False)

    # Annotate which topology each row is using (right of right panel)
    for i, r in enumerate(rows):
        axR.text(0.17, y[i], r['topo_label'], va='center', ha='right',
                 fontsize=8.5, color=COLOR[r['topo_label']],
                 fontweight='bold')

    fig.suptitle('Finding #5 — Energy-optimal topology is per-benchmark, not universal',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.04,
             'Each benchmark\'s energy-optimal topology delivers 5-90% energy reduction at zero or near-zero accuracy cost.  '
             'Five prose benchmarks favor IS-family interventions (drop debate rounds); SWE-bench requires structural rounds, '
             'so only speaker-side reduction (terse) saves energy without collapsing accuracy.',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444', wrap=True)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")
    for r in rows:
        print(f"  {r['bench']:<18} {r['topo_label']:<22} ΔE={r['pct_E']:+5.1f}%  ΔAcc={r['delta_acc']:+.3f}")


if __name__ == "__main__":
    main()
