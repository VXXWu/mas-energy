"""Finding #4 figure: iteration-required mechanism axis.

Shows independent_share's paired ΔAcc per benchmark, with the predictability
axis (predictable retrieval vs unpredictable environment feedback) as the
mechanism explanation. SWE-bench (unpredictable env feedback) is the only
collapse case; BrowseComp+ (predictable retrieval, long tool returns) is the
negative control that does NOT collapse.
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
PHASE_B2 = 'mas-energy/results/a5000_phase_b2_terse'

BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}

# Predictability classification per benchmark (mechanism axis from finding #4)
MECHANISM = {
    'qampari':         'predictable',  # Wikipedia retrieval (training-set-like corpus)
    'fanoutqa':        'predictable',  # entity attribute retrieval
    'math':            'predictable',  # symbolic reasoning, no novel feedback
    'workbench':       'predictable',  # typed tool returns (CRM records)
    'browsecomp_plus': 'predictable',  # retrieval (long, but informationally redundant)
    'swebench':        'unpredictable',  # unit-test pass/fail = environment state
}
COLOR = {'predictable': '#2c7fb8', 'unpredictable': '#a50f15'}


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


def paired_delta(bench, topo='independent_share'):
    # Locate baseline + ablation files
    base_dir = Path(BASE_ROOT) / bench
    abl_dir  = Path(PHASE_B2) / bench
    if not base_dir.exists() or not abl_dir.exists():
        return None
    base_files = list(base_dir.glob('*decentralized_k*.jsonl'))
    abl_files  = list(abl_dir.glob(f'*_{topo}_k*.jsonl'))
    if not base_files or not abl_files:
        return None
    base = {}
    for f in base_files:
        for r in load(f): base[(r['task_id'], r.get('rep', 0))] = r
    abl = {}
    for f in abl_files:
        for r in load(f): abl[(r['task_id'], r.get('rep', 0))] = r
    keys = sorted(set(base) & set(abl))
    if len(keys) < 5: return None
    deltas = [acc(abl[k]) - acc(base[k]) for k in keys]
    lo, hi = bootstrap_ci(deltas)
    return dict(n=len(keys), delta=mean(deltas), ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_iteration_mechanism.png')
    args = ap.parse_args()

    benches = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus', 'swebench']
    rows = []
    for b in benches:
        r = paired_delta(b)
        if r is None:
            print(f"  no data for {b}"); continue
        r['bench'] = b
        rows.append(r)

    # Order: predictable (left), unpredictable (right). Within group, sort by mean ΔAcc.
    pred = sorted([r for r in rows if MECHANISM[r['bench']]=='predictable'],
                  key=lambda r: r['delta'], reverse=True)
    unpred = sorted([r for r in rows if MECHANISM[r['bench']]=='unpredictable'],
                    key=lambda r: r['delta'], reverse=True)
    ordered = pred + unpred

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(ordered))[::-1]   # top-down

    for i, r in enumerate(ordered):
        b = r['bench']
        c = COLOR[MECHANISM[b]]
        delta = r['delta']
        lo, hi = r['ci_lo'], r['ci_hi']
        ax.barh(y[i], delta, color=c, edgecolor='black', linewidth=0.8,
                alpha=0.85, height=0.65)
        # Error bars
        if lo is not None:
            ax.errorbar(delta, y[i], xerr=[[delta-lo], [hi-delta]],
                        fmt='none', ecolor='black', linewidth=1.2, capsize=4, zorder=5)
        # Numeric label
        x_lab = delta + 0.025 if delta >= 0 else delta - 0.025
        ha = 'left' if delta >= 0 else 'right'
        ax.text(x_lab, y[i],
                f'{delta:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  n={r["n"]}',
                va='center', ha=ha, fontsize=9)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(-0.05, len(ordered)+0.3, 'pre-committed\ndecision threshold (−0.05)',
            ha='center', va='bottom', fontsize=8, style='italic', color='gray')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{BENCH_LABEL[r["bench"]]}\n({MECHANISM[r["bench"]]})'
                        for r in ordered], fontsize=10)
    ax.set_xlabel('Paired ΔAcc:  independent_share vs full Decent (R=2)', fontsize=11)
    ax.set_title('Finding #4 — Iteration is load-bearing iff tool returns are unpredictable\n'
                 'Dropping debate rounds: null on predictable-retrieval tasks, collapses on SWE-bench',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.7, 0.15)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Legend explaining mechanism axis
    from matplotlib.patches import Patch
    handles = [
        Patch(color=COLOR['predictable'],
              label='Predictable retrieval (rounds wasted; iteration not load-bearing)'),
        Patch(color=COLOR['unpredictable'],
              label='Unpredictable environment feedback (rounds load-bearing)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.95)

    fig.text(0.5, -0.03,
             'Mechanism: rounds are load-bearing iff tool returns supply information the agent could not predict from its prior. '
             'BrowseComp+ has long retrievals (~corpus chunks) but is informationally redundant — it does NOT collapse, '
             'serving as the negative control on the surface "long tool returns" property. SWE-bench is the only positive '
             'case: unit-test pass/fail is unpredictable environment state.',
             ha='center', va='top', fontsize=8.5, style='italic', color='#444', wrap=True)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")
    print(f"  benchmarks: {[(r['bench'], f'{r['delta']:+.3f}') for r in ordered]}")


if __name__ == "__main__":
    main()
