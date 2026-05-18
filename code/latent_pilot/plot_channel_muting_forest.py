"""Headline figure: forest plot showing paired ΔAcc for `empty`, `terse`,
and `independent_share` modes across benchmarks with 95% CIs.

Establishes: channel content is null universally (`empty` ΔAcc straddles 0
on every benchmark; aggregate CI tightly excludes ±0.03). The other modes
show the practical-deployment recommendations.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


def load(p):
    rows = []
    if not Path(p).exists(): return rows
    for line in open(p):
        try:
            d = json.loads(line)
            if d.get('error') and 'answer' not in d: continue
            rows.append(d)
        except: pass
    return rows


def acc(r):
    la = r.get('loose_accuracy')
    if la is not None: return float(la)
    return 1.0 if r.get('correct') else 0.0


def by_id(rows): return {(r['task_id'], r.get('rep', 0)): r for r in rows}


def find_jsonl(root, bench, topo):
    p = Path(root) / bench
    if not p.exists(): return None
    matches = list(p.glob(f'*_{topo}_k*.jsonl'))
    return matches[0] if matches else None


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=42):
    if len(deltas) < 5: return (None, None)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randint(0, n-1)] for _ in range(n))/n for _ in range(n_boot))
    return means[int(alpha/2 * n_boot)], means[int((1-alpha/2) * n_boot)]


BENCHES = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus', 'swebench']
BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}

MODES = [
    ('decentralized_empty',     'empty (channel muted)',  '#2c7fb8'),
    ('decentralized_terse',     'terse-speaker',           '#fdae6b'),
    ('independent_share',       'independent + share',     '#a50f15'),
]

ROOTS = {
    'baseline': 'mas-energy/results/a5000_latent_transcripts',
    'phase_a':  'mas-energy/results/a5000_phase_a_ablation',
    'phase_b2': 'mas-energy/results/a5000_phase_b2_terse',
}


def root_for_topo(topo):
    if topo in ('independent_share', 'decentralized_terse', 'decentralized_terse_answer_only'):
        return ROOTS['phase_b2']
    return ROOTS['phase_a']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_channel_muting_forest.png')
    args = ap.parse_args()

    # Compute per-benchmark per-mode paired ΔAcc + CI
    per_mode = defaultdict(list)  # mode -> list of (bench, n, delta, lo, hi)
    aggregate = {topo: [] for topo, _, _ in MODES}

    for bench in BENCHES:
        base_f = find_jsonl(ROOTS['baseline'], bench, 'decentralized')
        if not base_f: continue
        base_by = by_id(load(base_f))
        for topo, label, _ in MODES:
            abl_f = find_jsonl(root_for_topo(topo), bench, topo)
            if not abl_f: continue
            abl_by = by_id(load(abl_f))
            keys = sorted(set(base_by) & set(abl_by))
            if len(keys) < 5: continue
            deltas = [acc(abl_by[k]) - acc(base_by[k]) for k in keys]
            lo, hi = bootstrap_ci(deltas)
            per_mode[topo].append((bench, len(keys), mean(deltas), lo, hi))
            aggregate[topo].extend(deltas)

    # 3 horizontal panels, one per mode
    fig, axes = plt.subplots(1, len(MODES), figsize=(15, 5.5), sharey=True)
    fig.suptitle('Channel-muting decomposition: paired ΔAcc vs full Decent across benchmarks',
                 fontsize=13, fontweight='bold')

    y_positions = list(range(len(BENCHES) + 2))  # +2 for spacer + aggregate row
    bench_y = list(range(len(BENCHES)))
    spacer_y = len(BENCHES)
    agg_y = len(BENCHES) + 1

    for ax_idx, (topo, label, color) in enumerate(MODES):
        ax = axes[ax_idx]
        # Build a list keyed by bench order so missing benches show up as gaps
        d_by_bench = {b: (n, d, lo, hi) for b, n, d, lo, hi in per_mode[topo]}

        for i, bench in enumerate(BENCHES):
            if bench in d_by_bench:
                n, d, lo, hi = d_by_bench[bench]
                xerr = None
                if lo is not None and hi is not None:
                    xerr = [[d - lo], [hi - d]]
                ax.errorbar(d, i, xerr=xerr, marker='o', color=color, markersize=9,
                            capsize=4, linewidth=2, ecolor=color, alpha=0.95)
                ax.text(d + (0.01 if d >= 0 else -0.01), i,
                        f'  n={n}', va='center',
                        ha='left' if d >= 0 else 'right',
                        fontsize=8, color='#444')
            else:
                ax.text(0, i, '(no data)', va='center', ha='center',
                        fontsize=8, color='#aaa', style='italic')

        # Aggregate row
        agg_deltas = aggregate[topo]
        if agg_deltas:
            n_agg = len(agg_deltas)
            mean_agg = mean(agg_deltas)
            lo_agg, hi_agg = bootstrap_ci(agg_deltas)
            xerr = [[mean_agg - lo_agg], [hi_agg - mean_agg]] if lo_agg is not None else None
            ax.errorbar(mean_agg, agg_y, xerr=xerr, marker='D', color='black',
                        markersize=11, capsize=5, linewidth=2.5, ecolor='black')
            ax.text(mean_agg + (0.01 if mean_agg >= 0 else -0.01), agg_y,
                    f'  n={n_agg}', va='center',
                    ha='left' if mean_agg >= 0 else 'right',
                    fontsize=9, color='black', fontweight='bold')

        ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(spacer_y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-0.65, 0.18)
        ax.set_yticks(bench_y + [agg_y])
        ax.set_yticklabels([BENCH_LABEL[b] for b in BENCHES] + ['Aggregate'])
        ax.set_xlabel('Δ Accuracy vs full Decent (paired)', fontsize=10)
        ax.set_title(label, fontsize=11, color=color, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()  # benchmarks top-down, aggregate at bottom

    axes[0].set_ylabel('Benchmark', fontsize=11)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
