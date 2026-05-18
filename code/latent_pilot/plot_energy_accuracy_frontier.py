"""Energy-accuracy frontier figure: per-mode aggregate energy savings vs
accuracy preservation. Shows the deployment trade-off space.
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

MODES = [
    ('decentralized_answer_only',          'answer-only',          'o', '#9ecae1'),
    ('decentralized_truncate100',          'truncate-100',         'o', '#3182bd'),
    ('decentralized_empty',                'empty (channel muted)','D', '#08306b'),
    ('decentralized_terse',                'terse-speaker',        's', '#fdae6b'),
    ('decentralized_terse_answer_only',    'terse + answer-only',  's', '#e6550d'),
    ('independent_share',                  'independent + share',  '^', '#a50f15'),
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
    ap.add_argument('--out', default='figures/fig_energy_accuracy_frontier.png')
    args = ap.parse_args()

    # Aggregate paired ΔAcc and ΔE across all benchmarks per mode
    aggregate = {topo: {'da': [], 'de': []} for topo, _, _, _ in MODES}
    for bench in BENCHES:
        base_f = find_jsonl(ROOTS['baseline'], bench, 'decentralized')
        if not base_f: continue
        base_by = by_id(load(base_f))
        for topo, label, marker, color in MODES:
            abl_f = find_jsonl(root_for_topo(topo), bench, topo)
            if not abl_f: continue
            abl_by = by_id(load(abl_f))
            keys = sorted(set(base_by) & set(abl_by))
            if len(keys) < 5: continue
            for k in keys:
                aggregate[topo]['da'].append(acc(abl_by[k]) - acc(base_by[k]))
                aggregate[topo]['de'].append(
                    ((abl_by[k].get('gpu_dynamic_energy_joules') or 0) -
                     (base_by[k].get('gpu_dynamic_energy_joules') or 0)) / 1000
                )

    fig, ax = plt.subplots(figsize=(11, 7))

    # Quadrant shading: top-left (negative ΔE = saves energy, positive ΔAcc = better)
    # is the Pareto-improving region.
    ax.axvspan(-30, 0, ymin=0.5, ymax=1.0, alpha=0.06, color='green',
               zorder=0)
    ax.axvspan(0, 5, ymin=0, ymax=0.5, alpha=0.06, color='red', zorder=0)

    ax.axhline(0, color='black', linewidth=1.0, alpha=0.6, zorder=1)
    ax.axvline(0, color='black', linewidth=1.0, alpha=0.6, zorder=1)

    # Collect all results to position labels avoiding overlap
    points = []
    for topo, label, marker, color in MODES:
        d = aggregate[topo]
        if not d['da']: continue
        n = len(d['da'])
        m_da = mean(d['da'])
        m_de = mean(d['de'])
        lo_da, hi_da = bootstrap_ci(d['da'])
        lo_de, hi_de = bootstrap_ci(d['de'])
        points.append({
            'topo': topo, 'label': label, 'marker': marker, 'color': color, 'n': n,
            'da': m_da, 'de': m_de, 'da_lo': lo_da, 'da_hi': hi_da,
            'de_lo': lo_de, 'de_hi': hi_de,
        })

    # Plot full Decent reference at origin
    ax.scatter(0, 0, marker='*', s=400, color='#2ca02c', edgecolor='black',
               linewidth=1.5, zorder=10)
    ax.annotate('full Decent\n(baseline)', (0, 0),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#2ca02c')

    # Plot each mode with hand-tuned label offsets to avoid overlap
    label_offsets = {
        'decentralized_answer_only':         (8, -16),
        'decentralized_truncate100':         (8, 14),
        'decentralized_empty':               (-8, 14),
        'decentralized_terse':               (-12, -18),
        'decentralized_terse_answer_only':   (-12, 14),
        'independent_share':                 (12, -4),
    }
    for p in points:
        xerr = [[p['de'] - p['de_lo']], [p['de_hi'] - p['de']]] if p['de_lo'] is not None else None
        yerr = [[p['da'] - p['da_lo']], [p['da_hi'] - p['da']]] if p['da_lo'] is not None else None
        ax.errorbar(p['de'], p['da'], xerr=xerr, yerr=yerr, marker=p['marker'],
                    color=p['color'], markersize=15, markeredgecolor='black',
                    markeredgewidth=1.2, capsize=4, linewidth=1.5,
                    label=f"{p['label']}  (n={p['n']})", zorder=5)
        dx, dy = label_offsets.get(p['topo'], (8, 8))
        ha = 'left' if dx > 0 else 'right'
        ax.annotate(p['label'], (p['de'], p['da']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9.5, color=p['color'], fontweight='bold',
                    ha=ha)

    # Set limits with breathing room
    all_de = [p['de'] for p in points]
    all_da = [p['da'] for p in points]
    ax.set_xlim(min(all_de + [0]) - 4, max(all_de + [0]) + 5)
    ax.set_ylim(min(all_da + [0]) - 0.025, max(all_da + [0]) + 0.045)

    # Quadrant labels (corners)
    ax.text(0.02, 0.96, 'Pareto-improving\n(saves energy + accuracy ≥ baseline)',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='#2a8a2a', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0fff0', ec='#2a8a2a', alpha=0.85))
    ax.text(0.98, 0.04, 'Worse on both axes',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#cc4444', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff0f0', ec='#cc4444', alpha=0.85))

    ax.set_xlabel('Δ Energy vs full Decent  (kJ/task)   ←   more savings', fontsize=11)
    ax.set_ylabel('Δ Accuracy vs full Decent  (paired)', fontsize=11)
    ax.set_title('Energy–accuracy frontier across 6 benchmarks (aggregate, paired)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95, ncol=1)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
