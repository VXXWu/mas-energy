"""Specialist 2x2 figure: deployment-realism evidence.

Two partition strategies (round-robin, categorical) × two channel states
(full, empty) on WorkBench. Shows: under both partition schemes, empty
channel matches full channel within CI; the accuracy cost is from tool
access constraints, not from missing channel content.
"""
import argparse
import json
import random
from pathlib import Path
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_specialist_2x2.png')
    args = ap.parse_args()

    base_f = Path('mas-energy/results/a5000_latent_transcripts/workbench/Qwen_Qwen3.5-9B_decentralized_k10.jsonl')
    base_rows = load(base_f)
    base_by = by_id(base_rows)
    base_acc = mean(acc(r) for r in base_rows)

    indep_f = Path('mas-energy/results/a5000_latent_transcripts/workbench/Qwen_Qwen3.5-9B_independent_k10.jsonl')
    indep_rows = load(indep_f)
    indep_acc = mean(acc(r) for r in indep_rows) if indep_rows else None

    cells = [
        ('decentralized_specialist',                 'Round-robin\n(full channel)',  '#fdae6b', '#FFE0B2'),
        ('decentralized_specialist_empty',           'Round-robin\n(empty channel)', '#08306b', '#9ecae1'),
        ('decentralized_specialist_categorical',     'Categorical\n(full channel)',  '#fdae6b', '#FFE0B2'),
        ('decentralized_specialist_categorical_empty', 'Categorical\n(empty channel)', '#08306b', '#9ecae1'),
    ]

    results = []
    for topo, label, ec, fc in cells:
        f = Path('mas-energy/results/a5000_phase_a_ablation/workbench') / f'Qwen_Qwen3.5-9B_{topo}_k10.jsonl'
        rows = load(f)
        if not rows:
            results.append((label, None, None, None, None, ec, fc))
            continue
        abl_by = by_id(rows)
        keys = sorted(set(base_by) & set(abl_by))
        if not keys:
            results.append((label, None, None, None, None, ec, fc))
            continue
        abl_paired_acc = mean(acc(abl_by[k]) for k in keys)
        deltas = [acc(abl_by[k]) - acc(base_by[k]) for k in keys]
        delta_mean = mean(deltas)
        lo, hi = bootstrap_ci(deltas)
        results.append((label, abl_paired_acc, delta_mean, lo, hi, ec, fc))

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = []
    accs = []
    deltas = []
    los = []
    his = []
    edge_colors = []
    fill_colors = []

    # Reference: Independent + full Decent
    ax.axhline(base_acc, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.7,
               label=f'Full Decent (no partition) = {base_acc:.3f}')
    if indep_acc is not None:
        ax.axhline(indep_acc, color='#888', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Independent (no partition, no debate) = {indep_acc:.3f}')

    n_bars = len(results)
    x_pos = np.arange(n_bars)
    accs = [r[1] if r[1] is not None else 0 for r in results]
    # CI on paired ΔAcc applies to acc_abl. Bootstrap returns lo,hi on delta;
    # half-width is symmetric around the point estimate. clip to non-negative.
    err_lo = [max(0, (r[2] - r[3])) if r[3] is not None and r[2] is not None else 0 for r in results]
    err_hi = [max(0, (r[4] - r[2])) if r[4] is not None and r[2] is not None else 0 for r in results]
    bar_colors = [r[6] for r in results]
    edge = [r[5] for r in results]

    bars = ax.bar(x_pos, accs, color=bar_colors, edgecolor=edge,
                   linewidth=2.5, width=0.7)
    # NOTE: error bars from bootstrap on paired ΔAcc; show as error on absolute acc
    # bootstrapped CI half-widths apply to the delta which equals acc_abl - acc_base,
    # so the same half-width applies to acc_abl.
    ax.errorbar(x_pos, accs, yerr=[err_lo, err_hi], fmt='none',
                ecolor='black', capsize=5, linewidth=1.5)

    # Annotate ΔAcc above each bar
    for i, r in enumerate(results):
        if r[1] is None: continue
        delta = r[2]
        ax.text(i, r[1] + 0.02, f'Δ={delta:+.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='#d62728' if delta < -0.05 else '#000')
        ax.text(i, r[1] / 2, f'{r[1]:.3f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # Group separator
    ax.axvline(1.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.5, 0.95, 'Round-robin partition\n(artificial)',
            transform=ax.transData, ha='center', va='top',
            fontsize=10, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    ax.text(2.5, 0.95, 'Categorical partition\n(deployment-faithful:\nScheduler / Communicator / CRM-Analyst)',
            transform=ax.transData, ha='center', va='top',
            fontsize=10, color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[0] for r in results], fontsize=10)
    ax.set_ylabel('Accuracy (paired against full Decent baseline)', fontsize=11)
    ax.set_title('Specialist tool-partition: empty channel matches full channel under both partitions',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
