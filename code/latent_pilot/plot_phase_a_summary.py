"""Generate the Phase A / B-2 / channel-muting decomposition summary figure.

Two-panel grid:
  Top:    ΔAcc vs full-Decent baseline, per benchmark, per ablation mode (paired)
  Bottom: ΔEnergy vs full-Decent baseline, per benchmark, per ablation mode

Shows the channel-muting decomposition: empty/answer_only/truncate preserve accuracy
on most benchmarks; terse + independent_share also preserve while saving substantial
energy; swebench is the exception where independent_share collapses.
"""
import argparse
import json
import re
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


# ─── Configuration ───
BENCHES = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus', 'swebench']
BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}

MODES = [
    ('decentralized_answer_only',  'answer-only',         '#9ecae1'),
    ('decentralized_truncate300',  'truncate-300',        '#6baed6'),
    ('decentralized_truncate100',  'truncate-100',        '#3182bd'),
    ('decentralized_empty',        'empty (placeholder)', '#08519c'),
    ('decentralized_empty_silent', 'empty-silent (no framing)', '#08306b'),
    ('decentralized_terse',        'terse-speaker',       '#fdae6b'),
    ('decentralized_terse_answer_only', 'terse + answer-only', '#e6550d'),
    ('independent_share',          'independent + share', '#a50f15'),
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
    ap.add_argument('--out', default='figures/phase_a_b2_decomposition.png')
    args = ap.parse_args()

    # Build the data table: bench → mode → (delta_acc, delta_E_kJ, n)
    data = defaultdict(lambda: {})
    for bench in BENCHES:
        base_f = find_jsonl(ROOTS['baseline'], bench, 'decentralized')
        if not base_f: continue
        base_rows = load(base_f)
        base_by = by_id(base_rows)
        if not base_rows: continue
        for topo, label, color in MODES:
            abl_f = find_jsonl(root_for_topo(topo), bench, topo)
            if not abl_f: continue
            abl_rows = load(abl_f)
            if not abl_rows: continue
            abl_by = by_id(abl_rows)
            keys = sorted(set(base_by) & set(abl_by))
            if not keys: continue
            deltas_acc = [acc(abl_by[k]) - acc(base_by[k]) for k in keys]
            deltas_E = [(abl_by[k].get('gpu_dynamic_energy_joules') or 0) -
                        (base_by[k].get('gpu_dynamic_energy_joules') or 0)
                        for k in keys]
            data[bench][topo] = dict(
                delta_acc=mean(deltas_acc),
                delta_E_kJ=mean(deltas_E) / 1000,
                n=len(keys),
            )

    # ─── Plot ───
    fig, (ax_acc, ax_E) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle('Phase A / B-2 / channel-muting decomposition vs full Decent (paired)',
                 fontsize=13, fontweight='bold')

    n_modes = len(MODES)
    bar_w = 0.85 / n_modes
    x_centers = np.arange(len(BENCHES))

    for j, (topo, label, color) in enumerate(MODES):
        offsets = x_centers + (j - n_modes/2 + 0.5) * bar_w
        accs = []
        es = []
        ns = []
        for bench in BENCHES:
            d = data.get(bench, {}).get(topo)
            if d is None:
                accs.append(np.nan); es.append(np.nan); ns.append(0)
            else:
                accs.append(d['delta_acc']); es.append(d['delta_E_kJ']); ns.append(d['n'])
        ax_acc.bar(offsets, accs, width=bar_w, color=color, label=label, edgecolor='black', linewidth=0.4)
        ax_E.bar(offsets, es, width=bar_w, color=color, edgecolor='black', linewidth=0.4)
        # n labels above each ΔAcc bar
        for x, a, n in zip(offsets, accs, ns):
            if not np.isnan(a) and n > 0:
                ax_acc.text(x, a + (0.005 if a >= 0 else -0.020), f'{n}',
                            ha='center', va='bottom' if a >= 0 else 'top', fontsize=6, color='gray')

    ax_acc.axhline(0, color='black', linewidth=0.6)
    ax_acc.set_ylabel('Δ Accuracy vs full Decent (paired)', fontsize=11)
    ax_acc.set_title('(a) Accuracy delta — channel-muting preserves accuracy on most benchmarks; '
                     'independent_share collapses on SWE-bench', fontsize=11)
    ax_acc.legend(fontsize=8, ncol=4, loc='lower left', framealpha=0.9)
    ax_acc.grid(True, alpha=0.3, axis='y')
    ax_acc.set_ylim(-0.6, 0.15)

    ax_E.axhline(0, color='black', linewidth=0.6)
    ax_E.set_ylabel('Δ Energy vs full Decent (kJ/task)', fontsize=11)
    ax_E.set_xticks(x_centers)
    ax_E.set_xticklabels([BENCH_LABEL[b] for b in BENCHES], fontsize=10)
    ax_E.set_title('(b) Energy delta — terse and independent_share save substantial energy; '
                   'empty saves only via reduced peer prefill', fontsize=11)
    ax_E.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved {out}")

    # Also print the summary table to console for inclusion in writeups
    print('\n' + '=' * 100)
    print(f"{'Benchmark':<18}", end='')
    for _, label, _ in MODES:
        print(f"  {label[:18]:<18}", end='')
    print()
    print('-' * 100)
    for bench in BENCHES:
        print(f"{BENCH_LABEL[bench]:<18}", end='')
        for topo, _, _ in MODES:
            d = data.get(bench, {}).get(topo)
            if d is None:
                cell = '   —   '
            else:
                cell = f'{d["delta_acc"]:+.3f}({d["n"]:>3})'
            print(f"  {cell:<18}", end='')
        print()


if __name__ == "__main__":
    main()
