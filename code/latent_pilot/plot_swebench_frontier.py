"""SWE-bench-only frontier figure.

Plots the (energy, accuracy) Pareto frontier across SAS, Independent, and
Decentralized k-scans (Centralized excluded — different mechanism). Pareto-
optimal points are highlighted; a log-linear fit is overlaid on the Pareto
subset.

IMPORTANT — what this figure does NOT show:
The Pareto log-linear (slope ≈ +0.42/dec, R² ≈ 0.92 on swebench) is a
*derived consequence* of the per-topology saturation curves, not an
independent scaling law. Predicting the Pareto envelope from per-topology
saturation fits alone (no Pareto data used) yields slope=+0.41, R²=0.98 —
matching the observed Pareto fit within 0.01 on slope. Mechanism:
acc(E) = ceiling − drop·E^(−α) is approximately log-linear in E whenever α
is small and we operate far from the ceiling; the Pareto envelope picks
the topology that is farthest from its ceiling at each E, so the envelope
lives in the slow-approach regime where saturation looks log-linear. The
genuine law is the per-topology saturation form (R²≈0.90-0.97 each on
swebench); the Pareto log-linear is a useful summary of the engineer-
optimal frontier but should NOT be sold as a new scaling phenomenon.

Why swebench-only: the cross-benchmark unified figure averages across 6
benchmarks of varying difficulty, which forces per-bench normalization and
introduces averaging structure. Swebench in isolation has the deepest k-scan
we have (k up to 800 for SAS, 200 for Independent, 50 for Decentralized).

Usage: python plot_swebench_frontier.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


SWEBENCH_DIR = Path('mas-energy/results/a5000_swebench')

TOPO_STYLE = {
    'sas':           dict(color='#1f77b4', marker='o', label='SAS'),
    'independent':   dict(color='#ff7f0e', marker='s', label='Independent'),
    'decentralized': dict(color='#d62728', marker='D', label='Decentralized R=2'),
}


def parse_filename(p: Path):
    """Qwen_Qwen3.5-9B_<topo>_k<k>[_R<R>].jsonl → (topo, k, R)."""
    s = p.stem
    if not s.startswith('Qwen_Qwen3.5-9B_'):
        return None
    rest = s[len('Qwen_Qwen3.5-9B_'):]
    if '_R' in rest:
        topo_k, R_part = rest.rsplit('_R', 1)
        try: R = int(R_part)
        except ValueError: return None
    else:
        topo_k, R = rest, None
    if '_k' not in topo_k:
        return None
    topo, k_part = topo_k.rsplit('_k', 1)
    try: k = int(k_part)
    except ValueError: return None
    return topo, k, R


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return float(bool(r.get('correct')))


def load_cells():
    cells = []
    for path in sorted(SWEBENCH_DIR.glob('Qwen_Qwen3.5-9B_*.jsonl')):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        topo, k, R = parsed
        if topo not in TOPO_STYLE:
            continue
        rows = [json.loads(l) for l in path.open()]
        if not rows:
            continue
        # Default R for non-decentralized topologies
        if R is None and topo == 'decentralized':
            R = 2
        accs = [acc(r) for r in rows]
        es = [r.get('gpu_dynamic_energy_joules') or 0 for r in rows]
        cells.append({
            'topo': topo, 'k': k, 'R': R, 'n': len(rows),
            'acc': mean(accs), 'energy': mean(es),
        })
    return cells


def pareto_filter(cells):
    """Return cells that are Pareto-optimal: no other cell has lower energy
    AND higher accuracy. Maximizing accuracy, minimizing energy."""
    pareto = []
    for c in cells:
        dominated = False
        for o in cells:
            if o is c: continue
            if o['energy'] <= c['energy'] and o['acc'] >= c['acc'] and (
                o['energy'] < c['energy'] or o['acc'] > c['acc']):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    return pareto


def fit_loglinear(cells):
    valid = [c for c in cells if c['energy'] > 0]
    if len(valid) < 3:
        return None
    xs = np.log10([c['energy'] for c in valid])
    ys = np.array([c['acc'] for c in valid])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    ss_res = ((ys - pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, intercept, r2, valid


def main():
    cells = load_cells()
    print(f'Loaded {len(cells)} swebench cells (SAS/Indep/Decent only).')
    print(f'{"topo":<16} {"k":>5} {"R":>3} {"n":>5} {"acc":>8} {"E (J)":>10}')
    print('-' * 55)
    for c in sorted(cells, key=lambda c: (c['topo'], c['k'])):
        R_s = '-' if c['R'] is None else c['R']
        print(f'{c["topo"]:<16} {c["k"]:>5} {str(R_s):>3} {c["n"]:>5} {c["acc"]:>8.3f} {c["energy"]:>10.0f}')

    pareto = pareto_filter(cells)
    print(f'\n{len(pareto)} Pareto-optimal cells:')
    for c in sorted(pareto, key=lambda c: c['energy']):
        print(f'  {c["topo"]:<16} k={c["k"]:>3} → acc={c["acc"]:.3f}  E={c["energy"]:>9.0f} J')

    fit_pareto = fit_loglinear(pareto)
    fit_all = fit_loglinear(cells)
    if fit_pareto:
        s, b, r2, _ = fit_pareto
        print(f'\nPareto log-linear (n={len(pareto)}): acc = {s:+.3f}·log10(E) + {b:+.3f}, R²={r2:.3f}')
    if fit_all:
        s, b, r2, _ = fit_all
        print(f'All-cells log-linear (n={len(cells)}): acc = {s:+.3f}·log10(E) + {b:+.3f}, R²={r2:.3f}')

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    pareto_set = {(c['topo'], c['k']) for c in pareto}
    for topo in ('sas', 'independent', 'decentralized'):
        topo_cells = [c for c in cells if c['topo'] == topo]
        if not topo_cells:
            continue
        s = TOPO_STYLE[topo]
        # Non-Pareto cells: lighter
        non_pareto = [c for c in topo_cells if (c['topo'], c['k']) not in pareto_set]
        if non_pareto:
            ax.scatter([c['energy'] for c in non_pareto],
                       [c['acc'] for c in non_pareto],
                       s=60, color=s['color'], marker=s['marker'],
                       edgecolor='gray', linewidth=0.4, alpha=0.35, zorder=3)
        # Pareto cells: bold
        pareto_cells = [c for c in topo_cells if (c['topo'], c['k']) in pareto_set]
        if pareto_cells:
            ax.scatter([c['energy'] for c in pareto_cells],
                       [c['acc'] for c in pareto_cells],
                       s=110, color=s['color'], marker=s['marker'],
                       edgecolor='black', linewidth=1.0, alpha=0.95, zorder=5,
                       label=s['label'])

    # Annotate every Pareto cell with its k
    for c in pareto:
        topo = c['topo']
        col = TOPO_STYLE[topo]['color']
        ax.annotate(f'k={c["k"]}', xy=(c['energy'], c['acc']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=8.5, color=col,
                    fontweight='bold')

    # Pareto fit line — labeled as derived, not independent
    if fit_pareto:
        s, b, r2, valid = fit_pareto
        x_lo = min(c['energy'] for c in valid)
        x_hi = max(c['energy'] for c in valid)
        x_grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 100)
        y_grid = s * np.log10(x_grid) + b
        ax.plot(x_grid, y_grid, color='black', linewidth=1.6, linestyle='--',
                alpha=0.7, zorder=4,
                label=f'Pareto envelope (derived):  acc ≈ {s:+.2f}·log₁₀(E) + {b:+.2f}   (R²={r2:.2f})')

    ax.set_xscale('log')
    ax.set_xlabel('Mean GPU dynamic energy per task (J, log scale)', fontsize=11)
    ax.set_ylabel('Mean accuracy (loose / pass@1)', fontsize=11)
    ax.set_title(
        'SWE-bench Pareto envelope across SAS / Independent / Decentralized\n'
        'Envelope is log-linear (slope +0.42, R²=0.92), but this is a *derived* '
        'consequence\nof the per-topology saturation curves — not an independent '
        'scaling law.',
        fontsize=10.5
    )

    # In-figure caveat box — make sure the reader doesn't take the line at face value
    ax.text(0.02, 0.97,
            'Caveat: Pareto envelope = upper hull of three saturation curves.\n'
            'Predicted from per-topology fits alone (no Pareto data used):\n'
            '  slope = +0.41, R² = 0.98 — matches observed within 0.01.\n'
            'The genuine law is per-topology saturation; this line is derived.',
            transform=ax.transAxes, fontsize=8.0, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.92))
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

    out = Path('figures/frontier/fig_frontier_swebench.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
