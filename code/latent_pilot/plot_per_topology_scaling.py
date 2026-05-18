"""Per-topology scaling overlay (dual fit form).

Companion to plot_unified_scaling.py. Each topology gets its own fit on the
(energy_ratio, ΔAcc) plane. Two fit forms supported:

  Log-linear (default): ΔAcc = α·log10(E_ratio) + β
    - SAS:           shallow slope, sits ~at the SAS-k=10 origin
    - Independent:   steeper slope, lower intercept (coordination tax)
    - Decentralized: steeper slope, lower intercept
    - R² varies with k cut: at k≥1, Decent R² drops to 0.91 and pooled
      R² is 0.71; at k≥3, pooled R² lifts to 0.83 (k=1,2 are the
      coordination-tax regime where MAS topologies pay overhead they
      have not yet earned back).

  Saturation curve (--saturation): ΔAcc(k) = ceiling − drop·k^(−α)
    - All three R² ≈ 0.97 over full k range — better than log-linear at k≥1
    - Each topology has its own asymptote and approach rate
    - SAS ceiling +0.21 (α=0.41), Decent +0.31 (α=0.53), Independent very
      high but loosely identified ceiling with slow approach (α=0.12)

Use --k-min 3 to exclude the coordination-tax regime from the log-linear
fit, exposing the cross-topology shared axis at slope ~+0.18.
Use --saturation to overlay the saturation curves alongside.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import plot_unified_scaling as p


TOPO_STYLE = {
    'sas':           dict(color='#1f77b4', marker='o', label='SAS'),
    'independent':   dict(color='#ff7f0e', marker='s', label='Independent'),
    'decentralized': dict(color='#d62728', marker='D', label='Decentralized'),
    'centralized':   dict(color='#2ca02c', marker='P', label='Centralized'),
}

ALL_TOPOS = ('sas', 'independent', 'decentralized', 'centralized')


def fit(cells, cost_field='energy_ratio', k_min=1):
    valid = [c for c in cells if c.get(cost_field) and c.get('delta_acc') is not None
             and c['k'] >= k_min]
    if len(valid) < 3:
        return None
    xs = np.log10([c[cost_field] for c in valid])
    ys = np.array([c['delta_acc'] for c in valid])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    ss_res = ((ys - pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, intercept, r2, valid


def fit_saturation(cells):
    """ΔAcc(k) = ceiling - drop * k^(-alpha). Approaches an asymptote as k→∞."""
    from scipy.optimize import curve_fit
    valid = [c for c in cells if c.get('delta_acc') is not None]
    if len(valid) < 4:
        return None
    ks = np.array([c['k'] for c in valid], dtype=float)
    ys = np.array([c['delta_acc'] for c in valid])
    def model(k, ceiling, drop, alpha):
        return ceiling - drop * np.power(k, -alpha)
    try:
        popt, _ = curve_fit(model, ks, ys, p0=[0.3, 0.5, 0.5], maxfev=5000)
    except Exception:
        return None
    pred = model(ks, *popt)
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    ss_res = ((ys - pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return popt, r2, valid, model


def crossover_x(s1, b1, s2, b2):
    """log10(x) where line1 == line2. Returns x (linear, not log)."""
    if abs(s1 - s2) < 1e-9:
        return None
    return 10 ** ((b2 - b1) / (s1 - s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/frontier/fig_per_topology_scaling.png')
    ap.add_argument('--cost-axis',
                    choices=['energy_ratio', 'flops_ratio'],
                    default='energy_ratio')
    ap.add_argument('--prose-only', action='store_true',
                    help='Exclude swebench (which dominates the slope) and fit on '
                         'prose benchmarks only.')
    ap.add_argument('--k-min', type=int, default=1,
                    help='Minimum k for the log-linear fit. Use --k-min 3 to '
                         'exclude the k=1,2 coordination-tax regime where MAS '
                         'topologies pay overhead they have not yet earned back; '
                         'lifts pooled R² substantially without changing slopes.')
    ap.add_argument('--saturation', action='store_true',
                    help='Overlay a saturation-curve fit per topology: '
                         'ΔAcc(k) = ceiling − drop·k^(−α). Each topology has '
                         'its own asymptote and approach rate; R² ~0.97 each, '
                         'higher than log-linear at k≥1.')
    ap.add_argument('--benches', nargs='+', default=None,
                    help='Override benchmark list (e.g. fanoutqa workbench browsecomp_plus swebench)')
    ap.add_argument('--exclude-topos', nargs='+', default=[],
                    help='Topologies to exclude from plot (e.g. centralized)')
    args = ap.parse_args()

    if args.benches:
        benches = args.benches
    else:
        prose = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus']
        benches = prose if args.prose_only else prose + ['swebench']

    # Toggle plot_unified_scaling's centralized exclusion so centralized
    # cells are loaded into main_cells.
    p.INCLUDE_CENT = True
    main_cells = p.gather_main_study(benches)
    abl_cells = p.gather_ablations(benches)
    all_cells = main_cells + abl_cells

    sas_k10 = next((c for c in main_cells
                    if c['topo'] == 'sas' and c['k'] == 10 and c['R'] is None), None)
    sas_k10_per_bench = {b: d['a'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_E_per_bench = {b: d['e'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_T_per_bench = {b: d['t'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_D_per_bench = {b: d['d'] for b, d in sas_k10['per_bench'].items()}
    p.attach_summary(all_cells, sas_k10_per_bench,
                     sas_k10_E_per_bench=sas_k10_E_per_bench,
                     sas_k10_T_per_bench=sas_k10_T_per_bench,
                     sas_k10_D_per_bench=sas_k10_D_per_bench)

    # k-scan only: skip channel-muting and R-scan; keep centralized so it can
    # be plotted alongside the other topologies (mechanism is genuinely
    # different but visualizing it on the same plot lets us see how far it
    # sits from the SAS / Indep / Decent saturation curves). Dedupe on
    # (topo, k) since centralized has both R=None and R=2 entries that
    # encode the same canonical experiment.
    excluded_topos = set(args.exclude_topos)
    groups = {}
    seen = set()
    for c in all_cells:
        if c.get('family') == 'ablation':
            continue
        if c.get('R') is not None and c['R'] != 2:
            continue
        if c['topo'] in excluded_topos:
            continue
        key = (c['topo'], c['k'])
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(c['topo'], []).append(c)

    # Per-topology log-linear fits
    fits = {}
    sat_fits = {}
    print(f'Log-linear fit (k≥{args.k_min}):')
    print(f'{"topo":<16} {"n":>3} {"slope":>9} {"intcpt":>9} {"R²":>8}   k values')
    print('-' * 70)
    for topo in ALL_TOPOS:
        if topo not in groups:
            continue
        result = fit(groups[topo], args.cost_axis, k_min=args.k_min)
        if result is None:
            continue
        slope, intercept, r2, valid = result
        fits[topo] = result
        ks = sorted({c['k'] for c in valid})
        print(f'{topo:<16} {len(valid):>3d} {slope:>+9.3f} {intercept:>+9.3f} {r2:>8.3f}   {ks}')

    # Per-topology saturation-curve fits (over ALL k including k=1,2 — the
    # power-law form naturally accommodates the early-k regime since
    # ΔAcc → -drop as k → 1; doesn't need the k≥3 cut)
    if args.saturation:
        print()
        print('Saturation-curve fit (over all k): ΔAcc(k) = ceiling - drop * k^(-α)')
        print(f'{"topo":<16} {"n":>3} {"ceiling":>9} {"drop":>7} {"α":>7} {"R²":>7}')
        print('-' * 60)
        for topo in ALL_TOPOS:
            if topo not in groups:
                continue
            result = fit_saturation(groups[topo])
            if result is None:
                continue
            popt, r2, valid, model = result
            sat_fits[topo] = (popt, r2, valid, model)
            print(f'{topo:<16} {len(valid):>3d} {popt[0]:>+9.3f} {popt[1]:>7.3f} {popt[2]:>7.3f} {r2:>7.3f}')

    # Crossover energy_ratios (where MAS overtakes SAS)
    print()
    print('Crossover energy_ratio vs SAS (above this, MAS topology wins):')
    if 'sas' in fits:
        s_sas, b_sas, _, _ = fits['sas']
        for topo in ('independent', 'decentralized'):
            if topo not in fits:
                continue
            s, b, _, _ = fits[topo]
            xc = crossover_x(s_sas, b_sas, s, b)
            yc = s_sas * np.log10(xc) + b_sas if xc else None
            if xc:
                print(f'  SAS vs {topo:<14}: E_ratio = {xc:6.2f}  (ΔAcc at crossover = {yc:+.3f})')

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    x_min = min(c[args.cost_axis] for cs in groups.values()
                for c in cs if c.get(args.cost_axis))
    x_max = max(c[args.cost_axis] for cs in groups.values()
                for c in cs if c.get(args.cost_axis))
    x_grid = np.logspace(np.log10(x_min) - 0.05, np.log10(x_max) + 0.05, 200)

    # Plot all cells (k≥1), shading k<args.k_min as open markers (excluded
    # from log-linear fit but kept on plot so the early-k overhead regime
    # is visible).
    for topo in ALL_TOPOS:
        if topo not in groups:
            continue
        s = TOPO_STYLE[topo]
        in_fit = [c for c in groups[topo] if c['k'] >= args.k_min
                  and c.get(args.cost_axis) and c.get('delta_acc') is not None]
        excluded = [c for c in groups[topo] if c['k'] < args.k_min
                    and c.get(args.cost_axis) and c.get('delta_acc') is not None]

        slope, intercept, r2, _ = fits[topo]
        legend_lbl = f"{s['label']}  slope={slope:+.2f}/dec  R²={r2:.2f}"
        if topo in sat_fits:
            popt, sat_r2, _, _ = sat_fits[topo]
            legend_lbl += f"  | sat R²={sat_r2:.2f}"

        ax.scatter([c[args.cost_axis] for c in in_fit],
                   [c['delta_acc'] for c in in_fit],
                   s=80, color=s['color'], marker=s['marker'],
                   edgecolor='black', linewidth=0.6, alpha=0.85, zorder=4,
                   label=legend_lbl)
        if excluded:
            ax.scatter([c[args.cost_axis] for c in excluded],
                       [c['delta_acc'] for c in excluded],
                       s=80, facecolors='none', edgecolor=s['color'],
                       marker=s['marker'], linewidth=1.2, alpha=0.7, zorder=3)

        # Log-linear fit over k≥k_min range (don't extrapolate to excluded region)
        if in_fit:
            x_lo = min(c[args.cost_axis] for c in in_fit)
            x_hi = max(c[args.cost_axis] for c in in_fit)
            x_line = np.logspace(np.log10(x_lo), np.log10(x_hi), 50)
            y_line = slope * np.log10(x_line) + intercept
            ax.plot(x_line, y_line, color=s['color'], linewidth=1.8,
                    alpha=0.6, linestyle='--', zorder=2)

        # Saturation curve over full range (in cost-axis space). Map k → cost
        # via the topology's actual energy_ratio(k) data, not a model.
        if topo in sat_fits:
            popt, _, sat_valid, model = sat_fits[topo]
            sat_sorted = sorted(sat_valid, key=lambda c: c['k'])
            ks_arr = np.array([c['k'] for c in sat_sorted])
            cost_arr = np.array([c[args.cost_axis] for c in sat_sorted])
            k_dense = np.logspace(np.log10(ks_arr.min()), np.log10(ks_arr.max()), 100)
            cost_dense = np.interp(k_dense, ks_arr, cost_arr)
            y_sat = model(k_dense, *popt)
            ax.plot(cost_dense, y_sat, color=s['color'], linewidth=2.0,
                    alpha=0.85, linestyle='-', zorder=2)

    # Per-cell k labels (every (topology, k) cell annotated with its k value).
    # Offsets staggered per topology so labels don't collide where curves cross.
    TOPO_OFFSET = {
        'sas':           (5, -10),
        'independent':   (5, 7),
        'decentralized': (-12, -12),
        'centralized':   (5, 10),
    }
    for topo in ALL_TOPOS:
        if topo not in groups:
            continue
        col = TOPO_STYLE[topo]['color']
        dx, dy = TOPO_OFFSET[topo]
        for c in groups[topo]:
            x = c.get(args.cost_axis)
            y = c.get('delta_acc')
            if x is None or y is None:
                continue
            ax.annotate(f'k={c["k"]}', xy=(x, y),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=7.5, color=col, fontweight='bold',
                        zorder=7)

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Energy ratio (per-bench geomean of E / E_SAS_k=10)')
    ax.set_ylabel('ΔAcc vs SAS k=10 (per-bench differenced, mean)')
    bench_label = f'{len(benches)} benchmarks ({", ".join(benches)})'
    title_lines = [f'Per-topology scaling — {bench_label}']
    if args.k_min > 1:
        title_lines.append(f'Log-linear fit on k≥{args.k_min} (open markers = k<{args.k_min}, excluded from fit)')
    if args.saturation:
        title_lines.append('Solid line: saturation curve (ΔAcc = ceiling − drop·k^−α). '
                           'Dashed: log-linear fit')
    ax.set_title('\n'.join(title_lines))
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

    out = Path(args.out)
    if args.prose_only and out.name == 'fig_per_topology_scaling.png':
        out = out.with_name('fig_per_topology_scaling_prose.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
