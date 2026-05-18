"""Energy-accuracy frontier across the 4 baseline topologies plus all
channel-muting / communication ablations, averaged across benchmarks.

Two figures:
  1. figures/fig_frontier_prose.png       — averaged across 5 prose benchmarks
  2. figures/fig_frontier_all.png         — averaged across all 6 benchmarks

Each topology contributes one (mean_energy, mean_accuracy) point. Baselines
(SAS / Independent / Centralized / Decentralized) and ablations are colored
distinctly.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt


# ─── data sources ───
MAIN_STUDY_DIRS = {
    'qampari':         'a5000_qampari_v4',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'workbench':       'a5000_workbench_v2',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'swebench':        'a5000_swebench',
    'math':            'a5000_math_pilot',
}
PHASE_A = 'a5000_phase_a_ablation'
PHASE_B2 = 'a5000_phase_b2_terse'

# Topologies on the frontier are restricted to mechanically coherent variants:
# Independent and Decentralized form the no-comms / full-comms endpoints; all
# other variants sit on the spectrum between them as bandwidth or round-count
# restrictions on the same peer-to-peer mechanism.
# DROPPED:
#   - Centralized (hub-and-spoke is a structurally different mechanism, not a
#     restriction on the peer-to-peer channel; finding #6's β_R diverges)
#   - truncate variants (partial peer text actively confuses the model rather
#     than imposing a clean restriction; ΔAcc goes NEGATIVE on prose)
TOPOLOGIES = [
    # ─ Baseline endpoints at k=10 R=2 ─
    ('sas',                                 None, 'SAS',                        'baseline'),
    ('independent',                         None, 'Independent',                'baseline'),
    ('decentralized',                       None, 'Decentralized R=2',          'baseline'),
    # ─ Receiver-side channel-muting ─
    ('decentralized_answer_only',           PHASE_A, 'Decent + answer_only',       'rcv'),
    ('decentralized_empty',                 PHASE_A, 'Decent + empty',             'rcv'),
    ('decentralized_empty_silent',          PHASE_A, 'Decent + empty_silent',      'rcv'),
    # ─ Speaker-side decode reduction ─
    ('decentralized_terse',                 PHASE_B2, 'Decent + terse',             'spk'),
    ('decentralized_terse_answer_only',     PHASE_B2, 'Decent + terse_answer_only', 'spk'),
    ('decentralized_minimal',               PHASE_A, 'Decent + minimal',           'spk'),
    ('decentralized_minimal_empty',         PHASE_A, 'Decent + minimal_empty',     'spk'),
    # ─ Structural simplifications ─
    ('independent_share',                   PHASE_B2, 'independent_share',          'struct'),
    ('independent_share_minimal',           PHASE_B2, 'independent_share + minimal','struct'),
]

# Centralized at k=10 R=2 — added when --include-cent is passed. Hub-and-spoke
# is a structurally different mechanism (orchestrator multiplicatively
# compounds context, finding #6's β_R = 0.78 vs Decent's 0.25). Visualizing
# it on the frontier lets you see how far below the trend line it sits.
CENTRALIZED_TOPO = ('centralized', None, 'Centralized R=2', 'cent')

# Family colors and markers. Shape sizes chosen so single- and double-digit
# numbers fit fully within markers. Triangles (^) and diamonds (D) get extra
# size to compensate for their smaller visible interior vs circles/squares.
FAMILY_STYLE = {
    'baseline': dict(color='#2c2c2c', marker='o', size=400, edgewidth=1.4),
    'rcv':      dict(color='#2c7fb8', marker='s', size=380, edgewidth=1.0),
    'spk':      dict(color='#fdae6b', marker='^', size=520, edgewidth=1.0),
    'struct':   dict(color='#a50f15', marker='D', size=380, edgewidth=1.0),
    'cent':     dict(color='#2ca02c', marker='P', size=420, edgewidth=1.4),
}
FAMILY_LABEL = {
    'baseline': 'Baseline (SAS / Indep / Decent)',
    'rcv':      'Receiver-side channel ablation',
    'spk':      'Speaker-side decode reduction',
    'struct':   'Structural simplification (no debate)',
    'cent':     'Centralized (hub-and-spoke; off-trend)',
}


def load(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in open(path):
        try:
            d = json.loads(line)
            rows.append(d)
        except Exception:
            pass
    return rows


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return 1.0 if r.get('correct') else 0.0


def find_main_study_file(bench, topo):
    """For baseline topologies. Prefer _R2 for cent/decent; else default."""
    sub_dir = MAIN_STUDY_DIRS.get(bench)
    if not sub_dir:
        return None
    p = Path('mas-energy/results') / sub_dir
    if not p.exists():
        return None
    if topo in ('centralized', 'decentralized'):
        candidates = [
            p / f"Qwen_Qwen3.5-9B_{topo}_k10_R2.jsonl",
            p / f"Qwen_Qwen3.5-9B_{topo}_k10.jsonl",
        ]
    else:
        candidates = [p / f"Qwen_Qwen3.5-9B_{topo}_k10.jsonl"]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_ablation_file(bench, topo, root):
    p = Path('mas-energy/results') / root / bench
    if not p.exists():
        return None
    matches = sorted(p.glob(f"*_{topo}_k*.jsonl"))
    if matches:
        return matches[0]
    return None


def find_decent_baseline(bench):
    """Use a5000_latent_transcripts where available (most up-to-date Decent
    paired with ablations); fall back to main-study."""
    p = Path('mas-energy/results/a5000_latent_transcripts') / bench
    if p.exists():
        m = sorted(p.glob('*_decentralized_k*.jsonl'))
        if m:
            return m[0]
    return find_main_study_file(bench, 'decentralized')


def gather_rows(topo, root, benchmarks):
    """Load all rows for a topology across the given benchmarks."""
    out = []
    for bench in benchmarks:
        if root is None:
            path = find_main_study_file(bench, topo)
        else:
            path = find_ablation_file(bench, topo, root)
        if not path:
            continue
        rows = load(path)
        if not rows:
            continue
        out.append((bench, rows))
    return out


MIN_N_PER_CELL = 25  # exclude PARTIAL runs whose easy-task subset would
                      # bias the cross-benchmark mean (e.g. browsecomp_plus
                      # terse_answer_only at n=17 has an artificially high
                      # paired baseline of 0.59 on those 17 ids vs 0.40 typical).
                      # 25 keeps math main-study at n=30 (the full Level-5
                      # subset, not a biased partial).


def per_benchmark_stats(topo, root, benchmarks):
    """Return list of (bench, n, mean_E, mean_acc) for benches that have data
    above MIN_N_PER_CELL — partial runs are easy-subset-biased and excluded."""
    out = []
    bench_data = gather_rows(topo, root, benchmarks)
    for bench, rows in bench_data:
        n = len(rows)
        if n < MIN_N_PER_CELL:
            continue
        es = [(r.get('gpu_dynamic_energy_joules') or 0) for r in rows]
        accs = [acc(r) for r in rows]
        out.append((bench, n, mean(es), mean(accs)))
    return out


def aggregate(per_bench):
    """Average across benchmarks (each bench weighted equally)."""
    if not per_bench:
        return None
    e = mean(b[2] for b in per_bench)
    a = mean(b[3] for b in per_bench)
    n = sum(b[1] for b in per_bench)
    return dict(n=n, n_benches=len(per_bench), energy=e, acc=a)


def plot_frontier(benchmarks, title_suffix, out_path):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    points_by_family = defaultdict(list)
    all_points = []  # (x, y, idx, label, family) for the side legend

    # SAS k=10 reference: per-benchmark mean accuracy AND energy.
    # x-axis: geomean of per-bench E / E_SAS_k10[bench] (bench-symmetric ratio).
    # y-axis: per-bench differenced ΔAcc, then averaged.
    # Both axes apply the same per-benchmark normalization principle so swebench's
    # absolute scale doesn't dominate the cross-benchmark mean.
    import math as _math
    sas_stats = per_benchmark_stats('sas', None, benchmarks)
    sas_acc_per_bench = {b: a for b, n, e, a in sas_stats}
    sas_E_per_bench   = {b: e for b, n, e, a in sas_stats}

    for topo, root, label, family in TOPOLOGIES:
        per_bench = per_benchmark_stats(topo, root, benchmarks)
        if not per_bench:
            print(f"  skip {topo}: no data on any of {benchmarks}")
            continue
        # Restrict to benches where SAS k=10 also has data
        valid = [(b, n, e, a) for b, n, e, a in per_bench
                 if b in sas_acc_per_bench and b in sas_E_per_bench
                 and sas_E_per_bench[b] > 0 and e > 0]
        if not valid:
            print(f"  skip {topo}: no overlap with SAS benches")
            continue
        deltas = [a - sas_acc_per_bench[b] for b, n, e, a in valid]
        delta_mean = mean(deltas)
        # Geomean of per-bench energy ratios
        log_ratios = [_math.log(e / sas_E_per_bench[b]) for b, n, e, a in valid]
        energy_ratio = _math.exp(mean(log_ratios))
        n_total = sum(n for b, n, e, a in valid)
        points_by_family[family].append((energy_ratio, delta_mean, label,
                                         dict(n=n_total, n_benches=len(valid))))
        print(f"  {topo:<32} n_benches={len(valid)}  n={n_total:>4}  "
              f"E/SAS_k10={energy_ratio:>6.2f}×  ΔAcc(vs SAS)={delta_mean:+.3f}")

    # Assign one numeric ID per point, plotted in family order (baseline first).
    # Per-family text offsets compensate for marker geometry: triangles (^) have
    # their visual centroid below the matplotlib center anchor, so numbers are
    # nudged down 2 points to sit in the triangle's visual bulk.
    family_order = ['baseline', 'rcv', 'spk', 'struct', 'cent']
    text_y_offset = {'baseline': 0, 'rcv': 0, 'spk': -2, 'struct': 0, 'cent': 0}
    text_color = {'baseline': 'white', 'rcv': 'white', 'spk': 'black',
                  'struct': 'white', 'cent': 'white'}
    legend_entries = []  # (idx, label, family)
    next_id = 1
    for family in family_order:
        pts = points_by_family.get(family, [])
        style = FAMILY_STYLE[family]
        first_in_family = True
        for x, y, lab, agg in pts:
            ax.scatter(x, y, s=style['size'], color=style['color'],
                       marker=style['marker'], edgecolor='black',
                       linewidth=style['edgewidth'], zorder=5,
                       label=FAMILY_LABEL[family] if first_in_family else None)
            first_in_family = False
            ax.annotate(str(next_id), (x, y),
                        xytext=(0, text_y_offset[family]),
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=8.5, fontweight='bold',
                        color=text_color[family],
                        zorder=6)
            legend_entries.append((next_id, lab, family))
            next_id += 1

    ax.set_xscale('log')

    # x-axis label is updated post-fit below to include the slope finding.
    ax.set_ylabel('Mean ΔAccuracy vs SAS', fontsize=12)
    ax.set_title(f'Energy-vs-accuracy-lift over SAS, across topologies {title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

    # ─── Log-linear fit ───
    # Fit to all topology points with valid benchmark coverage (n_benches ≥ 5).
    # The line captures the broader regularity: structurally distinct
    # communication-restriction interventions all land on a common log-linear
    # tradeoff between mean energy and mean accuracy, not just the Pareto-
    # optimal subset. Pareto-optimal points are then highlighted with a thin
    # outline so the strict frontier remains visible.
    all_pts = []  # (x, y, label, family)
    for fam, pts in points_by_family.items():
        for x, y, lab, agg in pts:
            if agg.get('n_benches', 0) >= 5:
                all_pts.append((x, y, lab, fam))
    pareto = []
    for x, y, lab, fam in all_pts:
        dominated = any(
            (x2 <= x and y2 > y) or (x2 < x and y2 >= y)
            for x2, y2, _, _ in all_pts
            if (x2, y2) != (x, y)
        )
        if not dominated:
            pareto.append((x, y, lab, fam))
    pareto.sort()

    # Fit only on mechanically coherent peer-to-peer points; if Centralized
    # was added via --include-cent, plot it but exclude from the fit (different
    # mechanism — orchestrator hub).
    fit_pts = [p for p in all_pts if p[3] != 'cent']

    if len(fit_pts) >= 3:
        import numpy as _np
        xs = _np.log10([p[0] for p in fit_pts])
        ys = _np.array([p[1] for p in fit_pts])
        slope, intercept = _np.polyfit(xs, ys, 1)
        ss_tot = ((ys - ys.mean()) ** 2).sum()
        ss_res = ((ys - (slope * xs + intercept)) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # Extend the line slightly past the data on both ends
        all_xs = _np.log10([p[0] for p in all_pts])
        x_fit = _np.linspace(all_xs.min() - 0.05, all_xs.max() + 0.05, 50)
        y_fit = slope * x_fit + intercept
        ax.plot(10 ** x_fit, y_fit, '--', color='#444', linewidth=1.6,
                alpha=0.7, zorder=2,
                label=f'Log-linear fit  ΔAcc = {slope:.3f}·log₁₀(E) + {intercept:.3f}')

        # Outline Pareto-optimal points so the strict frontier remains visible
        for x, y, _, _ in pareto:
            ax.scatter(x, y, s=900, facecolors='none', edgecolor='#333',
                       linewidth=1.2, zorder=4)

        ax.text(0.02, 0.96,
                f'log-linear fit:\n'
                f'  slope = +{slope:.3f}/decade\n'
                f'  R² = {r2:.3f}  (n={len(fit_pts)})\n'
                f'  (Pareto-optimal: {len(pareto)} pts, outlined)',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', fc='#fffced', ec='#aaa',
                          alpha=0.95))
        # x-axis label encodes the headline finding inline
        ax.set_xlabel(
            f'Energy ratio  E_config / E_SAS_k=10  (geomean across benchmarks, log scale)  —  '
            f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})',
            fontsize=11)
        print(f"  Log-linear fit (n={len(fit_pts)}): slope={slope:.3f}/decade, R²={r2:.3f}")
        print(f"    Pareto-optimal subset ({len(pareto)} pts): {[lab for _, _, lab, _ in pareto]}")
    else:
        ax.set_xlabel('Energy ratio E/E_SAS_k=10 (log scale)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Two-part legend on the right side:
    #   (a) family color/marker key (top right)
    #   (b) numbered list of all points, grouped by family
    handles, labels_ = ax.get_legend_handles_labels()
    seen = set()
    family_handles = []
    for h, l in zip(handles, labels_):
        if l in seen or l is None:
            continue
        seen.add(l)
        family_handles.append((h, l))

    # Family-key legend — compact text with shrunken icons
    leg1 = ax.legend([h for h, _ in family_handles],
                     [l for _, l in family_handles],
                     loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     fontsize=9, framealpha=0.95, title='Family',
                     title_fontsize=10, markerscale=0.4)
    leg1._legend_box.align = 'left'
    ax.add_artist(leg1)

    # Build the numbered legend text, family-grouped
    grouped = defaultdict(list)
    for idx, lab, fam in legend_entries:
        grouped[fam].append((idx, lab))
    family_titles = {
        'baseline': 'Baseline topologies',
        'rcv':      'Receiver-side ablation',
        'spk':      'Speaker-side reduction',
        'struct':   'Structural simplification',
        'cent':     'Centralized (off-trend, excluded from fit)',
    }
    lines = []
    for fam in family_order:
        if fam not in grouped:
            continue
        lines.append(family_titles[fam])
        for idx, lab in grouped[fam]:
            lines.append(f"  {idx:>2}. {lab}")
        lines.append("")
    txt = "\n".join(lines).rstrip()
    fig.text(0.83, 0.55, txt, fontsize=8.5, va='center', ha='left',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#fafafa', ec='#888'))

    fig.text(0.5, 0.005,
             f'Each point: one topology cell averaged across {len(benchmarks)} benchmarks (equal-weighted).  '
             f'k=10, M=3, R=2.  All ablations baseline = full Decent.',
             ha='center', va='bottom', fontsize=9, style='italic', color='#444')

    # Reserve right side for the two-part legend
    plt.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.10)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-prose', default='figures/fig_frontier_prose.png')
    ap.add_argument('--out-all',   default='figures/fig_frontier_all.png')
    ap.add_argument('--include-cent', action='store_true',
                    help='Append Centralized R=2 to the plot. Plotted but '
                         'EXCLUDED from the log-linear fit (different mechanism).')
    args = ap.parse_args()

    if args.include_cent:
        global TOPOLOGIES
        TOPOLOGIES = list(TOPOLOGIES) + [CENTRALIZED_TOPO]

    prose = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus']
    all_b = prose + ['swebench']

    print("=== Frontier across 5 prose benchmarks ===")
    plot_frontier(prose, '(5 prose benchmarks)', args.out_prose)
    print("\n=== Frontier across all 6 benchmarks ===")
    plot_frontier(all_b, '(all 6 benchmarks)', args.out_all)


if __name__ == "__main__":
    main()
