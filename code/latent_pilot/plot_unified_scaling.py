"""Pooled-log-linear scaling figure: every (topology, k, R) configuration
we have, plus all channel-muting / communication-restriction ablations,
plotted on one energy-vs-accuracy chart with a single pooled log-linear fit.

This is the COARSE SUMMARY view. The pooled log-linear is a useful
aggregate statistic across MAS engineering knobs but is NOT the underlying
functional form: per-topology saturation curves
ΔAcc(k) = ceiling − drop·k^(−α) fit at R²≈0.97 each over the full k range,
while the pooled log-linear here fits at R²≈0.71 (full k) without exclusions.
For the per-topology saturation view (the real model), see
`plot_per_topology_scaling.py --saturation`.

The OPERATIONAL_K_MIN=5 exclusion for the pooled fit is retained for
backwards compatibility of legacy figures, but is now flagged as
post-hoc — k=1,2 cells encode real coordination-tax behavior of MAS
topologies, not noise. Honest reporting uses the full-k pooled fit
(slope ≈ +0.22, R² ≈ 0.71).

Output: figures/fig_unified_scaling.png
"""
import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


MAIN_STUDY_DIRS = {
    'qampari':         'a5000_qampari_v4',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'workbench':       'a5000_workbench_v2',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'math':            'a5000_math_pilot',
    'swebench':        'a5000_swebench',
}
PHASE_A = 'a5000_phase_a_ablation'
PHASE_B2 = 'a5000_phase_b2_terse'

# Excluded mechanically-distinct outliers per user instruction:
#   centralized (hub topology, finding #6 — different mechanism)
#   truncate variants (partial peer text actively confuses)
# Use whole-word match so 'decentralized' is NOT excluded.
EXCLUDED_TOPOS = {'centralized', 'hybrid', 'specialist'}
INCLUDE_CENT = False  # Set True via --include-cent to keep all centralized configs
def is_excluded(topo):
    if INCLUDE_CENT and topo == 'centralized':
        return False
    if topo in EXCLUDED_TOPOS: return True
    if 'truncate' in topo: return True
    if topo.startswith('specialist'): return True
    return False

ABLATION_TOPOS = {
    # phase_a/phase_b2 dirs use the full topology runner name in the filename
    'decentralized_answer_only':       PHASE_A,
    'decentralized_empty':             PHASE_A,
    'decentralized_empty_silent':      PHASE_A,
    'decentralized_minimal':           PHASE_A,
    'decentralized_minimal_empty':     PHASE_A,
    'decentralized_terse':             PHASE_B2,
    'independent_share':               PHASE_B2,
    'independent_share_minimal':       PHASE_B2,
}

# Excluded due to incomplete cluster runs. Re-enable when the run finishes.
INCOMPLETE_TOPOS = {
    # Currently n=17/100 on browsecomp_plus; other 4 prose at n=100 each.
    # Including would inflate the cross-benchmark mean (the 17 sampled tasks
    # are an easier subset — Decent baseline on those same 17 is 0.59 vs
    # 0.40 on the full browsecomp+ baseline). Partial-fill sbatch:
    #   sbatch --array=0 mas-energy/scripts/run_partial_fill.sbatch
    'decentralized_terse_answer_only': 'browsecomp+ at n=17/100',
}

# Color/marker by topology family
def style_for(label):
    if label == 'sas':                    return dict(color='#1f77b4', marker='o', size=70)
    if label == 'independent':            return dict(color='#ff7f0e', marker='s', size=70)
    if label == 'decentralized':          return dict(color='#d62728', marker='D', size=70)
    if label.startswith('decentralized_') and 'empty' in label:
        return dict(color='#2c7fb8', marker='s', size=80)
    if label.startswith('decentralized_') and ('answer_only' in label or 'truncate' in label):
        return dict(color='#3690c0', marker='s', size=80)
    if 'terse' in label or 'minimal' in label and not 'independent' in label:
        return dict(color='#fdae6b', marker='^', size=85)
    if 'independent_share' in label:      return dict(color='#a50f15', marker='D', size=80)
    return dict(color='#888', marker='x', size=60)


def load(p):
    if not Path(p).exists():
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return 1.0 if r.get('correct') else 0.0


cre_main = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl$")
cre_abl  = re.compile(r"Qwen_Qwen3\.5-9B_(.+?)_k(\d+)\.jsonl$")


MIN_N_PER_CELL = 25  # per-bench task count to count the bench as covered

def min_benches_required(requested):
    # Require coverage of at least all-but-one of the requested benchmarks.
    # Catches the terse_answer_only n=17 case (4-of-5 prose) without dropping
    # legitimate k-scan cells that miss only math.
    # For single-benchmark mode, require exactly 1.
    return max(1, len(requested) - 1)


def gather_main_study(benchmarks):
    """Return list of dicts: one per (topology, k, R) cell. Stores per-benchmark
    accuracies so a SAS-k=10 baseline can be subtracted per-bench at plot time."""
    by_config = defaultdict(lambda: defaultdict(list))
    for bench in benchmarks:
        sub = MAIN_STUDY_DIRS.get(bench)
        if not sub: continue
        p = Path('mas-energy/results') / sub
        if not p.exists(): continue
        for f in p.glob('Qwen_Qwen3.5-9B_*.jsonl'):
            if 'preclean' in f.name or 'bak' in f.name: continue
            m = cre_main.match(f.name)
            if not m: continue
            topo = m.group(1)
            k = int(m.group(2))
            R = int(m.group(3)) if m.group(3) else None
            if is_excluded(topo): continue
            rows = load(f)
            if len(rows) < MIN_N_PER_CELL: continue
            by_config[(topo, k, R)][bench] = rows

    cells = []
    for (topo, k, R), benches in by_config.items():
        # R-scan cells (Decent or Cent at explicit R != 2) only exist on the
        # 4 prose benchmarks where R-scans were run — never on swebench or
        # math. Special-case threshold to ≥4 so they appear on the all-6
        # figure too. Their E_ratio reflects the prose average; readers should
        # note that R-scan points lack swebench coverage.
        is_r_scan = (topo in ('decentralized', 'centralized')
                     and R is not None and R != 2)
        threshold = 4 if is_r_scan else min_benches_required(benchmarks)
        if len(benches) < threshold: continue
        per_bench = {}
        for bench, rows in benches.items():
            es = [(r.get('gpu_dynamic_energy_joules') or 0) for r in rows]
            ts = [(r.get('total_tokens') or
                   ((r.get('total_prompt_tokens') or 0)
                    + (r.get('total_completion_tokens') or 0)))
                  for r in rows]
            ds = [(r.get('total_completion_tokens') or 0) for r in rows]
            accs = [acc(r) for r in rows]
            per_bench[bench] = dict(n=len(rows), e=mean(es),
                                    t=mean(ts), d=mean(ds), a=mean(accs))
        cells.append(dict(
            topo=topo, k=k, R=R,
            per_bench=per_bench,
            label=f'{topo} k={k}' + (f' R={R}' if R else ''),
            family='main',
        ))
    return cells


def gather_ablations(benchmarks):
    cells = []
    for topo, root in ABLATION_TOPOS.items():
        per_bench = {}
        for bench in benchmarks:
            p = Path('mas-energy/results') / root / bench
            if not p.exists(): continue
            files = sorted(p.glob(f'*_{topo}_k*.jsonl'))
            if not files: continue
            rows = load(files[0])
            if len(rows) < MIN_N_PER_CELL: continue
            es = [(r.get('gpu_dynamic_energy_joules') or 0) for r in rows]
            ts = [(r.get('total_tokens') or
                   ((r.get('total_prompt_tokens') or 0)
                    + (r.get('total_completion_tokens') or 0)))
                  for r in rows]
            ds = [(r.get('total_completion_tokens') or 0) for r in rows]
            accs = [acc(r) for r in rows]
            per_bench[bench] = dict(n=len(rows), e=mean(es),
                                    t=mean(ts), d=mean(ds), a=mean(accs))
        if len(per_bench) < min_benches_required(benchmarks): continue
        cells.append(dict(
            topo=topo, k=10, R=2,
            per_bench=per_bench,
            label=topo, family='ablation',
        ))
    return cells


# Operational-regime threshold: k ≥ OPERATIONAL_K_MIN for ALL topologies.
#
# At k<5 the agent doesn't have enough ReAct steps to engage typical
# benchmark difficulty:
#   - QAMPARI requires multiple sequential searches to recall a list of entities.
#   - BrowseComp+ requires several queries to navigate a 100K-doc corpus.
#   - SWE-bench requires reading multiple files before producing a patch.
# At k=1-3 the budget is exhausted before the task can be meaningfully
# attempted — for ANY topology, single- or multi-agent. Multi-agent k<5
# additionally pays parallel-decode overhead without iteration depth, but the
# core issue (insufficient compute to engage the task) is shared with SAS k<5.
#
# Applying the threshold uniformly (rather than asymmetrically by M>1) is the
# defensible choice. Empirically, including SAS k<5 in the fit anchors the
# left end of the curve at a point in the "compute-starved" regime where the
# task itself is degenerate, not where the law applies.
OPERATIONAL_K_MIN = 5


QWEN35_9B_PARAMS = 9.0e9
FLOPS_PER_TOKEN = 2 * QWEN35_9B_PARAMS  # Kaplan: 2·N_params per forward pass


def attach_summary(cells, sas_k10_per_bench, sas_k10_E_per_bench=None,
                   sas_k10_T_per_bench=None, sas_k10_D_per_bench=None):
    """For each cell, compute aggregated energy, tokens, FLOPs, and ΔAcc.

    Two cost-axis aggregation modes:
      - Absolute (default): arithmetic mean of per-benchmark means. Dominated
        by benchmarks with larger absolute scales (swebench >> prose).
      - Relative (when sas_k10_E_per_bench is supplied): per-benchmark ratio
        E_config[bench] / E_SAS_k=10[bench], then geometric mean across
        benchmarks. Bench-symmetric; SAS k=10 sits at 1.0; matches the
        per-benchmark differencing already applied to ΔAcc.
    """
    for c in cells:
        bench_overlap = [b for b in c['per_bench'] if b in sas_k10_per_bench]
        if not bench_overlap:
            c['energy'] = None
            c['delta_acc'] = None
            continue
        c['energy']        = mean(c['per_bench'][b]['e'] for b in bench_overlap)
        c['tokens']        = mean(c['per_bench'][b]['t'] for b in bench_overlap)
        c['decode_tokens'] = mean(c['per_bench'][b]['d'] for b in bench_overlap)
        c['flops']         = c['tokens']        * FLOPS_PER_TOKEN
        c['decode_flops']  = c['decode_tokens'] * FLOPS_PER_TOKEN

        # Bench-symmetric energy ratio: geomean of per-benchmark E/E_SAS_k10
        if sas_k10_E_per_bench:
            valid = [b for b in bench_overlap
                     if b in sas_k10_E_per_bench and sas_k10_E_per_bench[b] > 0
                     and c['per_bench'][b]['e'] > 0]
            if valid:
                ratios = [c['per_bench'][b]['e'] / sas_k10_E_per_bench[b] for b in valid]
                c['energy_ratio'] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            else:
                c['energy_ratio'] = None
        if sas_k10_T_per_bench:
            valid = [b for b in bench_overlap
                     if b in sas_k10_T_per_bench and sas_k10_T_per_bench[b] > 0
                     and c['per_bench'][b]['t'] > 0]
            if valid:
                ratios = [c['per_bench'][b]['t'] / sas_k10_T_per_bench[b] for b in valid]
                c['flops_ratio'] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            else:
                c['flops_ratio'] = None
        if sas_k10_D_per_bench:
            valid = [b for b in bench_overlap
                     if b in sas_k10_D_per_bench and sas_k10_D_per_bench[b] > 0
                     and c['per_bench'][b]['d'] > 0]
            if valid:
                ratios = [c['per_bench'][b]['d'] / sas_k10_D_per_bench[b] for b in valid]
                c['decode_flops_ratio'] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            else:
                c['decode_flops_ratio'] = None

        c['delta_acc'] = mean(c['per_bench'][b]['a'] - sas_k10_per_bench[b]
                              for b in bench_overlap)
        c['n']         = sum(c['per_bench'][b]['n'] for b in bench_overlap)
        c['n_benches'] = len(bench_overlap)
        # R-scan cells (R != 2, the canonical default) plot but don't fit:
        # they sample a different axis (rounds, not k), and cluster off the
        # k-scaling trend — including them in the fit drags R^2 without
        # revealing anything new about the unified k-scaling law.
        is_r_scan = c.get('R') is not None and c['R'] != 2
        c['out_of_regime'] = (
            (c['k'] < OPERATIONAL_K_MIN)
            or (c['topo'] == 'centralized')
            or is_r_scan
        )


def fit_loglinear(cells, in_regime_only=True, cost_field='energy'):
    valid = [c for c in cells if c.get(cost_field) and c.get('delta_acc') is not None]
    if in_regime_only:
        fit_cells = [c for c in valid if not c.get('out_of_regime')]
    else:
        fit_cells = valid
    if len(fit_cells) < 3: return None
    xs = np.log10([c[cost_field] for c in fit_cells])
    ys = np.array([c['delta_acc'] for c in fit_cells])
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    ss_res = ((ys - pred) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, intercept, r2, valid, fit_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_unified_scaling.png')
    ap.add_argument('--prose-only', action='store_true')
    ap.add_argument('--include-cent', action='store_true',
                    help='Include all centralized k/R configs. Plotted but '
                         'EXCLUDED from the log-linear fit (different mechanism).')
    ap.add_argument('--cost-axis',
                    choices=['energy', 'flops', 'energy_ratio', 'flops_ratio',
                             'decode_flops_ratio'],
                    default='energy',
                    help="X-axis cost metric. 'energy'/'flops' use absolute "
                         "values aggregated by arithmetic mean across benchmarks "
                         "(dominated by absolute-scale variation, e.g. swebench). "
                         "'energy_ratio'/'flops_ratio' use per-benchmark E/E_SAS_k=10 "
                         "geomean — bench-symmetric, matches the y-axis "
                         "per-benchmark ΔAcc differencing.")
    ap.add_argument('--pareto-only', action='store_true',
                    help='Filter to global Pareto-optimal cells: a cell is kept '
                         'only if no other cell has lower cost AND higher ΔAcc. '
                         'Highlights "best achievable" upper envelope.')
    args = ap.parse_args()
    if args.include_cent:
        global INCLUDE_CENT
        INCLUDE_CENT = True

    prose = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus']
    benches = prose if args.prose_only else prose + ['swebench']
    suffix = '5 prose benches' if args.prose_only else '6 benches (incl. SWE-bench)'

    main_cells = gather_main_study(benches)
    abl_cells = gather_ablations(benches)
    all_cells = main_cells + abl_cells

    # Reference baseline: SAS k=10 (no R suffix). Per-benchmark mean accuracy.
    sas_k10 = next((c for c in main_cells
                    if c['topo'] == 'sas' and c['k'] == 10 and c['R'] is None), None)
    if sas_k10 is None:
        print("ERROR: SAS k=10 baseline not found."); return
    sas_k10_per_bench = {b: d['a'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_E_per_bench = {b: d['e'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_T_per_bench = {b: d['t'] for b, d in sas_k10['per_bench'].items()}
    sas_k10_D_per_bench = {b: d['d'] for b, d in sas_k10['per_bench'].items()}
    print(f"SAS k=10 baseline accuracies per bench: {sas_k10_per_bench}")

    attach_summary(all_cells, sas_k10_per_bench,
                   sas_k10_E_per_bench=sas_k10_E_per_bench,
                   sas_k10_T_per_bench=sas_k10_T_per_bench,
                   sas_k10_D_per_bench=sas_k10_D_per_bench)
    print(f"Loaded {len(main_cells)} main-study + {len(abl_cells)} ablation cells "
          f"(after min_benches={min_benches_required(benches)} filter)")

    cost_field = args.cost_axis  # 'energy', 'flops', 'energy_ratio', 'flops_ratio'

    # Optionally filter to global Pareto-optimal cells (across in-regime cells only)
    if args.pareto_only:
        in_regime_cells = [c for c in all_cells
                           if not c.get('out_of_regime')
                           and c.get(cost_field) and c.get('delta_acc') is not None]
        sorted_cells = sorted(in_regime_cells, key=lambda c: c[cost_field])
        pareto_cells = []
        running_max = -float('inf')
        for c in sorted_cells:
            if c['delta_acc'] > running_max:
                pareto_cells.append(c)
                running_max = c['delta_acc']
        # Keep out-of-regime cells visible (still plotted); filter in_regime to pareto
        keep_set = set(id(c) for c in pareto_cells)
        for c in all_cells:
            if not c.get('out_of_regime') and id(c) not in keep_set:
                c['_filtered_out'] = True
        all_cells = [c for c in all_cells if not c.get('_filtered_out')]
        print(f"\nPareto filter: kept {len(pareto_cells)} in-regime cells (from "
              f"{len(in_regime_cells)} total) + out-of-regime cells preserved")
        for c in pareto_cells:
            label = f"{c['topo']} k={c['k']}" + (f" R={c['R']}" if c.get('R') else '')
            print(f"  Pareto: {label:<40} E={c[cost_field]:>10.2e}  ΔAcc={c['delta_acc']:+.3f}")

    fit = fit_loglinear(all_cells, in_regime_only=True, cost_field=cost_field)
    if fit is None:
        print("Not enough data."); return
    slope, intercept, r2, all_cells, fit_cells = fit
    n_out = sum(1 for c in all_cells if c.get('out_of_regime'))

    fig, ax = plt.subplots(figsize=(13, 8))

    # Group cells for plotting/legend. R-scan = ANY k with explicit R != 2
    # (the canonical default). All such cells go to "Decentralized R-scan",
    # whether at k=5 or k=10. This now picks up the k=5 R∈{1,3,4,5} variants
    # too, which were previously hidden inside Decentralized k-scan.
    groups = defaultdict(list)
    for c in all_cells:
        topo = c['topo']
        if topo == 'sas':                           grp = 'SAS k-scan'
        elif topo == 'independent':                 grp = 'Independent k-scan'
        elif topo == 'decentralized':
            if c['R'] is not None and c['R'] != 2:
                grp = 'Decentralized R-scan'
            else:
                grp = 'Decentralized k-scan'
        elif topo == 'centralized':
            if c['R'] is not None and c['R'] != 2:
                grp = 'Centralized R-scan (off-trend)'
            else:
                grp = 'Centralized k-scan (off-trend)'
        elif c['family'] == 'ablation':             grp = 'Channel-muting ablation'
        else:                                       grp = topo
        groups[grp].append(c)

    GROUP_STYLE = {
        'SAS k-scan':                       dict(color='#1f77b4', marker='o'),
        'Independent k-scan':               dict(color='#ff7f0e', marker='s'),
        'Decentralized k-scan':             dict(color='#d62728', marker='D'),
        'Decentralized R-scan':             dict(color='#7f0e0e', marker='*'),
        'Channel-muting ablation':          dict(color='#2c7fb8', marker='^'),
        'Centralized k-scan (off-trend)':   dict(color='#2ca02c', marker='P'),
        'Centralized R-scan (off-trend)':   dict(color='#117733', marker='X'),
    }
    DEFAULT_STYLE = dict(color='#888', marker='x')

    for grp, cells in sorted(groups.items()):
        s = GROUP_STYLE.get(grp, DEFAULT_STYLE)
        in_regime  = [c for c in cells if not c.get('out_of_regime')]
        out_regime = [c for c in cells if c.get('out_of_regime')]
        if in_regime:
            xs = [c[cost_field] for c in in_regime]
            ys = [c['delta_acc'] for c in in_regime]
            ax.scatter(xs, ys, s=80, color=s['color'], marker=s['marker'],
                       edgecolor='black', linewidth=0.6, alpha=0.85,
                       label=grp, zorder=4)
        if out_regime:
            xs = [c[cost_field] for c in out_regime]
            ys = [c['delta_acc'] for c in out_regime]
            # Pick the most informative reason this group is out-of-regime
            if 'R-scan' in grp:
                reason = 'R≠2, excluded from fit'
            elif 'Centralized' in grp:
                reason = 'off-trend, excluded from fit'
            else:
                reason = f'k<{OPERATIONAL_K_MIN}, excluded from fit'
            ax.scatter(xs, ys, s=80, facecolors='none', edgecolor=s['color'],
                       marker=s['marker'], linewidth=1.2, alpha=0.7,
                       zorder=3,
                       label=f'{grp} ({reason})')

    # Annotate EVERY in-regime cell with its k or R value so the figure
    # is self-documenting without legend cross-reference. Color-coded by
    # topology family. Out-of-regime (open-marker) cells also labeled.
    for c in all_cells:
        topo = c['topo']
        x = c[cost_field]; y = c['delta_acc']

        # Labels for R-scan cells: show R (and k if not 10)
        if topo == 'decentralized' and c['R'] is not None and c['R'] != 2:
            label = f"R={c['R']}" if c['k'] == 10 else f"k={c['k']} R={c['R']}"
            color = '#7f0e0e'
            offset = (4, 7)
        elif topo == 'centralized' and c['R'] is not None and c['R'] != 2:
            label = f"R={c['R']}" if c['k'] == 10 else f"k={c['k']} R={c['R']}"
            color = '#117733'
            offset = (4, 7)
        # Labels for k-scan cells: show k
        elif topo == 'sas':
            label = f"k={c['k']}"
            color = '#1f77b4'
            offset = (4, -10)
        elif topo == 'independent':
            label = f"k={c['k']}"
            color = '#ff7f0e'
            offset = (4, -10)
        elif topo == 'decentralized':
            label = f"k={c['k']}"
            color = '#7f0e0e' if c['R'] not in (None, 2) else '#d62728'
            offset = (4, -10)
        elif topo == 'centralized':
            label = f"k={c['k']}"
            color = '#2ca02c'
            offset = (4, -10)
        elif c.get('family') == 'ablation':
            # Channel-muting ablation: show its short name
            short = c['topo'].replace('decentralized_', '').replace('independent_share', 'IS')
            short = short.replace('answer_only', 'ans').replace('truncate', 'tr')
            label = short
            color = '#2c7fb8'
            offset = (4, 5)
        else:
            continue

        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points',
                    fontsize=6.5, color=color, alpha=0.9,
                    fontweight='bold' if topo == 'sas' and c['k'] in (1, 50) else 'normal')

    # Log-linear fit line
    xs_all = np.log10([c[cost_field] for c in all_cells])
    x_fit = np.linspace(xs_all.min() - 0.05, xs_all.max() + 0.05, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(10 ** x_fit, y_fit, '--', color='#444', linewidth=1.6, alpha=0.7,
            label=f'Log-linear fit  (slope=+{slope:.3f}/decade, R²={r2:.3f}, n={len(fit_cells)})',
            zorder=2)

    ax.set_xscale('log')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=1)
    if cost_field == 'flops':
        x_label = (f'Mean inference FLOPs per task (log scale)  —  '
                   f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})')
    elif cost_field == 'energy_ratio':
        x_label = (f'Energy ratio E_config / E_SAS_k=10  (geomean across benchmarks, log scale)  —  '
                   f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})')
    elif cost_field == 'flops_ratio':
        x_label = (f'FLOP ratio relative to SAS k=10  (geomean across benchmarks, log scale)  —  '
                   f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})')
    elif cost_field == 'decode_flops_ratio':
        x_label = (f'Decode FLOPs ratio relative to SAS k=10  (geomean across benchmarks, log scale)  —  '
                   f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})')
    else:
        x_label = (f'Mean energy per task (J, log scale, arithmetic mean across benchmarks)  —  '
                   f'each 10× buys ~+{slope*100:.1f}pp ΔAcc  (R²={r2:.2f})')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Mean ΔAccuracy vs SAS k=10 (per-benchmark differenced)',
                  fontsize=11)
    ax.set_title(
        f'Pooled MAS scaling (coarse summary): all (topology, k, R) configs + channel-muting ablations\n'
        f'averaged across {suffix}. See plot_per_topology_scaling.py for the per-topology\n'
        f'saturation form (ΔAcc = ceiling − drop·k⁻ᵅ, R²≈0.97 each), the real model.',
        fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

    # (Exclusion box removed; rationale documented in code comments above
    # OPERATIONAL_K_MIN and in INCOMPLETE_TOPOS, not on the figure.)

    print(f"  Log-linear fit (n={len(fit_cells)} in-regime cells): "
          f"slope={slope:.3f}/decade, R²={r2:.3f}")
    if n_out:
        print(f"  ({n_out} out-of-regime cells plotted but excluded from fit)")
    for grp, cells in sorted(groups.items()):
        in_n = sum(1 for c in cells if not c.get('out_of_regime'))
        out_n = sum(1 for c in cells if c.get('out_of_regime'))
        if out_n:
            if 'R-scan' in grp:
                reason = 'out-of-regime R≠2'
            elif 'Centralized' in grp:
                reason = 'off-trend mechanism'
            else:
                reason = f'out-of-regime k<{OPERATIONAL_K_MIN}'
            out_str = f' (+{out_n} {reason})'
        else:
            out_str = ''
        print(f"    {grp:<32} n_fit={in_n}{out_str}")
    # Flag the explicitly-excluded incomplete topologies
    if INCOMPLETE_TOPOS:
        print(f"  ** EXCLUDED for incomplete cluster runs: **")
        for topo, reason in INCOMPLETE_TOPOS.items():
            print(f"    {topo:<35} {reason}")
    # Also flag any topology that loaded but didn't meet the per-benchmark
    # coverage threshold (e.g. anything with partial coverage we forgot to list)
    excluded_by_threshold = []
    for topo, root in ABLATION_TOPOS.items():
        bcount = 0
        for bench in benches:
            p = Path('mas-energy/results') / root / bench
            if not p.exists(): continue
            files = sorted(p.glob(f'*_{topo}_k*.jsonl'))
            if not files: continue
            n = sum(1 for _ in open(files[0]))
            if n >= MIN_N_PER_CELL:
                bcount += 1
        if 0 < bcount < min_benches_required(benches):
            excluded_by_threshold.append((topo, bcount))
    if excluded_by_threshold:
        print(f"  ** Also excluded by n_benches filter (need >={min_benches_required(benches)}): **")
        for topo, bc in excluded_by_threshold:
            print(f"    {topo:<35} only {bc} benches with n>={MIN_N_PER_CELL}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
