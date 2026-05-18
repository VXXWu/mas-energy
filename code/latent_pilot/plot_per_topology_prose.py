"""Per-topology k-scaling on prose-only benchmarks.

Strips swebench out (the one bench that keeps climbing through k=50) and
shows what the per-topology k-scan actually looks like on the 5 prose
benches, where the user-flagged saturation is.

Output: figures/fig_per_topology_scaling.png
"""
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt


DIRS = {
    'qampari':         'a5000_qampari_v4',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'math':            'a5000_math_pilot',
    'workbench':       'a5000_workbench_v2',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'swebench':        'a5000_swebench',  # used only for the contrast panel
}
BENCH_LABEL = {
    'qampari': 'QAMPARI', 'fanoutqa': 'FanOutQA', 'math': 'MATH-L5',
    'workbench': 'WorkBench', 'browsecomp_plus': 'BrowseComp+', 'swebench': 'SWE-bench',
}
BENCH_COLOR = {
    'qampari': '#1f77b4', 'fanoutqa': '#2ca02c', 'math': '#9467bd',
    'workbench': '#ff7f0e', 'browsecomp_plus': '#8c564b', 'swebench': '#d62728',
}
TOPO_LABEL = {
    'sas': 'SAS (single agent)',
    'independent': 'Independent (M=3, parallel)',
    'decentralized': 'Decentralized (M=3, R=2 debate)',
}
PROSE = ['qampari', 'fanoutqa', 'math', 'workbench', 'browsecomp_plus']


def load(p):
    if not Path(p).exists(): return []
    out = []
    for line in open(p):
        try: out.append(json.loads(line))
        except: pass
    return out


def acc(r):
    if r.get('loose_accuracy') is not None:
        return float(r['loose_accuracy'])
    return 1.0 if r.get('correct') else 0.0


def get_kscan(bench, topo, R_filter=2):
    """Return [(k, mean_acc, energy)] sorted by k."""
    sub = DIRS[bench]
    base = Path('mas-energy/results') / sub
    out = []
    for f in sorted(base.glob(f'Qwen_Qwen3.5-9B_{topo}_k*.jsonl')):
        if 'preclean' in f.name or 'bak' in f.name: continue
        n = f.name.split('_k')[-1].replace('.jsonl', '')
        if '_R' in n:
            k_str, R_str = n.split('_R')
            try:
                if topo == 'decentralized' and int(R_str) != R_filter: continue
            except: continue
        else:
            k_str = n
        try: k = int(k_str)
        except: continue
        rows = load(f)
        if len(rows) < 25: continue
        a = mean(acc(r) for r in rows)
        e = mean((r.get('gpu_dynamic_energy_joules', 0) or 0) for r in rows)
        out.append((k, a, e))
    return sorted(out, key=lambda t: t[0])


def fit_loglinear(xs_log10, ys):
    if len(xs_log10) < 3: return None
    xs = np.array(xs_log10)
    ys = np.array(ys)
    slope, inter = np.polyfit(xs, ys, 1)
    pred = slope*xs + inter
    ss_tot = ((ys - ys.mean())**2).sum()
    ss_res = ((ys - pred)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return slope, inter, r2


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                             gridspec_kw=dict(hspace=0.35, wspace=0.30))
    topos = ['sas', 'independent', 'decentralized']

    # Get SAS k=10 baseline per bench (for ΔAcc differencing)
    sas_k10_baseline = {}
    for b in PROSE + ['swebench']:
        scan = get_kscan(b, 'sas')
        for k, a, _ in scan:
            if k == 10: sas_k10_baseline[b] = a

    # ROW 1: raw accuracy per benchmark, 5 prose benches overlaid per topology
    for col, topo in enumerate(topos):
        ax = axes[0, col]
        for bench in PROSE:
            scan = get_kscan(bench, topo)
            if not scan: continue
            ks  = [t[0] for t in scan]
            accs = [t[1] for t in scan]
            ax.plot(ks, accs, '-o', color=BENCH_COLOR[bench],
                    label=BENCH_LABEL[bench], markersize=6, linewidth=1.5,
                    alpha=0.9)
        ax.set_xscale('log')
        ax.set_xlabel('k (max ReAct steps)', fontsize=10)
        if col == 0: ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{TOPO_LABEL[topo]}\nraw accuracy, prose benchmarks',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ROW 2: ΔAcc-vs-SAS-k=10 per bench, with per-topology log-linear fit
    # (use prose-only data for fit)
    for col, topo in enumerate(topos):
        ax = axes[1, col]
        all_log_k = []
        all_delta = []
        for bench in PROSE:
            if bench not in sas_k10_baseline: continue
            scan = get_kscan(bench, topo)
            if not scan: continue
            base = sas_k10_baseline[bench]
            ks    = [t[0] for t in scan if t[0] >= 5]   # operational regime
            delta = [t[1] - base for t in scan if t[0] >= 5]
            if ks:
                ax.plot(ks, delta, '-o', color=BENCH_COLOR[bench],
                        label=BENCH_LABEL[bench], markersize=6,
                        linewidth=1.5, alpha=0.85)
                all_log_k.extend([math.log10(k) for k in ks])
                all_delta.extend(delta)

        # Log-linear fit, prose-only
        fit = fit_loglinear(all_log_k, all_delta)
        if fit is not None:
            slope, inter, r2 = fit
            xs_fit = np.linspace(min(all_log_k)-0.05, max(all_log_k)+0.05, 100)
            ys_fit = slope*xs_fit + inter
            ax.plot(10**xs_fit, ys_fit, '--', color='black', linewidth=1.7,
                    alpha=0.7,
                    label=f'fit: +{slope:.3f}/decade  R²={r2:.2f}')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('k (max ReAct steps, k≥5)', fontsize=10)
        if col == 0: ax.set_ylabel('ΔAcc vs SAS k=10', fontsize=11)
        ax.set_title(f'{TOPO_LABEL[topo]}\nΔAcc-baselined  (prose only, k≥5)',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Per-topology k-scaling on prose benchmarks\n'
                 'top: raw accuracy (saturation visible per bench)  |  '
                 'bottom: ΔAcc vs SAS k=10, with log-linear fit',
                 fontsize=13, fontweight='bold', y=0.995)

    out = Path('figures/fig_per_topology_scaling.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f'saved {out}')

    # Now also report numerical comparison: prose-only vs all-6 unified slopes
    print('\n=== Per-topology log-linear slopes (prose only, k≥5) ===')
    for topo in topos:
        all_log_k = []
        all_delta = []
        for bench in PROSE:
            if bench not in sas_k10_baseline: continue
            scan = get_kscan(bench, topo)
            base = sas_k10_baseline[bench]
            for k, a, _ in scan:
                if k >= 5:
                    all_log_k.append(math.log10(k))
                    all_delta.append(a - base)
        fit = fit_loglinear(all_log_k, all_delta)
        if fit:
            print(f'  {topo:<14}  slope=+{fit[0]:.3f}/decade  R²={fit[2]:.3f}  n={len(all_log_k)}')

    print('\n=== Same fits but INCLUDING swebench (for comparison) ===')
    for topo in topos:
        all_log_k = []
        all_delta = []
        for bench in PROSE + ['swebench']:
            if bench not in sas_k10_baseline: continue
            scan = get_kscan(bench, topo)
            base = sas_k10_baseline[bench]
            for k, a, _ in scan:
                if k >= 5 and k <= 50:  # cap swebench at 50 like the unified figure
                    all_log_k.append(math.log10(k))
                    all_delta.append(a - base)
        fit = fit_loglinear(all_log_k, all_delta)
        if fit:
            print(f'  {topo:<14}  slope=+{fit[0]:.3f}/decade  R²={fit[2]:.3f}  n={len(all_log_k)}')


if __name__ == '__main__':
    main()
