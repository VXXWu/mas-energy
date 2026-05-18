"""Test the corrected Brown-bound saturation formula against the SAS k-scan data.

Predicted form (oracle verifier, autoregressive correlation factor λ):
    Acc(k) = 1 − (1 − p)^(k·λ)

For each benchmark, fit (p, λ) to the SAS k-scan, then check:
  - Goodness of fit (R²)
  - Predicted saturation point k* = 1 / (λ · q)  where q = −log(1−p)
  - Empirical "elbow" of the curve

Also: use the fit p to predict per-benchmark slope α via the closed form
α = ln(10)/e · v · λ · θ from the prior turn, and compare to empirical
slopes from the unified-law per-benchmark fit.

This validates (or falsifies) the corrected geometric-marginal derivation.
"""
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.optimize import curve_fit


MAIN = {
    'qampari':         'a5000_qampari_v4',
    'fanoutqa':        'a5000_fanoutqa_v4',
    'workbench':       'a5000_workbench_v2',
    'browsecomp_plus': 'a5000_browsecomp_pilot',
    'math':            'a5000_math_pilot',
    'swebench':        'a5000_swebench',
}


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


def sas_kscan(bench):
    """Return list of (k, mean_acc, n) for SAS at every k available."""
    sub = MAIN[bench]
    out = []
    for f in sorted(Path(f'mas-energy/results/{sub}').glob('Qwen_Qwen3.5-9B_sas_k*.jsonl')):
        if 'preclean' in f.name or 'bak' in f.name: continue
        # parse k
        name = f.name
        k_str = name.split('_k')[-1].replace('.jsonl', '').split('_')[0]
        try: k = int(k_str)
        except: continue
        rows = load(f)
        if len(rows) < 25: continue
        out.append((k, mean(acc(r) for r in rows), len(rows)))
    out.sort(key=lambda t: t[0])
    return out


def indep_kscan(bench):
    """Independent topology k-scan (M=3 by default)."""
    sub = MAIN[bench]
    out = []
    for f in sorted(Path(f'mas-energy/results/{sub}').glob('Qwen_Qwen3.5-9B_independent_k*.jsonl')):
        if 'preclean' in f.name or 'bak' in f.name: continue
        name = f.name
        k_str = name.split('_k')[-1].replace('.jsonl', '').split('_')[0]
        try: k = int(k_str)
        except: continue
        rows = load(f)
        if len(rows) < 25: continue
        out.append((k, mean(acc(r) for r in rows), len(rows)))
    out.sort(key=lambda t: t[0])
    return out


def brown_curve(k, p, lam, ceiling=1.0):
    """Acc(k) = ceiling · (1 − (1−p)^(k·λ))."""
    return ceiling * (1.0 - (1.0 - p) ** (k * lam))


def fit_brown(ks, accs, fix_ceiling=None):
    """Fit (p, λ) [and optionally ceiling] to (k, acc) data."""
    ks_arr = np.array(ks, dtype=float)
    accs_arr = np.array(accs, dtype=float)
    if fix_ceiling is not None:
        f = lambda k, p, lam: brown_curve(k, p, lam, fix_ceiling)
        try:
            popt, _ = curve_fit(f, ks_arr, accs_arr,
                                p0=[0.2, 0.3], bounds=([1e-3, 1e-3], [0.99, 5.0]),
                                maxfev=10000)
            p, lam = popt
            ceiling = fix_ceiling
        except Exception as e:
            return None
    else:
        f = lambda k, p, lam, ce: brown_curve(k, p, lam, ce)
        try:
            popt, _ = curve_fit(f, ks_arr, accs_arr,
                                p0=[0.2, 0.3, max(accs)+0.05],
                                bounds=([1e-3, 1e-3, 0.05], [0.99, 5.0, 1.0]),
                                maxfev=10000)
            p, lam, ceiling = popt
        except Exception as e:
            return None
    pred = brown_curve(ks_arr, p, lam, ceiling)
    ss_tot = ((accs_arr - accs_arr.mean())**2).sum()
    ss_res = ((accs_arr - pred)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return dict(p=p, lam=lam, ceiling=ceiling, r2=r2,
                rmse=math.sqrt(ss_res/len(ks_arr)))


def empirical_elbow(ks, accs, frac=0.9):
    """Smallest k at which acc(k) ≥ frac · max_acc."""
    if not accs: return None
    target = frac * max(accs)
    for k, a in zip(ks, accs):
        if a >= target:
            return k
    return ks[-1]


def main():
    print("\n=== SAS k-scan: fitting Acc(k) = ceiling · (1 − (1−p)^(k·λ)) ===\n")
    print(f"{'bench':<18} {'n_k':<5} {'k=1 acc':<10} {'k=max acc':<11} "
          f"{'p̂':<8} {'λ̂':<8} {'ceiling':<9} {'R²':<6} "
          f"{'pred k*':<8} {'obs elbow':<10}")
    print('-' * 110)

    bench_fits = {}
    for bench in MAIN:
        scan = sas_kscan(bench)
        if not scan or len(scan) < 4: continue
        ks  = [t[0] for t in scan]
        accs = [t[1] for t in scan]
        # Use fixed ceiling = max observed acc + 0.02 to avoid over-fitting
        fit = fit_brown(ks, accs)
        if fit is None: continue
        # Saturation point: where (1−p)^(k·λ) = e^(−1) ⇒ k·λ·q = 1 ⇒ k* = 1/(λ·q)
        q = -math.log(1 - fit['p']) if fit['p'] < 1 else float('inf')
        k_star = 1.0 / (fit['lam'] * q) if (fit['lam']*q) > 0 else float('inf')
        elbow = empirical_elbow(ks, accs, frac=0.9)
        bench_fits[bench] = dict(fit=fit, k_star=k_star, elbow=elbow,
                                 q=q, ks=ks, accs=accs)
        print(f"{bench:<18} {len(ks):<5} {accs[0]:<10.3f} {max(accs):<11.3f} "
              f"{fit['p']:<8.3f} {fit['lam']:<8.3f} {fit['ceiling']:<9.3f} "
              f"{fit['r2']:<6.3f} {k_star:<8.1f} {elbow}")

    # Show curve and prediction overlay for a couple of benches
    print(f"\n=== Per-k predicted vs observed (qampari, math, swebench) ===")
    for bench in ['qampari', 'math', 'swebench']:
        if bench not in bench_fits: continue
        d = bench_fits[bench]
        print(f"\n{bench}:")
        print(f"  fit: p={d['fit']['p']:.3f} λ={d['fit']['lam']:.3f} "
              f"ceiling={d['fit']['ceiling']:.3f} q={d['q']:.3f} k*={d['k_star']:.1f}")
        for k, a in zip(d['ks'], d['accs']):
            pred = brown_curve(k, d['fit']['p'], d['fit']['lam'], d['fit']['ceiling'])
            print(f"  k={k:>3}: obs={a:.3f}  pred={pred:.3f}  Δ={a-pred:+.3f}")

    # Predict α via the closed form, compare to empirical per-benchmark slopes
    # α ≈ ln(10)/e · v · λ · θ
    # where θ = N·exp(−N·q) / max[N·exp(−N·q)] = (N·q) · exp(1 − N·q)  (peaked at Nq=1)
    # For SAS k-scan operating range, take N = mean(k_eff) over the scan
    print(f"\n=== Predicted vs empirical per-benchmark unified-law slope α ===\n")
    print(f"{'bench':<18} {'p':<8} {'q':<8} {'λ_fit':<8} "
          f"{'mean k·λ':<10} {'θ̂':<8} {'pred α (v=1)':<14} "
          f"{'pred α (v=0.7)':<16} {'observed α':<12}")
    print('-' * 110)
    # Empirical per-benchmark slopes from handoff doc
    obs_alpha = {
        'math':            0.009,
        'qampari':         0.042,
        'browsecomp_plus': 0.176,
        'fanoutqa':        0.208,
        'workbench':       0.245,
        'swebench':        0.427,
    }
    for bench in MAIN:
        if bench not in bench_fits: continue
        d = bench_fits[bench]
        p, lam, q = d['fit']['p'], d['fit']['lam'], d['q']
        ks = d['ks']
        N_eff_mean = mean(k*lam for k in ks)
        # Operating-position factor: peak slope is at N·q=1, value 1/e.
        # θ measures fraction-of-peak achieved at the operating N.
        x = N_eff_mean * q
        theta = (x * math.exp(-x)) / (1.0/math.e) if x > 0 else 0.0
        # Predicted α = ln(10) · v · θ · (1/e), substituting at the operating point
        # Equivalently: α = ln(10)/e · v · θ
        pred_v1 = (math.log(10) / math.e) * 1.0 * theta
        pred_v07 = (math.log(10) / math.e) * 0.7 * theta
        obs = obs_alpha.get(bench, float('nan'))
        print(f"{bench:<18} {p:<8.3f} {q:<8.3f} {lam:<8.3f} "
              f"{N_eff_mean:<10.2f} {theta:<8.3f} "
              f"{pred_v1:<14.3f} {pred_v07:<16.3f} {obs:<12.3f}")

    # Also test against Independent k-scan where the formula applies more
    # cleanly (parallel independent agents).
    print(f"\n=== Independent k-scan: Brown formula should fit BETTER (cleaner λ) ===\n")
    print(f"{'bench':<18} {'n_k':<5} {'k=1 acc':<10} {'k=max acc':<11} "
          f"{'p̂':<8} {'λ̂':<8} {'ceiling':<9} {'R²':<6} {'pred k*':<8}")
    print('-' * 100)
    for bench in MAIN:
        scan = indep_kscan(bench)
        if not scan or len(scan) < 4: continue
        ks  = [t[0] for t in scan]
        accs = [t[1] for t in scan]
        fit = fit_brown(ks, accs)
        if fit is None: continue
        q = -math.log(1 - fit['p']) if fit['p'] < 1 else float('inf')
        k_star = 1.0 / (fit['lam'] * q) if (fit['lam']*q) > 0 else float('inf')
        print(f"{bench:<18} {len(ks):<5} {accs[0]:<10.3f} {max(accs):<11.3f} "
              f"{fit['p']:<8.3f} {fit['lam']:<8.3f} {fit['ceiling']:<9.3f} "
              f"{fit['r2']:<6.3f} {k_star:<8.1f}")


if __name__ == "__main__":
    main()
