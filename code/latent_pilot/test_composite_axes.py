"""Refit the unified scaling law against multiple cost axes (composites of
P and C tokens) to find the most precise mechanistic axis.

Tested axes:
  C            — decode tokens only
  P            — prompt tokens only
  P+C          — total tokens (current FLOPs proxy)
  DET          — C + P/256, decode-token equivalents from b/c ratio
  DET_int      — DET plus the P·C interaction term from energy regression
  E_hat        — full predicted energy from per-call regression coefficients
  Cw_alpha     — C + P*alpha for alpha ∈ {1/16, 1/64, 1/256, 1/1024}
                  (sensitivity sweep)

For each axis, computes per-benchmark ratio to SAS k=10, geomeans across
benchmarks, fits log-linear ΔAcc vs log10(cost ratio), reports slope, R²,
n_fit. Also reports the per-bench-averaged absolute composite.

The point: identify which composite of (P, C) best explains ΔAcc variance,
not just which has the right mechanistic weighting.
"""
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np

# Energy regression coefficients (consolidated_findings.md finding #1)
B_PREFILL = 0.020   # J per prompt token
C_DECODE  = 5.14    # J per completion token
B_INT     = 5.73e-5 # J per (P·C)
A_INTER   = -23     # J intercept (drop in ratio form)


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
ABLATION_TOPOS = {
    'decentralized_answer_only':       PHASE_A,
    'decentralized_empty':             PHASE_A,
    'decentralized_empty_silent':      PHASE_A,
    'decentralized_minimal':           PHASE_A,
    'decentralized_minimal_empty':     PHASE_A,
    'decentralized_terse':             PHASE_B2,
    'independent_share':               PHASE_B2,
    'independent_share_minimal':       PHASE_B2,
}
EXCLUDED_TOPOS = {'centralized', 'hybrid', 'specialist'}
OPERATIONAL_K_MIN = 5
MIN_N_PER_CELL = 25

import re
cre_main = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl$")


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


def is_excluded(topo):
    if topo in EXCLUDED_TOPOS: return True
    if 'truncate' in topo: return True
    if topo.startswith('specialist'): return True
    return False


def cell_stats(rows):
    """Return per-cell (P, C, n, mean_acc) plus standard composites."""
    es = [r.get('gpu_dynamic_energy_joules', 0) or 0 for r in rows]
    ps = [r.get('total_prompt_tokens', 0) or 0 for r in rows]
    cs = [r.get('total_completion_tokens', 0) or 0 for r in rows]
    accs = [acc(r) for r in rows]
    n = len(rows)
    p, c = mean(ps), mean(cs)
    return dict(
        n=n,
        p=p, c=c,
        e=mean(es),
        t=p + c,
        e_hat = max(B_PREFILL*p + C_DECODE*c + B_INT*p*c, 1.0),  # drop intercept
        det     = c + p/256.0,
        det_int = c + p/256.0 + (B_INT/C_DECODE)*p*c,  # add interaction in C-units
        det_64  = c + p/64.0,
        det_1k  = c + p/1024.0,
        a=mean(accs),
    )


def gather_main(benches):
    by_config = defaultdict(dict)
    for bench in benches:
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
            by_config[(topo, k, R)][bench] = cell_stats(rows)
    cells = []
    for (topo, k, R), per_bench in by_config.items():
        if len(per_bench) < max(3, len(benches)-1): continue
        cells.append(dict(topo=topo, k=k, R=R, per_bench=per_bench, family='main',
                          out_of_regime=(k < OPERATIONAL_K_MIN)))
    return cells


def gather_ablations(benches):
    cells = []
    for topo, root in ABLATION_TOPOS.items():
        per_bench = {}
        for bench in benches:
            p = Path('mas-energy/results') / root / bench
            if not p.exists(): continue
            files = sorted(p.glob(f'*_{topo}_k*.jsonl'))
            if not files: continue
            rows = load(files[0])
            if len(rows) < MIN_N_PER_CELL: continue
            per_bench[bench] = cell_stats(rows)
        if len(per_bench) < max(3, len(benches)-1): continue
        cells.append(dict(topo=topo, k=10, R=2, per_bench=per_bench, family='abl',
                          out_of_regime=False))
    return cells


def fit(cells, axis_field, sas_per_bench):
    """Per-bench ratio of axis_field to SAS k=10, geomean, log-fit vs ΔAcc."""
    xs, ys, labels = [], [], []
    for c in cells:
        if c['out_of_regime']: continue
        bench_overlap = [b for b in c['per_bench']
                         if b in sas_per_bench and sas_per_bench[b][axis_field] > 0
                         and c['per_bench'][b][axis_field] > 0]
        if len(bench_overlap) < 3: continue
        ratios = [c['per_bench'][b][axis_field] / sas_per_bench[b][axis_field]
                  for b in bench_overlap]
        log_ratio = sum(math.log(r) for r in ratios) / len(ratios)
        delta_acc = mean(c['per_bench'][b]['a'] - sas_per_bench[b]['a']
                         for b in bench_overlap)
        xs.append(log_ratio / math.log(10))  # log10
        ys.append(delta_acc)
        labels.append((c['topo'], c['k'], c.get('R')))
    if len(xs) < 3: return None
    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope*xs + intercept
    ss_tot = ((ys - ys.mean())**2).sum()
    ss_res = ((ys - pred)**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return dict(slope=slope, r2=r2, n=len(xs), xs=xs, ys=ys,
                labels=labels, intercept=intercept)


def main():
    benches = ['qampari', 'fanoutqa', 'math', 'workbench',
               'browsecomp_plus', 'swebench']
    main_cells = gather_main(benches)
    abl_cells  = gather_ablations(benches)
    cells = main_cells + abl_cells

    sas10 = next(c for c in main_cells
                 if c['topo']=='sas' and c['k']==10 and c['R'] is None)
    sas_per_bench = sas10['per_bench']

    axes = ['e', 'e_hat', 't', 'c', 'p', 'det', 'det_int', 'det_64', 'det_1k']
    desc = {
        'e':       'Energy (measured, NVML)              ',
        'e_hat':   'E_hat = b·P + c·C + b_int·P·C        ',
        't':       'P+C (total tokens, FLOPs proxy)       ',
        'c':       'C alone (decode tokens)               ',
        'p':       'P alone (prompt tokens)               ',
        'det':     'DET = C + P/256                       ',
        'det_int': 'DET + P·C interaction (in C-units)    ',
        'det_64':  'C + P/64 (P weighted heavier)         ',
        'det_1k':  'C + P/1024 (P weighted lighter)       ',
    }
    print(f"\nUnified-law refit across {len(cells)} configs (29 in-regime + ablations)")
    print(f"{'Axis':<40}  slope/decade   R²    n_fit")
    print('-' * 75)
    results = {}
    for ax in axes:
        r = fit(cells, ax, sas_per_bench)
        if r is None:
            print(f"{desc[ax]}  no data"); continue
        results[ax] = r
        print(f"{desc[ax]}    +{r['slope']:.3f}     {r['r2']:.3f}    {r['n']}")

    # Sweep alpha to find R²-maximizing weighting C + alpha*P
    print(f"\nWeight sweep: cost = C + α·P  (α from 0 to 1; α=1/256 ≈ b/c)")
    print(f"{'α':<10} {'1/α':<10} {'slope':<10} {'R²':<8} ")
    print('-' * 38)
    best = (0, 0, 0)
    for alpha in [0, 1/8192, 1/4096, 1/2048, 1/1024, 1/512, 1/256,
                  1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0]:
        for c in cells:
            for b in c['per_bench']:
                pb = c['per_bench'][b]
                pb['cw'] = pb['c'] + alpha * pb['p']
        for b in sas_per_bench:
            sas_per_bench[b]['cw'] = sas_per_bench[b]['c'] + alpha * sas_per_bench[b]['p']
        r = fit(cells, 'cw', sas_per_bench)
        inv = 1/alpha if alpha > 0 else float('inf')
        print(f"{alpha:<10.5g} {inv:<10.4g} +{r['slope']:.3f}    {r['r2']:.4f}")
        if r['r2'] > best[2]:
            best = (alpha, r['slope'], r['r2'])
    print(f"\n  BEST α: {best[0]:.5g}  (1/α = {1/best[0]:.4g})")
    print(f"  slope = +{best[1]:.3f}/decade,  R² = {best[2]:.4f}")

    # Pareto compression check: where do Decent k=20 and Indep k=50 land
    # under each axis?
    print(f"\n=== Per-config positions on different axes (E_ratio, T_ratio, DET_ratio) ===")
    interesting = ['decentralized k=20 R=None',
                   'independent k=50 R=None',
                   'decentralized k=10 R=2',
                   'sas k=50 R=None']
    for c in cells:
        label = f"{c['topo']} k={c['k']} R={c.get('R')}"
        if label not in interesting: continue
        bench_overlap = [b for b in c['per_bench'] if b in sas_per_bench]
        if len(bench_overlap) < 3: continue
        def gm(field):
            rs = [c['per_bench'][b][field]/sas_per_bench[b][field]
                  for b in bench_overlap if sas_per_bench[b][field]>0
                  and c['per_bench'][b][field]>0]
            return math.exp(sum(math.log(r) for r in rs)/len(rs)) if rs else float('nan')
        delta_acc = mean(c['per_bench'][b]['a'] - sas_per_bench[b]['a']
                         for b in bench_overlap)
        print(f"  {label:<35} ΔAcc={delta_acc:+.3f}  E={gm('e'):.2f}×  T={gm('t'):.2f}×  C={gm('c'):.2f}×  DET={gm('det'):.2f}×")


if __name__ == "__main__":
    main()
