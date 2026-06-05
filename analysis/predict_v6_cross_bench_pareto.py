"""HistGBM Pareto-dominant prediction applied to ALL 4 main benchmarks.

Goal: produce a methodologically-symmetric sampling protocol across benchmarks.
SWE-bench has 87+ cube-interior cells; the other 3 have only 1D sweeps. A
reviewer can argue we didn't search the cube on FanOutQA/WorkBench/BrowseComp+,
so our claim "SWE-bench is the only 3-axis-active benchmark" rests on negative
evidence we never actually probed.

This script:
1. Loads cells for all 4 benchmarks
2. Fits HistGBM per-benchmark on existing data (same architecture used for v5)
3. Generates Pareto-dominant predictions per benchmark
4. Outputs top 4 candidates per benchmark for v6 validation

Expected outcome: 0-1 Pareto extensions on prose benchmarks (saturated),
≥1 on SWE-bench (already extensively cube-sampled). Either way, defensible
cross-benchmark sampling story.
"""
import json, glob, os, re, itertools
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")
BENCH_DIRS = {
    "FanOutQA":   "a5000_fanoutqa_v4",
    "WorkBench":  "a5000_workbench_v2",
    "BrowseComp+":"a5000_browsecomp_pilot",
    "SWE-bench":  "a5000_swebench",
}


def load_bench(bench_dir):
    base = fos.path.join(MAS_ENERGY_ROOT, "results/{bench_dir}")
    cells = []
    for f in sorted(glob.glob(f"{base}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo not in ("centralized", "decentralized"): continue
        R = int(r_str) if r_str else 2
        M = int(m_str) if m_str else 3
        n=0; acc=0; energy=0; P=0; C=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            n += 1
            a = (float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                 else (1.0 if d.get("correct") else 0.0))
            acc += a
            energy += d.get("gpu_dynamic_energy_joules", 0) or 0
            P += d.get("total_prompt_tokens", 0) or 0
            C += d.get("total_completion_tokens", 0) or 0
        if n < 30: continue
        cells.append({"topo": topo, "k": k, "R": R, "M": M,
                      "acc": 100*acc/n, "energy_kJ": energy/n/1000,
                      "P": P/n, "C": C/n, "n": n})
    return pd.DataFrame(cells)


def prep_xy(df, col):
    X = np.column_stack([
        np.log(df["k"].values), np.log(df["R"].values), np.log(df["M"].values),
        (df["topo"] == "centralized").astype(int).values,
    ])
    return X, df[col].values


def predict_pareto_candidates(df, k_grid, r_grid, m_grid):
    """Fit HistGBM for acc, P, C; linear regression for E. Return Pareto-dominant cells."""
    X, y_acc = prep_xy(df, "acc")
    _, y_P = prep_xy(df, "P")
    _, y_C = prep_xy(df, "C")
    acc_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, y_acc)
    p_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, y_P)
    c_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, y_C)
    PC = np.column_stack([df["P"].values, df["C"].values])
    e_reg = LinearRegression().fit(PC, df["energy_kJ"].values)

    # Existing cells
    obs = set((r["topo"], r["k"], r["R"], r["M"]) for _, r in df.iterrows())
    # Observed Pareto frontier
    obs_sorted = df.sort_values("energy_kJ")
    pareto = []
    max_acc = -1
    for _, r in obs_sorted.iterrows():
        if r["acc"] > max_acc:
            pareto.append((r["energy_kJ"], r["acc"]))
            max_acc = r["acc"]

    # Score candidates
    candidates = []
    for topo, k, R, M in itertools.product(["centralized", "decentralized"], k_grid, r_grid, m_grid):
        if (topo, k, R, M) in obs: continue
        x = np.array([[np.log(k), np.log(R), np.log(M), 1 if topo == "centralized" else 0]])
        pred_p = max(0, p_m.predict(x)[0])
        pred_c = max(0, c_m.predict(x)[0])
        pred_e = max(1.0, float(e_reg.predict(np.array([[pred_p, pred_c]]))[0]))
        pred_acc = acc_m.predict(x)[0]
        # Count how many observed Pareto points this candidate strictly dominates
        dom = sum(1 for (oe, oa) in pareto if pred_e <= oe and pred_acc >= oa and (pred_e < oe or pred_acc > oa))
        if dom >= 1:
            candidates.append({"topo": topo, "k": k, "R": R, "M": M,
                               "pred_acc": pred_acc, "pred_e": pred_e, "dom": dom})
    if not candidates:
        return pd.DataFrame(columns=["topo","k","R","M","pred_acc","pred_e","dom"])
    return pd.DataFrame(candidates).sort_values(["dom", "pred_e"], ascending=[False, True])


def main():
    # Reasonable cube grids per benchmark
    KS = [3, 5, 7, 10, 15, 20, 22, 30, 36]
    RS = [1, 2, 3, 4, 5]
    MS = [2, 3, 4, 5]

    print("=" * 100)
    print("v6 cross-benchmark HistGBM Pareto-dominant predictions")
    print("=" * 100)

    all_picks = {}
    for bench, dir_ in BENCH_DIRS.items():
        df = load_bench(dir_)
        if df.empty:
            print(f"\n--- {bench}: no data ---"); continue
        cand = predict_pareto_candidates(df, KS, RS, MS)
        print(f"\n--- {bench} ({len(df)} observed cells; {len(cand)} candidates predicted Pareto-dominant) ---")
        if len(cand) == 0:
            print("  No candidates predicted to dominate observed frontier.")
            all_picks[bench] = []
            continue
        # Deduplicate by (pred_acc, pred_e) cluster
        seen_acc_rounded = set()
        unique_picks = []
        for _, r in cand.iterrows():
            key = (round(r["pred_acc"], 1), round(r["pred_e"]))
            if key in seen_acc_rounded: continue
            seen_acc_rounded.add(key)
            unique_picks.append(r)
            if len(unique_picks) >= 4: break
        print(f"  {'cell':<32} {'pred acc':>9} {'pred E':>9} {'dom':>5}")
        for r in unique_picks:
            cell = f"{r['topo']} k={r['k']} R={r['R']} M={r['M']}"
            print(f"  {cell:<32} {r['pred_acc']:>8.1f}% {r['pred_e']:>7.0f}kJ {int(r['dom']):>4d}")
        all_picks[bench] = unique_picks

    # Generate sbatch task table
    print("\n" + "=" * 100)
    print("Recommended v6 sbatch — 12 cells (4 per benchmark × 3 non-SWE benchmarks)")
    print("=" * 100)
    print("Skipping SWE-bench (already cube-sampled). Picks for the other 3:")
    print(f"  {'idx':>3} {'bench':<14} {'topo':<14} {'k':>4} {'R':>3} {'M':>3} {'pred acc':>9} {'pred E':>9}")
    idx = 0
    for bench in ["FanOutQA", "WorkBench", "BrowseComp+"]:
        picks = all_picks.get(bench, [])
        for r in picks:
            print(f"  {idx:>3d} {bench:<14} {r['topo']:<14} {r['k']:>3d} {r['R']:>3d} {r['M']:>3d} {r['pred_acc']:>8.1f}% {r['pred_e']:>7.0f}kJ")
            idx += 1


if __name__ == "__main__":
    main()
