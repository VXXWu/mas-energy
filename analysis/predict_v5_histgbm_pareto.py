"""Generate v5 Pareto-dominant SWE-bench predictions using HistGBM.

Per the 2026-05-29 architecture comparison, HistGBM cut v4 prediction error
30% vs GP. Use HistGBM on the FULL current data (cube interior + v4 cells now
included) to predict Pareto-dominant cells for v5 validation.

Hypothesis: with both architecture switch AND v4 cells now anchoring the
corner regions, v5 predictions should be ~6-12pp accuracy error instead of
v3's 25-48pp.
"""
import json, glob, os, re, itertools
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")


def load_cells():
    cells = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
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
            acc += (1 if d.get("correct") else 0)
            energy += d.get("gpu_dynamic_energy_joules", 0) or 0
            P += d.get("total_prompt_tokens", 0) or 0
            C += d.get("total_completion_tokens", 0) or 0
        if n < 30: continue
        cells.append({"topo": topo, "k": k, "R": R, "M": M,
                      "acc": 100*acc/n, "energy_kJ": energy/n/1000,
                      "P": P/n, "C": C/n, "n": n})
    return pd.DataFrame(cells)


def prep_xy(df, col="acc"):
    X = np.column_stack([
        np.log(df["k"].values), np.log(df["R"].values), np.log(df["M"].values),
        (df["topo"] == "centralized").astype(int).values,
    ])
    return X, df[col].values


def main():
    df = load_cells()
    print(f"Loaded {len(df)} SWE-bench cells (cent + decent, n>=30)")

    # Fit HistGBM for accuracy (handles regime structure)
    # Use HistGBM for P and C (token counts) too — they're smoother than accuracy
    # Then E = α + β·P + γ·C (linear, physically constrained, no negative-energy
    # extrapolation artifacts that pure tree regression on energy produces).
    X, y_acc = prep_xy(df, "acc")
    _, y_P = prep_xy(df, "P")
    _, y_C = prep_xy(df, "C")

    acc_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05, random_state=0,
    ).fit(X, y_acc)
    p_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05, random_state=0,
    ).fit(X, y_P)
    c_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05, random_state=0,
    ).fit(X, y_C)

    # Per-call energy regression (from analysis/key_findings.md Finding 2):
    # E_kJ_per_task = α + β·P + γ·C (P, C are per-task totals)
    # Fit on actual data:
    PC = np.column_stack([df["P"].values, df["C"].values])
    e_y = df["energy_kJ"].values
    from sklearn.linear_model import LinearRegression
    e_reg = LinearRegression().fit(PC, e_y)
    print(f"Energy regression: E_kJ = {e_reg.intercept_:.1f} + {e_reg.coef_[0]:.5f}·P + {e_reg.coef_[1]:.5f}·C")
    print(f"  R² = {e_reg.score(PC, e_y):.3f}")

    # Build candidate grid: all (topo, k, R, M) combinations not already in data
    KS = [1, 2, 3, 5, 7, 10, 15, 20, 22, 25, 30, 36, 50, 75]
    RS = [1, 2, 3, 4, 5, 7, 10]
    MS = [2, 3, 4, 5, 7, 10, 15]
    TOPOS = ["centralized", "decentralized"]

    observed = set((r["topo"], r["k"], r["R"], r["M"]) for _, r in df.iterrows())
    candidates = []
    for topo, k, R, M in itertools.product(TOPOS, KS, RS, MS):
        if (topo, k, R, M) in observed: continue
        x = np.array([[np.log(k), np.log(R), np.log(M),
                       1 if topo == "centralized" else 0]])
        pred_p = max(0, p_model.predict(x)[0])
        pred_c = max(0, c_model.predict(x)[0])
        pred_e = float(e_reg.predict(np.array([[pred_p, pred_c]]))[0])
        # Clamp energy to be non-negative
        pred_e = max(1.0, pred_e)
        candidates.append({"topo": topo, "k": k, "R": R, "M": M,
                           "pred_acc": acc_model.predict(x)[0],
                           "pred_e": pred_e,
                           "pred_P": pred_p, "pred_C": pred_c})
    cand_df = pd.DataFrame(candidates)
    print(f"Generated {len(cand_df)} candidate cells (not in observed data)")

    # Filter: only keep candidates with predicted acc >= 70% (Pareto-relevant region)
    cand_df = cand_df[cand_df["pred_acc"] >= 70].copy()
    print(f"After filter (pred_acc >= 70%): {len(cand_df)} candidates")

    # Compute observed Pareto frontier
    obs_sorted = df.sort_values("energy_kJ").reset_index(drop=True)
    pareto = []
    max_acc = -1
    for _, r in obs_sorted.iterrows():
        if r["acc"] > max_acc:
            pareto.append((r["energy_kJ"], r["acc"]))
            max_acc = r["acc"]

    # For each candidate, count how many observed Pareto points it strictly dominates
    def dominates_count(pe, pa):
        return sum(1 for (oe, oa) in pareto if pe <= oe and pa >= oa and (pe < oe or pa > oa))

    cand_df["dominates"] = cand_df.apply(lambda r: dominates_count(r["pred_e"], r["pred_acc"]), axis=1)
    cand_df = cand_df[cand_df["dominates"] >= 1].copy()
    cand_df = cand_df.sort_values(["dominates", "pred_e"], ascending=[False, True])

    print()
    print("=" * 90)
    print("v5 PARETO-DOMINANT PREDICTIONS (HistGBM, all data including v4)")
    print("=" * 90)
    print(f"  {'cell':<40} {'pred acc':>9} {'pred E':>9} {'dominates':>11}")
    print("  " + "-" * 75)
    for _, r in cand_df.head(20).iterrows():
        cell = f"{r['topo']} k={r['k']} R={r['R']} M={r['M']}"
        print(f"  {cell:<40} {r['pred_acc']:>8.1f}% {r['pred_e']:>7.0f}kJ {r['dominates']:>10d}")

    print()
    print("Top 6 picks for v5 validation sbatch:")
    for _, r in cand_df.head(6).iterrows():
        print(f"  {r['topo']:<14} k={r['k']:>3d} R={r['R']:>2d} M={r['M']:>2d}  → pred {r['pred_acc']:.1f}% @ {r['pred_e']:.0f}kJ  (dominates {r['dominates']})")


if __name__ == "__main__":
    main()
