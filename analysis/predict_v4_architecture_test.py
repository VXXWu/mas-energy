"""Head-to-head test: GP vs Gradient Boosting vs Random Forest on the v4 cells.

Premise: v3 (smooth GP + floor + saturation) failed catastrophically on 6 v4
Pareto-dominant predictions (25-48pp accuracy errors). Hypothesis: the failure
is architectural — the SWE-bench accuracy surface has sharp AND-gate-like
regime structure (k_gate × R_gate × M_gate) that smooth GPs cannot represent.

This script trains three model families on the SAME training data v3 used
(standard grid + 37 cube-interior cells, EXCLUDING the 6 v4 cells), then
predicts the 6 v4 cells with each, then reports which architecture best
captures the regime structure.

Training architectures:
  1. GP (matches v3): RBF kernel over (log k, log R, log M), per-topology
  2. Gradient Boosting (HistGradientBoostingRegressor): tree-based, handles
     sharp regime boundaries via splits, naturally captures interactions
  3. Random Forest: ensemble of trees, smoother than single tree but still
     captures regime boundaries

If trees substantially beat GP on the v4 cells, architecture is the bottleneck
and v5 Pareto-dominant predictions should use the tree model. If trees also
fail, the regime structure is intrinsically unlearnable from the available
data (then the reframe-as-finding stands).
"""
import json, glob, os, re
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

# v4 cells (held out from training, used for evaluation)
V4_CELLS = {
    ("centralized",   50, 1, 2),
    ("decentralized",  3, 5, 4),
    ("centralized",   30, 1, 2),
    ("decentralized",  3, 5, 2),
    ("decentralized",  3, 5, 3),
    ("decentralized",  3, 4, 4),
}


def load_cells():
    cells = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo not in ("centralized", "decentralized"): continue
        R = int(r_str) if r_str else 2
        M = int(m_str) if m_str else 3
        n=0; acc=0; energy=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            n+=1; acc+=(1 if d.get("correct") else 0); energy+=d.get("gpu_dynamic_energy_joules",0) or 0
        if n < 30: continue
        cells.append({
            "topo": topo, "k": k, "R": R, "M": M,
            "acc": 100*acc/n, "energy_kJ": energy/n/1000, "n": n,
        })
    return pd.DataFrame(cells)


def prep_xy(df):
    """Return X (log-scale features) and y (accuracy). One-hot encodes topology."""
    X = np.column_stack([
        np.log(df["k"].values),
        np.log(df["R"].values),
        np.log(df["M"].values),
        (df["topo"] == "centralized").astype(int).values,
    ])
    y = df["acc"].values
    return X, y


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main():
    df = load_cells()
    print(f"Loaded {len(df)} cells (cent + decent, n>=30)")

    # Mark v4 cells
    df["is_v4"] = df.apply(lambda r: (r["topo"], r["k"], r["R"], r["M"]) in V4_CELLS, axis=1)
    train = df[~df["is_v4"]].copy()
    test = df[df["is_v4"]].copy()
    print(f"Training cells: {len(train)} (cube interior, standard grid, etc.)")
    print(f"v4 test cells: {len(test)}")
    print()

    X_train, y_train = prep_xy(train)
    X_test, y_test = prep_xy(test)

    models = {
        "GP (matches v3)": GaussianProcessRegressor(
            kernel=ConstantKernel(50.0) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0]),
            normalize_y=True, n_restarts_optimizer=3, random_state=0,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05, random_state=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=0,
        ),
    }

    results = {}
    for name, model in models.items():
        preds = fit_predict(model, X_train, y_train, X_test)
        errors = preds - y_test
        results[name] = (preds, errors)
        print(f"=== {name} ===")
        print(f"  {'cell':<35} {'true':>6} {'pred':>6} {'err':>7}")
        for (_, row), p, e in zip(test.iterrows(), preds, errors):
            cell = f"{row['topo']} k={row['k']} R={row['R']} M={row['M']}"
            print(f"  {cell:<35} {row['acc']:>5.1f}% {p:>5.1f}% {e:>+6.1f}pp")
        median_abs_err = np.median(np.abs(errors))
        max_abs_err = np.max(np.abs(errors))
        print(f"  median |err| = {median_abs_err:.1f}pp,  max |err| = {max_abs_err:.1f}pp")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY: which architecture best predicts the v4 cells?")
    print("=" * 70)
    print(f"  {'architecture':<25} {'median |err|':>13} {'max |err|':>11}")
    for name, (preds, errors) in results.items():
        median_abs_err = np.median(np.abs(errors))
        max_abs_err = np.max(np.abs(errors))
        print(f"  {name:<25} {median_abs_err:>12.1f}pp {max_abs_err:>10.1f}pp")
    print()
    print("Interpretation: if a tree-based model has substantially lower v4 error,")
    print("the v3 GP architecture is the bottleneck. Tree splits naturally capture")
    print("the AND-gate regime structure (k_gate AND R_gate AND M_gate ⇒ high acc).")


if __name__ == "__main__":
    main()
