"""Tests whether iterative point-collection actually improves predictions.

Setup: we have 89 training cells + 6 v4 cells (from yesterday's failed validation).
Question: if we ADD the v4 cells to training, do predictions in NEARBY untrained
regions improve? This tests the user's "will the model improve iteratively"
question directly.

Approach:
1. Pick a 7th "v5 test cell" near a v4 cell (similar (k, R, M) region)
2. Train models WITHOUT v4 cells, predict v5 test cell → record error
3. Train models WITH v4 cells included, predict v5 test cell → record error
4. Compare. Improvement = the iterative-learning works.

But we don't have v5 actuals to compare against. Instead: leave-one-v4-out
cross-validation. For each v4 cell, train on (everything else including the
other 5 v4 cells), predict that v4 cell, see if having the 5 nearest neighbors
helped.

If LOO-v4 errors are much smaller than v4-as-held-out errors, then YES,
iterative collection in the corner regions does converge the model.
"""
import json, glob, os, re
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

V4_CELLS = [
    ("centralized",   50, 1, 2),
    ("decentralized",  3, 5, 4),
    ("centralized",   30, 1, 2),
    ("decentralized",  3, 5, 2),
    ("decentralized",  3, 5, 3),
    ("decentralized",  3, 4, 4),
]


def load_cells():
    cells = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo not in ("centralized", "decentralized"): continue
        R = int(r_str) if r_str else 2
        M = int(m_str) if m_str else 3
        n=0; acc=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            n+=1; acc+=(1 if d.get("correct") else 0)
        if n < 30: continue
        cells.append({"topo": topo, "k": k, "R": R, "M": M, "acc": 100*acc/n, "n": n})
    return pd.DataFrame(cells)


def prep_xy(df):
    X = np.column_stack([
        np.log(df["k"].values), np.log(df["R"].values), np.log(df["M"].values),
        (df["topo"] == "centralized").astype(int).values,
    ])
    y = df["acc"].values
    return X, y


def fit_predict(model_factory, X_tr, y_tr, X_te):
    m = model_factory()
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def make_models():
    return {
        "GP": lambda: GaussianProcessRegressor(
            kernel=ConstantKernel(50.0) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0]),
            normalize_y=True, n_restarts_optimizer=3, random_state=0,
        ),
        "HistGBM": lambda: HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05, random_state=0,
        ),
        "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0),
    }


def main():
    df = load_cells()
    df["v4_idx"] = -1
    for i, (t, k, r, m) in enumerate(V4_CELLS):
        mask = (df["topo"] == t) & (df["k"] == k) & (df["R"] == r) & (df["M"] == m)
        df.loc[mask, "v4_idx"] = i
    v4_present = sorted(df[df["v4_idx"] >= 0]["v4_idx"].unique())
    print(f"Loaded {len(df)} cells. v4 cells present: {len(v4_present)} of 6")
    print()

    # --- Experiment 1: ALL v4 held out (baseline = yesterday's v4 failure) ---
    train = df[df["v4_idx"] < 0]
    test = df[df["v4_idx"] >= 0].copy()
    X_tr, y_tr = prep_xy(train)
    X_te, y_te = prep_xy(test)

    print("=" * 80)
    print("EXP 1 (baseline): ALL 6 v4 cells held out — replays yesterday's v4 test")
    print("=" * 80)
    models = make_models()
    exp1 = {}
    for name, factory in models.items():
        pred = fit_predict(factory, X_tr, y_tr, X_te)
        err = pred - y_te
        exp1[name] = (np.median(np.abs(err)), np.max(np.abs(err)))
        print(f"  {name:<10}: median |err|={exp1[name][0]:.1f}pp, max |err|={exp1[name][1]:.1f}pp")

    print()
    print("=" * 80)
    print("EXP 2: Leave-one-v4-out — train on 5 v4 + all others, predict the 1 held-out")
    print("(answers: does collecting v4 cells help predict NEARBY cube-corner cells?)")
    print("=" * 80)
    exp2_errs = {n: [] for n in models}
    for hold_idx in v4_present:
        train2 = df[df["v4_idx"] != hold_idx]
        test2 = df[df["v4_idx"] == hold_idx].copy()
        X_tr2, y_tr2 = prep_xy(train2)
        X_te2, y_te2 = prep_xy(test2)
        cell = test2.iloc[0]
        cell_label = f"{cell['topo']} k={cell['k']} R={cell['R']} M={cell['M']}"
        line = f"  hold-out {cell_label:<32} true={cell['acc']:.1f}%"
        for name, factory in models.items():
            pred = fit_predict(factory, X_tr2, y_tr2, X_te2)[0]
            err = pred - cell["acc"]
            exp2_errs[name].append(abs(err))
            line += f"  | {name}: pred={pred:.1f}% (err {err:+.1f}pp)"
        print(line)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'arch':<12} {'EXP 1 (no v4)':>18} {'EXP 2 (5 of 6 v4)':>20} {'improvement':>14}")
    for name in models:
        e1 = exp1[name][0]
        e2 = np.median(exp2_errs[name])
        delta = e1 - e2
        print(f"  {name:<12} {e1:>15.1f}pp {e2:>17.1f}pp {delta:>+12.1f}pp")
    print()
    print("Interpretation: if EXP 2 error is much lower than EXP 1, the iterative")
    print("learning works — collecting just 5 nearby corner cells dramatically")
    print("improves predictions for the 6th. If EXP 2 ≈ EXP 1, more data in the")
    print("corner doesn't help because the regime structure isn't capturable by")
    print("the current architectures even with neighbors.")


if __name__ == "__main__":
    main()
