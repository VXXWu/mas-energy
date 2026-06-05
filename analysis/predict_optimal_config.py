"""Predictive models for optimal (topology, k, R, M) at given energy budgets.

Two models fit per (benchmark, topology):
1. SIMPLE log-log regression:
     log(101 - acc) = a + b_k·log(k) + b_R·log(R) + b_M·log(M)
     log(energy)    = a + b_k·log(k) + b_R·log(R) + b_M·log(M)
2. GAUSSIAN PROCESS (sklearn, Matern 2.5 kernel) on (log k, log R, log M).
   Mean predictions + std for OOS uncertainty.

Outputs per benchmark: in-sample R², LOO MAE, OOS recommendations at energy budgets.
"""
import json, glob, os, re
import numpy as np
import pandas as pd

BENCH = {
    "FanOutQA":   "a5000_fanoutqa_v4",
    "WorkBench":  "a5000_workbench_v2",
    "BrowseComp+":"a5000_browsecomp_pilot",
    "SWE-bench":  "a5000_swebench",
}
DEFAULT_R = 2
DEFAULT_M = 3
TOPOS = ["sas", "independent", "centralized", "decentralized"]
K_GRID = [1,2,3,5,7,10,15,20,30,50,75]
R_GRID = [1,2,3,4,5]
M_GRID = [2,3,4,5,7,10,15,20]
BUDGETS = [50, 100, 200, 400, 800]

cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")


def load_cell(f):
    acc, e = [], []
    for line in open(f):
        try: dd = json.loads(line)
        except: continue
        if dd.get("error"): continue
        a = (float(dd["loose_accuracy"]) if dd.get("loose_accuracy") is not None
             else (1.0 if dd.get("correct") else 0.0))
        acc.append(a); e.append(dd.get("gpu_dynamic_energy_joules",0)/1000)
    if not acc: return None
    return np.mean(acc)*100, np.mean(e), len(acc)


def load_all_cells(bd):
    cells = []
    for f in sorted(glob.glob(f"mas-energy/results/{bd}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = cre.match(os.path.basename(f))
        if not m: continue
        topo, k, R, M = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo == "hybrid": continue
        if topo == "centralized" and R is None and M is None: continue
        out = load_cell(f)
        if not out: continue
        a, e, n = out
        if n < 30: continue
        cells.append({"topo": topo, "k": k,
                      "R": int(R) if R else DEFAULT_R,
                      "M": int(M) if M else DEFAULT_M,
                      "acc": a, "energy": e, "n": n})
    return cells


def build_dense_grid():
    grid = []
    for t in TOPOS:
        for k in K_GRID:
            if t in ("sas", "independent"):
                grid.append({"topo": t, "k": k, "R": DEFAULT_R, "M": DEFAULT_M})
            else:
                for R in R_GRID:
                    for M in M_GRID:
                        grid.append({"topo": t, "k": k, "R": R, "M": M})
    return grid


def fit_loglinear(sub_df, target):
    X = np.column_stack([np.ones(len(sub_df)), np.log(sub_df["k"]),
                          np.log(sub_df["R"]), np.log(sub_df["M"])])
    y = (np.log(np.clip(101 - sub_df["acc"].values, 0.5, None))
         if target == "acc" else np.log(sub_df["energy"].values))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    def predict(c):
        x = np.array([1, np.log(c["k"]), np.log(c["R"]), np.log(c["M"])])
        py = x @ coef
        return (101 - np.exp(py)) if target == "acc" else np.exp(py)
    pred_train = X @ coef
    r2 = (1 - np.sum((y-pred_train)**2)/np.sum((y-y.mean())**2)
          if y.var() > 1e-9 else float('nan'))
    return predict, r2, coef


def fit_gp(sub_df, target):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    X = np.column_stack([np.log(sub_df["k"]), np.log(sub_df["R"]), np.log(sub_df["M"])])
    y = (np.log(np.clip(101 - sub_df["acc"].values, 0.5, None))
         if target == "acc" else np.log(sub_df["energy"].values))
    kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
    try:
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=3, random_state=0)
        gp.fit(X, y)
    except Exception:
        return None, float('nan')
    def predict(c):
        x = np.array([[np.log(c["k"]), np.log(c["R"]), np.log(c["M"])]])
        py, std = gp.predict(x, return_std=True)
        if target == "acc":
            return 101 - np.exp(py[0]), float(std[0])
        return np.exp(py[0]), float(std[0])
    pred_train = gp.predict(X)
    r2 = (1 - np.sum((y-pred_train)**2)/np.sum((y-y.mean())**2)
          if y.var() > 1e-9 else float('nan'))
    return predict, r2


def loo_eval(cells, model_fn):
    errs_a, errs_e = [], []
    df = pd.DataFrame(cells)
    for i, c in enumerate(cells):
        train = df.drop(i).reset_index(drop=True)
        sub = train[train["topo"] == c["topo"]]
        if len(sub) < 4: continue
        try:
            out_a = model_fn(sub, "acc")
            out_e = model_fn(sub, "energy")
            pa_fn = out_a[0]; pe_fn = out_e[0]
            if pa_fn is None or pe_fn is None: continue
            pa = pa_fn(c); pe = pe_fn(c)
            if isinstance(pa, tuple): pa = pa[0]
            if isinstance(pe, tuple): pe = pe[0]
            errs_a.append(abs(pa - c["acc"]))
            errs_e.append(abs(pe - c["energy"]) / max(c["energy"], 1))
        except Exception:
            continue
    return errs_a, errs_e


print("=" * 110)
print("MODEL 1: SIMPLE LOG-LINEAR REGRESSION")
print("=" * 110)
print("log(101-acc) = a + b_k·log(k) + b_R·log(R) + b_M·log(M)")
print("log(energy)  = a + b_k·log(k) + b_R·log(R) + b_M·log(M)")

results = {}
for bench, bd in BENCH.items():
    cells = load_all_cells(bd)
    df = pd.DataFrame(cells)
    print(f"\n{bench} -- {len(cells)} cells")
    fitted = {}
    for topo in TOPOS:
        sub = df[df["topo"] == topo]
        if len(sub) < 4: continue
        pa_fn, r2_a, _ = fit_loglinear(sub, "acc")
        pe_fn, r2_e, coef_e = fit_loglinear(sub, "energy")
        fitted[topo] = (pa_fn, pe_fn)
        print(f"  {topo:<14} n={len(sub):>3}  acc R²={r2_a:.3f}  E R²={r2_e:.3f}  "
              f"β_k_E={coef_e[1]:+.2f} β_R_E={coef_e[2]:+.2f} β_M_E={coef_e[3]:+.2f}")
    results[bench] = (cells, df, fitted)

# OOS recommendations
print("\n" + "=" * 110)
print("OOS RECOMMENDATIONS (simple model)")
print("=" * 110)
for bench, (cells, df, fitted) in results.items():
    grid = build_dense_grid()
    measured = {(c["topo"], c["k"], c["R"], c["M"]) for c in cells}
    for c in grid:
        if c["topo"] not in fitted:
            c["pred_acc"] = float('nan'); c["pred_energy"] = float('nan'); continue
        pa_fn, pe_fn = fitted[c["topo"]]
        try:
            c["pred_acc"] = pa_fn(c); c["pred_energy"] = pe_fn(c)
        except Exception:
            c["pred_acc"] = float('nan'); c["pred_energy"] = float('nan')
        c["is_oos"] = (c["topo"], c["k"], c["R"], c["M"]) not in measured

    print(f"\n{bench}:")
    print(f"  {'Budget':>7}  {'Best observed':<42}  {'Best predicted (any)':<48}  {'Top OOS rec':<48}")
    for B in BUDGETS:
        obs = max((c for c in cells if c["energy"] <= B),
                  key=lambda c: c["acc"], default=None)
        feas = [c for c in grid if not np.isnan(c.get("pred_acc", float('nan')))
                and c["pred_energy"] <= B]
        pred = max(feas, key=lambda c: c["pred_acc"], default=None)
        oos = max([c for c in feas if c.get("is_oos")],
                   key=lambda c: c["pred_acc"], default=None)
        obs_s = (f"{obs['topo'][:5]} k={obs['k']} R={obs['R']} M={obs['M']} "
                 f"→{obs['acc']:.0f}%@{obs['energy']:.0f}kJ") if obs else "—"
        pred_s = (f"{pred['topo'][:5]} k={pred['k']} R={pred['R']} M={pred['M']} "
                  f"→pred {pred['pred_acc']:.0f}%@{pred['pred_energy']:.0f}kJ") if pred else "—"
        oos_s = (f"{oos['topo'][:5]} k={oos['k']} R={oos['R']} M={oos['M']} "
                 f"→pred {oos['pred_acc']:.0f}%@{oos['pred_energy']:.0f}kJ") if oos else "—"
        print(f"  ≤{B:>5}  {obs_s:<42}  {pred_s:<48}  {oos_s:<48}")

# LOO
print("\n" + "=" * 110)
print("LEAVE-ONE-OUT CROSS-VALIDATION (simple)")
print("=" * 110)
print(f"{'Bench':<12}  {'n':>3}  {'median |acc err|':>17}  {'median %E err':>14}")
for bench, (cells, _, _) in results.items():
    ea, ee = loo_eval(cells, fit_loglinear)
    if ea:
        print(f"{bench:<12}  {len(ea):>3}  {np.median(ea):>15.1f}pp  {np.median(ee)*100:>12.0f}%")

print("\n\n" + "=" * 110)
print("MODEL 2: GAUSSIAN PROCESS (Matern 2.5 on log(k,R,M))")
print("=" * 110)
gp_results = {}
for bench, bd in BENCH.items():
    cells = load_all_cells(bd)
    df = pd.DataFrame(cells)
    print(f"\n{bench}:")
    fitted = {}
    for topo in TOPOS:
        sub = df[df["topo"] == topo]
        if len(sub) < 5: continue
        pa_fn, r2_a = fit_gp(sub, "acc")
        pe_fn, r2_e = fit_gp(sub, "energy")
        if pa_fn and pe_fn:
            fitted[topo] = (pa_fn, pe_fn)
            print(f"  {topo:<14} n={len(sub):>3}  acc R²={r2_a:.3f}  E R²={r2_e:.3f}")
    gp_results[bench] = (cells, df, fitted)

print("\n" + "=" * 110)
print("GP OOS RECOMMENDATIONS (top by pred_acc - 0.5*std)")
print("=" * 110)
for bench, (cells, df, fitted) in gp_results.items():
    grid = build_dense_grid()
    measured = {(c["topo"], c["k"], c["R"], c["M"]) for c in cells}
    for c in grid:
        if c["topo"] not in fitted:
            c["pred_acc"] = float('nan'); continue
        pa_fn, pe_fn = fitted[c["topo"]]
        try:
            pa, pa_std = pa_fn(c); pe, _ = pe_fn(c)
            c["pred_acc"] = pa; c["pred_acc_std"] = pa_std; c["pred_energy"] = pe
        except Exception:
            c["pred_acc"] = float('nan')
        c["is_oos"] = (c["topo"], c["k"], c["R"], c["M"]) not in measured

    print(f"\n{bench}:")
    for B in BUDGETS:
        feas = [c for c in grid if not np.isnan(c.get("pred_acc", float('nan')))
                and c["pred_energy"] <= B and c.get("is_oos")]
        if not feas:
            print(f"  ≤{B}: —"); continue
        feas.sort(key=lambda c: c["pred_acc"] - 0.5*c["pred_acc_std"], reverse=True)
        top = feas[:3]
        s = " | ".join(f"{c['topo'][:5]} k{c['k']}R{c['R']}M{c['M']}→"
                       f"{c['pred_acc']:.0f}±{c['pred_acc_std']:.1f}%@{c['pred_energy']:.0f}kJ"
                       for c in top)
        print(f"  ≤{B}: {s}")

# GP LOO
print("\n" + "=" * 110)
print("LEAVE-ONE-OUT CROSS-VALIDATION (GP)")
print("=" * 110)
print(f"{'Bench':<12}  {'n':>3}  {'median |acc err|':>17}  {'median %E err':>14}")
for bench, (cells, _, _) in gp_results.items():
    ea, ee = loo_eval(cells, fit_gp)
    if ea:
        print(f"{bench:<12}  {len(ea):>3}  {np.median(ea):>15.1f}pp  {np.median(ee)*100:>12.0f}%")


# ----------------------------------------------------------------------
# Model 3: Random Forest (handles low-k discontinuity)
# ----------------------------------------------------------------------
def fit_rf(sub_df, target):
    from sklearn.ensemble import RandomForestRegressor
    X = np.column_stack([np.log(sub_df["k"]), np.log(sub_df["R"]), np.log(sub_df["M"])])
    if target == "acc":
        # Predict accuracy directly (not log) so RF can learn the floor at low k
        y = sub_df["acc"].values
    else:
        y = np.log(sub_df["energy"].values)
    if len(sub_df) < 5:
        return None, float('nan')
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                max_features=None, random_state=0)
    rf.fit(X, y)
    def predict(c):
        x = np.array([[np.log(c["k"]), np.log(c["R"]), np.log(c["M"])]])
        py = rf.predict(x)[0]
        if target == "acc":
            return float(np.clip(py, 0, 100)), 0.0
        return float(np.exp(py)), 0.0
    pred_train = rf.predict(X)
    r2 = (1 - np.sum((y-pred_train)**2)/np.sum((y-y.mean())**2)
          if y.var() > 1e-9 else float('nan'))
    return predict, r2


print("\n\n" + "=" * 110)
print("MODEL 3: RANDOM FOREST (handles low-k cliff, captures non-smooth surface)")
print("=" * 110)
rf_results = {}
for bench, bd in BENCH.items():
    cells = load_all_cells(bd)
    df = pd.DataFrame(cells)
    print(f"\n{bench}:")
    fitted = {}
    for topo in TOPOS:
        sub = df[df["topo"] == topo]
        if len(sub) < 5: continue
        pa_fn, r2_a = fit_rf(sub, "acc")
        pe_fn, r2_e = fit_rf(sub, "energy")
        if pa_fn and pe_fn:
            fitted[topo] = (pa_fn, pe_fn)
            print(f"  {topo:<14} n={len(sub):>3}  acc R²={r2_a:.3f}  E R²={r2_e:.3f}")
    rf_results[bench] = (cells, df, fitted)

print("\n" + "=" * 110)
print("RF OOS RECOMMENDATIONS")
print("=" * 110)
for bench, (cells, df, fitted) in rf_results.items():
    grid = build_dense_grid()
    measured = {(c["topo"], c["k"], c["R"], c["M"]) for c in cells}
    for c in grid:
        if c["topo"] not in fitted:
            c["pred_acc"] = float('nan'); continue
        pa_fn, pe_fn = fitted[c["topo"]]
        try:
            pa, _ = pa_fn(c); pe, _ = pe_fn(c)
            c["pred_acc"] = pa; c["pred_energy"] = pe
        except Exception:
            c["pred_acc"] = float('nan')
        c["is_oos"] = (c["topo"], c["k"], c["R"], c["M"]) not in measured

    print(f"\n{bench}:")
    for B in BUDGETS:
        feas = [c for c in grid if not np.isnan(c.get("pred_acc", float('nan')))
                and c["pred_energy"] <= B and c.get("is_oos")]
        if not feas:
            print(f"  ≤{B}: —"); continue
        feas.sort(key=lambda c: c["pred_acc"], reverse=True)
        top = feas[:3]
        s = " | ".join(f"{c['topo'][:5]} k{c['k']}R{c['R']}M{c['M']}→"
                       f"{c['pred_acc']:.0f}%@{c['pred_energy']:.0f}kJ"
                       for c in top)
        print(f"  ≤{B}: {s}")

# RF LOO
print("\n" + "=" * 110)
print("LEAVE-ONE-OUT CROSS-VALIDATION (RF)")
print("=" * 110)
print(f"{'Bench':<12}  {'n':>3}  {'median |acc err|':>17}  {'median %E err':>14}")
for bench, (cells, _, _) in rf_results.items():
    ea, ee = loo_eval(cells, fit_rf)
    if ea:
        print(f"{bench:<12}  {len(ea):>3}  {np.median(ea):>15.1f}pp  {np.median(ee)*100:>12.0f}%")

# Cross-model comparison summary
print("\n\n" + "=" * 110)
print("MODEL COMPARISON: LOO median |acc err| (pp)")
print("=" * 110)
print(f"{'Bench':<12}  {'simple':>8}  {'GP':>8}  {'RF':>8}")
all_errs = {}
for bench in BENCH:
    cells = load_all_cells(BENCH[bench])
    e_simple = loo_eval(cells, fit_loglinear)[0]
    e_gp = loo_eval(cells, fit_gp)[0]
    e_rf = loo_eval(cells, fit_rf)[0]
    all_errs[bench] = (e_simple, e_gp, e_rf)
    print(f"{bench:<12}  {np.median(e_simple):>7.1f}  {np.median(e_gp):>7.1f}  {np.median(e_rf):>7.1f}")

# Validation point check (specific configs where we have ground truth)
print("\n" + "=" * 110)
print("PREDICTION ACCURACY ON THE 4 VALIDATION CELLS (SWE-bench)")
print("=" * 110)
print(f"{'Cell':<40} {'actual':>7} {'simple':>10} {'GP':>10} {'RF':>10}")
val_cells = [
    ("decentralized", 3, 4, 2),
    ("decentralized", 5, 5, 2),
    ("decentralized", 75, 1, 4),
    ("centralized",   50, 2, 2),
]
for topo, k, R, M in val_cells:
    cells_all = load_all_cells(BENCH["SWE-bench"])
    target = next((c for c in cells_all if c["topo"]==topo and c["k"]==k
                    and c["R"]==R and c["M"]==M), None)
    if not target: continue
    # Train without this cell
    df = pd.DataFrame([c for c in cells_all if not (
        c["topo"]==topo and c["k"]==k and c["R"]==R and c["M"]==M)])
    sub = df[df["topo"] == topo]
    if len(sub) < 4: continue
    p_simple, _, _ = fit_loglinear(sub, "acc")
    p_gp_fn, _ = fit_gp(sub, "acc")
    p_rf_fn, _ = fit_rf(sub, "acc")
    pred_s = p_simple(target)
    pred_g = p_gp_fn(target)[0]
    pred_r = p_rf_fn(target)[0]
    cell_label = f"{topo} k={k} R={R} M={M}"
    print(f"{cell_label:<40} {target['acc']:>6.1f}% {pred_s:>9.1f}% {pred_g:>9.1f}% {pred_r:>9.1f}%")
