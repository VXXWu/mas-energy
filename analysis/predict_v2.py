"""Improved predictive models, addressing v1 failure modes:
1. Two-stage model: predict (P, C) → apply validated energy regression
2. Floor + saturation: explicit accuracy floor at low k, saturation above

Test on the same 7 SWE-bench validation cells.
"""
import json, glob, os, re
import numpy as np
import pandas as pd
from collections import defaultdict

BENCH = {
    "FanOutQA":   "a5000_fanoutqa_v4",
    "WorkBench":  "a5000_workbench_v2",
    "BrowseComp+":"a5000_browsecomp_pilot",
    "SWE-bench":  "a5000_swebench",
}
DEFAULT_R = 2; DEFAULT_M = 3
TOPOS = ["sas", "independent", "centralized", "decentralized"]
cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

# Per-call energy regression coefficients (from existing fit, R²=0.88)
A, B, C, D = -23, 0.020, 5.14, 5.73e-5

def load_cell(f):
    """Return (mean_acc%, mean_energy_kJ, mean_P, mean_C, n)."""
    acc, e, p, c = [], [], [], []
    for line in open(f):
        try: dd = json.loads(line)
        except: continue
        if dd.get("error"): continue
        a = (float(dd["loose_accuracy"]) if dd.get("loose_accuracy") is not None
             else (1.0 if dd.get("correct") else 0.0))
        acc.append(a)
        e.append(dd.get("gpu_dynamic_energy_joules",0)/1000)
        p.append(dd.get("total_prompt_tokens",0))
        c.append(dd.get("total_completion_tokens",0))
    if not acc: return None
    return np.mean(acc)*100, np.mean(e), np.mean(p), np.mean(c), len(acc)

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
        a, e, P, Cc, n = out
        if n < 30: continue
        cells.append({"topo": topo, "k": k,
                      "R": int(R) if R else DEFAULT_R,
                      "M": int(M) if M else DEFAULT_M,
                      "acc": a, "energy": e, "P": P, "C": Cc, "n": n})
    return cells

# ----------------------------------------------------------------------
# Model A: Two-stage (P, C → E via regression)
# ----------------------------------------------------------------------
def fit_pc_predictor(sub_df):
    """Predict log(P) and log(C) from (log k, log R, log M). Returns (predict_PC_fn, R² for P, R² for C)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    X = np.column_stack([np.log(sub_df["k"]), np.log(sub_df["R"]), np.log(sub_df["M"])])
    yP = np.log(sub_df["P"].values.clip(1))
    yC = np.log(sub_df["C"].values.clip(1))
    kernel_P = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
    kernel_C = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
    import warnings; warnings.filterwarnings("ignore")
    gp_P = GaussianProcessRegressor(kernel=kernel_P, normalize_y=True,
                                     n_restarts_optimizer=3, random_state=0)
    gp_C = GaussianProcessRegressor(kernel=kernel_C, normalize_y=True,
                                     n_restarts_optimizer=3, random_state=0)
    gp_P.fit(X, yP); gp_C.fit(X, yC)
    pP = np.exp(gp_P.predict(X)); pC = np.exp(gp_C.predict(X))
    r2_P = 1 - np.sum((sub_df["P"].values - pP)**2) / np.sum((sub_df["P"].values - sub_df["P"].mean())**2)
    r2_C = 1 - np.sum((sub_df["C"].values - pC)**2) / np.sum((sub_df["C"].values - sub_df["C"].mean())**2)
    # Fit per-task energy regression on the same training set: E_kJ = α + β*P + γ*C
    # (linear since per-task aggregate doesn't preserve the per-call PC interaction)
    XE = np.column_stack([np.ones(len(sub_df)), sub_df["P"].values, sub_df["C"].values])
    yE = sub_df["energy"].values  # already in kJ
    coef_E, *_ = np.linalg.lstsq(XE, yE, rcond=None)
    def predict(c):
        x = np.array([[np.log(c["k"]), np.log(c["R"]), np.log(c["M"])]])
        P = float(np.exp(gp_P.predict(x)[0]))
        Cc = float(np.exp(gp_C.predict(x)[0]))
        E_kJ = float(coef_E[0] + coef_E[1]*P + coef_E[2]*Cc)
        return P, Cc, E_kJ
    return predict, r2_P, r2_C

# ----------------------------------------------------------------------
# Model B: Floor + Saturation accuracy model
# ----------------------------------------------------------------------
def fit_floor_saturation(sub_df, k_floor_threshold=0.1):
    """Find empirical k_min such that for k>=k_min, the saturation curve
    acc = ceiling - drop·E^(-α) fits well. Below k_min, predict 0 + small offset.
    """
    from scipy.optimize import curve_fit
    # Identify the floor: smallest k where accuracy > k_floor_threshold of max accuracy
    max_acc = sub_df["acc"].max()
    threshold = k_floor_threshold * max_acc
    valid_df = sub_df[sub_df["acc"] >= threshold].copy()
    if len(valid_df) < 4:
        valid_df = sub_df.copy()
    # Saturation in E space
    E = valid_df["energy"].values; acc = valid_df["acc"].values
    def sat(E, ceiling, drop, alpha):
        return ceiling - drop * np.power(E, -alpha)
    try:
        popt, _ = curve_fit(sat, E, acc, p0=[acc.max(), acc.max()-acc.min()+1, 0.3],
                            maxfev=5000, bounds=([0, 0, 0.01], [105, 200, 3.0]))
    except Exception:
        # Fall back to log-linear in E
        coefs = np.polyfit(np.log(E), acc, 1)
        popt = (acc.max(), max(acc)-min(acc), 0.3)  # placeholder; use lin instead
        def predict(actual_E_kJ, k):
            if k < min(valid_df["k"]): return max(0, acc.min() - 5)
            return min(acc.max() + 5, coefs[0]*np.log(actual_E_kJ) + coefs[1])
        return predict, float("nan")
    # The "floor" is the minimum measured k where acc>=threshold
    k_min = valid_df["k"].min()
    def predict(actual_E_kJ, k):
        if k < k_min:
            # Below floor, accuracy is near 0 (or the floor value)
            return max(0.0, sub_df[sub_df["k"] < k_min]["acc"].mean() if (sub_df["k"] < k_min).any() else 0.0)
        return float(np.clip(sat(actual_E_kJ, *popt), 0, 100))
    # Compute R² on valid
    pred = sat(E, *popt)
    r2 = 1 - np.sum((acc-pred)**2)/np.sum((acc-acc.mean())**2)
    return predict, r2

# Run validation
print("=" * 130)
print("TWO-STAGE MODEL (predict P,C → apply energy regression)")
print("=" * 130)
swe = load_all_cells(BENCH["SWE-bench"])
df = pd.DataFrame(swe)

# Fit P,C predictors per topology
fitted_pc = {}
for topo in TOPOS:
    sub = df[df["topo"] == topo]
    if len(sub) < 5: continue
    pred_fn, r2_P, r2_C = fit_pc_predictor(sub)
    fitted_pc[topo] = pred_fn
    print(f"  {topo:<14} n={len(sub):>3}  P R²={r2_P:.3f}  C R²={r2_C:.3f}")

# Fit floor+saturation accuracy per topology
print("\n" + "=" * 130)
print("FLOOR + SATURATION MODEL")
print("=" * 130)
fitted_sat = {}
for topo in TOPOS:
    sub = df[df["topo"] == topo]
    if len(sub) < 5: continue
    pred_fn, r2 = fit_floor_saturation(sub)
    fitted_sat[topo] = pred_fn
    print(f"  {topo:<14} n={len(sub):>3}  saturation R²={r2:.3f}")

# Validation cells with known actuals
validations = [
    ("decentralized", 3,  4, 2, 26.0,  35.0),
    ("decentralized", 5,  5, 2, 60.0,  73.0),
    ("decentralized", 75, 1, 4, 96.0, 571.0),
    ("centralized",   50, 2, 2, 98.0, 213.0),
    ("decentralized", 75, 1, 2, 98.0, 264.0),
    ("centralized",   75, 2, 2, 90.0, 294.0),
    ("decentralized", 50, 1, 4, 90.0, 453.0),
]

# IMPORTANT: hold out each validation cell from training, refit, predict
print("\n" + "=" * 130)
print("HOLD-OUT VALIDATION (refit without each cell, predict it)")
print("=" * 130)
print(f"{'Cell':<40} {'Act acc':>7} {'Act E':>7}  {'Pred acc (sat)':>15} {'Acc miss':>9}  {'Pred E (2-stage)':>17} {'E miss':>8}")
print("-"*130)

for topo, k, R, M, actual_a, actual_e in validations:
    # Exclude this cell from training
    holdout_mask = ~((df["topo"]==topo) & (df["k"]==k) & (df["R"]==R) & (df["M"]==M))
    df_train = df[holdout_mask].copy()
    sub = df_train[df_train["topo"] == topo]
    if len(sub) < 5:
        print(f"  Skip {topo} k={k} R={R} M={M}: insufficient training data")
        continue

    # Refit P,C predictor and saturation on training set
    pc_pred, _, _ = fit_pc_predictor(sub)
    sat_pred, _ = fit_floor_saturation(sub)

    # Predict
    target = {"k": k, "R": R, "M": M}
    P_pred, C_pred, E_pred_kJ = pc_pred(target)
    acc_pred = sat_pred(E_pred_kJ, k)

    acc_miss = actual_a - acc_pred
    e_miss = (E_pred_kJ - actual_e) / actual_e * 100
    label = f"{topo} k={k} R={R} M={M}"
    print(f"{label:<40} {actual_a:>6.1f}% {actual_e:>6.0f}kJ  "
          f"{acc_pred:>13.1f}%  {acc_miss:>+7.1f}pp  "
          f"{E_pred_kJ:>15.0f}kJ {e_miss:>+6.0f}%")

# ----------------------------------------------------------------------
# Model D: Unified C-saturation (single conceptual story, no hybrid)
#   Step 1: (topo, k, R, M) → (P, C)        GP
#   Step 2: C → accuracy                     saturation curve per topology
#   Step 3: (P, C) → energy                  linear regression
# All three steps are interpretable and mechanistically grounded.
# Low-k cliff emerges automatically: small predicted C → small predicted acc.
# ----------------------------------------------------------------------
def fit_c_saturation(sub_df):
    """acc = ceiling - drop * C^(-α) per topology."""
    from scipy.optimize import curve_fit
    C_vals = sub_df["C"].values.astype(float)
    acc_vals = sub_df["acc"].values.astype(float)
    def sat(C, ceiling, drop, alpha):
        return ceiling - drop * np.power(C, -alpha)
    try:
        popt, _ = curve_fit(sat, C_vals, acc_vals,
                            p0=[acc_vals.max()+1, 30, 0.3],
                            maxfev=10000,
                            bounds=([0, 0, 0.001], [105, 1000, 5.0]))
    except Exception:
        return None, float("nan"), None
    pred_train = sat(C_vals, *popt)
    ss_tot = np.sum((acc_vals - acc_vals.mean())**2)
    ss_res = np.sum((acc_vals - pred_train)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    def predict(C):
        if C <= 0: return 0.0
        return float(np.clip(sat(C, *popt), 0, 100))
    return predict, r2, popt

print("\n" + "=" * 130)
print("UNIFIED C-SATURATION MODEL  (acc = ceiling - drop·C^(-α), per topology)")
print("=" * 130)
for topo in TOPOS:
    sub = df[df["topo"] == topo]
    if len(sub) < 5: continue
    _, r2, popt = fit_c_saturation(sub)
    if popt is not None:
        print(f"  {topo:<14} n={len(sub):>3}  R²={r2:.3f}  "
              f"ceiling={popt[0]:.1f}%  drop={popt[1]:.2f}  α={popt[2]:.3f}")

print("\n" + "=" * 130)
print("HOLD-OUT VALIDATION: Unified C-Saturation Model")
print("=" * 130)
print(f"{'Cell':<40} {'Act acc':>7} {'Act E':>7}  {'Pred C':>8}  {'Pred acc':>9} {'Acc miss':>9}  {'Pred E':>8} {'E miss':>8}")
print("-"*130)
for topo, k, R, M, actual_a, actual_e in validations:
    holdout_mask = ~((df["topo"]==topo) & (df["k"]==k) & (df["R"]==R) & (df["M"]==M))
    df_train = df[holdout_mask].copy()
    sub = df_train[df_train["topo"] == topo]
    if len(sub) < 5: continue
    pc_pred, _, _ = fit_pc_predictor(sub)
    csat_pred, _, _ = fit_c_saturation(sub)
    if csat_pred is None: continue
    target = {"k": k, "R": R, "M": M}
    P_pred, C_pred, E_pred_kJ = pc_pred(target)
    acc_pred = csat_pred(C_pred)
    acc_miss = actual_a - acc_pred
    e_miss = (E_pred_kJ - actual_e) / actual_e * 100
    label = f"{topo} k={k} R={R} M={M}"
    print(f"{label:<40} {actual_a:>6.1f}% {actual_e:>6.0f}kJ  "
          f"{C_pred:>7.0f}  {acc_pred:>8.1f}% {acc_miss:>+7.1f}pp  "
          f"{E_pred_kJ:>6.0f}kJ {e_miss:>+6.0f}%")

# ----------------------------------------------------------------------
# Model E: HYBRID for comparison (already implemented above as fit_hybrid_acc)
# ----------------------------------------------------------------------
def fit_hybrid_acc(sub_df, k_floor_threshold=0.1):
    """GP on (log k, log R, log M) for accuracy directly, plus explicit floor at low k."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    import warnings; warnings.filterwarnings("ignore")
    X = np.column_stack([np.log(sub_df["k"]), np.log(sub_df["R"]), np.log(sub_df["M"])])
    y = np.log(np.clip(101 - sub_df["acc"].values, 0.5, None))
    kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3, random_state=0)
    gp.fit(X, y)
    # Determine floor (where accuracy collapses)
    max_acc = sub_df["acc"].max()
    threshold = k_floor_threshold * max_acc
    above_floor = sub_df[sub_df["acc"] >= threshold]
    k_min = above_floor["k"].min() if len(above_floor) else sub_df["k"].min()
    # Empirical low-k accuracy
    below = sub_df[sub_df["k"] < k_min]
    floor_acc = below["acc"].mean() if len(below) else 0.0

    def predict(c):
        if c["k"] < k_min:
            return float(floor_acc)
        x = np.array([[np.log(c["k"]), np.log(c["R"]), np.log(c["M"])]])
        py = gp.predict(x)[0]
        return float(np.clip(101 - np.exp(py), 0, 100))
    return predict, k_min, floor_acc

# Apply hybrid to validation cells
print("\n" + "=" * 130)
print("HYBRID MODEL (GP-accuracy + low-k floor + two-stage energy)")
print("=" * 130)
print(f"{'Cell':<40} {'Act acc':>7} {'Act E':>7}  {'Hybrid acc':>11} {'Acc miss':>9}  {'Hybrid E':>10} {'E miss':>8}")
print("-"*130)
for topo, k, R, M, actual_a, actual_e in validations:
    holdout_mask = ~((df["topo"]==topo) & (df["k"]==k) & (df["R"]==R) & (df["M"]==M))
    df_train = df[holdout_mask].copy()
    sub = df_train[df_train["topo"] == topo]
    if len(sub) < 5: continue
    pc_pred, _, _ = fit_pc_predictor(sub)
    acc_pred_fn, k_min, floor_acc = fit_hybrid_acc(sub)
    target = {"k": k, "R": R, "M": M}
    _, _, E_pred_kJ = pc_pred(target)
    acc_pred = acc_pred_fn(target)
    acc_miss = actual_a - acc_pred
    e_miss = (E_pred_kJ - actual_e) / actual_e * 100
    label = f"{topo} k={k} R={R} M={M}"
    print(f"{label:<40} {actual_a:>6.1f}% {actual_e:>6.0f}kJ  "
          f"{acc_pred:>10.1f}% {acc_miss:>+7.1f}pp  "
          f"{E_pred_kJ:>8.0f}kJ {e_miss:>+6.0f}%")

# LOO comparison
print("\n" + "=" * 130)
print("LEAVE-ONE-OUT COMPARISON (SWE-bench): old GP vs floor+sat+2stage vs hybrid")
print("=" * 130)
errs_acc_sat, errs_e_2stage = [], []
errs_acc_hybrid = []
for i, c in enumerate(swe):
    sub = df.drop(i).reset_index(drop=True)
    sub = sub[sub["topo"] == c["topo"]]
    if len(sub) < 5: continue
    try:
        pc_pred, _, _ = fit_pc_predictor(sub)
        sat_pred, _ = fit_floor_saturation(sub)
        hybrid_pred, _, _ = fit_hybrid_acc(sub)
        _, _, E_pred = pc_pred(c)
        acc_sat = sat_pred(E_pred, c["k"])
        acc_hyb = hybrid_pred(c)
        errs_acc_sat.append(abs(acc_sat - c["acc"]))
        errs_acc_hybrid.append(abs(acc_hyb - c["acc"]))
        errs_e_2stage.append(abs(E_pred - c["energy"]) / max(c["energy"], 1))
    except Exception:
        continue

print(f"  Old GP (direct):           n=75   median |acc err|=4.4pp    median %E err=~5%")
print(f"  Two-stage + floor+sat:     n={len(errs_acc_sat):>3}   "
      f"median |acc err|={np.median(errs_acc_sat):.1f}pp   "
      f"median %E err={np.median(errs_e_2stage)*100:.0f}%")
print(f"  HYBRID (GP+floor+2stg):    n={len(errs_acc_hybrid):>3}   "
      f"median |acc err|={np.median(errs_acc_hybrid):.1f}pp   "
      f"median %E err={np.median(errs_e_2stage)*100:.0f}%")

# Find Pareto-dominant OOS predictions
print("\n" + "=" * 130)
print("PARETO-DOMINANT PREDICTIONS (OOS configs predicted to beat observed frontier)")
print("=" * 130)
# Fit final models on ALL data (no holdout)
final_models = {}
for topo in TOPOS:
    sub = df[df["topo"] == topo]
    if len(sub) < 5: continue
    pc_pred, _, _ = fit_pc_predictor(sub)
    acc_pred, k_min, floor_acc = fit_hybrid_acc(sub)
    final_models[topo] = (pc_pred, acc_pred)

# Observed Pareto frontier
cells_sorted = sorted([{"topo":c["topo"],"k":c["k"],"R":c["R"],"M":c["M"],
                         "acc":c["acc"],"energy":c["energy"]} for c in swe],
                       key=lambda c: c["energy"])
pareto_obs = []
max_acc = -1
for c in cells_sorted:
    if c["acc"] > max_acc:
        pareto_obs.append(c); max_acc = c["acc"]
print(f"\nObserved Pareto frontier ({len(pareto_obs)} points):")
for c in pareto_obs:
    print(f"  {c['energy']:>6.0f}kJ {c['acc']:>5.1f}%  {c['topo']:<14} k={c['k']} R={c['R']} M={c['M']}")

# Build dense candidate grid
K_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
R_GRID = [1, 2, 3, 4, 5]
M_GRID = [2, 3, 4, 5, 7, 10, 15, 20]
measured = {(c["topo"], c["k"], c["R"], c["M"]) for c in swe}

def dominates(pred, obs):
    """pred Pareto-dominates obs if pred has lower energy AND higher accuracy."""
    return pred["E"] <= obs["energy"] and pred["acc"] >= obs["acc"] and \
           (pred["E"] < obs["energy"] or pred["acc"] > obs["acc"])

predictions = []
for topo in ["centralized", "decentralized"]:
    if topo not in final_models: continue
    pc_pred, acc_pred = final_models[topo]
    for k in K_GRID:
        for R in R_GRID:
            for M in M_GRID:
                if (topo, k, R, M) in measured: continue
                target = {"k": k, "R": R, "M": M}
                _, _, E = pc_pred(target)
                a = acc_pred(target)
                predictions.append({"topo": topo, "k": k, "R": R, "M": M,
                                     "acc": a, "E": E})

# Find predictions that Pareto-dominate ANY observed point
new_pareto = []
for p in predictions:
    # Check if p dominates any observed point (= would improve the frontier)
    # AND p is not dominated by any other observed point
    dominated_by_obs = any(o["acc"] >= p["acc"] and o["energy"] <= p["E"]
                            and (o["acc"] > p["acc"] or o["energy"] < p["E"])
                            for o in pareto_obs)
    if dominated_by_obs: continue
    # Find observed points it dominates
    dominated = [o for o in pareto_obs if dominates(p, o)]
    if dominated:
        new_pareto.append((p, dominated))

new_pareto.sort(key=lambda x: x[0]["E"])
print(f"\nPredicted OOS configs that Pareto-dominate ≥1 observed point ({len(new_pareto)} candidates):")
print(f"  {'config':<35} {'pred acc':>9} {'pred E':>8}  dominates:")
for p, dom in new_pareto[:15]:
    label = f"{p['topo']:<14} k={p['k']} R={p['R']} M={p['M']}"
    dom_str = ", ".join(f"{o['acc']:.0f}%@{o['energy']:.0f}kJ" for o in dom[:3])
    print(f"  {label:<35} {p['acc']:>7.1f}% {p['E']:>7.0f}kJ  {dom_str}")

# Pick top 3 most diverse candidates for a validation sbatch
top_candidates = []
seen_budget_buckets = set()
for p, dom in new_pareto:
    bucket = int(p["E"] / 50) * 50  # 50kJ buckets
    if bucket in seen_budget_buckets: continue
    seen_budget_buckets.add(bucket)
    top_candidates.append(p)
    if len(top_candidates) >= 4: break

print(f"\nTop {len(top_candidates)} diverse candidates for validation sbatch:")
for p in top_candidates:
    print(f"  {p['topo']:<14} k={p['k']} R={p['R']} M={p['M']} → pred {p['acc']:.0f}% @ {p['E']:.0f}kJ")

# Unified C-saturation LOO
errs_acc_csat = []
for i, c in enumerate(swe):
    sub = df.drop(i).reset_index(drop=True)
    sub = sub[sub["topo"] == c["topo"]]
    if len(sub) < 5: continue
    try:
        pc_pred, _, _ = fit_pc_predictor(sub)
        csat_pred, _, _ = fit_c_saturation(sub)
        if csat_pred is None: continue
        _, C_pred, _ = pc_pred(c)
        acc_pred = csat_pred(C_pred)
        errs_acc_csat.append(abs(acc_pred - c["acc"]))
    except Exception:
        continue

print(f"  UNIFIED C-saturation:      n={len(errs_acc_csat):>3}   "
      f"median |acc err|={np.median(errs_acc_csat):.1f}pp   "
      f"median %E err={np.median(errs_e_2stage)*100:.0f}%")
