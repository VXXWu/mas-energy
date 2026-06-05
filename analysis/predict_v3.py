"""Predictive model v3: Modular structural predictor.

Trains ONLY on standard scaling grid cells:
  - k-sweep:  (k ∈ standard, R=2, M=3)
  - R-sweep:  (k=10, R ∈ {1,3,4,5}, M=3)        [centralized/decentralized only]
  - M-sweep:  (k=10, R=2, M ∈ {2,4,5,7,10,15,20,30})  [centralized/decentralized only]

Evaluates on all OTHER cells as held-out "intermediate" tests
(cells like k=20 R=2 M=4, k=3 R=4 M=2, etc that aren't on any sweep).

Model structure (per benchmark, per topology):
  Stage 1: GP_P(log k, log R, log M) → P_tokens
           GP_C(log k, log R, log M) → C_tokens
  Stage 2: E_kJ = α + β·P + γ·C        [linear, per-task aggregate]
  Stage 3: acc = empirical_floor(topology)              if k < k_min
                = 101 - exp( GP_acc(log k, log R, log M) )  otherwise

k_min determination (per benchmark, per topology):
  threshold = 0.10 × max_observed_acc on the topology's training cells
  k_min = smallest k value where acc ≥ threshold
  floor_acc = mean acc of cells where acc < threshold
  (e.g., on SWE-bench Decent: max=92%, threshold=9.2%; k_min=5 because
   k=1,2,3 all fall below 9.2%, and k=5 first crosses it)
"""
import json, glob, os, re
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

BENCH = {
    "FanOutQA":   "a5000_fanoutqa_v4",
    "WorkBench":  "a5000_workbench_v2",
    "BrowseComp+":"a5000_browsecomp_pilot",
    "SWE-bench":  "a5000_swebench",
}
DEFAULT_R = 2; DEFAULT_M = 3
TOPOS = ["sas", "independent", "centralized", "decentralized"]
cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")


def load_cell(f):
    acc, e, p, c = [], [], [], []
    for line in open(f):
        try: dd = json.loads(line)
        except: continue
        if dd.get("error"): continue
        a = (float(dd["loose_accuracy"]) if dd.get("loose_accuracy") is not None
             else (1.0 if dd.get("correct") else 0.0))
        acc.append(a); e.append(dd.get("gpu_dynamic_energy_joules",0)/1000)
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


def is_standard_grid(c):
    """True iff cell is on the canonical k/R/M sweep grid."""
    topo, k, R, M = c["topo"], c["k"], c["R"], c["M"]
    if topo in ("sas", "independent"):
        # Only k-sweep at R=2, M=3 makes sense for these
        return R == DEFAULT_R and M == DEFAULT_M
    # k-sweep: (k, R=2, M=3)
    if R == 2 and M == 3: return True
    # R-sweep: (k=10, R, M=3)
    if k == 10 and M == 3: return True
    # M-sweep: (k=10, R=2, M)
    if k == 10 and R == 2: return True
    return False


# ----- Model components -----
def fit_pc_predictor(sub_df):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    X = np.column_stack([np.log(sub_df["k"]), np.log(sub_df["R"]), np.log(sub_df["M"])])
    yP = np.log(sub_df["P"].values.clip(1))
    yC = np.log(sub_df["C"].values.clip(1))
    def make_gp():
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         n_restarts_optimizer=3, random_state=0)
    gp_P = make_gp(); gp_P.fit(X, yP)
    gp_C = make_gp(); gp_C.fit(X, yC)
    # Energy linear regression on (P, C)
    XE = np.column_stack([np.ones(len(sub_df)), sub_df["P"].values, sub_df["C"].values])
    yE = sub_df["energy"].values
    coef_E, *_ = np.linalg.lstsq(XE, yE, rcond=None)
    def predict(k, R, M):
        x = np.array([[np.log(k), np.log(R), np.log(M)]])
        P = float(np.exp(gp_P.predict(x)[0]))
        Cc = float(np.exp(gp_C.predict(x)[0]))
        E = float(coef_E[0] + coef_E[1]*P + coef_E[2]*Cc)
        return P, Cc, E
    return predict


def fit_acc_predictor(sub_df, k_floor_threshold=0.10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    # Determine k_min: smallest k where acc >= 10% of max observed
    max_acc = sub_df["acc"].max()
    threshold = k_floor_threshold * max_acc
    above = sub_df[sub_df["acc"] >= threshold]
    k_min = above["k"].min() if len(above) else sub_df["k"].min()
    below = sub_df[sub_df["k"] < k_min]
    floor_acc = float(below["acc"].mean()) if len(below) else 0.0
    # GP on log(101-acc) for cells above the floor
    fit_df = sub_df[sub_df["k"] >= k_min]
    if len(fit_df) < 4:
        fit_df = sub_df  # fallback
    X = np.column_stack([np.log(fit_df["k"]), np.log(fit_df["R"]), np.log(fit_df["M"])])
    y = np.log(np.clip(101 - fit_df["acc"].values, 0.5, None))
    kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3, random_state=0)
    gp.fit(X, y)
    def predict(k, R, M):
        if k < k_min:
            return floor_acc
        x = np.array([[np.log(k), np.log(R), np.log(M)]])
        py = gp.predict(x)[0]
        return float(np.clip(101 - np.exp(py), 0, 100))
    return predict, k_min, floor_acc


# ----- Main loop -----
print("=" * 130)
print("MODULAR MODEL v3 -- training ONLY on standard grid, evaluating on intermediate cells")
print("=" * 130)

all_results = {}
for bench, bd in BENCH.items():
    cells = load_all_cells(bd)
    df_all = pd.DataFrame(cells)
    train_cells = [c for c in cells if is_standard_grid(c)]
    test_cells = [c for c in cells if not is_standard_grid(c)]
    df_train = pd.DataFrame(train_cells)

    print(f"\n{'='*130}\n{bench}: {len(train_cells)} train (standard grid) + {len(test_cells)} held-out intermediate")

    # Fit per topology on training set
    print(f"  {'Topology':<14} {'n_train':>7}  {'k_min':>5}  {'floor%':>7}  {'P R²':>6}  {'C R²':>6}  {'E_α':>7}  {'E_β':>7}  {'E_γ':>7}")
    fitted = {}
    for topo in TOPOS:
        sub = df_train[df_train["topo"] == topo]
        if len(sub) < 4: continue
        pc_pred = fit_pc_predictor(sub)
        acc_pred, k_min, floor_acc = fit_acc_predictor(sub)
        fitted[topo] = (pc_pred, acc_pred)
        # Report in-sample fit details
        XE = np.column_stack([np.ones(len(sub)), sub["P"].values, sub["C"].values])
        coef_E, *_ = np.linalg.lstsq(XE, sub["energy"].values, rcond=None)
        # P/C in-sample R²
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        X = np.column_stack([np.log(sub["k"]), np.log(sub["R"]), np.log(sub["M"])])
        # quick refit just for R²
        gp_P = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(length_scale=[1,1,1],nu=2.5)+WhiteKernel(0.1),
                                         normalize_y=True, random_state=0)
        gp_P.fit(X, np.log(sub["P"].values.clip(1)))
        gp_C = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(length_scale=[1,1,1],nu=2.5)+WhiteKernel(0.1),
                                         normalize_y=True, random_state=0)
        gp_C.fit(X, np.log(sub["C"].values.clip(1)))
        pP = np.exp(gp_P.predict(X)); pC = np.exp(gp_C.predict(X))
        r2_P = 1 - np.sum((sub["P"]-pP)**2)/np.sum((sub["P"]-sub["P"].mean())**2)
        r2_C = 1 - np.sum((sub["C"]-pC)**2)/np.sum((sub["C"]-sub["C"].mean())**2)
        print(f"  {topo:<14} {len(sub):>7}  {k_min:>5}  {floor_acc:>6.1f}%  {r2_P:>5.3f}  {r2_C:>5.3f}  "
              f"{coef_E[0]:>+7.2f}  {coef_E[1]:>+7.4f}  {coef_E[2]:>+7.4f}")

    # Evaluate on held-out intermediate cells
    if test_cells:
        print(f"\n  Held-out intermediate cell evaluation (test set):")
        print(f"  {'Cell':<40} {'Act acc':>7} {'Pred acc':>9} {'Acc miss':>9}  {'Act E':>8} {'Pred E':>8} {'E miss':>7}")
        print("  " + "-"*120)
        acc_errs, e_errs = [], []
        for c in test_cells:
            if c["topo"] not in fitted: continue
            pc_pred, acc_pred = fitted[c["topo"]]
            P_p, C_p, E_p = pc_pred(c["k"], c["R"], c["M"])
            a_p = acc_pred(c["k"], c["R"], c["M"])
            am = c["acc"] - a_p
            em = (E_p - c["energy"]) / max(c["energy"], 1) * 100
            acc_errs.append(abs(am)); e_errs.append(abs(em))
            label = f"{c['topo']} k={c['k']} R={c['R']} M={c['M']}"
            print(f"  {label:<40} {c['acc']:>6.1f}% {a_p:>8.1f}% {am:>+7.1f}pp  "
                  f"{c['energy']:>7.0f}kJ {E_p:>7.0f}kJ {em:>+6.0f}%")
        if acc_errs:
            print(f"\n  HELD-OUT MEDIAN: |acc err| = {np.median(acc_errs):.1f}pp, |%E err| = {np.median(e_errs):.0f}%")
            print(f"  HELD-OUT MAX:    |acc err| = {max(acc_errs):.1f}pp, |%E err| = {max(e_errs):.0f}%")

    all_results[bench] = {"train": train_cells, "test": test_cells,
                          "fitted": fitted, "df_all": df_all}

# Find Pareto-dominant predictions per benchmark
print("\n" + "=" * 130)
print("PARETO-DOMINANT PREDICTIONS PER BENCHMARK (configs predicted to beat observed frontier)")
print("=" * 130)
K_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 75]
R_GRID = [1, 2, 3, 4, 5]
M_GRID = [2, 3, 4, 5, 7, 10, 15, 20]

for bench, res in all_results.items():
    cells_obs = res["train"] + res["test"]
    fitted = res["fitted"]
    # Observed Pareto frontier on ALL measured cells
    sorted_c = sorted(cells_obs, key=lambda c: c["energy"])
    pareto = []; ma = -1
    for c in sorted_c:
        if c["acc"] > ma:
            pareto.append(c); ma = c["acc"]
    # Build candidate grid
    measured = {(c["topo"], c["k"], c["R"], c["M"]) for c in cells_obs}
    candidates = []
    for topo in ["centralized", "decentralized"]:
        if topo not in fitted: continue
        pc_pred, acc_pred = fitted[topo]
        for k in K_GRID:
            for R in R_GRID:
                for M in M_GRID:
                    if (topo, k, R, M) in measured: continue
                    _, _, E_p = pc_pred(k, R, M)
                    a_p = acc_pred(k, R, M)
                    candidates.append({"topo": topo, "k": k, "R": R, "M": M,
                                        "acc": a_p, "energy": E_p})
    # Find dominators
    dominators = []
    for p in candidates:
        dominated_by_obs = any(o["acc"] >= p["acc"] and o["energy"] <= p["energy"]
                                and (o["acc"] > p["acc"] or o["energy"] < p["energy"])
                                for o in pareto)
        if dominated_by_obs: continue
        dom_count = sum(1 for o in pareto
                         if p["energy"] <= o["energy"] and p["acc"] >= o["acc"]
                         and (p["energy"] < o["energy"] or p["acc"] > o["acc"]))
        if dom_count >= 1:
            dominators.append((p, dom_count))

    dominators.sort(key=lambda x: (-x[1], x[0]["energy"]))
    print(f"\n  {bench}: {len(dominators)} Pareto-dominant candidates")
    if dominators:
        print(f"  {'config':<40} {'pred acc':>9} {'pred E':>8}  obs_dominated")
        for p, n_dom in dominators[:8]:
            print(f"  {p['topo']:<14} k={p['k']:>3} R={p['R']} M={p['M']:>2}  "
                  f"{p['acc']:>7.1f}% {p['energy']:>7.0f}kJ   {n_dom}")


# ----- SWE-bench failure investigation: does adding intermediate cells help? -----
print("\n" + "=" * 130)
print("SWE-BENCH FAILURE INVESTIGATION: train on standard grid + N intermediate cells")
print("=" * 130)
print("Add intermediate cells one at a time to training set, hold out the rest, measure error\n")
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

bench, bd = "SWE-bench", "a5000_swebench"
res = all_results[bench]
train_std = res["train"]
intermediate = res["test"]

print(f"Standard grid: {len(train_std)} cells")
print(f"Intermediate cells: {len(intermediate)}")
print(f"\n  Add intermediate cells incrementally, measure held-out error on the rest:")
print(f"  {'n_added':<10} {'config added':<40} {'remaining test':>14} {'median acc err':>15} {'median %E err':>14}")

import random
random.seed(42)
# Order intermediate cells by "most informative" -- highest k_R_M product, then by topology balance
sorted_intermediate = sorted(intermediate, key=lambda c: -(c["k"] + c["R"]*10 + c["M"]*5))

added_cells = []
remaining = list(intermediate)

for n in range(0, len(intermediate) + 1):
    train_cells = train_std + added_cells
    df_train = pd.DataFrame(train_cells)
    fitted = {}
    for topo in TOPOS:
        sub = df_train[df_train["topo"] == topo]
        if len(sub) < 4: continue
        pc_pred = fit_pc_predictor(sub)
        acc_pred, _, _ = fit_acc_predictor(sub)
        fitted[topo] = (pc_pred, acc_pred)
    # Evaluate on remaining intermediate cells
    if not remaining:
        added_label = "—"
        print(f"  {n:<10} {added_label:<40} {0:>14}  all added")
        break
    acc_errs, e_errs = [], []
    for c in remaining:
        if c["topo"] not in fitted: continue
        pc_pred, acc_pred = fitted[c["topo"]]
        P_p, C_p, E_p = pc_pred(c["k"], c["R"], c["M"])
        a_p = acc_pred(c["k"], c["R"], c["M"])
        acc_errs.append(abs(a_p - c["acc"]))
        e_errs.append(abs(E_p - c["energy"]) / max(c["energy"], 1) * 100)
    if not acc_errs: break
    median_acc = np.median(acc_errs)
    median_e = np.median(e_errs)
    last_added = added_cells[-1] if added_cells else None
    added_label = (f"{last_added['topo']} k={last_added['k']} R={last_added['R']} M={last_added['M']}"
                    if last_added else "(standard grid only)")
    print(f"  {n:<10} {added_label:<40} {len(remaining):>14}  {median_acc:>14.1f}pp  {median_e:>12.0f}%")
    # Pick next cell to add (worst-predicted in current state)
    worst_idx = np.argmax(acc_errs) if acc_errs else 0
    next_cell = remaining[worst_idx]
    added_cells.append(next_cell)
    remaining.pop(worst_idx)
