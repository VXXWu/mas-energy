"""SAS-rollout-conditioned energy prediction pipeline.

For each task, use the (cheap) SAS rollout as a per-task fingerprint, then
predict the (expensive) MAS variants. Two predictors are built and compared:

  Direct:   (topology, k, R, sas_C, sas_P, sas_E, sas_n_react) → log(E_total)
  Composed: same features → log(C_total) and log(P_total) separately,
            then combine via E = N·a + b·P̂ + c·Ĉ

This script focuses on STRUCTURAL diagnostics rather than headline accuracy:
  1. Per-topology learned amplification: Ĉ_MAS / Ĉ_SAS
  2. Feature importance: which SAS features carry the signal?
  3. Residual analysis: where does the model still fail (and what does that say
     about the compound cost effect)?
  4. Composed vs direct: is the regression-aware composition better, equal, or
     worse than direct E prediction?

Outputs:
    analysis/a5000_figs/sas_rollout_predictor.png
    analysis/a5000_figs/sas_rollout_amplification.csv
    analysis/a5000_figs/sas_rollout_metrics.csv
"""
from __future__ import annotations
import os
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

A_INT, B_PR, C_DE = -84.0, 0.018, 5.54


# ----------------------------------------------------------------------
# Load + build SAS-anchor table
# ----------------------------------------------------------------------

def load_task_records() -> pd.DataFrame:
    rows = []
    cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        m = cre.match(os.path.basename(path))
        if not m:
            continue
        topo, k, r = m.group(1), int(m.group(2)), m.group(3)
        rounds = int(r) if r else 2
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            e = rec.get("gpu_dynamic_energy_joules", 0) or 0
            P = rec.get("total_prompt_tokens", 0) or 0
            C = rec.get("total_completion_tokens", 0) or 0
            pred = -84 + 0.018 * P + 5.54 * C
            if pred > 0 and e / pred < 0.1:
                continue
            rows.append(dict(
                benchmark=rec.get("benchmark", "?"),
                topology=topo,
                k=k, rounds=rounds,
                task_id=str(rec.get("task_id", "?")),
                rep=int(rec.get("rep", 0) or 0),
                P=int(P), C=int(C), E=float(e),
                n_react=int(rec.get("n_react_steps", 0) or 0),
                n_calls=int(rec.get("n_llm_calls", 0) or 0),
            ))
    return pd.DataFrame(rows)


def build_sas_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (benchmark, task_id), averaging over k and rep for SAS runs."""
    sas = df[df["topology"] == "sas"].copy()
    return (
        sas.groupby(["benchmark", "task_id"], as_index=False)
           .agg(
               sas_C=("C", "mean"),
               sas_P=("P", "mean"),
               sas_E=("E", "mean"),
               sas_n_react=("n_react", "mean"),
               sas_n_calls=("n_calls", "mean"),
           )
    )


def n_calls_estimate(topology: str, k: int, rounds: int, M: int = 3) -> int:
    if topology == "independent":
        return M * k + 1
    if topology == "centralized":
        return 1 + rounds * (M * k + 1) + 1
    if topology == "decentralized":
        return M * k * (rounds + 1)
    if topology == "hybrid":
        return 1 + rounds * (M * k + 1) + 1
    return k


# ----------------------------------------------------------------------
# Train + eval helpers
# ----------------------------------------------------------------------

def train_lgb(X_tr, y_tr, X_va, y_va, cat):
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat)
    valid_set = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=train_set)
    return lgb.train(
        dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=20, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbose=-1),
        train_set, num_boost_round=600, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )


def metric(y_true_log, y_pred_log):
    yt = np.expm1(y_true_log); yp = np.clip(np.expm1(y_pred_log), 0, None)
    return dict(
        r2=r2_score(yt, yp),
        mape_mean=float(np.mean(np.abs(yp - yt) / np.maximum(yt, 1))),
        mape_med=float(np.median(np.abs(yp - yt) / np.maximum(yt, 1))),
    )


def cv_loto(df: pd.DataFrame, target_col: str, cat: list, num: list, label: str):
    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(df[target_col].to_numpy(dtype=float))
    groups = (df["benchmark"].astype(str) + "::" + df["task_id"].astype(str)).to_numpy()
    gkf = GroupKFold(5)
    preds = np.zeros_like(y)
    importances = []
    for tr, va in gkf.split(X, y, groups):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat)
        preds[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        importances.append(m.feature_importance(importance_type="gain"))
    avg = metric(y, preds)
    print(f"  {label:<32} R²={avg['r2']:.3f}  MAPE_med={avg['mape_med']:.2%}")
    avg["preds_log"] = preds
    avg["importance"] = np.array(importances).mean(axis=0)
    avg["feature_names"] = list(X.columns)
    return avg


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    print("Loading task records...")
    df = load_task_records()
    print(f"  {len(df)} task rows, {df['task_id'].nunique()} unique tasks")

    print("Building SAS anchors...")
    sas_anchors = build_sas_anchors(df)
    print(f"  {len(sas_anchors)} (benchmark, task_id) anchors")

    # Restrict to MAS rows + join SAS anchors
    mas = df[df["topology"] != "sas"].copy()
    merged = mas.merge(sas_anchors, on=["benchmark", "task_id"], how="left")
    merged = merged.dropna(subset=["sas_C", "sas_P", "sas_E"])
    print(f"  MAS rows with SAS anchor: {len(merged)} / {len(mas)}")

    cat = ["benchmark", "topology"]
    num = ["k", "rounds", "sas_C", "sas_P", "sas_E", "sas_n_react", "sas_n_calls"]

    print("\n=== Predicting C_total (with SAS rollout features) ===")
    res_C = cv_loto(merged, "C", cat, num, "Ĉ | SAS")

    print("\n=== Predicting P_total (with SAS rollout features) ===")
    res_P = cv_loto(merged, "P", cat, num, "P̂ | SAS")

    print("\n=== Predicting E_total directly (with SAS rollout features) ===")
    res_E = cv_loto(merged, "E", cat, num, "Ê direct | SAS")

    # Compose energy from Ĉ and P̂
    print("\n=== Composing Ê = N·a + b·P̂ + c·Ĉ ===")
    merged["N_calls_est"] = [n_calls_estimate(r.topology, r.k, r.rounds) for r in merged.itertuples()]
    merged["C_pred"] = np.expm1(res_C["preds_log"])
    merged["P_pred"] = np.expm1(res_P["preds_log"])
    merged["E_pred_composed"] = (
        merged["N_calls_est"] * A_INT
        + B_PR * merged["P_pred"]
        + C_DE * merged["C_pred"]
    ).clip(lower=1)
    merged["E_pred_direct"] = np.expm1(res_E["preds_log"])

    e_true = merged["E"].to_numpy()
    e_comp = merged["E_pred_composed"].to_numpy()
    e_dir = merged["E_pred_direct"].to_numpy()
    r2_comp = r2_score(e_true, e_comp)
    r2_dir = r2_score(e_true, e_dir)
    mape_comp = float(np.median(np.abs(e_comp - e_true) / np.maximum(e_true, 1)))
    mape_dir = float(np.median(np.abs(e_dir - e_true) / np.maximum(e_true, 1)))
    print(f"  Composed:  R²={r2_comp:.3f}  MAPE_med={mape_comp:.2%}")
    print(f"  Direct:    R²={r2_dir:.3f}  MAPE_med={mape_dir:.2%}")

    # ------------------------------------------------------------------
    # STRUCTURAL DIAGNOSTICS
    # ------------------------------------------------------------------

    print("\n=== Diagnostic 1: Learned amplification per topology ===")
    print("Mean (Ĉ_MAS / sas_C) for each topology configuration:")
    merged["amp_C_pred"] = merged["C_pred"] / merged["sas_C"].clip(lower=1)
    merged["amp_C_true"] = merged["C"] / merged["sas_C"].clip(lower=1)
    amp = (
        merged.groupby(["benchmark", "topology", "k", "rounds"])
        .agg(amp_C_pred=("amp_C_pred", "median"),
             amp_C_true=("amp_C_true", "median"),
             n=("task_id", "count"))
        .reset_index()
    )
    print(amp.sort_values(["benchmark", "topology", "k", "rounds"])
            .to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    amp.to_csv(os.path.join(OUT_DIR, "sas_rollout_amplification.csv"), index=False)

    print("\n=== Diagnostic 2: Feature importance for Ĉ predictor ===")
    imp = pd.DataFrame({
        "feature": res_C["feature_names"],
        "gain_C": res_C["importance"],
        "gain_P": res_P["importance"],
        "gain_E": res_E["importance"],
    }).sort_values("gain_C", ascending=False)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    print("\n=== Diagnostic 3: Residual structure ===")
    merged["log_resid_C"] = np.log1p(merged["C"]) - np.log1p(merged["C_pred"])
    print("Median log-residual (Ĉ vs C) per topology — sign tells us systematic bias:")
    res_by_topo = merged.groupby("topology")["log_resid_C"].agg(["median", "std", "count"])
    print(res_by_topo.to_string(float_format=lambda x: f"{x:7.3f}"))

    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # Panel 1: Composed vs measured E (scatter)
    ax = fig.add_subplot(gs[0, 0])
    samp = np.random.RandomState(0).choice(len(merged), size=min(2000, len(merged)), replace=False)
    ax.scatter(e_true[samp], e_comp[samp], s=10, alpha=0.4, color="#1f77b4", label="Composed")
    ax.scatter(e_true[samp], e_dir[samp], s=10, alpha=0.3, color="#d62728", label="Direct")
    lo, hi = e_true.min(), e_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured E (J)")
    ax.set_ylabel("Predicted E (J)")
    ax.set_title(f"Energy prediction\ncomposed R²={r2_comp:.2f}  direct R²={r2_dir:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: feature importance bar
    ax = fig.add_subplot(gs[0, 1])
    imp_C = imp.set_index("feature")["gain_C"]
    imp_C = imp_C / imp_C.max()
    imp_C.sort_values().plot.barh(ax=ax, color="#2ca02c")
    ax.set_xlabel("Normalized gain (Ĉ predictor)")
    ax.set_title("Feature importance — Ĉ predictor")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 3: per-topology amplification (predicted vs true median)
    ax = fig.add_subplot(gs[0, 2])
    topo_order = sorted(amp["topology"].unique())
    by_topo_pred = amp.groupby("topology")["amp_C_pred"].median()
    by_topo_true = amp.groupby("topology")["amp_C_true"].median()
    x = np.arange(len(topo_order))
    w = 0.35
    ax.bar(x - w/2, [by_topo_true[t] for t in topo_order], w,
           label="True median amp", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, [by_topo_pred[t] for t in topo_order], w,
           label="Predicted median amp", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(topo_order, rotation=15, fontsize=9)
    ax.set_ylabel("Median Ĉ_MAS / sas_C")
    ax.set_title("Per-topology completion amplification\nlearned vs measured")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: residual structure by topology
    ax = fig.add_subplot(gs[1, 0])
    merged.boxplot(column="log_resid_C", by="topology", ax=ax, grid=True)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Topology"); ax.set_ylabel("log(C) − log(Ĉ)")
    ax.set_title("Residual distribution by topology")
    plt.suptitle("")  # remove default suptitle from boxplot

    # Panel 5: residual vs sas_C (looking for compound cost signal)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(merged["sas_C"], merged["log_resid_C"], s=4, alpha=0.2, color="#9467bd")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("sas_C (task verbosity baseline)")
    ax.set_ylabel("log(C) − log(Ĉ) residual")
    ax.set_title("Residual vs SAS verbosity\n(non-flat → compound cost not fully captured)")
    ax.grid(True, alpha=0.3)

    # Panel 6: amplification vs k by topology (true)
    ax = fig.add_subplot(gs[1, 2])
    for topo in topo_order:
        sub = amp[amp["topology"] == topo].groupby("k")["amp_C_true"].median()
        ax.plot(sub.index, sub.values, marker="o", label=topo, lw=1.5)
    ax.set_xlabel("k (max_react_steps)")
    ax.set_ylabel("Median true Ĉ_MAS / sas_C")
    ax.set_title("Empirical amplification vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SAS-Rollout Conditioned Energy Prediction — Structural Diagnostics",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sas_rollout_predictor.png"), dpi=140, bbox_inches="tight")
    print(f"\n  saved analysis/a5000_figs/sas_rollout_predictor.png")

    # Save metrics summary
    metrics_summary = pd.DataFrame([
        dict(model="C_predictor", r2=res_C["r2"], mape_med=res_C["mape_med"]),
        dict(model="P_predictor", r2=res_P["r2"], mape_med=res_P["mape_med"]),
        dict(model="E_direct", r2=res_E["r2"], mape_med=res_E["mape_med"]),
        dict(model="E_composed", r2=r2_comp, mape_med=mape_comp),
    ])
    metrics_summary.to_csv(os.path.join(OUT_DIR, "sas_rollout_metrics.csv"), index=False)
    print("\n=== Final summary ===")
    print(metrics_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
