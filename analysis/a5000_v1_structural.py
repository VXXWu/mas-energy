"""Structural diagnostics for V1 (no SAS rollout) energy predictor.

Features: (benchmark, topology, k, rounds) only. Targets: log(C), log(P), log(E).

Same diagnostic battery as the SAS-rollout version:
  1. Feature importance per target
  2. Per-cell learned amplification vs SAS (here: cell-mean vs sas cell-mean)
  3. Residual structure by topology
  4. Direct vs composed E prediction
  5. Per-cell prediction vs measured (since V1 outputs are constant per cell)

V1 is cleaner than the SAS-rollout version because it predicts only from the
hyperparameters that you choose when configuring an experiment — no need to
actually run SAS first.

Outputs:
    analysis/a5000_figs/v1_structural.png
    analysis/a5000_figs/v1_amplification.csv
    analysis/a5000_figs/v1_metrics.csv
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

A_INT, B_PR, C_DE = -84.0, 0.018, 5.54


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


def n_calls_estimate(topology: str, k: int, rounds: int, M: int = 3) -> int:
    if topology == "sas":
        return k
    if topology == "independent":
        return M * k + 1
    if topology in ("centralized", "hybrid"):
        return 1 + rounds * (M * k + 1) + 1
    if topology == "decentralized":
        return M * k * (rounds + 1)
    return k


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
        mape_med=float(np.median(np.abs(yp - yt) / np.maximum(yt, 1))),
        mape_mean=float(np.mean(np.abs(yp - yt) / np.maximum(yt, 1))),
    )


def cv_loto(df, target, cat, num, label):
    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(df[target].to_numpy(dtype=float))
    groups = (df["benchmark"].astype(str) + "::" + df["task_id"].astype(str)).to_numpy()
    gkf = GroupKFold(5)
    preds = np.zeros_like(y)
    importances = []
    for tr, va in gkf.split(X, y, groups):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat)
        preds[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        importances.append(m.feature_importance(importance_type="gain"))
    avg = metric(y, preds)
    print(f"  {label:<28} R²={avg['r2']:.3f}  MAPE_med={avg['mape_med']:.2%}")
    avg["preds_log"] = preds
    avg["importance"] = np.array(importances).mean(axis=0)
    avg["feature_names"] = list(X.columns)
    return avg


def main():
    print("Loading task records...")
    df = load_task_records()
    print(f"  {len(df)} task rows, {df['task_id'].nunique()} unique tasks")

    cat = ["benchmark", "topology"]
    num = ["k", "rounds"]

    print("\n=== V1 — Predicting C, P, E from (benchmark, topology, k, rounds) ===")
    res_C = cv_loto(df, "C", cat, num, "Ĉ V1")
    res_P = cv_loto(df, "P", cat, num, "P̂ V1")
    res_E = cv_loto(df, "E", cat, num, "Ê direct V1")

    # Compose
    df["N_calls_est"] = [n_calls_estimate(r.topology, r.k, r.rounds) for r in df.itertuples()]
    df["C_pred"] = np.expm1(res_C["preds_log"])
    df["P_pred"] = np.expm1(res_P["preds_log"])
    df["E_pred_composed"] = (
        df["N_calls_est"] * A_INT + B_PR * df["P_pred"] + C_DE * df["C_pred"]
    ).clip(lower=1)
    df["E_pred_direct"] = np.expm1(res_E["preds_log"])

    e_true = df["E"].to_numpy()
    r2_comp = r2_score(e_true, df["E_pred_composed"])
    r2_dir = r2_score(e_true, df["E_pred_direct"])
    mape_comp = float(np.median(np.abs(df["E_pred_composed"] - e_true) / np.maximum(e_true, 1)))
    mape_dir = float(np.median(np.abs(df["E_pred_direct"] - e_true) / np.maximum(e_true, 1)))
    print(f"\n  Composed Ê:  R²={r2_comp:.3f}  MAPE_med={mape_comp:.2%}")
    print(f"  Direct Ê:    R²={r2_dir:.3f}  MAPE_med={mape_dir:.2%}")

    # ------------------------------------------------------------------
    # Diagnostic 1: feature importance
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 1: Feature importance (V1) ===")
    imp = pd.DataFrame({
        "feature": res_C["feature_names"],
        "gain_C": res_C["importance"],
        "gain_P": res_P["importance"],
        "gain_E": res_E["importance"],
    }).sort_values("gain_C", ascending=False)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    # ------------------------------------------------------------------
    # Diagnostic 2: per-cell amplification
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 2: Per-cell amplification (V1 vs SAS cell mean) ===")
    # SAS cell means: averaged over k for fairness; collapse to one number per (benchmark, k_sas)
    sas_rows = df[df["topology"] == "sas"].copy()
    sas_cell_means = (
        sas_rows.groupby("benchmark", as_index=False)["C"].mean().rename(columns={"C": "sas_C_bench_mean"})
    )
    sas_cell_means_E = (
        sas_rows.groupby("benchmark", as_index=False)["E"].mean().rename(columns={"E": "sas_E_bench_mean"})
    )
    df = df.merge(sas_cell_means, on="benchmark", how="left")
    df = df.merge(sas_cell_means_E, on="benchmark", how="left")
    df["amp_C_pred"] = df["C_pred"] / df["sas_C_bench_mean"].clip(lower=1)
    df["amp_C_true"] = df["C"] / df["sas_C_bench_mean"].clip(lower=1)
    df["amp_E_pred_dir"] = df["E_pred_direct"] / df["sas_E_bench_mean"].clip(lower=1)
    df["amp_E_true"] = df["E"] / df["sas_E_bench_mean"].clip(lower=1)

    mas = df[df["topology"] != "sas"].copy()
    amp = (
        mas.groupby(["benchmark", "topology", "k", "rounds"])
           .agg(amp_C_pred=("amp_C_pred", "median"),
                amp_C_true=("amp_C_true", "median"),
                amp_E_pred=("amp_E_pred_dir", "median"),
                amp_E_true=("amp_E_true", "median"),
                n=("task_id", "count"))
           .reset_index()
    )
    print(amp.sort_values(["benchmark", "topology", "k", "rounds"])
            .to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    amp.to_csv(os.path.join(OUT_DIR, "v1_amplification.csv"), index=False)

    # ------------------------------------------------------------------
    # Diagnostic 3: residual structure
    # ------------------------------------------------------------------
    df["log_resid_C"] = np.log1p(df["C"]) - np.log1p(df["C_pred"])
    df["log_resid_E"] = np.log1p(df["E"]) - np.log1p(df["E_pred_direct"])
    print("\n=== Diagnostic 3: Median log-residual by topology ===")
    by_topo = df[df["topology"] != "sas"].groupby("topology")[["log_resid_C", "log_resid_E"]].agg(["median", "std", "count"])
    print(by_topo.to_string(float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------
    # Diagnostic 4: per-benchmark accuracy
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 4: Per-benchmark direct E prediction ===")
    for b in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == b]
        if len(sub) < 50: continue
        et = sub["E"].to_numpy(); ep = sub["E_pred_direct"].to_numpy()
        r2_b = r2_score(et, ep)
        mape_b = float(np.median(np.abs(ep - et) / np.maximum(et, 1)))
        print(f"  {b:<18} n={len(sub):4d}  R²={r2_b:.3f}  MAPE_med={mape_b:.2%}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    samp = np.random.RandomState(0).choice(len(df), size=min(2000, len(df)), replace=False)
    ax.scatter(e_true[samp], df["E_pred_composed"].to_numpy()[samp], s=10, alpha=0.4, color="#1f77b4", label=f"Composed R²={r2_comp:.2f}")
    ax.scatter(e_true[samp], df["E_pred_direct"].to_numpy()[samp], s=10, alpha=0.3, color="#d62728", label=f"Direct   R²={r2_dir:.2f}")
    lo, hi = e_true.min(), e_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured E (J)"); ax.set_ylabel("Predicted E (J)")
    ax.set_title("V1 energy predictions")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    imp_norm = imp.copy()
    imp_norm["gain_C"] = imp_norm["gain_C"] / imp_norm["gain_C"].max()
    imp_norm.set_index("feature")["gain_C"].sort_values().plot.barh(ax=ax, color="#2ca02c")
    ax.set_xlabel("Normalized gain (Ĉ predictor)")
    ax.set_title("V1 feature importance — Ĉ")
    ax.grid(True, axis="x", alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    topo_order = sorted(amp["topology"].unique())
    by_topo_pred = amp.groupby("topology")["amp_C_pred"].median()
    by_topo_true = amp.groupby("topology")["amp_C_true"].median()
    x = np.arange(len(topo_order)); w = 0.35
    ax.bar(x - w/2, [by_topo_true[t] for t in topo_order], w, label="True", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, [by_topo_pred[t] for t in topo_order], w, label="V1 pred", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(topo_order, rotation=15, fontsize=9)
    ax.set_ylabel("Median Ĉ_MAS / sas_C_bench_mean")
    ax.set_title("Per-topology amplification\n(V1 learns cell averages)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    df[df["topology"] != "sas"].boxplot(column="log_resid_C", by="topology", ax=ax, grid=True)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Topology"); ax.set_ylabel("log(C) − log(Ĉ)")
    ax.set_title("Residual distribution by topology")
    plt.suptitle("")

    ax = fig.add_subplot(gs[1, 1])
    # Per-cell true amplification vs k by topology, all benchmarks pooled
    for topo in topo_order:
        sub = amp[amp["topology"] == topo].groupby("k")["amp_C_true"].median()
        ax.plot(sub.index, sub.values, marker="o", label=topo, lw=1.5)
    ax.set_xlabel("k (max_react_steps)")
    ax.set_ylabel("Median true Ĉ_MAS / sas_C")
    ax.set_title("Empirical amplification vs k")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    # Per-benchmark MAPE bars
    bench_metrics = []
    for b in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == b]
        if len(sub) < 50: continue
        et = sub["E"].to_numpy(); ep = sub["E_pred_direct"].to_numpy()
        bench_metrics.append((b, r2_score(et, ep), float(np.median(np.abs(ep - et) / np.maximum(et, 1)))))
    bm_df = pd.DataFrame(bench_metrics, columns=["benchmark", "r2", "mape_med"])
    ax.bar(bm_df["benchmark"], bm_df["r2"], color="#9467bd", edgecolor="black")
    for xi, (r, m) in enumerate(zip(bm_df["r2"], bm_df["mape_med"])):
        ax.text(xi, r + 0.01, f"R²={r:.2f}\nMAPE={m:.0%}", ha="center", fontsize=8)
    ax.set_ylabel("Direct Ê R² (leave-task-out)")
    ax.set_title("Per-benchmark V1 accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=9)

    fig.suptitle("V1 (no SAS rollout) — Structural Diagnostics", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v1_structural.png"), dpi=140, bbox_inches="tight")
    print(f"\n  saved analysis/a5000_figs/v1_structural.png")

    pd.DataFrame([
        dict(model="C", r2=res_C["r2"], mape_med=res_C["mape_med"]),
        dict(model="P", r2=res_P["r2"], mape_med=res_P["mape_med"]),
        dict(model="E_direct", r2=res_E["r2"], mape_med=res_E["mape_med"]),
        dict(model="E_composed", r2=r2_comp, mape_med=mape_comp),
    ]).to_csv(os.path.join(OUT_DIR, "v1_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
