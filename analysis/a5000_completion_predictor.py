"""Train a completion-token (C) predictor on A5000 per-call data using LightGBM,
then validate under three CV regimes:
  (A) random per-call holdout    (within distribution, optimistic ceiling)
  (B) leave-task-out per benchmark (realistic generalization)
  (C) leave-topology-out          (the killer test for the paper)

Targets: log(C+1). Loss: standard squared error on log scale.

Outputs:
    analysis/a5000_figs/completion_predictor_scatter.png
    analysis/a5000_figs/completion_predictor_metrics.csv
    analysis/a5000_figs/completion_predictor_feature_importance.png
"""
from __future__ import annotations
import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"


def load_call_level() -> pd.DataFrame:
    """Per-call dataframe with engineered features for the C predictor."""
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            calls = rec.get("call_records") or []
            # Walk calls in order, accumulating per-trajectory state
            per_agent = defaultdict(int)  # turn index per agent
            obs_running_total = 0
            tool_running_total = 0
            for c in calls:
                ct = c.get("call_type", "?")
                if ct == "tool_execution":
                    obs_running_total += int(c.get("total_tokens", 0) or 0)
                    tool_running_total += 1
                    continue
                if "prompt_tokens" not in c or "completion_tokens" not in c:
                    continue
                agent = c.get("agent_id") or "default"
                turn = per_agent[agent]
                per_agent[agent] += 1

                P = int(c["prompt_tokens"])
                C = int(c["completion_tokens"])
                e = float(c.get("gpu_dynamic_energy_joules") or 0)
                if e <= 0 and (P + C) > 0:
                    continue

                # Coarse role labels: collapse worker_2_r1 → worker, debater_0_r0 → debater_r0, etc.
                if agent.startswith("worker"):
                    role = "worker_r1" if "_r1" in agent else "worker_r0"
                elif agent.startswith("debater"):
                    role = "debater_r1" if "_r1" in agent else "debater_r0"
                elif agent.startswith("orchestrator"):
                    role = "orchestrator"
                elif agent == "synthesizer" or "synth" in agent:
                    role = "synthesizer"
                elif agent.startswith("solver"):
                    role = "solver"
                else:
                    role = agent

                # Has-tools heuristic from call_type
                has_tools = ct.startswith("react_step")

                rows.append(dict(
                    benchmark=rec.get("benchmark", "?"),
                    topology=rec.get("topology", "?"),
                    n_rounds=int(rec.get("n_rounds_override", 2) or 2),
                    max_react_steps=int(rec.get("max_react_steps", 0) or 0),
                    task_id=str(rec.get("task_id", "?")),
                    rep=int(rec.get("rep", 0) or 0),
                    role=role,
                    call_type=ct,
                    turn_idx=turn,
                    has_tools=int(has_tools),
                    P=P, C=C, E=e,
                    obs_so_far=obs_running_total,
                    tools_so_far=tool_running_total,
                ))
    df = pd.DataFrame(rows)
    return df


CAT_FEATURES = ["benchmark", "topology", "role", "call_type"]
NUM_FEATURES = ["P", "turn_idx", "max_react_steps", "n_rounds",
                "has_tools", "obs_so_far", "tools_so_far"]


def make_xy(df: pd.DataFrame):
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    y = np.log1p(df["C"].to_numpy(dtype=float))
    return X, y


def train_lgb(X_tr, y_tr, X_va=None, y_va=None):
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=30,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        verbose=-1,
    )
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES)
    valid_sets = [train_set]
    if X_va is not None:
        valid_sets.append(lgb.Dataset(X_va, label=y_va, categorical_feature=CAT_FEATURES, reference=train_set))
    return lgb.train(params, train_set, num_boost_round=400, valid_sets=valid_sets,
                     callbacks=[lgb.early_stopping(30, verbose=False)] if X_va is not None else [])


def metric(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.abs(y_pred - y_true) / np.maximum(y_true, 1)).mean()
    return r2, mae, mape


def cv_random(df, n_folds=5):
    X, y = make_xy(df)
    kf = KFold(n_folds, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(X):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va])
        pred = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        scores.append(metric(y[va], pred))
    return np.array(scores).mean(axis=0)


def cv_leave_task_out(df, n_folds=5):
    X, y = make_xy(df)
    groups = df["task_id"].astype(str) + "::" + df["benchmark"].astype(str)
    gkf = GroupKFold(n_folds)
    scores = []
    for tr, va in gkf.split(X, y, groups):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va])
        pred = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        scores.append(metric(y[va], pred))
    return np.array(scores).mean(axis=0)


def cv_leave_topology_out(df):
    X, y = make_xy(df)
    topos = df["topology"].unique()
    rows = []
    for held_out in topos:
        tr = df["topology"] != held_out
        va = ~tr
        if va.sum() < 50 or tr.sum() < 100:
            continue
        m = train_lgb(X[tr], y[tr], X[va], y[va])
        pred = m.predict(X[va], num_iteration=m.best_iteration)
        r2, mae, mape = metric(y[va], pred)
        rows.append((held_out, va.sum(), r2, mae, mape))
    return pd.DataFrame(rows, columns=["held_out_topology", "n_test", "r2", "mae", "mape"])


def main():
    print("Loading per-call records...")
    df = load_call_level()
    print(f"  {len(df)} calls, {df['task_id'].nunique()} task IDs, "
          f"{df['benchmark'].nunique()} benchmarks, {df['topology'].nunique()} topologies")

    print("\n=== (A) Random 5-fold per-call CV ===")
    r2_a, mae_a, mape_a = cv_random(df)
    print(f"  R² = {r2_a:.3f}   MAE = {mae_a:.1f} tokens   MAPE = {mape_a:.2%}")

    print("\n=== (B) GroupKFold leave-task-out (5 folds) ===")
    r2_b, mae_b, mape_b = cv_leave_task_out(df)
    print(f"  R² = {r2_b:.3f}   MAE = {mae_b:.1f} tokens   MAPE = {mape_b:.2%}")

    print("\n=== (C) Leave-one-topology-out ===")
    lc = cv_leave_topology_out(df)
    print(lc.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print(f"  mean R² across held-out topologies: {lc['r2'].mean():.3f}")

    # Final training on all data, then plot
    print("\nFinal model on full data + scatter...")
    X, y = make_xy(df)
    final = train_lgb(X, y)
    pred = final.predict(X, num_iteration=final.best_iteration)

    # Scatter (sample to keep it readable)
    samp = np.random.RandomState(0).choice(len(df), size=min(8000, len(df)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(np.expm1(y[samp]), np.expm1(pred[samp]), s=3, alpha=0.18, color="#2ca02c")
    lo, hi = 1, max(np.expm1(y[samp]).max(), np.expm1(pred[samp]).max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured completion tokens C")
    ax.set_ylabel("Predicted completion tokens Ĉ")
    ax.set_title(f"LightGBM completion-length predictor\nrandom CV R² = {r2_a:.3f}")
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[1]
    fnames = list(X.columns)
    importances = final.feature_importance(importance_type="gain")
    order = np.argsort(importances)[::-1]
    ax.barh(np.array(fnames)[order][::-1], importances[order][::-1],
            color="#9467bd", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Gain")
    ax.set_title("Feature importance (gain)")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "completion_predictor_scatter.png"), dpi=140, bbox_inches="tight")
    print(f"  saved completion_predictor_scatter.png")

    # Metrics CSV
    metrics = pd.DataFrame([
        dict(split="random",         r2=r2_a, mae=mae_a, mape=mape_a),
        dict(split="leave_task_out", r2=r2_b, mae=mae_b, mape=mape_b),
        *[dict(split=f"leave_topology_out_{r['held_out_topology']}", r2=r['r2'], mae=r['mae'], mape=r['mape']) for _, r in lc.iterrows()],
    ])
    metrics.to_csv(os.path.join(OUT_DIR, "completion_predictor_metrics.csv"), index=False)
    print(f"  saved completion_predictor_metrics.csv")


if __name__ == "__main__":
    main()
