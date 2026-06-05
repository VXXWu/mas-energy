"""Comparative completion-token predictors on A5000 data.

Three variants, in order of increasing capability:
  V0  Per-call regression       (baseline, ~current script)
  V1  Per-task aggregate        (predict total task C from task features)
  V2  Per-task + SAS rollout    (predict MAS task C using SAS features as anchor)

For each variant: random 5-fold CV + leave-task-out CV. Report R² and MAPE.

Outputs:
    analysis/a5000_figs/predictor_comparison.csv
    analysis/a5000_figs/predictor_comparison.png
"""
from __future__ import annotations
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


def load_task_level() -> pd.DataFrame:
    """One row per (benchmark, topology, k, R, task_id, rep)."""
    rows = []
    import re
    cre = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        m = cre.match(os.path.basename(path))
        if not m:
            continue
        topo, k, r = m.group(1), int(m.group(2)), m.group(3)
        rounds_override = int(r) if r else 2  # default R=2 unless explicitly _R1
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
                rounds=rounds_override,
                k=k,
                task_id=str(rec.get("task_id", "?")),
                rep=int(rec.get("rep", 0) or 0),
                P=int(P),
                C=int(C),
                E=float(e),
                n_calls=int(rec.get("n_llm_calls", 0) or 0),
                n_tools=int(rec.get("n_tool_calls", 0) or 0),
                n_react=int(rec.get("n_react_steps", 0) or 0),
            ))
    return pd.DataFrame(rows)


def load_call_level() -> pd.DataFrame:
    """Used by the V0 baseline only."""
    from collections import defaultdict
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
            per_agent = defaultdict(int)
            obs_total = 0
            for c in rec.get("call_records") or []:
                ct = c.get("call_type", "?")
                if ct == "tool_execution":
                    obs_total += int(c.get("total_tokens", 0) or 0)
                    continue
                if "prompt_tokens" not in c or "completion_tokens" not in c:
                    continue
                agent = c.get("agent_id") or "default"
                turn = per_agent[agent]
                per_agent[agent] += 1
                P = int(c["prompt_tokens"])
                C = int(c["completion_tokens"])
                e = float(c.get("gpu_dynamic_energy_joules") or 0)
                if e <= 0:
                    continue
                role = (
                    "worker_r1" if "worker" in agent and "_r1" in agent
                    else "worker_r0" if "worker" in agent
                    else "debater_r1" if "debater" in agent and "_r1" in agent
                    else "debater_r0" if "debater" in agent
                    else "orchestrator" if agent.startswith("orchestrator")
                    else "synthesizer" if "synth" in agent
                    else "solver" if agent.startswith("solver")
                    else agent
                )
                rows.append(dict(
                    benchmark=rec.get("benchmark", "?"),
                    topology=rec.get("topology", "?"),
                    task_id=str(rec.get("task_id", "?")),
                    rep=int(rec.get("rep", 0) or 0),
                    role=role,
                    call_type=ct,
                    turn_idx=turn,
                    has_tools=int(ct.startswith("react_step")),
                    max_react_steps=int(rec.get("max_react_steps", 0) or 0),
                    P=P, C=C,
                    obs_so_far=obs_total,
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Common training/eval helpers
# ----------------------------------------------------------------------

def train_lgb(X_tr, y_tr, X_va=None, y_va=None, cat_features=None):
    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        verbose=-1,
    )
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features or "auto")
    valid_sets = [train_set]
    if X_va is not None:
        valid_sets.append(lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features or "auto", reference=train_set))
    return lgb.train(params, train_set, num_boost_round=400, valid_sets=valid_sets,
                     callbacks=[lgb.early_stopping(30, verbose=False)] if X_va is not None else [])


def metric(y_true_log, y_pred_log):
    yt = np.expm1(y_true_log)
    yp = np.clip(np.expm1(y_pred_log), 0, None)
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    mape = (np.abs(yp - yt) / np.maximum(yt, 1)).mean()
    return r2, mae, mape


def cv_random(X, y, cat_features=None, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(X):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_features)
        pred = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        scores.append(metric(y[va], pred))
    return np.array(scores).mean(axis=0)


def cv_group(X, y, groups, cat_features=None, n_folds=5):
    gkf = GroupKFold(n_folds)
    scores = []
    for tr, va in gkf.split(X, y, groups):
        m = train_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_features)
        pred = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        scores.append(metric(y[va], pred))
    return np.array(scores).mean(axis=0)


# ----------------------------------------------------------------------
# Variant V0: per-call baseline
# ----------------------------------------------------------------------

def variant_v0_per_call(call_df) -> dict:
    print("\n=== V0 — per-call (baseline) ===")
    cat = ["benchmark", "topology", "role", "call_type"]
    num = ["P", "turn_idx", "max_react_steps", "has_tools", "obs_so_far"]
    X = call_df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(call_df["C"].to_numpy(dtype=float))
    r_rand = cv_random(X, y, cat)
    r_grp = cv_group(X, y, call_df["task_id"].astype(str), cat)
    print(f"  random      R²={r_rand[0]:.3f}  MAE={r_rand[1]:.1f}  MAPE={r_rand[2]:.2%}")
    print(f"  leave-task  R²={r_grp[0]:.3f}  MAE={r_grp[1]:.1f}  MAPE={r_grp[2]:.2%}")
    return dict(
        variant="V0_per_call",
        rand_r2=r_rand[0], rand_mae=r_rand[1], rand_mape=r_rand[2],
        grp_r2=r_grp[0], grp_mae=r_grp[1], grp_mape=r_grp[2],
    )


# ----------------------------------------------------------------------
# Variant V1: per-task aggregate
# ----------------------------------------------------------------------

def variant_v1_per_task(task_df) -> dict:
    print("\n=== V1 — per-task aggregate (CLEAN — no run-derived leakage) ===")
    cat = ["benchmark", "topology"]
    # Only features knowable BEFORE running: benchmark, topology, k, rounds.
    # P/n_calls/n_react are leakage — they're measurements of the run we're trying to predict.
    num = ["k", "rounds"]
    X = task_df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(task_df["C"].to_numpy(dtype=float))
    r_rand = cv_random(X, y, cat)
    r_grp = cv_group(X, y, task_df["task_id"].astype(str), cat)
    print(f"  random      R²={r_rand[0]:.3f}  MAE={r_rand[1]:.1f}  MAPE={r_rand[2]:.2%}")
    print(f"  leave-task  R²={r_grp[0]:.3f}  MAE={r_grp[1]:.1f}  MAPE={r_grp[2]:.2%}")
    return dict(
        variant="V1_per_task",
        rand_r2=r_rand[0], rand_mae=r_rand[1], rand_mape=r_rand[2],
        grp_r2=r_grp[0], grp_mae=r_grp[1], grp_mape=r_grp[2],
    )


# ----------------------------------------------------------------------
# Variant V2: per-task + SAS rollout features
# ----------------------------------------------------------------------

def variant_v2_with_rollout(task_df) -> dict:
    print("\n=== V2 — per-task + SAS rollout features ===")
    # Build SAS feature table per (benchmark, task_id) — average over k and rep
    sas = task_df[task_df["topology"] == "sas"].copy()
    sas_feat = (
        sas.groupby(["benchmark", "task_id"], as_index=False)
           .agg(sas_C=("C", "mean"), sas_P=("P", "mean"), sas_E=("E", "mean"),
                sas_n_calls=("n_calls", "mean"), sas_n_react=("n_react", "mean"))
    )
    print(f"  SAS feature table: {len(sas_feat)} (benchmark × task_id) anchors")

    # Predict for non-SAS tasks (the interesting case: extrapolate from SAS to MAS)
    mas = task_df[task_df["topology"] != "sas"].copy()
    merged = mas.merge(sas_feat, on=["benchmark", "task_id"], how="left")
    # Drop tasks where SAS isn't available (shouldn't happen in practice)
    merged = merged.dropna(subset=["sas_C", "sas_E"])
    print(f"  MAS rows with SAS anchor: {len(merged)} / {len(mas)}")

    cat = ["benchmark", "topology"]
    # CLEAN: only the topology hyperparameters + SAS rollout anchor features.
    # No leakage from the MAS run we are trying to predict.
    num = ["k", "rounds",
           "sas_C", "sas_P", "sas_E", "sas_n_calls", "sas_n_react"]
    X = merged[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(merged["C"].to_numpy(dtype=float))
    r_rand = cv_random(X, y, cat)
    r_grp = cv_group(X, y, merged["task_id"].astype(str), cat)
    print(f"  random      R²={r_rand[0]:.3f}  MAE={r_rand[1]:.1f}  MAPE={r_rand[2]:.2%}")
    print(f"  leave-task  R²={r_grp[0]:.3f}  MAE={r_grp[1]:.1f}  MAPE={r_grp[2]:.2%}")
    return dict(
        variant="V2_per_task_+SAS",
        rand_r2=r_rand[0], rand_mae=r_rand[1], rand_mape=r_rand[2],
        grp_r2=r_grp[0], grp_mae=r_grp[1], grp_mape=r_grp[2],
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    print("Loading task-level records...")
    task_df = load_task_level()
    print(f"  {len(task_df)} task rows, {task_df['task_id'].nunique()} unique task IDs, {task_df['topology'].nunique()} topologies")

    print("Loading per-call records...")
    call_df = load_call_level()
    print(f"  {len(call_df)} call rows")

    results = []
    results.append(variant_v0_per_call(call_df))
    results.append(variant_v1_per_task(task_df))
    results.append(variant_v2_with_rollout(task_df))

    df = pd.DataFrame(results)
    print()
    print("=== Comparison ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(os.path.join(OUT_DIR, "predictor_comparison.csv"), index=False)

    # Bar chart of R² across variants × CV regimes
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(df))
    ax.bar(x - width/2, df["rand_r2"], width, label="Random CV", color="#1f77b4")
    ax.bar(x + width/2, df["grp_r2"], width, label="Leave-task-out", color="#ff7f0e")
    for xi, (r1, r2) in enumerate(zip(df["rand_r2"], df["grp_r2"])):
        ax.text(xi - width/2, r1 + 0.01, f"{r1:.2f}", ha="center", fontsize=9)
        ax.text(xi + width/2, r2 + 0.01, f"{r2:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=10, fontsize=9)
    ax.set_ylabel("R² (predicting log(C+1))")
    ax.set_title("Completion-length predictor variants — A5000")
    ax.set_ylim(0, max(1.0, df[["rand_r2", "grp_r2"]].max().max() * 1.15))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "predictor_comparison.png"), dpi=140, bbox_inches="tight")
    print(f"\n  saved analysis/a5000_figs/predictor_comparison.png")


if __name__ == "__main__":
    main()
