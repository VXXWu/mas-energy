"""Full ex-ante energy prediction pipeline for A5000 data.

Goal: highest-accuracy E_total prediction using only features observable BEFORE
running the model — i.e., the input prompt text plus topology hyperparameters.

Pipeline (factored):
   (benchmark, topology, k, R, question_text, gold_metadata)
                    │
                    ▼
        Question features:
          - length tokens, sentences, words
          - expected answer cardinality
          - sentence embedding (384-d MiniLM)
          - question type (regex)
                    │
                    ▼
        Topology features:
          - benchmark, topology, k, R, n_agents (M=3)
          - closed-form n_calls heuristic
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   V1_C (Ĉ_total)          V1_P (P̂_total)
        │                       │
        └───────────┬───────────┘
                    ▼
        Ê_total = N_calls·a + b·P̂ + c·Ĉ
                  (per-call regression coefficients)

Validation: leave-task-out CV per benchmark + pooled.

Outputs:
    analysis/a5000_figs/full_predictor_metrics.csv
    analysis/a5000_figs/full_predictor_scatter.png
"""
from __future__ import annotations
import os
import json
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

# Energy regression coefficients (canonical from project)
A_INT, B_PR, C_DE = -84.0, 0.018, 5.54


# ----------------------------------------------------------------------
# 1. Load task-level records
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
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. Load question text for each benchmark
# ----------------------------------------------------------------------

def load_qampari_questions() -> dict:
    """qid -> dict(question, answer_cardinality, n_entities)"""
    out = {}
    for line in open("mas-energy/data/qampari/qampari_data/dev_data.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["qid"]] = dict(
            question=r["question_text"],
            answer_cardinality=len(r.get("answer_list", [])),
            n_entities=len(r.get("entities", [])),
        )
    return out


def load_fanoutqa_questions() -> dict:
    """id -> dict(question, decomposition_size, n_evidence)"""
    try:
        import fanoutqa
        ds = fanoutqa.load_dev()
    except Exception as e:
        print(f"  fanoutqa load failed: {e}")
        return {}
    out = {}
    for q in ds:
        out[q.id] = dict(
            question=q.question,
            answer_cardinality=len(q.answer) if isinstance(q.answer, dict) else 1,
            decomposition_size=len(q.decomposition) if hasattr(q, "decomposition") else 0,
            n_evidence=len(q.necessary_evidence) if hasattr(q, "necessary_evidence") else 0,
        )
    return out


def load_swebench_questions() -> dict:
    """instance_id -> dict(question, n_patch_lines)"""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception as e:
        print(f"  swebench load failed: {e}")
        return {}
    out = {}
    for r in ds:
        out[r["instance_id"]] = dict(
            question=r["problem_statement"],
            answer_cardinality=len(str(r.get("patch", "")).split("\n")),
            n_evidence=0,
        )
    return out


# ----------------------------------------------------------------------
# 3. Engineer features per task
# ----------------------------------------------------------------------

QUESTION_TYPE_PATTERNS = [
    ("how_many", r"how many"),
    ("list", r"^(list|name|give|provide|all)"),
    ("who", r"^who"),
    ("what", r"^what"),
    ("when", r"^when"),
    ("where", r"^where"),
    ("explain", r"^(explain|why|how does|describe)"),
    ("implement", r"(implement|fix|modify|change|update|add)"),
]


def engineer_features(task_id: str, benchmark: str, q_lookup: dict) -> dict:
    qmeta = q_lookup.get(benchmark, {}).get(task_id, {})
    qtext = qmeta.get("question", "") or ""
    feats = dict(
        q_len_chars=len(qtext),
        q_len_words=len(qtext.split()),
        q_len_sentences=max(1, qtext.count(".") + qtext.count("?") + qtext.count("!")),
        q_n_named_caps=len(re.findall(r"\b[A-Z][a-z]{2,}", qtext)),
        answer_cardinality=qmeta.get("answer_cardinality", -1),
        decomposition_size=qmeta.get("decomposition_size", -1),
        n_entities=qmeta.get("n_entities", -1),
        n_evidence=qmeta.get("n_evidence", -1),
    )
    qlow = qtext.lower().strip()
    for name, pat in QUESTION_TYPE_PATTERNS:
        feats[f"qtype_{name}"] = int(bool(re.search(pat, qlow)))
    return feats


def add_embeddings(df: pd.DataFrame, q_lookup: dict, dim: int = 384) -> pd.DataFrame:
    """Add 384-d MiniLM embedding columns. Skips tasks without question text."""
    from sentence_transformers import SentenceTransformer
    print("  loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Cache embeddings per (benchmark, task_id) so we don't reembed the same question for many topologies
    unique_keys = df[["benchmark", "task_id"]].drop_duplicates()
    texts = []
    for _, r in unique_keys.iterrows():
        meta = q_lookup.get(r["benchmark"], {}).get(r["task_id"], {})
        texts.append(meta.get("question", "") or "")
    print(f"  embedding {len(texts)} unique questions ...")
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=64)
    emb_cache = {(r["benchmark"], r["task_id"]): embs[i] for i, (_, r) in enumerate(unique_keys.iterrows())}
    emb_matrix = np.array([emb_cache[(r["benchmark"], r["task_id"])] for _, r in df.iterrows()])
    emb_cols = [f"emb_{i}" for i in range(dim)]
    edf = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    return pd.concat([df, edf], axis=1)


# ----------------------------------------------------------------------
# 4. Train + evaluate
# ----------------------------------------------------------------------

def fit_lgb(X_tr, y_tr, X_va, y_va, cat):
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat)
    valid_set = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=train_set)
    return lgb.train(
        dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=63, min_data_in_leaf=20, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=5, verbose=-1),
        train_set, num_boost_round=600, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )


def metric(y_true_log, y_pred_log):
    yt = np.expm1(y_true_log)
    yp = np.clip(np.expm1(y_pred_log), 0, None)
    return dict(
        r2=r2_score(yt, yp),
        mae=mean_absolute_error(yt, yp),
        mape_mean=float((np.abs(yp - yt) / np.maximum(yt, 1)).mean()),
        mape_med=float(np.median(np.abs(yp - yt) / np.maximum(yt, 1))),
    )


def cv_loto(df: pd.DataFrame, target_col: str, cat: list, num: list, label: str) -> dict:
    """Leave-task-out CV by GroupKFold on task_id."""
    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = np.log1p(df[target_col].to_numpy(dtype=float))
    groups = (df["benchmark"].astype(str) + "::" + df["task_id"].astype(str)).to_numpy()
    gkf = GroupKFold(5)
    metrics = []
    preds_all = np.zeros_like(y)
    for tr, va in gkf.split(X, y, groups):
        m = fit_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat)
        p = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        preds_all[va] = p
        metrics.append(metric(y[va], p))
    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    print(f"  {label:<32} R²={avg['r2']:.3f}  MAE={avg['mae']:9.0f}  MAPE_med={avg['mape_med']:.2%}  MAPE_mean={avg['mape_mean']:.2%}")
    return dict(**avg, preds_log=preds_all)


# ----------------------------------------------------------------------
# 5. Closed-form n_calls (rough)
# ----------------------------------------------------------------------

def n_calls_estimate(topology: str, k: int, rounds: int, M: int = 3) -> int:
    """Approximate LLM call count, ignoring ReAct early termination."""
    if topology == "sas":
        return k  # up to k react steps
    if topology == "independent":
        return M * k + 1  # M parallel solvers + 1 synthesizer
    if topology == "centralized":
        return 1 + rounds * (M * k + 1) + 1  # decompose + rounds*(M workers + review) + synthesis
    if topology == "decentralized":
        return M * k * (rounds + 1)  # M debaters * k react * (R+1) rounds
    if topology == "hybrid":
        return 1 + rounds * (M * k + 1) + 1  # similar to centralized
    return k


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    print("Loading task records...")
    df = load_task_records()
    print(f"  {len(df)} task rows, {df['task_id'].nunique()} unique tasks, "
          f"{df['benchmark'].nunique()} benchmarks")

    print("\nLoading question text per benchmark...")
    q_lookup = {}
    print("  qampari ...")
    q_lookup["qampari"] = load_qampari_questions()
    print(f"    {len(q_lookup['qampari'])} questions")
    print("  fanoutqa ...")
    q_lookup["fanoutqa"] = load_fanoutqa_questions()
    print(f"    {len(q_lookup['fanoutqa'])} questions")
    print("  swebench ...")
    q_lookup["swebench"] = load_swebench_questions()
    print(f"    {len(q_lookup['swebench'])} questions")
    # browsecomp+ and workbench: question text not easily accessible — skip semantic features
    q_lookup["browsecomp_plus"] = {}
    q_lookup["workbench"] = {}

    # Engineer per-task features
    print("\nEngineering features...")
    feat_rows = []
    for _, r in df.iterrows():
        f = engineer_features(r["task_id"], r["benchmark"], q_lookup)
        feat_rows.append(f)
    feats_df = pd.DataFrame(feat_rows, index=df.index)
    df = pd.concat([df, feats_df], axis=1)
    print(f"  added {len(feats_df.columns)} content features")
    print(f"  question coverage by benchmark:")
    for b in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == b]
        coverage = (sub["q_len_words"] > 0).mean()
        print(f"    {b:<20}  {coverage:.0%}")

    print("\nAdding sentence embeddings (only where question text is available)...")
    df = add_embeddings(df, q_lookup)

    # Filter to benchmarks where we have question text — others have zero embeddings and won't help
    has_text = df["q_len_words"] > 0
    df_text = df[has_text].copy()
    print(f"\n  rows with question text: {len(df_text)} / {len(df)}")
    print(f"  unique tasks with text:   {df_text['task_id'].nunique()}")

    # Define feature sets
    cat_features = ["benchmark", "topology"]
    structural_num = ["k", "rounds"]
    content_num = [
        "q_len_chars", "q_len_words", "q_len_sentences", "q_n_named_caps",
        "answer_cardinality", "decomposition_size", "n_entities", "n_evidence",
    ] + [f"qtype_{n}" for n, _ in QUESTION_TYPE_PATTERNS]
    emb_num = [c for c in df_text.columns if c.startswith("emb_")]

    print("\n=== Predicting C_total (completion tokens) ===")
    print("Leave-task-out R² (5-fold GroupKFold by task_id, on text-available subset):")
    res_C = {}
    res_C["structural_only"] = cv_loto(df_text, "C", cat_features, structural_num, "structural only")
    res_C["+content_features"] = cv_loto(df_text, "C", cat_features, structural_num + content_num, "+ content")
    res_C["+content+embeddings"] = cv_loto(df_text, "C", cat_features, structural_num + content_num + emb_num, "+ content + embeddings")

    print("\n=== Predicting P_total (prompt tokens) ===")
    res_P = {}
    res_P["structural_only"] = cv_loto(df_text, "P", cat_features, structural_num, "structural only")
    res_P["+content_features"] = cv_loto(df_text, "P", cat_features, structural_num + content_num, "+ content")
    res_P["+content+embeddings"] = cv_loto(df_text, "P", cat_features, structural_num + content_num + emb_num, "+ content + embeddings")

    print("\n=== Combining into Ê via per-call regression ===")
    # Use the BEST predictors (with embeddings) to compose energy
    df_text["N_calls_est"] = [n_calls_estimate(r.topology, r.k, r.rounds) for r in df_text.itertuples()]
    df_text["C_pred"] = np.expm1(res_C["+content+embeddings"]["preds_log"])
    df_text["P_pred"] = np.expm1(res_P["+content+embeddings"]["preds_log"])
    df_text["E_pred"] = (
        df_text["N_calls_est"] * A_INT
        + B_PR * df_text["P_pred"]
        + C_DE * df_text["C_pred"]
    ).clip(lower=1)

    e_true = df_text["E"].to_numpy()
    e_pred = df_text["E_pred"].to_numpy()
    r2 = r2_score(e_true, e_pred)
    mape_med = float(np.median(np.abs(e_pred - e_true) / np.maximum(e_true, 1)))
    mape_mean = float(np.mean(np.abs(e_pred - e_true) / np.maximum(e_true, 1)))
    print(f"  Composed Ê: R²={r2:.3f}  MAPE_med={mape_med:.2%}  MAPE_mean={mape_mean:.2%}")

    print("\n=== Direct Ê prediction with full features (for comparison) ===")
    res_E = {}
    res_E["structural_only"] = cv_loto(df_text, "E", cat_features, structural_num, "structural only")
    res_E["+content_features"] = cv_loto(df_text, "E", cat_features, structural_num + content_num, "+ content")
    res_E["+content+embeddings"] = cv_loto(df_text, "E", cat_features, structural_num + content_num + emb_num, "+ content + embeddings")

    # Save metrics CSV
    table = []
    for target, dct in [("C", res_C), ("P", res_P), ("E_direct", res_E)]:
        for variant, m in dct.items():
            table.append(dict(target=target, variant=variant, r2=m["r2"], mape_med=m["mape_med"], mape_mean=m["mape_mean"]))
    table.append(dict(target="E_composed", variant="best (C+embed × P+embed × N)", r2=r2, mape_med=mape_med, mape_mean=mape_mean))
    pd.DataFrame(table).to_csv(os.path.join(OUT_DIR, "full_predictor_metrics.csv"), index=False)

    # Scatter plot of composed energy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    samp_idx = np.random.RandomState(0).choice(len(df_text), size=min(2000, len(df_text)), replace=False)
    ax.scatter(e_true[samp_idx], e_pred[samp_idx], s=8, alpha=0.4, color="#1f77b4")
    lo, hi = max(1, e_true.min()), e_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured E (J)")
    ax.set_ylabel("Composed Ê = N·a + b·P̂ + c·Ĉ (J)")
    ax.set_title(f"Composed energy prediction\nR² = {r2:.3f}, MAPE_med = {mape_med:.1%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar comparison: structural / +content / +embeddings on E_composed-equivalent (approximate via C)
    ax = axes[1]
    variants = ["structural_only", "+content_features", "+content+embeddings"]
    r2_C = [res_C[v]["r2"] for v in variants]
    r2_P = [res_P[v]["r2"] for v in variants]
    r2_E = [res_E[v]["r2"] for v in variants]
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, r2_C, w, label="Ĉ R²", color="#d62728")
    ax.bar(x, r2_P, w, label="P̂ R²", color="#1f77b4")
    ax.bar(x + w, r2_E, w, label="Ê direct R²", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, fontsize=9)
    ax.set_ylabel("Leave-task-out R²")
    ax.set_title("Feature ablation: structural → +content → +embeddings")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "full_predictor_scatter.png"), dpi=140, bbox_inches="tight")
    print(f"\n  saved analysis/a5000_figs/full_predictor_scatter.png")
    print(f"  saved analysis/a5000_figs/full_predictor_metrics.csv")


if __name__ == "__main__":
    main()
