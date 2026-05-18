"""Train a classifier that predicts whether a task has high Collaboration
Dividend (CD) — i.e. whether paying for inter-agent communication is
worth the accuracy gain.

Reads collaboration_dividend.jsonl (from analyze_collaboration_dividend.py)
and joins with task-level features available at runtime:

  From Independent run alone (pre-Decent):
    - A_indep:              the Independent-topology accuracy on this task
    - E_indep:              Independent-topology energy (task complexity proxy)

  From task metadata (available pre-inference):
    - question_len:         character length of the question
    - n_commas_in_q:        proxy for "how many sub-entities asked about"

Label:
  high_CD = (delta_A > margin) AND (CD > CD_threshold)

Pooled 5-fold CV + leave-one-benchmark-out CV. Saves the final model for
use by simulate_orchestrator_pareto.py.

Usage:
    python train_cd_classifier.py \
        --cd-data mas-energy/results/latent_pilot/collaboration_dividend.jsonl \
        --out-path mas-energy/results/latent_pilot/cd_classifier.json
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


# IMPORTANT: feature selection affects deployability.
# `A_indep` is the ACCURACY of Independent topology on a given task. It
# requires ground-truth labels to compute, and is therefore NOT available
# at deployment time. Including it yields strong AUC (~0.74) but the
# resulting classifier is an "oracle-assisted" upper bound, not a
# deployable policy.
#
# To train a deployable classifier, use RUNTIME_FEATURES below. These are
# knowable immediately after running Independent without any ground truth.
# Current runtime-only AUC is ~0.53 (near random), indicating the signal
# we need is NOT captured by simple post-Indep scalar features — richer
# signals (Phase-1 agent output diversity, confidence, embeddings) are
# needed. That's the open feature-engineering problem.

ORACLE_FEATURE_NAMES = [
    "A_indep",            # leaks ground truth — NOT deployable
    "E_indep",
    "log_E_indep",
    "delta_A_sign_indep", # derived from A_indep — also leaks
    "id_length",
    "n_underscores",
]

RUNTIME_FEATURE_NAMES = [
    # Pre-inference features (available from question alone, before ANY topology runs):
    "q_n_words", "q_n_commas", "q_reasoning_markers", "q_cardinality_markers",
    "q_multihop_markers", "q_proper_nouns", "q_proper_noun_density",
    "q_n_subord_conjunctions", "q_mean_word_len",
    # Creative features aligned with Kim et al.'s parallelizability axis:
    "q_is_imperative", "q_propn_first_third", "q_propn_last_third",
    "q_n_definite_refs", "q_is_what_did", "q_is_which_of", "q_is_why_how_did",
    "q_is_both_and", "q_has_year", "q_n_relative_clauses", "q_word_len_var",
    # Derived: predicted answer cardinality from text-trained regressor:
    "predicted_answer_count",
    # LLM-as-judge difficulty rating (single forward pass per question,
    # cluster-computed via run_llm_judge_qwen3.sbatch):
    "llm_difficulty_rating",
    # Task-id hints:
    "id_length", "n_underscores",
]

# Default: runtime-only. Pass --include-oracle-features to get the upper bound.
FEATURE_NAMES = RUNTIME_FEATURE_NAMES


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def extract_features(row):
    """Compute features from a CD row.

    Runtime-only features (preferred for deployable classifier) come from
    question text alone (text_features dict). Oracle features (A_indep
    and derivatives) leak ground truth and are included only for upper-
    bound analysis.
    """
    a_i = row["A_indep"]
    e_i = row["E_indep"]
    tid_feats = row.get("task_id_features", {})
    text_feats = row.get("text_features", {}) or {}
    out = {
        # Oracle features (leak ground truth):
        "A_indep": a_i,
        "E_indep": e_i,
        "log_E_indep": math.log1p(e_i),
        "delta_A_sign_indep": 1.0 if a_i > 0.3 else 0.0,
        # Task-id features:
        "id_length": float(tid_feats.get("id_length", 0)),
        "n_underscores": float(tid_feats.get("n_underscores", 0)),
    }
    # Text features (when available from enrich_cd_features.py):
    for k in (
        "q_n_words", "q_n_commas", "q_reasoning_markers", "q_cardinality_markers",
        "q_multihop_markers", "q_proper_nouns", "q_proper_noun_density",
        "q_n_subord_conjunctions", "q_mean_word_len", "q_n_chars",
        "q_n_sentences", "q_n_conjunctions",
        # Creative features:
        "q_is_imperative", "q_has_question_mark",
        "q_propn_first_third", "q_propn_last_third",
        "q_n_definite_refs",
        "q_is_what_did", "q_is_which_of", "q_is_why_how_did", "q_is_both_and",
        "q_has_year", "q_n_relative_clauses", "q_word_len_var",
        # Predicted answer cardinality (from train_cardinality_predictor.py):
        "predicted_answer_count",
        # LLM-as-judge rating (from judge_difficulty.py):
        "llm_difficulty_rating",
    ):
        out[k] = float(text_feats.get(k, 0))
    return out


def standardize(X):
    d = len(X[0])
    means = [statistics.mean(r[k] for r in X) for k in range(d)]
    stds = []
    for k in range(d):
        col = [r[k] for r in X]
        s = statistics.stdev(col) if len(col) > 1 else 1.0
        stds.append(s if s > 1e-6 else 1.0)
    X_s = [[(r[k] - means[k]) / stds[k] for k in range(d)] for r in X]
    return X_s, means, stds


def apply_std(X, means, stds):
    return [[(r[k] - means[k]) / stds[k] for k in range(len(means))] for r in X]


def fit_lr(X, y, lr=0.05, epochs=500, l2=1e-3):
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        dw = [0.0] * d
        db = 0.0
        for xi, yi in zip(X, y):
            z = sum(w[k] * xi[k] for k in range(d)) + b
            p = _sigmoid(z)
            err = p - yi
            for k in range(d):
                dw[k] += err * xi[k]
            db += err
        for k in range(d):
            w[k] -= lr * (dw[k] / n + l2 * w[k])
        b -= lr * db / n
    return w, b


def predict(w, b, x):
    return _sigmoid(sum(w[k] * x[k] for k in range(len(w))) + b)


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n: wins += 1
            elif p == n: wins += 0.5
    return wins / (len(pos) * len(neg))


def cv_auc(X, y, k=5, seed=42):
    idx = list(range(len(X)))
    random.seed(seed); random.shuffle(idx)
    fold = max(len(idx) // k, 1)
    aucs = []
    for f in range(k):
        val_ids = idx[f * fold : (f + 1) * fold]
        train_ids = [i for i in idx if i not in set(val_ids)]
        if sum(y[i] for i in train_ids) in (0, len(train_ids)):
            continue
        X_tr = [X[i] for i in train_ids]
        X_va = [X[i] for i in val_ids]
        X_tr_s, m, s = standardize(X_tr)
        X_va_s = apply_std(X_va, m, s)
        w, b = fit_lr(X_tr_s, [y[i] for i in train_ids])
        scores = [predict(w, b, x) for x in X_va_s]
        aucs.append(auc(scores, [y[i] for i in val_ids]))
    return aucs


def loocv_by_benchmark(rows, X, y, benchmarks):
    unique = sorted(set(benchmarks))
    results = {}
    for held in unique:
        tr = [i for i, b in enumerate(benchmarks) if b != held]
        va = [i for i, b in enumerate(benchmarks) if b == held]
        if not va or not tr:
            continue
        if sum(y[i] for i in tr) in (0, len(tr)):
            results[held] = (None, len(va), "degenerate labels")
            continue
        X_tr = [X[i] for i in tr]
        X_va = [X[i] for i in va]
        X_tr_s, m, s = standardize(X_tr)
        X_va_s = apply_std(X_va, m, s)
        w, b = fit_lr(X_tr_s, [y[i] for i in tr])
        scores = [predict(w, b, x) for x in X_va_s]
        results[held] = (auc(scores, [y[i] for i in va]), len(va), None)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd-data", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="high_CD requires delta_A > margin")
    ap.add_argument("--cd-threshold", type=float, default=1e-5,
                    help="high_CD requires CD (acc/joule) > threshold")
    ap.add_argument("--include-oracle-features", action="store_true",
                    help="Include A_indep and delta_A_sign_indep — these leak "
                         "ground truth and yield a non-deployable upper-bound "
                         "classifier. Off by default.")
    args = ap.parse_args()

    global FEATURE_NAMES
    FEATURE_NAMES = ORACLE_FEATURE_NAMES if args.include_oracle_features else RUNTIME_FEATURE_NAMES
    print(f"Using features: {FEATURE_NAMES}  "
          f"({'ORACLE — not deployable' if args.include_oracle_features else 'RUNTIME — deployable'})")

    rows = [json.loads(l) for l in open(args.cd_data)]
    print(f"Loaded {len(rows)} paired tasks.")
    if not rows:
        print("No data. Run analyze_collaboration_dividend.py first.")
        return

    X = []
    y = []
    benchmarks = []
    for r in rows:
        feats = extract_features(r)
        X.append([feats[f] for f in FEATURE_NAMES])
        label = 1 if (r["delta_A"] > args.margin and r["CD"] > args.cd_threshold) else 0
        y.append(label)
        benchmarks.append(r.get("benchmark", "unknown"))

    pos = sum(y)
    print(f"Class balance: {pos}/{len(y)} positive ({pos/max(len(y),1):.1%}) — "
          f"label criterion: delta_A > {args.margin} AND CD > {args.cd_threshold}")

    if pos == 0 or pos == len(y):
        print("Degenerate labels — cannot train.")
        return

    # Pooled 5-fold CV
    aucs = cv_auc(X, y, k=5)
    print(f"\nPooled 5-fold CV AUC: mean={statistics.mean(aucs):.3f} "
          f"folds={[f'{a:.3f}' for a in aucs]}" if aucs else "CV skipped.")

    # A_indep-alone baseline: lower A_indep → more likely collaboration helps
    base_auc = auc([-x[0] for x in X], y)
    print(f"Baseline (A_indep alone, inverted): AUC = {base_auc:.3f}")

    # Leave-one-benchmark-out
    print(f"\nLeave-one-benchmark-out:")
    for bench, (val, n, err) in sorted(loocv_by_benchmark(rows, X, y, benchmarks).items()):
        if err:
            print(f"  {bench:<15} n={n:>3}  {err}")
        else:
            print(f"  {bench:<15} n={n:>3}  AUC={val:.3f}")

    # Final model
    X_std, means, stds = standardize(X)
    w, b = fit_lr(X_std, y)
    print(f"\nFinal pooled-data model:")
    for f, v in zip(FEATURE_NAMES, w):
        print(f"  {f:<20} {v:+.3f}")
    print(f"  {'bias':<20} {b:+.3f}")

    model = {
        "feature_names": FEATURE_NAMES,
        "weights": w,
        "bias": b,
        "standardize_means": means,
        "standardize_stds": stds,
        "label_criterion": {
            "margin": args.margin,
            "cd_threshold": args.cd_threshold,
        },
        "training_stats": {
            "n_train": len(rows),
            "positive_frac": pos / len(rows),
            "pooled_cv_auc_mean": statistics.mean(aucs) if aucs else None,
            "baseline_A_indep_auc": base_auc,
        },
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved CD classifier to {args.out_path}")


if __name__ == "__main__":
    main()
