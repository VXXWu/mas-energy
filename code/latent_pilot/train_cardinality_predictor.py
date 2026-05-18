"""Train a predictor of expected answer cardinality from question text.

Trained on QAMPARI's gold_answer_count targets using existing text features.
At test time, this regressor predicts expected answer count from the
question alone — a runtime-observable feature that maps to Kim et al.'s
parallelizability axis (high cardinality = breadth = parallelizable).

Output: adds `predicted_answer_count` feature to each CD row, which the
CD classifier can consume.

Usage:
    python train_cardinality_predictor.py \
        --cd-data mas-energy/results/latent_pilot/collaboration_dividend_enriched.jsonl \
        --out-path mas-energy/results/latent_pilot/collaboration_dividend_final.jsonl
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


FEATURES_FOR_CARDINALITY = [
    "q_n_words", "q_n_commas", "q_cardinality_markers",
    "q_proper_nouns", "q_proper_noun_density",
    "q_is_imperative", "q_is_which_of", "q_n_conjunctions",
    "q_n_definite_refs", "q_has_year",
]


def standardize(X):
    d = len(X[0])
    means = [statistics.mean(r[k] for r in X) for k in range(d)]
    stds = [max(statistics.stdev([r[k] for r in X]), 1e-6) if len(X) > 1 else 1.0
            for k in range(d)]
    X_s = [[(r[k] - means[k]) / stds[k] for k in range(d)] for r in X]
    return X_s, means, stds


def fit_linreg(X, y, lr=0.05, epochs=500, l2=1e-3):
    """Plain linear regression via gradient descent."""
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        dw = [0.0] * d
        db = 0.0
        for xi, yi in zip(X, y):
            pred = sum(w[k] * xi[k] for k in range(d)) + b
            err = pred - yi
            for k in range(d):
                dw[k] += err * xi[k]
            db += err
        for k in range(d):
            w[k] -= lr * (dw[k] / n + l2 * w[k])
        b -= lr * db / n
    return w, b


def predict(w, b, x):
    return sum(w[k] * x[k] for k in range(len(w))) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd-data", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cd_data)]
    # Train on rows that have gold_answer_count (QAMPARI)
    train_rows = [r for r in rows
                  if r.get("gold_answer_count") is not None
                  and r.get("text_features")]
    print(f"Train set: {len(train_rows)}/{len(rows)} rows with gold_answer_count")

    if not train_rows:
        print("No training data — saving pass-through output.")
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return

    X_raw = [[r["text_features"].get(f, 0) for f in FEATURES_FOR_CARDINALITY]
             for r in train_rows]
    y_raw = [float(r["gold_answer_count"]) for r in train_rows]
    # Log-transform targets (cardinality has heavy right tail)
    y = [math.log1p(v) for v in y_raw]

    X_s, means, stds = standardize(X_raw)
    # 5-fold CV on R² for sanity
    random.seed(42)
    idx = list(range(len(X_s)))
    random.shuffle(idx)
    fold = max(len(idx) // 5, 1)
    r2_folds = []
    for f in range(5):
        val_ids = idx[f * fold : (f + 1) * fold]
        tr_ids = [i for i in idx if i not in set(val_ids)]
        X_tr = [X_s[i] for i in tr_ids]
        y_tr = [y[i] for i in tr_ids]
        X_va = [X_s[i] for i in val_ids]
        y_va = [y[i] for i in val_ids]
        w, b = fit_linreg(X_tr, y_tr)
        preds = [predict(w, b, x) for x in X_va]
        ss_res = sum((y_va[i] - preds[i]) ** 2 for i in range(len(y_va)))
        ss_tot = sum((y_va[i] - statistics.mean(y_va)) ** 2 for i in range(len(y_va)))
        r2 = 1 - ss_res / max(ss_tot, 1e-6)
        r2_folds.append(r2)
    print(f"Cardinality regressor 5-fold CV R² on log-targets: "
          f"mean={statistics.mean(r2_folds):.3f}  folds={[f'{r:.3f}' for r in r2_folds]}")

    # Train final on all, apply to all rows (including non-QAMPARI)
    w, b = fit_linreg(X_s, y)
    print(f"\nFinal model coefficients (log-cardinality):")
    for name, val in zip(FEATURES_FOR_CARDINALITY, w):
        print(f"  {name:<26} {val:+.3f}")
    print(f"  {'bias':<26} {b:+.3f}")

    # Enrich all rows with predicted cardinality (where text features exist)
    n_enriched = 0
    for r in rows:
        tf = r.get("text_features", {}) or {}
        if tf:
            x_raw = [tf.get(f, 0) for f in FEATURES_FOR_CARDINALITY]
            x_s = [(x_raw[k] - means[k]) / stds[k] for k in range(len(x_raw))]
            log_pred = predict(w, b, x_s)
            tf["predicted_answer_count"] = math.expm1(log_pred)  # exp(pred)-1
            r["text_features"] = tf
            n_enriched += 1
    print(f"\nEnriched {n_enriched} rows with predicted_answer_count")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Output: {args.out_path}")


if __name__ == "__main__":
    main()
