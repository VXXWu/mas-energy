"""Train the topology-selection classifier with leave-one-benchmark-out CV.

Reads orchestrator_train.jsonl (from collect_orchestrator_data.py) and:
  1. Trains on pooled data with 5-fold CV (in-distribution AUC)
  2. Runs leave-one-benchmark-out CV (generalization across task types)
  3. Reports AUC vs agreement-only baseline at each evaluation level
  4. Saves the final pooled-data model to disk for run_orchestrator_eval.py

The classifier is plain Python logistic regression (no sklearn/numpy
dependency — mas-latent env already thrashed, avoid bare pip install).

Usage:
    python train_topology_classifier.py \
        --data orchestrator_train.jsonl \
        --out-path orchestrator_classifier.json
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

FEATURE_NAMES = ["agreement", "mean_calls", "dup_rate", "length_var", "phase1_energy"]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def standardize(X):
    """Return standardized X and (means, stds) for later inference-time use."""
    d = len(X[0])
    means = [statistics.mean(r[k] for r in X) for k in range(d)]
    stds = []
    for k in range(d):
        col = [r[k] for r in X]
        if len(col) > 1:
            s = statistics.stdev(col)
            stds.append(s if s > 1e-6 else 1.0)
        else:
            stds.append(1.0)
    X_std = [[(r[k] - means[k]) / stds[k] for k in range(d)] for r in X]
    return X_std, means, stds


def apply_standardize(X, means, stds):
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
            # L2 regularization
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


def cv_auc(X_raw, y, k=5, seed=42):
    idx = list(range(len(X_raw)))
    random.seed(seed)
    random.shuffle(idx)
    fold = max(len(idx) // k, 1)
    aucs = []
    for f in range(k):
        val_ids = idx[f * fold : (f + 1) * fold]
        train_ids = [i for i in idx if i not in set(val_ids)]
        if sum(y[i] for i in train_ids) in (0, len(train_ids)):
            continue
        X_tr = [X_raw[i] for i in train_ids]
        X_va = [X_raw[i] for i in val_ids]
        X_tr_std, m, s = standardize(X_tr)
        X_va_std = apply_standardize(X_va, m, s)
        w, b = fit_lr(X_tr_std, [y[i] for i in train_ids])
        scores = [predict(w, b, x) for x in X_va_std]
        aucs.append(auc(scores, [y[i] for i in val_ids]))
    return aucs


def loocv_by_benchmark(rows, X_raw, y, benchmarks):
    """Leave-one-benchmark-out: for each benchmark B, train on all OTHER
    benchmarks, eval on B. Tests cross-benchmark generalization."""
    unique_bench = sorted(set(benchmarks))
    results = {}
    for held in unique_bench:
        train_ids = [i for i, b in enumerate(benchmarks) if b != held]
        val_ids = [i for i, b in enumerate(benchmarks) if b == held]
        if not val_ids or not train_ids:
            continue
        if sum(y[i] for i in train_ids) in (0, len(train_ids)):
            results[held] = ("skipped: degenerate training labels", len(val_ids))
            continue
        X_tr = [X_raw[i] for i in train_ids]
        X_va = [X_raw[i] for i in val_ids]
        X_tr_std, m, s = standardize(X_tr)
        X_va_std = apply_standardize(X_va, m, s)
        w, b = fit_lr(X_tr_std, [y[i] for i in train_ids])
        scores = [predict(w, b, x) for x in X_va_std]
        val_y = [y[i] for i in val_ids]
        results[held] = (auc(scores, val_y), len(val_ids))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    X_raw = [[r["features"][f] for f in FEATURE_NAMES] for r in rows]
    y = [r["label"] for r in rows]
    benchmarks = [r["benchmark"] for r in rows]

    print(f"Loaded {len(rows)} examples.")
    print(f"Class balance: pos={sum(y)} neg={len(y)-sum(y)} ({sum(y)/max(len(y),1):.1%} pos)")
    print(f"Benchmarks: {sorted(set(benchmarks))}")

    if sum(y) == 0 or sum(y) == len(y):
        print("Degenerate labels — cannot train. Need more diverse data.")
        return

    # Pooled 5-fold CV
    aucs = cv_auc(X_raw, y, k=args.cv_folds)
    if aucs:
        print(f"\nPooled {args.cv_folds}-fold CV AUC: mean={statistics.mean(aucs):.3f} "
              f"folds={[f'{a:.3f}' for a in aucs]}")
    else:
        print("\nNot enough data for pooled CV.")

    # Agreement-only baseline
    agreement_scores = [x[0] for x in X_raw]
    base_auc = auc([-s for s in agreement_scores], y)  # high agreement → less likely debate
    print(f"Baseline (agreement alone): AUC = {base_auc:.3f}")

    # Leave-one-benchmark-out
    print("\nLeave-one-benchmark-out generalization:")
    loocv = loocv_by_benchmark(rows, X_raw, y, benchmarks)
    for bench, (val, n) in sorted(loocv.items()):
        if isinstance(val, float):
            print(f"  held-out {bench:<15} n={n:>3}  AUC={val:.3f}")
        else:
            print(f"  held-out {bench:<15} n={n:>3}  {val}")

    # Final model on full data
    X_std, means, stds = standardize(X_raw)
    w, b = fit_lr(X_std, y)
    print(f"\nFinal pooled-data model coefficients:")
    for f, v in zip(FEATURE_NAMES, w):
        print(f"  {f:<16} {v:+.3f}")
    print(f"  {'bias':<16} {b:+.3f}")

    # Save
    model = {
        "feature_names": FEATURE_NAMES,
        "weights": w,
        "bias": b,
        "standardize_means": means,
        "standardize_stds": stds,
        "training_stats": {
            "n_train": len(rows),
            "positive_frac": sum(y) / len(y),
            "pooled_cv_auc": statistics.mean(aucs) if aucs else None,
            "agreement_only_auc": base_auc,
            "loocv_aucs": {b: v[0] if isinstance(v[0], float) else None
                           for b, v in loocv.items()},
        },
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved classifier to {args.out_path}")


if __name__ == "__main__":
    main()
