"""Train a classifier that predicts whether debate will help on a task.

This is direction (b): learn a head that decides "run Phase 2 debate" vs
"skip to synthesis" from features extracted after Phase 1.

Features (all scalar, computable without model forward passes):
  - phase1_agreement (Jaccard across agents' Phase 1 answers)
  - mean tool calls per agent (more calls = harder task?)
  - tool-call duplicate rate across agents (high dup = agents converged early)
  - phase1 response length variance (disagreement in verbosity?)

Label:
  debate_helps = (text_loose_acc > single_loose_acc + margin)
  where margin = 0.05 by default to avoid noise triggering positives.

Training data: any jsonl that includes text_phase1_tool_calls and
text_phase1_agreement fields — i.e. produced after the metadata_out
instrumentation. Aggregates across multiple jsonls if given a glob.

Usage:
    python train_debate_classifier.py [jsonl_glob]

Output: LR coefficients, cross-val AUC, and a held-out confusion matrix.
Compare AUC vs "agreement alone" baseline to see if other features help.
"""
import json
import sys
import statistics
from pathlib import Path

RESULTS = Path("mas-energy/results/latent_pilot")


def extract_features(row):
    """Return dict of scalar features for one task row, or None if required
    fields are missing."""
    agreement = row.get("text_phase1_agreement")
    p1_calls = row.get("text_phase1_tool_calls")
    if agreement is None or p1_calls is None:
        return None
    # Mean calls per agent
    calls_per_agent = [len(c) for c in p1_calls]
    mean_calls = statistics.mean(calls_per_agent) if calls_per_agent else 0.0
    # Duplicate rate: pairs of calls across agents sharing the same normalized
    # (name, args_dict) key. Cheap approx.
    def norm(c):
        name = c.get("name", "")
        args = c.get("args", {}) or {}
        q = args.get("query", "") if isinstance(args, dict) else ""
        return f"{name}({q.strip().lower()})"
    flat = [norm(c) for calls in p1_calls for c in calls]
    n = len(flat)
    unique = len(set(flat))
    dup_rate = 1.0 - (unique / n) if n > 0 else 0.0
    # Response-length variance
    if "text_phase1_tool_calls" in row:
        # proxy for response length: number of calls × avg query length
        lens = [sum(len(c.get("args", {}).get("query", "")) for c in calls) for calls in p1_calls]
        length_var = statistics.variance(lens) if len(lens) > 1 else 0.0
    else:
        length_var = 0.0
    return {
        "agreement": float(agreement),
        "mean_calls": mean_calls,
        "dup_rate": dup_rate,
        "length_var": length_var,
    }


def extract_label(row, margin=0.05):
    """debate_helps = True iff text_loose_acc > single_loose_acc + margin"""
    text_la = row.get("text_loose_accuracy")
    single_la = row.get("single_loose_accuracy")
    if text_la is None or single_la is None:
        return None
    return 1 if (text_la - single_la) > margin else 0


def logistic_regression_fit(X, y, lr=0.05, epochs=500):
    """Plain NumPy-free LR via manual gradient descent. Avoids sklearn dep."""
    import math
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        dw = [0.0] * d
        db = 0.0
        for xi, yi in zip(X, y):
            z = sum(w[k] * xi[k] for k in range(d)) + b
            # sigmoid, numerically stable
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                e = math.exp(z)
                p = e / (1.0 + e)
            err = p - yi
            for k in range(d):
                dw[k] += err * xi[k]
            db += err
        for k in range(d):
            w[k] -= lr * dw[k] / n
        b -= lr * db / n
    return w, b


def predict(w, b, x):
    import math
    z = sum(w[k] * x[k] for k in range(len(w))) + b
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def auc(scores, labels):
    """ROC-AUC via pairwise concordance."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "eval_Qwen_Qwen3-8B_*.jsonl"
    paths = sorted(RESULTS.glob(pattern))

    X, y, feat_names = [], [], ["agreement", "mean_calls", "dup_rate", "length_var"]
    for p in paths:
        for line in open(p):
            row = json.loads(line)
            feats = extract_features(row)
            lab = extract_label(row)
            if feats is None or lab is None:
                continue
            X.append([feats[f] for f in feat_names])
            y.append(lab)

    print(f"Loaded {len(X)} labeled examples from {len(paths)} jsonl(s) matching '{pattern}'")
    if not X:
        print("NO training data. Need jsonls with text_phase1_tool_calls and text_phase1_agreement.")
        print("Run agentic_latentmas.py with current metadata_out instrumentation to generate them.")
        return
    pos = sum(y)
    print(f"Class balance: positive={pos}  negative={len(y)-pos}  ({pos/len(y):.1%} positive)")
    if pos == 0 or pos == len(y):
        print("Degenerate labels (all same class); skipping train.")
        return

    # Simple 5-fold cross-validation for AUC
    import random
    random.seed(42)
    idx = list(range(len(X)))
    random.shuffle(idx)
    k = 5
    fold_size = len(idx) // k
    fold_aucs = []
    for fold in range(k):
        val_ids = idx[fold * fold_size: (fold + 1) * fold_size]
        train_ids = [i for i in idx if i not in set(val_ids)]
        X_tr = [X[i] for i in train_ids]
        y_tr = [y[i] for i in train_ids]
        X_va = [X[i] for i in val_ids]
        y_va = [y[i] for i in val_ids]
        if sum(y_tr) == 0 or sum(y_tr) == len(y_tr):
            continue
        w, b = logistic_regression_fit(X_tr, y_tr)
        scores = [predict(w, b, x) for x in X_va]
        fold_aucs.append(auc(scores, y_va))

    print(f"\n5-fold CV AUC (classifier with {', '.join(feat_names)}):")
    if fold_aucs:
        print(f"  mean={statistics.mean(fold_aucs):.3f}  folds={[f'{a:.3f}' for a in fold_aucs]}")
    else:
        print("  (not enough data for CV)")

    # Baseline: use agreement alone
    agreement_scores = [x[0] for x in X]
    # Since higher agreement → less likely debate helps → negate for "debate_helps" score
    agreement_neg = [-s for s in agreement_scores]
    base_auc = auc(agreement_neg, y)
    print(f"\nBaseline: agreement-only AUC = {base_auc:.3f}")
    print(f"Classifier delta over agreement-only: {statistics.mean(fold_aucs) - base_auc:+.3f}"
          if fold_aucs else "")

    # Train final model on all data and report coefficients
    w, b = logistic_regression_fit(X, y)
    print(f"\nFinal model coefficients (on full data):")
    for k, v in zip(feat_names, w):
        print(f"  {k:<16} {v:+.3f}")
    print(f"  {'bias':<16} {b:+.3f}")


if __name__ == "__main__":
    main()
