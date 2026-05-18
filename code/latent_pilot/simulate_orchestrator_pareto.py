"""Simulate deployment of the CD classifier as an orchestrator and compute
the resulting energy/accuracy Pareto point vs always-Indep and always-Decent.

For each task in collaboration_dividend.jsonl:
  1. Compute runtime features (A_indep, E_indep — these ARE the cost of
     running Independent, which must happen anyway to compute the feature).
  2. Classifier predicts P(high_CD).
  3. If P > threshold, ESCALATE: pay E_decent - E_indep extra to get A_decent.
     Else: stop at Independent with A_indep.

Aggregate:
  - Total energy per task (Indep always + escalation cost where triggered)
  - Accuracy per task (Indep or Decent, depending on decision)
  - Escalation rate

Compare to:
  - Always-Independent: total_energy=sum(E_indep), accuracy=mean(A_indep)
  - Always-Decentralized: total_energy=sum(E_decent), accuracy=mean(A_decent)
  - Oracle (cheat): escalate only on truly high-CD tasks. Upper bound on
    deployed gain.

Output: per-benchmark and aggregate Pareto comparison.

Usage:
    python simulate_orchestrator_pareto.py \
        --cd-data mas-energy/results/latent_pilot/collaboration_dividend.jsonl \
        --classifier mas-energy/results/latent_pilot/cd_classifier.json
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def extract_features(row, feature_names):
    a_i = row["A_indep"]
    e_i = row["E_indep"]
    tid_feats = row.get("task_id_features", {})
    text_feats = row.get("text_features", {}) or {}
    feats = {
        "A_indep": a_i,
        "E_indep": e_i,
        "log_E_indep": math.log1p(e_i),
        "delta_A_sign_indep": 1.0 if a_i > 0.3 else 0.0,
        "id_length": float(tid_feats.get("id_length", 0)),
        "n_underscores": float(tid_feats.get("n_underscores", 0)),
    }
    # Text-derived features from enrich_cd_features.py
    for k in (
        "q_n_words", "q_n_commas", "q_reasoning_markers", "q_cardinality_markers",
        "q_multihop_markers", "q_proper_nouns", "q_proper_noun_density",
        "q_n_subord_conjunctions", "q_mean_word_len", "q_n_chars",
        "q_n_sentences", "q_n_conjunctions",
        "q_is_imperative", "q_has_question_mark",
        "q_propn_first_third", "q_propn_last_third",
        "q_n_definite_refs",
        "q_is_what_did", "q_is_which_of", "q_is_why_how_did", "q_is_both_and",
        "q_has_year", "q_n_relative_clauses", "q_word_len_var",
        "predicted_answer_count", "llm_difficulty_rating",
    ):
        feats[k] = float(text_feats.get(k, 0))
    return [feats[f] for f in feature_names]


def predict_p(model, feats_raw):
    x = [(feats_raw[k] - model["standardize_means"][k]) / model["standardize_stds"][k]
         for k in range(len(feats_raw))]
    z = sum(model["weights"][k] * x[k] for k in range(len(x))) + model["bias"]
    return _sigmoid(z)


def simulate(rows, model, threshold=0.5):
    """Return per-task policy decisions + aggregates."""
    results = []
    for r in rows:
        x = extract_features(r, model["feature_names"])
        p = predict_p(model, x)
        escalate = p >= threshold
        # Policy: always pay Indep cost, then pay extra to reach Decent if escalate
        if escalate:
            acc, energy = r["A_decent"], r["E_decent"]
        else:
            acc, energy = r["A_indep"], r["E_indep"]
        results.append({
            "task_id": r["task_id"], "benchmark": r.get("benchmark", "?"),
            "p_high_CD": p, "escalate": escalate,
            "orch_acc": acc, "orch_energy": energy,
            "indep_acc": r["A_indep"], "indep_energy": r["E_indep"],
            "decent_acc": r["A_decent"], "decent_energy": r["E_decent"],
            "delta_A": r["delta_A"], "CD": r["CD"],
        })
    return results


def summarize(sim_results, label):
    n = len(sim_results)
    if n == 0:
        return
    o_a = statistics.mean(r["orch_acc"] for r in sim_results)
    o_e = statistics.mean(r["orch_energy"] for r in sim_results)
    i_a = statistics.mean(r["indep_acc"] for r in sim_results)
    i_e = statistics.mean(r["indep_energy"] for r in sim_results)
    d_a = statistics.mean(r["decent_acc"] for r in sim_results)
    d_e = statistics.mean(r["decent_energy"] for r in sim_results)
    escalated = sum(1 for r in sim_results if r["escalate"])

    # Oracle: escalate iff delta_A > 0 (cheat: we use the ground truth)
    oracle = [r for r in sim_results]
    o_oracle_a = statistics.mean(max(r["indep_acc"], r["decent_acc"]) for r in oracle)
    # Oracle paying minimum energy: if decent is better, pay decent; else indep
    o_oracle_e = statistics.mean(
        r["decent_energy"] if r["decent_acc"] > r["indep_acc"] else r["indep_energy"]
        for r in oracle
    )

    print(f"\n  {label}  (n={n})")
    print(f"    {'Policy':<20} {'Accuracy':>10} {'Energy (J)':>12} {'vs Decent E':>12}")
    print(f"    {'-'*56}")
    print(f"    {'Always Decent':<20} {d_a:>10.3f} {d_e:>12.0f} {'baseline':>12}")
    print(f"    {'Always Indep':<20} {i_a:>10.3f} {i_e:>12.0f} "
          f"{(i_e-d_e)/max(d_e,1)*100:>+11.1f}%")
    print(f"    {'Classifier-gated':<20} {o_a:>10.3f} {o_e:>12.0f} "
          f"{(o_e-d_e)/max(d_e,1)*100:>+11.1f}%")
    print(f"    {'Oracle (cheat)':<20} {o_oracle_a:>10.3f} {o_oracle_e:>12.0f} "
          f"{(o_oracle_e-d_e)/max(d_e,1)*100:>+11.1f}%")
    print(f"    Escalation rate: {escalated}/{n} ({escalated/n:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd-data", type=str, required=True)
    ap.add_argument("--classifier", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-path", type=str,
                    default="mas-energy/results/latent_pilot/orchestrator_sim.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cd_data)]
    model = json.load(open(args.classifier))
    print(f"Loaded {len(rows)} CD rows.")
    print(f"Loaded classifier with features: {model['feature_names']}")

    sim = simulate(rows, model, threshold=args.threshold)
    by_bench = defaultdict(list)
    for r in sim:
        by_bench[r["benchmark"]].append(r)

    print(f"\n{'='*70}\nPareto simulation @ threshold={args.threshold}\n{'='*70}")
    summarize(sim, "ALL")
    for bench in sorted(by_bench.keys()):
        summarize(by_bench[bench], f"benchmark={bench}")

    # Threshold sweep — extend to low values to find the operating point
    # where the classifier actually triggers escalation (imbalanced classes
    # push optimal threshold below 0.5 since positive class is ~10%).
    print(f"\n{'='*70}\nThreshold sweep (aggregate)\n{'='*70}")
    print(f"  {'thresh':>7} {'acc':>8} {'energy':>10} {'escalate %':>12} "
          f"{'vs Always-Dec E':>16} {'vs Always-Indep acc':>21}")
    indep_a_mean = statistics.mean(r["A_indep"] for r in rows)
    for t in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        sim_t = simulate(rows, model, threshold=t)
        a = statistics.mean(r["orch_acc"] for r in sim_t)
        e = statistics.mean(r["orch_energy"] for r in sim_t)
        d_e = statistics.mean(r["decent_energy"] for r in sim_t)
        esc = sum(1 for r in sim_t if r["escalate"]) / max(len(sim_t), 1)
        print(f"  {t:>7.2f} {a:>8.3f} {e:>10.0f} {esc:>11.0%} "
              f"{(e-d_e)/max(d_e,1)*100:>+15.1f}% "
              f"{(a-indep_a_mean):>+20.3f}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in sim:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved per-task simulation to {args.out_path}")


if __name__ == "__main__":
    main()
