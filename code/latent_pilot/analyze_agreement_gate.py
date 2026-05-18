"""Paired analysis: Text MAS with vs without agreement-gated Phase 2 skip.

Expects two jsonls from run_agreement_gate_qwen3.sbatch on the same 30 tasks:
  - baseline:    eval_*_m0.jsonl                 (no gating, always-debate)
  - gated:       eval_*_m0_agreeskip.jsonl       (skip Phase 2 if Jaccard >= 0.5)

Answers (per task where possible):
  1. Does skipping preserve accuracy?
     - On tasks gated skipped: does text_loose_acc match baseline?
     - On tasks gated ran debate: should be unchanged.
  2. How much energy saved by skipping?
     - Aggregate and per-task delta.
  3. Does agreement score correctly identify "debate wasted" tasks?
     - Correlate agreement with baseline's (post-debate - pre-debate) accuracy.
"""
import json
import statistics
from pathlib import Path

RESULTS = Path("mas-energy/results/latent_pilot")


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def analyze_one(model_tag, benchmark, m=0):
    print(f"\n{'='*78}")
    print(f"{model_tag} / {benchmark} (m_latent={m})")
    print('=' * 78)

    base = load(f"eval_{model_tag}_{benchmark}_agentic_k10_m{m}.jsonl")
    gated = load(f"eval_{model_tag}_{benchmark}_agentic_k10_m{m}_agreeskip.jsonl")

    if base is None:
        print(f"MISSING baseline jsonl")
        return
    if gated is None:
        print(f"MISSING gated jsonl (--enable-agreement-skip run)")
        return
    _run_analysis(base, gated)


def _run_analysis(base, gated):

    base_by_id = {r["task_id"]: r for r in base}
    gated_by_id = {r["task_id"]: r for r in gated}
    common = sorted(set(base_by_id) & set(gated_by_id))

    if not common:
        print("No overlapping task_ids between baseline and gated runs")
        return

    # Distribution of agreement scores across tasks
    scores = [gated_by_id[t].get("text_phase1_agreement") or 0.0 for t in common]
    print(f"Agreement score distribution across {len(common)} tasks:")
    print(f"  mean={statistics.mean(scores):.3f}  median={statistics.median(scores):.3f}")
    print(f"  min={min(scores):.3f}  max={max(scores):.3f}")
    thresh = 0.5
    above = sum(1 for s in scores if s >= thresh)
    print(f"  above threshold {thresh}: {above}/{len(common)} tasks ({above/len(common):.0%})")

    # Partition tasks by gating decision
    skipped_ids = [t for t in common if gated_by_id[t].get("text_skipped_debate")]
    ran_debate_ids = [t for t in common if not gated_by_id[t].get("text_skipped_debate")]
    print(f"\nGating decisions: skipped={len(skipped_ids)}  ran debate={len(ran_debate_ids)}")

    def subset_stats(ids, label):
        if not ids:
            print(f"\n{label}: (empty)")
            return
        base_la = statistics.mean([base_by_id[t]["text_loose_accuracy"] for t in ids])
        gated_la = statistics.mean([gated_by_id[t]["text_loose_accuracy"] for t in ids])
        base_f1 = statistics.mean([base_by_id[t]["text_f1"] for t in ids])
        gated_f1 = statistics.mean([gated_by_id[t]["text_f1"] for t in ids])
        base_e = statistics.mean([base_by_id[t]["text_energy_j"] for t in ids])
        gated_e = statistics.mean([gated_by_id[t]["text_energy_j"] for t in ids])
        delta_la = gated_la - base_la
        delta_e_pct = (gated_e - base_e) / max(base_e, 1) * 100
        print(f"\n{label} (n={len(ids)}):")
        print(f"  loose_acc: baseline={base_la:.3f}  gated={gated_la:.3f}  delta={delta_la:+.3f}")
        print(f"  F1:        baseline={base_f1:.3f}  gated={gated_f1:.3f}")
        print(f"  energy:    baseline={base_e:>8.0f}J  gated={gated_e:>8.0f}J  delta={delta_e_pct:+.1f}%")

    subset_stats(skipped_ids, "Tasks where gate SKIPPED Phase 2")
    subset_stats(ran_debate_ids, "Tasks where gate kept Phase 2")
    subset_stats(common, "All tasks (aggregate)")

    # Correctness of gating: on SKIPPED tasks, did baseline's debate actually help?
    # If baseline's text_loose_acc is close to its single_loose_acc on these tasks,
    # debate didn't help — gating skipped correctly.
    if skipped_ids:
        single_la_on_skipped = statistics.mean(
            [base_by_id[t]["single_loose_accuracy"] for t in skipped_ids]
        )
        text_la_on_skipped = statistics.mean(
            [base_by_id[t]["text_loose_accuracy"] for t in skipped_ids]
        )
        diff = text_la_on_skipped - single_la_on_skipped
        print(f"\nGating correctness check (on SKIPPED tasks):")
        print(f"  baseline single_loose_acc: {single_la_on_skipped:.3f}")
        print(f"  baseline text_loose_acc:   {text_la_on_skipped:.3f}")
        print(f"  debate-lift on these tasks: {diff:+.3f}  "
              f"{'(gate correctly skipped — debate did not help)' if diff < 0.03 else '(gate skipped tasks where debate WOULD have helped — miss)'}")

    if ran_debate_ids:
        single_la = statistics.mean(
            [base_by_id[t]["single_loose_accuracy"] for t in ran_debate_ids]
        )
        text_la = statistics.mean(
            [base_by_id[t]["text_loose_accuracy"] for t in ran_debate_ids]
        )
        diff = text_la - single_la
        print(f"\nGating correctness check (on DEBATED tasks):")
        print(f"  baseline single_loose_acc: {single_la:.3f}")
        print(f"  baseline text_loose_acc:   {text_la:.3f}")
        print(f"  debate-lift on these tasks: {diff:+.3f}  "
              f"{'(gate correctly ran debate — it helped)' if diff > 0.03 else '(gate ran debate that did not help)'}")


def main():
    # Analyze both benchmarks if their jsonls are present
    for benchmark in ("qampari", "fanoutqa"):
        analyze_one("Qwen_Qwen3-8B", benchmark, m=0)


if __name__ == "__main__":
    main()
