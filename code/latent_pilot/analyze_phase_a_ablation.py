"""Phase A ablation analysis: compare counterfactual Decent variants vs the
full-decent baseline to measure how much peer text B actually needs.

Pairs each ablation jsonl against the corresponding full Decent baseline jsonl
(matched by benchmark + task_id). Reports per-ablation:
  - paired ΔAcc (helps/hurts/both-right/both-wrong on the same task ids)
  - paired ΔE
  - paired Δcompletion_tokens
  - bootstrap 95% CI on ΔAcc

Healthy compressibility signal: truncate300 (and even truncate100) preserve
accuracy within ~3 percentage points of full Decent, while empty baseline
hurts substantially. That gap = the empirical compression ceiling.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean


ABLATION_TOPOS = [
    # Phase A: receiver-side counterfactual ablation (modify what receiver sees)
    'decentralized_answer_only',
    'decentralized_truncate300',
    'decentralized_truncate100',
    'decentralized_empty',
    # Strict control for 'empty' — no peer framing at all
    'decentralized_empty_silent',
    # Phase B-2: speaker-side decode reduction (modify what speaker writes)
    'decentralized_terse',
    'decentralized_terse_answer_only',
    # Indep+share: skip debate phase entirely; one text-only refine round
    'independent_share',
    # Indep+share + minimal speaker: stacks structural (drop debate rounds)
    # and within-call (suppress narration) decode reductions.
    'independent_share_minimal',
    # Minimum-output: agent suppresses ALL text during trajectory (no narration
    # between tool calls; just tool calls + final answer). Stricter than terse.
    'decentralized_minimal',
    'decentralized_minimal_empty',
    # Specialist: each agent gets disjoint round-robin tool slice (workbench only — needs 27 tools)
    'decentralized_specialist',
    'decentralized_specialist_empty',
    # Categorical specialist: deployment-faithful WorkBench partition into
    # Scheduler / Communicator / CRM-Analyst roles (mirrors CrewAI/MetaGPT)
    'decentralized_specialist_categorical',
    'decentralized_specialist_categorical_empty',
]


def load_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(l) for l in open(path)]


def acc_field(rec):
    """Return float accuracy in [0,1]. Use loose_accuracy if present, else correct."""
    if rec.get('loose_accuracy') is not None:
        return float(rec['loose_accuracy'])
    return 1.0 if rec.get('correct') else 0.0


def find_jsonls_for_benchmark(root, benchmark, topo):
    """Find all jsonls matching a topology in benchmark's dir.

    Anchor the topology name to '_k<N>' so substring topo names don't
    match each other (e.g., 'decentralized_empty' must NOT match
    'decentralized_empty_silent_k10.jsonl').
    """
    p = Path(root) / benchmark
    if not p.exists(): return []
    return sorted(p.glob(f'*_{topo}_k*.jsonl'))


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05):
    """Bootstrap percentile CI on the mean of deltas."""
    if len(deltas) < 5: return (None, None)
    rng = random.Random(42)
    means = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [deltas[rng.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample)/n)
    means.sort()
    lo = means[int(alpha/2 * n_boot)]
    hi = means[int((1-alpha/2) * n_boot)]
    return (lo, hi)


def pair_topologies(baseline_rows, ablation_rows):
    """Match by task_id + rep, return list of (baseline, ablation) tuples."""
    bb = {(r['task_id'], r.get('rep', 0)): r for r in baseline_rows}
    aa = {(r['task_id'], r.get('rep', 0)): r for r in ablation_rows}
    keys = sorted(set(bb) & set(aa))
    return [(bb[k], aa[k]) for k in keys]


def analyze_pair(baseline_rows, ablation_rows, label):
    pairs = pair_topologies(baseline_rows, ablation_rows)
    n = len(pairs)
    if n == 0: return None

    deltas_acc = [acc_field(a) - acc_field(b) for b, a in pairs]
    deltas_E = [(a.get('gpu_dynamic_energy_joules', 0) or 0) -
                (b.get('gpu_dynamic_energy_joules', 0) or 0) for b, a in pairs]
    deltas_tok = [(a.get('total_completion_tokens', 0) or 0) -
                  (b.get('total_completion_tokens', 0) or 0) for b, a in pairs]

    helps = sum(1 for d in deltas_acc if d > 0)
    hurts = sum(1 for d in deltas_acc if d < 0)
    same = sum(1 for d in deltas_acc if d == 0)

    ci_lo, ci_hi = bootstrap_ci(deltas_acc)

    # Compute baseline/ablation accuracy ONLY over the paired set so they're
    # consistent with the paired ΔAcc. Reporting unpaired means here can give
    # misleading impressions when the two files have different row counts
    # (e.g., partial ablation runs vs full baseline).
    return dict(
        n=n,
        baseline_acc=mean(acc_field(b) for b, a in pairs),
        ablation_acc=mean(acc_field(a) for b, a in pairs),
        delta_acc_paired=mean(deltas_acc),
        delta_acc_ci=(ci_lo, ci_hi),
        helps=helps, hurts=hurts, same=same,
        delta_E_mean=mean(deltas_E),
        delta_completion_tokens_mean=mean(deltas_tok),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-root',
                    default='mas-energy/results/a5000_latent_transcripts',
                    help='Root containing full-decent baseline (per-benchmark subdirs)')
    ap.add_argument('--ablation-roots', nargs='+',
                    default=['mas-energy/results/a5000_phase_a_ablation',
                             'mas-energy/results/a5000_phase_b2_terse'],
                    help='Roots containing ablation jsonls (per-benchmark subdirs). '
                         'First match wins per (bench, topo).')
    ap.add_argument('--benchmarks', nargs='+',
                    default=['qampari', 'fanoutqa', 'math', 'workbench',
                             'browsecomp_plus', 'swebench', 'humaneval'])
    args = ap.parse_args()

    print(f"{'benchmark':<18} {'mode':<28} {'n':>4} {'baseAcc':>8} {'ablAcc':>8} "
          f"{'ΔAcc':>7} {'CI95':<16} {'h/H/=':<10} {'ΔE_J':>8} {'Δtok':>7}")
    print("-" * 122)

    overall = {topo: dict(deltas_acc=[], deltas_E=[], deltas_tok=[]) for topo in ABLATION_TOPOS}

    for bench in args.benchmarks:
        baseline_files = find_jsonls_for_benchmark(args.baseline_root, bench, 'decentralized')
        if not baseline_files:
            print(f"{bench:<18}   (no baseline decent jsonl found)")
            continue
        baseline = []
        for f in baseline_files:
            baseline.extend(load_jsonl(f))
        for topo in ABLATION_TOPOS:
            # Search multiple ablation roots; first one with matching files wins.
            ablation_files = []
            for root in args.ablation_roots:
                ablation_files = find_jsonls_for_benchmark(root, bench, topo)
                if ablation_files:
                    break
            if not ablation_files:
                continue
            ablation = []
            for f in ablation_files:
                ablation.extend(load_jsonl(f))
            res = analyze_pair(baseline, ablation, f"{bench}/{topo}")
            if res is None: continue
            ci = res['delta_acc_ci']
            ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci[0] is not None else "n/a"
            print(f"{bench:<18} {topo:<28} {res['n']:>4} "
                  f"{res['baseline_acc']:>8.3f} {res['ablation_acc']:>8.3f} "
                  f"{res['delta_acc_paired']:>+7.3f} {ci_str:<16} "
                  f"{res['helps']}/{res['hurts']}/{res['same']:<4} "
                  f"{res['delta_E_mean']:>+8.0f} {res['delta_completion_tokens_mean']:>+7.0f}")

            # Aggregate
            pairs = pair_topologies(baseline, ablation)
            overall[topo]['deltas_acc'].extend(acc_field(a) - acc_field(b) for b, a in pairs)
            overall[topo]['deltas_E'].extend((a.get('gpu_dynamic_energy_joules', 0) or 0) -
                                             (b.get('gpu_dynamic_energy_joules', 0) or 0) for b, a in pairs)
            overall[topo]['deltas_tok'].extend((a.get('total_completion_tokens', 0) or 0) -
                                                (b.get('total_completion_tokens', 0) or 0) for b, a in pairs)

    print("\n=== Aggregate across all benchmarks ===")
    print(f"{'mode':<28} {'n':>4} {'ΔAcc':>7} {'CI95':<16} {'ΔE_J':>9} {'Δtok':>7}")
    print("-" * 78)
    for topo in ABLATION_TOPOS:
        d = overall[topo]
        n = len(d['deltas_acc'])
        if n == 0: continue
        ci = bootstrap_ci(d['deltas_acc'])
        ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci[0] is not None else "n/a"
        print(f"{topo:<28} {n:>4} {mean(d['deltas_acc']):>+7.3f} {ci_str:<16} "
              f"{mean(d['deltas_E']):>+9.0f} {mean(d['deltas_tok']):>+7.0f}")


if __name__ == "__main__":
    main()
