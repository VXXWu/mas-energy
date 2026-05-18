"""Paired 4-way structural comparison analysis.

Reads per-task jsonls from the three latent modes (kv_share, cocunut,
phase1_latent) produced by run_structural_comparison_qwen3.sbatch and
reports per-task comparison across text MAS / kv_share / cocunut /
phase1_latent.

Each jsonl contains 30 task rows with fields:
  task_id, single_loose_accuracy, text_loose_accuracy, latent_loose_accuracy,
  single_f1, text_f1, latent_f1,
  single_energy_j, text_energy_j, latent_energy_j

The `latent_*` columns refer to each run's specific mode; the `single_*`
and `text_*` columns should be approximately identical across modes
(they run the same text-baseline in each invocation).

Usage:
    python analyze_structural.py <results_dir> <model_tag>
where <model_tag> is e.g. "Qwen_Qwen3-8B".
"""
import json
import sys
import statistics
from pathlib import Path

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
                       "mas-energy/results/latent_pilot")
    model_tag = sys.argv[2] if len(sys.argv) > 2 else "Qwen_Qwen3-8B"

    base = f"eval_{model_tag}_qampari_agentic_k10_m5"
    kv_share_path = results_dir / f"{base}.jsonl"
    cocunut_path = results_dir / f"{base}_cocunut.jsonl"
    phase1_path = results_dir / f"{base}_phase1_latent.jsonl"

    for p in (kv_share_path, cocunut_path, phase1_path):
        if not p.exists():
            print(f"MISSING: {p}")
            return

    print(f"Loading:\n  {kv_share_path.name}\n  {cocunut_path.name}\n  {phase1_path.name}")
    kv = load_jsonl(kv_share_path)
    co = load_jsonl(cocunut_path)
    p1 = load_jsonl(phase1_path)

    # Index by task_id for paired comparison
    kv_by_id = {r['task_id']: r for r in kv}
    co_by_id = {r['task_id']: r for r in co}
    p1_by_id = {r['task_id']: r for r in p1}

    common = set(kv_by_id) & set(co_by_id) & set(p1_by_id)
    print(f"\nCommon task_ids across all three runs: {len(common)}")

    # Text MAS baselines should be consistent across the three runs
    # (they each re-run the text MAS condition). Report the variance.
    text_f1_ranges = []
    text_la_ranges = []
    text_en_ranges = []
    for tid in common:
        f1s = [kv_by_id[tid]['text_f1'], co_by_id[tid]['text_f1'], p1_by_id[tid]['text_f1']]
        las = [kv_by_id[tid]['text_loose_accuracy'], co_by_id[tid]['text_loose_accuracy'],
               p1_by_id[tid]['text_loose_accuracy']]
        ens = [kv_by_id[tid]['text_energy_j'], co_by_id[tid]['text_energy_j'],
               p1_by_id[tid]['text_energy_j']]
        text_f1_ranges.append(max(f1s) - min(f1s))
        text_la_ranges.append(max(las) - min(las))
        text_en_ranges.append((max(ens) - min(ens)) / max(min(ens), 1))
    print(f"\nText MAS baseline consistency (range across 3 runs, avg over tasks):")
    print(f"  F1 range: {statistics.mean(text_f1_ranges):.3f}")
    print(f"  loose_acc range: {statistics.mean(text_la_ranges):.3f}")
    print(f"  energy relative range: {statistics.mean(text_en_ranges):.2%}")

    # Aggregate each mode's metrics (using its own text baseline)
    print(f"\n{'='*85}")
    print(f"{'Mode':<18} {'loose_acc':>12} {'F1':>10} {'Energy (J)':>14} {'vs Text energy':>18}")
    print(f"{'='*85}")

    # Text MAS (averaged across the three runs as a single estimate)
    text_la = statistics.mean([kv_by_id[t]['text_loose_accuracy'] for t in common] +
                               [co_by_id[t]['text_loose_accuracy'] for t in common] +
                               [p1_by_id[t]['text_loose_accuracy'] for t in common])
    text_f1 = statistics.mean([kv_by_id[t]['text_f1'] for t in common] +
                               [co_by_id[t]['text_f1'] for t in common] +
                               [p1_by_id[t]['text_f1'] for t in common])
    text_en = statistics.mean([kv_by_id[t]['text_energy_j'] for t in common] +
                               [co_by_id[t]['text_energy_j'] for t in common] +
                               [p1_by_id[t]['text_energy_j'] for t in common])
    print(f"{'Text MAS':<18} {text_la:>12.3f} {text_f1:>10.3f} {text_en:>14.0f} {'baseline':>18}")

    # Single (same averaging)
    single_la = statistics.mean([kv_by_id[t]['single_loose_accuracy'] for t in common] +
                                 [co_by_id[t]['single_loose_accuracy'] for t in common] +
                                 [p1_by_id[t]['single_loose_accuracy'] for t in common])
    single_f1 = statistics.mean([kv_by_id[t]['single_f1'] for t in common] +
                                 [co_by_id[t]['single_f1'] for t in common] +
                                 [p1_by_id[t]['single_f1'] for t in common])
    single_en = statistics.mean([kv_by_id[t]['single_energy_j'] for t in common] +
                                 [co_by_id[t]['single_energy_j'] for t in common] +
                                 [p1_by_id[t]['single_energy_j'] for t in common])
    single_vs_text = (single_en - text_en) / text_en * 100
    print(f"{'Single':<18} {single_la:>12.3f} {single_f1:>10.3f} {single_en:>14.0f} {single_vs_text:>17.1f}%")

    # Each latent mode
    for label, rows_by_id in [
        ('kv_share', kv_by_id),
        ('cocunut (LCoT-synth)', co_by_id),
        ('phase1_latent (LCoT-all)', p1_by_id),
    ]:
        las = [rows_by_id[t]['latent_loose_accuracy'] for t in common]
        f1s = [rows_by_id[t]['latent_f1'] for t in common]
        ens = [rows_by_id[t]['latent_energy_j'] for t in common]
        la = statistics.mean(las)
        f1 = statistics.mean(f1s)
        en = statistics.mean(ens)
        # Use THIS run's text baseline, not the averaged one
        this_text_en = statistics.mean([rows_by_id[t]['text_energy_j'] for t in common])
        vs_text = (en - this_text_en) / this_text_en * 100
        print(f"{label:<18} {la:>12.3f} {f1:>10.3f} {en:>14.0f} {vs_text:>17.1f}%")

    # Per-task winner counts for loose_acc and F1
    print(f"\nPer-task winners (strict ties broken by order below):")
    labels = ['text', 'kv', 'co', 'p1']
    la_wins = {l: 0 for l in labels}
    f1_wins = {l: 0 for l in labels}
    for tid in common:
        # Use each run's own text baseline for comparison
        scores_la = {
            'text': p1_by_id[tid]['text_loose_accuracy'],  # any of the 3 text runs
            'kv': kv_by_id[tid]['latent_loose_accuracy'],
            'co': co_by_id[tid]['latent_loose_accuracy'],
            'p1': p1_by_id[tid]['latent_loose_accuracy'],
        }
        scores_f1 = {
            'text': p1_by_id[tid]['text_f1'],
            'kv': kv_by_id[tid]['latent_f1'],
            'co': co_by_id[tid]['latent_f1'],
            'p1': p1_by_id[tid]['latent_f1'],
        }
        la_wins[max(scores_la, key=scores_la.get)] += 1
        f1_wins[max(scores_f1, key=scores_f1.get)] += 1
    print(f"  loose_acc wins: text={la_wins['text']}  kv={la_wins['kv']}  co={la_wins['co']}  p1={la_wins['p1']}  (n={len(common)})")
    print(f"  F1 wins:        text={f1_wins['text']}  kv={f1_wins['kv']}  co={f1_wins['co']}  p1={f1_wins['p1']}  (n={len(common)})")

    # Energy savings at matched accuracy (tasks where latent mode is within
    # 0.05 loose_acc of text MAS)
    print(f"\nEnergy savings on tasks where latent matches text within 0.05 loose_acc:")
    for label, rows_by_id in [
        ('kv_share', kv_by_id),
        ('cocunut', co_by_id),
        ('phase1_latent', p1_by_id),
    ]:
        matched = [t for t in common
                   if abs(rows_by_id[t]['latent_loose_accuracy']
                          - rows_by_id[t]['text_loose_accuracy']) <= 0.05]
        if matched:
            text_e = statistics.mean([rows_by_id[t]['text_energy_j'] for t in matched])
            lat_e = statistics.mean([rows_by_id[t]['latent_energy_j'] for t in matched])
            pct = (lat_e - text_e) / text_e * 100
            print(f"  {label:<18} matched n={len(matched):>3}  "
                  f"text_E={text_e:>7.0f}J  latent_E={lat_e:>7.0f}J  delta={pct:>+6.1f}%")
        else:
            print(f"  {label:<18} no tasks matched within 0.05 tolerance")


if __name__ == "__main__":
    main()
