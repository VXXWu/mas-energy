"""Final ablation table: kv_share m_latent sweep across architectures
and benchmarks.

Tests whether LatentMAS's mechanism claim (latent thoughts replace text
decode at lower energy while preserving reasoning) holds:

  - QAMPARI (parallelizable, ceiling-limited): expected m=0 ≈ m>0 because
    debate itself doesn't help on this task. Attribution ambiguous here.
  - FanOutQA (reasoning-heavy, multi-hop): if m>0 > m=0 AND text MAS > single,
    loopback is load-bearing and LatentMAS's claim is validated. If m>0 ≈ m=0
    even when text MAS > single, loopback fails even where debate helps.

Architectures: Qwen3-8B (softmax) and Qwen3.5-9B (hybrid Gated DeltaNet).
"""
import json
import statistics
from pathlib import Path

RESULTS = Path("mas-energy/results/latent_pilot")


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return [json.loads(l) for l in open(path)]


def agg(rows, field):
    if not rows:
        return None
    return statistics.mean([r[field] for r in rows])


def report(model_tag, benchmark, m_values):
    header = f"{model_tag} / {benchmark}"
    print(f"\n{header}")
    print(f"  {'m':>3} {'Single':>8} {'TextMAS':>8} {'kv_share':>10} {'Lat F1':>8} "
          f"{'Text E (J)':>11} {'Lat E (J)':>11} {'vs Text':>10}")
    print(f"  {'-'*3} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*11} {'-'*11} {'-'*10}")
    for m in m_values:
        fname = f"eval_{model_tag}_{benchmark}_agentic_k10_m{m}.jsonl"
        rows = load(fname)
        if rows is None:
            print(f"  {m:>3}  (PENDING — {fname})")
            continue
        single_la = agg(rows, 'single_loose_accuracy')
        text_la = agg(rows, 'text_loose_accuracy')
        lat_la = agg(rows, 'latent_loose_accuracy')
        lat_f1 = agg(rows, 'latent_f1')
        text_e = agg(rows, 'text_energy_j')
        lat_e = agg(rows, 'latent_energy_j')
        pct = (lat_e - text_e) / text_e * 100
        print(f"  {m:>3} {single_la:>8.3f} {text_la:>8.3f} {lat_la:>10.3f} {lat_f1:>8.3f} "
              f"{text_e:>11.0f} {lat_e:>11.0f} {pct:>+9.1f}%")


def main():
    print("=" * 92)
    print("kv_share m_latent ablation — does latent loopback validate LatentMAS's claim?")
    print("=" * 92)

    print("\n[QAMPARI — ceiling-limited: debate doesn't help on this task anyway]")
    report("Qwen_Qwen3-8B", "qampari", [0, 3, 5, 10])
    report("Qwen_Qwen35-9B", "qampari", [0, 5])

    print("\n[FanOutQA — reasoning-heavy: proper test of loopback mechanism]")
    report("Qwen_Qwen3-8B", "fanoutqa", [0, 5, 10])

    print()
    print("Interpretation key:")
    print("  QAMPARI: m=0 ≈ m>0 only shows ceiling. Says nothing about mechanism on")
    print("    tasks where debate helps.")
    print("  FanOutQA: PROPER test.")
    print("    If text MAS > single AND kv_share m>0 > kv_share m=0:")
    print("      → loopback IS load-bearing; LatentMAS's claim validated.")
    print("    If text MAS > single BUT kv_share m>0 ≈ kv_share m=0:")
    print("      → loopback fails even where debate matters; LatentMAS invalidated.")
    print("    If text MAS ≈ single on FanOutQA:")
    print("      → wrong benchmark; need another reasoning-heavy task.")


if __name__ == "__main__":
    main()
