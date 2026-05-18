"""Analyze the predictability-axis test.

Pairs tasks within each topology by task_index across variants P and U,
then compares topologies within each variant. Reports the falsifiable test:
  H1: P-Decent ΔAcc(vs IS) ≈ 0
  H2: U-Decent ΔAcc(vs IS) ≥ 0.10
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def variant_of(row):
    """Pull the variant tag from row's task_id (format pred-P-NNNN / pred-U-NNNN)."""
    tid = row.get("task_id", "")
    if "pred-P-" in tid: return "P"
    if "pred-U-" in tid: return "U"
    # Fallback: nested task dict if present
    return (row.get("task") or {}).get("variant")


def task_index(row):
    tid = row.get("task_id", "")
    if "pred-P-" in tid or "pred-U-" in tid:
        return int(tid.split("-")[-1])
    return None


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05):
    if len(deltas) < 5: return (None, None)
    rng = random.Random(42)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randint(0, n-1)] for _ in range(n))/n for _ in range(n_boot))
    return means[int(alpha/2 * n_boot)], means[int((1-alpha/2) * n_boot)]


def acc(r):
    if r.get("loose_accuracy") is not None:
        return float(r["loose_accuracy"])
    return 1.0 if r.get("correct") else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="mas-energy/results/predictability_axis")
    ap.add_argument("--k", default="k5")
    args = ap.parse_args()

    topologies = ["decentralized", "independent_share", "decentralized_empty"]
    rows_by_topo = {}
    for topo in topologies:
        path = Path(args.root) / f"Qwen_Qwen3.5-9B_{topo}_{args.k}.jsonl"
        rows_by_topo[topo] = load(path)
        if not rows_by_topo[topo]:
            print(f"  (no data for {topo} at {path})")

    # Index by (variant, task_index) within each topology
    indexed = {topo: {} for topo in topologies}
    for topo, rs in rows_by_topo.items():
        for r in rs:
            v = variant_of(r)
            ti = task_index(r)
            if v is not None and ti is not None:
                indexed[topo][(v, ti)] = r

    # Per-variant per-topology mean F1
    print(f"{'topology':<24} {'variant':<8} {'n':>5} {'F1_mean':>9} {'exact_mean':>11} {'E_J':>8}")
    print("-" * 70)
    for topo in topologies:
        for v in ("P", "U"):
            rs = [r for (vv, _ti), r in indexed[topo].items() if vv == v]
            if not rs: continue
            f1m = mean(acc(r) for r in rs)
            exm = mean(1.0 if r.get("correct") else 0.0 for r in rs)
            em  = mean((r.get("gpu_dynamic_energy_joules") or 0) for r in rs)
            print(f"{topo:<24} {v:<8} {len(rs):>5} {f1m:>9.3f} {exm:>11.3f} {em:>8.0f}")

    # Paired comparisons: within each variant, Decent vs IS (and vs empty)
    print("\n=== Paired ΔF1: Decent vs IS, within variant ===")
    for v in ("P", "U"):
        decent_by_ti = {ti: r for (vv, ti), r in indexed["decentralized"].items() if vv == v}
        is_by_ti     = {ti: r for (vv, ti), r in indexed["independent_share"].items() if vv == v}
        common = sorted(set(decent_by_ti) & set(is_by_ti))
        if not common: continue
        deltas = [acc(decent_by_ti[ti]) - acc(is_by_ti[ti]) for ti in common]
        lo, hi = bootstrap_ci(deltas)
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
        print(f"  Variant {v}: n={len(common)}  ΔF1(Decent-IS) = {mean(deltas):+.3f}  CI95 {ci}")

    # Decision rule for H1 / H2
    p_delta = None; u_delta = None
    decent_by_ti_P = {ti: r for (vv, ti), r in indexed["decentralized"].items() if vv == "P"}
    is_by_ti_P     = {ti: r for (vv, ti), r in indexed["independent_share"].items() if vv == "P"}
    common_P = sorted(set(decent_by_ti_P) & set(is_by_ti_P))
    if common_P:
        p_delta = mean(acc(decent_by_ti_P[ti]) - acc(is_by_ti_P[ti]) for ti in common_P)
    decent_by_ti_U = {ti: r for (vv, ti), r in indexed["decentralized"].items() if vv == "U"}
    is_by_ti_U     = {ti: r for (vv, ti), r in indexed["independent_share"].items() if vv == "U"}
    common_U = sorted(set(decent_by_ti_U) & set(is_by_ti_U))
    if common_U:
        u_delta = mean(acc(decent_by_ti_U[ti]) - acc(is_by_ti_U[ti]) for ti in common_U)

    print("\n=== Hypothesis test ===")
    if p_delta is None or u_delta is None:
        print("  (not enough data yet)")
    else:
        h1_ok = abs(p_delta) <= 0.05
        h2_ok = u_delta >= 0.10
        print(f"  H1 (P-Decent ≈ P-IS, |ΔF1| ≤ 0.05):    ΔF1 = {p_delta:+.3f}   {'PASS' if h1_ok else 'FAIL'}")
        print(f"  H2 (U-Decent − U-IS ≥ 0.10):           ΔF1 = {u_delta:+.3f}   {'PASS' if h2_ok else 'FAIL'}")
        if h1_ok and h2_ok:
            print("  → Predictability axis CONFIRMED as the mechanism for iteration-required.")
        elif not h2_ok:
            print("  → Predictability axis FALSIFIED — rounds aren't load-bearing even when tool returns are unpredictable.")
        else:
            print("  → Mixed: rounds appear to matter even on predictable tool returns; mechanism story needs revision.")

    # If empty cell exists, the deeper mechanism question
    if rows_by_topo.get("decentralized_empty"):
        print("\n=== Empty-channel test (if Decent shows U-effect, is it from text or from extra tool turns?) ===")
        for v in ("P", "U"):
            decent_by_ti = {ti: r for (vv, ti), r in indexed["decentralized"].items() if vv == v}
            empty_by_ti  = {ti: r for (vv, ti), r in indexed["decentralized_empty"].items() if vv == v}
            is_by_ti     = {ti: r for (vv, ti), r in indexed["independent_share"].items() if vv == v}
            common = sorted(set(decent_by_ti) & set(empty_by_ti) & set(is_by_ti))
            if not common: continue
            d_full = mean(acc(decent_by_ti[ti]) for ti in common)
            d_emp  = mean(acc(empty_by_ti[ti]) for ti in common)
            d_is   = mean(acc(is_by_ti[ti]) for ti in common)
            print(f"  Variant {v}: n={len(common)}  Decent={d_full:.3f}  empty={d_emp:.3f}  IS={d_is:.3f}")


if __name__ == "__main__":
    main()
