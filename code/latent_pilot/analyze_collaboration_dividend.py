"""Compute per-task collaboration dividend (CD) across main-study data.

CD(task) = (A_decent - A_indep) / max(E_decent - E_indep, eps)

Reads paired Independent and Decentralized jsonls from the main-study
result directories, pairs by task_id, and computes CD per task. Reports
distribution + quantiles + per-benchmark breakdowns.

Expected jsonl fields per row:
    task_id, topology, loose_accuracy, gpu_dynamic_energy_joules

Input: a list of result directories. By default, the main study's
benchmark-specific directories (a5000_qampari_v4, a5000_fanoutqa_v4, etc.).

Usage:
    python analyze_collaboration_dividend.py
    python analyze_collaboration_dividend.py --results-root mas-energy/results
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_config_from_filename(path):
    """Extract (model, topology, params) from main-study jsonl filename:
    <Model>_<topology>_<params>.jsonl   e.g. Qwen_Qwen3.5-9B_decentralized_k20.jsonl
    """
    stem = path.stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p in ("sas", "independent", "centralized", "decentralized", "hybrid"):
            model = "_".join(parts[:i])
            topo = p
            extra = "_".join(parts[i + 1:])
            return model, topo, extra
    return None, None, None


def infer_benchmark_from_dir(path):
    """Best-effort inference from parent directory name."""
    name = path.parent.name
    for bench in ("qampari", "fanoutqa", "browsecomp", "workbench", "swebench"):
        if bench in name.lower():
            return bench
    return "unknown"


def load_jsonl(path):
    try:
        return [json.loads(l) for l in open(path)]
    except Exception as e:
        print(f"  error reading {path}: {e}")
        return []


def _get_accuracy(row):
    """Extract per-task accuracy: prefer loose_accuracy if present AND non-None,
    else fall back to `correct` (bool → 0/1). WorkBench specifically has
    `loose_accuracy` key present but None, with only `correct` populated."""
    la = row.get("loose_accuracy")
    if la is not None:
        return float(la)
    c = row.get("correct")
    if c is not None:
        return 1.0 if c else 0.0
    return 0.0


def bucket_by_task(rows):
    """Many rows per task due to n_reps; average energy and accuracy per task."""
    by_id = defaultdict(list)
    for r in rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        by_id[tid].append(r)
    result = {}
    for tid, rs in by_id.items():
        accs = [_get_accuracy(r) for r in rs]
        energies = [float(r.get("gpu_dynamic_energy_joules", 0) or 0) for r in rs]
        result[tid] = {
            "loose_accuracy": statistics.mean(accs) if accs else 0.0,
            "gpu_dynamic_energy_joules": statistics.mean(energies) if energies else 0.0,
            "n_reps": len(rs),
        }
    return result


def compute_cd(indep, decent, eps=100.0):
    """CD = (A_decent - A_indep) / max(E_decent - E_indep, eps)
    where eps (in joules) avoids divide-by-near-zero when decent ≈ indep energy.
    """
    common = sorted(set(indep.keys()) & set(decent.keys()))
    out = []
    for tid in common:
        a_i = indep[tid]["loose_accuracy"]
        a_d = decent[tid]["loose_accuracy"]
        e_i = indep[tid]["gpu_dynamic_energy_joules"]
        e_d = decent[tid]["gpu_dynamic_energy_joules"]
        d_e = max(e_d - e_i, eps)
        cd = (a_d - a_i) / d_e
        out.append({
            "task_id": tid,
            "A_indep": a_i, "A_decent": a_d,
            "E_indep": e_i, "E_decent": e_d,
            "delta_A": a_d - a_i, "delta_E": e_d - e_i,
            "CD": cd,
        })
    return out


def _bootstrap_ci(values, n_iters=1000, seed=42, conf=0.95):
    """Bootstrap CI for the mean of `values`."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        return (None, None)
    means = []
    for _ in range(n_iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((1 - conf) / 2 * n_iters)]
    hi = means[int((1 + conf) / 2 * n_iters) - 1]
    return (lo, hi)


def _mcnemar_test(cd_rows):
    """McNemar-style paired comparison on binary win (ΔA > 0) vs loss (ΔA < 0).

    Returns (n_decent_wins, n_indep_wins, n_ties, mcnemar_p).
    p-value via two-sided exact binomial test on the discordant pairs.
    """
    wins = sum(1 for r in cd_rows if r["delta_A"] > 0)
    losses = sum(1 for r in cd_rows if r["delta_A"] < 0)
    ties = sum(1 for r in cd_rows if r["delta_A"] == 0)
    discordant = wins + losses
    if discordant == 0:
        return wins, losses, ties, 1.0
    # Two-sided exact binomial p-value: P(min >= min(wins, losses) under H0: p=0.5)
    import math
    k = min(wins, losses)
    n = discordant
    p_value = 0.0
    for i in range(k + 1):
        p_value += math.comb(n, i) * (0.5 ** n)
    p_value = min(1.0, 2 * p_value)  # two-sided
    return wins, losses, ties, p_value


def summarize(cd_rows, label):
    if not cd_rows:
        print(f"  {label}: no paired tasks")
        return
    cds = [r["CD"] for r in cd_rows]
    dAs = [r["delta_A"] for r in cd_rows]
    dEs = [r["delta_E"] for r in cd_rows]
    n = len(cds)

    # Bin classification
    never_helps = sum(1 for r in cd_rows if r["delta_A"] <= 0)
    neutral = sum(1 for r in cd_rows if 0 < r["delta_A"] <= 0.05)
    helps = sum(1 for r in cd_rows if r["delta_A"] > 0.05)

    # Statistical tests
    lo_dA, hi_dA = _bootstrap_ci(dAs)
    decent_wins, indep_wins, ties, mcnemar_p = _mcnemar_test(cd_rows)

    print(f"\n  {label}  (n={n} paired tasks)")
    print(f"    ΔA (decent - indep):  mean={statistics.mean(dAs):+.3f}  "
          f"median={statistics.median(dAs):+.3f}  "
          f"95% CI=[{lo_dA:+.3f}, {hi_dA:+.3f}]" if lo_dA is not None else "")
    print(f"    ΔE (joules):          mean={statistics.mean(dEs):+.0f}  "
          f"median={statistics.median(dEs):+.0f}  ")
    print(f"    CD (ΔA per kJ):       mean={statistics.mean(cds)*1000:+.2f}  "
          f"median={statistics.median(cds)*1000:+.2f}")
    print(f"    Paired win/loss/tie:  Decent_wins={decent_wins}  Indep_wins={indep_wins}  "
          f"ties={ties}")
    print(f"    McNemar two-sided p:  {mcnemar_p:.4f}  "
          f"{'(Decent sig. different)' if mcnemar_p < 0.05 else '(no sig. difference)'}")
    print(f"    Task bins:")
    print(f"      collab HURTS/no-op (ΔA ≤ 0):        {never_helps:>3d} ({never_helps/n:.0%})")
    print(f"      collab marginal (0 < ΔA ≤ 0.05):    {neutral:>3d} ({neutral/n:.0%})")
    print(f"      collab HELPS (ΔA > 0.05):           {helps:>3d} ({helps/n:.0%})")


def find_pairs(results_root):
    """Find all (indep, decent) filename pairs under result dirs."""
    root = Path(results_root)
    pairs = []  # (indep_path, decent_path, model, benchmark, extra_key)
    by_key = defaultdict(dict)
    for path in root.rglob("*.jsonl"):
        model, topo, extra = parse_config_from_filename(path)
        if topo not in ("independent", "decentralized"):
            continue
        bench = infer_benchmark_from_dir(path)
        key = (bench, model, extra)
        by_key[key][topo] = path
    for key, topos in by_key.items():
        if "independent" in topos and "decentralized" in topos:
            bench, model, extra = key
            pairs.append((topos["independent"], topos["decentralized"], model, bench, extra))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str,
                    default="mas-energy/results",
                    help="Root directory containing main-study result subdirs.")
    args = ap.parse_args()

    pairs = find_pairs(args.results_root)
    if not pairs:
        print(f"No (independent, decentralized) pairs found under {args.results_root}.")
        print("Expected main-study jsonls matching *_independent_*.jsonl + *_decentralized_*.jsonl")
        return

    print(f"Found {len(pairs)} (indep, decent) pairs.")
    # Group by benchmark for summary
    all_rows = []
    by_bench = defaultdict(list)
    for indep_path, decent_path, model, bench, extra in pairs:
        indep_rows = load_jsonl(indep_path)
        decent_rows = load_jsonl(decent_path)
        if not indep_rows or not decent_rows:
            continue
        indep_tasks = bucket_by_task(indep_rows)
        decent_tasks = bucket_by_task(decent_rows)
        cd_rows = compute_cd(indep_tasks, decent_tasks)
        label = f"{bench} / {model} / {extra}"
        summarize(cd_rows, label)
        for r in cd_rows:
            r["benchmark"] = bench
            r["model"] = model
            r["config_extra"] = extra
        all_rows.extend(cd_rows)
        by_bench[bench].extend(cd_rows)

    # Overall
    if all_rows:
        print(f"\n{'='*70}\nAggregate across all configs (n={len(all_rows)})\n{'='*70}")
        summarize(all_rows, "ALL")
        # Per-benchmark aggregate
        print(f"\nPer-benchmark aggregate:")
        for bench in sorted(by_bench.keys()):
            summarize(by_bench[bench], bench)

    # Enrich with task-text features (question length, word count, comma count)
    # by joining back to source jsonls. These enable richer classifier features.
    for r in all_rows:
        # Try to re-load the source row for its question text
        # (finding source across a5000_* dirs is expensive; we embed what's
        # already in the main-study jsonl — task_id alone usually encodes
        # benchmark structure enough for the feature)
        r["task_id_features"] = _task_id_features(r.get("task_id", ""))

    # Save a jsonl for downstream classifier training
    out_path = Path("mas-energy/results/latent_pilot/collaboration_dividend.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(all_rows)} per-task CD rows to {out_path}")


def _task_id_features(task_id):
    """Extract cheap features from task_id strings. Benchmark-specific.

    QAMPARI: "123__category__split" — category carries signal
    FanOutQA: hex ID — no info
    WorkBench: "app_NN" — app name carries signal
    BrowseComp: "0xNN" — no info
    SWE-bench: "repo__issue-NNN" — repo carries signal
    """
    import re
    tid = str(task_id)
    features = {
        "id_length": len(tid),
        "n_underscores": tid.count("_"),
        "has_double_underscore": "__" in tid,
    }
    # WorkBench app category (e.g., "calendar", "email", "analytics")
    m = re.match(r"^([a-z_]+?)_\d+$", tid)
    if m:
        features["task_category"] = m.group(1)
    # QAMPARI category (after __)
    m = re.match(r"^\d+__([a-z_]+)__\w+$", tid)
    if m:
        features["qampari_category"] = m.group(1)
    return features


if __name__ == "__main__":
    main()
