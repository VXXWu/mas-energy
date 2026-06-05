"""Merge SWE-bench harness results back into JSONL records.

Pipeline:
  1. swebench_real_eval_dedupe.py writes predictions_pareto.json / dedup.json
     with model_name_or_path = "mas-energy/<patch_hash>", plus dedup_mapping.json
     mapping {cell|instance_id: patch_hash}.
  2. The harness writes per-instance reports to:
       <invocation_cwd>/logs/run_evaluation/<run_id>/mas-energy__<patch_hash>/<instance_id>/report.json
     plus a summary file: <invocation_cwd>/mas-energy__<patch_hash>.<run_id>.json
     (we only need the per-instance report.json files)
  3. This script walks all matching per-instance reports, builds
     {(patch_hash, instance_id): resolved}, then for each JSONL record:
       a. find patch_hash via dedup_mapping
       b. look up (patch_hash, instance_id) in harness dict
       c. if found, set d["real_correct"] = resolved
                      d["patch_presence_correct"] = (old `correct`)

Records whose patch was not evaluated are NOT modified.

Outputs:
  - JSONLs rewritten in-place (with `.pre_real_eval.bak` backup)
  - `analysis/swebench_real_vs_patch_presence.csv` — per-cell comparison
"""
import json, os, glob, re, csv, shutil
from pathlib import Path

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR    = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
INPUT_DIR    = os.path.join(MAS_ENERGY_ROOT, "results/swebench_real_eval_inputs")
# Harness output relative to the dir we invoke `swebench.harness.run_evaluation` from.
# Both the smoke test and the Pareto/full runs invoke from INPUT_DIR, so logs land there.
HARNESS_LOGS = f"{INPUT_DIR}/logs/run_evaluation"
CRE = re.compile(r"(Qwen_Qwen3\.5-9B_[a-z]+_k\d+(?:_R\d+)?(?:_M\d+)?)\.jsonl$")


def load_dedup_mapping():
    """Return {(cell_name, instance_id): patch_hash}."""
    p = f"{INPUT_DIR}/dedup_mapping.json"
    if not os.path.exists(p):
        print(f"ERROR: {p} missing — run swebench_real_eval_dedupe.py first.")
        return {}
    raw = json.load(open(p))
    return {tuple(k.split("|", 1)): v for k, v in raw.items()}


def load_harness_results(run_ids=None):
    """Return {(patch_hash, instance_id): resolved}.

    Walks every `logs/run_evaluation/<run_id>/mas-energy__<patch_hash>/<instance_id>/report.json`
    under HARNESS_LOGS. If `run_ids` is provided, restricts to that set.
    """
    out = {}
    if not os.path.isdir(HARNESS_LOGS):
        print(f"WARNING: {HARNESS_LOGS} does not exist — no harness output yet.")
        return out
    for run_dir in sorted(glob.glob(f"{HARNESS_LOGS}/*")):
        rid = os.path.basename(run_dir)
        if run_ids is not None and rid not in run_ids:
            continue
        if not os.path.isdir(run_dir): continue
        for model_dir in glob.glob(f"{run_dir}/mas-energy__*"):
            patch_hash = os.path.basename(model_dir).replace("mas-energy__", "", 1)
            for report in glob.glob(f"{model_dir}/*/report.json"):
                instance_id = os.path.basename(os.path.dirname(report))
                try:
                    d = json.load(open(report))
                    resolved = bool(d.get(instance_id, {}).get("resolved", False))
                    out[(patch_hash, instance_id)] = resolved
                except Exception as e:
                    print(f"  warning: failed to parse {report}: {e}")
    return out


def main(run_ids=None):
    mapping = load_dedup_mapping()
    if not mapping:
        return
    harness = load_harness_results(run_ids=run_ids)
    if not harness:
        print("No harness results found — exiting.")
        return
    print(f"Loaded {len(harness)} (patch_hash, instance_id) -> resolved results")
    print(f"Loaded {len(mapping)} (cell, instance_id) -> patch_hash mappings")

    summary = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.search(os.path.basename(f))
        if not m: continue
        cell_name = m.group(1)
        records = []
        patch_presence_correct = 0
        real_correct_count = 0
        real_eval_count = 0
        total = 0

        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"):
                records.append(d)
                continue
            tid = d.get("task_id") or d.get("instance_id")
            old_correct = bool(d.get("correct"))
            patch_hash = mapping.get((cell_name, tid))
            real_correct = harness.get((patch_hash, tid)) if patch_hash else None
            if real_correct is not None:
                d["real_correct"] = real_correct
                d["patch_presence_correct"] = old_correct
                real_correct_count += int(real_correct)
                real_eval_count += 1
            total += 1
            patch_presence_correct += int(old_correct)
            records.append(d)

        backup = f + ".pre_real_eval.bak"
        if not os.path.exists(backup):
            shutil.copy(f, backup)
        with open(f, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        if total:
            summary.append({
                "cell": cell_name,
                "n_total": total,
                "n_real_evaluated": real_eval_count,
                "patch_presence_rate_pct": round(100 * patch_presence_correct / total, 1),
                "real_pass_rate_pct": (round(100 * real_correct_count / real_eval_count, 1)
                                       if real_eval_count else None),
                "delta_pp": (round(100 * real_correct_count / real_eval_count
                                   - 100 * patch_presence_correct / total, 1)
                             if real_eval_count else None),
            })

    sum_path = os.path.join(PROJECT_ROOT, "analysis/swebench_real_vs_patch_presence.csv")
    with open(sum_path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "n_total", "n_real_evaluated",
                                            "patch_presence_rate_pct",
                                            "real_pass_rate_pct", "delta_pp"])
        w.writeheader()
        for s in sorted(summary, key=lambda r: -(r["real_pass_rate_pct"] or -1)):
            w.writerow(s)
    print(f"\nWrote {sum_path}")
    cells_with_real = sum(1 for s in summary if s["n_real_evaluated"])
    print(f"Cells touched by real eval: {cells_with_real} / {len(summary)}")


if __name__ == "__main__":
    main()
