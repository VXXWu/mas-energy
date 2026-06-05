"""Bucket SWE-bench predictions so the harness can evaluate all unique patches.

The harness dedupes by instance_id (run_evaluation.py:515:
  predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
), so a single predictions file can only carry one patch per instance.

To evaluate ALL unique patches, we split the deduped predictions into
buckets where each bucket has at most one patch per instance_id, and run
the harness once per bucket. Total work = unique-patch count; just spread
across N runs where N = max(patches_per_instance).

Inputs (from swebench_real_eval_dedupe.py):
  - predictions_pareto.json     (556 patches)
  - predictions_dedup.json      (3546 patches)

Outputs:
  - predictions_pareto_bucket_<i>.json    for i in 0..(max-1)
  - predictions_dedup_bucket_<i>.json     for i in 0..(max-1)
"""
import json, os
from collections import defaultdict

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


OUT = os.path.join(MAS_ENERGY_ROOT, "results/swebench_real_eval_inputs")


def bucket(predictions, prefix):
    """Split list of predictions into per-instance buckets and write JSON files."""
    by_instance = defaultdict(list)
    for p in predictions:
        by_instance[p["instance_id"]].append(p)
    # Sort within each instance for determinism (by model_name_or_path = patch hash)
    for tid in by_instance:
        by_instance[tid].sort(key=lambda r: r["model_name_or_path"])
    max_bucket = max(len(v) for v in by_instance.values())
    buckets = [[] for _ in range(max_bucket)]
    for tid, preds in by_instance.items():
        for i, p in enumerate(preds):
            buckets[i].append(p)
    # Write
    for i, b in enumerate(buckets):
        path = f"{OUT}/{prefix}_bucket_{i}.json"
        with open(path, "w") as fh:
            json.dump(b, fh, indent=2)
    return max_bucket, [len(b) for b in buckets]


def main():
    for fname, prefix in [("predictions_pareto.json", "predictions_pareto"),
                          ("predictions_dedup.json",  "predictions_dedup")]:
        preds = json.load(open(f"{OUT}/{fname}"))
        n_buckets, sizes = bucket(preds, prefix)
        print(f"{fname}:")
        print(f"  total patches:   {len(preds)}")
        print(f"  buckets created: {n_buckets}  (sizes: {sizes})")
        print(f"  total spread:    {sum(sizes)} == {len(preds)} ? {sum(sizes)==len(preds)}")


if __name__ == "__main__":
    main()
