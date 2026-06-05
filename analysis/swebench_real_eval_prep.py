"""Extract per-cell SWE-bench patches into predictions.json files for the
official SWE-bench harness.

Each JSONL file (one per cell) has records with a `patch` field containing
the full git diff the agent produced. This script:
  1. Reads every SWE-bench JSONL
  2. Extracts (instance_id, patch) per task
  3. Writes a per-cell predictions JSON in the format the harness expects:
       [{"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}, ...]

Output: one predictions_<cell>.json per JSONL in `swebench_real_eval_inputs/`.
"""
import json, glob, os, re
from pathlib import Path

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
OUT_DIR = os.path.join(MAS_ENERGY_ROOT, "results/swebench_real_eval_inputs")
CRE = re.compile(r"(Qwen_Qwen3\.5-9B_[a-z]+_k\d+(?:_R\d+)?(?:_M\d+)?)\.jsonl$")


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    n_cells = 0
    n_patches_total = 0
    n_missing = 0

    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.search(os.path.basename(f))
        if not m: continue
        cell_name = m.group(1)
        predictions = []
        seen_ids = set()
        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            tid = d.get("task_id") or d.get("instance_id")
            if not tid or tid in seen_ids: continue
            seen_ids.add(tid)
            # The eval data dict from evaluate_swebench is merged into the
            # record; check several plausible locations
            patch = (d.get("patch")
                     or d.get("answer")
                     or d.get("model_patch")
                     or "")
            if not patch:
                n_missing += 1
                continue
            predictions.append({
                "instance_id": tid,
                "model_name_or_path": f"mas-energy/{cell_name}",
                "model_patch": patch,
            })
            n_patches_total += 1
        if predictions:
            out = f"{OUT_DIR}/predictions_{cell_name}.json"
            with open(out, "w") as fh:
                json.dump(predictions, fh, indent=2)
            n_cells += 1

    print(f"Wrote predictions for {n_cells} cells")
    print(f"Total patches: {n_patches_total}")
    print(f"Missing patches: {n_missing}")
    print(f"Output dir: {OUT_DIR}")
    print()
    print("Next: rsync OUT_DIR to cluster, then run the harness — see")
    print("  analysis/swebench_real_eval_prep.py docstring for the cluster command.")


if __name__ == "__main__":
    main()
