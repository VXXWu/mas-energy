"""Dedupe SWE-bench predictions across cells and prioritize Pareto-relevant ones.

Many cells share (instance_id, patch) pairs because `seed=42` in load_tasks
samples the same tasks deterministically. Identical patches at the same
instance_id need only ONE harness evaluation regardless of how many cells
generated them.

This script:
  1. Reads every JSONL, extracts (instance_id, patch_content_hash) per record
  2. Deduplicates into a single global predictions list
  3. Tags Pareto-relevant cells (configurable) so you can run those first
  4. Writes ONE predictions_dedup.json + a mapping table

Outputs:
  - predictions_dedup.json : flat list of unique (instance_id, patch) tuples
  - dedup_mapping.json     : {(cell, instance_id): patch_hash} for re-merging
  - predictions_pareto.json: subset covering only Pareto-frontier cells (~15 cells)
"""
import json, glob, os, re, hashlib
from pathlib import Path

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
OUT_DIR = os.path.join(MAS_ENERGY_ROOT, "results/swebench_real_eval_inputs")
CRE = re.compile(r"(Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?)\.jsonl$")

# Pareto-frontier cells from the most recent analysis (see plot_pareto_canonical.py)
PARETO_CELLS = {
    ("centralized",    15,  2, 3),
    ("decentralized",  15,  2, 2),
    ("decentralized",   7,  5, 2),
    ("centralized",    30,  2, 2),
    ("centralized",    20,  3, 2),
    ("decentralized",  36,  1, 2),
    ("decentralized",  15,  4, 2),
    ("decentralized",  20,  2, 4),
    ("decentralized",  10, 10, 3),
    # Add high-accuracy "near-Pareto" cells worth confirming (within ~2pp of frontier)
    ("decentralized",  10,  5, 3),  # the canonical n=131 cell
    ("centralized",    50,  2, 3),  # 100% old champion
    ("centralized",    36,  3, 5),  # 100% LHS cell
}


def hash_patch(p):
    return hashlib.sha1(p.strip().encode()).hexdigest()[:12]


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # patch_index: {(instance_id, patch_hash): patch_content}
    patch_index = {}
    # mapping: {(cell, instance_id): patch_hash}
    mapping = {}
    # Track which cells touched which patches
    pareto_patches = set()

    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.search(os.path.basename(f))
        if not m: continue
        cell_name, topo, k, R, M = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5)
        R = int(R) if R else (5 if topo == "centralized" else 2)
        M = int(M) if M else 3
        is_pareto = (topo, k, R, M) in PARETO_CELLS

        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            tid = d.get("task_id") or d.get("instance_id")
            patch = d.get("patch") or d.get("answer") or d.get("model_patch") or ""
            if not tid or not patch: continue
            ph = hash_patch(patch)
            patch_index[(tid, ph)] = patch
            mapping[f"{cell_name}|{tid}"] = ph
            if is_pareto:
                pareto_patches.add((tid, ph))

    # Write deduped predictions
    all_preds = [
        {"instance_id": tid,
         "model_name_or_path": f"mas-energy/{ph}",
         "model_patch": patch}
        for (tid, ph), patch in patch_index.items()
    ]
    dedup_path = f"{OUT_DIR}/predictions_dedup.json"
    with open(dedup_path, "w") as fh:
        json.dump(all_preds, fh, indent=2)

    # Write Pareto-only subset
    pareto_preds = [p for p in all_preds
                    if (p["instance_id"], p["model_name_or_path"].split("/")[-1])
                       in pareto_patches]
    pareto_path = f"{OUT_DIR}/predictions_pareto.json"
    with open(pareto_path, "w") as fh:
        json.dump(pareto_preds, fh, indent=2)

    # Write mapping for re-merge step
    map_path = f"{OUT_DIR}/dedup_mapping.json"
    with open(map_path, "w") as fh:
        json.dump(mapping, fh, indent=2)

    # Summary
    total_records = sum(1 for _ in patch_index)  # equals len(patch_index)
    unique_instances = len(set(tid for tid, _ in patch_index.keys()))

    print(f"Total cell-record entries across all JSONLs:   {len(mapping)}")
    print(f"Unique (instance_id, patch_hash) pairs:        {len(patch_index)}")
    print(f"Unique instance IDs (across all patches):      {unique_instances}")
    print(f"Pareto-only subset:                            {len(pareto_preds)}")
    print()
    print(f"Speedup from dedup: {len(mapping) / max(len(patch_index), 1):.1f}× fewer evals")
    print()
    print(f"Time estimate at ~5 min/eval × 4 Docker workers:")
    print(f"  Pareto-only:  ~{len(pareto_preds) * 5 / 4 / 60:.1f} hours")
    print(f"  Full dedup:   ~{len(patch_index) * 5 / 4 / 60:.1f} hours")
    print()
    print(f"Outputs in: {OUT_DIR}")
    print(f"  predictions_pareto.json   (start here — quick validation)")
    print(f"  predictions_dedup.json    (complete eval)")
    print(f"  dedup_mapping.json        (for re-merge step)")


if __name__ == "__main__":
    main()
