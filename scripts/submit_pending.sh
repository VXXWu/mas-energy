#!/bin/bash
# Submit/resubmit the pending fill jobs in one shot.
#
# Run on cluster login (sc.stanford.edu). Stages submissions with a short
# delay between sbatches so SLURM accounting registers them in order.
#
# All sbatches now have the SGLang health-check window bumped to 20 min
# (was 6 min — the prior failures on 15316653/15323604/15316650 were all
# health-check timeouts during first-time cuda graph capture, NOT env or
# OOM issues).
#
# Submits:
#   1. qampari mscan (full array 0-5; resume logic skips completed cells)
#   2. swebench mscan arrays 4,5 only (Decent M=4, M=5 — failed in
#      job 15323604; arrays 0,3 already complete; arrays 1,2 still running)
#   3. BCB ablation array 0 only (Decentralized — failed in job 15316650;
#      arrays 1,2 already complete at orphan path; array 3 already running)

set -e

SCRIPTS=/atlas2/u/$USER/mas_project/mas-energy/scripts

echo "=== submitting qampari mscan (all 6 cells) ==="
sbatch "$SCRIPTS/a5000_mscan_qampari.sbatch"
sleep 2

echo "=== submitting swebench mscan arrays 4,5 (Decent M=4, M=5) ==="
sbatch --array=4,5 "$SCRIPTS/a5000_mscan_swebench.sbatch"
sleep 2

echo "=== submitting BCB ablation array 0 (Decentralized) ==="
sbatch --array=0 "$SCRIPTS/run_bigcodebench_ablation.sbatch"

echo
echo "=== queue snapshot ==="
squeue -u $USER --format='%.18i %.9P %.10j %.8u %.2t %.10M %.6D %R'
