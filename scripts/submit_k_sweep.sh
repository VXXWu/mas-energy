#!/bin/bash
# Submit the k-sweep for Phase B latent encoder training.
# Each k value gets its own sbatch (separate output dir per k).
#
# Usage (run on cluster, or via ssh wrapper):
#   ssh vincewu8@scdt.stanford.edu "bash /atlas2/u/vincewu8/mas_project/mas-energy/scripts/submit_k_sweep.sh"
#
# Override K_VALUES, EPOCHS via env vars:
#   K_VALUES="4 16 32" EPOCHS=3 bash submit_k_sweep.sh

set -e

K_VALUES=${K_VALUES:-"4 8 16 32"}
EPOCHS=${EPOCHS:-2}
SBATCH_FILE=/atlas2/u/$USER/mas_project/mas-energy/scripts/train_latent_encoder.sbatch

if [ ! -f "$SBATCH_FILE" ]; then
    echo "ERROR: $SBATCH_FILE not found. Sync code first:"
    echo "  bash mas-energy/scripts/sync_to_cluster.sh vincewu8"
    exit 1
fi

echo "Submitting k-sweep: K_VALUES=[$K_VALUES] EPOCHS=$EPOCHS"
echo "(each job ~15-20 min on A6000 at full corpus)"
echo ""

JOB_IDS=()
for K in $K_VALUES; do
    OUT=$(sbatch --export=ALL,EPOCHS=$EPOCHS,K=$K "$SBATCH_FILE")
    JOB_ID=$(echo "$OUT" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    echo "  K=$K  -> job $JOB_ID"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
echo "Monitor: squeue -u \$USER --format='%.18i %.20j %.8T %.10M %R'"
echo "Output:  results/latent_encoder/k{K}_epochs${EPOCHS}/encoder_epoch{N}.pt"
