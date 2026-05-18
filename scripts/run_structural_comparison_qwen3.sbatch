#!/bin/bash
#SBATCH --job-name=structural-qwen3
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/structural_qwen3_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/structural_qwen3_%j.err

# Structural comparison of latent-communication decompositions on Qwen3-8B
# QAMPARI. All three latent modes run on the same 30 tasks, paired against
# text MAS baseline. Answers: "where in the topology should we apply
# latent to maximize energy savings at matched accuracy?"
#
# Modes compared:
#   kv_share       — Phase 2 only (LatentMAS-style seq KV accumulation)
#   cocunut        — Phase 3 only (hidden-state loopback at synthesis)
#   phase1_latent  — All phases latent (Phase 1 + Phase 2 + Phase 3)
#
# Each mode's run produces its own jsonl (model_tag fix ensures no
# collision). Analysis is on paired per-task energy + F1 + loose_acc.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== Structural latent comparison on Qwen3-8B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

for MODE in kv_share cocunut phase1_latent; do
    echo ""
    echo "======================================================"
    echo "MODE: $MODE"
    echo "======================================================"
    python -m latent_pilot.agentic_latentmas \
        --benchmark qampari \
        --n-tasks 30 --m-latent 5 --max-react-steps 10 --n-agents 3 \
        --model-name Qwen/Qwen3-8B \
        --data-dir "$QAMPARI_DIR" \
        --latent-mode $MODE
done

echo ""
echo "=== Done: $(date) ==="
