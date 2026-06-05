#!/bin/bash
#SBATCH --job-name=structural-qwen35
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/structural_qwen35_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/structural_qwen35_%j.err

# Structural comparison on Qwen3.5-9B (hybrid Gated DeltaNet) — architecture
# transfer test for the three latent modes. Qwen3-8B already established:
#   kv_share:       -47% energy at +25% loose_acc (clean win)
#   cocunut:        -45% energy at matched loose_acc (close win)
#   phase1_latent:  +18% energy at -13% loose_acc (structural negative)
#
# This run tests whether those findings transfer to hybrid attention.
# kv_share likely struggles on hybrid (DeltaNet state doesn't share via
# KV cache); cocunut likely struggles differently (zero-shot loopback
# fails on DeltaNet per our earlier 14% agreement baseline).

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== Structural latent comparison on Qwen3.5-9B (QAMPARI, hybrid attention) ==="
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
        --model-name Qwen/Qwen3.5-9B \
        --data-dir "$QAMPARI_DIR" \
        --latent-mode $MODE
done

echo ""
echo "=== Done: $(date) ==="
