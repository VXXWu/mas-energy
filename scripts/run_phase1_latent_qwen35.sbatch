#!/bin/bash
#SBATCH --job-name=phase1-latent-qwen35
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/phase1_latent_qwen35_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/phase1_latent_qwen35_%j.err

# Phase-1-latent parallel decentralized on Qwen3.5-9B (hybrid attention
# transfer test). Same code path as the Qwen3-8B baseline, just a different
# model. This is the architecture-transfer test: does raw hidden-state
# loopback (no W_e alignment) produce coherent continuations through
# Gated DeltaNet layers in Phase 1 ReAct?
#
# If F1 is comparable to text MAS: thesis transfers to the main-paper model.
# If F1 collapses: hybrid attention rejects hidden-state injection at both
# alignment (LatentMAS-style) and raw (CoCoNuT-style), and the latent-MAS
# approach is softmax-only.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
    echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
    exit 1
fi

echo "=== Phase-1-latent on Qwen3.5-9B (QAMPARI) — architecture transfer test ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 5 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3.5-9B \
    --data-dir "$QAMPARI_DIR" \
    --latent-mode phase1_latent

echo "=== Done: $(date) ==="
