#!/bin/bash
#SBATCH --job-name=cocunut-qwen35
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/cocunut_qwen35_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/cocunut_qwen35_%j.err

# CoCoNuT (architecture-agnostic hidden-state loopback) synthesis on Qwen3.5-9B.
# KV-sharing LatentMAS fails on Qwen3.5's hybrid Gated DeltaNet attention
# (14% agreement vs 72-86% on Qwen3-8B). CoCoNuT operates at the top of the
# residual stack (no KV surgery), so it should work on any architecture.
#
# Phase 1: parallel text ReAct (identical to text baseline)
# Phase 2: compact debate transcript
# Phase 3: CoCoNuT synthesis (m_latent hidden-state loopback + decode)
#
# Compares single / text-MAS / cocunut-MAS on QAMPARI loose_accuracy + energy.

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

echo "=== CoCoNuT synthesis on Qwen3.5-9B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 40 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3.5-9B \
    --data-dir "$QAMPARI_DIR" \
    --latent-mode cocunut

echo "=== Done: $(date) ==="
