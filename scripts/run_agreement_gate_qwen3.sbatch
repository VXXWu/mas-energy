#!/bin/bash
#SBATCH --job-name=agreegate-qwen3
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/agreegate_qwen3_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/agreegate_qwen3_%j.err

# Agreement-gate experiment on Qwen3-8B QAMPARI. Two paired runs on same
# 30 tasks:
#   1. No gating (baseline): standard Decentralized with Phase 2 debate.
#   2. Gated: skip Phase 2 when agents' Phase 1 Jaccard agreement > 0.5.
# Both also log per-agent tool calls (for cross-agent duplicate-rate analysis).
#
# Outcomes examined:
#   (a) Does gating preserve accuracy? Compare text_loose_acc gated vs baseline.
#   (b) How much energy saved by skipping? Compare text_energy_j.
#   (c) What fraction of tasks got skipped? Count text_skipped_debate.
#   (d) Tool-call duplicate rate (works from either run): informs cache ROI.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== Agreement-gate vs baseline on Qwen3-8B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""
echo "======================================================"
echo "Run 1: baseline (no agreement gate)"
echo "======================================================"
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --data-dir "$QAMPARI_DIR" \
    --latent-mode kv_share

echo ""
echo "======================================================"
echo "Run 2: with agreement gate (skip Phase 2 if Jaccard >= 0.5)"
echo "======================================================"
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --data-dir "$QAMPARI_DIR" \
    --latent-mode kv_share \
    --enable-agreement-skip --agreement-threshold 0.5

echo ""
echo "=== Done: $(date) ==="
