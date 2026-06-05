#!/bin/bash
#SBATCH --job-name=kvshare-m0-qwen35
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_m0_qwen35_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_m0_qwen35_%j.err

# Attribution validation: does the Qwen3-8B finding that m=0 equals m=5
# transfer to Qwen3.5-9B? On Qwen3-8B, m=0 achieved -48.9% energy at
# 0.225 loose_acc, matching m=5's 0.221 / -47.3%. This confirms the
# latent loopback contributes nothing; savings come from sequential KV
# continuation (topology) alone.
#
# Qwen3.5 kv_share at m=5 achieved -50.8% energy but 0.218 loose_acc
# (down from 0.309 text). If m=0 matches (similar loose_acc, lower
# energy), attribution holds: savings are topological on hybrid too.
# If m=0 diverges from m=5, hybrid needs different treatment.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== kv_share m=0 attribution test on Qwen3.5-9B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3.5-9B \
    --data-dir "$QAMPARI_DIR" \
    --latent-mode kv_share

echo "=== Done: $(date) ==="
