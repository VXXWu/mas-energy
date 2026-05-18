#!/bin/bash
#SBATCH --job-name=kvshare-m-abl
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_m_abl_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_m_abl_%j.err

# m_latent ablation for the WORKING kv_share mode on Qwen3-8B QAMPARI.
# Baseline result (m=5): -47.2% energy, +25% loose_acc vs text MAS.
#
# Grid: m_latent ∈ {0, 3, 5, 10}
#   m=0  — pure KV continuation, no latent steps. Tests if loopback is
#          load-bearing or if just passing KV between agents is enough.
#   m=3  — smaller latent budget. Tests if the mechanism works with less.
#   m=5  — current working config (control / reproducibility check).
#   m=10 — larger latent budget. Tests if more thinking helps or starts to
#          break (like phase1_latent at m=20 did).

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== kv_share m_latent ablation on Qwen3-8B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

for M in 0 3 5 10; do
    echo ""
    echo "======================================================"
    echo "m_latent = $M"
    echo "======================================================"
    python -m latent_pilot.agentic_latentmas \
        --benchmark qampari \
        --n-tasks 30 --m-latent $M --max-react-steps 10 --n-agents 3 \
        --model-name Qwen/Qwen3-8B \
        --data-dir "$QAMPARI_DIR" \
        --latent-mode kv_share
done

echo ""
echo "=== Done: $(date) ==="
