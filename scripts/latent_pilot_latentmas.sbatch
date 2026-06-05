#!/bin/bash
#SBATCH --job-name=latentmas-base
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_base_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_base_%j.err

# LatentMAS baseline: latent thought generation on Qwen3-8B.
# Sweeps m ∈ {10, 20, 40, 80} latent steps on 50 pairs.
#
# Expected runtime: ~1-2 hrs. Each pair requires:
#   - 1 forward (A input) + m forward passes (latent thoughts) + 1 forward (B latent)
#   - 1 forward (B text baseline)
#   Total: m+3 forward passes per pair. At m=80: 83 passes × 50 pairs = 4150 passes.
#
# 4hr time limit to accommodate the full sweep.

set -e

source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONPATH=/atlas2/u/$USER/mas_project/mas-energy/code:$PYTHONPATH

python -c "import torch, torch._dynamo, transformers, sympy" 2>/dev/null || {
    echo "ERROR: mas-latent env is broken."
    echo "  bash mas-energy/scripts/build_latent_env.sh"
    exit 1
}

echo "=== LatentMAS baseline (Qwen3-8B) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.latentmas_baseline --n-pairs 50 --m-values 10 20 40 80

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/latentmas_summary.json"
