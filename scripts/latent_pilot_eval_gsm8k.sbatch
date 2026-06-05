#!/bin/bash
#SBATCH --job-name=latent-gsm8k
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_gsm8k_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_gsm8k_%j.err

# LatentMAS evaluation on GSM8K — the benchmark LatentMAS was validated on.
# Short numeric answers minimize sensitivity to extended-generation corruption.
# 100 tasks, 3 conditions (single, text debate, latent debate).
#
# Expected runtime: ~4-8 hrs.

set -e

source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONPATH=/atlas2/u/$USER/mas_project/mas-energy/code:$PYTHONPATH
export PYTHONUNBUFFERED=1

python -c "import torch, torch._dynamo, transformers, sympy" 2>/dev/null || {
    echo "ERROR: mas-latent env is broken."
    echo "  bash mas-energy/scripts/build_latent_env.sh"
    exit 1
}

echo "=== LatentMAS GSM8K evaluation ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.eval_latentmas_gsm8k --n-tasks 100 --m 40 --n-agents 3

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/eval_gsm8k_summary.json"
