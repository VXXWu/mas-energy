#!/bin/bash
#SBATCH --job-name=latent-earlystop
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_earlystop_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_earlystop_%j.err

# Phase 3: Learned early stopping for latent thoughts.
# Runs m=80 with checkpoints every 5 steps on 30 pairs, trains a stop
# classifier, evaluates energy reduction.
#
# Expected runtime: ~3-4 hrs (30 pairs × 80 latent steps with 16
# checkpoint evaluations each = ~30 × 96 forward passes + training).
#
# REQUIRES: Phase 1 (latentmas_baseline) to have validated the mechanism.

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

echo "=== Latent pilot: Phase 3 — early stopping ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.latentmas_early_stop --n-pairs 30 --max-m 80 --checkpoint-interval 5

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/early_stop_summary.json"
