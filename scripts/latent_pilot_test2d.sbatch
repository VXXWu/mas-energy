#!/bin/bash
#SBATCH --job-name=latent-test2d
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2d_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2d_%j.err

# Test 2d: KV injection on Qwen3-8B (pure softmax attention, 36 layers).
# All layers participate in injection — no DeltaNet bottleneck.
# LatentMAS validated this approach on this model family.
#
# Qwen3-8B BF16 is ~16GB; fits on A5000 24GB.
# Expected runtime: ~15-20 min (100 pairs × 3 forward passes).
#
# IMPORTANT: Qwen3-8B must be cached in HF_HOME. If not, the first run
# downloads ~16GB. Verify: ls $HF_HOME/hub/models--Qwen--Qwen3-8B/

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

echo "=== Latent pilot: Test 2d (KV injection on Qwen3-8B) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""
echo "--- Full cache (k=0, no compression, LatentMAS protocol) ---"
python -m latent_pilot.test2d_qwen3_injection --n-pairs 100 --k 0

echo ""
echo "--- Compressed cache (k=64) ---"
python -m latent_pilot.test2d_qwen3_injection --n-pairs 100 --k 64

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/test2d_qwen3_summary_k64.json"
