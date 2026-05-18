#!/bin/bash
#SBATCH --job-name=latent-test2b
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2b_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2b_%j.err

# Test 2b: Hidden-state soft-embedding injection (Option 2 pilot).
#
# Unlike Test 2 (KV injection into 8 attn layers only), this injects Agent A's
# last-layer hidden states as soft embeddings into Agent B's input. All 32
# layers -- including the 24 GatedDeltaNet layers -- process the signal.
#
# Runs three compression methods at K=64 to compare:
#   last_k: last 64 hidden states from A's output
#   stride: uniformly sampled 64 hidden states
#   pool:   mean-pooled into 64 chunks
#
# No dependency on Test 1 or Test 2 results -- this is an independent approach.

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

echo "=== Latent pilot: Test 2b (hidden-state injection) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""
echo "--- last_k compression (K=64) ---"
python -m latent_pilot.test2b_hidden_state --n-pairs 100 --k 64 --method last_k

echo ""
echo "--- stride compression (K=64) ---"
python -m latent_pilot.test2b_hidden_state --n-pairs 100 --k 64 --method stride

echo ""
echo "--- pool compression (K=64) ---"
python -m latent_pilot.test2b_hidden_state --n-pairs 100 --k 64 --method pool

echo ""
echo "=== Done: $(date) ==="
echo "Results in: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/test2b_summary_k64_*.json"
