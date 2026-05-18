#!/bin/bash
#SBATCH --job-name=latent-test2
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test2_%j.err

# Test 2: KV-injection functional equivalence at K=64 (generous K).
#
# REQUIRES: test1 verdict != "KILL" and take_last_k_cache() / forward_with_cache()
# filled in based on model_structure.json from Step 0.
#
# Expected wall time ~2 hr for 100 pairs. Each pair requires 3 forward passes:
#   A (capture KV), B_text (baseline), B_latent (with injected cache).

set -e

source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONPATH=/atlas2/u/$USER/mas_project/mas-energy/code:$PYTHONPATH

# Preflight: verify required deps resolve before model load.
python -c "import torch, torch._dynamo, transformers, sympy" 2>/dev/null || {
    echo "ERROR: mas-latent env is broken (likely cascading dep breakage)."
    echo "Use the project's rebuild script rather than fixing individual packages:"
    echo "  bash mas-energy/scripts/build_latent_env.sh"
    echo "Full diagnostic:"
    python -c "import torch, transformers, sympy"
    exit 1
}

# Sanity: Test 1 summary should exist and not have verdict=KILL
TEST1=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/test1_summary.json
if [ ! -f "$TEST1" ]; then
    echo "ERROR: Test 1 summary not found. Run latent_pilot_test1.sbatch first."
    exit 1
fi
if grep -q "\"verdict\":.*KILL" "$TEST1"; then
    echo "ERROR: Test 1 verdict is KILL. Option 1 abandoned. Do not run Test 2."
    echo "See: $TEST1"
    exit 1
fi

echo "=== Latent pilot: Test 2 (KV injection equivalence) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.test2_injection --n-pairs 100 --k 64

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/test2_summary_k64.json"
