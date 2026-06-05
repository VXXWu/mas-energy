#!/bin/bash
#SBATCH --job-name=latent-test1
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test1_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_test1_%j.err

# Test 1: layer-attribution ablation. The gate test for Option 1 viability.
#
# Expected wall time ~1 hr for 200 prompts. Three forward passes per prompt:
#   baseline, softmax-attn-ablated, deltanet-ablated.
#
# Decision rule (see latent_pilot/README.md):
#   attn disagreement > 0.50  → PROCEED to Test 2
#   attn disagreement < 0.20  → KILL: switch to Option 2
#   in between                → MARGINAL: proceed cautiously

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

# Verify model-structure report from Step 0 exists
REPORT=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/model_structure.json
if [ ! -f "$REPORT" ]; then
    echo "ERROR: $REPORT not found. Run latent_pilot_inspect.sbatch first."
    exit 1
fi

echo "=== Latent pilot: Test 1 (layer ablation) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.test1_layer_ablation --n-prompts 200 --last-k-positions 50

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/test1_summary.json"
