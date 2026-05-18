#!/bin/bash
#SBATCH --job-name=latent-inspect
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_inspect_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_inspect_%j.err

# Step 0 of latent-communication pilot: dump Qwen3.5-9B module structure.
# Runs in ~5 minutes, writes model_structure.json. Subsequent tests depend
# on its output for correct hook targeting.

set -e

source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONPATH=/atlas2/u/$USER/mas_project/mas-energy/code:$PYTHONPATH

# Preflight: verify required deps resolve before model load. sympy is an
# implicit torch._dynamo dep whose absence produces an 80-line unhelpful
# traceback from transformers' lazy importer.
python -c "import torch, torch._dynamo, transformers, sympy" 2>/dev/null || {
    echo "ERROR: mas-latent env is broken (likely cascading dep breakage)."
    echo "Use the project's rebuild script rather than fixing individual packages:"
    echo "  bash mas-energy/scripts/build_latent_env.sh"
    echo "Full diagnostic:"
    python -c "import torch, transformers, sympy"
    exit 1
}

echo "=== Latent pilot: model inspection ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.inspect_model

echo "=== Done: $(date) ==="
