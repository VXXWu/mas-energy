#!/bin/bash
#SBATCH --job-name=latent-hybrid
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_hybrid_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_hybrid_%j.err

# Hybrid evaluation: text-debate vs latent-debate on QAMPARI with pre-retrieval.
# Three conditions per task: single agent, text-debate (3 agents R=2),
# latent-debate (3 agents, m=40 latent steps in Round 2).
# All conditions see identical BM25-retrieved passages.
#
# Expected runtime: ~6-10 hrs. Each task runs 3 conditions:
#   - Single agent: 1 generate call
#   - Text debate: 3 Round1 + 3 Round2 = 6 generate calls
#   - Latent debate: 3 Round1 generate + 2×40 latent steps + 1 decode = 3+81
#   Total: ~90 forward-pass-equivalents per task × 50 tasks

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

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
    echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
    echo "Download: https://aggreg-qa.s3.amazonaws.com/qampari.zip"
    exit 1
fi

echo "=== Hybrid evaluation: text vs latent debate on QAMPARI ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.eval_hybrid_qampari \
    --n-tasks 50 --m 40 --n-agents 3 \
    --qampari-data-dir "$QAMPARI_DIR"

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/eval_hybrid_summary.json"
