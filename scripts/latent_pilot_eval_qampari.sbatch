#!/bin/bash
#SBATCH --job-name=latent-qampari
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_qampari_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_qampari_%j.err

# LatentMAS QAMPARI evaluation (closed-book, no tools).
# Runs sequential protocol: 2 agents do latent thoughts + 1 solver decodes.
# Compares to single-agent closed-book baseline.
# 50 tasks, m=40 latent steps.
#
# Expected runtime: ~2-4 hrs (50 tasks × ~3 min each for 3 agents with
# m=40 latent steps + text decode + single-agent baseline).

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

# Check QAMPARI data exists
QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
    echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
    echo "Download: https://aggreg-qa.s3.amazonaws.com/qampari.zip"
    exit 1
fi

echo "=== LatentMAS QAMPARI evaluation (closed-book) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.eval_latentmas_qampari \
    --n-tasks 50 --m 40 --n-agents 3 \
    --qampari-data-dir "$QAMPARI_DIR"

echo "=== Done: $(date) ==="
echo "Summary: /atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/eval_qampari_summary.json"
