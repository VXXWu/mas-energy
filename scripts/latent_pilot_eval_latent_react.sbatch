#!/bin/bash
#SBATCH --job-name=latent-react
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_react_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_react_%j.err

# Latent ReAct on QAMPARI: interleaved latent thinking + text tool calling.
# 30 tasks, single-agent ReAct vs 3-agent Latent-ReAct MAS.
# m=15 latent think steps per react round, max 5 react steps per agent.

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
    exit 1
fi

echo "=== Latent ReAct QAMPARI evaluation ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:  $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
echo "--- max-react-steps=5 ---"
python -m latent_pilot.eval_latent_react_qampari \
    --n-tasks 30 --m-think 15 --max-react-steps 5 --n-agents 3 \
    --qampari-data-dir "$QAMPARI_DIR"

echo "--- max-react-steps=10 ---"
python -m latent_pilot.eval_latent_react_qampari \
    --n-tasks 30 --m-think 15 --max-react-steps 10 --n-agents 3 \
    --qampari-data-dir "$QAMPARI_DIR"

echo "=== Done: $(date) ==="
