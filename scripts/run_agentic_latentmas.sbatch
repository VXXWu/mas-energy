#!/bin/bash
#SBATCH --job-name=agentic-latentmas
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/agentic_latentmas_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/agentic_latentmas_%j.err

# Agentic LatentMAS on QAMPARI: text ReAct for tools, latent for communication.
# Three conditions: single agent, text decentralized, latent decentralized.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
    echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
    exit 1
fi

echo "=== Agentic LatentMAS on QAMPARI ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 40 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --data-dir "$QAMPARI_DIR"

echo "=== Done: $(date) ==="
