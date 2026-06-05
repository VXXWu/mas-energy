#!/bin/bash
#SBATCH --job-name=latent-fanoutqa
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_fanoutqa_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_fanoutqa_%j.err

# Agentic LatentMAS on FanOutQA.
# Main experiment baselines: SAS k=10 = 0.577, Decentralized k=10 = 0.724
# Three conditions: single, text MAS, latent MAS with energy measurement.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

python -c "import torch, transformers, fanoutqa" 2>/dev/null || {
    echo "ERROR: mas-latent env missing deps."
    exit 1
}

echo "=== Agentic LatentMAS on FanOutQA ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code
python -m latent_pilot.agentic_latentmas \
    --benchmark fanoutqa \
    --n-tasks 30 --m-latent 40 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B

echo "=== Done: $(date) ==="
