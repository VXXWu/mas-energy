#!/bin/bash
#SBATCH --job-name=phase1-latent-msweep
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/phase1_latent_msweep_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/phase1_latent_msweep_%j.err

# m_latent hyperparameter sweep on Qwen3-8B QAMPARI.
# m=5 produced coherent outputs (-47% energy, F1 0.197 vs text 0.129).
# m=20 drifted into prompt-echo failure (F1 0.000).
# Sweep m ∈ {3, 5, 8, 10} at 15 tasks each to locate the stability cliff.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

echo "=== Phase-1-latent m-sweep on Qwen3-8B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

for M in 3 5 8 10; do
    echo ""
    echo "--- m_latent=$M ---"
    python -m latent_pilot.agentic_latentmas \
        --benchmark qampari \
        --n-tasks 15 --m-latent $M --max-react-steps 10 --n-agents 3 \
        --model-name Qwen/Qwen3-8B \
        --data-dir "$QAMPARI_DIR" \
        --latent-mode phase1_latent
done

echo ""
echo "=== Done: $(date) ==="
