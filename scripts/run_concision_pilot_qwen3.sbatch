#!/bin/bash
#SBATCH --job-name=concision-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/concision_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/concision_pilot_%j.err

# Concision fine-tuning pilot on Qwen3-8B.
#
# Stages:
#   1. Data gen: run SAS on 500 QAMPARI train, extract (verbose, concise)
#      pairs via regex — programmatic concision distillation, no GPT-4 judge.
#   2. LoRA SFT training: rank 8, 2 epochs. ~2-3 hours.
#   3. Eval: compare trained adapter vs baseline on decode-token count and
#      F1/loose_acc — does concision tuning get the same F1 wins as
#      grammar-constrained decoding at LOWER decode cost?

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
DATA_OUT=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/concise_pairs.jsonl
LORA_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/concision_lora

echo "=== Concision pilot on Qwen3-8B ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""; echo "[1/3] Generating (verbose, concise) SFT pairs..."
python -m latent_pilot.concision_data_gen \
    --n-tasks 500 \
    --split train \
    --out-path $DATA_OUT \
    --data-dir $QAMPARI_DIR \
    --model-name Qwen/Qwen3-8B

echo ""; echo "[2/3] Training concision LoRA (rank=8, 2 epochs)..."
pip show peft >/dev/null 2>&1 || pip install 'peft==0.13.2' --no-deps
python -m latent_pilot.concision_train \
    --data $DATA_OUT \
    --model Qwen/Qwen3-8B \
    --out-dir $LORA_DIR \
    --epochs 2 \
    --lora-rank 8

echo ""; echo "[3/3] Evaluating concision adapter on 30 QAMPARI eval tasks..."
python -m latent_pilot.agentic_latentmas \
    --benchmark qampari \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --data-dir $QAMPARI_DIR \
    --latent-mode kv_share \
    --lora-adapter $LORA_DIR

echo ""
echo "=== Done: $(date) ==="
