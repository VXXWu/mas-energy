#!/bin/bash
#SBATCH --job-name=llm-judge-qwen3
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/llm_judge_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/llm_judge_%j.err

# LLM-as-judge feature computation for the orchestrator classifier.
# Rates each QAMPARI question's reasoning complexity on a 1-5 scale
# using Qwen3-8B (same model as main study). Adds
# `llm_difficulty_rating` to the enriched CD jsonl.
#
# Runtime: ~500 questions × 1 short forward pass each ≈ 2-3 minutes on A5000.
# 2h wall-time budget is generous headroom.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== LLM-as-judge difficulty ratings on QAMPARI questions ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

IN=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/collaboration_dividend_final.jsonl
OUT=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/cd_with_llm_judge.jsonl

# If the enriched CD jsonl doesn't exist yet on cluster, synthesize it locally
# from whatever is present. (Expected that user syncs before running this.)
if [ ! -f "$IN" ]; then
    echo "ERROR: $IN does not exist. Sync your latent_pilot/ results dir first."
    exit 1
fi

python -m latent_pilot.judge_difficulty \
    --in "$IN" \
    --out "$OUT" \
    --model Qwen/Qwen3-8B \
    --device cuda

echo "=== Done: $(date) ==="
echo "Output at $OUT"
