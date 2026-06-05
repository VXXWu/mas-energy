#!/bin/bash
#SBATCH --job-name=latentmas-qwen35
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_qwen35_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_qwen35_%j.err

# Critical test: does LatentMAS mechanism work on Qwen3.5-9B (hybrid attention)?
# The earlier test2/2b/2c failures used the WRONG approach (KV injection with
# DeltaNet reset, or hidden state re-injection). The LatentMAS approach (W_e +
# full cache sharing INCLUDING DeltaNet state) was never tested on this model.
#
# If this works, latent communication runs on the same model as the main
# energy paper — no model-difference confound.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONPATH=/atlas2/u/$USER/mas_project/mas-energy/code:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== LatentMAS mechanism test on Qwen3.5-9B (hybrid attention) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

# Test the LatentMAS mechanism (W_e alignment + latent generation + full cache)
# on Qwen3.5-9B. Uses the same latentmas_baseline.py that got 72-86% on Qwen3-8B.
# Key difference: Qwen3.5-9B has hybrid Gated DeltaNet attention.
# The full cache includes DeltaNet recurrent state — no reset.
python -m latent_pilot.latentmas_baseline \
    --model-name Qwen/Qwen3.5-9B \
    --n-pairs 50 --m-values 10 20 40 80 --seed 42

echo "=== Done: $(date) ==="
