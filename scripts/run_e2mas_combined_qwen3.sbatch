#!/bin/bash
#SBATCH --job-name=e2mas-combined
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/e2mas_combined_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/e2mas_combined_%j.err

# E2-MAS combined system test: three engineering optimizations stacked,
# measuring the incremental contribution of each on Qwen3-8B QAMPARI.
#
# Runs 4 configs on the same 30 tasks with seed control:
#   (1) baseline           — no optimizations
#   (2) +agree-skip        — skip Phase 2 when Phase 1 agents agree
#   (3) +tool-cache        — dedupe cross-agent tool queries per task
#   (4) +grammar           — stop-strings on synthesis to cut prose tails
#   (5) +all three         — the full E2-MAS system
#
# Ablation design: each flag isolates one mechanism's contribution. The
# "+all" run shows whether they compound or interfere.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
COMMON="--benchmark qampari --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 --model-name Qwen/Qwen3-8B --data-dir $QAMPARI_DIR --latent-mode kv_share"

echo "=== E2-MAS mechanism ablation on Qwen3-8B (QAMPARI) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""; echo "=== (1) baseline — no optimizations ==="
python -m latent_pilot.agentic_latentmas $COMMON

echo ""; echo "=== (2) agreement gate only ==="
python -m latent_pilot.agentic_latentmas $COMMON --enable-agreement-skip --agreement-threshold 0.5

echo ""; echo "=== (3) tool cache only ==="
python -m latent_pilot.agentic_latentmas $COMMON --enable-tool-cache

echo ""; echo "=== (4) grammar constraint only ==="
python -m latent_pilot.agentic_latentmas $COMMON --enable-grammar

echo ""; echo "=== (5) all three combined ==="
python -m latent_pilot.agentic_latentmas $COMMON \
    --enable-agreement-skip --agreement-threshold 0.5 \
    --enable-tool-cache --enable-grammar

echo ""
echo "=== Done: $(date) ==="
