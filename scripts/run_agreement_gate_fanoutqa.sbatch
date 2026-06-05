#!/bin/bash
#SBATCH --job-name=agreegate-fanoutqa
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/agreegate_fanoutqa_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/agreegate_fanoutqa_%j.err

# Agreement-gate experiment on FanOutQA (reasoning-heavy, prose answers).
# Same paired design as the QAMPARI pilot but on a benchmark where:
#   - Debate is more likely to add value (reasoning task, not just breadth)
#   - Answers are prose, so the agreement metric uses content-word fallback
#
# Two paired runs on same 30 FanOutQA tasks:
#   Run 1 (baseline):  Decentralized with Phase 2 debate (always run).
#   Run 2 (gated):     Skip Phase 2 when agents' Phase 1 token-set Jaccard
#                      >= threshold. Logs gate decisions.
#
# Expected outcomes (if gate is well-calibrated):
#   - Tasks where agents converged (high Jaccard): skip debate, save energy,
#     accuracy unchanged (debate wasn't adding value).
#   - Tasks where agents diverged (low Jaccard): run debate, accuracy may
#     improve (Decentralized > Independent per main study).
#
# On FanOutQA specifically: debate is more likely to help, so gate should
# trigger on FEWER tasks than on QAMPARI. Still worth measuring.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== Agreement-gate vs baseline on Qwen3-8B (FanOutQA) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

echo ""
echo "======================================================"
echo "Run 1: baseline (no agreement gate)"
echo "======================================================"
python -m latent_pilot.agentic_latentmas \
    --benchmark fanoutqa \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --latent-mode kv_share

echo ""
echo "======================================================"
echo "Run 2: with agreement gate (skip Phase 2 if Jaccard >= 0.5)"
echo "======================================================"
python -m latent_pilot.agentic_latentmas \
    --benchmark fanoutqa \
    --n-tasks 30 --m-latent 0 --max-react-steps 10 --n-agents 3 \
    --model-name Qwen/Qwen3-8B \
    --latent-mode kv_share \
    --enable-agreement-skip --agreement-threshold 0.5

echo ""
echo "=== Done: $(date) ==="
