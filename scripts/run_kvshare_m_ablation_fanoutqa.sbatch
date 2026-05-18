#!/bin/bash
#SBATCH --job-name=kvshare-fanoutqa
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_fanoutqa_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/kvshare_fanoutqa_%j.err

# PROPER test of LatentMAS's mechanism claim: kv_share m_latent ablation on
# FanOutQA (reasoning-heavy, multi-hop decomposition, benefits from debate).
#
# QAMPARI ablation was ceiling-limited: debate itself doesn't help on
# QAMPARI (parallelizable, no overlap to resolve), so loopback can't help
# either. Any latent-debate mechanism looks equivalent to no-debate there.
#
# FanOutQA is a reasoning task where cross-agent debate should add
# accuracy. Grid:
#   m=0  → no loopback (tests if text debate > no-debate here)
#   m=5  → small loopback (LatentMAS config)
#   m=10 → medium loopback
# If m>0 > m=0 on accuracy AND text MAS > m=0: loopback is load-bearing.
# That validates LatentMAS's mechanism claim on reasoning-heavy tasks.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== kv_share m_latent ablation on FanOutQA (Qwen3-8B) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

cd /atlas2/u/$USER/mas_project/mas-energy/code

for M in 0 5 10; do
    echo ""
    echo "======================================================"
    echo "m_latent = $M"
    echo "======================================================"
    python -m latent_pilot.agentic_latentmas \
        --benchmark fanoutqa \
        --n-tasks 30 --m-latent $M --max-react-steps 10 --n-agents 3 \
        --model-name Qwen/Qwen3-8B \
        --latent-mode kv_share
done

echo ""
echo "=== Done: $(date) ==="
