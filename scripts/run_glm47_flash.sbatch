#!/bin/bash
#SBATCH --job-name=mas-glm47
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/glm47_flash_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/glm47_flash_%j.err

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

BENCHMARK=${1:-workbench}
echo "=== GLM-4.7-Flash (MoE) — ${BENCHMARK} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# GLM-4.7-Flash: no official GPTQ-Int4, uses FP8 or BF16
# FP8 (~30GB) fits on 48GB with less KV cache headroom
python -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --port 30000 \
    --tp 1 \
    --attention-backend triton \
    --mem-fraction-static 0.80 \
    --context-length 131072 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!

for i in $(seq 1 120); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code
python run_experiments.py \
    --model glm47-flash \
    --benchmarks ${BENCHMARK} \
    --topologies sas independent centralized decentralized hybrid \
    --n-reps 5 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/main \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
