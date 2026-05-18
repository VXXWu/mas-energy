#!/bin/bash
#SBATCH --job-name=mas-q35a3b
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/qwen35b_a3b_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/qwen35b_a3b_%j.err

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

BENCHMARK=${1:-workbench}
echo "=== Qwen3.5-35B-A3B (MoE) — ${BENCHMARK} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --port 30000 \
    --tp 1 \
    --quantization gptq \
    --mem-fraction-static 0.80 \
    --context-length 131072 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
--host 0.0.0.0 &
SGLANG_PID=$!

for i in $(seq 1 120); do
    curl -s http://localhost:30000/health > /dev/null 2>&1 && break
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code
python run_experiments.py \
    --model qwen35b-a3b \
    --benchmarks ${BENCHMARK} \
    --topologies sas independent centralized decentralized hybrid \
    --n-reps 5 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/main \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
