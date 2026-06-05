#!/bin/bash
#SBATCH --job-name=mas-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/pilot_%j.err

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

echo "=== MAS Energy Pilot ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Start SGLang server (toy model: Qwen3.5-9B)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port 30000 \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 131072 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
--host 0.0.0.0 &
SGLANG_PID=$!

# Wait for server to be ready
echo "Waiting for SGLang server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:30000/health > /dev/null 2>&1; then
    echo "ERROR: SGLang failed to start"
    kill $SGLANG_PID 2>/dev/null
    exit 1
fi

# Run pilot test
cd /atlas2/u/$USER/mas_project/mas-energy/code
python pilot_test.py \
    --model toy \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
    --n-tasks 5

echo "=== Pilot complete ==="

# Cleanup
kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
