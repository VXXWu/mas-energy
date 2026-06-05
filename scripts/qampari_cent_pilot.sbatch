#!/bin/bash
#SBATCH --job-name=qampari-cent
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/qampari_cent_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/qampari_cent_%j.err

# Centralized pilot with improved decomposition prompt.
# Runs centralized default + R=1 at k=5, N=30 tasks.

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000

echo "=== QAMPARI Centralized Pilot (improved decompose prompt) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "WARNING: stale process on port $PORT, killing..."
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 131072 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!

echo "Waiting for SGLang server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: SGLang failed to start"
    kill $SGLANG_PID 2>/dev/null
    exit 1
fi

cd /atlas2/u/$USER/mas_project/mas-energy/code

# Centralized default rounds, k=5
python run_experiments.py \
    --model toy \
    --benchmarks qampari \
    --topologies centralized \
    --max-react-steps 5 \
    --n-tasks 30 --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/qampari_v3_cent \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: centralized default failed"

# Centralized R=1, k=5
python run_experiments.py \
    --model toy \
    --benchmarks qampari \
    --topologies centralized \
    --max-react-steps 5 \
    --rounds 1 \
    --n-tasks 30 --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/qampari_v3_cent \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: centralized R=1 failed"

# Also run SAS k=5 as baseline (same 30 tasks)
python run_experiments.py \
    --model toy \
    --benchmarks qampari \
    --topologies sas \
    --max-react-steps 5 \
    --n-tasks 30 --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/qampari_v3_cent \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: sas failed"

echo "=== QAMPARI Centralized Pilot complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
