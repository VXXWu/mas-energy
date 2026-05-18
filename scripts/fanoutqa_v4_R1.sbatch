#!/bin/bash
#SBATCH --job-name=fanoutqa-v4-R1
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/fanoutqa_v4_R1_%A.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/fanoutqa_v4_R1_%A.err

# Single-round centralized ablation (R=1).
# Tests the "deployed MAS" pattern: decompose → execute once → synthesize.
# No iterative review loop. Compare against R=5 (default) centralized.

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30005
N_TASKS=${N_TASKS:-50}

echo "=== FanOutQA v4: centralized R=1 ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "N_TASKS: $N_TASKS"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "WARNING: stale process on port $PORT, killing..."
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

start_sglang() {
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
            return 0
        fi
        sleep 2
    done
    echo "ERROR: SGLang failed to start"
    kill $SGLANG_PID 2>/dev/null
    return 1
}

start_sglang || exit 1

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks fanoutqa \
    --topologies centralized \
    --max-react-steps 2 5 10 \
    --rounds 1 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/fanoutqa_v4 \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: centralized R=1 failed"

echo "=== FanOutQA v4 centralized R=1 complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
