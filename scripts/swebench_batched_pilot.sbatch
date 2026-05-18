#!/bin/bash
#SBATCH --job-name=swe-batch
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/swe_batch_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/swe_batch_%j.err

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SWEBENCH_REPOS=/atlas2/u/$USER/mas_project/swebench_repos

PORT=30000
N_TASKS=5
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/swebench_batched_pilot
WORKBENCH_PATH=/atlas2/u/$USER/mas_project/WorkBench

echo "=== SWE-bench Batched Pilot ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
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
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: SGLang failed to start"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

echo "=== SAS k=50 ==="
python run_experiments.py \
    --model toy \
    --benchmarks swebench_batched \
    --topologies sas \
    --max-react-steps 50 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH \
|| echo "WARNING: SAS failed"

echo "=== Centralized k=50 R=1 ==="
python run_experiments.py \
    --model toy \
    --benchmarks swebench_batched \
    --topologies centralized \
    --max-react-steps 50 \
    --rounds 1 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH \
|| echo "WARNING: Centralized failed"

echo "=== Decentralized k=50 ==="
python run_experiments.py \
    --model toy \
    --benchmarks swebench_batched \
    --topologies decentralized \
    --max-react-steps 50 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH \
|| echo "WARNING: Decentralized failed"

echo "=== Pilot complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
