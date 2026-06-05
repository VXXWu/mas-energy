#!/bin/bash
#SBATCH --job-name=a5000-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5000_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5000_pilot_%j.err

# A5000 feasibility pilot: BrowseComp+ Decentralized k=5 (5 tasks).
# Stress test: longest context topology on most context-heavy benchmark.
#
# Memory budget (Qwen3.5-9B BF16 on A5000 24GB):
#   Model weights: ~18GB (includes GDN layer params)
#   KV cache: ~32 KB/token (only 25% attention layers need KV; 75% DeltaNet)
# Stress test results (job 15084595):
#   Model weights: 17.62 GB, Mamba/SSM state: 1.39 GB, total: 19.0 GB
#   At mem_fraction=0.88: KV allocated for 49,594 tokens (32.2 KB/tok)
#   131K and 98K context requested but actual max usable = ~49K tokens
#   Covers P95 of A6000 calls (35K). Misses P99 (81K) and max (127K).
#   Set context-length=98304 (SGLang will allocate what fits, ~49K usable).

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000

echo "=== A5000 Pilot ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 98304 \
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

echo "=== BrowseComp+ Decentralized k=5 (5 tasks) ==="
python run_experiments.py \
    --model toy \
    --benchmarks browsecomp_plus \
    --topologies decentralized \
    --max-react-steps 5 \
    --n-tasks 5 --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/a5000_pilot \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
|| echo "WARNING: Decentralized failed"

echo "=== Pilot complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
