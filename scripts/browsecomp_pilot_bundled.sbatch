#!/bin/bash
#SBATCH --job-name=browsecomp-all
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/browsecomp_bundled_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/browsecomp_bundled_%j.err

# Bundled: runs ALL topologies sequentially in one GPU allocation.
# Extra memory (64G) for BrowseComp 100K corpus + BM25 index.

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000
N_TASKS=${N_TASKS:-50}

echo "=== BrowseComp-Plus BUNDLED ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.90 \
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
    if [ $i -eq 120 ]; then
        echo "ERROR: SGLang failed to start"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

echo "Ensuring BrowseComp-Plus datasets are cached..."
python -c "
from datasets import load_dataset
load_dataset('Tevatron/browsecomp-plus', split='test')
load_dataset('Tevatron/browsecomp-plus-corpus', split='train')
print('Datasets ready.')
" || { echo "ERROR: Failed to load datasets"; kill $SGLANG_PID 2>/dev/null; exit 1; }

cd /atlas2/u/$USER/mas_project/mas-energy/code

CONFIGS=(
    "sas          2 5 10  "
    "independent  2 5 10  "
    "centralized  2 5 10  "
    "centralized  2 5 10  --rounds 1"
    "decentralized 2 5 10  "
)

for config in "${CONFIGS[@]}"; do
    read -r TOPO K1 K2 K3 EXTRA <<< "$config"
    K_VALUES="$K1 $K2"
    [ -n "$K3" ] && K_VALUES="$K1 $K2 $K3"

    echo ""
    echo "=== Running: ${TOPO} k=${K_VALUES} ${EXTRA} ($(date)) ==="

    python run_experiments.py \
        --model toy \
        --benchmarks browsecomp_plus \
        --topologies $TOPO \
        --max-react-steps $K_VALUES \
        $EXTRA \
        --n-tasks $N_TASKS --n-reps 1 \
        --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/browsecomp_pilot \
        --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: ${TOPO} ${EXTRA} failed, continuing..."

    echo "=== Done: ${TOPO} ${EXTRA} ($(date)) ==="
done

echo ""
echo "=== ALL BrowseComp configs complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
