#!/bin/bash
#SBATCH --job-name=browsecomp-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4%3
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/browsecomp_pilot_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/browsecomp_pilot_%A_%a.err

# BrowseComp-Plus pilot: 830 fact-seeking questions over 100K doc corpus.
# Uses BM25 retrieval (Recall@5 = 1.2% -- expect low accuracy).
# Pilot validates pipeline: tool calling, energy measurement, evaluation.
# Dense retrieval (Qwen3-Embed-8B) needed for main runs.
#
# Extra memory (64G) for loading 100K corpus + BM25 index (~4-6GB).
# First run builds BM25 index (cached to pickle for subsequent runs).
#
# Array tasks:
#   0: sas           (k=2,5,10)
#   1: independent   (k=2,5,10)
#   2: centralized   (k=2,5,10)
#   3: centralized   (k=2,5,10, R=1)
#   4: decentralized (k=2,5,10)

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-50}

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;            K_VALUES="2 5 10"; ROUNDS_ARG="" ;;
    1) TOPO=independent;    K_VALUES="2 5 10"; ROUNDS_ARG="" ;;
    2) TOPO=centralized;    K_VALUES="2 5 10"; ROUNDS_ARG="" ;;
    3) TOPO=centralized;    K_VALUES="2 5 10"; ROUNDS_ARG="--rounds 1" ;;
    4) TOPO=decentralized;  K_VALUES="2 5 10"; ROUNDS_ARG="" ;;
esac

echo "=== BrowseComp-Plus Pilot: ${TOPO} ${ROUNDS_ARG} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "N_TASKS: $N_TASKS"
echo "K_VALUES: $K_VALUES"
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

# Pre-download datasets if not cached (first run only)
echo "Ensuring BrowseComp-Plus datasets are cached..."
python -c "
from datasets import load_dataset
load_dataset('Tevatron/browsecomp-plus', split='test')
load_dataset('Tevatron/browsecomp-plus-corpus', split='train')
print('Datasets ready.')
" || { echo "ERROR: Failed to load datasets"; kill $SGLANG_PID 2>/dev/null; exit 1; }

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks browsecomp_plus \
    --topologies $TOPO \
    --max-react-steps $K_VALUES \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/browsecomp_pilot \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} ${ROUNDS_ARG} failed"

echo "=== BrowseComp-Plus Pilot complete for ${TOPO} ${ROUNDS_ARG}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
