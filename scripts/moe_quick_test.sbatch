#!/bin/bash
#SBATCH --job-name=moe-test
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/moe_test_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/moe_test_%A_%a.err

# Quick MoE vs Dense energy comparison on BrowseComp-Plus.
# Tests whether MoE (3B active) has different energy profile
# than Dense (9B active) under MAS context growth.
#
# Array tasks:
#   0: sas           k=5,10
#   1: independent   k=5,10
#   2: centralized   k=5,10  R=1
#   3: decentralized k=5,10

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=$((30000 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-50}
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/moe_quick_test

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;            K_VALUES="5 10"; ROUNDS_ARG="" ;;
    1) TOPO=independent;    K_VALUES="5 10"; ROUNDS_ARG="" ;;
    2) TOPO=centralized;    K_VALUES="5 10"; ROUNDS_ARG="--rounds 1" ;;
    3) TOPO=decentralized;  K_VALUES="5 10"; ROUNDS_ARG="" ;;
esac

echo "=== MoE Quick Test: ${TOPO} ${ROUNDS_ARG} ==="
echo "Model: Qwen3.5-35B-A3B (MoE, 3B active, GPTQ-Int4)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
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

python run_experiments.py \
    --model qwen35b-a3b \
    --benchmarks browsecomp_plus \
    --topologies $TOPO \
    --max-react-steps $K_VALUES \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} ${ROUNDS_ARG} failed"

echo "=== Done: ${TOPO} ${ROUNDS_ARG}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
