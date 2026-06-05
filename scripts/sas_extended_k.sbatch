#!/bin/bash
#SBATCH --job-name=sas-ext-k
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-7%1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/sas_ext_k_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/sas_ext_k_%A_%a.err

# SAS with k=15,20 across all 4 benchmarks.
# Token-matched comparison: SAS at higher k uses similar total tokens
# as MAS topologies, isolating whether MAS coordination adds value
# beyond simply giving a single agent more compute.
#
# Writes to existing results dirs so data is co-located.
#
# Array tasks:
#   0: qampari       k=15
#   1: qampari       k=20
#   2: fanoutqa      k=15
#   3: fanoutqa      k=20
#   4: browsecomp    k=15
#   5: browsecomp    k=20
#   6: workbench     k=15
#   7: workbench     k=20

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=$((30000 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-50}
BASE=/atlas2/u/$USER/mas_project/mas-energy/results
WORKBENCH_PATH=/atlas2/u/$USER/mas_project/WorkBench

case $SLURM_ARRAY_TASK_ID in
    0) BENCH=qampari;         K=15; OUTPUT_DIR=$BASE/qampari_v4 ;;
    1) BENCH=qampari;         K=20; OUTPUT_DIR=$BASE/qampari_v4 ;;
    2) BENCH=fanoutqa;        K=15; OUTPUT_DIR=$BASE/fanoutqa_v4 ;;
    3) BENCH=fanoutqa;        K=20; OUTPUT_DIR=$BASE/fanoutqa_v4 ;;
    4) BENCH=browsecomp_plus; K=15; OUTPUT_DIR=$BASE/browsecomp_pilot ;;
    5) BENCH=browsecomp_plus; K=20; OUTPUT_DIR=$BASE/browsecomp_pilot ;;
    6) BENCH=workbench;       K=15; OUTPUT_DIR=$BASE/workbench_v2 ;;
    7) BENCH=workbench;       K=20; OUTPUT_DIR=$BASE/workbench_v2 ;;
esac

echo "=== SAS Extended k: ${BENCH} k=${K} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Output: $OUTPUT_DIR"
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

python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies sas \
    --max-react-steps $K \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH \
|| echo "WARNING: ${BENCH} k=${K} failed"

echo "=== Done: ${BENCH} k=${K}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
