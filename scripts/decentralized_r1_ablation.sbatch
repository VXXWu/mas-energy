#!/bin/bash
#SBATCH --job-name=dec-r1
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3%1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/dec_r1_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/dec_r1_%A_%a.err

# Decentralized R=1 ablation across all 4 benchmarks.
# Tests whether 1 debate round captures most of the benefit
# at ~half the energy of R=2 (default).
#
# Comparison: R=0 is Independent (already have data).
#             R=2 is default Decentralized (already have data).
#             R=1 fills the gap.
#
# Array tasks (sequential):
#   0: QAMPARI       k=2,5,10
#   1: FanOutQA      k=2,5,10
#   2: BrowseComp+   k=2,5,10
#   3: WorkBench     k=2,5,10

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000
N_TASKS=50
WORKBENCH_PATH=/atlas2/u/$USER/mas_project/WorkBench
BASE=/atlas2/u/$USER/mas_project/mas-energy/results

case $SLURM_ARRAY_TASK_ID in
    0) BENCH=qampari;         OUTPUT_DIR=$BASE/qampari_v4;       WB_N=50 ;;
    1) BENCH=fanoutqa;        OUTPUT_DIR=$BASE/fanoutqa_v4;       WB_N=50 ;;
    2) BENCH=browsecomp_plus; OUTPUT_DIR=$BASE/browsecomp_pilot;  WB_N=50 ;;
    3) BENCH=workbench;       OUTPUT_DIR=$BASE/workbench_v2;      WB_N=100 ;;
esac

# WorkBench uses n=100 (2 reps x 50 tasks), others use n=50
if [ "$BENCH" = "workbench" ]; then
    N_TASKS=$WB_N
fi

echo "=== Decentralized R=1: ${BENCH} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "N_TASKS: $N_TASKS"
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
    --topologies decentralized \
    --max-react-steps 2 5 10 \
    --rounds 1 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH \
|| echo "WARNING: ${BENCH} failed"

echo "=== Done: ${BENCH}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
