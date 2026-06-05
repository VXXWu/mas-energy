#!/bin/bash
#SBATCH --job-name=ks-cd-r2
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_all_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_all_%A_%a.err

# Full k-sweep: 4 topologies × 4 benchmarks = 16 array tasks, 8 concurrent.
# Resume logic skips any k values already at 50 records.
# Centralized uses --rounds 2 (standardized with decentralized).
# SAS/Independent already complete — resume will skip them instantly.

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then
    pip install -q --upgrade "jinja2>=3.1.4"
fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50

# Map array index to (topology, benchmark)
# 0-3:   Centralized (R=2) × 4 benchmarks
# 4-7:   Decentralized × 4 benchmarks

BENCHMARKS=(qampari fanoutqa workbench browsecomp_plus)
OUTPUT_DIRS=(
    /atlas2/u/$USER/mas_project/mas-energy/results/a5000_qampari_v4
    /atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4
    /atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2
    /atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot
)

TOPO_IDX=$((SLURM_ARRAY_TASK_ID / 4))
BENCH_IDX=$((SLURM_ARRAY_TASK_ID % 4))

BENCH=${BENCHMARKS[$BENCH_IDX]}
OUTPUT_DIR=${OUTPUT_DIRS[$BENCH_IDX]}

case $TOPO_IDX in
    0) TOPO=centralized;    K_VALUES="1 2 3 5 7 10 15 20 30 50"; ROUNDS_ARG="--rounds 2" ;;
    1) TOPO=decentralized;  K_VALUES="1 2 3 5 7 10 15 20 30 50"; ROUNDS_ARG="" ;;
esac

EXTRA=""
if [ "$BENCH" = "workbench" ]; then
    EXTRA="--workbench-path /atlas2/u/$USER/mas_project/WorkBench"
fi

echo "=== k-scan: ${BENCH} / ${TOPO} ${ROUNDS_ARG} k={$K_VALUES} ==="
echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID (topo=$TOPO_IDX bench=$BENCH_IDX)"
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
    --topologies $TOPO \
    --max-react-steps $K_VALUES \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA \
|| echo "WARNING: ${BENCH} ${TOPO} failed"

echo "=== ${BENCH} / ${TOPO} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
