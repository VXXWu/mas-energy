#!/bin/bash
#SBATCH --job-name=minimal-out
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/minimal_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/minimal_%A_%a.err

# Minimum-output ablation: tests whether the agent's intermediate ReAct
# narration (between tool calls) is decode-waste or load-bearing for the
# agent's own reasoning.
#
# Variants:
#   decentralized_minimal       : suppress all agent text during trajectory;
#                                 receiver still sees full peer text
#   decentralized_minimal_empty : suppress all agent text + receiver sees
#                                 empty peer placeholder
#
# Distinct from `terse`: terse only restricts the FINAL response. Minimal
# also suppresses narration BETWEEN tool calls. The system prompt includes:
#   "When making tool calls, output ONLY the tool calls — do not write any
#    explanation, reasoning, or commentary alongside them."
#
# Run on all 6 benchmarks at n=100, k=10, R=2 to compare against existing
# Decent baseline + terse.
#
# Submit:
#   sbatch mas-energy/scripts/run_minimal_output.sbatch

set -e

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then sleep $STAGGER; fi

source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Port budget: TCP max is 65535. Formula must yield ports below that for any
# (job_id, task_id). With (job_id % 4000) * 10 + task_id, max offset is ~40K,
# so BASE must be ≤ 25500. Using 20000 base.
PORT=$((20000 + SLURM_ARRAY_JOB_ID % 4000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}
K_VALUES="10"

# 6 benchmarks × 2 modes (minimal, minimal_empty) = 12 array tasks
# array index = bench_idx * 2 + mode_idx
EXTRA_ARGS=""
BENCH_IDX=$((SLURM_ARRAY_TASK_ID / 2))
MODE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

case $BENCH_IDX in
    0) BENCH=qampari        ; CTX=65536  ;;
    1) BENCH=fanoutqa       ; CTX=65536  ;;
    2) BENCH=math           ; CTX=49152  ;;
    3) BENCH=workbench      ; CTX=49152
       EXTRA_ARGS="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
    4) BENCH=browsecomp_plus; CTX=131072 ;;
    5) BENCH=swebench       ; CTX=131072 ;;
esac

case $MODE_IDX in
    0) MODE=decentralized_minimal       ;;
    1) MODE=decentralized_minimal_empty ;;
esac

export SGLANG_CONTEXT_LENGTH=$CTX
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_a_ablation/$BENCH
mkdir -p "$OUT_DIR"

echo "=== Minimum-output: ${BENCH} / ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS: $N_TASKS"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length $CTX \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!
trap "kill $SGLANG_PID 2>/dev/null || true" EXIT

echo "Waiting for SGLang server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "ready ${i}s"; break; fi
    if [ $i -eq 180 ]; then echo "ERROR: SGLang failed"; exit 1; fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies $MODE \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA_ARGS \
    || echo "WARNING: ${BENCH}/${MODE} run had errors"

echo "=== Done at $(date) ==="
ls -lh "$OUT_DIR"/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl 2>/dev/null
