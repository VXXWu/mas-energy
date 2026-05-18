#!/bin/bash
#SBATCH --job-name=is-minimal
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/is_minimal_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/is_minimal_%A_%a.err

# independent_share + minimal_output: stack structural decode reduction
# (drop R=2 debate rounds, replace with one short tool-free refine) AND
# within-call decode reduction (suppress narration alongside tool calls).
#
# Hypothesis: on the 5 prose benchmarks where independent_share already
# matches Decent accuracy (-0.026 to +0.017), adding minimal-output should
# preserve that accuracy and recover an additional ~30-50% of remaining
# energy by also stripping Phase-1 ReAct narration. SWE-bench is included
# as a control: independent_share already collapses there (-0.580); we
# expect minimal to give an equal-or-worse result, confirming iteration
# loss is structural and not recoverable via speaker-side reduction.
#
# Decision rule:
#   - If ΔAcc on each prose bench remains within ±0.05 of Decent: independent_share_minimal
#     becomes the recommended energy-optimal protocol for non-iterative tasks.
#   - If ΔAcc collapses on any prose bench by ≥0.10: minimal can't compose
#     with independent_share's already-stripped Phase-1; speaker narration
#     is load-bearing for the agent's own work even when it's not load-bearing
#     in debate. Keep them as separate interventions.
#
# 6 array tasks: one per benchmark (5 prose hypothesis tests + swebench
# control). No separate _empty variant — comms in independent_share Phase 2
# are final-answer-only, not full peer text, so there's no comm content to
# strip beyond what's already there.
#
# Submit:
#   sbatch mas-energy/scripts/run_independent_share_minimal.sbatch

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
# (job_id, task_id). With (job_id % 1000) * 10 + task_id, max offset is ~10K,
# so BASE must be ≤ 55000. Using 26000 base.
PORT=$((26000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}
K_VALUES="10"

# 6 benchmarks × 1 mode = 6 array tasks
EXTRA_ARGS=""
case $SLURM_ARRAY_TASK_ID in
    0) BENCH=qampari        ; CTX=65536  ;;
    1) BENCH=fanoutqa       ; CTX=65536  ;;
    2) BENCH=math           ; CTX=49152  ;;
    3) BENCH=workbench      ; CTX=49152
       EXTRA_ARGS="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
    4) BENCH=browsecomp_plus; CTX=131072 ;;
    5) BENCH=swebench       ; CTX=131072 ;;
esac

MODE=independent_share_minimal

export SGLANG_CONTEXT_LENGTH=$CTX
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_b2_terse/$BENCH
mkdir -p "$OUT_DIR"

echo "=== independent_share_minimal: ${BENCH} ==="
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
