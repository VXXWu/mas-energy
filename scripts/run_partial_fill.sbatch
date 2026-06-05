#!/bin/bash
#SBATCH --job-name=fill
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/fill_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/fill_%A_%a.err

# Resubmits incomplete ablation cells. Each array task is (benchmark, mode);
# resume logic in run_experiments.py skips already-completed task IDs.
#
# Cells (audited 2026-04-29):
#   0: browsecomp_plus / terse_answer_only  (n=17 → 100)  *** fixes fig_frontier outlier ***
#   1: browsecomp_plus / empty              (n=75 → 100)
#   2: browsecomp_plus / minimal            (n=54 → 100)
#   3: browsecomp_plus / minimal_empty      (n=54 → 100)
#   4: swebench / terse                     (n=51 → 100)
#   5: swebench / truncate300               (n=69 → 100)
#   6: swebench / minimal                   (n=22 → 100)
#   7: swebench / minimal_empty             (n=96 → 100)
#   8: workbench / minimal                  (n=82 → 100)
#   9: workbench / minimal_empty            (n=92 → 100)
#
# To run only the figure-critical cell (#0):
#   sbatch --array=0 mas-energy/scripts/run_partial_fill.sbatch
# To run all 10:
#   sbatch mas-energy/scripts/run_partial_fill.sbatch

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

# Port: BASE 50000 + (job_id % 1000)*10 + task_id; max ~60000. Safe.
PORT=$((50000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}
K_VALUES="10"

EXTRA_ARGS=""
case $SLURM_ARRAY_TASK_ID in
    0) BENCH=browsecomp_plus; MODE=decentralized_terse_answer_only; CTX=131072
       OUT_ROOT=results/a5000_phase_b2_terse ;;
    1) BENCH=browsecomp_plus; MODE=decentralized_empty             ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    2) BENCH=browsecomp_plus; MODE=decentralized_minimal           ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    3) BENCH=browsecomp_plus; MODE=decentralized_minimal_empty     ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    4) BENCH=swebench       ; MODE=decentralized_terse             ; CTX=131072
       OUT_ROOT=results/a5000_phase_b2_terse ;;
    5) BENCH=swebench       ; MODE=decentralized_truncate300       ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    6) BENCH=swebench       ; MODE=decentralized_minimal           ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    7) BENCH=swebench       ; MODE=decentralized_minimal_empty     ; CTX=131072
       OUT_ROOT=results/a5000_phase_a_ablation ;;
    8) BENCH=workbench      ; MODE=decentralized_minimal           ; CTX=49152
       OUT_ROOT=results/a5000_phase_a_ablation
       EXTRA_ARGS="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
    9) BENCH=workbench      ; MODE=decentralized_minimal_empty     ; CTX=49152
       OUT_ROOT=results/a5000_phase_a_ablation
       EXTRA_ARGS="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
esac

export SGLANG_CONTEXT_LENGTH=$CTX
OUT_DIR=/atlas2/u/$USER/mas_project/$OUT_ROOT/$BENCH
mkdir -p "$OUT_DIR"

echo "=== Fill: ${BENCH} / ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS: $N_TASKS"
RESULT_FILE="$OUT_DIR/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl"
if [ -f "$RESULT_FILE" ]; then
    echo "Existing rows: $(wc -l < $RESULT_FILE)"
fi

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
[ -f "$RESULT_FILE" ] && echo "Final rows: $(wc -l < $RESULT_FILE)"
