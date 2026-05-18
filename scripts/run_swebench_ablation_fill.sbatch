#!/bin/bash
#SBATCH --job-name=swe-fill
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/swe_fill_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/swe_fill_%A_%a.err

# Fill the SWE-bench ablation coverage gap.
#
# SWE-bench is the benchmark where peer-information-sharing is most plausibly
# load-bearing (long-context, multi-file edits, decomposable structure with
# verifiable rewards). Yet most ablation modes were never run on it because
# of swebench's per-task latency.
#
# Existing coverage:
#   Independent:       100/100 ✓
#   Decent (full):     43/100 (still partial — limits paired analysis)
#   answer_only:       100/100 ✓ (paired 43)
#   truncate300:       63/100 (paired 43)
#   terse:             51/100 (paired 43)
# Missing:
#   empty              ← MOST CRITICAL: the channel-muting test
#   empty_silent       ← strict control for empty
#   truncate100        ← completes Phase A receiver-side ablation
#   independent_share  ← most aggressive simplification
#
# Each array task runs ONE mode at n=100, so each has a full 36h to itself.
# Per-task latency on swebench is 3-5 min; n=100 ≈ 6-8h per mode.
#
# Output dir aligned with phase_a_ablation/ so the existing analyzer
# auto-pairs against the same Decent baseline.
#
# Submit:
#   sbatch mas-energy/scripts/run_swebench_ablation_fill.sbatch

set -e

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then
    echo "Staggering ${STAGGER}s..."
    sleep $STAGGER
fi

source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export SGLANG_CONTEXT_LENGTH=131072

PORT=$((30000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}

# One mode per array task — keeps each within the 36h time limit comfortably
case $SLURM_ARRAY_TASK_ID in
    0) MODE=decentralized_empty            ; OUT_ROOT=results/a5000_phase_a_ablation ;;
    1) MODE=decentralized_empty_silent     ; OUT_ROOT=results/a5000_phase_a_ablation ;;
    2) MODE=decentralized_truncate100      ; OUT_ROOT=results/a5000_phase_a_ablation ;;
    3) MODE=independent_share              ; OUT_ROOT=results/a5000_phase_b2_terse   ;;
esac

OUT_DIR="/atlas2/u/$USER/mas_project/mas-energy/${OUT_ROOT}/swebench"
mkdir -p "$OUT_DIR"

echo "=== swebench ablation fill: ${MODE} (n=${N_TASKS}, k=10, R=2) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Port: $PORT  Output: $OUT_DIR"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 131072 \
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
    --benchmarks swebench \
    --topologies $MODE \
    --max-react-steps 10 \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: ${MODE} run had errors"

echo "=== Done: ${MODE} at $(date) ==="
ls -lh "$OUT_DIR"/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl 2>/dev/null
