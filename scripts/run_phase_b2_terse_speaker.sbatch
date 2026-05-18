#!/bin/bash
#SBATCH --job-name=phaseB2-terse
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/phase_b2_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/phase_b2_%A_%a.err

# Phase B-2: speaker-side decode reduction.
# Decentralized + terse-speaker prompt (instructs the agent to output only the
# final answer, no reasoning, no commentary). Directly attacks decode cost —
# the dominant energy term.
#
# Three modes per benchmark (run sequentially within each array task to share
# SGLang warm-up):
#   decentralized_terse              : terse output, full peer info shown
#                                       (isolates speaker-side change)
#   decentralized_terse_answer_only  : terse output + answer-only peer info
#                                       (combined Phase A + B-2)
#   independent_share                : Independent + one text-only refinement
#                                       round — most aggressive simplification.
#                                       Tests whether Decent's accuracy benefit
#                                       comes from agents reading peer answers
#                                       (preserved here) or from receiver-side
#                                       additional tool calls (skipped here).
#
# Compared against the existing full-text Decent baseline in
# results/a5000_latent_transcripts/<bench>/Qwen_Qwen3.5-9B_decentralized_k10.jsonl
# the analyzer pairs by task_id and reports ΔAcc + Δ-tokens + Δ-energy.
#
# Submit:
#   sbatch mas-energy/scripts/run_phase_b2_terse_speaker.sbatch

set -e

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then
    echo "Staggering ${STAGGER}s for cache/pip/SGLang init contention..."
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

PORT=$((26000 + SLURM_ARRAY_JOB_ID % 4000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}
K_VALUES="10"
OUT_ROOT=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_b2_terse

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

export SGLANG_CONTEXT_LENGTH=$CTX
OUT_DIR="${OUT_ROOT}/${BENCH}"
mkdir -p "$OUT_DIR"

echo "=== Phase B-2 terse-speaker: ${BENCH} (n=${N_TASKS}, k=${K_VALUES}) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

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

# Two modes:
#   terse              : terse-speaker, FULL peer info (isolates speaker change)
#   terse_answer_only  : terse-speaker + answer-only peer info (combined)
TERSE_TOPOS="decentralized_terse decentralized_terse_answer_only independent_share"

python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies $TERSE_TOPOS \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA_ARGS \
    || echo "WARNING: ${BENCH} run had errors"

echo "=== Done: ${BENCH} at $(date) ==="
ls -lh "$OUT_DIR"/*.jsonl 2>/dev/null
