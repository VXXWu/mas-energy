#!/bin/bash
#SBATCH --job-name=he-abl
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/he_abl_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/he_abl_%A_%a.err

# HumanEval+ second-code-benchmark ablation. Tests whether the swebench
# iteration-collapse claim generalizes to atomic code+test problems
# (no multi-file edits, no repo dependencies).
#
# Pilot showed pass@1 = 0.60 SAS at n=20 — well below saturation (0.70 threshold),
# plenty of headroom for ablation differences.
#
# 4 cells × n=164 (full HumanEval+):
#   array idx 0: decentralized              (baseline)
#   array idx 1: independent_share          (iteration collapse test — core)
#   array idx 2: decentralized_minimal      (cross-bench decode-waste replication)
#   array idx 3: independent_share_minimal  (stacking interaction)
#
# Decision rules:
# - independent_share collapses (ΔAcc ≤ -0.10): swebench iteration claim
#   generalizes to atomic code; n=2 in positive class.
# - independent_share matches Decent (ΔAcc within ±0.05): swebench-specific
#   factor (multi-file editing? repo complexity?) drove the collapse, not
#   write-test-fix iteration alone. Claim must scope.
# - independent_share partial collapse (-0.05 to -0.10): ambiguous; report
#   as suggestive.
#
# Submit:
#   sbatch mas-energy/scripts/run_humaneval_ablation.sbatch
# Or just the two essential cells:
#   sbatch --array=0-1 mas-energy/scripts/run_humaneval_ablation.sbatch

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
export SGLANG_CONTEXT_LENGTH=32768

# Port: BASE 44000, max offset (job_id%1000)*10 + 3 = ~10003 → max 54003. Safe.
PORT=$((44000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-164}
K_VALUES="10"

# 4 array tasks: one topology each
case $SLURM_ARRAY_TASK_ID in
    0) MODE=decentralized                ; OUT_ROOT=results/a5000_latent_transcripts ;;
    1) MODE=independent_share            ; OUT_ROOT=results/a5000_phase_b2_terse     ;;
    2) MODE=decentralized_minimal        ; OUT_ROOT=results/a5000_phase_a_ablation   ;;
    3) MODE=independent_share_minimal    ; OUT_ROOT=results/a5000_phase_b2_terse     ;;
esac

OUT_DIR=/atlas2/u/$USER/mas_project/$OUT_ROOT/humaneval
mkdir -p "$OUT_DIR"

echo "=== HumanEval+: ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS: $N_TASKS"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
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
    --benchmarks humaneval \
    --topologies $MODE \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: humaneval/${MODE} run had errors"

echo "=== Done at $(date) ==="
ls -lh "$OUT_DIR"/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl 2>/dev/null
