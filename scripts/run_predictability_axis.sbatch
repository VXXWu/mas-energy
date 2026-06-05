#!/bin/bash
#SBATCH --job-name=pred-ax
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/pred_ax_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/pred_ax_%A_%a.err

# Direct test of the predictability axis (analysis/predictability_axis_test_design.md).
# Synthetic task: N=30 items with hidden integer labels; find all items
# whose label is divisible by 3. Tool: query(item_id) returns the label.
# Two variants (P/U) toggle whether the prompt lists the labels verbatim
# (predictable tool returns) or omits them (unpredictable).
#
# Falsifiable hypothesis (paired by task_index across P/U):
#   H1: P-Decent ≈ P-IS              (rounds wasted when tool returns predictable)
#   H2: U-Decent − U-IS ≥ 0.10       (rounds load-bearing when tool returns unpredictable)
#   H1 ∧ H2 confirms the predictability mechanism.
#   ¬H2 falsifies it (rounds aren't doing the work the mechanism claims).
#
# 3 array tasks (one topology each, runs both variants per task):
#   0: decentralized      (M=3, R=2 — full debate)
#   1: independent_share  (M=3, single share — drops debate rounds)
#   2: decentralized_empty (M=3, R=2 — rounds preserved, peer text muted)
#
# Cell 2 isolates whether the iteration property (if observed) is from
# additional tool-call turns OR from text-channel observation sharing.
# If U-Decent ≈ U-empty > U-IS  → tool-call turns matter; text doesn't
# If U-Decent > U-empty ≈ U-IS  → text observations matter
#
# n=50 per variant per topology (100 paired tasks per cell, 6 cells total).
# Submit:
#   sbatch mas-energy/scripts/run_predictability_axis.sbatch
# Just the headline 2-topology test:
#   sbatch --array=0-1 mas-energy/scripts/run_predictability_axis.sbatch

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
export SGLANG_CONTEXT_LENGTH=16384

# Port: BASE 48000 + (job_id % 1000) * 10 + task_id, max ~58000. Safe.
PORT=$((48000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))

# 50 per variant; benchmark generates 50 P + 50 U = 100 tasks total.
N_TASKS=${N_TASKS:-50}
K_VALUES="5"

case $SLURM_ARRAY_TASK_ID in
    0) MODE=decentralized        ;;
    1) MODE=independent_share    ;;
    2) MODE=decentralized_empty  ;;
esac

OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/predictability_axis
mkdir -p "$OUT_DIR"

echo "=== Predictability-axis: ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS_per_variant: $N_TASKS"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 16384 \
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

# --n-tasks here is per-variant. Benchmark generates 2× this many total tasks
# (one P, one U for each task_index). They share base_seed within a task_index
# so paired analysis pairs P-task[k] with U-task[k].
python run_experiments.py \
    --model toy \
    --benchmarks predictability \
    --topologies $MODE \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: predictability/${MODE} run had errors"

echo "=== Done at $(date) ==="
RESULT_FILE="$OUT_DIR/Qwen_Qwen3.5-9B_${MODE}_k5.jsonl"
if [ -f "$RESULT_FILE" ]; then
    python -c "
import json
rows = [json.loads(l) for l in open('$RESULT_FILE')]
by_var = {'P': [], 'U': []}
for r in rows:
    v = r.get('task', {}).get('variant') or ('P' if 'pred-P-' in r.get('task_id','') else ('U' if 'pred-U-' in r.get('task_id','') else None))
    if v: by_var[v].append(r)
for v in ('P','U'):
    rs = by_var[v]
    n = len(rs)
    if n == 0: continue
    f1_mean = sum(r.get('loose_accuracy',0) or 0 for r in rs)/n
    correct = sum(1 for r in rs if r.get('correct'))/n
    e_mean = sum(r.get('gpu_dynamic_energy_joules',0) or 0 for r in rs)/n
    print(f'  Variant {v}: n={n}  F1={f1_mean:.3f}  exact={correct:.3f}  E={e_mean:.0f}J')
"
fi
