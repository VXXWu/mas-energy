#!/bin/bash
#SBATCH --job-name=he-sas
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/he_sas_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/he_sas_pilot_%j.err

# HumanEval+ SAS-WITH-TOOLS pilot. Calibrates the actual ablation regime:
# k=10 ReAct steps with run_tests tool feedback per task.
#
# Decision rule (revised — this is the regime the ablation runs in):
#   pass@1 < 0.85  → Decent has headroom; ablation comparisons valid for
#                    both collapse tests (IS) and subtle effects (minimal).
#   pass@1 0.85-0.92 → IS collapse still detectable; minimal-output cells
#                       borderline. Run --array=0-1 (essential cells only).
#   pass@1 > 0.92  → near-ceiling. Pivot to LiveCodeBench for ablation;
#                     use HumanEval+ only as a confirmatory replication
#                     of the IS collapse if it shows up.
#
# Wall time: ~20 min (~5 min SGLang warmup + ~30s avg per task × 20 tasks).
# Submit:
#   sbatch mas-energy/scripts/pilot_humaneval_sas_tools.sbatch

set -e

source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export SGLANG_CONTEXT_LENGTH=32768

PORT=$((43000 + SLURM_JOB_ID % 4000))
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/humaneval_pilot
mkdir -p "$OUT_DIR"

echo "=== HumanEval+ SAS-WITH-TOOLS pilot (k=10, n=20) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT"

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

# SAS at k=10 with run_tests tool — same regime the ablation will run in
python run_experiments.py \
    --model toy \
    --benchmarks humaneval \
    --topologies sas \
    --max-react-steps 10 \
    --n-tasks 20 --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1"

echo "=== Done at $(date) ==="
# Print pass@1
RESULT_FILE="$OUT_DIR/Qwen_Qwen3.5-9B_sas_k10.jsonl"
if [ -f "$RESULT_FILE" ]; then
    PASS=$(python -c "
import json
rows = [json.loads(l) for l in open('$RESULT_FILE')]
n = len(rows)
p = sum(1 for r in rows if r.get('correct'))
print(f'  pass@1 = {p/n:.3f} ({p}/{n})')
if p/n < 0.85: print('  verdict: BELOW SATURATION — full 4-cell ablation OK')
elif p/n < 0.92: print('  verdict: TIGHT — run --array=0-1 (essential cells only)')
else: print('  verdict: NEAR-CEILING — pivot to LiveCodeBench')
"
    )
    echo "$PASS"
else
    echo "  (no result file produced)"
fi
