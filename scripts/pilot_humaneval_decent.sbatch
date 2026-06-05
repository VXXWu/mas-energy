#!/bin/bash
#SBATCH --job-name=he-dec
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/he_decent_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/he_decent_pilot_%j.err

# HumanEval+ Decent (R=2, k=10) pilot at n=20.
# Calibrates the actual ceiling for ablation comparisons. SAS+tools pilot
# bounds from below; this bounds from above (Decent will be the highest-
# accuracy cell in the ablation).
# If Decent here is ≥0.95, IS-vs-Decent collapse magnitudes < 0.10 won't be
# detectable at n=164 (CI noise floor).
#
# Wall time: ~1-2h (~5 min × 20 tasks).
# Submit:
#   sbatch mas-energy/scripts/pilot_humaneval_decent.sbatch

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

PORT=$((45000 + SLURM_JOB_ID % 4000))
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/humaneval_pilot
mkdir -p "$OUT_DIR"

echo "=== HumanEval+ Decent pilot (R=2, k=10, n=20) ==="
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

# Decent at R=2, k=10 — same regime the full ablation will run in
python run_experiments.py \
    --model toy \
    --benchmarks humaneval \
    --topologies decentralized \
    --max-react-steps 10 \
    --rounds 2 \
    --n-tasks 20 --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1"

echo "=== Done at $(date) ==="
RESULT_FILE="$OUT_DIR/Qwen_Qwen3.5-9B_decentralized_k10.jsonl"
if [ -f "$RESULT_FILE" ]; then
    python -c "
import json
rows = [json.loads(l) for l in open('$RESULT_FILE')]
n = len(rows)
p = sum(1 for r in rows if r.get('correct'))
rate = p/n if n else 0
print(f'  Decent pass@1 = {rate:.3f} ({p}/{n})')
print(f'  total tokens (mean): {sum(r.get(\"total_completion_tokens\",0) for r in rows)/n if n else 0:.0f}')
print(f'  energy J (mean): {sum(r.get(\"gpu_dynamic_energy_joules\",0) or 0 for r in rows)/n if n else 0:.0f}')
"
else
    echo "  (no result file)"
fi
