#!/bin/bash
#SBATCH --job-name=swe-decent
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/swe_decent_extend_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/swe_decent_extend_%j.err

# Extend swebench Decent baseline from 43 → 100 tasks.
# This is the gating sample for ALL swebench paired-ablation analysis.
# At n=43, paired CIs are ±0.10 to ±0.20; at n=100, they shrink to ±0.07 to ±0.13.
# Resume logic in run_experiments.py picks up where the existing 43 left off.
#
# Specifically tightens: empty vs empty_silent gap on swebench (currently
# borderline-significant suggestion that meta-instruction carries signal on code).
#
# Submit:
#   sbatch mas-energy/scripts/extend_swebench_decent_baseline.sbatch

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export SGLANG_CONTEXT_LENGTH=131072

PORT=$((34000 + SLURM_JOB_ID % 4000))
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_latent_transcripts/swebench
mkdir -p "$OUT_DIR"

echo "=== Extending swebench Decent baseline to n=100 ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT"
echo "Resume: existing 43 rows in $OUT_DIR/Qwen_Qwen3.5-9B_decentralized_k10.jsonl"

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

# Resume: run_experiments.py skips already-completed task_ids in the jsonl
python run_experiments.py \
    --model toy \
    --benchmarks swebench \
    --topologies decentralized \
    --max-react-steps 10 \
    --n-tasks 100 --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: run had errors"

echo "=== Done at $(date) ==="
wc -l "$OUT_DIR/Qwen_Qwen3.5-9B_decentralized_k10.jsonl"
