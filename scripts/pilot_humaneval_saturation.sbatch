#!/bin/bash
#SBATCH --job-name=he-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/he_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/he_pilot_%j.err

# HumanEval+ saturation pilot. n=20 SAS-style single-agent generation,
# pass@1 measurement, decision-rule output.
# Decides whether HumanEval+ is usable as the second code benchmark or
# whether to pivot to LiveCodeBench.
# Expected wall time: ~10-15 min (20 tasks × ~30s gen + ~30s testing).
# Submit:
#   sbatch mas-energy/scripts/pilot_humaneval_saturation.sbatch

set -e

source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Pre-fetch dataset on a node with internet (HuggingFace cache will reuse on later runs)
python -c "from datasets import load_dataset; load_dataset('evalplus/humanevalplus', split='test')" 2>&1 | head -5

# Port: non-array job, max offset SLURM_JOB_ID % 4000 = 3999. Base 42000 → max 45999.
PORT=$((42000 + SLURM_JOB_ID % 4000))
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/humaneval_pilot
mkdir -p "$OUT_DIR"

echo "=== HumanEval+ saturation pilot ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
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

cd /atlas2/u/$USER/mas_project/mas-energy

python scripts/pilot_humaneval_saturation.py \
    --sglang-url "http://localhost:$PORT/v1" \
    --model Qwen/Qwen3.5-9B \
    --n-tasks 20 \
    --temperature 0.0 \
    --max-tokens 2048 \
    --out-jsonl "$OUT_DIR/sas_pilot_n20.jsonl"

echo "=== Done at $(date) ==="
