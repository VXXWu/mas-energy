#!/bin/bash
#SBATCH --job-name=browse-share
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/browse_share_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/browse_share_%j.err

# Fill the missing browsecomp_plus / independent_share cell.
# The Phase B-2 sbatch's browsecomp_plus array task hit time limit before
# reaching independent_share (third mode in its sequence after terse +
# terse_answer_only). At n=100 paired against the 78-row Decent baseline,
# this completes the "independent_share works on prose, breaks on code"
# story across all prose benchmarks.

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

PORT=$((36000 + SLURM_JOB_ID % 4000))
OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_b2_terse/browsecomp_plus
mkdir -p "$OUT_DIR"

echo "=== browsecomp_plus / independent_share fill ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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
    --benchmarks browsecomp_plus \
    --topologies independent_share \
    --max-react-steps 10 \
    --n-tasks 100 --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: run had errors"

echo "=== Done at $(date) ==="
ls -lh "$OUT_DIR"/Qwen_Qwen3.5-9B_independent_share_k10.jsonl 2>/dev/null
