#!/bin/bash
#SBATCH --job-name=math-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/math_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/math_pilot_%j.err

# Pilot: MATH (Level 5) reasoning benchmark with stateful Python interpreter.
# Tests whether Decentralized debate produces a meaningful collaboration
# dividend on reasoning-heavy agentic tasks — filling the gap our existing
# retrieval-heavy matrix has where Decent ≈ Indep aggregate.
#
# Predictions before running:
#   - Single (SAS): ~30-50% on Level 5 (model-dependent)
#   - Independent + voting: marginal improvement (math errors poison votes)
#   - Decentralized: largest improvement (Du et al. 2023 mechanism: debate
#     resolves arithmetic errors)
# If Decent significantly beats Indep here, our paper's claim becomes
# scoped to "retrieval-heavy → no, reasoning-heavy → yes" (clean).
# If Decent doesn't beat Indep here either, the claim is universal (strong).

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Self-heal env drift: the mas-energy env has been in a degraded state.
# Each missing/outdated dep is a separate failure mode; patch each guarded
# and pinned (per project memory: never bare pip install in conda env).
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
# jinja2 3.0.3 is too old for transformers' apply_chat_template (needs >=3.1.0).
# Without the upgrade, SGLang's tool-call-parser path returns 500 errors.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

OUT_ROOT=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_math_pilot
mkdir -p "$OUT_ROOT"

cd /atlas2/u/$USER/mas_project/mas-energy/code

# Start SGLang server (background) — must include --tool-call-parser
# qwen3_coder so the model's XML <tool_call><function=...> format gets
# parsed into structured tool_calls. Without it, the agent ReAct loop
# never sees parsed tool calls and exits after step 0 (giving 0% on SAS).
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port 30000 \
    --mem-fraction-static 0.80 \
    --tp 1 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --context-length 32768 \
    --host 0.0.0.0 &
SGLANG_PID=$!
trap "kill $SGLANG_PID 2>/dev/null || true" EXIT

echo "Waiting for SGLang server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:30000/health >/dev/null 2>&1; then
        echo "  ready after ${i}x5s"
        break
    fi
    sleep 5
done

echo "=== MATH Level-5 pilot on Qwen3.5-9B ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# 30 tasks per topology, 1 rep, k=10 ReAct steps. Quick pilot to confirm
# (a) the benchmark + tool integration works end-to-end
# (b) sample-level CD distribution differs meaningfully from retrieval benchmarks
# Model alias: 'toy' is Qwen3.5-9B per config.TOY_MODEL (the 9B Dense model
# we've been using for pilots). Other registered keys: qwen35b-a3b (MoE),
# glm47-flash (MoE), qwen35-27b (Dense).
python run_experiments.py \
    --model toy \
    --benchmarks math \
    --topologies sas independent centralized decentralized \
    --n-tasks 30 --n-reps 1 \
    --max-react-steps 10 \
    --save-transcripts \
    --output-dir "$OUT_ROOT"

echo "=== Done: $(date) ==="
echo "Output: $OUT_ROOT"
