#!/bin/bash
#SBATCH --job-name=swe-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/swe_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/swe_pilot_%j.err

# SWE-bench-Lite MINIMUM pilot: 2 tasks, SAS only, k=5.
# Just validates the pipeline works end-to-end:
#   1. Repo cloning
#   2. Git worktree creation at correct commit
#   3. Agent bash tool execution in worktree
#   4. Patch extraction via git diff
#   5. Energy measurement
#   6. Cleanup (worktree removal)
#
# NOT an array job -- single sequential run for debugging.

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000
REPOS_DIR=/atlas2/u/$USER/mas_project/swebench_repos
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/swebench_pilot
WORKBENCH_PATH=/atlas2/u/$USER/mas_project/WorkBench

echo "=== SWE-bench Minimum Pilot ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Clone sympy repo if not present (full clone, not bare)
mkdir -p $REPOS_DIR
if [ ! -d "$REPOS_DIR/sympy/.git" ]; then
    echo "Cloning sympy repository (full history needed for worktrees)..."
    git clone https://github.com/sympy/sympy.git $REPOS_DIR/sympy
    echo "Sympy clone complete: $(date)"
else
    echo "Sympy repo already present"
fi

# Verify repo is usable
echo "Verifying sympy repo..."
cd $REPOS_DIR/sympy && git log --oneline -1 && cd -
if [ $? -ne 0 ]; then
    echo "ERROR: sympy repo is broken"
    exit 1
fi

# Start SGLang
if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 131072 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!

echo "Waiting for SGLang server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: SGLang failed to start"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

export SWEBENCH_REPOS=$REPOS_DIR

echo "=== Running SAS k=5, 2 tasks ==="
python run_experiments.py \
    --model toy \
    --benchmarks swebench \
    --topologies sas \
    --max-react-steps 50 \
    --n-tasks 2 --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path $WORKBENCH_PATH

EXITCODE=$?
echo "Exit code: $EXITCODE"

if [ $EXITCODE -eq 0 ]; then
    echo "=== SAS pilot succeeded, running Centralized k=5 R=1, 2 tasks ==="
    python run_experiments.py \
        --model toy \
        --benchmarks swebench \
        --topologies centralized \
        --max-react-steps 50 \
        --rounds 1 \
        --n-tasks 2 --n-reps 1 \
        --output-dir $OUTPUT_DIR \
        --sglang-url "http://localhost:$PORT/v1" \
        --workbench-path $WORKBENCH_PATH \
    || echo "WARNING: Centralized failed"

    echo "=== Running Decentralized k=5, 2 tasks ==="
    python run_experiments.py \
        --model toy \
        --benchmarks swebench \
        --topologies decentralized \
        --max-react-steps 50 \
        --n-tasks 2 --n-reps 1 \
        --output-dir $OUTPUT_DIR \
        --sglang-url "http://localhost:$PORT/v1" \
        --workbench-path $WORKBENCH_PATH \
    || echo "WARNING: Decentralized failed"
else
    echo "SAS pilot FAILED -- skipping other topologies"
fi

echo "=== Pilot complete: $(date) ==="
echo "Check results: ls $OUTPUT_DIR/"
echo "Check patches: ls /atlas2/u/$USER/mas_project/swebench_patches/"

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
