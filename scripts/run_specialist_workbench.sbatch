#!/bin/bash
#SBATCH --job-name=specialist
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/specialist_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/specialist_%A_%a.err

# Specialist tool-partition ablation on WorkBench.
#
# Tests whether the channel-muting result holds when agents have GENUINELY
# disjoint capabilities (not just overlapping access to a shared tool env).
# Each agent gets a round-robin disjoint slice of WorkBench's 27 tools
# (~9 tools per agent at M=3). Agents physically cannot complete some
# subtasks alone — coordination is genuinely necessary.
#
# Four modes (one per array task), two partition strategies × two channel states:
#   _0: decentralized_specialist                    (round-robin partition, FULL channel)
#   _1: decentralized_specialist_empty              (round-robin partition, EMPTY channel)
#   _2: decentralized_specialist_categorical        (deployment-faithful Scheduler/Communicator/CRM-Analyst, FULL)
#   _3: decentralized_specialist_categorical_empty  (deployment-faithful, EMPTY)
#
# The round-robin variant tests whether the channel-muting result holds
# under arbitrary disjoint capabilities. The categorical variant maps to
# real production specialist patterns (CrewAI, MetaGPT, Salesforce Agentforce
# all use domain-specialist agents). If channel-muting holds in the
# categorical variant, the deployment-relevance argument lands directly.
#
# Headline comparison: ΔAcc(specialist_empty − specialist).
#   ≈ 0  → channel content carries no signal even when agents have disjoint
#          capabilities. Strongest possible version of the channel-muting
#          finding.
#   ≪ 0  → channel content IS necessary specifically when capabilities are
#          disjoint. Bounds the original claim's scope.
#
# Output: results/a5000_phase_a_ablation/workbench/Qwen_Qwen3.5-9B_decentralized_specialist[_empty]_k10.jsonl
# The existing analyze_phase_a_ablation.py picks them up automatically when
# added to ABLATION_TOPOS.

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
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((32000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}

case $SLURM_ARRAY_TASK_ID in
    0) MODE=decentralized_specialist                    ;;
    1) MODE=decentralized_specialist_empty              ;;
    2) MODE=decentralized_specialist_categorical        ;;
    3) MODE=decentralized_specialist_categorical_empty  ;;
esac

OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_a_ablation/workbench
mkdir -p "$OUT_DIR"

echo "=== Specialist tool-partition: workbench / ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS: $N_TASKS"

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 49152 \
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
    --benchmarks workbench \
    --topologies $MODE \
    --max-react-steps 10 \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
    || echo "WARNING: ${MODE} run had errors"

echo "=== Done: ${MODE} at $(date) ==="
ls -lh "$OUT_DIR"/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl 2>/dev/null
