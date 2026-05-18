#!/bin/bash
#SBATCH --job-name=phaseA-abl
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/phase_a_abl_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/phase_a_abl_%A_%a.err

# Phase A counterfactual ablation for the latent inter-agent communication pilot.
#
# For each benchmark, runs Decentralized at four progressively-truncated peer
# message modes (registered in topologies.TOPOLOGY_RUNNERS as the
# decentralized_<mode> variants). Comparison vs the standard Decent baseline
# (collected by collect_latent_transcripts.sbatch) reveals how much of the peer
# text B actually needs.
#
# Modes:
#   decentralized_answer_only   : peer slot = final answer only (drop tool calls + reasoning)
#   decentralized_truncate300   : peer slot = first 300 chars of final answer
#   decentralized_truncate100   : peer slot = first 100 chars of final answer
#   decentralized_empty         : peer slot = "(output not shared)" — control, should hurt
#
# Hypothesis test: if accuracy stays within ~3% of full Decent at truncate300
# or even truncate100, then ~70-90% of debate-round decode is replaceable with
# a small latent payload — empirical validation of the compressibility hypothesis
# beyond the n-gram heuristic.
#
# Output: results/a5000_phase_a_ablation/<bench>/Qwen_Qwen3.5-9B_<topology>_k<K>.jsonl
#
# Submit (after or alongside collect_latent_transcripts.sbatch):
#   sbatch mas-energy/scripts/run_phase_a_ablation.sbatch
#
# Override n_tasks for a smoke test:
#   sbatch --export=ALL,N_TASKS=20 mas-energy/scripts/run_phase_a_ablation.sbatch

set -e

# Stagger by array index to avoid HF cache / pip / SGLang init contention
STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then
    echo "Staggering: sleeping ${STAGGER}s (array idx $SLURM_ARRAY_TASK_ID)"
    sleep $STAGGER
fi

source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Different port range from collect_latent_transcripts (22000+) to avoid overlap
# if the two jobs run concurrently
PORT=$((24000 + SLURM_ARRAY_JOB_ID % 4000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-100}
K_VALUES="10"
OUT_ROOT=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_phase_a_ablation

EXTRA_ARGS=""
case $SLURM_ARRAY_TASK_ID in
    0) BENCH=qampari        ; CTX=65536  ;;
    1) BENCH=fanoutqa       ; CTX=65536  ;;
    2) BENCH=math           ; CTX=49152  ;;
    3) BENCH=workbench      ; CTX=49152
       EXTRA_ARGS="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
    4) BENCH=browsecomp_plus; CTX=131072 ;;
    5) BENCH=swebench       ; CTX=131072 ;;
esac

export SGLANG_CONTEXT_LENGTH=$CTX
OUT_DIR="${OUT_ROOT}/${BENCH}"
mkdir -p "$OUT_DIR"

echo "=== Phase A ablation: ${BENCH} (n=${N_TASKS}, k=${K_VALUES}, ctx=${CTX}) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task: $SLURM_ARRAY_TASK_ID  Port: $PORT  Output: $OUT_DIR"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length $CTX \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!
trap "kill $SGLANG_PID 2>/dev/null || true" EXIT

echo "Waiting for SGLang server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: SGLang failed to start"
        exit 1
    fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

# Run all 4 ablation modes in one process to share SGLang warm-up.
# Each mode writes to its own jsonl (run_experiments.py keys output filename
# by topology name). Resume logic skips completed tasks if rerun.
# Order: most-info -> least-info, so partial completions still produce
# a meaningful comparison if wall time runs out.
ABLATION_TOPOS="decentralized_answer_only decentralized_truncate300 decentralized_truncate100 decentralized_empty"

python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies $ABLATION_TOPOS \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA_ARGS \
    || echo "WARNING: ${BENCH} ablation had errors"

echo "=== Done: ${BENCH} ablation at $(date) ==="
ls -lh "$OUT_DIR"/*.jsonl 2>/dev/null
