#!/bin/bash
#SBATCH --job-name=latent-trans
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_transcripts_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_transcripts_%A_%a.err

# Bulk transcript collection for the latent inter-agent communication study.
#
# Six array tasks, ONE BENCHMARK EACH, run in parallel (no %N throttle).
# Each gets its own A5000 (24G constraint). SLURM places them on whatever
# a5000 nodes are free.
#
# Per array task: Independent + Decentralized topologies, --save-transcripts ON.
#   Indep = paired control (no debate rounds, all decode is task-essential).
#   Decent = debate-round responses (the training target for latent compression).
#
# All standard accuracy/energy/token fields are still written by run_experiments.py
# (correct, loose_accuracy, gpu_dynamic_energy_joules, total_tokens, ...). The
# only addition from --save-transcripts is the response text + request_messages
# inside call_records — the records remain compatible with main-study analysis
# pipelines and can be merged into per-benchmark results if desired.
#
# Output: results/a5000_latent_transcripts/<bench>/Qwen_Qwen3.5-9B_<topo>_k<K>.jsonl
# Sizes:  prose benchmarks ~5-15MB; tool/code benchmarks ~30-80MB per topology
#
# Why all 6 benchmarks (including tool/code-heavy):
#   prose-heavy (qampari/fanoutqa/math/browsecomp) = positive cases for the
#     redundancy hypothesis (n-gram overlap heuristics designed for prose).
#   tool-heavy (workbench) and code-heavy (swebench) = negative-case calibration.
#     If compression ceiling drops to <10% for these, that scopes the claim
#     to prose-MAS rather than over-claiming universality.
#
# Submit:
#   sbatch mas-energy/scripts/collect_latent_transcripts.sbatch
#
# Override n_tasks for a quick smoke test:
#   sbatch --export=ALL,N_TASKS=20 mas-energy/scripts/collect_latent_transcripts.sbatch

set -e

# Stagger launch by array index. With 6 tasks starting concurrently, contention
# happens on (a) the shared HF cache when loading ~17GB of Qwen3.5-9B weights,
# (b) the conda env's pip during self-heal installs, (c) SGLang's torch.dist init.
# 90s per index gives task 0 enough head-start to load + finish self-heal before
# task 1 hits the same paths. Total stagger: 6×90s ≈ 9 min, trivial vs 10h run.
STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then
    echo "Staggering: sleeping ${STAGGER}s (array idx $SLURM_ARRAY_TASK_ID) to avoid HF/pip/SGLang contention..."
    sleep $STAGGER
fi

source /sailhome/$USER/.bashrc
conda activate mas-energy

# Self-heal env drift
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Per-array-task port to avoid collisions when jobs co-locate on a node
PORT=$((22000 + SLURM_ARRAY_JOB_ID % 4000 * 10 + SLURM_ARRAY_TASK_ID))

OUT_ROOT=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_latent_transcripts

# Uniform experimental parameters: N_TASKS=100, k=10, R=2 (config default).
# Only context length varies (serving-side ceiling, not experimental).
# Override per-benchmark n_tasks via env var if needed.
N_TASKS=${N_TASKS:-100}
K_VALUES="10"

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

echo "=== Latent transcript collection: ${BENCH} (n=${N_TASKS}, k=${K_VALUES}, ctx=${CTX}) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task: $SLURM_ARRAY_TASK_ID  Port: $PORT  Output: $OUT_DIR"
echo "Date: $(date)"

# Free port if a stale process holds it
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

# Independent + Decentralized in one process so SGLang stays warm.
# All standard run_experiments.py output fields are preserved; --save-transcripts
# only adds response/request_messages inside call_records.
python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies independent decentralized \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA_ARGS \
    || echo "WARNING: ${BENCH} run had errors"

echo "=== Done: ${BENCH} at $(date) ==="
ls -lh "$OUT_DIR"/*.jsonl 2>/dev/null
