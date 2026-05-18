#!/bin/bash
#SBATCH --job-name=a5k-trans
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_transcripts_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_transcripts_%A_%a.err

# Transcript-logging spot-check run on A5000.
# Runs a SMALL number of tasks per topology with --save-transcripts enabled,
# producing rich call records that include the actual request messages and
# response text for downstream visualization with analysis/visualize_trace.py.
#
# - n_tasks=5 per (topology, k) — keeps file sizes manageable (~10x larger
#   than normal due to embedded text)
# - One topology per array task to avoid SGLang state mixing
# - Single benchmark (QAMPARI) to keep wall time under 2 hours
# - Outputs to a dedicated directory (a5000_transcripts_qampari) so the main
#   energy dataset is untouched
#
# Submit:
#   sbatch mas-energy/scripts/a5000_transcripts_qampari.sbatch
#
# To extend to other benchmarks: copy this file, change BENCH and the
# output dir.

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then
    echo "auto-fixing jinja2 ($JVER -> 3.1.4)"
    pip install -q --upgrade "jinja2>=3.1.4"
fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=5
BENCH=qampari
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_transcripts_qampari

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;            K_VALUES="5";    ROUNDS_ARG="" ;;
    1) TOPO=independent;    K_VALUES="5";    ROUNDS_ARG="" ;;
    2) TOPO=centralized;    K_VALUES="5";    ROUNDS_ARG="" ;;
    3) TOPO=decentralized;  K_VALUES="5";    ROUNDS_ARG="" ;;
    4) TOPO=hybrid;         K_VALUES="5";    ROUNDS_ARG="" ;;
esac

echo "=== A5000 transcript run: ${TOPO} ${ROUNDS_ARG} (${BENCH}, n=$N_TASKS) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
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
    --context-length 98304 \
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

python run_experiments.py \
    --model toy \
    --benchmarks $BENCH \
    --topologies $TOPO \
    --max-react-steps $K_VALUES \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} ${ROUNDS_ARG} failed"

echo "=== Transcript run complete for ${TOPO} ${ROUNDS_ARG}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
