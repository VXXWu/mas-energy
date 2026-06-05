#!/bin/bash
#SBATCH --job-name=a5k-swe
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-2%1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_swe_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_swe_%A_%a.err

# SWE-bench on A5000. Context 49K fits all SWE-bench calls (max ~20K prompt).
# Array tasks (sequential):
#   0: sas
#   1: centralized R=1
#   2: decentralized

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then echo "auto-fixing jinja2 ($JVER -> 3.1.4)"; pip install -q --upgrade "jinja2>=3.1.4"; fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SWEBENCH_REPOS=/atlas2/u/$USER/mas_project/swebench_repos

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;            ROUNDS_ARG="" ;;
    1) TOPO=centralized;    ROUNDS_ARG="--rounds 1" ;;
    2) TOPO=decentralized;  ROUNDS_ARG="" ;;
esac

echo "=== SWE-bench A5000: ${TOPO} ${ROUNDS_ARG} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
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
    --benchmarks swebench \
    --topologies $TOPO \
    --max-react-steps 50 \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
|| echo "WARNING: ${TOPO} ${ROUNDS_ARG} failed"

echo "=== Done: ${TOPO} ${ROUNDS_ARG}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
