#!/bin/bash
#SBATCH --job-name=ks-swe-hk
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_swebench_highk_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_swebench_highk_%A_%a.err

# SWE-bench high-k extension: SAS and Independent at k=100,200.
# Array: 0=SAS k=100, 1=SAS k=200, 2=Independent k=100, 3=Independent k=200

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then
    pip install -q --upgrade "jinja2>=3.1.4"
fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SWEBENCH_REPOS=/atlas2/u/$USER/mas_project/swebench_repos
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;         K=100 ;;
    1) TOPO=sas;         K=200 ;;
    2) TOPO=independent; K=100 ;;
    3) TOPO=independent; K=200 ;;
esac

echo "=== SWE-bench high-k: ${TOPO} k=${K} ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Stagger model loading to avoid concurrent shard OOM
STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
echo "Staggering start by ${STAGGER}s to avoid concurrent model loads"
sleep $STAGGER

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
    --max-react-steps $K \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} k=${K} failed"

echo "=== SWE-bench ${TOPO} k=${K} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
