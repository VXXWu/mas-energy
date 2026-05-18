#!/bin/bash
#SBATCH --job-name=rs-qa-k10
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_rscan_k10_qampari_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_rscan_k10_qampari_%A_%a.err

# R-scaling sweep on QAMPARI at k=10, n=100.
# Array: 0-4 = centralized R=1..5, 5-9 = decentralized R=1..5
# Resume logic skips completed tasks.

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then
    pip install -q --upgrade "jinja2>=3.1.4"
fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=100
K=10
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_qampari_v4

TOPO_IDX=$((SLURM_ARRAY_TASK_ID / 5))
R=$(( (SLURM_ARRAY_TASK_ID % 5) + 1 ))

case $TOPO_IDX in
    0) TOPO=centralized ;;
    1) TOPO=decentralized ;;
esac

echo "=== R-scan QAMPARI: ${TOPO} k=${K} R=${R} n=${N_TASKS} ==="
echo "Node: $(hostname)"
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
    --benchmarks qampari \
    --topologies $TOPO \
    --max-react-steps $K \
    --rounds $R \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
     \
|| echo "WARNING: QAMPARI ${TOPO} k=${K} R=${R} failed"

echo "=== R-scan QAMPARI ${TOPO} R=${R} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
