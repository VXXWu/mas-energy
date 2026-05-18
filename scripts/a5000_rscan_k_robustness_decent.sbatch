#!/bin/bash
#SBATCH --job-name=rsk-wb-r1
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_rscan_k_robust_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_rscan_k_robust_%A_%a.err

# Robustness check for the Decent saturation curve shape: rerun the k-scan at
# R=1 (smallest non-degenerate debate budget) on WorkBench, to verify the
# (ceiling, drop, alpha) fit is not an artifact of the R=2 operating-point
# choice. Existing data has R=1 only at k=5 and k=10; this fills in the
# remaining k values.
#
# Tests whether saturation shape (drop, alpha) shifts when R changes.
# Already have at R=1: k=5 (n=50), k=10 (n=100). This array runs:
# k in {1, 2, 3, 7, 15, 20, 30, 50}, all at R=1, Decent, n=100.
# Compare resulting (ceiling, drop, alpha) fit to the published R=2 fit
# (ceiling=0.31, drop=0.42, alpha=0.53). If shape parameters differ by less
# than 0.10, R=2 choice is robust; otherwise the saturation curve is
# operating-point-specific and that should be stated in the paper.
#
# Resume logic skips already-completed tasks.

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
R=1
TOPO=decentralized
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2

# Array index → k value (k=5, 10 already exist with R=1, so skip)
K_VALUES=(1 2 3 7 15 20 30 50)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== R-robustness Decent k-scan: ${TOPO} k=${K} R=${R} n=${N_TASKS} ==="
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
    --benchmarks workbench \
    --topologies $TOPO \
    --max-react-steps $K \
    --rounds $R \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
|| echo "WARNING: WorkBench ${TOPO} k=${K} R=${R} failed"

echo "=== R-robust scan ${TOPO} k=${K} R=${R} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
