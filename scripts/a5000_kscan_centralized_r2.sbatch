#!/bin/bash
#SBATCH --job-name=ks-cent-r2
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_cent_r2_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_cent_r2_%A_%a.err

# Centralized k-sweep at R=2 (standardized with decentralized R=2).
# Writes _R2 suffixed files to existing benchmark dirs.
# Resume skips any R=2 k-values that already exist (e.g., k=5 from R-sweep).
#
# Array: 0=QAMPARI, 1=FanOutQA, 2=WorkBench, 3=BrowseComp+, 4=SWE-bench

source /sailhome/$USER/.bashrc
conda activate mas-energy
JVER=$(python -c "import jinja2; print(jinja2.__version__)" 2>/dev/null)
if [[ "$JVER" && "$JVER" < "3.1" ]]; then
    pip install -q --upgrade "jinja2>=3.1.4"
fi
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
K_VALUES="1 2 3 5 7 10 15 20 30 50"

case $SLURM_ARRAY_TASK_ID in
    0) BENCH=qampari;        OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_qampari_v4;       EXTRA="" ;;
    1) BENCH=fanoutqa;       OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4;      EXTRA="" ;;
    2) BENCH=workbench;      OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2;     EXTRA="--workbench-path /atlas2/u/$USER/mas_project/WorkBench" ;;
    3) BENCH=browsecomp_plus; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot; EXTRA="" ;;
    4) BENCH=swebench;       OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench;         EXTRA="" ;;
esac

echo "=== Centralized R=2 k-scan: ${BENCH} k={$K_VALUES} ==="
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
    --benchmarks $BENCH \
    --topologies centralized \
    --max-react-steps $K_VALUES \
    --rounds 2 \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
    $EXTRA \
|| echo "WARNING: ${BENCH} centralized R=2 failed"

echo "=== Centralized R=2 k-scan ${BENCH} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
