#!/bin/bash
#SBATCH --job-name=mas-pareto
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/pareto_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/pareto_%A_%a.err

source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

TOPOLOGIES=(sas independent centralized decentralized hybrid)
TOPO=${TOPOLOGIES[$SLURM_ARRAY_TASK_ID]}
PORT=$((30000 + SLURM_ARRAY_TASK_ID))
if [ -n "$STEPS_OVERRIDE" ]; then
    IFS=',' read -ra STEPS <<< "$STEPS_OVERRIDE"
else
    STEPS=(2 5 10 20)
fi

echo "=== MAS Pareto Sweep: ${TOPO} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"

# Kill any zombie SGLang from a previous job on this port
if lsof -ti :$PORT > /dev/null 2>&1; then
    echo "WARNING: stale process on port $PORT, killing..."
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

start_sglang() {
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-9B \
        --port $PORT \
        --tp 1 \
        --mem-fraction-static 0.85 \
        --context-length 65536 \
        --tool-call-parser qwen3_coder \
        --trust-remote-code \
        --host 0.0.0.0 &
    SGLANG_PID=$!

    echo "Waiting for SGLang server..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "SGLang ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: SGLang failed to start"
    kill $SGLANG_PID 2>/dev/null
    return 1
}

start_sglang || exit 1

cd /atlas2/u/$USER/mas_project/mas-energy/code

for S in "${STEPS[@]}"; do
    echo ""
    echo "--- ${TOPO} / max_react_steps=${S} ---"
    echo "Start: $(date)"

    # Restart SGLang if our process died (don't trust port alone -- could be zombie)
    if ! kill -0 $SGLANG_PID 2>/dev/null || ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang server dead, restarting..."
        kill $SGLANG_PID 2>/dev/null
        wait $SGLANG_PID 2>/dev/null
        # Kill any zombie on the port
        if lsof -ti :$PORT > /dev/null 2>&1; then
            kill $(lsof -ti :$PORT) 2>/dev/null
            sleep 3
        fi
        start_sglang || exit 1
    fi

    python run_experiments.py \
        --model toy \
        --benchmarks workbench \
        --topologies $TOPO \
        --max-react-steps $S \
        --n-tasks 50 --n-reps 1 \
        --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/workbench_v1 \
        --workbench-path /atlas2/u/$USER/mas_project/WorkBench \
        --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: ${TOPO}/steps=${S} failed, continuing..."

    echo "Done ${TOPO}/steps=${S}: $(date)"
done

echo "=== Pareto sweep complete for ${TOPO} ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
