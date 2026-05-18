#!/bin/bash
#SBATCH --job-name=fanoutqa-ablation
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/fanoutqa_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/fanoutqa_%A_%a.err

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

TOPOLOGIES=(sas independent centralized decentralized hybrid)
TOPO=${TOPOLOGIES[$SLURM_ARRAY_TASK_ID]}
PORT=$((30000 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-50}

# Ablation design: topology-specific compute levels
# SAS/Independent: vary k (agent depth)
# Centralized/Hybrid: vary R (orchestrator rounds), fixed k=3
# Decentralized: vary debate rounds (1-3), fixed k=3
case $TOPO in
    sas|independent)
        LEVELS=("k=2"  "k=5"  "k=10")
        STEPS_LIST=(2 5 10)
        ROUNDS_LIST=(0 0 0)  # unused
        ;;
    centralized|hybrid)
        LEVELS=("R=1,k=3" "R=3,k=3" "R=5,k=3")
        STEPS_LIST=(3 3 3)
        ROUNDS_LIST=(1 3 5)
        ;;
    decentralized)
        LEVELS=("D=1,k=3" "D=2,k=3" "D=3,k=3")
        STEPS_LIST=(3 3 3)
        ROUNDS_LIST=(1 2 3)  # debate rounds (total = debate + 1 init)
        ;;
esac

echo "=== FanOutQA Ablation: ${TOPO} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "N_TASKS: $N_TASKS"
echo "Levels: ${LEVELS[*]}"
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
        --context-length 131072 \
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

for idx in 0 1 2; do
    S=${STEPS_LIST[$idx]}
    R=${ROUNDS_LIST[$idx]}
    echo ""
    echo "--- ${TOPO} / ${LEVELS[$idx]} ---"
    echo "Start: $(date)"

    # Restart SGLang if our process died
    if ! kill -0 $SGLANG_PID 2>/dev/null || ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang server dead, restarting..."
        kill $SGLANG_PID 2>/dev/null
        wait $SGLANG_PID 2>/dev/null
        if lsof -ti :$PORT > /dev/null 2>&1; then
            kill $(lsof -ti :$PORT) 2>/dev/null
            sleep 3
        fi
        start_sglang || exit 1
    fi

    ROUNDS_ARG=""
    if [ "$R" -gt 0 ]; then
        ROUNDS_ARG="--rounds $R"
    fi

    python run_experiments.py \
        --model toy \
        --benchmarks fanoutqa \
        --topologies $TOPO \
        --max-react-steps $S \
        $ROUNDS_ARG \
        --n-tasks $N_TASKS --n-reps 1 \
        --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/fanoutqa_v3 \
        --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: ${TOPO}/${LEVELS[$idx]} failed, continuing..."

    echo "Done ${TOPO}/${LEVELS[$idx]}: $(date)"
done

echo "=== FanOutQA ablation complete for ${TOPO} ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
