#!/bin/bash
#SBATCH --job-name=ks-fanoutqa
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_fanoutqa_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_kscan_fanoutqa_%A_%a.err

# k-scaling sweep on FanOutQA: adds new k values to existing k={2,5,10}.
# Writes to the existing a5000_fanoutqa_v4 dir; resume logic skips existing configs.
# SAS/Independent: k ∈ {1,3,7,15,20,30,50}
# Centralized/Decentralized: k ∈ {1,3,7,15,20} (k=30,50 too slow at R=2)

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
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;            K_VALUES="1 3 7 15 20 30 50"; ROUNDS_ARG="" ;;
    1) TOPO=independent;    K_VALUES="1 3 7 15 20 30 50"; ROUNDS_ARG="" ;;
    2) TOPO=centralized;    K_VALUES="1 3 7 15 20 30 50";       ROUNDS_ARG="" ;;
    3) TOPO=decentralized;  K_VALUES="1 3 7 15 20 30 50";       ROUNDS_ARG="" ;;
esac

echo "=== k-scan FanOutQA: ${TOPO} k={$K_VALUES} ==="
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
    --benchmarks fanoutqa \
    --topologies $TOPO \
    --max-react-steps $K_VALUES \
    $ROUNDS_ARG \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} failed"

echo "=== k-scan FanOutQA ${TOPO} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
