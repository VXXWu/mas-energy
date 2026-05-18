#!/bin/bash
#SBATCH --job-name=bc-thinking
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/bc_thinking_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/bc_thinking_%A_%a.err

# Thinking vs MAS ablation on BrowseComp-Plus.
# Compares enable_thinking=True (SAS, Independent) against
# the baseline enable_thinking=False results in browsecomp_pilot/.
#
# Array tasks:
#   0: sas thinking          (k=5,10)
#   1: independent thinking  (k=5,10)
#   2: sas no-thinking       (k=5,10)  -- control (should match browsecomp_pilot)
#   3: independent no-think  (k=5,10)  -- control

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Self-heal env drift: jinja2 3.0.3 breaks SGLang's --tool-call-parser path
# (transformers' apply_chat_template needs jinja2>=3.1). Pinned + --no-deps per project memory.
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=$((30000 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-50}

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;         THINKING="--thinking"; LABEL="thinking" ;;
    1) TOPO=independent; THINKING="--thinking"; LABEL="thinking" ;;
    2) TOPO=sas;         THINKING="";           LABEL="no-thinking" ;;
    3) TOPO=independent; THINKING="";           LABEL="no-thinking" ;;
esac

echo "=== BrowseComp Thinking Ablation: ${TOPO} ${LABEL} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

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
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: SGLang failed to start"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks browsecomp_plus \
    --topologies $TOPO \
    --max-react-steps 5 10 \
    $THINKING \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/browsecomp_thinking \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} ${LABEL} failed"

echo "=== Done: ${TOPO} ${LABEL}: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
