#!/bin/bash
#SBATCH --job-name=a5000-ctx
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5000_context_stress_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5000_context_stress_%j.err

# Find max context length for Qwen3.5-9B on A5000 (24GB).
# Tries starting SGLang at decreasing context lengths until one works,
# then sends a prompt that fills the full context to confirm it's usable.

source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=30000

echo "=== A5000 Max Context Length Test ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)"
echo "Date: $(date)"

try_context() {
    local CTX=$1
    local FRAC=$2
    echo ""
    echo "=== Trying context_length=$CTX, mem_fraction=$FRAC ==="

    # Kill any leftover server
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3

    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-9B \
        --port $PORT \
        --tp 1 \
        --mem-fraction-static $FRAC \
        --context-length $CTX \
        --trust-remote-code \
        --host 0.0.0.0 &
    local PID=$!

    for i in $(seq 1 180); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server started OK (ctx=$CTX, frac=$FRAC) after ${i}s"

            # Send a prompt that fills ~90% of the context
            local TARGET_TOKENS=$((CTX * 9 / 10))
            echo "Sending prompt targeting ~$TARGET_TOKENS tokens..."

            python3 -c "
import openai, time
client = openai.OpenAI(base_url='http://localhost:$PORT/v1', api_key='none')
chunk = 'The quick brown fox jumps over the lazy dog. '
prompt = chunk * ($TARGET_TOKENS // 10)
prompt = 'Read this document then reply with one word confirmed.\n\n' + prompt + '\n\nReply:'
try:
    t0 = time.time()
    r = client.chat.completions.create(
        model='Qwen/Qwen3.5-9B',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=20, temperature=0.0,
    )
    elapsed = time.time() - t0
    print(f'  OK: prompt={r.usage.prompt_tokens}, completion={r.usage.completion_tokens}, time={elapsed:.1f}s')
    print(f'  Response: {r.choices[0].message.content[:80]}')
except Exception as e:
    print(f'  INFERENCE FAILED: {str(e)[:200]}')
"
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
            return 0
        fi

        # Check if process died
        if ! kill -0 $PID 2>/dev/null; then
            echo "Server CRASHED (ctx=$CTX, frac=$FRAC)"
            wait $PID 2>/dev/null
            return 1
        fi

        sleep 2
    done

    echo "Server TIMEOUT (ctx=$CTX, frac=$FRAC)"
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    return 1
}

# Try from highest to lowest. Stop at first success.
# mem_fraction 0.88 = 21.1GB reserved, ~3GB free for loading
# Then test context lengths from high to low.
for CTX in 131072 98304 65536 49152 32768; do
    if try_context $CTX 0.88; then
        echo ""
        echo "=== MAX WORKING CONTEXT: $CTX tokens (mem_fraction=0.88) ==="
        break
    fi
done

echo ""
echo "=== Test complete: $(date) ==="
