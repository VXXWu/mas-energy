# SGLang helper functions for sbatch scripts.
# Source this file: source scripts/sglang_helpers.sh
#
# Usage:
#   start_sglang "Qwen/Qwen3.5-9B" 30000 "--trust-remote-code"
#   # ... run experiments ...
#   stop_sglang

SGLANG_PID=""

start_sglang() {
    # Args: model_path port [extra_args]
    local MODEL=$1
    local PORT=$2
    local EXTRA_ARGS=${3:-""}

    # Kill any existing process on this port
    if lsof -ti :$PORT > /dev/null 2>&1; then
        echo "Killing stale process on port $PORT..."
        kill $(lsof -ti :$PORT) 2>/dev/null
        sleep 5
    fi

    echo "Starting SGLang: $MODEL on port $PORT"
    python -m sglang.launch_server \
        --model-path $MODEL \
        --port $PORT \
        --tp 1 \
        --mem-fraction-static 0.85 \
        --context-length 131072 \
        --tool-call-parser qwen3_coder \
        --host 0.0.0.0 \
        $EXTRA_ARGS &
    SGLANG_PID=$!

    # Wait for inference readiness by polling the actual chat endpoint.
    # DO NOT use /health or /model_info -- these respond before inference
    # is ready, causing 500 errors on the first real request.
    echo "Waiting for SGLang inference readiness (up to 10 min)..."
    local READY=0
    for i in $(seq 1 200); do
        local RESP=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
            http://localhost:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null)
        if [ "$RESP" = "200" ]; then
            echo "SGLang ready after $((i*3))s"
            READY=1
            break
        fi
        # Check if process died
        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo "ERROR: SGLang process died during startup"
            return 1
        fi
        sleep 3
    done

    if [ $READY -eq 0 ]; then
        echo "ERROR: SGLang failed to start after 10 min"
        kill $SGLANG_PID 2>/dev/null
        return 1
    fi
    return 0
}

stop_sglang() {
    if [ -n "$SGLANG_PID" ]; then
        kill $SGLANG_PID 2>/dev/null
        wait $SGLANG_PID 2>/dev/null
        SGLANG_PID=""
    fi
}
