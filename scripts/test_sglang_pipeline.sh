#!/bin/bash
# End-to-end SGLang pipeline test for the mas-energy env.
# Run from an interactive session on atlas24:
#   srun --account=atlas --partition=atlas --gres=gpu:1 --nodelist=atlas24 \
#        --mem=32G --cpus-per-task=8 --time=00:30:00 --pty bash
#   bash mas-energy/scripts/test_sglang_pipeline.sh

source ~/.bashrc && conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

PORT=31999
lsof -ti :$PORT 2>/dev/null | xargs -r kill 2>/dev/null
sleep 2

echo "=== env check ==="
python -c 'import jinja2; print("jinja2:", jinja2.__version__)'
python -c 'import sympy; print("sympy:", sympy.__version__)'

echo "=== launching SGLang ==="
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 8192 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 > /tmp/sglang_test.log 2>&1 &
SGPID=$!

echo "waiting for actual inference endpoint..."
for i in $(seq 1 120); do
    RESP=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"say hi"}],"max_tokens":10}' 2>/dev/null)
    if echo "$RESP" | grep -q '"content"'; then
        echo "PIPELINE OK after ${i}s"
        echo "$RESP" | head -c 400
        echo
        kill $SGPID 2>/dev/null
        wait $SGPID 2>/dev/null
        exit 0
    fi
    if echo "$RESP" | grep -q 'jinja2'; then
        echo "JINJA2 STILL BROKEN:"
        echo "$RESP"
        kill $SGPID 2>/dev/null
        wait $SGPID 2>/dev/null
        exit 1
    fi
    sleep 2
done

echo "TIMEOUT — sglang did not become ready"
echo "=== last 40 lines of sglang log ==="
tail -40 /tmp/sglang_test.log
kill $SGPID 2>/dev/null
wait $SGPID 2>/dev/null
exit 1
