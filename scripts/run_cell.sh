#!/usr/bin/env bash
# Run experiments on one GPU: start an SGLang server for the model, wait until it
# is healthy, run run_experiments.py with the given benchmark / topology /
# parameters, shut the server down.
#
#   scripts/run_cell.sh <model-key> [--dry-run] <run_experiments.py args...>
#
# One cell:
#   scripts/run_cell.sh toy --benchmarks swebench --topologies decentralized \
#       --max-react-steps 10 --rounds 2 --n-agents 3 --n-tasks 50 --output-dir results/demo
# A whole K-sweep (run_experiments loops over every benchmark x topology x K):
#   scripts/run_cell.sh toy --benchmarks fanoutqa workbench --topologies sas independent \
#       --max-react-steps 1 2 3 5 7 10 15 20 30 50 --n-tasks 50 --output-dir results/ksweep
#
# <model-key>: "toy" (Qwen3.5-9B) or a key of MODELS in code/config.py
# (qwen35b-a3b-fp8, gemma4-31b-qat-w4a16). Model path, tool-call parser and extra
# server flags are read from config.py; the memory fraction, context length and
# quantization below are the values used for the paper.
#
# Environment overrides: PYTHON (python), SGLANG_PORT (30000), MEM_FRACTION,
# CONTEXT_LENGTH, SGLANG_EXTRA (appended to the server command), plus the data
# locations HF_HOME, WORKBENCH_PATH, SWEBENCH_REPOS and MAS_SAVE_TRANSCRIPTS (1).
#
# The GPU must be allocated exclusively: energy is read from the device's NVML
# counter, so anything else running on it is counted. Runs are resume-safe; a
# second invocation with the same --output-dir only runs the missing tasks.
set -euo pipefail

MODEL_KEY=${1:?usage: run_cell.sh <model-key> [--dry-run] <run_experiments args...>}
shift
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE/code"

read -r MODEL_PATH PARSER EXTRA <<<"$("$PYTHON" - "$MODEL_KEY" <<'PY'
import sys, config
k = sys.argv[1]
m = config.TOY_MODEL if k == "toy" else config.MODELS[k]
print(m["model_path"], m.get("tool_call_parser") or "none", " ".join(m.get("sglang_extra_args") or []))
PY
)"

# Serving values used for the paper (appendix "Model and serving deployment").
case "$MODEL_KEY" in
  toy)                  MEM=${MEM_FRACTION:-0.88}; CTX=${CONTEXT_LENGTH:-98304}; QUANT="--max-running-requests 1" ;;
  qwen35b-a3b-fp8)      MEM=${MEM_FRACTION:-0.90}; CTX=${CONTEXT_LENGTH:-49152}; QUANT="--quantization fp8 --max-running-requests 4 --cuda-graph-max-bs 4" ;;
  gemma4-31b-qat-w4a16) MEM=${MEM_FRACTION:-0.85}; CTX=${CONTEXT_LENGTH:-65536}; QUANT="--max-running-requests 1" ;;
  *)                    MEM=${MEM_FRACTION:-0.88}; CTX=${CONTEXT_LENGTH:-65536}; QUANT="--max-running-requests 1" ;;
esac
PORT=${SGLANG_PORT:-30000}
export SGLANG_CONTEXT_LENGTH=$CTX          # client-side truncation and tool-return budgets
export MAS_SAVE_TRANSCRIPTS=${MAS_SAVE_TRANSCRIPTS:-1}
PARSER_FLAG=""; [ "$PARSER" != "none" ] && PARSER_FLAG="--tool-call-parser $PARSER"

SERVER_CMD="$PYTHON -m sglang.launch_server --model-path $MODEL_PATH --port $PORT --tp 1 \
--mem-fraction-static $MEM --context-length $CTX $PARSER_FLAG $QUANT $EXTRA ${SGLANG_EXTRA:-}"
RUNNER_CMD="$PYTHON run_experiments.py --model $MODEL_KEY --sglang-url http://localhost:$PORT/v1 $*"

echo "== server: $SERVER_CMD"
echo "== runner: (cd code && $RUNNER_CMD)"
[ "$DRY" = 1 ] && exit 0

$SERVER_CMD &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for i in $(seq 1 600); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "== server ready"; break; }
    kill -0 $SERVER 2>/dev/null || { echo "server exited during startup" >&2; exit 1; }
    [ "$i" -eq 600 ] && { echo "server did not become healthy in 20 min" >&2; exit 1; }
    sleep 2
done

"$PYTHON" run_experiments.py --model "$MODEL_KEY" --sglang-url "http://localhost:$PORT/v1" "$@"
