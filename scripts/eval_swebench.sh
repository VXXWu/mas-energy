#!/bin/bash
# Evaluate SWE-bench patches from experiment JSONL files.
# Requires: Docker/OrbStack running, eval_venv with swebench installed.
#
# Usage:
#   bash scripts/eval_swebench.sh [results_dir]
#   bash scripts/eval_swebench.sh ../results/swebench_v1
#
# Steps:
#   1. Extracts patches from JSONL -> predictions JSON
#   2. Cleans incomplete evaluation log dirs (from prior killed runs)
#   3. Evaluates each topology's predictions via SWE-bench Docker harness
#   4. Prints final summary

set -e

RESULTS_DIR=${1:-"$(dirname "$0")/../results/swebench_v1"}
EVAL_DIR="$RESULTS_DIR/eval"
EVAL_VENV="$(dirname "$0")/../eval_venv/bin/python"
LOG_BASE="logs/run_evaluation"
MAX_WORKERS=4
TIMEOUT=600

# Check prerequisites
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start OrbStack/Docker first."
    exit 1
fi

if [ ! -f "$EVAL_VENV" ]; then
    echo "ERROR: eval_venv not found at $EVAL_VENV"
    echo "Create it: python3 -m venv mas-energy/eval_venv && mas-energy/eval_venv/bin/pip install swebench datasets"
    exit 1
fi

echo "=== Step 1: Extract predictions ==="
$EVAL_VENV "$(dirname "$0")/../code/eval_swebench.py" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$EVAL_DIR" \
    --skip-eval

echo ""
echo "=== Step 2: Clean incomplete log dirs ==="
# SWE-bench skips instances that have a log dir, even if empty.
# Remove dirs without report.json so they get retried.
for PRED in "$EVAL_DIR"/predictions_*.json; do
    RUN_ID=$(basename "$PRED" .json | sed 's/predictions_//')
    INSTANCE_LOG_DIR="$LOG_BASE/$RUN_ID"
    if [ -d "$INSTANCE_LOG_DIR" ]; then
        for d in "$INSTANCE_LOG_DIR"/*/; do
            if [ -d "$d" ] && [ ! -f "$d/report.json" ]; then
                echo "  Removing incomplete: $(basename "$d")"
                rm -rf "$d"
            fi
        done
    fi
done

echo ""
echo "=== Step 3: Evaluate ==="
for PRED in "$EVAL_DIR"/predictions_*.json; do
    RUN_ID=$(basename "$PRED" .json | sed 's/predictions_//')
    N_PREDS=$(python3 -c "import json; print(len(json.load(open('$PRED'))))")

    if [ "$N_PREDS" -eq 0 ]; then
        echo "Skipping $RUN_ID (0 predictions)"
        continue
    fi

    echo ""
    echo "--- Evaluating: $RUN_ID ($N_PREDS predictions) ---"
    $EVAL_VENV -m swebench.harness.run_evaluation \
        --predictions_path "$PRED" \
        --run_id "$RUN_ID" \
        --max_workers $MAX_WORKERS \
        --timeout $TIMEOUT \
        --cache_level env 2>&1 | grep -E 'Running|Evaluation|instances|resolved|All|skipping|Cleaning|error'
done

echo ""
echo "=== Step 4: Summary ==="
for PRED in "$EVAL_DIR"/predictions_*.json; do
    RUN_ID=$(basename "$PRED" .json | sed 's/predictions_//')
    INSTANCE_LOG_DIR="$LOG_BASE/$RUN_ID"

    if [ ! -d "$INSTANCE_LOG_DIR" ]; then
        echo "$RUN_ID: not evaluated"
        continue
    fi

    python3 -c "
import json, os
log_dir = '$INSTANCE_LOG_DIR'
passed, failed = [], []
for d in os.listdir(log_dir):
    report = os.path.join(log_dir, d)
    rf = os.path.join(report, 'report.json')
    if os.path.isdir(report) and os.path.exists(rf):
        with open(rf) as f:
            r = json.load(f)
        for iid, res in r.items():
            (passed if res.get('resolved') else failed).append(iid)
total = len(passed) + len(failed)
rate = len(passed)/total*100 if total else 0
print(f'$RUN_ID: {len(passed)}/{total} resolved ({rate:.1f}%)')
if passed:
    print(f'  Passed: {passed}')
"
done
