#!/bin/bash
# Pull and analyze the crash logs for jobs 15335111 (BCB array 0) and
# 15335112 (qampari mscan, all 6 arrays). Designed as one-shot for the
# user to run from their local terminal where SSH actually works.
#
# Output: per-array summary + best-guess root cause for each failure.
# Usage: bash mas-energy/scripts/diagnose_crashes.sh

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}
REMOTE_LOGS=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/logs
LOCAL_LOGS=/Users/vincewu/Papers/energy_scaling/mas-energy/logs

CM_SOCK="/tmp/ssh-cm-${USER_REMOTE}-${HOST}.sock"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CM_SOCK} -o ControlPersist=10m"

echo "[0] SSH auth (one prompt)..."
ssh ${SSH_OPTS} "${REMOTE}" "echo connected" || { echo "auth failed"; exit 1; }
trap 'ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true' EXIT

echo "[1] Pulling crash logs..."
mkdir -p "${LOCAL_LOGS}"
rsync -avz -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_LOGS}/bcb_abl_15335111_*" \
    "${REMOTE}:${REMOTE_LOGS}/a5k_mscan_qampari_15335112_*" \
    "${LOCAL_LOGS}/" 2>/dev/null

echo
echo "============================================================"
echo "BCB ablation array 0 (job 15335111) — Decentralized BCB"
echo "============================================================"
F_OUT=${LOCAL_LOGS}/bcb_abl_15335111_0.out
F_ERR=${LOCAL_LOGS}/bcb_abl_15335111_0.err
if [ -f "$F_OUT" ]; then
    echo "[.out size=$(stat -f%z $F_OUT 2>/dev/null || stat -c%s $F_OUT)]"
    cat "$F_OUT"
    echo
    echo "--- err signature (filtered) ---"
    grep -m20 -E "ERROR|Traceback|Error|FAIL|OOM|out of memory|cancelled|killed|exit|command not found|ModuleNotFoundError|RuntimeError|FileNotFoundError|connection refused" "$F_ERR" 2>/dev/null
fi

echo
echo "============================================================"
echo "qampari mscan (job 15335112) — array tasks 0-5"
echo "============================================================"
for i in 0 1 2 3 4 5; do
    F_OUT=${LOCAL_LOGS}/a5k_mscan_qampari_15335112_${i}.out
    F_ERR=${LOCAL_LOGS}/a5k_mscan_qampari_15335112_${i}.err
    [ ! -f "$F_OUT" ] && continue
    echo
    echo "--- array $i [.out size=$(stat -f%z $F_OUT 2>/dev/null || stat -c%s $F_OUT)] ---"
    cat "$F_OUT"
    echo "[.err filtered:]"
    grep -m10 -E "ERROR|Traceback|Error|FAIL|OOM|cancelled|killed|exit|command not found|ModuleNotFoundError|RuntimeError|FileNotFoundError" "$F_ERR" 2>/dev/null
done

echo
echo "============================================================"
echo "ROOT CAUSE GUESSES"
echo "============================================================"
# Pattern match the .out files to classify failure
classify() {
    local f=$1
    local label=$2
    [ ! -f "$f" ] && { echo "$label: log file missing"; return; }
    if grep -q "command not found" "$f" "${f%.out}.err" 2>/dev/null; then
        echo "$label: ENV — python/bash not on PATH (conda activate failed)"
    elif grep -q "ModuleNotFoundError" "$f" "${f%.out}.err" 2>/dev/null; then
        echo "$label: ENV — required python module missing (sglang/sympy/jinja2)"
    elif grep -q "SGLang failed to start" "$f" 2>/dev/null; then
        echo "$label: SGLANG TIMEOUT — health-check expired (was 360s, supposedly bumped to 1200s)"
    elif grep -qi "out of memory\|OOM" "$f" "${f%.out}.err" 2>/dev/null; then
        echo "$label: OOM — GPU/RAM exhausted"
    elif grep -q "DUE TO TIME LIMIT" "${f%.out}.err" 2>/dev/null; then
        echo "$label: TIME LIMIT — exceeded SLURM --time"
    elif grep -q "FileNotFoundError\|No such file" "${f%.out}.err" 2>/dev/null; then
        echo "$label: PATH — input file not found"
    elif grep -q "AttributeError\|TypeError\|NameError" "${f%.out}.err" 2>/dev/null; then
        echo "$label: PYTHON BUG — stack trace; investigate"
    elif grep -q "Final rows" "$f" 2>/dev/null; then
        echo "$label: COMPLETED (not actually a crash)"
    else
        echo "$label: UNKNOWN — read .err manually"
    fi
}

classify "${LOCAL_LOGS}/bcb_abl_15335111_0.out" "BCB array 0"
for i in 0 1 2 3 4 5; do
    classify "${LOCAL_LOGS}/a5k_mscan_qampari_15335112_${i}.out" "qampari array $i"
done
