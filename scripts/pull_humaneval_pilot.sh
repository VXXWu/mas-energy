#!/bin/bash
# One-shot pull for the HumanEval+ saturation pilot. The main pull_results.sh
# uses an allow-list that doesn't include humaneval_pilot/ or he_pilot_* logs.
#
# Usage:
#   bash mas-energy/scripts/pull_humaneval_pilot.sh

set -e

REMOTE=vincewu8@scdt.stanford.edu
REMOTE_BASE=/atlas2/u/vincewu8/mas_project/mas-energy
LOCAL_BASE=/Users/vincewu/Papers/energy_scaling/mas-energy

mkdir -p "${LOCAL_BASE}/results/humaneval_pilot" "${LOCAL_BASE}/logs"

echo "[1/2] Pulling humaneval_pilot results..."
rsync -avz \
    "${REMOTE}:${REMOTE_BASE}/results/humaneval_pilot/" \
    "${LOCAL_BASE}/results/humaneval_pilot/" \
    || echo "  (no results dir on cluster yet — pilot may not have produced output)"

echo ""
echo "[2/2] Pulling he pilot logs (sas/decent)..."
rsync -avz \
    --include='he_pilot_*' \
    --include='he_sas_pilot_*' \
    --include='he_decent_pilot_*' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_BASE}/logs/" \
    "${LOCAL_BASE}/logs/" \
    || echo "  (no pilot logs on cluster — pilot was never submitted?)"

echo ""
echo "==== Summary ===="
echo "Local humaneval_pilot files:"
ls -la "${LOCAL_BASE}/results/humaneval_pilot/" 2>/dev/null || echo "  (empty)"
echo ""
echo "Local he_pilot logs:"
ls -la "${LOCAL_BASE}/logs/"he_pilot_* 2>/dev/null || echo "  (none)"
echo ""
echo "Pilot summary line (from .out log if present):"
grep -E "pass@1|verdict" "${LOCAL_BASE}/logs/"he_pilot_*.out 2>/dev/null | tail -5 || echo "  (no log yet)"

echo ""
echo "Done."
