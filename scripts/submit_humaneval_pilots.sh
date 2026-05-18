#!/bin/bash
# Submit both HumanEval+ calibration pilots via ssh. No pulling, no auto-decide.
# After both jobs finish, run pull_humaneval_pilot.sh and bring results back to inspect.
#
# Usage:
#   bash mas-energy/scripts/submit_humaneval_pilots.sh

set -e

REMOTE=vincewu8@scdt.stanford.edu
REMOTE_BASE=/atlas2/u/vincewu8/mas_project/mas-energy

echo "Submitting SAS+tools pilot (k=10, n=20, ~20 min)..."
ssh "${REMOTE}" "sbatch ${REMOTE_BASE}/scripts/pilot_humaneval_sas_tools.sbatch"

echo ""
echo "Submitting Decent pilot (R=2, k=10, n=20, ~1-2h)..."
ssh "${REMOTE}" "sbatch ${REMOTE_BASE}/scripts/pilot_humaneval_decent.sbatch"

echo ""
echo "Both submitted. Check status:"
echo "  ssh ${REMOTE} 'squeue -u vincewu8'"
echo ""
echo "When both finish, pull and analyze locally:"
echo "  bash mas-energy/scripts/pull_humaneval_pilot.sh"
