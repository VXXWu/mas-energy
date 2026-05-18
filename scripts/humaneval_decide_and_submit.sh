#!/bin/bash
# After the SAS-with-tools pilot finishes on the cluster, run this locally:
#   bash mas-energy/scripts/humaneval_decide_and_submit.sh
#
# Flow:
#   1. Pulls the SAS+tools pilot result, prints pass@1 and verdict.
#   2. Submits a Decent (R=2, k=10, n=20) pilot to bound the ablation
#      ceiling. Does NOT auto-submit the full ablation — you decide based
#      on both pilot results.
#   3. Prints next-step instructions.

set -e

REMOTE=vincewu8@scdt.stanford.edu
REMOTE_BASE=/atlas2/u/vincewu8/mas_project/mas-energy
LOCAL_BASE=/Users/vincewu/Papers/energy_scaling/mas-energy
SAS_FILE="${LOCAL_BASE}/results/humaneval_pilot/Qwen_Qwen3.5-9B_sas_k10.jsonl"

echo "[1/3] Pulling SAS+tools pilot result..."
mkdir -p "${LOCAL_BASE}/results/humaneval_pilot"
rsync -avz \
    "${REMOTE}:${REMOTE_BASE}/results/humaneval_pilot/Qwen_Qwen3.5-9B_sas_k10.jsonl" \
    "${SAS_FILE}" 2>&1 | tail -3

if [ ! -f "${SAS_FILE}" ]; then
    echo ""
    echo "ERROR: SAS pilot result not on cluster yet."
    echo "  Check job status:  ssh ${REMOTE} 'squeue -u vincewu8 -n he-sas'"
    echo "  Or check logs:     ssh ${REMOTE} 'tail -30 ${REMOTE_BASE}/logs/he_sas_pilot_*.out'"
    exit 1
fi

echo ""
echo "[2/3] Computing SAS+tools pass@1..."
PASS_RATE=$(python3 -c "
import json
rows = [json.loads(l) for l in open('${SAS_FILE}')]
n = len(rows)
p = sum(1 for r in rows if r.get('correct'))
print(f'{p/n:.4f}')
")
N_TASKS=$(wc -l < "${SAS_FILE}" | tr -d ' ')
VERDICT=$(python3 -c "
p = float('${PASS_RATE}')
if p < 0.85:    print('BELOW SATURATION (full 4-cell ablation safe)')
elif p < 0.92:  print('TIGHT (essential 2 cells only)')
else:           print('NEAR-CEILING (consider LiveCodeBench pivot)')
")
echo "  SAS+tools pass@1 = ${PASS_RATE} (${N_TASKS} tasks)"
echo "  verdict (lower bound): ${VERDICT}"

echo ""
echo "[3/3] Submitting Decent (R=2, k=10, n=20) pilot to find the ceiling..."
ssh "${REMOTE}" "sbatch ${REMOTE_BASE}/scripts/pilot_humaneval_decent.sbatch"

echo ""
echo "=== Next steps ==="
echo "Decent pilot wall time: ~1-2h."
echo ""
echo "When it finishes, pull and inspect:"
echo "  bash ${LOCAL_BASE}/scripts/pull_humaneval_pilot.sh"
echo "  python3 -c \"import json; rows=[json.loads(l) for l in open('${LOCAL_BASE}/results/humaneval_pilot/Qwen_Qwen3.5-9B_decentralized_k10.jsonl')]; n=len(rows); p=sum(1 for r in rows if r.get('correct')); print(f'Decent pass@1 = {p/n:.3f} ({p}/{n})')\""
echo ""
echo "Then make the call:"
echo "  - Decent < 0.85 → run all 4 cells:   ssh ${REMOTE} 'sbatch ${REMOTE_BASE}/scripts/run_humaneval_ablation.sbatch'"
echo "  - Decent 0.85-0.95 → essential 2:    ssh ${REMOTE} 'sbatch --array=0-1 ${REMOTE_BASE}/scripts/run_humaneval_ablation.sbatch'"
echo "  - Decent > 0.95 → ceiling problem:   either run --array=0-1 as confirmatory IS-collapse check, or pivot to LiveCodeBench"
echo ""
echo "Done."
