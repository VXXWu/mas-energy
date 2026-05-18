#!/bin/bash
# Recover from the failed submissions of jobs 15335111 (BCB array 0) and
# 15335112 (qampari mscan). Hypothesis: the patched sbatches were never
# rsync'd to the cluster before submission, so the prior submit ran the
# OLD sbatches which still have the 360s SGLang timeout (qampari) and
# the broken OUT_ROOT=results/... path (BCB).
#
# This script does in one auth round-trip:
#   1. Pulls the .out from both crashed jobs to confirm the hypothesis
#   2. Syncs the patched sbatches to the cluster
#   3. Resubmits the affected arrays with the fixed scripts
#   4. Prints squeue snapshot
#
# Usage: bash mas-energy/scripts/recover_15335.sh

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}
REMOTE_LOGS=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/logs
REMOTE_SCRIPTS=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/scripts
LOCAL_LOGS=/Users/vincewu/Papers/energy_scaling/mas-energy/logs
LOCAL_SCRIPTS=/Users/vincewu/Papers/energy_scaling/mas-energy/scripts

CM_SOCK="/tmp/ssh-cm-${USER_REMOTE}-${HOST}.sock"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CM_SOCK} -o ControlPersist=10m"

echo "[0] SSH auth (one prompt for the entire script)..."
ssh ${SSH_OPTS} "${REMOTE}" "echo connected" || { echo "auth failed"; exit 1; }
trap 'ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true' EXIT

# ─── 1. Pull crash logs to confirm hypothesis ───
echo
echo "[1] Pulling crash logs from 15335111 (BCB array 0) + 15335112 (qampari mscan)..."
mkdir -p "${LOCAL_LOGS}"
rsync -avz -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_LOGS}/bcb_abl_15335111_*" \
    "${REMOTE}:${REMOTE_LOGS}/a5k_mscan_qampari_15335112_*" \
    "${LOCAL_LOGS}/" 2>/dev/null

# Show the .out for both — they're tiny on crashes, useful diagnostic
echo
echo "  --- BCB array 0 .out ---"
[ -f "${LOCAL_LOGS}/bcb_abl_15335111_0.out" ] && cat "${LOCAL_LOGS}/bcb_abl_15335111_0.out"
echo "  --- qampari array 0 .out ---"
[ -f "${LOCAL_LOGS}/a5k_mscan_qampari_15335112_0.out" ] && cat "${LOCAL_LOGS}/a5k_mscan_qampari_15335112_0.out"

# ─── 2. Sync patched sbatches ───
echo
echo "[2] Syncing patched sbatches to cluster..."
rsync -avz -e "ssh ${SSH_OPTS}" \
    "${LOCAL_SCRIPTS}/run_bigcodebench_ablation.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_qampari.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_swebench.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_math.sbatch" \
    "${LOCAL_SCRIPTS}/submit_pending.sh" \
    "${LOCAL_SCRIPTS}/diagnose_crashes.sh" \
    "${REMOTE}:${REMOTE_SCRIPTS}/"

# ─── 3. Verify the patch IS now on the cluster (last-100-line sanity check) ───
echo
echo "[3] Verifying patches are live on cluster..."
ssh ${SSH_OPTS} "${REMOTE}" "
    echo '  qampari mscan health-check (should say 600):'
    grep -m1 'for i in \$(seq 1' ${REMOTE_SCRIPTS}/a5000_mscan_qampari.sbatch
    echo '  BCB sbatch OUT_ROOT (should start with mas-energy/):'
    grep -m1 OUT_ROOT= ${REMOTE_SCRIPTS}/run_bigcodebench_ablation.sbatch
"

# ─── 4. Resubmit with the patched sbatches ───
echo
echo "[4] Resubmitting failed arrays with patched sbatches..."
ssh ${SSH_OPTS} "${REMOTE}" "
    cd /atlas2/u/${USER_REMOTE}/mas_project
    sbatch --array=0 ${REMOTE_SCRIPTS}/run_bigcodebench_ablation.sbatch
    sleep 2
    sbatch ${REMOTE_SCRIPTS}/a5000_mscan_qampari.sbatch
    sleep 2
    echo '--- queue snapshot ---'
    squeue -u ${USER_REMOTE} --format='%.18i %.9P %.10j %.8u %.2t %.10M %R'
"

echo
echo "Done. Wait ~30 min, then 'bash mas-energy/scripts/pull_results.sh' to pull progress."
