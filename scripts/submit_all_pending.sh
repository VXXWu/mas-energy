#!/bin/bash
# Sync the patched sbatches and submit ALL pending jobs in one auth round-trip.
# Bundles every pending submission identified in the May-2..May-4 work window:
#   - BCB ablation array 0  (Decentralized; failed jinja2 in 15335219, now fixed)
#   - qampari mscan          (failed jinja2 in 15335221, now fixed)
#   - swebench mscan 4,5     (Decent M=4, M=5 — never landed)
#   - math mscan             (new sbatch, never submitted)
#   - 5 midstream sbatches   (new — tests R-as-compute claim against decent_midstream)
#
# All sbatches now have:
#   - 20-min SGLang health-check window (was 6 min)
#   - jinja2 force-reinstall + post-verify (soft pip install was no-op'ing
#     due to corrupted ~etuptools metadata in cluster env)
#   - 2-min stagger per array task
#
# Usage: bash mas-energy/scripts/submit_all_pending.sh

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}
REMOTE_SCRIPTS=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/scripts
LOCAL_SCRIPTS=/Users/vincewu/Papers/energy_scaling/mas-energy/scripts

CM_SOCK="/tmp/ssh-cm-${USER_REMOTE}-${HOST}.sock"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CM_SOCK} -o ControlPersist=10m"

echo "[0] SSH auth (one prompt)..."
ssh ${SSH_OPTS} "${REMOTE}" "echo connected" || { echo "auth failed"; exit 1; }
trap 'ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true' EXIT

# ─── 1. Sync patched sbatches ───
echo
echo "[1] Syncing patched sbatches to cluster..."
rsync -avz -e "ssh ${SSH_OPTS}" \
    "${LOCAL_SCRIPTS}/run_bigcodebench_ablation.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_qampari.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_swebench.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_math.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_mscan_browsecomp_high.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_joint_km.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_midstream_browsecomp.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_midstream_fanoutqa.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_midstream_workbench.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_midstream_swebench.sbatch" \
    "${LOCAL_SCRIPTS}/a5000_midstream_math.sbatch" \
    "${REMOTE}:${REMOTE_SCRIPTS}/"

# ─── 2. Verify a key fix landed on the cluster ───
echo
echo "[2] Verifying patches are live on cluster..."
ssh ${SSH_OPTS} "${REMOTE}" "
    echo '  qampari mscan jinja2 fix (should say force-reinstall):'
    grep -m1 'force-reinstall' ${REMOTE_SCRIPTS}/a5000_mscan_qampari.sbatch
    echo '  BCB sbatch OUT_ROOT (should start with mas-energy/):'
    grep -m1 'OUT_ROOT=mas-energy' ${REMOTE_SCRIPTS}/run_bigcodebench_ablation.sbatch
"

# ─── 3. Submit all pending jobs ───
echo
echo "[3] Submitting all pending jobs..."
ssh ${SSH_OPTS} "${REMOTE}" "
    cd /atlas2/u/${USER_REMOTE}/mas_project
    S=${REMOTE_SCRIPTS}

    echo '--- BCB ablation array 0 (Decentralized) ---'
    sbatch --array=0 \$S/run_bigcodebench_ablation.sbatch
    sleep 2

    echo '--- qampari mscan (full array, all 6 cells) ---'
    sbatch \$S/a5000_mscan_qampari.sbatch
    sleep 2

    echo '--- swebench mscan arrays 4,5 (Decent M=4, M=5) ---'
    sbatch --array=4,5 \$S/a5000_mscan_swebench.sbatch
    sleep 2

    echo '--- math mscan (full array, all 6 cells) ---'
    sbatch \$S/a5000_mscan_math.sbatch
    sleep 2

    echo '--- midstream: 5 benchmarks ---'
    for B in workbench fanoutqa browsecomp swebench math; do
        sbatch \$S/a5000_midstream_\${B}.sbatch
        sleep 2
    done

    echo '--- BrowseComp+ high-M extension (M=7, M=10 × Cent/Decent) ---'
    sbatch \$S/a5000_mscan_browsecomp_high.sbatch
    sleep 2

    echo '--- joint (k, M) validation: 4 cells across non-qampari benches ---'
    sbatch \$S/a5000_joint_km.sbatch
    sleep 2

    echo
    echo '--- queue snapshot ---'
    squeue -u ${USER_REMOTE} --format='%.18i %.9P %.10j %.8u %.2t %.10M %R'
"

echo
echo "Done. Wait ~30 min, then run pull_results.sh to see progress."
echo "If any array task fails fast (<1 min), the jinja2 fix didn't take —"
echo "look at the .err for 'jinja2 still at X.Y' which is the new fast-fail message."
