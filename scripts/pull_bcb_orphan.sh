#!/bin/bash
# One-shot pull for BCB ablation results that landed at the wrong cluster
# path due to the path bug in run_bigcodebench_ablation.sbatch (job 15316650
# used OUT_ROOT=results/... instead of OUT_ROOT=mas-energy/results/...).
#
# Source paths on cluster (orphan location):
#   /atlas2/u/$USER/mas_project/results/a5000_latent_transcripts/bigcodebench/   (Decent — array 0)
#   /atlas2/u/$USER/mas_project/results/a5000_phase_b2_terse/bigcodebench/       (IS, IS+min — arrays 1, 3)
#   /atlas2/u/$USER/mas_project/results/a5000_phase_a_ablation/bigcodebench/     (Decent-min — array 2)
#
# Lands at canonical local path under mas-energy/results/<...>/bigcodebench/.
# After this, the main pull_results.sh handles future runs (sbatch is fixed).
#
# Usage: bash mas-energy/scripts/pull_bcb_orphan.sh

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}
ORPHAN=/atlas2/u/${USER_REMOTE}/mas_project/results
LOCAL=/Users/vincewu/Papers/energy_scaling/mas-energy/results

CM_SOCK="/tmp/ssh-cm-${USER_REMOTE}-${HOST}.sock"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CM_SOCK} -o ControlPersist=10m"

echo "[0] Establishing SSH connection..."
ssh ${SSH_OPTS} "${REMOTE}" "echo connected" || { echo "auth failed"; exit 1; }
trap 'ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true' EXIT

for sub in a5000_latent_transcripts a5000_phase_b2_terse a5000_phase_a_ablation; do
    src="${ORPHAN}/${sub}/bigcodebench"
    dst="${LOCAL}/${sub}/bigcodebench"
    mkdir -p "${dst}"
    echo "  ${sub}/bigcodebench/ ..."
    rsync -avz \
        -e "ssh ${SSH_OPTS}" \
        --include='*/' \
        --include='*.jsonl' \
        --include='*.json' \
        --exclude='*' \
        "${REMOTE}:${src}/" "${dst}/" 2>/dev/null \
        || echo "    (skipped: ${src} not found)"
done

echo
echo "=== Resulting BCB jsonls ==="
for sub in a5000_latent_transcripts a5000_phase_b2_terse a5000_phase_a_ablation; do
    for f in "${LOCAL}/${sub}/bigcodebench/"*.jsonl; do
        [ -f "$f" ] && echo "  rows=$(wc -l < $f)  $(echo $f | sed 's|.*/results/||')"
    done
done

echo
echo "Done. Once these are pulled, the main pull_results.sh will handle"
echo "future runs (run_bigcodebench_ablation.sbatch path bug fixed)."
