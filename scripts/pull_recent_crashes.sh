#!/bin/bash
# Pull all recent crash logs from the cluster for local analysis.
# Targets the most recent ~24h of bcb_abl, a5k_mscan_*, a5k_joint_*,
# and a5k_midstream_* logs. Single SSH auth round-trip.
#
# Usage: bash mas-energy/scripts/pull_recent_crashes.sh

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

mkdir -p "${LOCAL_LOGS}"

# Find logs modified within last 24 hours on cluster, then rsync those.
# Use find -newer trick: get a list, then rsync just those files.
echo "[1] Listing recent logs on cluster..."
RECENT=$(ssh ${SSH_OPTS} "${REMOTE}" \
    "find ${REMOTE_LOGS} -maxdepth 1 -type f \\( -name 'bcb_abl_*' -o -name 'a5k_mscan_*' -o -name 'a5k_joint_*' -o -name 'a5k_midstream_*' \\) -mtime -1 -printf '%f\\n' 2>/dev/null | sort")
n_files=$(echo "$RECENT" | grep -v '^$' | wc -l | tr -d ' ')
echo "  Found ${n_files} log files modified in last 24h"

if [ "$n_files" -eq 0 ]; then
    echo "  No recent logs to pull."
    exit 0
fi

# Build a temp list and rsync from-files
TMP_LIST=$(mktemp)
echo "$RECENT" | grep -v '^$' > "$TMP_LIST"

echo "[2] Pulling logs..."
rsync -avz --files-from="$TMP_LIST" \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_LOGS}/" \
    "${LOCAL_LOGS}/"
rm -f "$TMP_LIST"

# Summary by job ID
echo
echo "[3] Summary of pulled logs by job ID:"
ls -lt "${LOCAL_LOGS}" | grep -E "bcb_abl_|a5k_mscan_|a5k_joint_|a5k_midstream_" | head -40 | awk '{print "  " $NF " (" $5 "B)"}'

echo
echo "Done. Recent logs in ${LOCAL_LOGS}/"
