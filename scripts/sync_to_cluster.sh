#!/bin/bash
# Sync local code to the Stanford SC cluster via scdt.
#
# Usage:
#   bash scripts/sync_to_cluster.sh YOUR_CSID
#
# This pushes code/ and scripts/ to the cluster. Does NOT sync
# results, logs, or model weights.

CSID=${1:?Usage: sync_to_cluster.sh YOUR_CSID}

rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='archive/' \
    "$(dirname "$0")/../" \
    "${CSID}@scdt.stanford.edu:/atlas2/u/${CSID}/mas_project/mas-energy/"

echo ""
echo "Synced to /atlas2/u/${CSID}/mas_project/mas-energy/"
echo "Next: ssh ${CSID}@scdt.stanford.edu"
