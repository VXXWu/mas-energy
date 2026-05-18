#!/bin/bash
# Single unified pull from Stanford cluster — merges pull_a5000_results.sh
# and pull_latent_results.sh into one script with one auth prompt.
#
# Pulls:
#   - Main study result dirs (qampari/fanoutqa/workbench/browsecomp/swebench/math)
#   - Latent pilot dirs (transcripts, phase_a_ablation, phase_b2_terse, latent_pilot,
#     latent_encoder, transcripts_pilot)
#   - All relevant logs across the project
#
# Uses SSH ControlMaster: one auth prompt at the start, all subsequent
# transfers reuse the connection. Uses rsync (not scp -r), so re-runs are
# incremental and partial pulls resume.
#
# Usage:
#   bash mas-energy/scripts/pull_results.sh

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}

REMOTE_RESULTS_ROOT=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/results
REMOTE_LOGS=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/logs

LOCAL_ROOT=/Users/vincewu/Papers/energy_scaling/mas-energy
LOCAL_RESULTS_ROOT=${LOCAL_ROOT}/results
LOCAL_LOGS=${LOCAL_ROOT}/logs

mkdir -p "${LOCAL_RESULTS_ROOT}" "${LOCAL_LOGS}"

# ─── ssh connection multiplexing ───
# One control socket; all rsync calls reuse the auth.
CM_SOCK="/tmp/ssh-cm-${USER_REMOTE}-${HOST}.sock"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CM_SOCK} -o ControlPersist=10m"

echo "[0] Establishing master SSH connection (one-time auth)..."
ssh ${SSH_OPTS} "${REMOTE}" "echo 'connected'" || {
    echo "ERROR: could not open SSH connection."
    exit 1
}
trap 'ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true' EXIT

# ─── Helper: rsync a remote results dir, accepting common file types ───
pull_results_dir() {
    local subdir="$1"
    local label="${2:-$subdir}"
    local remote_path="${REMOTE_RESULTS_ROOT}/${subdir}"
    local local_path="${LOCAL_RESULTS_ROOT}/${subdir}"
    mkdir -p "${local_path}"
    echo "  ${label}..."
    rsync -avz \
        -e "ssh ${SSH_OPTS}" \
        --include='*/' \
        --include='*.jsonl' \
        --include='*.json' \
        --include='*.bak' \
        --include='*.pt' \
        --exclude='*' \
        "${REMOTE}:${remote_path}/" \
        "${local_path}/" 2>/dev/null \
        || echo "    (skipped: ${subdir} not on cluster yet)"
}

# ─── Pull result directories ───
echo ""
echo "[1] Pulling main-study result dirs..."
for d in a5000_qampari_v4 a5000_fanoutqa_v4 a5000_workbench_v2 \
         a5000_browsecomp_pilot a5000_swebench a5000_math_pilot; do
    pull_results_dir "$d"
done

echo ""
echo "[2] Pulling latent-comm pilot dirs..."
for d in a5000_latent_transcripts a5000_phase_a_ablation a5000_phase_b2_terse \
         a5000_transcripts_pilot a5000_transcripts_qampari \
         latent_pilot latent_encoder humaneval_pilot livecodebench_pilot \
         bigcodebench_pilot predictability_axis; do
    pull_results_dir "$d"
done

# ─── Pull logs ───
echo ""
echo "[3] Pulling logs..."
rsync -avz \
    -e "ssh ${SSH_OPTS}" \
    --include='math_pilot_*' \
    --include='latent_transcripts_*' \
    --include='phase_a_abl_*' \
    --include='phase_b2_*' \
    --include='latent_train_*' \
    --include='swe_fill_*' \
    --include='empty_silent_*' \
    --include='specialist_*' \
    --include='kvshare_m_abl_*' \
    --include='kvshare_m0_qwen35_*' \
    --include='structural_qwen3*' \
    --include='structural_qwen35*' \
    --include='phase1_latent_qwen3_*' \
    --include='phase1_latent_qwen35_*' \
    --include='latentmas_qwen35_*' \
    --include='latent_noreencode_*' \
    --include='agentic_latentmas_*' \
    --include='agreegate_qwen3_*' \
    --include='agreegate_fanoutqa_*' \
    --include='e2mas_combined_*' \
    --include='coconut_pilot_*' \
    --include='concision_pilot_*' \
    --include='llm_judge_*' \
    --include='transcript_dup_pilot_*' \
    --include='he_pilot_*' \
    --include='he_sas_pilot_*' \
    --include='he_decent_pilot_*' \
    --include='he_cal_*' \
    --include='he_abl_*' \
    --include='lcb_cal_*' \
    --include='lcb_abl_*' \
    --include='bcb_cal_*' \
    --include='bcb_abl_*' \
    --include='pred_ax_*' \
    --include='a5k_kscan_*' \
    --include='a5k_rscan_*' \
    --include='a5k_mscan_*' \
    --include='fill_*' \
    --include='is_minimal_*' \
    --include='brc_decent_extend_*' \
    --include='swe_decent_extend_*' \
    --include='minimal_*' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_LOGS}/" \
    "${LOCAL_LOGS}/"

# ─── Summary ───
echo ""
echo "==== Summary ===="
echo "Latest result jsonls (any dir):"
find "${LOCAL_RESULTS_ROOT}" -name '*.jsonl' -type f -mtime -7 2>/dev/null \
    | xargs -I {} stat -f '%Sm %z %N' -t '%Y-%m-%d_%H:%M' {} 2>/dev/null \
    | sort -r | head -10 | awk '{ printf "  %s  %10s  %s\n", $1, $2, $3 }'

echo ""
echo "Latest encoder checkpoints:"
find "${LOCAL_RESULTS_ROOT}/latent_encoder" -name '*.pt' -type f 2>/dev/null \
    | xargs -I {} stat -f '%Sm %z %N' -t '%Y-%m-%d_%H:%M' {} 2>/dev/null \
    | sort -r | head -5 | awk '{ printf "  %s  %10s  %s\n", $1, $2, $3 }'

echo ""
echo "Latest logs:"
ls -lat "${LOCAL_LOGS}"/*.out 2>/dev/null | head -5 | awk '{ printf "  %s  %s\n", $9, $6" "$7" "$8 }'

echo ""
echo "Done."
