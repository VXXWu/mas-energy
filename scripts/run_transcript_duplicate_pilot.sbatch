#!/bin/bash
#SBATCH --job-name=transcript-dup-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/transcript_dup_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/transcript_dup_pilot_%j.err

# Transcript-enabled pilot: measures cross-agent tool-call duplicate rates
# across all 4 benchmarks x decentralized topology, n=10 tasks each.
#
# Purpose: confirm the hypothesis that open-ended benchmarks (SWE-bench,
# BrowseComp, WorkBench) have lower duplicate rates than structured
# list-answer tasks (QAMPARI) because agents' exploration paths diverge
# naturally.
#
# Main study's a5000_transcripts_qampari/ already exists (n=5 per topology).
# This pilot adds the other 3 benchmarks at matched n=10 for a direct
# cross-benchmark comparison.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

OUT_ROOT=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_transcripts_pilot
mkdir -p "$OUT_ROOT"

cd /atlas2/u/$USER/mas_project/mas-energy/code

# Start SGLang server for Qwen3.5-9B in background (main study uses this model)
# NOTE: The main study's run_experiments.py expects SGLang already running.
# Launch it here as a background process and wait for it to come up.
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port 30000 \
    --mem-fraction-static 0.80 \
    --tp 1 &
SGLANG_PID=$!
trap "kill $SGLANG_PID 2>/dev/null || true" EXIT

# Wait for server to be healthy (up to 5 minutes)
echo "Waiting for SGLang server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:30000/health >/dev/null 2>&1; then
        echo "SGLang ready after ${i}x5s"
        break
    fi
    sleep 5
done

echo "=== Transcript-enabled duplicate-rate pilot ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Run each benchmark x decentralized with transcripts enabled.
# QAMPARI already measured; re-running it here for internal consistency.
for BENCH in qampari fanoutqa workbench browsecomp; do
    echo ""
    echo "======================================================"
    echo "Benchmark: $BENCH (decentralized, n=10, with transcripts)"
    echo "======================================================"
    python run_experiments.py \
        --model qwen35-9b \
        --benchmarks "$BENCH" \
        --topologies decentralized \
        --n-tasks 10 --n-reps 1 \
        --max-react-steps 5 \
        --save-transcripts \
        --output-dir "$OUT_ROOT" \
        || echo "  (benchmark $BENCH failed or skipped)"
done

# SWE-bench separately — its runtime per task is longer, run fewer tasks
echo ""
echo "======================================================"
echo "Benchmark: swebench (decentralized, n=5, with transcripts)"
echo "======================================================"
python run_experiments.py \
    --model qwen35-9b \
    --benchmarks swebench \
    --topologies decentralized \
    --n-tasks 5 --n-reps 1 \
    --max-react-steps 10 \
    --save-transcripts \
    --output-dir "$OUT_ROOT" \
    || echo "  (swebench failed or skipped)"

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Transcripts saved under $OUT_ROOT"
echo "After pulling, run:"
echo "  python mas-energy/code/latent_pilot/analyze_mainstudy_duplicates.py \\"
echo "      mas-energy/results/a5000_transcripts_pilot/*/*.jsonl"
