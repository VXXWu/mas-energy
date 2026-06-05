#!/bin/bash
#SBATCH --job-name=lat-train
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latent_train_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latent_train_%j.err

# Phase B training: latent encoder + adapter via KL distillation.
# Frozen Qwen3.5-9B backbone (~17GB bf16); trainable encoder+adapter (~few MB).
# Single A5000 24GB; batch_size=1 fits comfortably with peer up to 2048 tokens.
#
# Submit:
#   sbatch mas-energy/scripts/train_latent_encoder.sbatch
#
# Override hyperparams via env:
#   sbatch --export=ALL,K=16,EPOCHS=3,LR=5e-5 <this>
#
# Smoke test on a small subset:
#   sbatch --export=ALL,MAX_PAIRS=100,EPOCHS=1 <this>

set -e
source /sailhome/$USER/.bashrc
conda activate mas-energy

python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || pip install 'sympy==1.13.3' --no-deps --quiet
python -c "import jinja2; assert tuple(map(int, jinja2.__version__.split('.')[:2])) >= (3, 1)" 2>/dev/null \
    || pip install 'jinja2>=3.1.4,<3.2' --no-deps --quiet

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
# Reproducibility
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

K=${K:-8}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_PAIRS=${MAX_PAIRS:-}
# Defaults assume A6000 (48G) for full-fidelity training.
# On A5000 (24G), drop both: MAX_PEER=1024 MAX_TARGET=256 GRAD_CKPT=1
MAX_PEER=${MAX_PEER:-2048}
MAX_TARGET=${MAX_TARGET:-512}
# GRAD_CKPT=1 enables backbone gradient checkpointing (slower, lower memory).
# Default off (faster) — assumes A6000. Set GRAD_CKPT=1 for A5000.
GRAD_CKPT=${GRAD_CKPT:-0}

OUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/latent_encoder/k${K}_epochs${EPOCHS}
mkdir -p "$OUT_DIR"

PAIRS_PATH=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/training_pairs.jsonl
mkdir -p "$(dirname $PAIRS_PATH)"

# Regenerate training pairs from cluster's transcripts. Always runs (cheap;
# scans local jsonls and writes one output file). This makes the sbatch
# self-contained — sync_to_cluster.sh excludes results/ by design, so we
# can't push a locally-generated training_pairs.jsonl to the cluster anyway.
echo "Regenerating training_pairs.jsonl from cluster transcripts..."
cd /atlas2/u/$USER/mas_project
python mas-energy/code/latent_pilot/extract_training_pairs.py \
    --source-roots \
        mas-energy/results/a5000_latent_transcripts \
        mas-energy/results/a5000_math_pilot \
        mas-energy/results/a5000_transcripts_qampari \
    --output "$PAIRS_PATH" \
    || { echo "ERROR: pair extraction failed"; exit 1; }

if [ ! -f "$PAIRS_PATH" ]; then
    echo "ERROR: training_pairs.jsonl still missing after regen"
    exit 1
fi

EXTRA_ARGS=""
if [ -n "$MAX_PAIRS" ]; then EXTRA_ARGS="--max-pairs $MAX_PAIRS"; fi
if [ "$GRAD_CKPT" = "0" ]; then EXTRA_ARGS="$EXTRA_ARGS --no-gradient-checkpointing"; fi

echo "=== Phase B: latent encoder training ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Hyperparams: k=$K epochs=$EPOCHS lr=$LR batch=$BATCH_SIZE max_peer=$MAX_PEER"
echo "Pairs:  $PAIRS_PATH ($(wc -l < $PAIRS_PATH) records)"
echo "Output: $OUT_DIR"

cd /atlas2/u/$USER/mas_project/mas-energy/code

python latent_pilot/train_latent_encoder.py \
    --training-pairs "$PAIRS_PATH" \
    --backbone Qwen/Qwen3.5-9B \
    --output-dir "$OUT_DIR" \
    --k $K \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --max-peer-tokens $MAX_PEER \
    --max-target-tokens $MAX_TARGET \
    --dtype bfloat16 \
    $EXTRA_ARGS

echo "=== Training complete: $(date) ==="
ls -lh "$OUT_DIR"
