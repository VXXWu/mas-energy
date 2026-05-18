#!/bin/bash
#SBATCH --job-name=diff-bridge-pilot
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/diff_bridge_pilot_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/diff_bridge_pilot_%j.err

# Hybrid latent-text MAS with conditional diffusion bridge.
# Trains the bridge on existing inter-agent transcripts, then evaluates
# three conditions on QAMPARI: TextMAS, LatentMAS-W_e, LatentMAS-Diffusion.
#
# Reuses:
#   - latent_pilot.diffusion_bridge (new module, this PR)
#   - latent_pilot.train_diffusion_bridge (new, this PR)
#   - latent_pilot.eval_hybrid_diffusion (new, this PR)
#   - agentic_latentmas runners (existing)
#   - latentmas_baseline.compute_alignment (existing W_e)
#
# See LATENT_HANDOFF.md (2026-05 resumption section) for context.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH

# ---- Config ----
MODEL=Qwen/Qwen3-8B
RUN_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/diffusion_bridge/pilot_$(date +%Y%m%d)
PAIRS=/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/training_pairs_qampari.jsonl
QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/eval"

echo "=== Diffusion Bridge Pilot ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Run dir: $RUN_DIR"
echo "Model: $MODEL"

# ---- Sanity checks ----
if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
    echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
    exit 1
fi
if [ ! -f "$PAIRS" ]; then
    echo "WARNING: training pairs not found at $PAIRS"
    echo "Generating from existing transcripts..."
    cd /atlas2/u/$USER/mas_project/mas-energy/code
    python -m latent_pilot.extract_training_pairs \
        --source-roots /atlas2/u/$USER/mas_project/mas-energy/results/a5000_latent_transcripts/qampari \
        --output "$PAIRS"
fi
echo "Pairs: $(wc -l < $PAIRS) lines"

# ---- Phase 1: Train diffusion bridge ----
echo "=== Phase 1: Train diffusion bridge ==="
cd /atlas2/u/$USER/mas_project/mas-energy/code

python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS" \
    --output-dir "$RUN_DIR" \
    --epochs 2 \
    --batch-size 2 \
    --lr 3e-4 \
    --k-soft-prompts 16 \
    --m-positions 32 \
    --n-source-layers 3 \
    --n-heads 8 \
    --K-sample-eval 20 \
    --eval-every 500 \
    --save-every 2000 \
    --warmup-steps 200 \
    --dtype bfloat16 \
    --n-blocks 2

CKPT="$RUN_DIR/bridge_epoch1.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Final checkpoint $CKPT missing — training likely failed"
    exit 1
fi
echo "Trained checkpoint: $CKPT"

# ---- Phase 2: End-to-end eval (3 conditions) ----
echo "=== Phase 2: End-to-end eval on QAMPARI ==="
EVAL_OUT="$RUN_DIR/eval/qampari_pilot.jsonl"

python -m latent_pilot.eval_hybrid_diffusion \
    --model-name "$MODEL" \
    --bridge-ckpt "$CKPT" \
    --qampari-data "$QAMPARI_DIR" \
    --output "$EVAL_OUT" \
    --n-tasks 50 \
    --n-agents 3 \
    --m-latent 20 \
    --max-react-steps 5 \
    --K-sample 20 \
    --conditions text latent_we latent_diffusion \
    --dtype bfloat16

echo "Eval results: $EVAL_OUT"

# ---- Quick summary ----
python -c "
import json
from collections import defaultdict

agg = defaultdict(lambda: {'n': 0, 'f1': 0.0, 'energy': 0.0, 'wall': 0.0})
with open('$EVAL_OUT') as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('_session'):
            continue
        c = rec['condition']
        agg[c]['n'] += 1
        agg[c]['f1'] += rec['f1']
        agg[c]['energy'] += rec['energy']['gpu_dynamic_energy_joules']
        agg[c]['wall'] += rec['energy']['wall_seconds']

print()
print('=== Summary ===')
print(f'{\"condition\":>20} {\"n\":>4} {\"f1\":>7} {\"energy_J\":>12} {\"wall_s\":>10}')
for c, v in sorted(agg.items()):
    n = max(v['n'], 1)
    print(f'{c:>20} {v[\"n\"]:>4} {v[\"f1\"]/n:>7.3f} {v[\"energy\"]/n:>12.1f} {v[\"wall\"]/n:>10.1f}')
"

echo "=== Done: $(date) ==="
