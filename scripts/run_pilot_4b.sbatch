#!/bin/bash
#SBATCH --job-name=diff-pilot-4b
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/pilot_4b_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/pilot_4b_%j.err

# Staged pilot for hybrid latent-text MAS + diffusion bridge.
#
# Defaults: Qwen3-4B + FanOutQA. The 2026-05-16 first pilot on Qwen3-4B +
# QAMPARI hit floor-level accuracy on both text and latent conditions
# (33/40 tasks scored 0 under both), making the mechanism comparison
# uninterpretable. FanOutQA has looser eval (loose accuracy is a real
# graded score, not list F1) and Qwen3-4B handles it better. QAMPARI and
# 8B remain available via env vars:
#
#   MODEL=Qwen/Qwen3-8B BENCHMARK=qampari sbatch run_pilot_4b.sbatch
#
# Stage 0: SGLang + run_experiments.py decentralized at chosen model
#          → produces call_records-format transcripts for pair extraction.
# Stage 1: agentic_latentmas mechanism validation (HF transformers).
#          Now captures per-condition response strings for diagnosis.
# Stage 1b: HARD GATE — abort if text-MAS baseline doesn't produce signal.
#          Prevents wasting GPU-hours on bridge training when the underlying
#          MAS pipeline can't tell text from latent.
# Stage 2-7: pair extraction, smoke test, bridge training, ablation, KL eval.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE:$PYTHONPATH

python -c "import sglang" 2>/dev/null || {
    echo "Installing sglang..."
    pip install -q "sglang[all]"
}

# ---- Config (env-overridable) ----
MODEL=${MODEL:-Qwen/Qwen3-4B}
BENCHMARK=${BENCHMARK:-fanoutqa}
N_TRANSCRIPT_TASKS=${N_TRANSCRIPT_TASKS:-30}
N_MECHANISM_TASKS=${N_MECHANISM_TASKS:-20}
# Minimum mean text-mas F1 to consider baseline viable for bridge training.
# Set to 0 to skip the gate.
MIN_TEXT_F1=${MIN_TEXT_F1:-0.10}
MIN_TEXT_SCORED=${MIN_TEXT_SCORED:-5}  # min tasks where text_f1 > 0

MODEL_TAG=$(echo "$MODEL" | tr '/.' '_-')
RESULTS_BASE=/atlas2/u/$USER/mas_project/mas-energy/results
PILOT_DIR=$RESULTS_BASE/diffusion_pilot_${BENCHMARK}_$(echo $MODEL | tr '/.' '_')_$(date +%Y%m%d_%H%M)
TRANSCRIPTS_DIR=$PILOT_DIR/transcripts
PAIRS_PATH=$PILOT_DIR/pairs.jsonl
BRIDGE_RUN=$PILOT_DIR/bridge_main
BRIDGE_ABLATION=$PILOT_DIR/bridge_layer1

mkdir -p "$PILOT_DIR" "$TRANSCRIPTS_DIR" "$BRIDGE_RUN" "$BRIDGE_ABLATION"
cd /atlas2/u/$USER/mas_project/mas-energy/code

echo "=== Hybrid Latent + Diffusion Bridge Pilot ==="
echo "Model:     $MODEL"
echo "Benchmark: $BENCHMARK"
echo "Pilot dir: $PILOT_DIR"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Benchmark-specific data path checks
DATA_DIR_ARG=""
if [ "$BENCHMARK" = "qampari" ]; then
    QAMPARI_DIR=/atlas2/u/$USER/mas_project/mas-energy/data/qampari/qampari_data
    if [ ! -f "$QAMPARI_DIR/dev_data.jsonl" ]; then
        echo "ERROR: QAMPARI data not found at $QAMPARI_DIR"
        exit 1
    fi
    DATA_DIR_ARG="--data-dir $QAMPARI_DIR"
fi

# ────────────────────────────────────────────────────────────────────
# Stage 0: SGLang + run_experiments → call_records transcripts
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 0: text decentralized transcripts via run_experiments ==="
PORT=$((20000 + (SLURM_JOB_ID % 4000)))
if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null || true
    sleep 3
fi

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 \
    > "$PILOT_DIR/sglang.log" 2>&1 &
SGLANG_PID=$!

echo "  Waiting for SGLang on port $PORT ..."
for i in $(seq 1 180); do
    if curl -fs "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  SGLang ready after ${i}s"
        break
    fi
    sleep 1
    if [ $i -eq 180 ]; then
        echo "ERROR: SGLang failed to come up in 3 min."
        tail -30 "$PILOT_DIR/sglang.log"
        kill $SGLANG_PID 2>/dev/null || true
        exit 1
    fi
done

python -m run_experiments \
    --model "$MODEL" \
    --benchmarks "$BENCHMARK" \
    --topologies decentralized \
    --n-tasks $N_TRANSCRIPT_TASKS \
    --n-reps 1 \
    --max-react-steps 10 \
    --rounds 1 \
    --n-agents 3 \
    --output-dir "$TRANSCRIPTS_DIR" \
    --sglang-url "http://localhost:$PORT" \
    --save-transcripts

echo "  Shutting down SGLang ..."
kill $SGLANG_PID 2>/dev/null || true
sleep 5
if lsof -ti :$PORT > /dev/null 2>&1; then
    kill -9 $(lsof -ti :$PORT) 2>/dev/null || true
fi
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader

TRANSCRIPT_FILE=$(find "$TRANSCRIPTS_DIR" -name "*decentralized*.jsonl" | head -1)
if [ -z "$TRANSCRIPT_FILE" ] || [ ! -s "$TRANSCRIPT_FILE" ]; then
    echo "ERROR: no transcripts produced."
    find "$TRANSCRIPTS_DIR" -type f
    exit 1
fi
echo "  Transcript: $TRANSCRIPT_FILE ($(wc -l < $TRANSCRIPT_FILE) records)"

# ────────────────────────────────────────────────────────────────────
# Stage 1: agentic_latentmas mechanism validation
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 1: agentic_latentmas mechanism validation ==="
python -m latent_pilot.agentic_latentmas \
    --benchmark "$BENCHMARK" \
    --n-tasks $N_MECHANISM_TASKS \
    --m-latent 20 \
    --max-react-steps 5 \
    --n-agents 3 \
    --model-name "$MODEL" \
    $DATA_DIR_ARG \
    --latent-mode kv_share

# Locate the produced file (model + benchmark tagged)
MECH_TAG=$(echo "$MODEL" | tr '/.' '_-' | sed 's/-/_/g')
MECH_JSONL=$(find "$RESULTS_BASE/latent_pilot" -name "eval_*${BENCHMARK}_agentic*.jsonl" \
    -newer "$PILOT_DIR" 2>/dev/null | head -1)
if [ -z "$MECH_JSONL" ]; then
    # Fallback: just take the newest matching
    MECH_JSONL=$(ls -t "$RESULTS_BASE/latent_pilot"/eval_*"${BENCHMARK}"_agentic*.jsonl 2>/dev/null | head -1)
fi

if [ -z "$MECH_JSONL" ] || [ ! -f "$MECH_JSONL" ]; then
    echo "ERROR: could not locate agentic_latentmas output file"
    ls "$RESULTS_BASE/latent_pilot" | grep -i "$BENCHMARK" | head
    exit 1
fi
echo "  Mechanism output: $MECH_JSONL"

# Run diagnostic tool on the output
python -m latent_pilot.diagnose_agentic_run \
    --jsonl "$MECH_JSONL" \
    --out "$PILOT_DIR/mechanism_diagnosis.json" \
    --top-n 5 || true

# ────────────────────────────────────────────────────────────────────
# Stage 1b: HARD GATE on baseline signal
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 1b: baseline signal gate ==="
python3 - << PYEOF
import json, sys
with open("$PILOT_DIR/mechanism_diagnosis.json") as f:
    d = json.load(f)
text_f1 = d["means"]["text"]["f1"]
n_text_pos = d["pattern"]["n_text_pos"]
n_total = d["n_records"]
print(f"  text MAS mean F1: {text_f1:.3f} (threshold {$MIN_TEXT_F1})")
print(f"  text MAS scored:  {n_text_pos}/{n_total} (threshold {$MIN_TEXT_SCORED})")
fail = False
if text_f1 < $MIN_TEXT_F1:
    print(f"  GATE FAIL: text mean F1 below {$MIN_TEXT_F1}.")
    fail = True
if n_text_pos < $MIN_TEXT_SCORED:
    print(f"  GATE FAIL: fewer than {$MIN_TEXT_SCORED} tasks scored.")
    fail = True
if fail:
    print()
    print("  Baseline doesn't produce enough signal to measure a mechanism effect.")
    print("  The diffusion bridge cannot earn its complexity over W_e if the")
    print("  underlying MAS pipeline already fails on this model/benchmark combo.")
    print()
    print("  Try one of:")
    print(f"    MODEL=Qwen/Qwen3-8B BENCHMARK=$BENCHMARK sbatch \$0")
    print(f"    MODEL=$MODEL BENCHMARK=qampari sbatch \$0      # may not help")
    print(f"    MIN_TEXT_F1=0 sbatch \$0                          # force-skip gate")
    sys.exit(2)
print("  GATE PASS — baseline produces sufficient signal. Proceeding to bridge work.")
PYEOF

# If the python block exited 2, set -e in bash would normally kill the
# script, but heredoc python is run in a subshell that swallows the exit
# code. Re-check explicitly:
if grep -q "GATE FAIL" "$SLURM_SUBMIT_DIR/../logs/pilot_4b_$SLURM_JOB_ID.out" 2>/dev/null; then
    # Belt-and-suspenders; main check is the python sys.exit above
    :
fi

# ────────────────────────────────────────────────────────────────────
# Stage 2: extract training pairs from Stage 0 transcripts
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 2: extract training pairs ==="
python -m latent_pilot.extract_training_pairs \
    --source-roots "$TRANSCRIPTS_DIR" \
    --output "$PAIRS_PATH" \
    --min-target-chars 30

N_PAIRS=$(wc -l < "$PAIRS_PATH")
echo "  Pairs extracted: $N_PAIRS"
if [ "$N_PAIRS" -lt 20 ]; then
    echo "ERROR: fewer than 20 training pairs. Cannot train bridge."
    exit 1
fi

# ────────────────────────────────────────────────────────────────────
# Stage 3: trainer smoke test
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 3: trainer smoke test ==="
SMOKE_DIR=$PILOT_DIR/smoke
mkdir -p "$SMOKE_DIR"

python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir "$SMOKE_DIR" \
    --max-pairs 64 \
    --epochs 1 \
    --batch-size 1 \
    --k-soft-prompts 16 \
    --m-positions 16 \
    --n-source-layers 3 \
    --n-blocks 2 \
    --K-sample-eval 5 \
    --eval-every 50 \
    --save-every 100 \
    --warmup-steps 20 \
    --dtype bfloat16

python3 - << PYEOF
import json
losses, kls = [], []
with open("$SMOKE_DIR/train_log.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "loss" in r: losses.append(r["loss"])
        if "eval_kl" in r: kls.append(r["eval_kl"])
if losses:
    print(f"  Smoke loss: first={losses[0]:.4f}, last={losses[-1]:.4f}")
if kls:
    print(f"  Smoke KL: {kls}")
if losses and losses[-1] > losses[0]:
    print("  WARNING: smoke loss did not decrease — possible trainer bug.")
PYEOF

# ────────────────────────────────────────────────────────────────────
# Stage 4: full bridge training (multi-layer source)
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 4: full bridge training (3-layer source) ==="
python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir "$BRIDGE_RUN" \
    --epochs 2 \
    --batch-size 2 \
    --lr 3e-4 \
    --k-soft-prompts 16 \
    --m-positions 32 \
    --n-source-layers 3 \
    --n-blocks 2 \
    --K-sample-eval 20 \
    --eval-every 500 \
    --save-every 2000 \
    --warmup-steps 200 \
    --dtype bfloat16

CKPT_MAIN="$BRIDGE_RUN/bridge_epoch1.pt"
if [ ! -f "$CKPT_MAIN" ]; then
    echo "ERROR: main bridge checkpoint missing"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────
# Stage 5: layer ablation (single-layer source)
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 5: ablation — single-layer source ==="
python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir "$BRIDGE_ABLATION" \
    --epochs 2 \
    --batch-size 2 \
    --lr 3e-4 \
    --k-soft-prompts 16 \
    --m-positions 32 \
    --n-source-layers 1 \
    --n-blocks 2 \
    --K-sample-eval 20 \
    --eval-every 500 \
    --save-every 2000 \
    --warmup-steps 200 \
    --dtype bfloat16

CKPT_ABL="$BRIDGE_ABLATION/bridge_epoch1.pt"

# ────────────────────────────────────────────────────────────────────
# Stage 6: KL comparison
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 6: KL comparison (W_e vs diffusion) ==="
KL_MAIN=$PILOT_DIR/kl_main.json
KL_ABL=$PILOT_DIR/kl_ablation.json

python -m latent_pilot.eval_bridge_kl \
    --model-name "$MODEL" \
    --bridge-ckpt "$CKPT_MAIN" \
    --pairs-path "$PAIRS_PATH" \
    --output "$KL_MAIN" \
    --K-values 0 5 20 \
    --max-eval-batches 32

if [ -f "$CKPT_ABL" ]; then
    python -m latent_pilot.eval_bridge_kl \
        --model-name "$MODEL" \
        --bridge-ckpt "$CKPT_ABL" \
        --pairs-path "$PAIRS_PATH" \
        --output "$KL_ABL" \
        --K-values 0 5 20 \
        --max-eval-batches 32
fi

# ────────────────────────────────────────────────────────────────────
# Stage 7: summary
# ────────────────────────────────────────────────────────────────────
echo
echo "=== Stage 7: Summary ==="
python3 - << PYEOF
import json, os
def load(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)
main = load("$KL_MAIN")
abl = load("$KL_ABL")

print()
print("KL nats/token (lower is better)")
print("=" * 60)
print(f'{"bridge":>20} {"K=0 (W_e)":>12} {"K=5":>10} {"K=20":>10}')
for label, d in [("main (3-layer)", main), ("ablation (1-layer)", abl)]:
    if not d: continue
    klp = d["kl_per_K"]
    print(f'{label:>20} {klp.get("0", 0):>12.4f} {klp.get("5", 0):>10.4f} {klp.get("20", 0):>10.4f}')

print()
print("Decision rules:")
if main:
    k20 = main["kl_per_K"].get("20", 999)
    if k20 < 0.55:
        print(f"  K=20 = {k20:.3f} < 0.55 → BRIDGE HAS HEADROOM. Proceed to end-to-end three-condition eval.")
    elif k20 < 0.60:
        print(f"  K=20 = {k20:.3f} in [0.55, 0.60] → ambiguous. Try MLP adapter or more layers.")
    else:
        print(f"  K=20 = {k20:.3f} ≥ 0.60 → bridge not earning its complexity. Re-scope.")
if main and abl:
    delta = main["kl_per_K"].get("20", 999) - abl["kl_per_K"].get("20", 999)
    print(f"  Multi-layer vs single-layer KL delta: {delta:+.4f}")
    print("    → multi-layer source materially helps" if delta < -0.05
          else "    → multi-layer source not earning its compute")
PYEOF

echo
echo "=== Pilot done: $(date) ==="
ls -la "$PILOT_DIR"
