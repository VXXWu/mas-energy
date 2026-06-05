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
conda activate ${PILOT_ENV:-mas-energy}
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export LATENTMAS_REPO=/atlas2/u/$USER/mas_project/LatentMAS
export MAS_ENERGY_CODE=/atlas2/u/$USER/mas_project/mas-energy/code
# Reset PYTHONPATH to avoid inheriting swebench_worktrees pollution from
# .bashrc — that path contains an old sympy worktree that shadows the conda
# env's sympy 1.14.0 and breaks `from sympy.core.numbers import equal_valued`
# inside torch._dynamo. Symptom: ImportError loading torch's modules at
# pilot startup, every Stage 8 job crashes during Stage 6 init.
unset PYTHONPATH
export PYTHONPATH=$LATENTMAS_REPO:$MAS_ENERGY_CODE

# Cleanup: remove any .pth/.egg-link files left behind by SWE-bench
# `pip install -e .` operations on worktree sympy fixtures. Without this,
# any swebench_worktrees-pointing .pth file shadows the conda env's sympy
# and breaks torch._dynamo at import time. This is the canonical pattern
# used by a5000_swebench_*.sbatch — copied here verbatim.
SITE_PKG=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
if [ -n "$SITE_PKG" ] && [ -d "$SITE_PKG" ]; then
    for f in "$SITE_PKG"/*.pth "$SITE_PKG"/*.egg-link; do
        [ -f "$f" ] && grep -q "swebench_worktrees" "$f" 2>/dev/null && {
            echo "  Removing SWE-bench worktree pollution: $f"
            rm -f "$f"
        }
    done
fi

# Runtime guard: verify sympy is loaded from the conda env's site-packages,
# not a swebench worktree. Fails fast with a clear message if the polluted
# import path is still being picked up (e.g., from a .pth file we can't see
# from local checkout). Avoids the obscure torch._dynamo ImportError chain.
set +e
python - << 'PYGUARD'
"""Pre-flight env guard.

Verifies every package whose ABI/version drift has historically broken pilot
jobs is importable AND coming from the mas-energy conda env. Each check
short-circuits at the first failure with a specific diagnostic.

History (see ~/.claude/bug_fixes.md for full entries):
  - 2026-04-27: sympy pre-1.13 missing equal_valued → torch._dynamo fails
  - 2026-06-02: SWE-bench worktree shadowed sympy via PYTHONPATH
  - 2026-06-03: SWE-bench .pth dropped sympy; .pth cleanup found dropped sympy
  - 2026-06-03: SWE-bench downgraded numpy → scipy ABI mismatch
"""
import os, sys, traceback

EXPECTED_PREFIX = ("/atlas2/u/" + os.environ["USER"] + "/miniconda3/envs/"
                   + os.environ.get("CONDA_DEFAULT_ENV", "mas-energy"))

def _fail(pkg, msg):
    print(f"FAIL [{pkg}]: {msg}", file=sys.stderr)
    sys.exit(1)

def _check_from_env(pkg, mod):
    """Module file must live inside the mas-energy site-packages."""
    if not mod.__file__.startswith(EXPECTED_PREFIX):
        _fail(pkg, f"loaded from {mod.__file__}\n"
                   f"               expected prefix: {EXPECTED_PREFIX}\n"
                   f"               likely a .pth/.egg-link in site-packages "
                   f"pointing elsewhere (e.g. swebench_worktrees).")

# 1. sympy — version drift + .pth shadowing class (2026-04-27, 2026-06-02/03).
try:
    import sympy
except ImportError as e:
    _fail("sympy", f"missing: {e}\n"
                   f"               Reinstall: pip install --no-deps --force-reinstall 'sympy==1.14.0'")
_check_from_env("sympy", sympy)
sver = tuple(int(x) for x in sympy.__version__.split(".")[:2])
if sver < (1, 13):
    _fail("sympy", f"{sympy.__version__} too old (need >=1.13 for equal_valued)")

# 2. numpy — must come from mas-energy env (path-prefix check), but version
# is whatever rebuild_env.sh pinned (currently 1.26.4). We do NOT version-pin
# here because the env's own scipy/sklearn are built against that specific
# numpy. If a SWE-bench task changes numpy, the scipy ABI check below catches
# the actual symptom — no need to second-guess the version itself.
try:
    import numpy
except ImportError as e:
    _fail("numpy", f"missing: {e}")
_check_from_env("numpy", numpy)

# 3. scipy — actually triggers the ABI dtype-size error if numpy was downgraded
# AFTER scipy was built. Exercise an import path that USES the C extension.
try:
    from scipy.spatial.transform import Rotation  # noqa: F401
    import scipy
except (ImportError, ValueError) as e:
    msg = f"{type(e).__name__}: {e}"
    if "dtype size changed" in str(e) or "binary incompatibility" in str(e):
        msg += "\n               numpy/scipy ABI mismatch (numpy was downgraded after scipy built)."
        msg += "\n               Fix: pip install --no-deps --force-reinstall 'numpy>=2.0,<2.3' 'scipy>=1.13'"
    _fail("scipy", msg)
_check_from_env("scipy", scipy)

# 4. sklearn — analyze_end_to_end and several pilot helpers depend on it.
# Same ABI mismatch class as scipy.
try:
    import sklearn
    from sklearn.metrics import f1_score  # exercise compiled paths
except (ImportError, ValueError) as e:
    _fail("sklearn", f"{type(e).__name__}: {e}\n"
                     f"               If 'dtype size changed': numpy/scipy ABI mismatch; reinstall as in [3].")
_check_from_env("sklearn", sklearn)

# 5. torch — final goal of all imports. If something earlier is wrong this
# usually surfaces here via torch._dynamo.utils → symbolic_shapes → sympy chain.
try:
    import torch
except ImportError as e:
    _fail("torch", f"{e}\n"
                   f"               Sometimes this is a cascading failure from sympy/numpy upstream.")
_check_from_env("torch", torch)

# 6. transformers — pilot can't run without it.
try:
    import transformers
except ImportError as e:
    _fail("transformers", f"{e}")
_check_from_env("transformers", transformers)

print(f"env guard OK:")
print(f"  sympy        v{sympy.__version__}")
print(f"  numpy        v{numpy.__version__}")
print(f"  scipy        v{scipy.__version__}")
print(f"  sklearn      v{sklearn.__version__}")
print(f"  torch        v{torch.__version__}")
print(f"  transformers v{transformers.__version__}")
PYGUARD
GUARD_RC=$?
set -e
if [ $GUARD_RC -ne 0 ]; then
  echo "ERROR: env guard failed — see specific package failure above."
  echo ""
  echo "  Common causes (and fixes — run inside 'conda activate mas-energy'):"
  echo ""
  echo "  1) SWE-bench .pth file shadowing real install:"
  echo "       SITE_PKG=\$(python -c \"import sysconfig; print(sysconfig.get_paths()['purelib'])\")"
  echo "       for f in \"\$SITE_PKG\"/*.pth \"\$SITE_PKG\"/*.egg-link; do"
  echo "         [ -f \"\$f\" ] && grep -q swebench_worktrees \"\$f\" && rm -f \"\$f\""
  echo "       done"
  echo ""
  echo "  2) scipy/sklearn ABI mismatch (numpy changed under them):"
  echo "       Best fix: bash mas-energy/scripts/rebuild_env.sh"
  echo "       OR (faster): pip install --no-deps --force-reinstall scipy scikit-learn"
  echo ""
  echo "  3) sympy missing or pre-1.13:"
  echo "       pip install --no-deps --force-reinstall 'sympy==1.14.0'"
  echo ""
  echo "  See ~/.claude/bug_fixes.md for full history of this class of bug."
  exit 1
fi

# Env switched 2026-06-02 from `mas-latent` to `mas-energy` (the main study's
# env). mas-latent had a multi-week saga of broken sgl_kernel binaries
# (sm100-only, missing libnvrtc.so.13) and a torch 2.11+cu130 vs torch
# 2.9+cu128 ABI gap. mas-energy has working torch + transformers + SGLang
# proven by the main study running in it daily. The pilot's only LatentMAS-
# repo dependency (`from models import ModelWrapper`) was found to be a
# type-annotation-only import never instantiated at runtime — made optional
# so the pilot now runs in mas-energy without LatentMAS installed. Override
# with PILOT_ENV=mas-latent if needed for backward compatibility.

# ---- Config (env-overridable) ----
MODEL=${MODEL:-Qwen/Qwen3-4B}
BENCHMARK=${BENCHMARK:-fanoutqa}
# Smoke mode: tiny inputs end-to-end, validates every stage's wiring in
# ~10 min wall instead of ~5 hours. Use `SMOKE=1 sbatch ...` whenever
# changing pipeline code; only run the full pilot once smoke passes.
# Bridge depth. The bridge scales like O(n_blocks · d_model²); at d=4096
# (8B backbone) n_blocks=2 alone is ~1B params which OOMs on a 24G A5000
# alongside the backbone + Adam optimizer state. n_blocks=1 (~500M at 8B)
# fits comfortably. Override via env var if your model needs more capacity
# and you have a larger GPU.
BRIDGE_N_BLOCKS=${BRIDGE_N_BLOCKS:-1}

# Parallelism: STAGES env var selects which stage subset to run, so multiple
# SLURM jobs can share one PILOT_DIR via REUSE_PILOT_DIR and run in parallel
# on separate GPUs. Examples:
#   STAGES=mechanism     → Stage 1 + Stage 1b only (mechanism diagnosis)
#   STAGES=bridge_main   → Stages 2-4 only (pairs + smoke + main bridge train)
#   STAGES=bridge_ablation → Stages 2 + 5 only (pairs + ablation training)
#   STAGES=eval          → Stages 6 + 8 (KL eval + end-to-end; needs bridges)
#   STAGES=all (default) → full sequential pipeline (Stages 0-8)
STAGES=${STAGES:-all}
case "$STAGES" in
    all|mechanism|bridge_main|bridge_ablation|eval) ;;
    *) echo "ERROR: unknown STAGES=$STAGES"; exit 1 ;;
esac
do_stage() {
    # Returns 0 if the stage should run, 1 if it should be skipped given $STAGES.
    case "$STAGES" in
        all) return 0 ;;
        mechanism)
            [[ "$1" == "stage0" || "$1" == "stage1" || "$1" == "stage1b" ]] && return 0
            return 1 ;;
        bridge_main)
            [[ "$1" == "stage2" || "$1" == "stage3" || "$1" == "stage4" ]] && return 0
            return 1 ;;
        bridge_ablation)
            [[ "$1" == "stage2" || "$1" == "stage5" ]] && return 0
            return 1 ;;
        eval)
            [[ "$1" == "stage6" || "$1" == "stage7" || "$1" == "stage8" ]] && return 0
            return 1 ;;
    esac
    return 1
}
echo "STAGES filter: $STAGES"

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    N_TRANSCRIPT_TASKS=${N_TRANSCRIPT_TASKS:-2}
    N_MECHANISM_TASKS=${N_MECHANISM_TASKS:-2}
    SMOKE_MAX_PAIRS=8
    SMOKE_EPOCHS=${BRIDGE_EPOCHS:-1}
    SMOKE_K_SAMPLE=${BRIDGE_K_SAMPLE:-2}
    SMOKE_EVAL_BATCHES=${BRIDGE_EVAL_BATCHES:-2}
    MIN_TEXT_F1=${MIN_TEXT_F1:-0}
    MIN_TEXT_SCORED=${MIN_TEXT_SCORED:-0}
    echo "SMOKE MODE: tiny inputs throughout. Use for pipeline validation only."
else
    N_TRANSCRIPT_TASKS=${N_TRANSCRIPT_TASKS:-30}
    N_MECHANISM_TASKS=${N_MECHANISM_TASKS:-20}
    SMOKE_MAX_PAIRS=""
    # Default full-mode training budget. With 6,523 main-study pairs at
    # batch_size 2, 5 epochs ≈ 16,000 gradient steps — a realistic budget
    # for fine-tuning a 588M-param warm-started diffusion bridge. Earlier
    # default (2 epochs) was inherited from the 87-pair tiny-corpus pilot
    # and severely under-trained the bridge.
    SMOKE_EPOCHS=${BRIDGE_EPOCHS:-5}
    SMOKE_K_SAMPLE=${BRIDGE_K_SAMPLE:-20}
    SMOKE_EVAL_BATCHES=${BRIDGE_EVAL_BATCHES:-32}
    # Minimum mean text-mas F1 to consider baseline viable for bridge training.
    # Set to 0 to skip the gate.
    MIN_TEXT_F1=${MIN_TEXT_F1:-0.10}
    MIN_TEXT_SCORED=${MIN_TEXT_SCORED:-5}  # min tasks where text_f1 > 0
fi
# Batch size and warmup are also configurable. Larger batch = more stable
# training and better GPU utilization at the cost of more memory.
BRIDGE_BATCH_SIZE=${BRIDGE_BATCH_SIZE:-2}
BRIDGE_WARMUP_STEPS=${BRIDGE_WARMUP_STEPS:-500}
BRIDGE_LR=${BRIDGE_LR:-3e-4}

MODEL_TAG=$(echo "$MODEL" | tr '/.' '_-')
RESULTS_BASE=/atlas2/u/$USER/mas_project/mas-energy/results

# Resume mode: if REUSE_PILOT_DIR is set, point at an existing pilot dir
# and skip stages whose outputs already exist. Lets us salvage runs where
# a late-stage bug burned a re-run of expensive Stage 0/1. Defaults to a
# fresh timestamped dir.
if [ -n "${REUSE_PILOT_DIR:-}" ]; then
    PILOT_DIR=$REUSE_PILOT_DIR
    if [ ! -d "$PILOT_DIR" ]; then
        echo "ERROR: REUSE_PILOT_DIR=$PILOT_DIR does not exist."
        exit 1
    fi
    echo "Resume mode: reusing $PILOT_DIR"
else
    # Stable dir keyed on model+benchmark (no timestamp). This means parallel
    # STAGES=mechanism / bridge_main / bridge_ablation / eval jobs auto-share
    # the same dir without needing a manual REUSE_PILOT_DIR. Resume mode is
    # implicit: if a stage's outputs already exist, the do_stage guards (and
    # STAGE_0_DONE / STAGE_1_DONE flags below) skip it. Override with
    # REUSE_PILOT_DIR=<other-path> if you want a different dir.
    #
    # PILOT_DIR_SUFFIX appends a tag to the dir name, e.g.
    # PILOT_DIR_SUFFIX=_specmatched produces diffusion_pilot_<bench>_<model>_specmatched.
    # Use this to keep results from different parameter regimes (truncation cap,
    # ablation variant, etc.) separate without overwriting prior runs.
    PILOT_DIR=$RESULTS_BASE/diffusion_pilot_${BENCHMARK}_$(echo $MODEL | tr '/.' '_')${PILOT_DIR_SUFFIX:-}
fi
TRANSCRIPTS_DIR=$PILOT_DIR/transcripts
PAIRS_PATH=$PILOT_DIR/pairs.jsonl
BRIDGE_RUN=$PILOT_DIR/bridge_main
BRIDGE_ABLATION=$PILOT_DIR/bridge_layer1

# Main-study corpus mode: instead of generating fresh transcripts in Stage 0
# and extracting ~87 pairs from them, use the ~6,500 pairs already extracted
# from the main study's Qwen3.5-9B decentralized runs (across 6 benchmarks).
# Set USE_MAIN_STUDY_CORPUS=1 with MODEL=Qwen/Qwen3.5-9B to enable.
# Optionally filter to a single benchmark via MAIN_CORPUS_BENCH=fanoutqa
# (defaults to all 6 benchmarks pooled).
USE_MAIN_STUDY_CORPUS=${USE_MAIN_STUDY_CORPUS:-0}
MAIN_STUDY_PAIRS=$RESULTS_BASE/latent_pilot/training_pairs.jsonl
MAIN_CORPUS_BENCH=${MAIN_CORPUS_BENCH:-}

mkdir -p "$PILOT_DIR" "$TRANSCRIPTS_DIR" "$BRIDGE_RUN" "$BRIDGE_ABLATION"
cd /atlas2/u/$USER/mas_project/mas-energy/code

echo "=== Hybrid Latent + Diffusion Bridge Pilot ==="
echo "Model:     $MODEL"
echo "Benchmark: $BENCHMARK"
echo "Pilot dir: $PILOT_DIR"
if [ "$USE_MAIN_STUDY_CORPUS" = "1" ]; then
    echo "Mode:      USE_MAIN_STUDY_CORPUS=1 (using 6,523-pair Qwen3.5-9B corpus)"
    if [ -n "$MAIN_CORPUS_BENCH" ]; then
        echo "           filtered to benchmark=$MAIN_CORPUS_BENCH"
    fi
fi
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

# Detect existing outputs from prior partial runs so we can skip completed
# stages. Each STAGE_X_DONE flag is read by the Stage X block to decide
# whether to run or skip.
EXISTING_TRANSCRIPT=$(find "$TRANSCRIPTS_DIR" -name "*decentralized*.jsonl" 2>/dev/null | head -1)
EXISTING_MECH_DIAG="$PILOT_DIR/mechanism_diagnosis.json"
STAGE_0_DONE=0; [ -n "$EXISTING_TRANSCRIPT" ] && [ -s "$EXISTING_TRANSCRIPT" ] && STAGE_0_DONE=1
STAGE_1_DONE=0; [ -f "$EXISTING_MECH_DIAG" ] && STAGE_1_DONE=1
# In main-study-corpus mode, we don't need fresh transcripts at all.
# Pretend Stage 0 is done so it's skipped; Stage 2 takes its pairs from
# the main-study extraction instead.
if [ "$USE_MAIN_STUDY_CORPUS" = "1" ]; then
    STAGE_0_DONE=1
    echo "Main-study corpus mode → Stage 0 will be skipped (pairs come from $MAIN_STUDY_PAIRS)"
fi
if [ "$STAGE_0_DONE" = "1" ]; then
    echo "Stage 0 output exists ($EXISTING_TRANSCRIPT) — will skip."
fi
if [ "$STAGE_1_DONE" = "1" ]; then
    echo "Stage 1 output exists ($EXISTING_MECH_DIAG) — will skip."
fi

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
# Stage 0: produce call_records-format decentralized text-MAS transcripts.
# Prefers SGLang (batched, fast) but falls back to a pure-HF generator
# (gen_transcripts_hf.py) when SGLang's launcher isn't healthy. The HF
# fallback is slower per task but has no sgl_kernel dependency, which is
# the failure mode that has been blocking this pilot since mid-May.
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage0; then
echo "=== Stage 0: text decentralized transcripts (HF generator) ==="
if [ "$STAGE_0_DONE" = "1" ]; then
    echo "  SKIPPING — transcripts already present at $EXISTING_TRANSCRIPT"
else
    python -m latent_pilot.gen_transcripts_hf \
        --model-name "$MODEL" \
        --benchmark "$BENCHMARK" \
        --n-tasks $N_TRANSCRIPT_TASKS \
        --n-agents 3 \
        --n-rounds 1 \
        --max-react-steps 5 \
        --output-dir "$TRANSCRIPTS_DIR" \
        $DATA_DIR_ARG
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader

if [ "$USE_MAIN_STUDY_CORPUS" = "1" ]; then
    echo "  (Stage 0 transcript check skipped — pairs come from main-study corpus at Stage 2)"
else
    TRANSCRIPT_FILE=$(find "$TRANSCRIPTS_DIR" -name "*decentralized*.jsonl" | head -1)
    if [ -z "$TRANSCRIPT_FILE" ] || [ ! -s "$TRANSCRIPT_FILE" ]; then
        echo "ERROR: no transcripts produced."
        find "$TRANSCRIPTS_DIR" -type f
        exit 1
    fi
    echo "  Transcript: $TRANSCRIPT_FILE ($(wc -l < $TRANSCRIPT_FILE) records)"
fi
fi  # end if do_stage stage0

# ────────────────────────────────────────────────────────────────────
# Stage 1: agentic_latentmas mechanism validation
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage1; then
echo "=== Stage 1: agentic_latentmas mechanism validation ==="
# Default config matches the main study's Decentralized topology (config.py):
# MAX_REACT_STEPS=10, DECENTRALIZED_ROUNDS=2. Earlier pilot runs used k=5, R=1
# which capped the text-MAS baseline; the +0.157 F1 latent-vs-text result
# from the May 30 run reflects that capped comparison, not main-study quality.
# Override via env vars if you need to reproduce the prior reduced budget.
MECH_K=${MECH_K:-10}
MECH_R=${MECH_R:-2}
MECH_M_LATENT=${MECH_M_LATENT:-20}
if [ "$STAGE_1_DONE" = "1" ]; then
    echo "  SKIPPING — mechanism_diagnosis.json already present"
    echo "  (delete that file to force a re-run at MECH_K=$MECH_K MECH_R=$MECH_R)"
else
    echo "  Config: max-react-steps=$MECH_K n-rounds=$MECH_R m-latent=$MECH_M_LATENT"
    python -m latent_pilot.agentic_latentmas \
        --benchmark "$BENCHMARK" \
        --n-tasks $N_MECHANISM_TASKS \
        --m-latent $MECH_M_LATENT \
        --max-react-steps $MECH_K \
        --n-rounds $MECH_R \
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
fi
fi  # end if do_stage stage1

# ────────────────────────────────────────────────────────────────────
# Stage 1b: HARD GATE on baseline signal
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage1b; then
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
fi  # end if do_stage stage1b

# ────────────────────────────────────────────────────────────────────
# Stage 2: extract training pairs from Stage 0 transcripts
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage2; then
echo "=== Stage 2: extract training pairs ==="
if [ "$USE_MAIN_STUDY_CORPUS" = "1" ]; then
    if [ ! -s "$MAIN_STUDY_PAIRS" ]; then
        echo "ERROR: USE_MAIN_STUDY_CORPUS=1 but $MAIN_STUDY_PAIRS doesn't exist."
        echo "Regenerate via:  python -m latent_pilot.extract_training_pairs \\"
        echo "    --source-roots $RESULTS_BASE/a5000_latent_transcripts \\"
        echo "    --output $MAIN_STUDY_PAIRS"
        exit 1
    fi
    # Pool all 6 benchmarks by default; filter via MAIN_CORPUS_BENCH=fanoutqa
    if [ -n "$MAIN_CORPUS_BENCH" ]; then
        echo "  Filtering main-study corpus to benchmark='$MAIN_CORPUS_BENCH'"
        python3 -c "
import json
out_path = '$PAIRS_PATH'
src_path = '$MAIN_STUDY_PAIRS'
target_bench = '$MAIN_CORPUS_BENCH'
with open(src_path) as src, open(out_path, 'w') as dst:
    n_in, n_out = 0, 0
    for line in src:
        try:
            r = json.loads(line)
        except: continue
        n_in += 1
        if r.get('benchmark') == target_bench:
            dst.write(line)
            n_out += 1
print(f'  Filtered {n_out}/{n_in} pairs to benchmark={target_bench!r}')
"
    else
        cp "$MAIN_STUDY_PAIRS" "$PAIRS_PATH"
        echo "  Using all $(wc -l < $PAIRS_PATH) pairs from main-study corpus"
    fi
else
    python -m latent_pilot.extract_training_pairs \
        --source-roots "$TRANSCRIPTS_DIR" \
        --output "$PAIRS_PATH" \
        --min-target-chars 30
fi

N_PAIRS=$(wc -l < "$PAIRS_PATH")
echo "  Pairs extracted: $N_PAIRS"
PAIR_MIN=20
[ "$SMOKE" = "1" ] && PAIR_MIN=2
if [ "$N_PAIRS" -lt "$PAIR_MIN" ]; then
    echo "ERROR: fewer than $PAIR_MIN training pairs. Cannot train bridge."
    exit 1
fi
fi  # end if do_stage stage2

# ────────────────────────────────────────────────────────────────────
# Stage 3: trainer smoke test
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage3; then
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
    --n-blocks $BRIDGE_N_BLOCKS \
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
fi  # end if do_stage stage3

# ────────────────────────────────────────────────────────────────────
# Stage 4: full bridge training (multi-layer source)
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage4; then
echo "=== Stage 4: full bridge training (3-layer source) ==="
echo "  Config: epochs=$SMOKE_EPOCHS batch_size=$BRIDGE_BATCH_SIZE lr=$BRIDGE_LR warmup=$BRIDGE_WARMUP_STEPS"
python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir "$BRIDGE_RUN" \
    --epochs $SMOKE_EPOCHS \
    --batch-size $BRIDGE_BATCH_SIZE \
    --lr $BRIDGE_LR \
    --k-soft-prompts 16 \
    --m-positions 32 \
    --n-source-layers 3 \
    --n-blocks $BRIDGE_N_BLOCKS \
    --K-sample-eval $SMOKE_K_SAMPLE \
    --eval-every 500 \
    --save-every 2000 \
    --warmup-steps $BRIDGE_WARMUP_STEPS \
    --dtype bfloat16 \
    ${SMOKE_MAX_PAIRS:+--max-pairs $SMOKE_MAX_PAIRS}

# Pick the highest-numbered epoch checkpoint that actually exists. In
# SMOKE mode only epoch 0 runs (SMOKE_EPOCHS=1), so bridge_epoch0.pt
# is the only checkpoint; in full mode there's epoch0 + epoch1.
CKPT_MAIN=$(ls -1 "$BRIDGE_RUN"/bridge_epoch*.pt 2>/dev/null | sort | tail -1)
if [ -z "$CKPT_MAIN" ] || [ ! -f "$CKPT_MAIN" ]; then
    echo "ERROR: no main bridge checkpoint found in $BRIDGE_RUN"
    ls -la "$BRIDGE_RUN" 2>/dev/null || true
    exit 1
fi
echo "  main bridge checkpoint: $CKPT_MAIN"
fi  # end if do_stage stage4

# ────────────────────────────────────────────────────────────────────
# Stage 5: layer ablation (single-layer source)
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage5; then
echo "=== Stage 5: ablation — single-layer source ==="
if [ "$SMOKE" = "1" ]; then
    echo "  SKIPPING in smoke mode (Stage 4 covers the training path)"
else
python -m latent_pilot.train_diffusion_bridge \
    --model-name "$MODEL" \
    --pairs-path "$PAIRS_PATH" \
    --output-dir "$BRIDGE_ABLATION" \
    --epochs $SMOKE_EPOCHS \
    --batch-size $BRIDGE_BATCH_SIZE \
    --lr $BRIDGE_LR \
    --k-soft-prompts 16 \
    --m-positions 32 \
    --n-source-layers 1 \
    --n-blocks $BRIDGE_N_BLOCKS \
    --K-sample-eval $SMOKE_K_SAMPLE \
    --eval-every 500 \
    --save-every 2000 \
    --warmup-steps $BRIDGE_WARMUP_STEPS \
    --dtype bfloat16
fi  # end SMOKE skip of ablation

CKPT_ABL=$(ls -1 "$BRIDGE_ABLATION"/bridge_epoch*.pt 2>/dev/null | sort | tail -1)
fi  # end if do_stage stage5

# Re-resolve checkpoints unconditionally so eval-only runs (STAGES=eval) find
# them even when stage4/stage5 blocks above were skipped. If the prior block
# already set these to a valid path, that value is preserved.
if [ -z "$CKPT_MAIN" ] || [ ! -f "$CKPT_MAIN" ]; then
    CKPT_MAIN=$(ls -1 "$BRIDGE_RUN"/bridge_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT_ABL" ] || [ ! -f "$CKPT_ABL" ]; then
    CKPT_ABL=$(ls -1 "$BRIDGE_ABLATION"/bridge_epoch*.pt 2>/dev/null | sort | tail -1)
fi

# ────────────────────────────────────────────────────────────────────
# Stage 6: KL comparison
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage6; then
echo "=== Stage 6: KL comparison (W_e vs diffusion) ==="
if [ -z "$CKPT_MAIN" ] || [ ! -f "$CKPT_MAIN" ]; then
    echo "ERROR: Stage 6 needs a main bridge checkpoint but none found at $BRIDGE_RUN"
    ls -la "$BRIDGE_RUN" 2>/dev/null || true
    exit 1
fi
echo "  using main checkpoint:     $CKPT_MAIN"
echo "  using ablation checkpoint: ${CKPT_ABL:-<none>}"
KL_MAIN=$PILOT_DIR/kl_main.json
KL_ABL=$PILOT_DIR/kl_ablation.json

if [ "$SMOKE" = "1" ]; then
    KL_K_VALUES="0 2"
else
    KL_K_VALUES="0 5 20"
fi

python -m latent_pilot.eval_bridge_kl \
    --model-name "$MODEL" \
    --bridge-ckpt "$CKPT_MAIN" \
    --pairs-path "$PAIRS_PATH" \
    --output "$KL_MAIN" \
    --K-values $KL_K_VALUES \
    --max-eval-batches $SMOKE_EVAL_BATCHES

if [ -f "$CKPT_ABL" ]; then
    python -m latent_pilot.eval_bridge_kl \
        --model-name "$MODEL" \
        --bridge-ckpt "$CKPT_ABL" \
        --pairs-path "$PAIRS_PATH" \
        --output "$KL_ABL" \
        --K-values $KL_K_VALUES \
        --max-eval-batches $SMOKE_EVAL_BATCHES
fi
fi  # end if do_stage stage6

# ────────────────────────────────────────────────────────────────────
# Stage 7: summary
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage7; then
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

# Read the actual K values present in the output rather than hardcoding
# K=5/20 (which produced misleading "0.0000" output in SMOKE mode where
# only K=0 and K=2 were evaluated).
def k_values_of(d):
    if not d: return []
    return sorted([int(k) for k in d["kl_per_K"].keys()])

all_K = sorted(set(k_values_of(main)) | set(k_values_of(abl)))
header = f'{"bridge":>20} ' + ' '.join(f'K={k:<4}'.rjust(10) for k in all_K)
print(header)
for label, d in [("main (3-layer)", main), ("ablation (1-layer)", abl)]:
    if not d: continue
    klp = d["kl_per_K"]
    row_cells = []
    for k in all_K:
        v = klp.get(str(k))
        row_cells.append(f'{v:>10.4f}' if v is not None else f'{"—":>10}')
    print(f'{label:>20} ' + ' '.join(row_cells))

# Diffusion-vs-warm-start delta (K_max vs K=0). Meaningful regardless of
# whether K_max is 2 (smoke) or 20 (full).
print()
if main and all_K:
    k_max = max(all_K)
    k0 = main["kl_per_K"].get("0")
    km = main["kl_per_K"].get(str(k_max))
    if k0 is not None and km is not None:
        delta = km - k0
        print(f"  Diffusion delta: K={k_max} KL - K=0 KL = {delta:+.4f}")
        if delta < -0.10:
            print(f"  → diffusion bridge is doing real work over the W_e warm-start")
        elif delta < 0:
            print(f"  → diffusion bridge marginally improves over warm-start")
        else:
            print(f"  → diffusion bridge does NOT improve over warm-start (W_e is sufficient)")

# Original decision rule still applies in non-smoke runs (K_max = 20).
if main:
    klp = main["kl_per_K"]
    if "20" in klp:
        k20 = klp["20"]
        if k20 < 0.55:
            print(f"  K=20 = {k20:.3f} < 0.55 → BRIDGE HAS HEADROOM. Proceed to end-to-end three-condition eval.")
        elif k20 < 0.60:
            print(f"  K=20 = {k20:.3f} in [0.55, 0.60] → ambiguous. Try MLP adapter or more layers.")
        else:
            print(f"  K=20 = {k20:.3f} ≥ 0.60 → bridge not earning its complexity. Re-scope.")
    else:
        print(f"  (K=20 not present; decision rule for full eval applies after non-smoke run)")

if main and abl:
    k_max = max(all_K) if all_K else None
    if k_max is not None:
        m_k = main["kl_per_K"].get(str(k_max))
        a_k = abl["kl_per_K"].get(str(k_max))
        if m_k is not None and a_k is not None:
            delta = m_k - a_k
            print(f"  Multi-layer vs single-layer KL delta at K={k_max}: {delta:+.4f}")
            print("    → multi-layer source materially helps" if delta < -0.05
                  else "    → multi-layer source not earning its compute")
PYEOF
fi  # end if do_stage stage7


# ────────────────────────────────────────────────────────────────────
# Stage 8: end-to-end three-condition eval (text MAS / W_e hybrid /
# diffusion-bridge hybrid) with NVML energy + per-task accuracy.
# This is the reportable result.
# ────────────────────────────────────────────────────────────────────
echo
if do_stage stage8; then
echo "=== Stage 8: end-to-end three-condition eval ==="
E2E_OUT=$PILOT_DIR/end_to_end_results.jsonl
if [ "$SMOKE" = "1" ]; then
    E2E_N_TASKS=2
    E2E_CONDITIONS=${E2E_CONDITIONS:-"text latent_we latent_diffusion"}
    E2E_K=$SMOKE_K_SAMPLE
else
    E2E_N_TASKS=${E2E_N_TASKS:-30}
    E2E_CONDITIONS=${E2E_CONDITIONS:-"text latent_we latent_diffusion"}
    E2E_K=$SMOKE_K_SAMPLE
fi
E2E_N_ROUNDS=${E2E_N_ROUNDS:-1}
# Per-condition compute budgets — all env-overridable so experiments can
# reduce them to make pure-latent diffusion clearly cheaper than text MAS
# (the design intent). Defaults match the original baseline runs.
E2E_M_LATENT=${E2E_M_LATENT:-20}
E2E_MAX_REACT_STEPS=${E2E_MAX_REACT_STEPS:-5}
E2E_M_SOURCE_POSITIONS=${E2E_M_SOURCE_POSITIONS:-32}
echo "  n-tasks=$E2E_N_TASKS  conditions=$E2E_CONDITIONS  K=$E2E_K  R=$E2E_N_ROUNDS"
echo "  m-latent=$E2E_M_LATENT  max-react-steps=$E2E_MAX_REACT_STEPS  m-source-positions=$E2E_M_SOURCE_POSITIONS"

python -m latent_pilot.eval_hybrid_diffusion \
    --model-name "$MODEL" \
    --bridge-ckpt "$CKPT_MAIN" \
    --benchmark "$BENCHMARK" \
    --output "$E2E_OUT" \
    --n-tasks $E2E_N_TASKS \
    --n-agents 3 \
    --m-latent $E2E_M_LATENT \
    --max-react-steps $E2E_MAX_REACT_STEPS \
    --K-sample $E2E_K \
    --m-source-positions $E2E_M_SOURCE_POSITIONS \
    --conditions $E2E_CONDITIONS \
    --n-rounds $E2E_N_ROUNDS \
    --dtype bfloat16 \
    $DATA_DIR_ARG || {
        echo "WARNING: end-to-end eval failed; see above. Earlier stages still valid."
    }

# Aggregate the three-condition results
python3 - << PYEOF
import json, os
from collections import defaultdict

path = "$E2E_OUT"
if not os.path.exists(path):
    print("  (no end_to_end_results.jsonl produced)")
else:
    agg = defaultdict(lambda: {"n": 0, "f1": 0.0, "loose": 0.0, "energy_J": 0.0, "wall_s": 0.0})
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except: continue
            if r.get("_session"):
                continue
            c = r.get("condition")
            if not c: continue
            agg[c]["n"] += 1
            agg[c]["f1"] += r.get("f1", 0.0)
            agg[c]["loose"] += r.get("loose_accuracy", 0.0)
            e = r.get("energy") or {}
            agg[c]["energy_J"] += e.get("gpu_dynamic_energy_joules", 0.0)
            agg[c]["wall_s"] += e.get("wall_seconds", 0.0)

    print()
    print("=== End-to-end three-condition results ===")
    print(f'{"condition":>20} {"n":>4} {"f1":>7} {"loose":>7} {"energy_J":>12} {"wall_s":>10}')
    for c in ["text", "latent_we", "latent_diffusion"]:
        v = agg.get(c)
        if not v: continue
        n = max(v["n"], 1)
        print(f'{c:>20} {v["n"]:>4} {v["f1"]/n:>7.3f} {v["loose"]/n:>7.3f} '
              f'{v["energy_J"]/n:>12.1f} {v["wall_s"]/n:>10.1f}')
    # Energy savings
    tE = agg.get("text", {}).get("energy_J", 0) / max(agg.get("text", {}).get("n", 1), 1)
    wE = agg.get("latent_we", {}).get("energy_J", 0) / max(agg.get("latent_we", {}).get("n", 1), 1)
    dE = agg.get("latent_diffusion", {}).get("energy_J", 0) / max(agg.get("latent_diffusion", {}).get("n", 1), 1)
    if tE > 0:
        print()
        if wE > 0:
            print(f"  W_e hybrid energy vs text:        {100*(wE-tE)/tE:+.1f}%")
        if dE > 0:
            print(f"  Diffusion hybrid energy vs text:  {100*(dE-tE)/tE:+.1f}%")
PYEOF
fi  # end if do_stage stage8

echo
echo "=== Pilot done: $(date) ==="
ls -la "$PILOT_DIR"
