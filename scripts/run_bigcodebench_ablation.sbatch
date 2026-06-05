#!/bin/bash
#SBATCH --job-name=bcb-abl
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/bcb_abl_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/bcb_abl_%A_%a.err

# BigCodeBench second-code-benchmark ablation. Tests whether the swebench
# iteration-collapse claim generalizes to function-calling code with diverse
# library use (~1140 tasks; 148 BCB-Hard subset available).
#
# Pilot results (n=20 each, calibration sbatch):
#   SAS k=10:  pass@1 = 0.650, ~22 kJ/task   (well below saturation)
#   Decent R=2: pass@1 = 0.882, ~123 kJ/task (n=17 partial; clear MAS lift)
# This is the right hardness regime: SAS-Decent gap = +0.23 ΔAcc, comparable
# to swebench's MAS-over-SAS lift. Plenty of headroom for IS to either
# collapse (replicating swebench) or stay near Decent (scoping the claim).
#
# 4 cells × n=200:
#   array idx 0: decentralized              (baseline)
#   array idx 1: independent_share          (iteration collapse test — core)
#   array idx 2: decentralized_minimal      (cross-bench decode-waste replication)
#   array idx 3: independent_share_minimal  (stacking interaction)
#
# Decision rules:
# - independent_share collapses (ΔAcc ≤ -0.10): swebench iteration claim
#   generalizes to function-calling code; n=2 in iteration-required class.
# - independent_share matches Decent (ΔAcc within ±0.05): swebench-specific
#   factor (repo-level multi-file editing) drove the collapse, not write→
#   test→fix iteration alone. Claim narrows to repo-scale code workflows.
# - Partial collapse (-0.05 to -0.10): ambiguous; report as suggestive.
#
# Time budget: BCB tasks run slower than HumanEval+ (more diverse libraries,
# longer prompts, longer test suites). Pilot: Decent ~5min/task average,
# bursts to 30+ min. n=200 × ~5 min × 1.5 (overhead) ≈ 25h worst case.
# 2026-04-30 attempt at 18h hit the time limit (job 15301139); only
# IS+min reached n=195. Bumped to 48h for clean buffer; resume logic
# picks up any partial JSONLs from the prior run, so re-submission
# only does the leftover tasks.
#
# Submit:
#   sbatch mas-energy/scripts/run_bigcodebench_ablation.sbatch
# Or just the two essential cells (Decent + IS):
#   sbatch --array=0-1 mas-energy/scripts/run_bigcodebench_ablation.sbatch

set -e

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
if [ "$STAGGER" -gt 0 ]; then sleep $STAGGER; fi

# Extra random stagger to desynchronize concurrent SGLang launches across
# different sbatches landing on atlas24. Prior crashes (15341274 etc.) had
# 6 simultaneous SGLang processes hitting model-loading peak together;
# random sleep spreads the peaks.
EXTRA_SLEEP=$(( RANDOM % 240 ))
echo "Extra random stagger: ${EXTRA_SLEEP}s"
sleep $EXTRA_SLEEP

source /sailhome/$USER/.bashrc
conda activate mas-energy
# Sanitize CWD and PYTHONPATH before any python invocation. Concurrent
# swebench tasks cd into worktrees containing their own sympy/ subdirs;
# new array tasks inheriting that cwd hit "import sympy" → broken
# worktree version → "module 'collections' has no attribute 'MutableSet'"
# (Python 2 era code). cd to home and clear PYTHONPATH to force conda
# env imports.
cd $HOME
unset PYTHONPATH


python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'sympy==1.13.3' 'mpmath>=1.3.0'
# Robust jinja2 fix — soft pip install silently no-ops due to corrupted
# ~etuptools metadata in this env (job 15335219 hit this). Force-reinstall.
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1), jinja2.__version__" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1), f'jinja2 still at {jinja2.__version__}'" \
    || { echo "ERROR: jinja2 upgrade failed on $(hostname); aborting"; exit 1; }

# Restore numpy <2.0 to match the env's pandas binary. A prior sklearn
# install (without --no-deps) bumped numpy to 2.4.4, which broke pandas
# C extensions (96 vs 88 byte dtype layout). Pinning numpy back is
# strictly simpler than rebuilding pandas + sklearn against numpy 2.x.
python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'numpy<2.0'
python -c "import numpy, pandas; print('numpy=' + numpy.__version__ + ' pandas=' + pandas.__version__)" \
    || { echo "ERROR: numpy/pandas state still broken on $(hostname); aborting"; exit 1; }

# scipy was rebuilt against numpy 2.x by an earlier sklearn install; pin
# back to scipy 1.11.4 — the last release before numpy 2.0 existed, so its
# wheels are guaranteed numpy-1.x-only. scipy 1.13.x wheels are forward-
# compat-built (numpy 2.x headers, references "_signature_descriptor")
# and break on numpy 1.26.
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scipy==1.11.4'
# Fallback: --ignore-installed leaves stale .so files; --force-reinstall
# overwrites them. Use only if the first install didn't take.
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --force-reinstall --quiet 'scipy==1.11.4'
python -c "import scipy.optimize; print('scipy=' + __import__('scipy').__version__)" \
    || { echo "ERROR: scipy/numpy ABI still broken on $(hostname); aborting"; exit 1; }

# sklearn was rebuilt against numpy 2.x by the same earlier install.
# scikit-learn 1.5.2 is the last release before numpy 2.0 existed (May
# 2024 vs numpy 2.0 June 2024); its wheels are guaranteed numpy-1.x-only.
# 1.6+ wheels reference numpy 2.x via Cython numpy.import_array and break
# on numpy 1.26 (job 15371703 caught this 2026-05-08 via deep-import).
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scikit-learn==1.5.2'
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --force-reinstall --quiet 'scikit-learn==1.5.2'
python -c "from sklearn.metrics import roc_curve; import sklearn; print('sklearn=' + sklearn.__version__)" \
    || { echo "ERROR: sklearn/numpy ABI still broken on $(hostname); aborting"; exit 1; }

# Deep-import validation: trigger the EXACT lazy-load chain SGLang's
# launch_server runs when starting the model. Pre-flight `import sglang;
# import transformers` is INSUFFICIENT — transformers uses _get_module
# lazy loading, so scipy/sklearn and other deep modules don't fault until
# ~60s into model load, then a 20-min health-check timeout. This forces
# all lazy loads at sbatch start so any other broken module surfaces in
# 30s with a named traceback, not 20min of mystery.
python -c "
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.generation import GenerationMixin
from sklearn.metrics import roc_curve
import scipy.optimize, scipy.linalg, codecarbon, pynvml, torch, openai
print('deep-import validation OK')
" || { echo "ERROR: deep-import validation failed on $(hostname). Some lazy-loaded module is broken; identify failing import from traceback above and patch THIS sbatch's env-fix block BEFORE relaunching."; exit 1; }

# BCB-specific runtime libs (not in default env). Install with --no-deps
# to avoid touching numpy/pandas.
python -c "import faker, rsa, scipy, requests" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet faker rsa scipy requests

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1
export SGLANG_CONTEXT_LENGTH=32768

# Port: BASE 51000, max offset (job_id%1000)*10 + 3 = ~10003 → max 61003. Safe.
PORT=$((51000 + SLURM_ARRAY_JOB_ID % 1000 * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=${N_TASKS:-200}
K_VALUES="10"

# 4 array tasks: one topology each
case $SLURM_ARRAY_TASK_ID in
    0) MODE=decentralized                ; OUT_ROOT=mas-energy/results/a5000_latent_transcripts ;;
    1) MODE=independent_share            ; OUT_ROOT=mas-energy/results/a5000_phase_b2_terse     ;;
    2) MODE=decentralized_minimal        ; OUT_ROOT=mas-energy/results/a5000_phase_a_ablation   ;;
    3) MODE=independent_share_minimal    ; OUT_ROOT=mas-energy/results/a5000_phase_b2_terse     ;;
esac

# IMPORTANT: OUT_ROOT must start with mas-energy/ so output lands under
# mas_project/mas-energy/results/, which is where pull_results.sh targets.
# An earlier version omitted "mas-energy/" and BCB results from job 15316650
# landed at /atlas2/u/$USER/mas_project/results/... (orphan path) — they
# need to be rsync'd from the orphan location, see pull_bcb_orphan.sh.
OUT_DIR=/atlas2/u/$USER/mas_project/$OUT_ROOT/bigcodebench
mkdir -p "$OUT_DIR"

echo "=== BigCodeBench: ${MODE} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)  Port: $PORT  N_TASKS: $N_TASKS"
RESULT_FILE="$OUT_DIR/Qwen_Qwen3.5-9B_${MODE}_k10.jsonl"
if [ -f "$RESULT_FILE" ]; then
    echo "Existing rows (resume target): $(wc -l < $RESULT_FILE)"
fi

if lsof -ti :$PORT > /dev/null 2>&1; then kill $(lsof -ti :$PORT) 2>/dev/null; sleep 3; fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!
trap "kill $SGLANG_PID 2>/dev/null || true" EXIT

echo "Waiting for SGLang server..."
# Bump health-check window to 20 min (was 6 min). Decent BCB array 0 in
# job 15316650 failed because Qwen3.5-9B model load (~3.5 min) + cuda
# graph capture (~3 min on first run) exceeded the prior 360s window.
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "ready ${i}s"; break; fi
    if [ $i -eq 600 ]; then echo "ERROR: SGLang failed to come up within 20min"; exit 1; fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks bigcodebench \
    --topologies $MODE \
    --max-react-steps $K_VALUES \
    --n-tasks $N_TASKS --n-reps 1 \
    --save-transcripts \
    --output-dir "$OUT_DIR" \
    --sglang-url "http://localhost:$PORT/v1" \
    || echo "WARNING: bigcodebench/${MODE} run had errors"

echo "=== Done at $(date) ==="
[ -f "$RESULT_FILE" ] && echo "Final rows: $(wc -l < $RESULT_FILE)"
ls -lh "$RESULT_FILE" 2>/dev/null
