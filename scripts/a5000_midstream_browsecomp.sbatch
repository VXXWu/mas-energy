#!/bin/bash
#SBATCH --job-name=ms-browsecomp
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_midstream_browsecomp_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_midstream_browsecomp_%A_%a.err

# Mid-stream injection sweep on BrowseComp+ at k=10, n=100.
# Tests whether continuous-flow peer-injection (run_decentralized_midstream)
# produces the same compute-vs-accuracy curve as standard debate-rounds R.
# If yes → R is essentially raw compute injection (no structural mechanism).
# If no → standard Decent's commit→debate→revise IS structurally meaningful.
#
# Array: 0-3 = n_checkpoints (1,2,3,5) at k=10, M=3, decentralized_midstream.
# Compare to existing a5000_rscan_k10_browsecomp.sbatch (standard decent R-scan).

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
# Stagger BEFORE pip install so concurrent array tasks don't race on
# site-packages writes. (Originally the stagger was placed right before
# SGLang launch, but pip install ran first and could collide across tasks.)
STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
echo "Staggering start by ${STAGGER}s"
sleep $STAGGER
EXTRA_SLEEP=$(( RANDOM % 240 ))
echo "Extra random stagger: ${EXTRA_SLEEP}s"
sleep $EXTRA_SLEEP




CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"
which python
# Install sympy if missing — recurring env drift on this cluster (see
# ~/.claude/bug_fixes.md "energy_scaling SGLang launch fails with ModuleNotFoundError: sympy").
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'sympy==1.13.3' 'mpmath>=1.3.0'


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

python -c "import sglang; import sympy; import jinja2; import openai; import pandas; import pynvml; import codecarbon; import torch; import transformers; print('env ok: sglang=' + sglang.__version__ + ' sympy=' + sympy.__version__ + ' jinja2=' + jinja2.__version__)" \
    || { echo "ERROR: env validation failed on $(hostname)"; exit 1; }

# Robust jinja2 version fix.
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1), jinja2.__version__" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1), f'jinja2 still at {jinja2.__version__}'" \
    || { echo "ERROR: jinja2 upgrade failed on $(hostname); aborting"; exit 1; }
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152



PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=100
K=10
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot

# n_checkpoints values: 1, 2, 3, 5 (skip 4 to match R-scan parity)
R_VALUES=(1 2 3 5)
R=${R_VALUES[$SLURM_ARRAY_TASK_ID]}
TOPO=decentralized_midstream

echo "=== Midstream BrowseComp+: ${TOPO} k=${K} n_checkpoints=${R} n=${N_TASKS} ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT \
    --tp 1 \
    --mem-fraction-static 0.88 \
    --context-length 98304 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!

# 20-min health-check window (matches mscan/BCB after diagnosed timeout issue)
for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    if [ $i -eq 600 ]; then
        echo "ERROR: SGLang failed to start within 20min"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy \
    --benchmarks browsecomp_plus \
    --topologies $TOPO \
    --max-react-steps $K \
    --rounds $R \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: midstream BrowseComp+ k=${K} R=${R} failed"

echo "=== Midstream BrowseComp+ R=${R} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
