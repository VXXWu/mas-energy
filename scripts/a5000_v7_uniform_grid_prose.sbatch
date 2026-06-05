#!/bin/bash
#SBATCH --job-name=v7-grid
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-26%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_v7_grid_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_v7_grid_%A_%a.err

# v7 uniform cube-interior grid for prose benchmarks.
#
# Motivation: v6 HistGBM cross-benchmark predictions over-predicted FanOutQA
# accuracy by 37-54pp (e.g., Cent k=20 R=3 M=3 predicted 67%, actual 30%).
# Root cause: prose benchmarks have ZERO cube-interior training cells — only
# 3 orthogonal 1D scaling sweeps (k-axis, R-axis, M-axis at default anchors).
# HistGBM has nothing to constrain interpolation off those axis lines.
#
# Fix: methodologically symmetric uniform grid sampling. 9 cells per benchmark
# × 3 benchmarks = 27 total. Direct empirical Pareto evidence (no model
# uncertainty), AND provides training data so any future HistGBM picks for
# prose benchmarks are anchored against real interior observations.
#
# Grid (Decentralized, M=2 — matches SWE-bench Pareto pattern):
#   k ∈ {7, 15, 30}  (mid-knee, well-past-cliff, near-saturation)
#   R ∈ {2, 3, 5}    (default, mid, max-studied)
#   M = 2            (Pareto-dominant M per SWE-bench analysis)
#
# Per-benchmark timing (5hr median per cell × 9 cells, %4 concurrency on
# atlas24 with this sbatch alone): ~11 hours wall-clock per benchmark.
# With %4 across all 27 array tasks: ~9-12 hours total. Fits comfortably
# in 2-day --time cap.
#
# Array tasks (3 benchmarks × 9 cells each):
#   0-8:   FanOutQA   (k=7/15/30 × R=2/3/5, M=2 fixed)
#   9-17:  WorkBench  (same pattern)
#   18-26: BrowseComp+ (same pattern)

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
sleep $STAGGER
sleep $(( RANDOM % 180 ))

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Scrub stale swebench-worktree pollution
SITE_PKG="${CONDA_ENV}/lib/python3.11/site-packages"
if [ -d "$SITE_PKG" ]; then
    for f in "$SITE_PKG"/*.pth "$SITE_PKG"/*.egg-link; do
        [ -f "$f" ] && grep -q "swebench_worktrees" "$f" 2>/dev/null && rm -f "$f"
    done
fi

# Env self-heal stack (same as v6_prose sbatch)
python -c "import requests; assert 'site-packages' in requests.__file__" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'requests>=2.31,<3'
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'sympy==1.13.3' 'mpmath>=1.3.0'
python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'numpy<2.0'
python -c "import pandas; assert pandas.__version__.startswith('2.')" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'pandas==2.1.4' pytz tzdata
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scipy==1.11.4'
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scikit-learn==1.5.2'
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'

# Torch+numpy ABI probe — catches the 2026-05-31 cascade pattern
python -c "
import torch, numpy
t = torch.tensor(numpy.array([1, 2, 3], dtype=numpy.int64))
assert t.dtype == torch.int64, f'torch dtype inference broken: got {t.dtype}'
" || { echo 'ERROR: torch+numpy ABI mismatch — env rebuild needed'; exit 1; }

python -c "
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from transformers.loss.loss_utils import LOSS_MAPPING
from sklearn.metrics import roc_curve
import scipy.optimize, codecarbon, pynvml, torch, openai
print('env OK')
" || { echo "ERROR: env validation failed"; exit 1; }

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
TOPO=decentralized
M=2

# Decode array index → (benchmark, k, R)
BENCH_IDX=$(( SLURM_ARRAY_TASK_ID / 9 ))
CELL_IDX=$(( SLURM_ARRAY_TASK_ID % 9 ))
# CELL_IDX is 0..8, decoded as (k_idx * 3 + r_idx)
K_IDX=$(( CELL_IDX / 3 ))
R_IDX=$(( CELL_IDX % 3 ))
K_VALUES=(7 15 30)
R_VALUES=(2 3 5)
K=${K_VALUES[$K_IDX]}
R=${R_VALUES[$R_IDX]}

case $BENCH_IDX in
    0) BENCH=fanoutqa;        OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4 ;;
    1) BENCH=workbench;       OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2 ;;
    2) BENCH=browsecomp_plus; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot ;;
esac

echo "=== v7 grid: ${BENCH} ${TOPO} k=${K} R=${R} M=${M} (array idx ${SLURM_ARRAY_TASK_ID}: bench ${BENCH_IDX}, cell ${CELL_IDX}) ==="

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B --port $PORT --tp 1 \
    --mem-fraction-static 0.88 --context-length 98304 \
    --tool-call-parser qwen3_coder --trust-remote-code \
    --host 0.0.0.0 &
SGLANG_PID=$!

for i in $(seq 1 600); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "SGLang ready"; break; fi
    if [ $i -eq 600 ]; then echo "SGLang failed"; kill $SGLANG_PID; exit 1; fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy --benchmarks $BENCH --topologies $TOPO \
    --max-react-steps $K --rounds $R --n-agents $M \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${BENCH} ${TOPO} k=${K} R=${R} M=${M} failed"

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
