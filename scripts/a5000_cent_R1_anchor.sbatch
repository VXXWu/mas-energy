#!/bin/bash
#SBATCH --job-name=cent-R1
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_cent_R1_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_cent_R1_%A_%a.err

# Active-learning anchor cells for the Cent R=1 regime — the one regime where
# even tree models fail (per 2026-05-29 architecture comparison).
#
# Current Cent R=1 coverage (sparse, hence the prediction failures):
#   k=8  M=5: 4%    (LHS cell)
#   k=10 M=3: 11%   (R-scan baseline)
#   k=30 M=2: 50%   (v4, predicted 84% by HistGBM, error +34pp)
#   k=50 M=2: 74%   (v4, predicted 88% by HistGBM, error +14pp)
#   k=50 M=3: 80%   (R-scan)
#
# Gap: no Cent R=1 cells in k ∈ [11, 29]. HistGBM tree splits jump from
# 11% (k=10) to 50% (k=30) with no intermediate constraint. Adding k=15, k=20,
# k=22 anchors the regime so future tree predictions of (Cent, low-R, mid-k)
# cells are bounded by observed data, not by tree extrapolation guesswork.
#
# After these land, retraining HistGBM should produce reliable v6 Pareto
# predictions for any Cent low-R configuration.
#
# Array tasks (all at R=1, M=2, the most-failing combo):
#  0 → centralized k=15 R=1 M=2
#  1 → centralized k=20 R=1 M=2
#  2 → centralized k=22 R=1 M=2
#
# Cost: low-R Cent is fast (1 orchestrator call + M=2 workers + 1 review +
# 1 synthesize = 5 LLM calls per round × 1 round). Each cell ~30-90 min/task,
# n=50 → ~1-2 days per cell. --time=2d is comfortable.

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
sleep $STAGGER
sleep $(( RANDOM % 240 ))

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Scrub stale swebench-worktree pollution from site-packages (see a5000_swebench_lhs.sbatch).
SITE_PKG="${CONDA_ENV}/lib/python3.11/site-packages"
if [ -d "$SITE_PKG" ]; then
    for f in "$SITE_PKG"/*.pth "$SITE_PKG"/*.egg-link; do
        [ -f "$f" ] && grep -q "swebench_worktrees" "$f" 2>/dev/null && rm -f "$f"
    done
fi

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

# Torch+numpy ABI compatibility probe: catches the 2026-05-31 SGLang KV pool init
# failure ("Could not infer dtype of numpy.int64") at sbatch startup so we don't
# burn GPU minutes loading the model just to crash. See rebuild_env.sh for context.
python -c "
import torch, numpy
t = torch.tensor(numpy.array([1, 2, 3], dtype=numpy.int64))
assert t.dtype == torch.int64, f'torch dtype inference broken: got {t.dtype}'
" || { echo 'ERROR: torch+numpy ABI mismatch — see rebuild_env.sh torch comment'; exit 1; }

python -c "
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from transformers.loss.loss_utils import LOSS_MAPPING
from sklearn.metrics import roc_curve
import scipy.optimize, codecarbon, pynvml, torch, openai
print('env OK')
" || { echo "ERROR: env validation failed"; exit 1; }

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SWEBENCH_REPOS=/atlas2/u/$USER/mas_project/swebench_repos
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
TOPO=centralized
R=1
M=2
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench

case $SLURM_ARRAY_TASK_ID in
    0) K=15 ;;
    1) K=20 ;;
    2) K=22 ;;
esac

echo "=== Cent R=1 anchor: ${TOPO} k=${K} R=${R} M=${M} ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

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
    --model toy --benchmarks swebench --topologies $TOPO \
    --max-react-steps $K --rounds $R --n-agents $M \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} k=${K} R=${R} M=${M} failed"

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
