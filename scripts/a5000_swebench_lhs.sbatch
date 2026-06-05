#!/bin/bash
#SBATCH --job-name=swe-lhs
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-15%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_swe_lhs_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_swe_lhs_%A_%a.err

# SWE-bench LHS exploration grid: 16 cells across (k, R, M) cube using a
# Latin Hypercube design. Motivation: R-scaling appears most energy-efficient
# in isolation, but individual axes show diminishing returns on log-scale.
# True compute-optimal configs likely sit at multi-axis interiors where each
# dimension is in its log-knee. Pure axis-sweeps + 14 prior off-default cells
# (intermediate_grid + v3) leave the interior under-sampled.
#
# Design (LHS with stratification):
#   k:  log-spaced 1-50 → strata centers {1, 2, 3, 5, 8, 13, 22, 36}
#   R:  uniform 1-5     → strata {1, 1, 2, 3, 3, 4, 4, 5} (paired across topo)
#   M:  uniform 2-5     → strata {2, 2, 3, 3, 4, 4, 5, 5} (paired across topo)
#   topology: 8 cent + 8 decent
#
# Combinations checked against intermediate_grid + v3 + standard sweep cells;
# no duplicates.
#
# Array tasks:
#  0 → cent   k=2  R=4 M=3      8 → decent k=5  R=4 M=2
#  1 → cent   k=8  R=1 M=5      9 → decent k=22 R=5 M=4
#  2 → cent   k=22 R=3 M=2     10 → decent k=2  R=3 M=5
#  3 → cent   k=3  R=5 M=4     11 → decent k=13 R=4 M=3
#  4 → cent   k=13 R=2 M=2     12 → decent k=36 R=1 M=2
#  5 → cent   k=36 R=3 M=5     13 → decent k=8  R=2 M=4
#  6 → cent   k=5  R=2 M=4     14 → decent k=3  R=3 M=5
#  7 → cent   k=1  R=5 M=3     15 → decent k=1  R=2 M=2

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
sleep $STAGGER
sleep $(( RANDOM % 240 ))

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Scrub stale swebench-worktree pollution from site-packages. SWE-bench tasks
# sometimes run `pip install -e .` inside their worktree (notably swe_psf__requests
# tasks where the package under test IS `requests`), which writes a .pth file
# pointing into the worktree. When the worktree is later removed, the .pth file
# persists in site-packages and shadows the conda env's installed packages,
# producing ModuleNotFoundError: No module named 'requests' (or worse, a Python-2
# era `requests` is loaded that fails on `from collections import Mapping`).
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
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench

case $SLURM_ARRAY_TASK_ID in
    0)  TOPO=centralized;   K=2;  R=4; M=3 ;;
    1)  TOPO=centralized;   K=8;  R=1; M=5 ;;
    2)  TOPO=centralized;   K=22; R=3; M=2 ;;
    3)  TOPO=centralized;   K=3;  R=5; M=4 ;;
    4)  TOPO=centralized;   K=13; R=2; M=2 ;;
    5)  TOPO=centralized;   K=36; R=3; M=5 ;;
    6)  TOPO=centralized;   K=5;  R=2; M=4 ;;
    7)  TOPO=centralized;   K=1;  R=5; M=3 ;;
    8)  TOPO=decentralized; K=5;  R=4; M=2 ;;
    9)  TOPO=decentralized; K=22; R=5; M=4 ;;
    10) TOPO=decentralized; K=2;  R=3; M=5 ;;
    11) TOPO=decentralized; K=13; R=4; M=3 ;;
    12) TOPO=decentralized; K=36; R=1; M=2 ;;
    13) TOPO=decentralized; K=8;  R=2; M=4 ;;
    14) TOPO=decentralized; K=3;  R=3; M=5 ;;
    15) TOPO=decentralized; K=1;  R=2; M=2 ;;
esac

echo "=== SWE-bench LHS: ${TOPO} k=${K} R=${R} M=${M} ==="
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
