#!/bin/bash
#SBATCH --job-name=v6-prose
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-8%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_v6_prose_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_v6_prose_%A_%a.err

# v6 cross-benchmark HistGBM validation — PROSE benchmarks (FanOutQA, WorkBench, BCP+).
#
# Purpose: methodological symmetry with SWE-bench cube sampling. HistGBM applied
# per-benchmark predicts these cells are NEAR the existing Pareto frontier
# (within 80% of observed max accuracy). Running them gives direct empirical
# evidence that prose benchmarks are saturated, vs the current paper claim
# resting on negative model output alone.
#
# Per-cell timing estimates (n=50, from historical data):
#   FanOutQA  : median 4.2h, p90 5.9h
#   WorkBench : median 3.1h, p90 4.9h
#   BCP+      : median 7.1h, p90 16.0h
# Total with %4 concurrency: ~7-8 hours wall-clock. 2-day cap is comfortable.
#
# Array tasks (9 cells total):
#  0-3 → FanOutQA  (Decent k=30 R=5 M=5, Decent k=7 R=4 M=3, Cent k=20 R=3 M=3, Cent k=7 R=4 M=3)
#  4-7 → WorkBench (Decent k=7 R=2 M=2, Decent k=30 R=2 M=2, Cent k=7 R=2 M=4, Cent k=30 R=2 M=2)
#  8   → BCP+      (Decent k=10 R=3 M=2)
#
# Expected outcome: most cells confirm saturation (predicted acc within 5pp of
# observed max). If any cell substantially extends the frontier (>2pp above
# previous best for its energy budget), HistGBM cross-bench prediction is even
# more valuable than v6 baseline suggested.

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 90))
sleep $STAGGER
sleep $(( RANDOM % 180 ))

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Scrub stale swebench-worktree pollution from site-packages
SITE_PKG="${CONDA_ENV}/lib/python3.11/site-packages"
if [ -d "$SITE_PKG" ]; then
    for f in "$SITE_PKG"/*.pth "$SITE_PKG"/*.egg-link; do
        [ -f "$f" ] && grep -q "swebench_worktrees" "$f" 2>/dev/null && rm -f "$f"
    done
fi

# Env self-heal (full stack)
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

# Torch+numpy ABI probe (catches 2026-05-31 SGLang KV pool init failure pattern)
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
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50

case $SLURM_ARRAY_TASK_ID in
    0) BENCH=fanoutqa;        TOPO=decentralized;   K=30; R=5; M=5; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4 ;;
    1) BENCH=fanoutqa;        TOPO=decentralized;   K=7;  R=4; M=3; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4 ;;
    2) BENCH=fanoutqa;        TOPO=centralized;     K=20; R=3; M=3; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4 ;;
    3) BENCH=fanoutqa;        TOPO=centralized;     K=7;  R=4; M=3; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_fanoutqa_v4 ;;
    4) BENCH=workbench;       TOPO=decentralized;   K=7;  R=2; M=2; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2 ;;
    5) BENCH=workbench;       TOPO=decentralized;   K=30; R=2; M=2; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2 ;;
    6) BENCH=workbench;       TOPO=centralized;     K=7;  R=2; M=4; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2 ;;
    7) BENCH=workbench;       TOPO=centralized;     K=30; R=2; M=2; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_workbench_v2 ;;
    8) BENCH=browsecomp_plus; TOPO=decentralized;   K=10; R=3; M=2; OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot ;;
esac

echo "=== v6 prose-bench cell: ${BENCH} ${TOPO} k=${K} R=${R} M=${M} ==="

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
