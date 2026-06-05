#!/bin/bash
#SBATCH --job-name=val-swe3
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-5%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_validate_swe_v3_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_validate_swe_v3_%A_%a.err

# THIRD-ROUND validation: Pareto-DOMINANT predictions from the modular model.
# Each of these configs is predicted to beat the observed Pareto frontier
# at its budget. If validated, they extend the frontier.
#
# Observed frontier had a gap between 83 kJ (74%) and 141 kJ (92%).
# All 6 candidates fall in this 80-135 kJ band, predicted to fill that gap.
#
# 0 → decentralized k=10 R=3 M=2 → 82% @ 97 kJ   (dominates 80%@119kJ SAS)
# 1 → centralized   k=7  R=5 M=2 → 81% @ 99 kJ   (dominates 80%@119kJ SAS)
# 2 → decentralized k=15 R=2 M=2 → 84% @101 kJ   (dominates 80,82%)
# 3 → decentralized k=7  R=5 M=2 → 83% @109 kJ   (dominates 80,82%)
# 4 → decentralized k=10 R=4 M=2 → 89% @131 kJ   (dominates 82%@136kJ)
# 5 → centralized   k=30 R=2 M=2 → 85% @135 kJ   (dominates 82%@136kJ)

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
sleep $STAGGER
EXTRA_SLEEP=$(( RANDOM % 240 ))
sleep $EXTRA_SLEEP

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Env self-heal
# Scrub stale swebench-worktree pollution from site-packages (see a5000_swebench_lhs.sbatch
# for full explanation). Triggered by 2026-05-25 crash batch: 11 cells died on
# ModuleNotFoundError: No module named 'requests' from leftover swe_psf__requests worktree .pth.
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
    0) TOPO=decentralized; K=10; R=3; M=2 ;;
    1) TOPO=centralized;   K=7;  R=5; M=2 ;;
    2) TOPO=decentralized; K=15; R=2; M=2 ;;
    3) TOPO=decentralized; K=7;  R=5; M=2 ;;
    4) TOPO=decentralized; K=10; R=4; M=2 ;;
    5) TOPO=centralized;   K=30; R=2; M=2 ;;
esac

echo "=== v3 Pareto-dominant validation: ${TOPO} k=${K} R=${R} M=${M} ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

if lsof -ti :$PORT > /dev/null 2>&1; then
    kill $(lsof -ti :$PORT) 2>/dev/null
    sleep 3
fi

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port $PORT --tp 1 \
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
