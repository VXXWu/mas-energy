#!/bin/bash
#SBATCH --job-name=lcb-disc
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_lcb_disc_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_lcb_disc_%A_%a.err

# LiveCodeBench discrimination experiment -- tests whether SWE-bench's weird
# scaling behaviors (low-k cliff, strong R-scaling, sharp k×R×M interactions)
# come from verifiable+hard+iterative alone, OR require multi-file decomposability.
#
# LiveCodeBench shares with SWE-bench: verifiable (unit tests), iterative
# (write→test→fix), hard (~15-35% pass@1 base).
# LiveCodeBench differs from SWE-bench: single-function (not multi-file).
#
# If LCB shows SWE-bench-like behavior → multi-file is NOT the driver (n=2 positive class)
# If LCB shows BCB-like flat behavior → multi-file IS the driver (scopes story)
#
# Minimal design: k-sweep (R=2 M=3) + one R-scan point + one M-scan point.
# 10 cells total at n=50 each.
#
# Array tasks:
#  0-5: SAS k-sweep at k = {1, 3, 7, 15, 30, 50}  → reveals floor structure
#  6:   Decent k=10 R=2 M=3                       → baseline MAS comparison
#  7:   Decent k=10 R=5 M=3                       → tests R-scaling (vs R=2)
#  8:   Decent k=10 R=2 M=5                       → tests M-scaling (vs M=3)
#  9:   Cent   k=10 R=2 M=3                       → orchestrator comparison

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
sleep $STAGGER
sleep $(( RANDOM % 240 ))

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"

# Scrub stale swebench-worktree pollution from site-packages (see a5000_swebench_lhs.sbatch
# for full explanation). LCB doesn't write to swebench_worktrees but jobs share site-packages
# across all sbatches on this env, so the pollution from SWE-bench tasks affects LCB jobs.
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
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scipy==1.11.4'
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scikit-learn==1.5.2'
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'

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
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_livecodebench

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=sas;           K=1;  R=2; M=3 ;;
    1) TOPO=sas;           K=3;  R=2; M=3 ;;
    2) TOPO=sas;           K=7;  R=2; M=3 ;;
    3) TOPO=sas;           K=15; R=2; M=3 ;;
    4) TOPO=sas;           K=30; R=2; M=3 ;;
    5) TOPO=sas;           K=50; R=2; M=3 ;;
    6) TOPO=decentralized; K=10; R=2; M=3 ;;
    7) TOPO=decentralized; K=10; R=5; M=3 ;;
    8) TOPO=decentralized; K=10; R=2; M=5 ;;
    9) TOPO=centralized;   K=10; R=2; M=3 ;;
esac

echo "=== LCB discrimination: ${TOPO} k=${K} R=${R} M=${M} ==="

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
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then echo "ready"; break; fi
    if [ $i -eq 600 ]; then echo "failed"; kill $SGLANG_PID; exit 1; fi
    sleep 2
done

cd /atlas2/u/$USER/mas_project/mas-energy/code

python run_experiments.py \
    --model toy --benchmarks livecodebench --topologies $TOPO \
    --max-react-steps $K --rounds $R --n-agents $M \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} k=${K} R=${R} M=${M} failed"

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
