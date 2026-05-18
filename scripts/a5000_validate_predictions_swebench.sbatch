#!/bin/bash
#SBATCH --job-name=val-swe
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_validate_swe_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_validate_swe_%A_%a.err

# Validation of GP-predicted optimal configs on SWE-bench (n=50).
# Each config is a never-measured (topo, k, R, M) tuple from the predict_optimal_config.py
# OOS recommendations. If actual accuracy lands within GP std bounds, the model is validated
# for full (M, R, k) Pareto characterization.
#
# Array task → predicted result (model: GP Matern 2.5 on log(k,R,M))
# 0 → decentralized k=3   R=4 M=2 → 77±1.0% @ 46 kJ   (low-budget anchor)
# 1 → decentralized k=5   R=5 M=2 → 90±1.0% @ 91 kJ   (mid-budget headline)
# 2 → decentralized k=75  R=1 M=4 → 97±1.1% @ 176 kJ  (high-k, low-R, novel cell)
# 3 → centralized   k=50  R=2 M=2 → 100±0.3% @ 279 kJ (energy-saving variant of measured Cent k=50 M=3)

source /sailhome/$USER/.bashrc
conda activate mas-energy
cd $HOME
unset PYTHONPATH

STAGGER=$((SLURM_ARRAY_TASK_ID * 120))
echo "Staggering start by ${STAGGER}s"
sleep $STAGGER
EXTRA_SLEEP=$(( RANDOM % 240 ))
echo "Extra random stagger: ${EXTRA_SLEEP}s"
sleep $EXTRA_SLEEP

CONDA_ENV=/atlas2/u/$USER/miniconda3/envs/mas-energy
export PATH="${CONDA_ENV}/bin:${PATH}"
which python

# Env self-heal (sympy/numpy/scipy/sklearn drift)
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'sympy==1.13.3' 'mpmath>=1.3.0'
python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'numpy<2.0'
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scipy==1.11.4'
python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --force-reinstall --quiet 'scipy==1.11.4'
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scikit-learn==1.5.2'
python -c "from sklearn.metrics import roc_curve" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --force-reinstall --quiet 'scikit-learn==1.5.2'
python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'

# Deep-import validation
python -c "
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.generation import GenerationMixin
from sklearn.metrics import roc_curve
import scipy.optimize, scipy.linalg, codecarbon, pynvml, torch, openai
print('deep-import OK')
" || { echo "ERROR: deep-import validation failed"; exit 1; }

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SWEBENCH_REPOS=/atlas2/u/$USER/mas_project/swebench_repos
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=50
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_swebench

case $SLURM_ARRAY_TASK_ID in
    0) TOPO=decentralized; K=3;  R=4; M=2 ;;
    1) TOPO=decentralized; K=5;  R=5; M=2 ;;
    2) TOPO=decentralized; K=75; R=1; M=4 ;;
    3) TOPO=centralized;   K=50; R=2; M=2 ;;
esac

echo "=== Validate: ${TOPO} k=${K} R=${R} M=${M} (SWE-bench, n=${N_TASKS}) ==="
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
    --benchmarks swebench \
    --topologies $TOPO \
    --max-react-steps $K \
    --rounds $R \
    --n-agents $M \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: ${TOPO} k=${K} R=${R} M=${M} failed"

echo "=== Validate ${TOPO} k=${K} R=${R} M=${M} complete: $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
