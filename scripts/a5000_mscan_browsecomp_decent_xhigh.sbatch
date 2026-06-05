#!/bin/bash
#SBATCH --job-name=ma-bc-dxh
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=atlas24
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_mscan_browsecomp_decent_xhigh_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/a5k_mscan_browsecomp_decent_xhigh_%A_%a.err

# Extended high-M on BrowseComp+ DECENTRALIZED at k=10, R=2, n=100.
# Adds M=15, 20 — matches Cent's xhigh extension. Existing Decent on
# browsecomp+: M={2,4,5,7,10}. browsecomp+ is M-dominant (search diversity
# benchmark) so Decent should saturate at moderate M; this confirms the
# saturation point. Skipping M=30 (Decent's M^2 context > 98K limit).
#
# Array: 0=M=15, 1=M=20. n=100 each.

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

# Post-env-rebuild (2026-05-11): the env-fix block is a no-op safety net.
python -c "import sympy; v=tuple(map(int, sympy.__version__.split('.')[:2])); assert v >= (1, 13)" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'sympy==1.13.3' 'mpmath>=1.3.0'

python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'numpy<2.0'
python -c "import numpy, pandas; print('numpy=' + numpy.__version__ + ' pandas=' + pandas.__version__)" \
    || { echo "ERROR: numpy/pandas state still broken on $(hostname); aborting"; exit 1; }

python -c "import scipy.optimize" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'scipy==1.11.4'
python -c "import scipy.optimize; print('scipy=' + __import__('scipy').__version__)" \
    || { echo "ERROR: scipy/numpy ABI still broken on $(hostname); aborting"; exit 1; }

python -c "from sklearn.metrics import roc_curve; import sklearn; print('sklearn=' + sklearn.__version__)" \
    || { echo "ERROR: sklearn broken on $(hostname); aborting"; exit 1; }

python -c "
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.generation import GenerationMixin
from sklearn.metrics import roc_curve
import scipy.optimize, scipy.linalg, codecarbon, pynvml, torch, openai
print('deep-import validation OK')
" || { echo "ERROR: deep-import validation failed on $(hostname). Identify failing module from traceback above."; exit 1; }

python -c "import jinja2; v=tuple(map(int, jinja2.__version__.split('.')[:2])); assert v >= (3, 1), jinja2.__version__" 2>/dev/null \
    || python -m pip install --no-deps --no-cache-dir --ignore-installed --quiet 'jinja2==3.1.4'

export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export SGLANG_CONTEXT_LENGTH=49152

PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 4000) * 10 + SLURM_ARRAY_TASK_ID))
N_TASKS=100
K=10
R=2
TOPO=decentralized
OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/a5000_browsecomp_pilot

M_VALUES=(15 20)
M=${M_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=== M-ablation BrowseComp+ DECENT XHIGH: ${TOPO} k=${K} R=${R} M=${M} n=${N_TASKS} ==="
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
    --benchmarks browsecomp_plus \
    --topologies $TOPO \
    --max-react-steps $K \
    --rounds $R \
    --n-agents $M \
    --n-tasks $N_TASKS --n-reps 1 \
    --output-dir $OUTPUT_DIR \
    --sglang-url "http://localhost:$PORT/v1" \
|| echo "WARNING: BrowseComp+ DECENT XHIGH ${TOPO} k=${K} R=${R} M=${M} failed"

echo "=== Done at $(date) ==="

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
