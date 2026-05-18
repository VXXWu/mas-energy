#!/bin/bash
# Rebuild the mas-energy conda environment from scratch.
# Run from the cluster login node (sc).
#
# Usage:
#   bash mas-energy/scripts/rebuild_env.sh
#
# This removes and recreates the entire env to fix cascading pip breakage.
# Model weights, results, and code are untouched (they live outside the env).

set -e

echo "=== Removing old mas-energy env ==="
conda deactivate 2>/dev/null || true
conda env remove -n mas-energy -y

echo "=== Creating fresh mas-energy env (Python 3.11) ==="
conda create -n mas-energy python=3.11 -y

echo "=== Activating ==="
source ~/.bashrc
conda activate mas-energy

echo "=== Installing core packages (pinned versions) ==="
# torch + CUDA (must match cluster CUDA driver)
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128

# SGLang (the inference server)
pip install "sglang[all]==0.5.10"
# NOTE: Do NOT install sglang-kernel separately -- --force-reinstall pulls
# transitive deps that break sympy/torch. sglang[all]==0.5.10 installs the
# correct sgl_kernel for the GPU architecture.

# HuggingFace stack
pip install transformers==5.3.0 datasets==4.8.4 huggingface_hub tokenizers accelerate

# Energy measurement
pip install codecarbon pynvml

# ML / analysis
pip install scikit-learn==1.6.1 numpy==1.26.4 pandas scipy matplotlib seaborn

# LLM client
pip install openai==2.6.1

# Benchmark-specific
pip install fanoutqa rank_bm25

# Template engine (SGLang needs >=3.1)
pip install "jinja2>=3.1.4"

# Sympy (torch dependency — pin to known-good version)
pip install sympy==1.13.3

# NOTE: vllm removed -- it overwrites sgl_kernel with SM100-only binaries,
# breaking SGLang on SM86 (atlas24 A5000). Use a separate conda env for vllm.

# Misc
pip install tqdm

echo ""
echo "=== Verifying imports ==="
python -c "
import sys
print(f'Python {sys.version}')
failures = []
for name in ['sympy','sklearn','transformers','torch','jinja2',
             'sglang','openai','numpy','pandas','codecarbon',
             'pynvml','fanoutqa','datasets']:
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  {name:<20} {ver}')
    except Exception as e:
        print(f'  {name:<20} FAILED: {e}')
        failures.append(name)
try:
    from sglang.srt.server_args import prepare_server_args
    print(f'  sglang.server_args   ok')
except Exception as e:
    print(f'  sglang.server_args   FAILED: {e}')
    failures.append('sglang.server_args')
print()
if failures:
    print(f'BROKEN: {failures}')
    sys.exit(1)
else:
    print('ALL IMPORTS OK')
"

echo ""
echo "=== Cleanup stale pip artifacts ==="
rm -rf /atlas2/u/$USER/miniconda3/envs/mas-energy/lib/python3.11/site-packages/~etuptools* 2>/dev/null
rm -f /atlas2/u/$USER/miniconda3/envs/mas-energy/lib/python3.11/site-packages/matplotlib-*-nspkg.pth 2>/dev/null

echo ""
echo "=== Done. Env rebuilt successfully. ==="
