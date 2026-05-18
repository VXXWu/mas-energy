#!/bin/bash
# Build a separate conda env for latent communication experiments.
# vllm conflicts with sglang's sgl_kernel on SM86 (A5000), so latent
# experiments (which need vllm for LatentMAS repo) run in their own env.
#
# Usage (on cluster):
#   bash mas-energy/scripts/build_latent_env.sh

set -e

echo "=== Creating mas-latent env (Python 3.11) ==="
conda deactivate 2>/dev/null || true
conda env remove -n mas-latent -y 2>/dev/null || true
conda create -n mas-latent python=3.11 -y

source ~/.bashrc
conda activate mas-latent

echo "=== Installing packages ==="
# torch + CUDA
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128

# vllm (the reason this env exists — conflicts with sglang on SM86)
pip install vllm

# HuggingFace stack
pip install transformers datasets huggingface_hub tokenizers accelerate

# Energy measurement
pip install codecarbon pynvml

# ML / analysis
pip install scikit-learn numpy pandas scipy matplotlib seaborn

# Benchmark-specific
pip install fanoutqa rank_bm25

# Sympy
pip install sympy==1.13.3

# Misc
pip install tqdm openai jinja2

echo ""
echo "=== Verifying imports ==="
python -c "
import sys
print(f'Python {sys.version}')
for name in ['torch','vllm','transformers','datasets','sympy',
             'codecarbon','pynvml','fanoutqa','numpy','pandas']:
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  {name:<20} {ver}')
    except Exception as e:
        print(f'  {name:<20} FAILED: {e}')
print()
print('ALL OK')
"

echo ""
echo "=== Done. Activate with: conda activate mas-latent ==="
