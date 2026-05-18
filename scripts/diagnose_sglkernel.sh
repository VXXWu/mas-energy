#!/bin/bash
# Diagnose sgl_kernel SM86 compatibility issue.
# Run on cluster GPU node: srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=1:00:00 --account=atlas --partition=atlas-interactive --pty bash
# Then: bash mas-energy/scripts/diagnose_sglkernel.sh

source ~/.bashrc
conda activate mas-energy

echo "=== Current sgl_kernel state ==="
python -c "
import sgl_kernel, os, pathlib
pkg_dir = pathlib.Path(sgl_kernel.__file__).parent
print(f'sgl_kernel location: {pkg_dir}')
print(f'Available subdirs: {[d.name for d in pkg_dir.iterdir() if d.is_dir()]}')
for so in pkg_dir.rglob('*.so'):
    print(f'  {so.relative_to(pkg_dir)}')
" 2>&1

echo ""
echo "=== Test the failing import ==="
python -c "from sgl_kernel import sgl_per_token_quant_fp8; print('OK')" 2>&1

echo ""
echo "=== Package versions ==="
pip show sgl-kernel sglang-kernel sglang vllm 2>&1 | grep -E "^(Name|Version|Requires)"

echo ""
echo "=== Test: uninstall sgl-kernel, install sglang-kernel ==="
echo "(DRY RUN -- uncomment to execute)"
# pip uninstall -y sgl-kernel
# pip install sglang-kernel
# python -c "from sgl_kernel import sgl_per_token_quant_fp8; print('sglang-kernel OK')"

echo ""
echo "=== Test: does removing vllm fix sgl_kernel? ==="
echo "(DRY RUN -- uncomment to execute)"
# pip uninstall -y vllm sgl-kernel
# pip install --force-reinstall "sglang[all]==0.5.10"
# python -c "from sgl_kernel import sgl_per_token_quant_fp8; print('post-vllm-removal OK')"
