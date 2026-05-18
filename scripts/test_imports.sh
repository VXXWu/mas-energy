#!/bin/bash
# Test all critical imports for the mas-energy pipeline.
# Run on the cluster: bash mas-energy/scripts/test_imports.sh

source ~/.bashrc
conda activate mas-energy

python -c "
import sys
print(f'Python {sys.version}')
print()

failures = []

for name in ['sympy', 'sklearn', 'transformers', 'torch', 'jinja2',
             'sglang', 'openai', 'numpy', 'pandas', 'codecarbon',
             'pynvml', 'fanoutqa', 'rank_bm25', 'datasets']:
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  {name:<20} {ver}')
    except Exception as e:
        print(f'  {name:<20} FAILED: {e}')
        failures.append(name)

# Test SGLang can actually start (imports its server module)
print()
try:
    from sglang.srt.server_args import prepare_server_args
    print('  sglang.server_args   ok')
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
