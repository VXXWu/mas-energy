#!/bin/bash
# Download Hendrycks MATH dataset on scdt (data transfer node with internet)
# and save to canonical Hendrycks layout that benchmarks_math.py loads
# offline-first on compute nodes.
#
# Run ONCE before running run_math_pilot.sbatch:
#   bash mas-energy/scripts/download_math_dataset.sh
#
# Default target: /atlas2/u/vincewu8/mas_project/mas-energy/data/math/MATH/test/
# Layout: <subject>/<i>.json with {problem, level, type, solution} per file

set -e

USER_REMOTE=vincewu8
HOST=scdt.stanford.edu
REMOTE=${USER_REMOTE}@${HOST}

REMOTE_DATA=/atlas2/u/${USER_REMOTE}/mas_project/mas-energy/data/math

echo "[1/2] Fetching MATH dataset on ${HOST}..."

ssh "$REMOTE" "
set -e
source ~/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/${USER_REMOTE}/mas_project/hf_cache
mkdir -p '${REMOTE_DATA}'

python3 << 'PYEOF'
import json
import os
from pathlib import Path

OUT_ROOT = Path('${REMOTE_DATA}') / 'MATH' / 'test'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Try lighteval/MATH first; fall back to hendrycks/competition_math
ds = None
err = None
for repo, cfg in [('lighteval/MATH', 'all'),
                  ('hendrycks/competition_math', None),
                  ('EleutherAI/hendrycks_math', 'algebra')]:
    try:
        from datasets import load_dataset
        if cfg:
            ds = load_dataset(repo, cfg, split='test', trust_remote_code=True)
        else:
            ds = load_dataset(repo, split='test', trust_remote_code=True)
        print(f'Loaded from {repo}: {len(ds)} problems')
        source_repo = repo
        break
    except Exception as e:
        err = e
        print(f'  {repo} failed: {type(e).__name__}: {str(e)[:200]}')
        continue

if ds is None:
    raise RuntimeError(f'All HF sources failed. Last error: {err}')

# Special handling for EleutherAI mirror: it's split by subject — would need to load all subjects separately.
# But typical case is lighteval/MATH or hendrycks/competition_math, both provide unified test split.

counts = {}
for i, row in enumerate(ds):
    subject = (row.get('type') or 'unknown').lower().replace(' ', '_').replace('&', 'and')
    counts[subject] = counts.get(subject, 0) + 1
    subj_dir = OUT_ROOT / subject
    subj_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'problem': row.get('problem', ''),
        'level': row.get('level', ''),
        'type': row.get('type', ''),
        'solution': row.get('solution', ''),
    }
    fp = subj_dir / f'{counts[subject]-1}.json'
    with open(fp, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)

total = sum(counts.values())
print(f'Wrote {total} files to {OUT_ROOT}')
print(f'Per-subject counts: {dict(sorted(counts.items()))}')

# Sanity check: per-level distribution
levels = {}
for json_path in OUT_ROOT.rglob('*.json'):
    with open(json_path) as f:
        r = json.load(f)
    lv = r.get('level', 'unknown')
    levels[lv] = levels.get(lv, 0) + 1
print(f'Level distribution: {dict(sorted(levels.items()))}')
PYEOF
"

echo ""
echo "[2/2] Sanity check from local end..."
ssh "$REMOTE" "ls -d ${REMOTE_DATA}/MATH/test/*/ 2>/dev/null | head -8 && echo 'Total: '$(ssh "$REMOTE" "find ${REMOTE_DATA}/MATH/test -name '*.json' | wc -l")"

echo ""
echo "Done. The benchmark loader will now read offline from:"
echo "  ${REMOTE_DATA}/MATH/test/<subject>/*.json"
