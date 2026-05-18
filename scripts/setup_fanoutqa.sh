#!/bin/bash
# Set up FanOutQA on the cluster.
# Run on scdt.stanford.edu (data transfer node) for wiki cache download,
# or on a compute node for just the pip install.
#
# Usage:
#   ssh YOUR_CSID@scdt.stanford.edu
#   bash /atlas2/u/$USER/mas_project/mas-energy/scripts/setup_fanoutqa.sh
#
# Options:
#   --skip-cache   Skip wiki cache pre-warming (~9GB download)

set -e
echo "=== FanOutQA Setup ==="
echo "Node: $(hostname)"
echo "User: $USER"
echo "Date: $(date)"

source /sailhome/$USER/.bashrc
conda activate mas-energy

# --- Install fanoutqa + eval deps ---
echo ""
echo "[1/3] Installing fanoutqa..."
pip install -q "fanoutqa[retrieval,eval]"

echo "[2/3] Downloading spaCy model..."
python -m spacy download en_core_web_sm

# --- Pre-warm wiki cache ---
if [[ "$1" != "--skip-cache" ]]; then
    echo ""
    echo "[3/3] Pre-warming Wikipedia cache for dev set (~9GB)..."
    echo "      This downloads all necessary Wikipedia pages to ~/.cache/fanoutqa/wikicache/"
    echo "      so that experiment runs don't hit the network."
    python3 -c "
import fanoutqa
from fanoutqa.wiki import wiki_content
from tqdm import tqdm

questions = fanoutqa.load_dev()
all_evidence = []
for q in questions:
    all_evidence.extend(q.necessary_evidence)

# Deduplicate by pageid
seen = set()
unique = []
for e in all_evidence:
    if e.pageid not in seen:
        seen.add(e.pageid)
        unique.append(e)

print(f'Caching {len(unique)} unique Wikipedia pages...')
errors = 0
for e in tqdm(unique):
    try:
        wiki_content(e)
    except Exception as ex:
        errors += 1
        print(f'  Warning: failed to cache {e.title}: {ex}')

print(f'Done. {len(unique) - errors}/{len(unique)} pages cached.')
print(f'Cache dir: ~/.cache/fanoutqa/wikicache/')
"
else
    echo ""
    echo "[3/3] Skipping wiki cache (--skip-cache)"
fi

echo ""
echo "=== FanOutQA setup complete ==="
echo ""
echo "Verify with:"
echo "  python -c \"import fanoutqa; print(len(fanoutqa.load_dev()), 'questions loaded')\""
