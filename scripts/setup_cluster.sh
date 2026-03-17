#!/bin/bash
# Run on scdt.stanford.edu (data transfer node) to download models and data.
#
# Usage:
#   ssh YOUR_CSID@scdt.stanford.edu
#   bash /atlas2/u/$USER/mas_project/mas-energy/scripts/setup_cluster.sh

set -e
echo "=== MAS Energy Cluster Setup ==="
echo "Node: $(hostname)"
echo "User: $USER"
echo "Date: $(date)"

source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

# Create directories
mkdir -p /atlas2/u/$USER/mas_project/mas-energy/{results,logs}
mkdir -p /atlas2/u/$USER/mas_project/mas-energy/results/{pilot,toy,main}

# --- Install/update Python dependencies ---
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q sglang[all] openai codecarbon datasets pandas numpy scipy \
    matplotlib seaborn tqdm huggingface_hub

# --- Download toy model (Qwen3.5-9B) ---
echo ""
echo "[2/4] Downloading Qwen3.5-9B (toy model, ~18GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B')
print('  Done: Qwen3.5-9B')
"

# --- Download production models ---
# echo ""
# echo "[3/4] Downloading production models..."

# echo "  Downloading Qwen3.5-27B-GPTQ-Int4 (~17GB)..."
# python3 -c "
# from huggingface_hub import snapshot_download
# snapshot_download('Qwen/Qwen3.5-27B-GPTQ-Int4')
# print('  Done: Qwen3.5-27B-GPTQ-Int4')
# "

# echo "  Downloading Qwen3.5-35B-A3B-GPTQ-Int4 (~17GB)..."
# python3 -c "
# from huggingface_hub import snapshot_download
# snapshot_download('Qwen/Qwen3.5-35B-A3B-GPTQ-Int4')
# print('  Done: Qwen3.5-35B-A3B-GPTQ-Int4')
# "

# echo "  Downloading GLM-4.7-Flash (~30GB)..."
# python3 -c "
# from huggingface_hub import snapshot_download
# snapshot_download('zai-org/GLM-4.7-Flash')
# print('  Done: GLM-4.7-Flash')
"

# --- Clone/update WorkBench ---
echo ""
echo "[4/4] Setting up WorkBench..."
if [ -d "/atlas2/u/$USER/mas_project/WorkBench" ]; then
    echo "  WorkBench already exists, pulling latest..."
    cd /atlas2/u/$USER/mas_project/WorkBench && git pull
else
    echo "  Cloning WorkBench..."
    git clone https://github.com/olly-styles/WorkBench.git /atlas2/u/$USER/mas_project/WorkBench
fi

echo ""
echo "=== Setup complete ==="
echo "HF cache: $(du -sh $HF_HOME 2>/dev/null | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. SSH to sc.stanford.edu"
echo "  2. Get interactive GPU: srun --time=4:00:00 --account=atlas --partition=atlas-interactive --gres=gpu:1 --constraint=48G --mem=48G --cpus-per-task=8 --pty bash"
echo "  3. Run pilot: cd /atlas2/u/\$USER/mas_project/mas-energy/code && python pilot_test.py --model toy --workbench-path /atlas2/u/\$USER/mas_project/WorkBench"
echo "  4. Or submit batch: sbatch /atlas2/u/\$USER/mas_project/mas-energy/scripts/pilot.sbatch"
