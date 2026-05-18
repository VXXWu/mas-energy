#!/bin/bash
# Download Qwen3-8B to cluster HF cache.
# Run on scdt (data transfer node): bash mas-energy/scripts/download_qwen3_8b.sh

source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache

echo "Downloading Qwen/Qwen3-8B to $HF_HOME ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B')
print('Done')
"
echo "Verify: ls $HF_HOME/hub/models--Qwen--Qwen3-8B/"
