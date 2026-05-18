#!/bin/bash
#SBATCH --job-name=latentmas-repo
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_repo_%j.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/latentmas_repo_%j.err

# LatentMAS only (baseline and TextMAS already done: both 95%).
# Requires: vllm installed, Qwen3-8B added to run.py choices.

set -e
source /sailhome/$USER/.bashrc
conda activate mas-latent
export HF_HOME=/atlas2/u/$USER/mas_project/hf_cache
export PYTHONUNBUFFERED=1

# Install vllm if missing
python -c "import vllm" 2>/dev/null || {
    echo "Installing vllm..."
    pip install vllm
}

cd /atlas2/u/$USER/mas_project/LatentMAS

# Add Qwen3-8B to model choices if missing
python3 -c "
t = open('run.py').read()
if 'Qwen/Qwen3-8B' not in t:
    t = t.replace('choices=[\"Qwen/Qwen3-4B\", \"Qwen/Qwen3-4B\", \"Qwen/Qwen3-14B\"]',
                  'choices=[\"Qwen/Qwen3-4B\", \"Qwen/Qwen3-8B\", \"Qwen/Qwen3-14B\"]')
    open('run.py', 'w').write(t)
    print('Added Qwen3-8B to model choices')
else:
    print('Qwen3-8B already in choices')
"

echo "=== LatentMAS (sequential, m=40) on GSM8K ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --task gsm8k \
    --prompt sequential \
    --max_samples 100 \
    --max_new_tokens 2048 \
    --latent_steps 40 \
    --latent_space_realign \
    --generate_bs 1

echo "=== Done: $(date) ==="
