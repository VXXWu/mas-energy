#!/bin/bash
#SBATCH --job-name=gpu-diag
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/atlas2/u/%u/mas_project/mas-energy/logs/gpu_diag_%A_%a.out
#SBATCH --error=/atlas2/u/%u/mas_project/mas-energy/logs/gpu_diag_%A_%a.err

source /sailhome/$USER/.bashrc
conda activate mas-energy

cd /atlas2/u/$USER/mas_project/mas-energy/code
python gpu_diagnostic.py
