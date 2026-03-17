#!/bin/bash
#SBATCH --job-name=mas-qwen14b
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/atlas2/u/%u/mas-energy/logs/qwen14b_%j.out
#SBATCH --error=/atlas2/u/%u/mas-energy/logs/qwen14b_%j.err

source /sailhome/$USER/.bashrc
conda activate mas-energy
export HF_HOME=/atlas2/u/$USER/hf_cache

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --disable-log-requests \
    --port 8000 &
VLLM_PID=$!

for i in $(seq 1 60); do
    curl -s http://localhost:8000/health && break
    sleep 5
done

cd /atlas2/u/$USER/mas-energy/code
python run_experiments.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --gpu-index 0 \
    --output-dir /atlas2/u/$USER/mas-energy/results/qwen14b \
    --n-tasks 200 \
    --n-reps 5 \
    --benchmarks gsm8k math mmlu \
    --topologies sas independent centralized decentralized hybrid

kill $VLLM_PID
