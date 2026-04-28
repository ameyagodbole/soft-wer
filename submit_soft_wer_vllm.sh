#!/bin/bash
#SBATCH --job-name=soft-wer-vllm-sample
#SBATCH --output=/home/ec2-user/logs/soft-wer-vllm-sample-%A.txt
#SBATCH --partition=gpu-ondemand
#SBATCH --gres=gpu:1
#SBATCH --constraint="g6e.xlarge"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

export TRANSFORMERS_OFFLINE=1

MODEL="${MODEL:-Qwen/Qwen3-4B}"
INPUT="${INPUT:-sample_wer_log.json}"
OUTPUT="${OUTPUT:-results/sample_soft_wer_vllm.json}"

mkdir -p "$(dirname "$OUTPUT")"

nvidia-smi

set -x

uv run soft-wer \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --max-tokens 4096 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 50
