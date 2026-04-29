#!/bin/bash
set -euo pipefail

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
