#!/usr/bin/env bash
# Phase 5 — LoRA fine-tune of Llama-3-8B-Instruct 4-bit with MLX.
# Usage: scripts/train_lora.sh [ITERS]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ITERS="${1:-600}"
mkdir -p adapters logs

python -m mlx_lm lora \
  --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
  --train --test \
  --data data/processed \
  -c config/lora.yaml \
  --num-layers 16 \
  --batch-size 4 \
  --iters "$ITERS" \
  --learning-rate 1e-5 \
  --mask-prompt \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --save-every 50 \
  --val-batches 25 \
  --max-seq-length 512 \
  --grad-checkpoint \
  --adapter-path adapters 2>&1 | tee logs/train.log
