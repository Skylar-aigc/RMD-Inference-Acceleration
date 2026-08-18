#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_8gpu.sh [training-config] [extra rmd-train arguments]
# Example:
#   bash scripts/train_8gpu.sh configs/train_wan_14b.yaml \
#     --data_path /data/prompts.txt --output_dir /data/rmd-output

TRAIN_CONFIG="${1:-configs/train_wan_1_3b.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes 8 \
  src/rmd/cli.py \
  train \
  --config "${TRAIN_CONFIG}" \
  "$@"
