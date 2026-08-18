#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_7gpu.sh [training-config] [extra rmd-train arguments]
TRAIN_CONFIG="${1:-configs/train_wan_1_3b_7gpu.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

accelerate launch \
  --config_file configs/accelerate_fsdp_7.yaml \
  --num_processes 7 \
  src/rmd/cli.py train \
  --config "${TRAIN_CONFIG}" \
  "$@"
