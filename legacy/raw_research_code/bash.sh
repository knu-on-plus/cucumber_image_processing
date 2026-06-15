#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./bash.sh [DATA_ROOT] [OUT_ROOT]
# Example:
#   ./bash.sh ./data ./outputs

DATA_ROOT="${1:-./data}"
OUT_ROOT="${2:-./outputs}"

python modal_mask_generation.py \
  --data_root "${DATA_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --dataset_type train \
  --position random \
  --multi_leaves 2 \
  --random_ratio true \
  --sample_limit 100
