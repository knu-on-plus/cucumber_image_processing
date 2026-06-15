#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"
OUT_ROOT="${2:-./outputs}"

if ! command -v python >/dev/null 2>&1; then
  echo "Python not found in PATH. Install Python 3.10+ and retry." >&2
  exit 1
fi

python modal_mask_generation.py \
  --data_root "${DATA_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --dataset_type debugging \
  --position random \
  --multi_leaves 0 \
  --random_ratio true \
  --sample_limit 2
