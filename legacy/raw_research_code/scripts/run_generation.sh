#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"
OUT_ROOT="${2:-./outputs}"
DATASET_TYPE="${3:-debugging}"
POSITION="${4:-random}"
MULTI_LEAVES="${5:-0}"
RANDOM_RATIO="${6:-true}"
SAMPLE_LIMIT="${7:-5}"

python modal_mask_generation.py \
  --data_root "${DATA_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --dataset_type "${DATASET_TYPE}" \
  --position "${POSITION}" \
  --multi_leaves "${MULTI_LEAVES}" \
  --random_ratio "${RANDOM_RATIO}" \
  --sample_limit "${SAMPLE_LIMIT}"
