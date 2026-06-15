#!/usr/bin/env bash
set -euo pipefail

readonly ENV_NAME="OMG"
readonly CHECKPOINT="checkpoints/sam2_hiera_large.pt"
readonly CONFIG="configs/quickstart.json"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "Activate the conda environment first: conda activate ${ENV_NAME}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing ${CHECKPOINT}. Run ./scripts/download_sam2_checkpoint.sh first." >&2
  exit 1
fi

if ! python pipeline.py validate-inputs >/dev/null; then
  python pipeline.py prepare-sample --overwrite
fi

python pipeline.py segment-cucumbers \
  --checkpoint "${CHECKPOINT}" \
  --model-config sam2_hiera_l.yaml \
  --overwrite
python pipeline.py synthesize \
  --config "${CONFIG}" \
  --overwrite
python pipeline.py validate-masks
python pipeline.py validate-synthesis

echo "Quickstart complete: outputs/quickstart/synthesis"
