#!/usr/bin/env bash
set -euo pipefail

readonly ENV_NAME="OMG"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "Activate the conda environment first: conda activate ${ENV_NAME}" >&2
  exit 1
fi

python pipeline.py validate-inputs
python pipeline.py check-stage2
python pipeline.py validate-masks
python pipeline.py validate-synthesis
python -m pytest -q
ruff check occlusion_pipeline pipeline.py tests

echo "Release verification passed."
