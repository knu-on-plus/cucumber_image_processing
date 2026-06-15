#!/usr/bin/env bash
set -euo pipefail

readonly ENV_NAME="OMG"
readonly SAM2_REPOSITORY="https://github.com/facebookresearch/sam2.git"
readonly SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

conda_base="$(conda info --base)"
source "${conda_base}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file environment.yml --prune
else
  conda env create --file environment.yml
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade -r requirements.txt

if [[ ! -d external/sam2/.git ]]; then
  git clone "${SAM2_REPOSITORY}" external/sam2
  git -C external/sam2 checkout "${SAM2_COMMIT}"
fi

current_commit="$(git -C external/sam2 rev-parse HEAD)"
if [[ "${current_commit}" != "${SAM2_COMMIT}" ]]; then
  echo "external/sam2 must be at ${SAM2_COMMIT}, found ${current_commit}" >&2
  exit 1
fi

SAM2_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation -e external/sam2

python pipeline.py check-stage2

echo
echo "OMG environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
