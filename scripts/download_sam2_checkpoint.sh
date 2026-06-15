#!/usr/bin/env bash
set -euo pipefail

readonly CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
readonly CHECKPOINT_PATH="checkpoints/sam2_hiera_large.pt"
readonly CHECKPOINT_SHA256="7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"

mkdir -p checkpoints
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint already exists; verifying: ${CHECKPOINT_PATH}"
else
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location "${CHECKPOINT_URL}" --output "${CHECKPOINT_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget "${CHECKPOINT_URL}" --output-document="${CHECKPOINT_PATH}"
  else
    echo "curl or wget is required to download the SAM2 checkpoint." >&2
    exit 1
  fi
fi

actual_sha256="$(sha256sum "${CHECKPOINT_PATH}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint checksum mismatch: ${actual_sha256}" >&2
  exit 1
fi

echo "Checkpoint verified: ${CHECKPOINT_PATH}"
