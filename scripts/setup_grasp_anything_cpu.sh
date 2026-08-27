#!/usr/bin/env bash
# Install the pinned official Grasp-Anything model in gitignored directories.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENDOR_DIR="${PROJECT_DIR}/assets/vendor/grasp_anything"
VENV_DIR="${PROJECT_DIR}/.venv/grasp_anything"
UPSTREAM_URL="https://github.com/Fsoft-AIC/Grasp-Anything.git"
UPSTREAM_COMMIT="d7755f43c5518bd6590b25021054f862e65bddd5"
MODEL_SHA256="65984ef3364790c1ece107f22bcbeb67dc8fba21784087bb3d8ff183a3582e0a"

mkdir -p "${PROJECT_DIR}/assets/vendor" "${PROJECT_DIR}/.venv"
if [ ! -d "${VENDOR_DIR}/.git" ]; then
  git clone "${UPSTREAM_URL}" "${VENDOR_DIR}"
fi
git -C "${VENDOR_DIR}" fetch origin "${UPSTREAM_COMMIT}"
git -C "${VENDOR_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

printf '%s  %s\n' \
  "${MODEL_SHA256}" "${VENDOR_DIR}/weights/model_grasp_anything" | sha256sum -c -

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
"${VENV_DIR}/bin/python" -m pip install \
  numpy==2.5.2 scipy==1.18.1 scikit-image==0.26.0 pillow==12.3.0

"${VENV_DIR}/bin/python" -c \
  'import torch; print("Grasp-Anything CPU environment ready:", torch.__version__)'
