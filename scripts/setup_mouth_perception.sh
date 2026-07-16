#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/assets/mediapipe"
MODEL_PATH="${MODEL_DIR}/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

"${PROJECT_DIR}/venv/bin/pip" install -r "${PROJECT_DIR}/requirements/mouth_perception.txt"
mkdir -p "${MODEL_DIR}"
if [ ! -s "${MODEL_PATH}" ]; then
  curl -L --fail --show-error --output "${MODEL_PATH}" "${MODEL_URL}"
fi
echo "Mouth-perception dependencies and model are ready: ${MODEL_PATH}"
