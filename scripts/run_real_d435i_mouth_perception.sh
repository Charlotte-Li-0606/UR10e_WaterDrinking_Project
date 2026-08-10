#!/usr/bin/env bash
# Run the existing MediaPipe mouth-perception node against a real D435i.
#
# The RealSense driver owns d435i_link -> d435i_color_optical_frame.  This
# script attaches that tree to the UR tool with one project-local static TF,
# then passes real-topic parameters to the existing perception node.  The
# simulation defaults in run_mouth_perception.sh are intentionally untouched.
#
# The project-local JSON file records the current tool0 -> d435i_link mount
# transform and its calibration status. The checked-in provisional transform
# is the physically validated 2026-07-23 axis correction used by the successful
# real pre-mouth runs; the guarded execution path also compares the live TF to
# this file before it permits motion. scripts/calibrate_d435i_mount.py is the
# supported writer for a future metrology-grade verified replacement.
set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
set -u

RGB_TOPIC="${D435I_RGB_TOPIC:-/d435i/d435i/color/image_raw}"
DEPTH_TOPIC="${D435I_DEPTH_TOPIC:-/d435i/d435i/aligned_depth_to_color/image_raw}"
CAMERA_INFO_TOPIC="${D435I_CAMERA_INFO_TOPIC:-/d435i/d435i/color/camera_info}"
MOUNT_CONFIG="${D435I_MOUNT_CONFIG:-${PROJECT_DIR}/config/ur10e_real/d435i_mount_calibration.json}"

if [ ! -f "${MOUNT_CONFIG}" ]; then
  echo "D435i mount configuration is missing: ${MOUNT_CONFIG}" >&2
  exit 2
fi

read -r MOUNT_X MOUNT_Y MOUNT_Z MOUNT_ROLL MOUNT_PITCH MOUNT_YAW MOUNT_STATUS < <(
  python3 - "${MOUNT_CONFIG}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    config = json.load(stream)
transform = config["tool0_to_d435i_link"]
translation = transform["translation_m"]
rpy = transform["rpy_rad"]
status = config.get("calibration_status", "unknown")
if len(translation) != 3 or len(rpy) != 3 or not all(math.isfinite(float(value)) for value in translation + rpy):
    raise SystemExit("invalid tool0_to_d435i_link transform")
if status not in {"provisional", "verified"}:
    raise SystemExit("calibration_status must be provisional or verified")
print(*translation, *rpy, status)
PY
)

ros2 run tf2_ros static_transform_publisher \
  --x "${MOUNT_X}" \
  --y "${MOUNT_Y}" \
  --z "${MOUNT_Z}" \
  --roll "${MOUNT_ROLL}" \
  --pitch "${MOUNT_PITCH}" \
  --yaw "${MOUNT_YAW}" \
  --frame-id tool0 \
  --child-frame-id d435i_link &
TF_PID=$!

cleanup() {
  kill -INT "${TF_PID}" 2>/dev/null || true
  wait "${TF_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${PROJECT_DIR}/scripts/run_mouth_perception.sh" \
  --rgb-topic "${RGB_TOPIC}" \
  --depth-topic "${DEPTH_TOPIC}" \
  --camera-info-topic "${CAMERA_INFO_TOPIC}" \
  --output-topic /detected_mouth_pose \
  --candidates-topic /detected_mouth_candidates \
  --status-topic /mouth_detection/status \
  --normal-topic /detected_mouth_normal \
  --mount-calibration-status "${MOUNT_STATUS}" \
  --debug-image-topic /mouth_detection/debug_image \
  --base-frame base_link \
  --max-jump-m 0.35 \
  "$@"
