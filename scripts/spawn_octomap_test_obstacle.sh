#!/usr/bin/env bash
# Spawn/remove a Gazebo-only obstacle for validating MoveIt's dynamic OctoMap.
#
# This script talks only to Gazebo transport.  It intentionally does not
# publish a MoveIt PlanningScene message or create a CollisionObject.
set -eo pipefail

# Make the documented absolute-path invocation work from a fresh terminal.
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

WORLD_NAME="${GAZEBO_WORLD_NAME:-empty}"
MODEL_NAME="octomap_depth_test_obstacle"
ACTION=""
# Gazebo-world pose.  The UR base is at world (0.0, 0.55, 0.60), so this is
# just in front of the pre-mouth approach, without overlapping its target.
POSE="0.350 0.820 1.650 0 0 0"
SIZE="0.120 0.120 0.250"

usage() {
  cat <<'EOF'
Usage:
  scripts/spawn_octomap_test_obstacle.sh --spawn [--pose "x y z roll pitch yaw"] [--size "x y z"]
  scripts/spawn_octomap_test_obstacle.sh --remove

The pose and size are in Gazebo world coordinates/metres.  The model is
named octomap_depth_test_obstacle and is created only in Gazebo; it is never
added to MoveIt's PlanningScene as a fixed collision object.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spawn)
      ACTION="spawn"
      shift
      ;;
    --remove)
      ACTION="remove"
      shift
      ;;
    --pose)
      [[ $# -ge 2 ]] || { echo "--pose requires six quoted values" >&2; exit 2; }
      POSE="$2"
      shift 2
      ;;
    --size)
      [[ $# -ge 2 ]] || { echo "--size requires three quoted values" >&2; exit 2; }
      SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${ACTION}" ]] || { usage >&2; exit 2; }
command -v gz >/dev/null || {
  echo "'gz' is unavailable. Source /opt/ros/jazzy/setup.bash first." >&2
  exit 2
}

remove_model() {
  # A missing model is acceptable when replacing an old test obstacle.
  gz service \
    -s "/world/${WORLD_NAME}/remove/blocking" \
    --reqtype gz.msgs.Entity \
    --reptype gz.msgs.Boolean \
    --timeout 3000 \
    --req "name: \"${MODEL_NAME}\" type: MODEL" >/dev/null 2>&1 || true
}

if [[ "${ACTION}" == "remove" ]]; then
  remove_model
  python3 - "${WORLD_NAME}" "${MODEL_NAME}" <<'PY'
import json
import sys
print(json.dumps({
    "event": "octomap_depth_test_obstacle_removed",
    "gazebo_only": True,
    "model": sys.argv[2],
    "world": sys.argv[1],
}, sort_keys=True))
PY
  exit 0
fi

read -r px py pz roll pitch yaw extra <<<"${POSE}"
[[ -z "${extra:-}" && -n "${yaw:-}" ]] || {
  echo "--pose must contain exactly: x y z roll pitch yaw" >&2
  exit 2
}
read -r sx sy sz size_extra <<<"${SIZE}"
[[ -z "${size_extra:-}" && -n "${sz:-}" ]] || {
  echo "--size must contain exactly: x y z" >&2
  exit 2
}

# Validate numeric input before embedding it in SDF.  It is intentionally
# static, visible, and has Gazebo collision geometry, but no MoveIt object.
read -r px py pz roll pitch yaw sx sy sz < <(
  python3 - "${px}" "${py}" "${pz}" "${roll}" "${pitch}" "${yaw}" "${sx}" "${sy}" "${sz}" <<'PY'
import math
import sys

values = [float(value) for value in sys.argv[1:]]
if not all(math.isfinite(value) for value in values):
    raise SystemExit("pose and size values must be finite")
if any(value <= 0.0 for value in values[6:]):
    raise SystemExit("all obstacle size values must be positive")
print(*[f"{value:.6f}" for value in values])
PY
)

remove_model

# Keep a Gazebo collision geometry for scene inspection, but disable physical
# contacts.  This fixture must test only the depth-cloud -> OctoMap route;
# a Gazebo contact changing the robot joints would confound the result.
SDF="<sdf version=\"1.9\"><model name=\"${MODEL_NAME}\"><static>true</static><pose>${px} ${py} ${pz} ${roll} ${pitch} ${yaw}</pose><link name=\"link\"><collision name=\"collision\"><geometry><box><size>${sx} ${sy} ${sz}</size></box></geometry><surface><contact><collide_bitmask>0x00</collide_bitmask></contact></surface></collision><visual name=\"visual\"><geometry><box><size>${sx} ${sy} ${sz}</size></box></geometry><material><ambient>0.55 0.05 0.85 1</ambient><diffuse>0.70 0.10 1.00 1</diffuse></material></visual></link></model></sdf>"
REQUEST="$(SDF="${SDF}" python3 -c 'import json, os; print("sdf: " + json.dumps(os.environ["SDF"]))')"

RESULT="$(gz service \
  -s "/world/${WORLD_NAME}/create/blocking" \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 30000 \
  --req "${REQUEST}" 2>&1)" || {
  echo "Gazebo could not create ${MODEL_NAME}: ${RESULT}" >&2
  exit 1
}

if [[ "${RESULT}" != *"data: true"* ]]; then
  echo "Gazebo rejected ${MODEL_NAME}: ${RESULT}" >&2
  exit 1
fi

python3 - "${WORLD_NAME}" "${MODEL_NAME}" "${px}" "${py}" "${pz}" "${roll}" "${pitch}" "${yaw}" "${sx}" "${sy}" "${sz}" <<'PY'
import json
import sys

values = [float(value) for value in sys.argv[3:]]
print(json.dumps({
    "event": "octomap_depth_test_obstacle_spawned",
    "gazebo_only": True,
    "model": sys.argv[2],
    "pose_world": values[:6],
    "size_m": values[6:],
    "world": sys.argv[1],
}, sort_keys=True))
PY
