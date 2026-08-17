#!/usr/bin/env bash
# Codex-native entrypoint for the guarded real UR10e feed_water workflow.

set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

if [ ! -r "${ROS_SETUP}" ]; then
  printf 'ROS 2 setup not found: %s\n' "${ROS_SETUP}" >&2
  exit 1
fi

# ROS setup scripts may inspect unset variables, so enable nounset afterward.
# shellcheck disable=SC1091
source "${ROS_SETUP}"
if [ -r /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ] || ! "${PYTHON_BIN}" -c 'import rclpy' >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

# Serialize every plan/execution request. A busy lock means a current canonical
# workflow is still alive, so a new caller must refuse instead of interrupting
# an active real-motion session.
EXECUTION_LOCK="/tmp/ur_drinking_project_feed_water_execution.lock"
exec 9>"${EXECUTION_LOCK}"
if ! flock -n 9; then
  printf '%s\n' \
    '{"success":false,"stage":"feed_water_process_lock","reason":"another canonical feed-water workflow is still active; new execution refused","execution_attempted":false,"execution_sent":false}' >&2
  exit 2
fi

EXECUTION_REQUESTED=false
for argument in "$@"; do
  if [ "${argument}" = "--execute" ]; then
    EXECUTION_REQUESTED=true
    break
  fi
done

# A previous launcher can be terminated externally after spawning a child.
# Once the lock is ours, retire only exact orphaned workflow runners before a
# new real execution. Never match the long-running robot/MoveIt/Servo stack.
if [ "${EXECUTION_REQUESTED}" = true ]; then
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/feed_water_process_guard.py" >&2
fi

# This entrypoint is intentionally real-only. Simulation remains available
# through the preserved OpenClaw compatibility scripts, but Codex never infers
# or inherits the backend from a long-running gateway process.
unset ROS_STATIC_PEERS
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0
export UR10E_BACKEND=real
cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_feed_water_real_direct.py" "$@"
