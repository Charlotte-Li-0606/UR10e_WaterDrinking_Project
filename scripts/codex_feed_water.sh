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

# This entrypoint is intentionally real-only. Simulation remains available
# through the preserved OpenClaw compatibility scripts, but Codex never infers
# or inherits the backend from a long-running gateway process.
export UR10E_BACKEND=real
cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_feed_water_real_direct.py" "$@"
