#!/usr/bin/env bash
# Local OpenClaw bridge for one validated reusable tool call or safe tool plan.
set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi

PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/robot_layer/arm_ur10e/agent_server/feeding_safe_tool_runner.py" "$@"
