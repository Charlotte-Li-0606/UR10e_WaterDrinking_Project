#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

exec "${PYTHON_BIN}" "${PROJECT_DIR}/robot_layer/arm_ur10e/agent_tools/planning_scene_manager.py" --apply "$@"
