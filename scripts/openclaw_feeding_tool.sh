#!/usr/bin/env bash
# Local OpenClaw bridge for one validated reusable tool call or safe tool plan.
set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi

PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ] || ! "${PYTHON_BIN}" -c 'import rclpy' >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

# Any explicit execution request is simulator-only in this project. Ensure the
# local Gazebo/MoveIt stack is ready even when an older cached skill invokes
# this bridge directly instead of the higher-level compatibility entrypoint.
for argument in "$@"; do
  if [ "$argument" = "--execute" ]; then
    "$PROJECT_DIR/scripts/ensure_ur10e_feeding_sim.sh"
    exec "$PYTHON_BIN" "$PROJECT_DIR/robot_layer/arm_ur10e/agent_server/feeding_safe_tool_runner.py" "$@"
  fi
done

exec "${PYTHON_BIN}" \
  "${PROJECT_DIR}/robot_layer/arm_ur10e/agent_server/feeding_safe_tool_runner.py" "$@"
