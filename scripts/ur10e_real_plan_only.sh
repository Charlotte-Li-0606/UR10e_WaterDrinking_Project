#!/usr/bin/env bash
# Plan the reviewed straw_tip -> pre_mouth motion through MoveIt without motion.
set -euo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi

# This script is intentionally a hard plan-only safety boundary even if its
# caller inherited an execution-permission environment variable.
export UR10E_BACKEND=real
export UR10E_ALLOW_REAL_EXECUTION=0

exec python3 "$PROJECT_DIR/robot_layer/arm_ur10e/agent_server/feeding_safe_tool_runner.py" \
  --tool move_tool_to_target \
  --args-json '{"tool":"straw_tip","target":"pre_mouth","execute":false}'
