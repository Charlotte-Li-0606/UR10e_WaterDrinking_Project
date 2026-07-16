#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export UR10E_SDK_CONFIG="${UR10E_SDK_CONFIG:-${PROJECT_DIR}/config/ur10e_sdk_config.yaml}"

exec "${PROJECT_DIR}/venv/bin/python" \
  "${PROJECT_DIR}/robot_layer/arm_ur10e/agent_server/no_llm_feeding_agent_node.py" "$@"
