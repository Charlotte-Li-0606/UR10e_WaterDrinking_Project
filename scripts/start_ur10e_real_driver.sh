#!/usr/bin/env bash
# Start only the upstream UR10e driver. It does not start MoveIt or feed water.
source /opt/ros/jazzy/setup.bash
set -euo pipefail
: "${UR10E_ROBOT_IP:?Set UR10E_ROBOT_IP to the UR10e controller address before starting the driver.}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIBRATION_FILE="${UR10E_KINEMATICS_FILE:-${PROJECT_DIR}/config/ur10e_real/ur10e_calibration.yaml}"
if [ ! -f "${CALIBRATION_FILE}" ]; then
  echo "UR10e calibration file is required: ${CALIBRATION_FILE}" >&2
  exit 2
fi

# The UR controller is reached through its TCP robot IP, not ROS discovery.
# Keep this physical-driver graph local so a ROS/Gazebo instance on Wi-Fi
# cannot publish a competing robot_description or TF tree.
unset ROS_STATIC_PEERS
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0

exec ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:="${UR10E_ROBOT_IP}" \
  kinematics_params_file:="${CALIBRATION_FILE}" \
  launch_rviz:=false
