#!/usr/bin/env bash
# Start only the upstream UR10e driver. It does not start MoveIt or feed water.
set -euo pipefail

source /opt/ros/jazzy/setup.bash
: "${UR10E_ROBOT_IP:?Set UR10E_ROBOT_IP to the UR10e controller address before starting the driver.}"

exec ros2 launch ur_robot_driver ur10e.launch.py \
  robot_ip:="$UR10E_ROBOT_IP" \
  use_mock_hardware:=false \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  activate_joint_controller:=true
