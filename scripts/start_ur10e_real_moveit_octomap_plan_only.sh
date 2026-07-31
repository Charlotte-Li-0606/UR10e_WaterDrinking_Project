#!/usr/bin/env bash
# Start MoveIt against the physical UR description with D435i OctoMap input,
# while disabling MoveGroup trajectory execution.  This launcher is for the
# staged no-motion validation of dynamic obstacle plans only.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset ROS_STATIC_PEERS
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0

exec ros2 launch "${PROJECT_DIR}/launch/ur10e_moveit_with_kinematics.launch.py" \
  ur_type:=ur10e \
  launch_rviz:=false \
  launch_servo:=false \
  use_sim_time:=false \
  use_octomap:=true \
  allow_trajectory_execution:=false \
  trajectory_controller:=scaled_joint_trajectory_controller
