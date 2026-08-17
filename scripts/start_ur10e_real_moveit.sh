#!/usr/bin/env bash
# Start only MoveIt against an already-running real UR10e driver.
source /opt/ros/jazzy/setup.bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_SERVO="${UR10E_LAUNCH_SERVO:-false}"
USE_OCTOMAP="${UR10E_USE_OCTOMAP:-true}"

case "${LAUNCH_SERVO}" in
  true|false) ;;
  *)
    echo "UR10E_LAUNCH_SERVO must be true or false" >&2
    exit 2
    ;;
esac
case "${USE_OCTOMAP}" in
  true|false) ;;
  *)
    echo "UR10E_USE_OCTOMAP must be true or false" >&2
    exit 2
    ;;
esac

# The driver and MoveIt run locally on the ThinkPad.  Do not discover a
# second ROS graph (for example a Wi-Fi Gazebo instance) that could publish a
# competing robot_description or TF tree.
unset ROS_STATIC_PEERS
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0

# This project wrapper supplies the MoveIt kinematics parameters required for
# pose planning, loads the wrist-camera occupancy monitor used by the guarded
# real feed_water workflow, and advertises only the physical-robot controller.
exec ros2 launch "${PROJECT_DIR}/launch/ur10e_moveit_with_kinematics.launch.py" \
  ur_type:=ur10e \
  launch_rviz:=false \
  launch_servo:="${LAUNCH_SERVO}" \
  use_sim_time:=false \
  use_octomap:="${USE_OCTOMAP}" \
  trajectory_controller:=scaled_joint_trajectory_controller
