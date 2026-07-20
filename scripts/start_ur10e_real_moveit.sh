#!/usr/bin/env bash
# Start only MoveIt against an already-running real UR10e driver.
source /opt/ros/jazzy/setup.bash
set -euo pipefail

exec ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur10e \
  launch_rviz:=false \
  launch_servo:=false \
  use_sim_time:=false
