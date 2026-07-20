#!/usr/bin/env bash
# Start and verify the local UR10e Gazebo/MoveIt simulation for approved runs.
# This script never addresses a physical robot.
set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"

source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

simulator_ready() {
  local actions
  # Match the actual Gazebo executable, not another shell command whose text
  # happens to mention "gz sim" (for example a diagnostics command).
  if ! pgrep -f '^gz sim( |$)' >/dev/null 2>&1; then
    return 1
  fi
  actions="$(ros2 action list 2>/dev/null || true)"
  grep -Fxq '/move_action' <<<"${actions}" && \
    grep -Fxq '/joint_trajectory_controller/follow_joint_trajectory' <<<"${actions}"
}

if ! simulator_ready; then
  "$PROJECT_DIR/scripts/start_ur10e_feeding_sim.sh"
fi

for _attempt in $(seq 1 45); do
  if simulator_ready; then
    exit 0
  fi
  sleep 1
done

printf '%s\n' '{"success":false,"stage":"simulator_startup","reason":"Gazebo/MoveIt did not expose /move_action and the joint trajectory controller within 45 seconds; no reusable tool was executed."}'
exit 2
