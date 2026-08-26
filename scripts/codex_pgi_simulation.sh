#!/usr/bin/env bash
# Canonical simulation-only entrypoint for the PGI cup-grasp tool surface.

set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

if [ ! -r "${ROS_SETUP}" ]; then
  printf 'ROS 2 setup not found: %s\n' "${ROS_SETUP}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROS_SETUP}"
if [ -r /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-106}"
VALIDATE_ONLY=false
for argument in "$@"; do
  if [ "${argument}" = "--validate-only" ]; then
    VALIDATE_ONLY=true
    break
  fi
done
if [ "${ROS_DOMAIN_ID}" = "0" ] && [ "${VALIDATE_ONLY}" = false ]; then
  printf '%s\n' \
    '{"success":false,"stage":6,"reason":"ROS domain 0 is refused; Stage 6 is simulation-only","real_robot_command_sent":false}' >&2
  exit 2
fi
export GZ_PARTITION="${GZ_PARTITION:-pgi_140_80_domain_${ROS_DOMAIN_ID}}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0

# Real-execution gates are deliberately removed from this child environment.
unset UR10E_BACKEND UR10E_ALLOW_REAL_EXECUTION ROS_STATIC_PEERS

cd "${PROJECT_DIR}"
exec /usr/bin/python3 \
  "${PROJECT_DIR}/robot_layer/arm_ur10e/agent_server/pgi_simulation_safe_tool_runner.py" \
  "$@"
