#!/usr/bin/env bash
# Start the read-only ROS 2 live pendant. Any arguments override/add server flags.

set -eo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
ROS_SETUP="/opt/ros/jazzy/setup.bash"

if [ ! -r "${ROS_SETUP}" ]; then
  printf 'ROS 2 setup not found: %s\n' "${ROS_SETUP}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROS_SETUP}"
set -u
cd "${PROJECT_DIR}"
exec python3 -m live_pendant.server "$@"
