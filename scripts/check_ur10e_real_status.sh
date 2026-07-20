#!/usr/bin/env bash
# Read-only real-UR10e status probe. It never sends a trajectory or starts a driver.
set -euo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi

export UR10E_BACKEND=real
export UR10E_ALLOW_REAL_EXECUTION=0
exec python3 "$PROJECT_DIR/scripts/check_ur10e_backend_status.py" --backend real
