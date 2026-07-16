#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/view_wrist_rgb.py" "$@"
