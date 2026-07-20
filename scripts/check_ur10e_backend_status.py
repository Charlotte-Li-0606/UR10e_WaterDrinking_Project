#!/usr/bin/env python3
"""Read-only status probe for the canonical UR10e ROS SDK.

It creates no action goal, publishes no controller command, and is intended to
run only after a driver and MoveIt stack have already been started manually.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk import UR10eRobotEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "real"), default=os.environ.get("UR10E_BACKEND", "sim"))
    args = parser.parse_args()
    os.environ["UR10E_BACKEND"] = args.backend

    try:
        env = UR10eRobotEnv()
    except Exception as exc:
        print(json.dumps({"success": False, "stage": "sdk_initialization", "reason": str(exc)}))
        return 2
    try:
        print(json.dumps({"success": True, "stage": "read_only_status", "status": env.get_backend_status()}, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
