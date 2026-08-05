#!/usr/bin/env python3
"""Call the canonical feed_water tool directly against its guarded real backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_RUNNER = (
    PROJECT_ROOT
    / "robot_layer"
    / "arm_ur10e"
    / "agent_server"
    / "feeding_safe_tool_runner.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan the safe pre-mouth target and validate the fixed return target; never execute.",
    )
    mode.add_argument("--execute", action="store_true", help="Request guarded real pre-mouth execution.")
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Explicit runtime confirmation required with --execute.",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=5.0,
        help=(
            "Motionless pre-mouth dwell before the guarded return "
            "(2-5 seconds; default: 5)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.execute and os.environ.get("UR10E_BACKEND") != "real":
        print(
            json.dumps(
                {
                    "success": False,
                    "stage": "backend_selection",
                    "reason": "real execution requires UR10E_BACKEND=real",
                }
            )
        )
        return 2

    environment = dict(os.environ)
    if args.plan_only:
        # The command name plus --plan-only explicitly selects the real
        # backend, while deliberately removing the execution authorization.
        environment["UR10E_BACKEND"] = "real"
        environment.pop("UR10E_ALLOW_REAL_EXECUTION", None)

    tool_args = {
        "target_selection": "center",
        "execute": bool(args.execute),
        "allow_vertical_adjust": False,
        "hold_duration_sec": args.hold_duration,
    }
    command = [
        sys.executable,
        str(TOOL_RUNNER),
        "--tool",
        "feed_water",
        "--args-json",
        json.dumps(tool_args, separators=(",", ":")),
    ]
    if args.execute:
        command.append("--execute")
    if args.confirm_real_motion:
        command.append("--confirm-real-motion")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
