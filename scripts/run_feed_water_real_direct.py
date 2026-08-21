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
    parser.add_argument(
        "--target-selection",
        choices=("left", "center", "right"),
        default="center",
        help=(
            "Select the initially left, center, or right mouth in camera-image "
            "order and retain that person's 3D identity."
        ),
    )
    parser.add_argument(
        "--track-mouth-during-execution",
        action="store_true",
        help=(
            "Use the preserved legacy segmented MoveIt tracking path. "
            "New continuous tracking requests should use "
            "--continuous-mouth-tracking."
        ),
    )
    parser.add_argument(
        "--continuous-mouth-tracking",
        action="store_true",
        help="Use the opt-in continuous MoveIt Servo approach and hold mode.",
    )
    parser.add_argument(
        "--use-octomap",
        action="store_true",
        help="Enable the experimental dynamic OctoMap layer in continuous mode.",
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
        "target_selection": args.target_selection,
        "execute": bool(args.execute),
        "allow_vertical_adjust": False,
        "hold_duration_sec": args.hold_duration,
        "track_mouth_during_execution": bool(
            args.track_mouth_during_execution
        ),
        "continuous_mouth_tracking": bool(args.continuous_mouth_tracking),
        "use_octomap": bool(args.use_octomap),
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
