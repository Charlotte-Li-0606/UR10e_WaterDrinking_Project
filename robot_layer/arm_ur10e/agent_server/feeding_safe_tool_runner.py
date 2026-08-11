#!/usr/bin/env python3
"""CLI dispatcher for the approved UR10e feeding tool surface.

This is an integration boundary for OpenClaw-style callers, not a second
planning stack. Every call is validated, then delegated to the existing
``FeedingSkillLibrary`` implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_tools.feeding_tools import (  # noqa: E402
    SAFE_FEEDING_TOOL_NAMES,
    FeedingSkillLibrary,
    FeedingToolValidationError,
    safe_feeding_tool_dispatch,
    validate_safe_feeding_tool_call,
    validate_safe_feeding_tool_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tool", choices=sorted(SAFE_FEEDING_TOOL_NAMES))
    source.add_argument("--plan-json", help="JSON object containing an ordered safe-tool steps array.")
    parser.add_argument("--args-json", default="{}", help="JSON object containing only approved arguments.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicit operator permission for a validated motion-capable tool. Default is plan-only.",
    )
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required in addition to --execute and environment gates for the real backend.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and normalize the approved call or plan without initializing ROS, Gazebo, MoveIt, or the robot SDK.",
    )
    return parser.parse_args()


def _tool_dispatch(library: FeedingSkillLibrary) -> Mapping[str, Callable[..., dict[str, Any]]]:
    return safe_feeding_tool_dispatch(library)


def _call_tool(function: Callable[..., dict[str, Any]], call: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return function(**call["args"])
    except Exception as exc:
        return {
            "success": False,
            "tool": call["tool"],
            "reason": f"safe tool raised {exc.__class__.__name__}",
        }


def main() -> int:
    cli = _parse_args()
    try:
        if cli.plan_json is None:
            raw_args = json.loads(cli.args_json)
            calls = [validate_safe_feeding_tool_call(cli.tool, raw_args, cli_execute=bool(cli.execute))]
        else:
            raw_plan = json.loads(cli.plan_json)
            calls = validate_safe_feeding_tool_plan(raw_plan, cli_execute=bool(cli.execute))["steps"]
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "stage": "validation", "reason": f"tool JSON is invalid: {exc.msg}"}))
        return 2
    except FeedingToolValidationError as exc:
        print(json.dumps({"success": False, "stage": "validation", "reason": str(exc)}))
        return 2

    backend = os.environ.get("UR10E_BACKEND", "sim").strip().lower() or "sim"
    if backend not in {"sim", "real"}:
        print(json.dumps({"success": False, "stage": "backend_selection", "reason": "UR10E_BACKEND must be sim or real"}))
        return 2

    if backend == "real":
        if cli.plan_json is not None or len(calls) != 1 or calls[0]["tool"] != "feed_water":
            print(
                json.dumps(
                    {
                        "success": False,
                        "stage": "real_tool_policy",
                        "reason": (
                            "the real backend exposes only one high-level feed_water call; "
                            "plans and direct perception, pose, joint, gripper, retreat, or controller tools are refused"
                        ),
                    }
                )
            )
            return 2
        if calls[0]["args"].get("allow_vertical_adjust"):
            print(
                json.dumps(
                    {
                        "success": False,
                        "stage": "real_tool_policy",
                        "reason": "real feed_water does not permit a vertical adjustment, mouth contact, tilt, pour, or automatic retreat",
                    }
                )
            )
            return 2

    if cli.validate_only:
        if cli.execute:
            print(
                json.dumps(
                    {
                        "success": False,
                        "stage": "validation",
                        "reason": "--validate-only cannot be combined with --execute",
                    }
                )
            )
            return 2
        if cli.plan_json is None:
            item = calls[0]
            print(
                json.dumps(
                    {
                        "event": "safe_feeding_tool_validation_result",
                        "success": True,
                        "mode": "validate_only",
                        "call": item,
                        "final_state": "plan_validated",
                        "reason": None,
                        "note": "Validated the approved reusable call; no ROS, simulator, MoveIt, perception, or robot SDK was initialized.",
                    },
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "event": "safe_feeding_tool_plan_result",
                    "success": True,
                    "mode": "validate_only",
                    "final_state": "plan_validated",
                    "planned_steps": calls,
                    "results": [],
                    "failed_step": None,
                    "reason": None,
                    "note": "Validated the approved reusable plan; no ROS, simulator, MoveIt, perception, or robot SDK was initialized.",
                },
                sort_keys=True,
            )
        )
        return 0

    if backend == "real":
        from robot_layer.arm_ur10e.agent_server.real_feed_water_backend import run_real_feed_water

        call = calls[0]
        result = run_real_feed_water(
            execute=bool(call["args"]["execute"]),
            confirm_real_motion=bool(cli.confirm_real_motion),
            target_selection=str(call["args"]["target_selection"]),
            hold_duration_sec=float(call["args"]["hold_duration_sec"]),
            track_mouth_during_execution=bool(
                call["args"].get("track_mouth_during_execution", False)
            ),
            continuous_mouth_tracking=bool(
                call["args"].get("continuous_mouth_tracking", False)
            ),
            use_octomap=bool(call["args"].get("use_octomap", False)),
        )
        print(
            json.dumps(
                {"event": "safe_feeding_tool_result", "call": call, "result": result},
                sort_keys=True,
                default=str,
            )
        )
        return 0 if result.get("success") else 2

    try:
        library = FeedingSkillLibrary()
    except Exception as exc:
        detail = str(exc).strip()
        print(
            json.dumps(
                {
                    "success": False,
                    "stage": "library_initialization",
                    "reason": (
                        f"could not initialize safe feeding tools: {exc.__class__.__name__}"
                        f": {detail}" if detail else
                        f"could not initialize safe feeding tools: {exc.__class__.__name__}"
                    ),
                }
            )
        )
        return 2
    try:
        dispatch = _tool_dispatch(library)
        results: list[dict[str, Any]] = []
        for call in calls:
            result = _call_tool(dispatch[call["tool"]], call)
            results.append({"call": call, "result": result})
            if not result.get("success"):
                break
    finally:
        library.close()

    if cli.plan_json is None:
        item = results[0]
        print(json.dumps({"event": "safe_feeding_tool_result", **item}, sort_keys=True, default=str))
        return 0 if item["result"].get("success") else 2
    success = bool(results) and len(results) == len(calls) and all(item["result"].get("success") for item in results)
    print(
        json.dumps(
            {
                "event": "safe_feeding_tool_plan_result",
                "success": success,
                "planned_steps": calls,
                "results": results,
                "failed_step": None if success else results[-1]["call"]["tool"],
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
