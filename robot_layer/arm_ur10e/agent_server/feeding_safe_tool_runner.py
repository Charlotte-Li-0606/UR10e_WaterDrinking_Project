#!/usr/bin/env python3
"""CLI dispatcher for the approved UR10e feeding tool surface.

This is an integration boundary for OpenClaw-style callers, not a second
planning stack. Every call is validated, then delegated to the existing
``FeedingSkillLibrary`` implementation.
"""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def _tool_dispatch(library: FeedingSkillLibrary) -> Mapping[str, Callable[..., dict[str, Any]]]:
    return {
        "get_feeding_observation": library.get_feeding_observation,
        "detect_mouth": library.detect_mouth,
        "active_search_mouth": library.active_search_mouth,
        "select_target": library.select_target,
        "move_straw_tip_to_pre_mouth": library.move_straw_tip_to_pre_mouth,
        "check_feeding_progress": library.check_feeding_progress,
        "hold_pre_mouth": library.hold_pre_mouth,
        "retreat_to_ready": library.retreat_to_ready,
        "feed_water": library.feed_water,
    }


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

    try:
        library = FeedingSkillLibrary()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "stage": "library_initialization",
                    "reason": f"could not initialize safe feeding tools: {exc.__class__.__name__}",
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
