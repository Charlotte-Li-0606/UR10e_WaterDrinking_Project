#!/usr/bin/env python3
"""CLI smoke test for the conservative reusable feeding tools.

All motion-capable commands are plan-only unless ``--execute`` is supplied.
The CLI never enables direct mouth contact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_tools.feeding_tools import FeedingSkillLibrary  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--observe", action="store_true", help="Read compact robot/camera/mouth observation.")
    actions.add_argument(
        "--active-target-state",
        action="store_true",
        help="Read the active left/center/right target queue state.",
    )
    actions.add_argument(
        "--select-target",
        choices=("left", "center", "right"),
        help="Select one logical mouth target without sending motion.",
    )
    actions.add_argument(
        "--search-mouth",
        action="store_true",
        help="Find a stable selected mouth with the fixed safe scan; --execute permits scan motion.",
    )
    actions.add_argument("--plan-pre-mouth", action="store_true", help="Validate and plan the detected pre-mouth target.")
    actions.add_argument("--move-pre-mouth", action="store_true", help="Preflight the detected pre-mouth target; --execute sends it.")
    actions.add_argument("--retreat", action="store_true", help="Preflight the configured ready retreat; --execute sends it.")
    actions.add_argument(
        "--adjust-cup-vertical",
        type=float,
        metavar="DELTA_Z",
        help="Move the rigid straw-tip control point only along base_link Z; --execute sends it.",
    )
    parser.add_argument("--execute", action="store_true", help="Permit the selected predefined MoveIt action.")
    planning_scene = parser.add_mutually_exclusive_group()
    planning_scene.add_argument(
        "--use-planning-scene",
        dest="use_planning_scene",
        action="store_true",
        default=None,
        help="Apply and verify deterministic mouth-derived PlanningScene obstacles before motion preflight.",
    )
    planning_scene.add_argument(
        "--no-planning-scene",
        dest="use_planning_scene",
        action="store_false",
        help="Disable the dynamic PlanningScene obstacle manager for this smoke-test run.",
    )
    parser.add_argument("--wait-timeout-sec", type=float, default=8.0, help="Maximum wait for a stable mouth pose.")
    parser.add_argument("--search-timeout", type=float, default=30.0, help="Maximum safe mouth-search duration (0.1–30 s).")
    args = parser.parse_args()
    if args.execute and (
        args.observe
        or args.active_target_state
        or args.select_target is not None
        or args.plan_pre_mouth
        or not any((args.search_mouth, args.move_pre_mouth, args.retreat, args.adjust_cup_vertical is not None))
    ):
        parser.error("--execute is valid only with --search-mouth, --move-pre-mouth, --retreat, or --adjust-cup-vertical")
    if not any(
        (
            args.observe,
            args.active_target_state,
            args.select_target is not None,
            args.search_mouth,
            args.plan_pre_mouth,
            args.move_pre_mouth,
            args.retreat,
            args.adjust_cup_vertical is not None,
        )
    ):
        args.observe = True
    return args


def main() -> int:
    args = _parse_args()
    tools = FeedingSkillLibrary(use_planning_scene=args.use_planning_scene)
    try:
        if args.observe:
            mouth = tools.wait_for_stable_mouth_pose(timeout_sec=args.wait_timeout_sec)
            observation = tools.get_robot_observation()
            result = {
                "success": bool(mouth.get("success")) and bool(observation.get("success")),
                "tool": "observe",
                "reason": None if mouth.get("success") and observation.get("success") else mouth.get("reason") or observation.get("reason"),
                "mouth": mouth,
                "observation": observation,
            }
        elif args.active_target_state:
            result = tools.get_active_target_state()
        elif args.select_target is not None:
            result = tools.select_active_target(args.select_target)
        elif args.search_mouth:
            result = tools.search_for_mouth(
                max_time_sec=args.search_timeout,
                selection="center",
                execute=args.execute,
            )
        elif args.plan_pre_mouth:
            mouth = tools.wait_for_stable_mouth_pose(timeout_sec=args.wait_timeout_sec)
            result = (
                tools.move_straw_tip_to_pre_mouth(mouth["mouth_pose"], execute=False)
                if mouth.get("success")
                else {"success": False, "tool": "plan_pre_mouth", "reason": mouth.get("reason"), "mouth": mouth}
            )
        elif args.move_pre_mouth:
            mouth = tools.wait_for_stable_mouth_pose(timeout_sec=args.wait_timeout_sec)
            result = (
                tools.move_straw_tip_to_pre_mouth(mouth["mouth_pose"], execute=args.execute)
                if mouth.get("success")
                else {"success": False, "tool": "move_straw_tip_to_pre_mouth", "reason": mouth.get("reason"), "mouth": mouth}
            )
        elif args.adjust_cup_vertical is not None:
            result = tools.adjust_cup_vertical(args.adjust_cup_vertical, execute=args.execute)
        else:
            result = tools.retreat_to_ready(execute=args.execute)
    except Exception as exc:
        result = {"success": False, "tool": "feeding_tools_smoke_test", "reason": str(exc)}
    finally:
        tools.close()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
