"""Unit checks for the explicit ABot/OpenClaw feeding tool surface."""

from __future__ import annotations

import unittest

from robot_layer.arm_ur10e.agent_tools.feeding_tools import (
    SAFE_FEEDING_TOOL_NAMES,
    FeedingToolValidationError,
    validate_safe_feeding_tool_call,
    validate_safe_feeding_tool_plan,
)


class SafeFeedingToolValidationTest(unittest.TestCase):
    def test_only_declared_tools_are_allowed(self) -> None:
        self.assertEqual(
            {
                "get_observation",
                "detect_target",
                "active_search",
                "select_target",
                "move_tool_to_target",
                "check_progress",
                "hold",
                "retreat",
                "feed_water",
            },
            set(SAFE_FEEDING_TOOL_NAMES),
        )
        with self.assertRaises(FeedingToolValidationError):
            validate_safe_feeding_tool_call("move_joints", {}, cli_execute=True)
        with self.assertRaises(FeedingToolValidationError):
            validate_safe_feeding_tool_call("detect_mouth", {}, cli_execute=True)

    def test_plan_only_forces_motion_flags_false(self) -> None:
        move = validate_safe_feeding_tool_call(
            "move_tool_to_target", {"tool": "straw_tip", "target": "pre_mouth", "execute": True}, cli_execute=False
        )
        search = validate_safe_feeding_tool_call(
            "active_search", {"execute": True}, cli_execute=False
        )
        self.assertFalse(move["args"]["execute"])
        self.assertFalse(search["args"]["execute"])

    def test_rejects_unsafe_or_out_of_range_arguments(self) -> None:
        rejected = (
            ("active_search", {"max_time_sec": 30.1}),
            ("detect_target", {"target_type": "person", "detector": "yolo"}),
            ("select_target", {"strategy": "front"}),
            ("move_tool_to_target", {"tool": "straw_tip", "target": "mouth"}),
            ("feed_water", {"allow_direct_mouth_contact": True}),
            ("feed_water", {"hold_duration_sec": 1.9}),
            ("feed_water", {"hold_duration_sec": 5.1}),
        )
        for tool, args in rejected:
            with self.subTest(tool=tool, args=args), self.assertRaises(FeedingToolValidationError):
                validate_safe_feeding_tool_call(tool, args, cli_execute=True)

    def test_feed_water_defaults_to_pre_mouth_hold_only(self) -> None:
        call = validate_safe_feeding_tool_call("feed_water", {}, cli_execute=False)
        self.assertFalse(call["args"]["allow_vertical_adjust"])
        self.assertEqual(3.0, call["args"]["hold_duration_sec"])
        self.assertFalse(call["args"]["execute"])

    def test_sequence_normalizes_each_safe_step(self) -> None:
        plan = validate_safe_feeding_tool_plan(
            {
                "steps": [
                    {"tool": "get_observation", "args": {}},
                    {"tool": "detect_target", "args": {"target_type": "mouth", "detector": "mediapipe"}},
                    {"tool": "active_search", "args": {"strategy": "safe_scan", "execute": True}},
                    {"tool": "select_target", "args": {"target_type": "mouth", "strategy": "center"}},
                    {"tool": "move_tool_to_target", "args": {"tool": "straw_tip", "target": "pre_mouth", "execute": True}},
                    {"tool": "check_progress", "args": {"task": "feed_water"}},
                ]
            },
            cli_execute=False,
        )
        self.assertEqual(6, len(plan["steps"]))
        self.assertEqual("safe_scan", plan["steps"][2]["args"]["strategy"])
        self.assertFalse(plan["steps"][4]["args"]["execute"])


if __name__ == "__main__":
    unittest.main()
