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
        self.assertIn("feed_water", SAFE_FEEDING_TOOL_NAMES)
        self.assertIn("get_feeding_observation", SAFE_FEEDING_TOOL_NAMES)
        with self.assertRaises(FeedingToolValidationError):
            validate_safe_feeding_tool_call("move_joints", {}, cli_execute=True)

    def test_plan_only_forces_motion_flags_false(self) -> None:
        move = validate_safe_feeding_tool_call(
            "move_straw_tip_to_pre_mouth", {"execute": True}, cli_execute=False
        )
        search = validate_safe_feeding_tool_call(
            "active_search_mouth", {"execute": True}, cli_execute=False
        )
        self.assertFalse(move["args"]["execute"])
        self.assertFalse(search["args"]["execute"])

    def test_rejects_unsafe_or_out_of_range_arguments(self) -> None:
        rejected = (
            ("active_search_mouth", {"max_search_time_sec": 30.1}),
            ("select_target", {"target_selection": "front"}),
            ("move_straw_tip_to_pre_mouth", {"pose": [0, 0, 0]}),
            ("feed_water", {"allow_direct_mouth_contact": True}),
        )
        for tool, args in rejected:
            with self.subTest(tool=tool, args=args), self.assertRaises(FeedingToolValidationError):
                validate_safe_feeding_tool_call(tool, args, cli_execute=True)

    def test_sequence_normalizes_each_safe_step(self) -> None:
        plan = validate_safe_feeding_tool_plan(
            {
                "steps": [
                    {"tool": "get_feeding_observation", "args": {}},
                    {"tool": "detect_mouth", "args": {}},
                    {"tool": "move_straw_tip_to_pre_mouth", "args": {"execute": True}},
                ]
            },
            cli_execute=False,
        )
        self.assertEqual(3, len(plan["steps"]))
        self.assertFalse(plan["steps"][-1]["args"]["execute"])


if __name__ == "__main__":
    unittest.main()
