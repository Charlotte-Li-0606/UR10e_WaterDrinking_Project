"""Focused safety checks for the reusable LLM feeding-tool contract."""

from __future__ import annotations

import json
import unittest

from robot_layer.arm_ur10e.agent_server.llm_feeding_agent import (
    PlanValidationError,
    SAFE_MODE,
    validate_plan,
)


def _plan(args: dict[str, object]) -> str:
    return json.dumps(
        {
            "task": "feed_water",
            "mode": SAFE_MODE,
            "steps": [{"tool": "feed_water", "args": args}],
        }
    )


class FeedWaterPlanValidationTest(unittest.TestCase):
    def test_plan_only_forces_execute_false(self) -> None:
        plan = validate_plan(
            _plan({"target_selection": "center", "execute": True, "max_search_time_sec": 30.0}),
            cli_execute=False,
        )
        self.assertFalse(plan["steps"][0]["args"]["execute"])

    def test_execute_requires_cli_permission(self) -> None:
        plan = validate_plan(
            _plan({"target_selection": "right", "execute": True, "max_search_time_sec": 30.0}),
            cli_execute=True,
        )
        self.assertTrue(plan["steps"][0]["args"]["execute"])
        self.assertEqual("right", plan["steps"][0]["args"]["target_selection"])

    def test_rejects_out_of_policy_arguments(self) -> None:
        invalid_args = (
            {"target_selection": "front"},
            {"max_search_time_sec": 30.1},
            {"allow_direct_mouth_contact": True},
        )
        for args in invalid_args:
            with self.subTest(args=args), self.assertRaises(PlanValidationError):
                validate_plan(_plan(args), cli_execute=True)

    def test_rejects_low_level_tool(self) -> None:
        plan = json.dumps(
            {
                "task": "feed_water",
                "mode": SAFE_MODE,
                "steps": [{"tool": "move_joints", "args": {}}],
            }
        )
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, cli_execute=True)

    def test_normalizes_reusable_plan_only_sequence(self) -> None:
        plan = json.dumps(
            {
                "task": "feed_water",
                "mode": SAFE_MODE,
                "steps": [
                    {"tool": "get_observation", "args": {}},
                    {"tool": "detect_target", "args": {"target_type": "mouth", "detector": "mediapipe"}},
                    {"tool": "active_search", "args": {"max_time_sec": 30.0, "strategy": "safe_scan", "execute": True}},
                    {"tool": "select_target", "args": {"target_type": "mouth", "strategy": "center"}},
                    {"tool": "move_tool_to_target", "args": {"tool": "straw_tip", "target": "pre_mouth", "execute": True}},
                    {"tool": "check_progress", "args": {"task": "feed_water", "critic": "rule_based"}},
                    {"tool": "hold", "args": {"duration_sec": 3.0}},
                ],
            }
        )
        normalized = validate_plan(plan, cli_execute=False)
        self.assertEqual("get_observation", normalized["steps"][0]["tool"])
        self.assertFalse(normalized["steps"][2]["args"]["execute"])
        self.assertFalse(normalized["steps"][4]["args"]["execute"])

    def test_rejects_legacy_task_specific_tool(self) -> None:
        plan = json.dumps(
            {
                "task": "feed_water",
                "mode": SAFE_MODE,
                "steps": [{"tool": "move_straw_tip_to_pre_mouth", "args": {}}],
            }
        )
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, cli_execute=True)


if __name__ == "__main__":
    unittest.main()
