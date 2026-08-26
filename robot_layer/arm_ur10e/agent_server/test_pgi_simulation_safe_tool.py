#!/usr/bin/env python3
"""Unit tests for the Stage-6 PGI simulation-only tool boundary."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from robot_layer.arm_ur10e.agent_tools.pgi_simulation_tools import (
    PgiSimulationToolValidationError,
    build_pgi_simulation_command,
    read_pgi_simulation_report,
    summarize_pgi_simulation_report,
    validate_safe_pgi_simulation_tool_call,
)


class PgiSimulationSafeToolTests(unittest.TestCase):
    def test_plan_and_execute_tools_are_approved_without_arguments(self) -> None:
        plan = validate_safe_pgi_simulation_tool_call("plan_cup_grasp_cycle")
        execute = validate_safe_pgi_simulation_tool_call("execute_cup_grasp_cycle", {})
        self.assertFalse(plan["motion_capable"])
        self.assertTrue(execute["motion_capable"])

    def test_low_level_and_arbitrary_motion_tools_are_rejected(self) -> None:
        for name in (
            "move_joints",
            "move_to_pose",
            "command_gripper",
            "switch_controller",
            "feed_water",
        ):
            with self.subTest(name=name), self.assertRaises(PgiSimulationToolValidationError):
                validate_safe_pgi_simulation_tool_call(name)

    def test_every_runtime_parameter_is_rejected(self) -> None:
        for name in ("joints", "pose", "cup_x", "force", "controller", "wrist_3_joint"):
            with self.subTest(name=name), self.assertRaises(PgiSimulationToolValidationError):
                validate_safe_pgi_simulation_tool_call(
                    "execute_cup_grasp_cycle", {name: 1}
                )

    def test_plan_command_cannot_enable_simulation_execution(self) -> None:
        call = validate_safe_pgi_simulation_tool_call("plan_cup_grasp_cycle")
        command = build_pgi_simulation_command(call, Path("/tmp/report.json"))
        self.assertNotIn("--execute-sim", command)
        self.assertNotIn("--confirm-simulation", command)
        self.assertIn("pgi_physical_grasp_demo.py", " ".join(command))
        self.assertNotIn("ur_robot_driver", " ".join(command))

    def test_execute_command_contains_both_simulation_gates(self) -> None:
        call = validate_safe_pgi_simulation_tool_call("execute_cup_grasp_cycle")
        command = build_pgi_simulation_command(call, Path("/tmp/report.json"))
        self.assertIn("--execute-sim", command)
        self.assertIn("--confirm-simulation", command)
        self.assertIn("--suppress-console-report", command)

    def test_report_reader_requires_a_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"stage": 5}), encoding="utf-8")
            self.assertEqual(read_pgi_simulation_report(path)["stage"], 5)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                read_pgi_simulation_report(path)

    def test_high_level_summary_keeps_safety_and_measured_results(self) -> None:
        summary = summarize_pgi_simulation_report(
            {
                "success": True,
                "stage": 5,
                "mode": "execute_sim",
                "real_robot_command_sent": False,
                "guards": {
                    "ros_domain_id": 106,
                    "move_group_plan_only": True,
                    "controllers": {
                        "joint_trajectory_controller": {"state": "inactive"}
                    },
                },
                "plans": {"lift": {"backend": "cartesian"}},
                "execution": {
                    "success": True,
                    "measured_lift_delta_m": [0.0, 0.0, 0.119],
                    "hold_drift_m": 0.000003,
                    "cup_after_lift": {"tilt_deg": 5.4},
                    "cup_attached_at_end": False,
                    "arm_controller_active_at_end": False,
                },
            }
        )
        self.assertTrue(summary["success"])
        self.assertFalse(summary["real_robot_command_sent"])
        self.assertAlmostEqual(summary["execution"]["measured_lift_z_m"], 0.119)
        self.assertEqual(summary["execution"]["maximum_observed_cup_tilt_deg"], 5.4)

    def test_plan_summary_reports_that_no_trajectory_was_sent(self) -> None:
        summary = summarize_pgi_simulation_report(
            {
                "success": True,
                "stage": 5,
                "mode": "plan_only",
                "real_robot_command_sent": False,
                "execution": {
                    "attempted": False,
                    "trajectory_sent": False,
                    "controller_switched": False,
                },
            }
        )
        self.assertEqual(
            summary["execution"],
            {
                "attempted": False,
                "trajectory_sent": False,
                "controller_switched": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
