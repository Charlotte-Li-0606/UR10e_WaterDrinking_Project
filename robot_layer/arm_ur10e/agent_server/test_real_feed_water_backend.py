"""No-motion unit checks for the guarded real feed_water adapter."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_layer.arm_ur10e.agent_server import real_feed_water_backend as backend


class RealFeedWaterBackendTest(unittest.TestCase):
    def test_missing_pipeline_report_preserves_bounded_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            ["pipeline"],
            1,
            stdout="",
            stderr="Traceback: startup exploded",
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run", return_value=completed):
            result = backend.run_real_feed_water(
                execute=False,
                environ={"UR10E_BACKEND": "real"},
            )

        self.assertFalse(result["success"])
        self.assertEqual("pipeline_report", result["stage"])
        self.assertIn("Traceback: startup exploded", result["reason"])
        self.assertEqual(1, result["safety_gates"]["pipeline_exit_code"])

    def test_pipeline_exception_does_not_claim_that_no_motion_was_sent(self) -> None:
        pipeline = {
            "success": False,
            "stage": "pipeline_exception",
            "reason": "RuntimeError: unexpected failure",
            "execution_attempted": None,
            "execution_sent": None,
            "execution_state_unknown": True,
            "final_state": "refused",
        }

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            report_argument = command.index("--report-file") + 1
            Path(command[report_argument]).write_text(
                json.dumps(pipeline), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 2, "", "")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run", side_effect=fake_run):
            result = backend.run_real_feed_water(
                execute=True,
                confirm_real_motion=True,
                environ={
                    "UR10E_BACKEND": "real",
                    "UR10E_ALLOW_REAL_EXECUTION": "1",
                },
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["execution_state_unknown"])
        self.assertIsNone(result["execution_attempted"])
        self.assertIsNone(result["execution_sent"])

    def test_execute_requires_environment_and_runtime_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run") as run:
            result = backend.run_real_feed_water(
                execute=True,
                confirm_real_motion=False,
                environ={"UR10E_BACKEND": "real"},
            )
        self.assertFalse(result["success"])
        self.assertFalse(result["execution_attempted"])
        run.assert_not_called()

    def test_real_backend_refuses_non_center_target_before_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run") as run:
            result = backend.run_real_feed_water(
                execute=False,
                target_selection="left",
                environ={"UR10E_BACKEND": "real"},
            )
        self.assertFalse(result["success"])
        self.assertEqual("target_selection", result["stage"])
        self.assertFalse(result["execution_attempted"])
        run.assert_not_called()

    def test_plan_only_delegates_to_corrected_pipeline_without_execute_flags(self) -> None:
        pipeline = {
            "success": True,
            "mode": "plan",
            "detected_mouth_pose": {"frame_id": "base_link", "position_m": [0.1, 0.2, 0.3]},
            "pre_mouth_pose": {"frame_id": "base_link", "position_m": [0.15, 0.2, 0.3]},
            "planned_tool0_translation_m": [0.01, 0.02, 0.03],
            "planned_tool0_translation_norm_m": 0.037,
            "checks": {
                "mouth_pose": {"stable": True},
                "camera_tf": {"available": True},
                "camera_mount_match": {"matches": True},
            },
            "execution_attempted": False,
            "execution_sent": False,
            "pre_mouth_hold": {
                "completed": False,
                "plan_only": True,
                "motion_command_sent": False,
            },
            "return_to_initial_position": {
                "success": True,
                "stage": "return_target_validated",
                "execution_attempted": False,
                "execution_sent": False,
                "automatic_retreat_sent": False,
            },
        }

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            report_argument = command.index("--report-file") + 1
            Path(command[report_argument]).write_text(json.dumps(pipeline), encoding="utf-8")
            self.assertEqual(
                str(backend.REAL_FEED_WATER_SCRIPT),
                command[1],
            )
            self.assertIn("--plan-only", command)
            target_selection_argument = command.index("--target-selection") + 1
            self.assertEqual("center", command[target_selection_argument])
            self.assertIn("--no-execute", command)
            self.assertNotIn("--confirm-real-motion", command)
            mouth_sample_argument = command.index("--mouth-sample-seconds") + 1
            self.assertEqual("1.0", command[mouth_sample_argument])
            velocity_argument = command.index("--trajectory-velocity-scaling") + 1
            acceleration_argument = command.index("--trajectory-acceleration-scaling") + 1
            hold_argument = command.index("--hold-duration") + 1
            self.assertEqual("0.3", command[velocity_argument])
            self.assertEqual("0.3", command[acceleration_argument])
            self.assertEqual("5.0", command[hold_argument])
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run", side_effect=fake_run):
            result = backend.run_real_feed_water(
                execute=False,
                environ={"UR10E_BACKEND": "real"},
            )
        self.assertTrue(result["success"])
        self.assertFalse(result["execution_attempted"])
        self.assertFalse(result["direct_mouth_contact"])
        self.assertFalse(result["cup_tilt_commanded"])
        self.assertFalse(result["pour_commanded"])
        self.assertFalse(result["automatic_retreat_sent"])
        self.assertEqual(
            "pre_mouth_and_return_target_validated", result["final_state"]
        )
        self.assertEqual(1.30, result["maximum_planned_displacement_m"])

    def test_execute_report_distinguishes_outbound_and_return_trajectories(self) -> None:
        pipeline = {
            "success": True,
            "stage": "returned_initial_position",
            "execution_attempted": True,
            "execution_sent": True,
            "pre_mouth_hold": {
                "completed": True,
                "duration_sec": 5.0,
                "motion_command_sent": False,
            },
            "return_to_initial_position": {
                "success": True,
                "stage": "returned_initial_position",
                "execution_attempted": True,
                "execution_sent": True,
                "automatic_retreat_sent": True,
            },
        }

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            report_argument = command.index("--report-file") + 1
            Path(command[report_argument]).write_text(
                json.dumps(pipeline), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run", side_effect=fake_run):
            result = backend.run_real_feed_water(
                execute=True,
                confirm_real_motion=True,
                hold_duration_sec=5.0,
                environ={
                    "UR10E_BACKEND": "real",
                    "UR10E_ALLOW_REAL_EXECUTION": "1",
                },
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["outbound_execution_sent"])
        self.assertTrue(result["return_execution_sent"])
        self.assertTrue(result["automatic_retreat_sent"])
        self.assertTrue(result["hold_completed"])
        self.assertEqual("initial_position", result["final_state"])

    def test_return_refusal_reports_that_robot_remains_at_pre_mouth(self) -> None:
        pipeline = {
            "success": False,
            "stage": "return_to_initial_position_refused",
            "reason": "return trajectory waypoint 4 collides with octomap",
            "final_state": "holding_pre_mouth",
            "execution_attempted": True,
            "execution_sent": True,
            "pre_mouth_hold": {
                "completed": True,
                "duration_sec": 5.0,
                "motion_command_sent": False,
            },
            "return_to_initial_position": {
                "success": False,
                "stage": "return_pre_execution_trajectory_refused",
                "execution_attempted": False,
                "execution_sent": False,
                "automatic_retreat_sent": False,
            },
        }

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            report_argument = command.index("--report-file") + 1
            Path(command[report_argument]).write_text(
                json.dumps(pipeline), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 2, "", "")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            backend, "REPORT_DIR", Path(directory)
        ), patch.object(backend.subprocess, "run", side_effect=fake_run):
            result = backend.run_real_feed_water(
                execute=True,
                confirm_real_motion=True,
                hold_duration_sec=5.0,
                environ={
                    "UR10E_BACKEND": "real",
                    "UR10E_ALLOW_REAL_EXECUTION": "1",
                },
            )

        self.assertFalse(result["success"])
        self.assertTrue(result["outbound_execution_sent"])
        self.assertFalse(result["return_execution_sent"])
        self.assertEqual("holding_pre_mouth", result["final_state"])
        self.assertEqual("return_to_initial_position_refused", result["stage"])


if __name__ == "__main__":
    unittest.main()
