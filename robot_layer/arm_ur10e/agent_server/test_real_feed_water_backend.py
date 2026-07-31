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
        }

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            report_argument = command.index("--report-file") + 1
            Path(command[report_argument]).write_text(json.dumps(pipeline), encoding="utf-8")
            self.assertIn("camera-ray", command)
            self.assertIn("--no-execute", command)
            self.assertNotIn("--confirm-real-motion", command)
            maximum_translation_argument = command.index("--maximum-plan-translation") + 1
            self.assertEqual("1.3", command[maximum_translation_argument])
            mouth_sample_argument = command.index("--mouth-sample-seconds") + 1
            self.assertEqual("1.0", command[mouth_sample_argument])
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


if __name__ == "__main__":
    unittest.main()
