"""No-motion unit checks for execution-scoped ROS 2 bag capture."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from robot_layer.arm_ur10e.agent_server.execution_rosbag import (
    ExecutionRosbagRecorder,
    capture_not_required,
)


class ExecutionRosbagTest(unittest.TestCase):
    def test_plan_only_summary_is_explicitly_not_required(self) -> None:
        summary = capture_not_required("plan only")

        self.assertFalse(summary["required"])
        self.assertFalse(summary["started"])
        self.assertIsNone(summary["verified"])
        self.assertFalse(summary["contains_identifiable_camera_data"])

    def test_command_uses_compressed_mcap_and_selected_topics(self) -> None:
        recorder = ExecutionRosbagRecorder(
            execution_id="20260814_120000_123456",
            environment={"PATH": "/usr/bin"},
            topics=("/joint_states", "/tf"),
        )

        command = recorder._command("/opt/ros/jazzy/bin/ros2")

        self.assertIn("mcap", command)
        self.assertIn("zstd_fast", command)
        self.assertIn("--include-unpublished-topics", command)
        self.assertEqual(["/joint_states", "/tf"], command[-2:])

    def test_missing_ros2_refuses_recorder_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "robot_layer.arm_ur10e.agent_server.execution_rosbag.shutil.which",
            return_value=None,
        ):
            recorder = ExecutionRosbagRecorder(
                execution_id="missing_ros2",
                environment={"PATH": "/missing"},
                output_root=Path(directory),
                minimum_free_bytes=0,
            )
            summary = recorder.start()

        self.assertFalse(summary["started"])
        self.assertFalse(summary["verified"])
        self.assertIn("ros2 executable not found", summary["reason"])

    def test_metadata_extracts_counts_duration_and_missing_topics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = ExecutionRosbagRecorder(
                execution_id="metadata",
                environment={},
                output_root=Path(directory),
                topics=("/joint_states", "/tf"),
                minimum_free_bytes=0,
            )
            recorder.bag_path.mkdir()
            metadata = {
                "rosbag2_bagfile_information": {
                    "message_count": 7,
                    "duration": {"nanoseconds": 2_500_000_000},
                    "topics_with_message_count": [
                        {
                            "topic_metadata": {"name": "/joint_states"},
                            "message_count": 7,
                        }
                    ],
                }
            }
            (recorder.bag_path / "metadata.yaml").write_text(
                yaml.safe_dump(metadata), encoding="utf-8"
            )

            information = recorder._load_metadata()
            self.assertIsNotNone(information)
            recorder._apply_metadata(information or {})
            summary = recorder.summary

        self.assertEqual(7, summary["message_count"])
        self.assertEqual(2.5, summary["duration_sec"])
        self.assertEqual(["/joint_states"], summary["topics_recorded"])
        self.assertEqual({"/joint_states": 7}, summary["topic_message_counts"])
        self.assertEqual(["/tf"], summary["missing_topics"])


if __name__ == "__main__":
    unittest.main()
