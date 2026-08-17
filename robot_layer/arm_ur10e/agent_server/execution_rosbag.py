#!/usr/bin/env python3
"""Bounded ROS 2 bag capture for guarded real feed-water executions.

The recorder is deliberately separate from the robot motion implementation. It
subscribes only; it never publishes a robot command or changes a controller.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_BAG_ROOT = PROJECT_ROOT / "reports" / "execution_rosbags"
MINIMUM_FREE_BYTES = 8 * 1024**3
RECORDER_READY_TIMEOUT_SEC = 5.0
RECORDER_STOP_TIMEOUT_SEC = 15.0

# Raw RGB and depth make perception failures reproducible. The debug-image
# topic is intentionally omitted because it duplicates the identifiable RGB
# stream. These bags must remain local and must never be committed.
EXECUTION_BAG_TOPICS = (
    "/tf",
    "/tf_static",
    "/joint_states",
    "/scaled_joint_trajectory_controller/controller_state",
    "/scaled_joint_trajectory_controller/joint_trajectory",
    "/speed_scaling_state_broadcaster/speed_scaling",
    "/io_and_status_controller/robot_mode",
    "/io_and_status_controller/safety_mode",
    "/io_and_status_controller/robot_program_running",
    "/force_torque_sensor_broadcaster/wrench_filtered",
    "/d435i/d435i/color/image_raw",
    "/d435i/d435i/color/camera_info",
    "/d435i/d435i/aligned_depth_to_color/image_raw",
    "/d435i/d435i/aligned_depth_to_color/camera_info",
    "/detected_mouth_pose",
    "/detected_mouth_candidates",
    "/detected_mouth_normal",
    "/mouth_detection/status",
    "/tracked_mouth_pose",
    "/mouth_tracking/status",
    "/continuous_mouth_tracking/status",
    "/continuous_mouth_tracking/target_pose",
    "/servo_node/status",
    "/servo_node/delta_twist_cmds",
    "/servo_node/pose_target_cmds",
    "/monitored_planning_scene",
    "/display_planned_path",
    "/trajectory_execution_event",
    "/diagnostics",
    "/rosout",
)


def capture_not_required(reason: str) -> dict[str, Any]:
    """Return the stable report shape used when no real execution may occur."""
    return {
        "required": False,
        "started": False,
        "stopped_cleanly": None,
        "verified": None,
        "reason": reason,
        "bag_path": None,
        "recorder_log_path": None,
        "storage_id": "mcap",
        "storage_preset": "zstd_fast",
        "topics_requested": list(EXECUTION_BAG_TOPICS),
        "topics_recorded": [],
        "topic_message_counts": {},
        "missing_topics": [],
        "message_count": 0,
        "duration_sec": 0.0,
        "size_bytes": 0,
        "contains_identifiable_camera_data": False,
    }


class ExecutionRosbagRecorder:
    """Start, verify, and cleanly finalize one execution-scoped rosbag."""

    def __init__(
        self,
        *,
        execution_id: str,
        environment: Mapping[str, str],
        output_root: Path | None = None,
        topics: tuple[str, ...] = EXECUTION_BAG_TOPICS,
        minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    ) -> None:
        self.execution_id = execution_id
        self.environment = dict(environment)
        self.output_root = output_root or EXECUTION_BAG_ROOT
        self.topics = tuple(topics)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.bag_path = self.output_root / f"feed_water_real_{execution_id}"
        self.log_path = self.output_root / f"feed_water_real_{execution_id}.rosbag.log.txt"
        self.node_name = f"feed_water_recorder_{execution_id}"
        self._process: subprocess.Popen[str] | None = None
        self._log_file: Any = None
        self._summary = self._base_summary()

    def _base_summary(self) -> dict[str, Any]:
        return {
            "required": True,
            "started": False,
            "stopped_cleanly": None,
            "verified": False,
            "reason": None,
            "bag_path": str(self.bag_path),
            "recorder_log_path": str(self.log_path),
            "storage_id": "mcap",
            "storage_preset": "zstd_fast",
            "topics_requested": list(self.topics),
            "topics_recorded": [],
            "topic_message_counts": {},
            "missing_topics": [],
            "message_count": 0,
            "duration_sec": 0.0,
            "size_bytes": 0,
            "contains_identifiable_camera_data": True,
        }

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self._summary)

    def _command(self, ros2_executable: str) -> list[str]:
        return [
            ros2_executable,
            "bag",
            "record",
            "--output",
            str(self.bag_path),
            "--storage",
            "mcap",
            "--storage-preset-profile",
            "zstd_fast",
            "--disable-keyboard-controls",
            "--include-unpublished-topics",
            "--node-name",
            self.node_name,
            "--custom-data",
            "workflow=feed_water",
            f"execution_id={self.execution_id}",
            "contains_identifiable_camera_data=true",
            "--topics",
            *self.topics,
        ]

    def start(self) -> dict[str, Any]:
        """Start the recorder and refuse readiness unless it stays alive."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.bag_path.exists():
            self._summary["reason"] = "execution rosbag output path already exists"
            return self.summary

        free_bytes = shutil.disk_usage(self.output_root).free
        self._summary["free_bytes_before"] = int(free_bytes)
        self._summary["minimum_free_bytes"] = self.minimum_free_bytes
        if free_bytes < self.minimum_free_bytes:
            self._summary["reason"] = (
                "insufficient free disk space for mandatory execution rosbag: "
                f"{free_bytes} bytes available, {self.minimum_free_bytes} required"
            )
            return self.summary

        ros2_executable = shutil.which("ros2", path=self.environment.get("PATH"))
        if not ros2_executable:
            self._summary["reason"] = "ros2 executable not found for mandatory execution rosbag"
            return self.summary

        self._log_file = self.log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self._command(ros2_executable),
                cwd=PROJECT_ROOT,
                env=self.environment,
                text=True,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._summary["reason"] = f"failed to start mandatory execution rosbag: {exc}"
            self._close_log()
            return self.summary

        deadline = time.monotonic() + RECORDER_READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                self._summary["reason"] = (
                    "mandatory execution rosbag exited during startup with code "
                    f"{return_code}"
                )
                self._close_log()
                return self.summary
            if self.bag_path.is_dir() and any(self.bag_path.iterdir()):
                self._summary["started"] = True
                self._summary["reason"] = "recording"
                return self.summary
            time.sleep(0.1)

        self._summary["reason"] = "mandatory execution rosbag did not become ready in time"
        self._stop_process()
        self._close_log()
        return self.summary

    def _stop_process(self) -> bool:
        if self._process is None:
            return False
        if self._process.poll() is not None:
            return self._process.returncode == 0
        try:
            os.killpg(self._process.pid, signal.SIGINT)
            self._process.wait(timeout=RECORDER_STOP_TIMEOUT_SEC)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        return self._process.returncode == 0

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _load_metadata(self) -> Mapping[str, Any] | None:
        metadata_path = self.bag_path / "metadata.yaml"
        try:
            document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(document, Mapping):
            return None
        information = document.get("rosbag2_bagfile_information")
        return information if isinstance(information, Mapping) else None

    def _apply_metadata(self, information: Mapping[str, Any]) -> None:
        topics_recorded: list[str] = []
        topic_message_counts: dict[str, int] = {}
        for entry in information.get("topics_with_message_count", []):
            if not isinstance(entry, Mapping):
                continue
            metadata = entry.get("topic_metadata")
            count = entry.get("message_count", 0)
            if isinstance(metadata, Mapping) and int(count or 0) > 0:
                name = metadata.get("name")
                if isinstance(name, str):
                    topics_recorded.append(name)
                    topic_message_counts[name] = int(count)
        duration = information.get("duration")
        duration_ns = duration.get("nanoseconds", 0) if isinstance(duration, Mapping) else 0
        self._summary["message_count"] = int(information.get("message_count", 0) or 0)
        self._summary["duration_sec"] = float(duration_ns or 0) / 1_000_000_000.0
        self._summary["topics_recorded"] = sorted(set(topics_recorded))
        self._summary["topic_message_counts"] = dict(sorted(topic_message_counts.items()))
        self._summary["missing_topics"] = sorted(set(self.topics) - set(topics_recorded))

    def stop(self) -> dict[str, Any]:
        """Finalize the bag and return metadata suitable for the JSON report."""
        clean = self._stop_process() if self._process is not None else False
        self._close_log()
        self._summary["stopped_cleanly"] = clean
        if self.bag_path.is_dir():
            self._summary["size_bytes"] = sum(
                path.stat().st_size for path in self.bag_path.rglob("*") if path.is_file()
            )
        information = self._load_metadata()
        if information is not None:
            self._apply_metadata(information)
        self._summary["verified"] = bool(
            self._summary["started"]
            and clean
            and information is not None
            and self._summary["message_count"] > 0
        )
        if self._summary["verified"]:
            self._summary["reason"] = "recording finalized and metadata verified"
        elif self._summary["started"]:
            self._summary["reason"] = (
                "execution rosbag started but did not finalize with verified message metadata"
            )
        return self.summary
