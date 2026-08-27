#!/usr/bin/env python3
"""Verify the opt-in Grasp-Anything observation pipeline without motion."""

from __future__ import annotations

import argparse
import json
import os
import time
from urllib.request import urlopen

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String


EXPECTED_MODEL_SHA256 = (
    "65984ef3364790c1ece107f22bcbeb67dc8fba21784087bb3d8ff183a3582e0a"
)


class ObservationVerifier(Node):
    def __init__(self) -> None:
        super().__init__("verify_pgi_grasp_anything_perception")
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.clock_values: list[float] = []
        self.debug_stamps: list[tuple[int, int]] = []
        self.status: dict | None = None
        self.metadata: dict | None = None
        self.pose: PoseStamped | None = None
        self.create_subscription(Clock, "/clock", self._on_clock, sensor_qos)
        self.create_subscription(
            Image,
            "/pgi/grasp_anything/debug_image",
            self._on_debug,
            sensor_qos,
        )
        self.create_subscription(
            String,
            "/pgi/grasp_anything/status",
            self._on_status,
            reliable_qos,
        )
        self.create_subscription(
            String,
            "/pgi/grasp_anything/candidate",
            self._on_metadata,
            reliable_qos,
        )
        self.create_subscription(
            PoseStamped,
            "/pgi/grasp_anything/candidate_pose",
            self._on_pose,
            reliable_qos,
        )

    def _on_clock(self, message: Clock) -> None:
        value = message.clock.sec + message.clock.nanosec * 1e-9
        if not self.clock_values or value != self.clock_values[-1]:
            self.clock_values.append(value)

    def _on_debug(self, message: Image) -> None:
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if not self.debug_stamps or stamp != self.debug_stamps[-1]:
            self.debug_stamps.append(stamp)

    def _on_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.status = payload

    def _on_metadata(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.metadata = payload

    def _on_pose(self, message: PoseStamped) -> None:
        self.pose = message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--health-url", default="http://127.0.0.1:8765/health"
    )
    return parser.parse_args()


def fail(reason: str, **details) -> int:
    print(
        json.dumps(
            {
                "success": False,
                "reason": reason,
                "observation_only": True,
                "controller_command_sent": False,
                "real_robot_command_sent": False,
                **details,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1


def main() -> int:
    args = parse_args()
    try:
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    except ValueError:
        return fail("invalid_ros_domain_id")
    if domain_id == 0:
        return fail("refusing_ros_domain_zero")
    if args.timeout <= 0.0:
        return fail("invalid_timeout")
    if not args.health_url.startswith("http://127.0.0.1:"):
        return fail("health_url_must_be_loopback")
    try:
        with urlopen(args.health_url, timeout=3.0) as response:
            health = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return fail("health_check_failed", detail=str(error))
    if health.get("success") is not True or health.get("motion_capable") is not False:
        return fail("invalid_health_contract", health=health)
    if health.get("model_sha256") != EXPECTED_MODEL_SHA256:
        return fail("model_checksum_mismatch", health=health)

    rclpy.init()
    node = ObservationVerifier()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if (
                len(node.clock_values) >= 2
                and len(node.debug_stamps) >= 2
                and node.status is not None
            ):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(node.clock_values) < 2 or node.clock_values[-1] <= node.clock_values[0]:
        return fail("gazebo_clock_not_advancing")
    if len(node.debug_stamps) < 2:
        return fail("debug_image_not_fresh")
    status = node.status
    if status is None:
        return fail("no_structured_status")
    if status.get("observation_only") is not True:
        return fail("status_not_observation_only", status=status)
    detected = status.get("detected")
    if not isinstance(detected, bool):
        return fail("status_missing_detected_boolean", status=status)
    if detected:
        if node.metadata is None or node.metadata.get("accepted") is not True:
            return fail("detected_without_candidate_metadata", status=status)
        if node.metadata.get("observation_only") is not True or node.pose is None:
            return fail("candidate_contract_incomplete", metadata=node.metadata)

    print(
        json.dumps(
            {
                "success": True,
                "reason": "observation_pipeline_verified",
                "ros_domain_id": domain_id,
                "model_sha256": health["model_sha256"],
                "clock_advanced_seconds": round(
                    node.clock_values[-1] - node.clock_values[0], 6
                ),
                "fresh_debug_frames": len(node.debug_stamps),
                "latest_status": status,
                "candidate_published": bool(detected),
                "observation_only": True,
                "controller_command_sent": False,
                "real_robot_command_sent": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
