#!/usr/bin/env python3
"""Plan-only MoveIt check using the filtered tracked-mouth topic.

This adapter never enables execution and never publishes a robot command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk import UR10eRobotEnv
from scripts.real_premouth_from_perception_plan import _adaptive_premouth_pose_candidates


class TargetCollector(Node):
    """Collect fresh tracked poses in the calibrated base frame."""

    def __init__(self) -> None:
        super().__init__("tracked_target_plan_only")
        self.samples: list[tuple[float, np.ndarray]] = []
        self.cloud_received: dict[str, float] = {"/wrist_rgbd/points": 0.0,
                                                  "/wrist_rgbd/filtered_cloud": 0.0}
        self.create_subscription(PoseStamped, "/tracked_mouth_pose", self._callback, 10)
        for topic in self.cloud_received:
            self.create_subscription(PointCloud2, topic, lambda _msg, t=topic: self._cloud(t),
                                     qos_profile_sensor_data)
        self.clear_octomap = self.create_client(Empty, "/clear_octomap")

    # Store only finite base_link samples; no motion interface is present.
    def _callback(self, message: PoseStamped) -> None:
        if message.header.frame_id.strip().lstrip("/") != "base_link":
            return
        position = np.array([message.pose.position.x, message.pose.position.y, message.pose.position.z], dtype=float)
        if np.all(np.isfinite(position)):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.samples.append((stamp, position))

    def _cloud(self, topic: str) -> None:
        self.cloud_received[topic] = time.monotonic()

    def rebuild_dynamic_octomap(self, timeout_sec: float = 6.0) -> dict[str, object]:
        """Clear only dynamic occupancy and verify fresh clouds repopulate it."""
        result: dict[str, object] = {
            "clear_attempted": False,
            "fixed_human_objects_preserved": True,
            "fresh_cloud_topics": [],
            "success": False,
        }
        if not self.clear_octomap.wait_for_service(timeout_sec=1.0):
            result["reason"] = "/clear_octomap unavailable"
            return result
        before = dict(self.cloud_received)
        future = self.clear_octomap.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.5)
        result["clear_attempted"] = True
        if future.result() is None:
            result["reason"] = "/clear_octomap timed out"
            return result
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            fresh = [topic for topic, stamp in self.cloud_received.items()
                     if stamp > before[topic]]
            if len(fresh) == len(self.cloud_received):
                result["fresh_cloud_topics"] = fresh
                result["success"] = True
                return result
        result["fresh_cloud_topics"] = [topic for topic, stamp in self.cloud_received.items()
                                         if stamp > before[topic]]
        result["reason"] = "fresh raw and filtered clouds did not arrive after clear"
        return result


def _camera_ray(env: UR10eRobotEnv, mouth: np.ndarray) -> list[float]:
    """Return the calibrated camera-to-mouth unit vector in base_link."""
    transform = env.node.tf_buffer.lookup_transform("base_link", "d435i_color_optical_frame", Time())
    t = transform.transform.translation
    camera = np.array([t.x, t.y, t.z], dtype=float)
    direction = mouth - camera
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise RuntimeError("camera and tracked mouth positions are coincident")
    return (direction / norm).tolist()


def _wait_for_robot_tf(env: UR10eRobotEnv, timeout_sec: float = 5.0) -> None:
    """Wait for the driver TF tree before reading tool0 or camera poses."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            env.node.tf_buffer.lookup_transform("base_link", "tool0", Time())
            env.node.tf_buffer.lookup_transform("base_link", "d435i_color_optical_frame", Time())
            return
        except Exception:
            rclpy.spin_once(env.node, timeout_sec=0.1)
    raise RuntimeError("TF tree did not connect base_link to tool0/camera before planning")


def main() -> int:
    """Collect a stable tracked target and run one MoveIt plan-only request."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--pre-mouth-offset", nargs=3, type=float, default=[0.0, -0.08, 0.0])
    parser.add_argument("--report-file", type=Path, default=Path("reports/tracked_target_plan_only.json"))
    args = parser.parse_args()
    if args.sample_seconds <= 0.0 or args.minimum_samples < 1:
        raise SystemExit("sample duration and minimum samples must be positive")

    rclpy.init()
    collector = TargetCollector()
    env = None
    report: dict[str, object] = {"mode": "plan_only", "execution_sent": False}
    try:
        deadline = time.monotonic() + args.sample_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(collector, timeout_sec=0.05)
        if len(collector.samples) < args.minimum_samples:
            report.update({"success": False, "stage": "tracking_readiness", "reason": "insufficient_fresh_tracked_samples",
                           "sample_count": len(collector.samples)})
        else:
            positions = np.asarray([item[1] for item in collector.samples[-args.minimum_samples:]])
            target = positions.mean(axis=0)
            spread = float(np.max(np.linalg.norm(positions - target, axis=1)))
            report.update({"tracked_target_position": target.tolist(), "sample_count": len(collector.samples),
                           "sample_spread_m": spread})
            env = UR10eRobotEnv(init_ros_node=False)
            _wait_for_robot_tf(env)
            rebuild = collector.rebuild_dynamic_octomap()
            report["dynamic_octomap_rebuild"] = rebuild
            if not rebuild.get("success"):
                report.update({"success": False, "stage": "dynamic_octomap_rebuild",
                               "reason": rebuild.get("reason"), "execution_sent": False})
            else:
                current = env.get_robot_end_pose()
                candidates = _adaptive_premouth_pose_candidates(
                mouth_position_m=target.tolist(),
                approach_offset_unit=_camera_ray(env, target),
                verified_flange_down_orientation_xyzw=list(current["orientation_quat"]),
                )
                # Try the safest larger standoffs first and bound plan-only time.
                candidates.sort(key=lambda item: (-item["standoff_m"], -abs(item["yaw_deg"])))
                candidates = candidates[: max(1, args.max_candidates)]
                attempts = []
                selected = None
                for candidate in candidates:
                    pose = candidate["tool0_pose"]
                    q = pose["orientation_quat_xyzw"]
                    endpose = [*pose["position_m"], *q]
                    result = env.move_to_pose(endpose, plan_only=True)
                    attempts.append({"standoff_m": candidate["standoff_m"], "yaw_deg": candidate["yaw_deg"],
                                     "success": bool(result.get("success")), "result": result})
                    if result.get("success"):
                        selected = {"candidate": candidate, "result": result}
                        break
                report.update({"success": selected is not None, "stage": "moveit_adaptive_plan_only",
                               "attempts": attempts, "selected_candidate": selected,
                               "execution_sent": False})
    except Exception as exc:
        report.update({"success": False, "stage": "exception", "reason": str(exc), "execution_sent": False})
    finally:
        if env is not None:
            env.close()
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
