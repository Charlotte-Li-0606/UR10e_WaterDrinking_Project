#!/usr/bin/env python3
"""Plan-only validation that a Gazebo depth obstacle reaches MoveIt OctoMap.

The test model is spawned by ``scripts/spawn_octomap_test_obstacle.sh`` using
Gazebo transport only.  This module never creates a MoveIt CollisionObject for
that model: it checks the raw/filtered PointCloud2 streams, then compares the
same safe pre-mouth MoveIt request with and without the depth-visible model.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_tools.feeding_tools import FeedingSkillLibrary  # noqa: E402


RAW_TOPIC = "/wrist_rgbd/points"
FILTERED_TOPIC = "/wrist_rgbd/filtered_cloud"
BASE_FRAME = "base_link"


@dataclass
class CloudRecord:
    message: PointCloud2
    received_monotonic: float
    sequence: int


def _rotation_matrix_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("transform quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class CloudProbe(Node):
    """Observe compact PointCloud2 metadata and obstacle-volume point counts."""

    def __init__(self) -> None:
        super().__init__("octomap_planning_validation_probe")
        self._records: dict[str, CloudRecord | None] = {RAW_TOPIC: None, FILTERED_TOPIC: None}
        self._sequences: dict[str, int] = {RAW_TOPIC: 0, FILTERED_TOPIC: 0}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self.create_subscription(PointCloud2, RAW_TOPIC, lambda message: self._record(RAW_TOPIC, message), qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2,
            FILTERED_TOPIC,
            lambda message: self._record(FILTERED_TOPIC, message),
            qos_profile_sensor_data,
        )

    def _record(self, topic: str, message: PointCloud2) -> None:
        self._sequences[topic] += 1
        self._records[topic] = CloudRecord(message, time.monotonic(), self._sequences[topic])

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.10, deadline - time.monotonic()))

    def sequence(self, topic: str) -> int:
        return self._sequences[topic]

    def wait_for_new(self, minimum_sequences: dict[str, int], timeout_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            if all(self._sequences[topic] > sequence for topic, sequence in minimum_sequences.items()):
                return
            rclpy.spin_once(self, timeout_sec=min(0.10, deadline - time.monotonic()))

    @staticmethod
    def _xyz_points(message: PointCloud2) -> np.ndarray:
        fields = {field.name: field for field in message.fields}
        required = ("x", "y", "z")
        if any(name not in fields for name in required):
            raise ValueError("PointCloud2 does not contain x/y/z fields")
        # MoveIt and this project's converter use FLOAT32 XYZ.  Avoid a huge
        # dump by decoding only the compact binary message in memory.
        if any(fields[name].datatype != 7 for name in required):
            raise ValueError("PointCloud2 x/y/z fields are not FLOAT32")
        count = int(message.width) * int(message.height)
        if count == 0:
            return np.empty((0, 3), dtype=np.float32)
        dtype = np.dtype(
            {
                "names": list(required),
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [fields[name].offset for name in required],
                "itemsize": int(message.point_step),
            }
        )
        if len(message.data) < count * int(message.point_step):
            raise ValueError("PointCloud2 data is shorter than width * height * point_step")
        structured = np.frombuffer(message.data, dtype=dtype, count=count)
        return np.column_stack((structured["x"], structured["y"], structured["z"]))

    def _points_in_base(self, message: PointCloud2) -> np.ndarray:
        points = self._xyz_points(message)
        points = points[np.all(np.isfinite(points), axis=1)]
        source = message.header.frame_id.strip().lstrip("/")
        if source == BASE_FRAME:
            return points.astype(np.float64, copy=False)
        transform = self._tf_buffer.lookup_transform(BASE_FRAME, source, Time())
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = _rotation_matrix_xyzw(rotation.x, rotation.y, rotation.z, rotation.w)
        return points.astype(np.float64, copy=False) @ matrix.T + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )

    def world_point_in_base(self, position_world: Sequence[float]) -> list[float]:
        transform = self._tf_buffer.lookup_transform(BASE_FRAME, "world", Time())
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = _rotation_matrix_xyzw(rotation.x, rotation.y, rotation.z, rotation.w)
        point = matrix @ np.asarray(position_world, dtype=np.float64) + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return [round(float(value), 6) for value in point]

    def summary(
        self,
        topic: str,
        obstacle_center_base: Sequence[float],
        obstacle_size: Sequence[float],
        margin_m: float,
    ) -> dict[str, Any]:
        record = self._records[topic]
        if record is None:
            return {"active": False, "topic": topic, "reason": "no PointCloud2 received"}
        message = record.message
        result: dict[str, Any] = {
            "active": (time.monotonic() - record.received_monotonic) <= 2.0,
            "frame_id": message.header.frame_id,
            "height": int(message.height),
            "point_count": int(message.width) * int(message.height),
            "sequence": record.sequence,
            "topic": topic,
            "width": int(message.width),
        }
        try:
            points = self._points_in_base(message)
            half_size = np.asarray(obstacle_size, dtype=np.float64) / 2.0 + float(margin_m)
            center = np.asarray(obstacle_center_base, dtype=np.float64)
            inside = np.all(np.abs(points - center) <= half_size, axis=1)
            result["points_in_obstacle_volume"] = int(np.count_nonzero(inside))
            result["finite_point_count"] = int(points.shape[0])
        except Exception as exc:
            result["point_volume_check_error"] = str(exc)
        return result


def _run_obstacle_helper(arguments: list[str]) -> dict[str, Any]:
    helper = PROJECT_ROOT / "scripts" / "spawn_octomap_test_obstacle.sh"
    completed = subprocess.run([str(helper), *arguments], text=True, capture_output=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {
        "command": [str(helper), *arguments],
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout.strip(),
        "success": completed.returncode == 0,
    }
    if lines:
        try:
            payload["result"] = json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    return payload


def _move_group_octomap_enabled() -> bool:
    completed = subprocess.run(
        ["ros2", "param", "get", "/move_group", "sensors"], text=True, capture_output=True, check=False
    )
    return completed.returncode == 0 and "wrist_rgbd_pointcloud" in completed.stdout


def _plan_summary(result: dict[str, Any]) -> dict[str, Any]:
    planner = result.get("planner_result") if isinstance(result.get("planner_result"), dict) else {}
    move = planner.get("move_result") if isinstance(planner.get("move_result"), dict) else {}
    target = result.get("pre_mouth_target")
    if target is None and isinstance(result.get("target"), dict):
        target = result["target"].get("pre_mouth_target")
    return {
        "final_target": target,
        "planner": planner.get("planner"),
        "planning_time": move.get("planning_time"),
        "points": move.get("points"),
        "reason": result.get("reason"),
        "stage": move.get("stage"),
        "success": bool(result.get("success")),
    }


def _visibility_effect(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_count = int(before.get("points_in_obstacle_volume", 0) or 0)
    after_count = int(after.get("points_in_obstacle_volume", 0) or 0)
    return bool(after.get("active")) and after_count >= max(3, before_count + 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obstacle-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=[0.350, 0.820, 1.650, 0.0, 0.0, 0.0],
        help="Gazebo-world obstacle pose. The default is calibrated near the pre-mouth approach.",
    )
    parser.add_argument(
        "--obstacle-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.120, 0.120, 0.250],
        help="Gazebo obstacle box size in metres.",
    )
    parser.add_argument("--settle-sec", type=float, default=5.0, help="Minimum cloud/OctoMap update wait after spawn.")
    parser.add_argument("--mouth-timeout-sec", type=float, default=10.0)
    parser.add_argument("--visibility-margin-m", type=float, default=0.025)
    parser.add_argument(
        "--remove-after",
        action="store_true",
        help="Remove the Gazebo obstacle after reporting. By default it remains for GUI inspection.",
    )
    args = parser.parse_args()
    if args.settle_sec <= 0.0 or args.mouth_timeout_sec <= 0.0 or args.visibility_margin_m < 0.0:
        parser.error("timings must be positive and visibility margin must be non-negative")
    if any(value <= 0.0 for value in args.obstacle_size):
        parser.error("obstacle dimensions must be positive")
    return args


def main() -> int:
    args = _parse_args()
    if not rclpy.ok():
        rclpy.init(args=None)
    probe = CloudProbe()
    tools: FeedingSkillLibrary | None = None
    spawned = False
    result: dict[str, Any] = {
        "event": "octomap_avoidance_validation",
        "octomap_enabled": _move_group_octomap_enabled(),
        "test_obstacle_spawned": False,
    }
    try:
        if not result["octomap_enabled"]:
            result.update(
                {
                    "avoidance_effect_demonstrated": False,
                    "reason": "MoveIt is not configured with wrist_rgbd_pointcloud; restart with USE_OCTOMAP=true",
                }
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        # Start from the required no-obstacle condition.  This only removes
        # the Gazebo model; it does not modify the fixed PlanningScene objects.
        _run_obstacle_helper(["--remove"])
        # Let fresh depth rays clear any voxels from a previous validation run
        # before collecting the no-obstacle baseline.
        probe.spin_for(max(2.0, args.settle_sec))
        obstacle_center_base = probe.world_point_in_base(args.obstacle_pose[:3])
        raw_baseline = probe.summary(RAW_TOPIC, obstacle_center_base, args.obstacle_size, args.visibility_margin_m)
        filtered_baseline = probe.summary(
            FILTERED_TOPIC, obstacle_center_base, args.obstacle_size, args.visibility_margin_m
        )

        tools = FeedingSkillLibrary(use_planning_scene=True)
        mouth = tools.wait_for_stable_mouth_pose(timeout_sec=args.mouth_timeout_sec)
        if not mouth.get("success"):
            result.update(
                {
                    "avoidance_effect_demonstrated": False,
                    "reason": str(mouth.get("reason") or "no stable mouth pose"),
                    "pointcloud_active": bool(raw_baseline.get("active")),
                    "filtered_cloud_active": bool(filtered_baseline.get("active")),
                }
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        # Reuse exactly one position-only mouth sample for both requests, so
        # target perception jitter cannot be mistaken for obstacle avoidance.
        mouth_pose = {
            "frame_id": str(mouth["mouth_pose"]["frame_id"]),
            "position": [float(value) for value in mouth["mouth_pose"]["position"]],
        }
        baseline_result = tools.move_straw_tip_to_pre_mouth(mouth_pose, execute=False)

        # Capture the sequence marker immediately before spawning.  Planning
        # and mouth acquisition above can take long enough for the previous
        # cloud to be several messages old; using an earlier marker would let
        # a pre-spawn message masquerade as an obstacle update.
        before_sequences = {RAW_TOPIC: probe.sequence(RAW_TOPIC), FILTERED_TOPIC: probe.sequence(FILTERED_TOPIC)}
        spawn = _run_obstacle_helper(
            [
                "--spawn",
                "--pose",
                " ".join(f"{value:.6f}" for value in args.obstacle_pose),
                "--size",
                " ".join(f"{value:.6f}" for value in args.obstacle_size),
            ]
        )
        spawned = bool(spawn.get("success"))
        result["test_obstacle_spawned"] = spawned
        if not spawned:
            result.update(
                {
                    "avoidance_effect_demonstrated": False,
                    "reason": "Gazebo could not spawn the test obstacle",
                    "spawn": spawn,
                    "baseline_plan": _plan_summary(baseline_result),
                }
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        # Do not return on the first message after creation: Gazebo/bridge
        # queues can contain one in-flight pre-spawn depth frame.  Consume the
        # full settling window, then require that both streams advanced.
        probe.spin_for(args.settle_sec)
        probe.wait_for_new(before_sequences, 1.0)
        raw_obstacle = probe.summary(RAW_TOPIC, obstacle_center_base, args.obstacle_size, args.visibility_margin_m)
        filtered_obstacle = probe.summary(FILTERED_TOPIC, obstacle_center_base, args.obstacle_size, args.visibility_margin_m)
        obstacle_result = tools.move_straw_tip_to_pre_mouth(mouth_pose, execute=False)

        raw_visible = _visibility_effect(raw_baseline, raw_obstacle)
        filtered_visible = _visibility_effect(filtered_baseline, filtered_obstacle)
        pointcloud_active = bool(raw_obstacle.get("active"))
        filtered_cloud_active = bool(filtered_obstacle.get("active"))
        baseline_plan = _plan_summary(baseline_result)
        obstacle_plan = _plan_summary(obstacle_result)
        safe_rejection = baseline_plan["success"] and not obstacle_plan["success"]
        point_count_changed = (
            baseline_plan["success"]
            and obstacle_plan["success"]
            and baseline_plan.get("points") is not None
            and obstacle_plan.get("points") is not None
            and baseline_plan["points"] != obstacle_plan["points"]
        )
        visibility_confirmed = raw_visible and filtered_visible
        avoidance_effect = visibility_confirmed and (safe_rejection or point_count_changed)
        if avoidance_effect and safe_rejection:
            evidence = "The depth-visible obstacle reached raw and filtered clouds, and MoveIt safely rejected the same pre-mouth request."
        elif avoidance_effect:
            evidence = "The depth-visible obstacle reached raw and filtered clouds, and the MoveIt trajectory point count changed for the same target."
        elif not visibility_confirmed:
            evidence = "Obstacle visibility in both raw and filtered cloud volumes was not confirmed; no planning conclusion is valid."
        else:
            evidence = "Cloud visibility was confirmed, but this plan-only request did not change point count or fail safely. Reposition the obstacle."

        result.update(
            {
                "avoidance_effect_demonstrated": avoidance_effect,
                "baseline_cloud": {"filtered": filtered_baseline, "raw": raw_baseline},
                "baseline_plan": baseline_plan,
                "evidence": evidence,
                "filtered_cloud_active": filtered_cloud_active,
                "obstacle_cloud": {"filtered": filtered_obstacle, "raw": raw_obstacle},
                "obstacle_center_base_link": obstacle_center_base,
                "obstacle_plan": obstacle_plan,
                "obstacle_visibility_confirmed": visibility_confirmed,
                "pointcloud_active": pointcloud_active,
                "spawn": spawn,
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if avoidance_effect else 3
    except Exception as exc:
        result.update({"avoidance_effect_demonstrated": False, "reason": f"validation error: {exc}"})
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 2
    finally:
        if spawned and args.remove_after:
            _run_obstacle_helper(["--remove"])
        if tools is not None:
            tools.close()
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
