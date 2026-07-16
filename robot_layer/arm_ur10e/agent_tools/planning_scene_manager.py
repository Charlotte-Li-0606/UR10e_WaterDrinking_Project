#!/usr/bin/env python3
"""Deterministic mouth-driven collision objects for the UR10e PlanningScene.

This is intentionally not an occupancy-grid or point-cloud integration.  It
uses only a small, explicit set of MoveIt primitive objects in ``base_link``:
the human head, torso, and a face safety zone.  Re-applying an object with the
same ID updates its pose as a new mouth pose is received.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive


MANAGED_OBJECT_IDS = (
    "human_head_collision",
    "human_torso_collision",
    "face_safety_zone",
    "table_platform_collision",
)


@dataclass(frozen=True)
class PlanningSceneObstacleConfig:
    """Conservative, deterministic primitive geometry in ``base_link``."""

    base_frame: str = "base_link"
    mouth_topic: str = "/detected_mouth_pose"
    head_radius_m: float = 0.12
    # The person faces -Y in the supplied scene.  Biasing the spheres behind
    # the detected mouth preserves the 8 cm pre-mouth standoff in front.
    head_offset_m: tuple[float, float, float] = (0.0, 0.10, 0.03)
    face_safety_radius_m: float = 0.16
    face_safety_offset_m: tuple[float, float, float] = (0.0, 0.15, 0.03)
    torso_offset_m: tuple[float, float, float] = (0.0, 0.10, -0.35)
    torso_size_m: tuple[float, float, float] = (0.40, 0.25, 0.60)
    include_table: bool = False
    table_center_m: tuple[float, float, float] = (0.40, 0.55, -0.05)
    table_size_m: tuple[float, float, float] = (1.20, 0.80, 0.10)
    service_timeout_sec: float = 5.0
    mouth_wait_timeout_sec: float = 5.0


def _xyz(value: Sequence[float]) -> list[float]:
    return [round(float(component), 6) for component in value]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[index]) + float(b[index]) for index in range(3)]


def _is_valid_xyz(value: Sequence[float]) -> bool:
    return len(value) == 3 and all(math.isfinite(float(component)) for component in value)


class PlanningSceneObstacleManager(Node):
    """Apply, verify, and remove the deterministic mouth-driven obstacles."""

    def __init__(self, config: PlanningSceneObstacleConfig | None = None) -> None:
        super().__init__("ur10e_planning_scene_obstacle_manager")
        self.config = config or PlanningSceneObstacleConfig()
        self._validate_config(self.config)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._latest_mouth: list[float] | None = None
        self._latest_mouth_received_at: float | None = None
        self._latest_mouth_frame: str | None = None
        self.create_subscription(PoseStamped, self.config.mouth_topic, self._mouth_callback, qos)
        self._apply_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._get_client = self.create_client(GetPlanningScene, "/get_planning_scene")

    @staticmethod
    def _validate_config(config: PlanningSceneObstacleConfig) -> None:
        if not config.base_frame.strip() or not config.mouth_topic.strip():
            raise ValueError("base_frame and mouth_topic must not be empty")
        if config.head_radius_m <= 0.0 or config.face_safety_radius_m <= 0.0:
            raise ValueError("head and face-safety radii must be positive")
        if config.service_timeout_sec <= 0.0 or config.mouth_wait_timeout_sec <= 0.0:
            raise ValueError("service and mouth wait timeouts must be positive")
        for name, value in (
            ("head_offset_m", config.head_offset_m),
            ("face_safety_offset_m", config.face_safety_offset_m),
            ("torso_offset_m", config.torso_offset_m),
            ("table_center_m", config.table_center_m),
        ):
            if not _is_valid_xyz(value):
                raise ValueError(f"{name} must contain three finite values")
        for name, value in (("torso_size_m", config.torso_size_m), ("table_size_m", config.table_size_m)):
            if not _is_valid_xyz(value) or any(float(component) <= 0.0 for component in value):
                raise ValueError(f"{name} must contain three positive finite values")

    def _mouth_callback(self, message: PoseStamped) -> None:
        frame = message.header.frame_id.strip().lstrip("/")
        if frame != self.config.base_frame.strip().lstrip("/"):
            self.get_logger().warning(
                f"Ignoring mouth pose in frame {frame!r}; deterministic PlanningScene objects require "
                f"{self.config.base_frame!r}"
            )
            return
        position = [message.pose.position.x, message.pose.position.y, message.pose.position.z]
        if not _is_valid_xyz(position):
            self.get_logger().warning("Ignoring non-finite detected mouth pose")
            return
        self._latest_mouth = [float(component) for component in position]
        self._latest_mouth_received_at = time.monotonic()
        self._latest_mouth_frame = frame

    def wait_for_mouth(self, timeout_sec: float | None = None) -> dict[str, Any]:
        """Wait for a fresh ``base_link`` mouth pose without commanding motion."""
        timeout = self.config.mouth_wait_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._latest_mouth is not None:
                return {
                    "success": True,
                    "mouth_position": _xyz(self._latest_mouth),
                    "frame_id": self.config.base_frame,
                    "received_age_sec": round(time.monotonic() - (self._latest_mouth_received_at or time.monotonic()), 4),
                }
            rclpy.spin_once(self, timeout_sec=min(0.10, max(0.0, deadline - time.monotonic())))
        return {
            "success": False,
            "reason": f"no fresh {self.config.mouth_topic} pose in {self.config.base_frame} within {timeout:.1f} s",
        }

    def _pose(self, position: Sequence[float]) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(component) for component in position)
        pose.orientation.w = 1.0
        return pose

    def _sphere(self, object_id: str, center: Sequence[float], radius: float) -> CollisionObject:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [float(radius)]
        collision = CollisionObject()
        collision.header.frame_id = self.config.base_frame
        collision.id = object_id
        collision.operation = CollisionObject.ADD
        collision.primitives.append(primitive)
        collision.primitive_poses.append(self._pose(center))
        return collision

    def _box(self, object_id: str, center: Sequence[float], size: Sequence[float]) -> CollisionObject:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(component) for component in size]
        collision = CollisionObject()
        collision.header.frame_id = self.config.base_frame
        collision.id = object_id
        collision.operation = CollisionObject.ADD
        collision.primitives.append(primitive)
        collision.primitive_poses.append(self._pose(center))
        return collision

    def build_collision_objects(self, mouth_position: Sequence[float]) -> list[CollisionObject]:
        """Build the complete deterministic obstacle set for a mouth position."""
        if not _is_valid_xyz(mouth_position):
            raise ValueError("mouth_position must contain three finite values")
        mouth = [float(component) for component in mouth_position]
        objects = [
            self._sphere(
                "human_head_collision",
                _add(mouth, self.config.head_offset_m),
                self.config.head_radius_m,
            ),
            self._box(
                "human_torso_collision",
                _add(mouth, self.config.torso_offset_m),
                self.config.torso_size_m,
            ),
            self._sphere(
                "face_safety_zone",
                _add(mouth, self.config.face_safety_offset_m),
                self.config.face_safety_radius_m,
            ),
        ]
        if self.config.include_table:
            objects.append(
                self._box(
                    "table_platform_collision",
                    self.config.table_center_m,
                    self.config.table_size_m,
                )
            )
        return objects

    def _apply_scene(self, scene: PlanningScene) -> dict[str, Any]:
        if not self._apply_client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {"success": False, "reason": "/apply_planning_scene is unavailable"}
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config.service_timeout_sec)
        response = future.result()
        if response is None or not response.success:
            return {"success": False, "reason": "MoveIt rejected the PlanningScene update"}
        return {"success": True}

    def _world_object_ids(self) -> dict[str, Any]:
        """Read world collision IDs from MoveIt's current PlanningScene."""
        if not self._get_client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {"success": False, "reason": "/get_planning_scene is unavailable", "object_ids": []}
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        future = self._get_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config.service_timeout_sec)
        response = future.result()
        if response is None:
            return {"success": False, "reason": "MoveIt did not return a PlanningScene", "object_ids": []}
        return {
            "success": True,
            "object_ids": sorted(collision.id for collision in response.scene.world.collision_objects),
        }

    def verify(self, object_ids: Sequence[str]) -> dict[str, Any]:
        """Verify that requested collision IDs appear in MoveIt's world scene."""
        current = self._world_object_ids()
        if not current.get("success"):
            return current
        present = set(current["object_ids"])
        expected = set(object_ids)
        missing = sorted(expected - present)
        return {
            "success": not missing,
            "present_object_ids": sorted(expected & present),
            "missing_object_ids": missing,
        }

    def apply(
        self,
        mouth_position: Sequence[float] | None = None,
        *,
        verify: bool = True,
    ) -> dict[str, Any]:
        """Add or update all dynamic objects; repeated calls update by ID."""
        if mouth_position is None:
            mouth_result = self.wait_for_mouth()
            if not mouth_result.get("success"):
                return mouth_result
            mouth_position = mouth_result["mouth_position"]
        if not _is_valid_xyz(mouth_position):
            return {"success": False, "reason": "mouth_position must contain three finite values"}
        try:
            objects = self.build_collision_objects(mouth_position)
        except ValueError as exc:
            return {"success": False, "reason": str(exc)}
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.extend(objects)
        applied = self._apply_scene(scene)
        if not applied.get("success"):
            return applied
        object_ids = [collision.id for collision in objects]
        result: dict[str, Any] = {
            "success": True,
            "operation": "apply",
            "frame_id": self.config.base_frame,
            "mouth_position": _xyz(mouth_position),
            "object_ids": object_ids,
            "config": asdict(self.config),
        }
        if verify:
            verification = self.verify(object_ids)
            result["verification"] = verification
            if not verification.get("success"):
                result["success"] = False
                result["reason"] = "PlanningScene update could not be verified"
        return result

    def remove(self, *, verify: bool = True) -> dict[str, Any]:
        """Remove all collision IDs managed by this module from MoveIt's world scene."""
        # MoveIt can reject a REMOVE diff for an absent object.  Read the
        # scene first so this is idempotent and cleans up the optional table
        # whenever it was actually added by an earlier apply.
        current = self._world_object_ids()
        if not current.get("success"):
            return current
        object_ids = sorted(set(MANAGED_OBJECT_IDS) & set(current["object_ids"]))
        if not object_ids:
            return {
                "success": True,
                "operation": "remove",
                "frame_id": self.config.base_frame,
                "object_ids": [],
                "verification": {"success": True, "present_object_ids": [], "missing_object_ids": []},
                "note": "No managed PlanningScene objects were present.",
            }
        scene = PlanningScene()
        scene.is_diff = True
        for object_id in object_ids:
            collision = CollisionObject()
            collision.header.frame_id = self.config.base_frame
            collision.id = object_id
            collision.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(collision)
        applied = self._apply_scene(scene)
        if not applied.get("success"):
            return applied
        result: dict[str, Any] = {
            "success": True,
            "operation": "remove",
            "frame_id": self.config.base_frame,
            "object_ids": object_ids,
        }
        if verify:
            remaining = self._world_object_ids()
            present_object_ids = sorted(set(object_ids) & set(remaining.get("object_ids", [])))
            verification = {
                "success": bool(remaining.get("success")) and not present_object_ids,
                "present_object_ids": present_object_ids,
                "missing_object_ids": sorted(set(object_ids) - set(present_object_ids)),
            }
            if not remaining.get("success"):
                verification["reason"] = remaining.get("reason")
            result["verification"] = verification
            if not verification["success"]:
                result["success"] = False
                result["reason"] = "PlanningScene removal could not be verified"
        return result


def _parse_xyz(value: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated finite values") from exc
    if not _is_valid_xyz(parsed):
        raise argparse.ArgumentTypeError("expected three comma-separated finite values")
    return parsed


def _positive(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true", help="Add or update deterministic mouth-driven obstacles.")
    action.add_argument("--remove", action="store_true", help="Remove this manager's collision objects.")
    parser.add_argument("--mouth-topic", default="/detected_mouth_pose")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--mouth", type=_parse_xyz, default=None, help="Explicit base-link mouth x,y,z; avoids waiting for a topic.")
    parser.add_argument("--mouth-wait-timeout-sec", type=_positive, default=5.0)
    parser.add_argument("--service-timeout-sec", type=_positive, default=5.0)
    parser.add_argument("--head-radius", type=_positive, default=0.12)
    parser.add_argument("--head-offset", type=_parse_xyz, default=(0.0, 0.10, 0.03))
    parser.add_argument("--face-safety-radius", type=_positive, default=0.16)
    parser.add_argument("--face-safety-offset", type=_parse_xyz, default=(0.0, 0.15, 0.03))
    parser.add_argument("--torso-offset", type=_parse_xyz, default=(0.0, 0.10, -0.35))
    parser.add_argument("--torso-size", type=_parse_xyz, default=(0.40, 0.25, 0.60))
    parser.add_argument("--include-table", action="store_true", help="Add the optional configured table/platform box.")
    parser.add_argument("--table-center", type=_parse_xyz, default=(0.40, 0.55, -0.05))
    parser.add_argument("--table-size", type=_parse_xyz, default=(1.20, 0.80, 0.10))
    parser.add_argument("--no-verify", action="store_true", help="Skip /get_planning_scene verification.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = PlanningSceneObstacleConfig(
        base_frame=args.base_frame,
        mouth_topic=args.mouth_topic,
        head_radius_m=args.head_radius,
        head_offset_m=args.head_offset,
        face_safety_radius_m=args.face_safety_radius,
        face_safety_offset_m=args.face_safety_offset,
        torso_offset_m=args.torso_offset,
        torso_size_m=args.torso_size,
        include_table=args.include_table,
        table_center_m=args.table_center,
        table_size_m=args.table_size,
        service_timeout_sec=args.service_timeout_sec,
        mouth_wait_timeout_sec=args.mouth_wait_timeout_sec,
    )
    rclpy.init(args=None)
    manager = PlanningSceneObstacleManager(config)
    try:
        result = (
            manager.apply(args.mouth, verify=not args.no_verify)
            if args.apply
            else manager.remove(verify=not args.no_verify)
        )
    except Exception as exc:
        result = {"success": False, "reason": str(exc)}
    finally:
        manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
