#!/usr/bin/env python3
"""Deterministic mouth-driven collision objects for the UR10e PlanningScene.

This is intentionally not an occupancy-grid or point-cloud integration.  It
uses a small, explicit set of MoveIt primitives: human safety objects in
``base_link`` and one conservative camera/cup-holder/straw box rigidly attached
to ``tool0``.  Re-applying an object with the same ID updates its pose.
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
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
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

MULTI_PERSON_OBJECT_PREFIX = "real_human_obstacle_"

# The physical camera, cup holder, and straw are conservatively represented as
# one rigid body attached to tool0.  The reviewed dimensions are 10 x 10 x
# 30 cm.  On this installation tool0 +Z is the flange's downward axis
# (base_link -Z while the cup is upright), so a center at +0.15 m makes the box
# start at the flange plane and extend 0.30 m downward.  The adjacent flange
# links are the only intentional touch links; collisions with every other
# robot link, world object, human safety object, and OctoMap remain enabled.
COMBINED_TOOL_COLLISION_OBJECT_ID = "combined_camera_cup_holder_straw_collision"
COMBINED_TOOL_COLLISION_LINK_NAME = "tool0"
COMBINED_TOOL_COLLISION_SIZE_M = (0.10, 0.10, 0.30)
COMBINED_TOOL_COLLISION_CENTER_TOOL0_M = (0.0, 0.0, 0.15)
COMBINED_TOOL_COLLISION_TOUCH_LINKS = ("tool0", "flange", "wrist_3_link")


def _normalized_quaternion_xyzw(value: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(component) ** 2 for component in value))
    if not math.isfinite(magnitude) or magnitude < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(component) / magnitude for component in value]


def _quaternion_multiply_xyzw(
    first: Sequence[float], second: Sequence[float]
) -> list[float]:
    x1, y1, z1, w1 = _normalized_quaternion_xyzw(first)
    x2, y2, z2, w2 = _normalized_quaternion_xyzw(second)
    return _normalized_quaternion_xyzw(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _rotate_vector_xyzw(
    orientation: Sequence[float], vector: Sequence[float]
) -> list[float]:
    quaternion = _normalized_quaternion_xyzw(orientation)
    # Use the equivalent rotation-matrix expression so the vector magnitude is
    # retained exactly.
    x, y, z, w = quaternion
    vx, vy, vz = (float(component) for component in vector)
    return [
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    ]


def combined_tool_collision_verification(
    attached_objects: Sequence[AttachedCollisionObject],
) -> dict[str, Any]:
    """Verify the exact attached primitive required for guarded execution."""
    matches = [
        item
        for item in attached_objects
        if str(item.object.id) == COMBINED_TOOL_COLLISION_OBJECT_ID
    ]
    base: dict[str, Any] = {
        "success": False,
        "real_execution_geometry_complete": False,
        "object_id": COMBINED_TOOL_COLLISION_OBJECT_ID,
        "modeling_strategy": "one conservative attached box for camera, cup holder, and straw",
        "expected_link_name": COMBINED_TOOL_COLLISION_LINK_NAME,
        "expected_frame_id": COMBINED_TOOL_COLLISION_LINK_NAME,
        "expected_dimensions_m": list(COMBINED_TOOL_COLLISION_SIZE_M),
        "expected_center_tool0_m": list(COMBINED_TOOL_COLLISION_CENTER_TOOL0_M),
        "expected_span_tool0_z_m": [0.0, COMBINED_TOOL_COLLISION_SIZE_M[2]],
        "expected_touch_links": list(COMBINED_TOOL_COLLISION_TOUCH_LINKS),
        "tool0_positive_z_is_base_link_negative_z_when_flange_down": True,
        "matched_object_count": len(matches),
    }
    if len(matches) != 1:
        return {
            **base,
            "reason": (
                f"expected exactly one attached {COMBINED_TOOL_COLLISION_OBJECT_ID!r} "
                f"object, found {len(matches)}"
            ),
        }

    attached = matches[0]
    collision = attached.object
    primitive_count = len(collision.primitives)
    primitive_pose_count = len(collision.primitive_poses)
    primitive = collision.primitives[0] if primitive_count == 1 else None
    primitive_pose = (
        collision.primitive_poses[0] if primitive_pose_count == 1 else None
    )
    actual_dimensions = (
        [float(value) for value in primitive.dimensions]
        if primitive is not None
        else []
    )
    primitive_center = (
        [
            float(primitive_pose.position.x),
            float(primitive_pose.position.y),
            float(primitive_pose.position.z),
        ]
        if primitive_pose is not None
        else []
    )
    primitive_orientation = (
        [
            float(primitive_pose.orientation.x),
            float(primitive_pose.orientation.y),
            float(primitive_pose.orientation.z),
            float(primitive_pose.orientation.w),
        ]
        if primitive_pose is not None
        else []
    )
    object_position = [
        float(collision.pose.position.x),
        float(collision.pose.position.y),
        float(collision.pose.position.z),
    ]
    object_orientation = [
        float(collision.pose.orientation.x),
        float(collision.pose.orientation.y),
        float(collision.pose.orientation.z),
        float(collision.pose.orientation.w),
    ]
    # MoveIt may canonicalize an attached shape by moving the submitted
    # primitive offset into CollisionObject.pose.  Verify the composed pose so
    # both equivalent message representations are accepted without weakening
    # the physical geometry check.
    actual_center = (
        [
            object_position[index] + rotated
            for index, rotated in enumerate(
                _rotate_vector_xyzw(object_orientation, primitive_center)
            )
        ]
        if primitive_pose is not None
        else []
    )
    actual_orientation = (
        _quaternion_multiply_xyzw(object_orientation, primitive_orientation)
        if primitive_pose is not None
        else []
    )
    link_ok = str(attached.link_name) == COMBINED_TOOL_COLLISION_LINK_NAME
    frame_ok = str(collision.header.frame_id).lstrip("/") == (
        COMBINED_TOOL_COLLISION_LINK_NAME
    )
    shape_ok = bool(
        primitive is not None and int(primitive.type) == SolidPrimitive.BOX
    )
    dimensions_ok = bool(
        len(actual_dimensions) == 3
        and all(
            abs(measured - expected) <= 1e-9
            for measured, expected in zip(
                actual_dimensions, COMBINED_TOOL_COLLISION_SIZE_M
            )
        )
    )
    center_ok = bool(
        len(actual_center) == 3
        and all(
            abs(measured - expected) <= 1e-9
            for measured, expected in zip(
                actual_center, COMBINED_TOOL_COLLISION_CENTER_TOOL0_M
            )
        )
    )
    orientation_ok = bool(
        len(actual_orientation) == 4
        and all(abs(value) <= 1e-9 for value in actual_orientation[:3])
        and abs(abs(actual_orientation[3]) - 1.0) <= 1e-9
    )
    actual_touch_links = sorted(str(value) for value in attached.touch_links)
    touch_links_ok = actual_touch_links == sorted(
        COMBINED_TOOL_COLLISION_TOUCH_LINKS
    )
    success = bool(
        link_ok
        and frame_ok
        and shape_ok
        and dimensions_ok
        and center_ok
        and orientation_ok
        and touch_links_ok
    )
    return {
        **base,
        "success": success,
        "real_execution_geometry_complete": success,
        "actual_link_name": str(attached.link_name),
        "actual_frame_id": str(collision.header.frame_id),
        "actual_shape": "box" if shape_ok else (
            None if primitive is None else str(int(primitive.type))
        ),
        "actual_dimensions_m": actual_dimensions,
        "actual_center_tool0_m": actual_center,
        "actual_orientation_quat_xyzw": actual_orientation,
        "object_pose_position_tool0_m": object_position,
        "object_pose_orientation_quat_xyzw": object_orientation,
        "primitive_pose_position_m": primitive_center,
        "primitive_pose_orientation_quat_xyzw": primitive_orientation,
        "actual_touch_links": actual_touch_links,
        "checks": {
            "link": link_ok,
            "frame": frame_ok,
            "shape": shape_ok,
            "dimensions": dimensions_ok,
            "center": center_ok,
            "orientation": orientation_ok,
            "touch_links": touch_links_ok,
        },
        "reason": (
            None
            if success
            else "attached combined-tool collision geometry does not match the reviewed specification"
        ),
    }


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


def _scale(vector: Sequence[float], scalar: float) -> list[float]:
    return [float(component) * float(scalar) for component in vector]


def _normalize(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(component) ** 2 for component in vector))
    if not math.isfinite(magnitude) or magnitude < 1e-6:
        raise ValueError("camera and mouth positions must not be coincident")
    return [float(component) / magnitude for component in vector]


def _is_valid_xyz(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(component)) for component in value)
    except (TypeError, ValueError):
        return False


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

    @staticmethod
    def build_combined_tool_attached_collision() -> AttachedCollisionObject:
        """Build the reviewed rigid 10 x 10 x 30 cm tool assembly box."""
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(COMBINED_TOOL_COLLISION_SIZE_M)

        primitive_pose = Pose()
        (
            primitive_pose.position.x,
            primitive_pose.position.y,
            primitive_pose.position.z,
        ) = COMBINED_TOOL_COLLISION_CENTER_TOOL0_M
        primitive_pose.orientation.w = 1.0

        collision = CollisionObject()
        collision.header.frame_id = COMBINED_TOOL_COLLISION_LINK_NAME
        collision.id = COMBINED_TOOL_COLLISION_OBJECT_ID
        collision.pose.orientation.w = 1.0
        collision.operation = CollisionObject.ADD
        collision.primitives.append(primitive)
        collision.primitive_poses.append(primitive_pose)

        attached = AttachedCollisionObject()
        attached.link_name = COMBINED_TOOL_COLLISION_LINK_NAME
        attached.object = collision
        attached.touch_links = list(COMBINED_TOOL_COLLISION_TOUCH_LINKS)
        return attached

    def _attached_collision_objects(self) -> dict[str, Any]:
        """Read attached bodies from MoveIt's monitored PlanningScene."""
        if not self._get_client.wait_for_service(
            timeout_sec=self.config.service_timeout_sec
        ):
            return {
                "success": False,
                "reason": "/get_planning_scene is unavailable",
                "attached_objects": [],
            }
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self._get_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.config.service_timeout_sec,
        )
        response = future.result()
        if response is None:
            return {
                "success": False,
                "reason": "MoveIt did not return attached collision objects",
                "attached_objects": [],
            }
        return {
            "success": True,
            "attached_objects": list(
                response.scene.robot_state.attached_collision_objects
            ),
        }

    def apply_combined_tool_collision(
        self, *, verify: bool = True
    ) -> dict[str, Any]:
        """Attach and optionally verify the combined physical tool geometry."""
        attached = self.build_combined_tool_attached_collision()
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)
        applied = self._apply_scene(scene)
        if not applied.get("success"):
            return {
                **applied,
                "operation": "attach_combined_tool_collision",
                "execution_sent": False,
            }

        result: dict[str, Any] = {
            "success": True,
            "operation": "attach_combined_tool_collision",
            "object_id": COMBINED_TOOL_COLLISION_OBJECT_ID,
            "link_name": COMBINED_TOOL_COLLISION_LINK_NAME,
            "frame_id": COMBINED_TOOL_COLLISION_LINK_NAME,
            "dimensions_m": list(COMBINED_TOOL_COLLISION_SIZE_M),
            "center_tool0_m": list(COMBINED_TOOL_COLLISION_CENTER_TOOL0_M),
            "span_tool0_z_m": [0.0, COMBINED_TOOL_COLLISION_SIZE_M[2]],
            "touch_links": list(COMBINED_TOOL_COLLISION_TOUCH_LINKS),
            "follows_tool0": True,
            "collision_checking_enabled_against_non_touch_links": True,
            "execution_sent": False,
        }
        if verify:
            current = self._attached_collision_objects()
            if not current.get("success"):
                result.update(
                    {
                        "success": False,
                        "reason": current.get("reason"),
                        "verification": current,
                    }
                )
                return result
            verification = combined_tool_collision_verification(
                current["attached_objects"]
            )
            result["verification"] = verification
            if not verification.get("success"):
                result["success"] = False
                result["reason"] = verification.get("reason")
        return result

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

    @staticmethod
    def _person_object_id(person_index: int, shape: str) -> str:
        return f"{MULTI_PERSON_OBJECT_PREFIX}{person_index}_{shape}"

    def build_collision_objects_for_people(
        self,
        mouth_positions: Sequence[Sequence[float]],
        *,
        camera_position: Sequence[float],
    ) -> tuple[list[CollisionObject], list[dict[str, Any]]]:
        """Build camera-facing head, torso, and face zones for every person.

        The legacy single-person geometry assumes the fixed Gazebo ``+Y``
        facing direction.  The real wrist camera can approach from another
        base-link direction, so this variant derives the direction behind each
        face from the camera-to-mouth ray while retaining the reviewed radii,
        vertical offsets, and torso dimensions.
        """
        if not _is_valid_xyz(camera_position):
            raise ValueError("camera_position must contain three finite values")
        if not isinstance(mouth_positions, Sequence) or not mouth_positions:
            raise ValueError("at least one visible mouth position is required")
        camera = [float(component) for component in camera_position]
        head_depth = math.hypot(float(self.config.head_offset_m[0]), float(self.config.head_offset_m[1]))
        face_depth = math.hypot(
            float(self.config.face_safety_offset_m[0]),
            float(self.config.face_safety_offset_m[1]),
        )
        torso_depth = math.hypot(
            float(self.config.torso_offset_m[0]),
            float(self.config.torso_offset_m[1]),
        )
        objects: list[CollisionObject] = []
        people: list[dict[str, Any]] = []
        for person_index, raw_mouth in enumerate(mouth_positions):
            if not _is_valid_xyz(raw_mouth):
                raise ValueError(f"mouth_positions[{person_index}] must contain three finite values")
            mouth = [float(component) for component in raw_mouth]
            behind_face = _normalize([mouth[axis] - camera[axis] for axis in range(3)])

            def center(depth: float, vertical: float) -> list[float]:
                result = _add(mouth, _scale(behind_face, depth))
                result[2] += float(vertical)
                return result

            head_id = self._person_object_id(person_index, "head")
            torso_id = self._person_object_id(person_index, "torso")
            face_id = self._person_object_id(person_index, "face_safety")
            head_center = center(head_depth, self.config.head_offset_m[2])
            torso_center = center(torso_depth, self.config.torso_offset_m[2])
            face_center = center(face_depth, self.config.face_safety_offset_m[2])
            objects.extend(
                [
                    self._sphere(head_id, head_center, self.config.head_radius_m),
                    self._box(torso_id, torso_center, self.config.torso_size_m),
                    self._sphere(face_id, face_center, self.config.face_safety_radius_m),
                ]
            )
            people.append(
                {
                    "person_index": person_index,
                    "mouth_position": _xyz(mouth),
                    "behind_face_unit_vector": _xyz(behind_face),
                    "head_center": _xyz(head_center),
                    "torso_center": _xyz(torso_center),
                    "face_safety_center": _xyz(face_center),
                    "object_ids": [head_id, torso_id, face_id],
                }
            )
        if self.config.include_table:
            objects.append(
                self._box(
                    "table_platform_collision",
                    self.config.table_center_m,
                    self.config.table_size_m,
                )
            )
        return objects, people

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

    def apply_people(
        self,
        mouth_positions: Sequence[Sequence[float]],
        *,
        camera_position: Sequence[float],
        verify: bool = True,
    ) -> dict[str, Any]:
        """Replace and verify the complete real-camera multi-person scene."""
        try:
            objects, people = self.build_collision_objects_for_people(
                mouth_positions,
                camera_position=camera_position,
            )
        except ValueError as exc:
            return {"success": False, "reason": str(exc)}
        current = self._world_object_ids()
        if not current.get("success"):
            return current
        expected_ids = [collision.id for collision in objects]
        expected_set = set(expected_ids)
        stale_ids = sorted(
            object_id
            for object_id in current["object_ids"]
            if (
                object_id in MANAGED_OBJECT_IDS
                or object_id.startswith(MULTI_PERSON_OBJECT_PREFIX)
            )
            and object_id not in expected_set
        )
        scene = PlanningScene()
        scene.is_diff = True
        for object_id in stale_ids:
            collision = CollisionObject()
            collision.header.frame_id = self.config.base_frame
            collision.id = object_id
            collision.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(collision)
        scene.world.collision_objects.extend(objects)
        applied = self._apply_scene(scene)
        if not applied.get("success"):
            return applied
        result: dict[str, Any] = {
            "success": True,
            "operation": "apply_people",
            "frame_id": self.config.base_frame,
            "camera_position": _xyz(camera_position),
            "person_count": len(people),
            "people": people,
            "object_ids": expected_ids,
            "removed_stale_object_ids": stale_ids,
            "config": asdict(self.config),
        }
        if verify:
            current_after = self._world_object_ids()
            present = set(current_after.get("object_ids", []))
            missing = sorted(expected_set - present)
            stale_present = sorted(set(stale_ids) & present)
            verification = {
                "success": bool(current_after.get("success")) and not missing and not stale_present,
                "present_object_ids": sorted(expected_set & present),
                "missing_object_ids": missing,
                "stale_object_ids_still_present": stale_present,
            }
            if not current_after.get("success"):
                verification["reason"] = current_after.get("reason")
            result["verification"] = verification
            if not verification["success"]:
                result["success"] = False
                result["reason"] = "multi-person PlanningScene update could not be verified"
        return result

    def remove(self, *, verify: bool = True) -> dict[str, Any]:
        """Remove all collision IDs managed by this module from MoveIt's world scene."""
        # MoveIt can reject a REMOVE diff for an absent object.  Read the
        # scene first so this is idempotent and cleans up the optional table
        # whenever it was actually added by an earlier apply.
        current = self._world_object_ids()
        if not current.get("success"):
            return current
        object_ids = sorted(
            object_id
            for object_id in current["object_ids"]
            if object_id in MANAGED_OBJECT_IDS or object_id.startswith(MULTI_PERSON_OBJECT_PREFIX)
        )
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
