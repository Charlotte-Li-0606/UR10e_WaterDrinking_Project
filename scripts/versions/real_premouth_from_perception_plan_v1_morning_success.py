#!/usr/bin/env python3
"""Guarded real-UR10e pre-mouth probe driven by /detected_mouth_pose.

This is deliberately independent of LLM, OpenClaw, feeding execution, and raw
trajectory actions.  Planning uses MoveIt's /move_action with ``plan_only``.
Execution is restricted to an explicitly validated camera-ray target and uses
MoveIt's ``/execute_trajectory`` action only after a fresh frozen observation
and a successful plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this probe on the physical robot's local ROS graph.  It never sends a
# trajectory goal; the scaled-controller client is inspection-only.
os.environ["UR10E_BACKEND"] = "real"
os.environ.pop("ROS_STATIC_PEERS", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["ROS_LOCALHOST_ONLY"] = "0"

import rclpy  # noqa: E402
import rclpy.time  # noqa: E402
import tf2_ros  # noqa: E402
from controller_manager_msgs.srv import ListControllers  # noqa: E402
from geometry_msgs.msg import Pose, PoseStamped, Vector3Stamped  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup  # noqa: E402
from moveit_msgs.msg import BoundingVolume, Constraints, OrientationConstraint, PositionConstraint  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402


BASE_FRAME = "base_link"
UR_BASE_FRAME = "base"
TOOL_FRAME = "tool0"
CAMERA_OPTICAL_FRAME = "d435i_color_optical_frame"
MOUTH_TOPIC = "/detected_mouth_pose"
MOUTH_NORMAL_TOPIC = "/detected_mouth_normal"
MOVE_ACTION = "/move_action"
EXECUTE_TRAJECTORY_ACTION = "/execute_trajectory"
GROUP_NAME = "ur_manipulator"
SCALED_CONTROLLER = "scaled_joint_trajectory_controller"
SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
EXPECTED_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Geometry is expressed in the real UR tool0/flange frame.
STRAW_TIP_OFFSET_TOOL0_M = (0.110, 0.0, 0.0)

# Pre-mouth policies.  ``camera-ray`` is the real-camera default: it stays on
# the camera side of the detected face rather than assuming a global axis.
PRE_MOUTH_APPROACH_AXIS_BASE_LINK = (1.0, 0.0, 0.0)
DEFAULT_PREMOUTH_POLICY = "camera-ray"
PREMOUTH_POLICIES = ("base-x", "camera-ray")
DEFAULT_SAFE_DISTANCE_M = 0.050
MIN_FACE_CLEARANCE_M = 0.050
# This start-state check only prevents crossing through the mouth plane.  It
# must not demand the final stand-off before a move that is itself increasing
# the separation to the final 5 cm pre-mouth point.
MIN_CURRENT_SIDE_MARGIN_M = 0.005
MAX_NORMAL_ANGULAR_SPREAD_RAD = math.radians(12.0)
MOUNT_CALIBRATION_CONFIG = PROJECT_ROOT / "config/ur10e_real/d435i_mount_calibration.json"

# The nominal UR10e reach is 1.30 m.  Exact reachability and collision checks
# are delegated to MoveIt, but a target beyond this gross physical radius is
# physically unreasonable and is rejected before planning.  In particular, do
# not use the simulated feeding-policy XYZ minimum/maximum here: base_link is
# rotated relative to the physical UR base on this real robot.
MAX_TOOL0_RADIUS_FROM_UR_BASE_M = 1.30
MAX_PLAN_TRANSLATION_M = 0.30
MIN_EXECUTION_SPEED_PERCENT = 5.0
MAX_EXECUTION_SPEED_PERCENT = 15.0
MAX_MOUTH_POSE_AGE_SEC = 1.0
MIN_STABLE_SAMPLES = 3
MAX_POSE_SPREAD_M = 0.025
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_RAD = 0.001
FINAL_ORIENTATION_TOLERANCE_RAD = 0.01
FINAL_STRAW_TARGET_TOLERANCE_M = 0.02
RETURN_START_MATCH_TOLERANCE_M = 0.02
RETURN_START_MATCH_ORIENTATION_TOLERANCE_RAD = 0.02
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PLANNER = "LIN"
ACTION_TIMEOUT_SEC = 60.0
JOINT_STATE_DISCOVERY_TIMEOUT_SEC = 5.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def _norm(vector: list[float] | tuple[float, float, float]) -> float:
    return math.sqrt(sum(float(component) * float(component) for component in vector))


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _add(left: list[float], right: tuple[float, float, float] | list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def _scale(vector: tuple[float, float, float], scalar: float) -> list[float]:
    return [float(component) * float(scalar) for component in vector]


def _dot(first: list[float], second: tuple[float, float, float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _rotate_tool_vector(
    orientation_xyzw: list[float], vector_tool: tuple[float, float, float] | list[float]
) -> list[float]:
    x, y, z, w = (float(component) for component in orientation_xyzw)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if magnitude == 0.0:
        raise RuntimeError("tool0 orientation quaternion has zero length")
    x, y, z, w = (component / magnitude for component in (x, y, z, w))
    vx, vy, vz = vector_tool
    return [
        (1.0 - 2.0 * (y * y + z * z)) * vx + 2.0 * (x * y - z * w) * vy + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx + (1.0 - 2.0 * (x * x + z * z)) * vy + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx + 2.0 * (y * z + x * w) * vy + (1.0 - 2.0 * (x * x + y * y)) * vz,
    ]


def _quaternion_distance_rad(first: list[float], second: list[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _trajectory_summary(trajectory: Any) -> dict[str, Any]:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    points = list(getattr(joint_trajectory, "points", []))
    duration = None
    if points:
        end = points[-1].time_from_start
        duration = float(end.sec) + float(end.nanosec) * 1e-9
    return {
        "joint_names": list(getattr(joint_trajectory, "joint_names", [])),
        "points": len(points),
        "duration_sec": duration,
    }


@dataclass(frozen=True)
class MouthSample:
    position_m: tuple[float, float, float]
    frame_id: str
    received_monotonic: float
    stamp_sec: float


class RealPreMouthFromPerceptionPlan(Node):
    """Read perception, validate it, then plan or guarded-execute in MoveIt."""

    def __init__(self, *, premouth_policy: str = DEFAULT_PREMOUTH_POLICY, safe_distance_m: float = DEFAULT_SAFE_DISTANCE_M) -> None:
        super().__init__("real_premouth_from_perception_plan")
        if premouth_policy not in PREMOUTH_POLICIES:
            raise ValueError(f"unsupported pre-mouth policy {premouth_policy}")
        if not math.isfinite(safe_distance_m) or safe_distance_m <= 0.0:
            raise ValueError("safe_distance_m must be positive and finite")
        self.premouth_policy = premouth_policy
        self.safe_distance_m = safe_distance_m
        self.latest_joint_state: JointState | None = None
        self.latest_speed_scaling: Float64 | None = None
        self.mouth_samples: list[MouthSample] = []
        self.normal_samples: list[MouthSample] = []
        self._validated_trajectory: Any | None = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.create_subscription(Float64, SPEED_SCALING_TOPIC, self._speed_scaling_callback, 10)
        self.create_subscription(PoseStamped, MOUTH_TOPIC, self._mouth_callback, 20)
        self.create_subscription(Vector3Stamped, MOUTH_NORMAL_TOPIC, self._normal_callback, 20)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.execute_trajectory = ActionClient(self, ExecuteTrajectory, EXECUTE_TRAJECTORY_ACTION)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")

    def _joint_state_callback(self, message: JointState) -> None:
        self.latest_joint_state = message

    def _speed_scaling_callback(self, message: Float64) -> None:
        self.latest_speed_scaling = message

    def _mouth_callback(self, message: PoseStamped) -> None:
        frame_id = message.header.frame_id.strip().lstrip("/")
        position = message.pose.position
        values = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in values):
            return
        self.mouth_samples.append(
            MouthSample(
                position_m=values,
                frame_id=frame_id,
                received_monotonic=time.monotonic(),
                stamp_sec=float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9,
            )
        )
        # Retain enough history for an 8-second check while bounding memory.
        if len(self.mouth_samples) > 512:
            del self.mouth_samples[:-256]

    def _normal_callback(self, message: Vector3Stamped) -> None:
        frame_id = message.header.frame_id.strip().lstrip("/")
        values = (float(message.vector.x), float(message.vector.y), float(message.vector.z))
        magnitude = _norm(values)
        if frame_id != BASE_FRAME or not math.isfinite(magnitude) or magnitude < 1e-8:
            return
        normalized = tuple(value / magnitude for value in values)
        self.normal_samples.append(
            MouthSample(
                position_m=normalized,
                frame_id=frame_id,
                received_monotonic=time.monotonic(),
                stamp_sec=float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9,
            )
        )
        if len(self.normal_samples) > 512:
            del self.normal_samples[:-256]

    @staticmethod
    def _mount_calibration() -> dict[str, Any]:
        try:
            configuration = json.loads(MOUNT_CALIBRATION_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"status": "missing", "reason": f"cannot read mount calibration config: {exc}"}
        status = configuration.get("calibration_status")
        if status not in {"provisional", "verified"}:
            return {"status": "invalid", "reason": "calibration_status must be provisional or verified"}
        return {
            "status": status,
            "metrics": configuration.get("calibration_metrics"),
            "config": str(MOUNT_CALIBRATION_CONFIG),
        }

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.10, max(0.0, deadline - time.monotonic())))

    def _wait_for_joint_state(self, timeout_sec: float = JOINT_STATE_DISCOVERY_TIMEOUT_SEC) -> None:
        """Allow discovery of the non-latched joint-state stream at startup."""
        deadline = time.monotonic() + timeout_sec
        while self.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _frame_transform(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0) -> dict[str, Any]:
        # A newly started probe needs to spin while discovering /tf_static.
        # A blocking lookup alone cannot populate this non-threaded listener's
        # buffer, so retry until the bounded timeout instead of falsely
        # declaring the camera mount absent during startup.
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        transform = None
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.0),
                )
                break
            except Exception as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.05)
        if transform is None:
            reason = "unknown TF lookup failure" if last_error is None else f"{last_error.__class__.__name__}: {last_error}"
            return {"available": False, "reason": reason}
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "available": True,
            "parent_frame": target_frame,
            "child_frame": source_frame,
            "position_m": [float(translation.x), float(translation.y), float(translation.z)],
            "orientation_quat_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
        }

    def _tool0_pose(self) -> dict[str, Any]:
        pose = self._frame_transform(BASE_FRAME, TOOL_FRAME)
        if pose.get("available"):
            pose["link_name"] = TOOL_FRAME
        return pose

    def _joint_state_status(self) -> dict[str, Any]:
        if self.latest_joint_state is None:
            return {"received": False, "complete": False}
        positions = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        return {
            "received": True,
            "complete": all(name in positions for name in EXPECTED_JOINTS),
            "received_expected_joint_count": sum(name in positions for name in EXPECTED_JOINTS),
            "expected_joint_count": len(EXPECTED_JOINTS),
        }

    def _controller_status(self) -> dict[str, Any]:
        if not self.controllers.wait_for_service(timeout_sec=0.5):
            return {"available": False, "reason": "/controller_manager/list_controllers is unavailable"}
        future = self.controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None:
            return {"available": False, "reason": "controller manager returned no response"}
        active = {controller.name for controller in response.controller if controller.state == "active"}
        return {
            "available": True,
            "joint_state_broadcaster_active": "joint_state_broadcaster" in active,
            "scaled_joint_trajectory_controller_active": SCALED_CONTROLLER in active,
        }

    def _collect_mouth_samples(self, duration_sec: float) -> dict[str, Any]:
        start = time.monotonic()
        self._spin_for(duration_sec)
        received = [sample for sample in self.mouth_samples if sample.received_monotonic >= start]
        now = time.monotonic()
        valid = [
            sample
            for sample in received
            if sample.frame_id == BASE_FRAME and now - sample.received_monotonic <= MAX_MOUTH_POSE_AGE_SEC
        ]
        if not received:
            return {"available": False, "reason": f"no {MOUTH_TOPIC} messages received in {duration_sec:.1f} seconds"}
        wrong_frame = sorted({sample.frame_id for sample in received if sample.frame_id != BASE_FRAME})
        if wrong_frame:
            return {
                "available": False,
                "reason": f"mouth pose must use {BASE_FRAME}; received {wrong_frame}",
                "received_count": len(received),
            }
        if not valid:
            return {
                "available": False,
                "reason": f"mouth pose is stale (over {MAX_MOUTH_POSE_AGE_SEC:.1f} s old)",
                "received_count": len(received),
            }
        coordinates = list(zip(*(sample.position_m for sample in valid)))
        mean = [sum(values) / len(values) for values in coordinates]
        std = [math.sqrt(sum((value - average) ** 2 for value in values) / len(values)) for values, average in zip(coordinates, mean)]
        max_distance = max(_norm(_subtract(list(sample.position_m), mean)) for sample in valid)
        latest = valid[-1]
        normal_received = [sample for sample in self.normal_samples if sample.received_monotonic >= start]
        normal_valid = [
            sample
            for sample in normal_received
            if sample.frame_id == BASE_FRAME and now - sample.received_monotonic <= MAX_MOUTH_POSE_AGE_SEC
        ]
        normal: dict[str, Any]
        if not normal_valid:
            normal = {"available": False, "reason": f"no recent {MOUTH_NORMAL_TOPIC} messages"}
        else:
            average_normal = [sum(values) / len(values) for values in zip(*(sample.position_m for sample in normal_valid))]
            normal_magnitude = _norm(average_normal)
            if normal_magnitude < 1e-8:
                normal = {"available": False, "reason": "mouth normals cancel instead of agreeing"}
            else:
                average_normal = [value / normal_magnitude for value in average_normal]
                angles = [
                    math.acos(max(-1.0, min(1.0, _dot(list(sample.position_m), tuple(average_normal)))))
                    for sample in normal_valid
                ]
                normal = {
                    "available": True,
                    "frame_id": BASE_FRAME,
                    "sample_count": len(normal_valid),
                    "mean_vector": average_normal,
                    "max_angular_spread_rad": max(angles),
                    "max_angular_spread_deg": math.degrees(max(angles)),
                    "stable": len(normal_valid) >= MIN_STABLE_SAMPLES
                    and max(angles) <= MAX_NORMAL_ANGULAR_SPREAD_RAD,
                    "stability_requirements": {
                        "minimum_samples": MIN_STABLE_SAMPLES,
                        "maximum_angular_spread_deg": math.degrees(MAX_NORMAL_ANGULAR_SPREAD_RAD),
                    },
                }
        return {
            "available": True,
            "frame_id": BASE_FRAME,
            "sample_duration_sec": duration_sec,
            "sample_count": len(valid),
            "mean_position_m": mean,
            "latest_position_m": list(latest.position_m),
            "jitter_stddev_m": std,
            "max_distance_from_mean_m": max_distance,
            "latest_received_age_sec": now - latest.received_monotonic,
            "latest_source_stamp_sec": latest.stamp_sec,
            "stable": len(valid) >= MIN_STABLE_SAMPLES and max_distance <= MAX_POSE_SPREAD_M,
            "stability_requirements": {
                "minimum_samples": MIN_STABLE_SAMPLES,
                "maximum_spread_m": MAX_POSE_SPREAD_M,
                "maximum_age_sec": MAX_MOUTH_POSE_AGE_SEC,
            },
            "surface_normal": normal,
        }

    def snapshot(self, mouth_sample_sec: float) -> dict[str, Any]:
        self._wait_for_joint_state()
        self._spin_for(0.2)
        move_group_available = self.move_group.wait_for_server(timeout_sec=2.0)
        execute_trajectory_available = self.execute_trajectory.wait_for_server(timeout_sec=1.0)
        nodes = [
            f"{namespace.rstrip('/')}/{name}".replace("//", "/")
            for name, namespace in self.get_node_names_and_namespaces()
        ]
        tool0 = self._tool0_pose()
        straw = None
        if tool0.get("available"):
            straw = _add(
                tool0["position_m"],
                _rotate_tool_vector(tool0["orientation_quat_xyzw"], STRAW_TIP_OFFSET_TOOL0_M),
            )
        return {
            "joint_state": self._joint_state_status(),
            "tool0_pose": tool0,
            "ur_base_tf": self._frame_transform(UR_BASE_FRAME, BASE_FRAME),
            "current_straw_tip_pose": None
            if straw is None
            else {
                "frame_id": BASE_FRAME,
                "position_m": straw,
                "orientation_quat_xyzw": tool0["orientation_quat_xyzw"],
            },
            "camera_tf": self._frame_transform(BASE_FRAME, CAMERA_OPTICAL_FRAME),
            "required_nodes": {
                "move_group_exists": "/move_group" in nodes,
                "controller_manager_exists": "/controller_manager" in nodes,
            },
            "move_group_available": move_group_available,
            "execute_trajectory_available": execute_trajectory_available,
            "controllers": self._controller_status(),
            "speed_slider_percent": None
            if self.latest_speed_scaling is None
            else float(self.latest_speed_scaling.data),
            "mouth_pose": self._collect_mouth_samples(mouth_sample_sec),
            "mount_calibration": self._mount_calibration(),
        }

    @staticmethod
    def readiness_failures(snapshot: dict[str, Any], *, require_stable_mouth: bool) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        if not snapshot["ur_base_tf"].get("available"):
            failures.append("TF base -> base_link is unavailable")
        if not snapshot["camera_tf"].get("available"):
            failures.append(f"TF base_link -> {CAMERA_OPTICAL_FRAME} is unavailable")
        nodes = snapshot["required_nodes"]
        if not nodes.get("move_group_exists") or not snapshot.get("move_group_available"):
            failures.append(f"{MOVE_ACTION} is unavailable")
        if not nodes.get("controller_manager_exists"):
            failures.append("controller_manager is unavailable")
        controllers = snapshot["controllers"]
        if not controllers.get("joint_state_broadcaster_active"):
            failures.append("joint_state_broadcaster is not active")
        if not controllers.get("scaled_joint_trajectory_controller_active"):
            failures.append("scaled_joint_trajectory_controller is not active")
        mouth = snapshot["mouth_pose"]
        if not mouth.get("available"):
            failures.append(mouth.get("reason", f"{MOUTH_TOPIC} is unavailable"))
        elif require_stable_mouth and not mouth.get("stable"):
            failures.append(
                "mouth pose is not stable: "
                f"{mouth.get('sample_count', 0)} samples, "
                f"max spread {mouth.get('max_distance_from_mean_m', float('nan')):.4f} m"
            )
        return failures

    @staticmethod
    def _point_in_ur_base(position_in_base_link: list[float], base_tf: dict[str, Any]) -> list[float]:
        """Transform a base_link point into physical UR ``base`` coordinates."""
        return _add(
            list(base_tf["position_m"]),
            _rotate_tool_vector(list(base_tf["orientation_quat_xyzw"]), position_in_base_link),
        )

    @staticmethod
    def _orientation_constraint(target_pose: Pose) -> OrientationConstraint:
        constraint = OrientationConstraint()
        constraint.header.frame_id = BASE_FRAME
        constraint.link_name = TOOL_FRAME
        constraint.orientation = target_pose.orientation
        constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        constraint.weight = 1.0
        return constraint

    def _goal_for_target(self, target: dict[str, Any]) -> MoveGroup.Goal:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target["position_m"]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = target["orientation_quat_xyzw"]
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE_M]
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose)
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = BASE_FRAME
        position_constraint.link_name = TOOL_FRAME
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0
        constraints = Constraints()
        constraints.name = "real_premouth_from_perception_plan_only"
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(self._orientation_constraint(pose))

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.pipeline_id = PILZ_PIPELINE
        goal.request.planner_id = PILZ_PLANNER
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.05
        goal.request.max_acceleration_scaling_factor = 0.05
        if self.latest_joint_state is not None:
            goal.request.start_state.joint_state = self.latest_joint_state
            goal.request.start_state.is_diff = False
        goal.request.goal_constraints.append(constraints)
        goal.request.path_constraints.orientation_constraints.append(self._orientation_constraint(pose))
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _run_plan(self, target: dict[str, Any]) -> dict[str, Any]:
        self._validated_trajectory = None
        future = self.move_group.send_goal_async(self._goal_for_target(target))
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {"success": False, "stage": "move_group_goal", "reason": "MoveGroup rejected plan-only goal"}
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=ACTION_TIMEOUT_SEC)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "move_group_timeout",
                "reason": f"MoveGroup did not return a plan within {ACTION_TIMEOUT_SEC:.0f} seconds; cancel requested",
            }
        result = wrapped_result.result
        success = int(result.error_code.val) == 1
        if success:
            self._validated_trajectory = result.planned_trajectory
        return {
            "success": success,
            "stage": "move_group_plan_only",
            "result_status": int(wrapped_result.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "execution_sent": False,
        }

    def _return_snapshot(self) -> dict[str, Any]:
        """Read only the robot state required for a recorded-pose return."""
        self._wait_for_joint_state()
        self._spin_for(0.2)
        nodes = [
            f"{namespace.rstrip('/')}/{name}".replace("//", "/")
            for name, namespace in self.get_node_names_and_namespaces()
        ]
        return {
            "joint_state": self._joint_state_status(),
            "tool0_pose": self._tool0_pose(),
            "ur_base_tf": self._frame_transform(UR_BASE_FRAME, BASE_FRAME),
            "required_nodes": {
                "move_group_exists": "/move_group" in nodes,
                "controller_manager_exists": "/controller_manager" in nodes,
            },
            "move_group_available": self.move_group.wait_for_server(timeout_sec=2.0),
            "execute_trajectory_available": self.execute_trajectory.wait_for_server(timeout_sec=1.0),
            "controllers": self._controller_status(),
            "speed_slider_percent": None
            if self.latest_speed_scaling is None
            else float(self.latest_speed_scaling.data),
        }

    @staticmethod
    def _valid_pose(pose: Any) -> bool:
        if not isinstance(pose, dict):
            return False
        position = pose.get("position_m")
        orientation = pose.get("orientation_quat_xyzw")
        return (
            isinstance(position, list)
            and len(position) == 3
            and isinstance(orientation, list)
            and len(orientation) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in position + orientation)
            and _norm([float(value) for value in orientation]) > 0.0
        )

    @staticmethod
    def _return_readiness_failures(snapshot: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        if not snapshot["ur_base_tf"].get("available"):
            failures.append("TF base -> base_link is unavailable")
        nodes = snapshot["required_nodes"]
        if not nodes.get("move_group_exists") or not snapshot.get("move_group_available"):
            failures.append(f"{MOVE_ACTION} is unavailable")
        if not snapshot.get("execute_trajectory_available"):
            failures.append(f"{EXECUTE_TRAJECTORY_ACTION} is unavailable")
        if not nodes.get("controller_manager_exists"):
            failures.append("controller_manager is unavailable")
        controllers = snapshot["controllers"]
        if not controllers.get("joint_state_broadcaster_active"):
            failures.append("joint_state_broadcaster is not active")
        if not controllers.get("scaled_joint_trajectory_controller_active"):
            failures.append("scaled_joint_trajectory_controller is not active")
        speed = snapshot.get("speed_slider_percent")
        if speed is None or not MIN_EXECUTION_SPEED_PERCENT <= float(speed) <= MAX_EXECUTION_SPEED_PERCENT:
            failures.append("speed slider is unavailable or outside the required 5%–15% range")
        return failures

    def return_to_recorded_start(self, report_path: Path | None, *, confirm_real_motion: bool) -> tuple[int, dict[str, Any]]:
        """Return only to the recorded start of one successful prior probe run."""
        response: dict[str, Any] = {
            "mode": "return",
            "execution_sent": False,
            "execution_disabled": False,
            "automatic_retreat_sent": False,
        }
        if report_path is None:
            response.update({"success": False, "stage": "return_input", "reason": "--return-report is required"})
            return 2, response
        try:
            recorded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            response.update({"success": False, "stage": "return_input", "reason": f"could not read return report: {exc}"})
            return 2, response
        start_pose = recorded.get("checks", {}).get("tool0_pose")
        end_pose = recorded.get("actual", {}).get("final_tool0_pose")
        execution_ok = bool(recorded.get("success")) and bool(recorded.get("execution_result", {}).get("success"))
        if not execution_ok or not self._valid_pose(start_pose) or not self._valid_pose(end_pose):
            response.update(
                {
                    "success": False,
                    "stage": "return_input",
                    "reason": "return report must be a successful executed probe with valid start and end tool0 poses",
                    "return_report": str(report_path),
                }
            )
            return 2, response

        snapshot = self._return_snapshot()
        failures = self._return_readiness_failures(snapshot)
        response.update({"return_report": str(report_path), "checks": snapshot, "recorded_start_tool0_pose": start_pose, "recorded_end_tool0_pose": end_pose})
        if failures:
            response.update({"success": False, "stage": "return_readiness", "failures": failures})
            return 2, response

        current = snapshot["tool0_pose"]
        start_match_error = _norm(_subtract(current["position_m"], end_pose["position_m"]))
        start_match_orientation_error = _quaternion_distance_rad(
            current["orientation_quat_xyzw"], end_pose["orientation_quat_xyzw"]
        )
        response.update(
            {
                "current_tool0_pose": current,
                "recorded_end_position_error_m": start_match_error,
                "recorded_end_orientation_error_rad": start_match_orientation_error,
            }
        )
        if (
            start_match_error > RETURN_START_MATCH_TOLERANCE_M
            or start_match_orientation_error > RETURN_START_MATCH_ORIENTATION_TOLERANCE_RAD
        ):
            response.update(
                {
                    "success": False,
                    "stage": "return_start_match_guard",
                    "reason": "robot is not at the recorded successful-run end pose; refusing an ambiguous return",
                    "position_tolerance_m": RETURN_START_MATCH_TOLERANCE_M,
                    "orientation_tolerance_rad": RETURN_START_MATCH_ORIENTATION_TOLERANCE_RAD,
                }
            )
            return 2, response

        target = {
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            "position_m": [float(value) for value in start_pose["position_m"]],
            "orientation_quat_xyzw": [float(value) for value in start_pose["orientation_quat_xyzw"]],
        }
        target_in_base = self._point_in_ur_base(target["position_m"], snapshot["ur_base_tf"])
        target_radius = _norm(target_in_base)
        if target_radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            response.update(
                {
                    "success": False,
                    "stage": "return_reach_guard",
                    "reason": "recorded return pose exceeds the UR10e nominal reach envelope",
                    "target_tool0_pose": target,
                    "target_tool0_radius_from_ur_base_m": target_radius,
                }
            )
            return 2, response

        plan_result = self._run_plan(target)
        planned_translation = _subtract(target["position_m"], current["position_m"])
        response.update(
            {
                "target_tool0_pose": target,
                "target_tool0_radius_from_ur_base_m": target_radius,
                "planned_tool0_translation_m": planned_translation,
                "planned_tool0_translation_norm_m": _norm(planned_translation),
                "orientation_difference_rad": _quaternion_distance_rad(
                    current["orientation_quat_xyzw"], target["orientation_quat_xyzw"]
                ),
                "plan_result": plan_result,
                "planned_trajectory_duration_sec": plan_result.get("planned_trajectory", {}).get("duration_sec"),
            }
        )
        guards: list[str] = []
        if not confirm_real_motion:
            guards.append("--confirm-real-motion is required")
        if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
            guards.append("UR10E_ALLOW_REAL_EXECUTION=1 is required")
        if not plan_result.get("success") or self._validated_trajectory is None:
            guards.append("MoveIt did not produce a validated return trajectory")
        if response["orientation_difference_rad"] > FINAL_ORIENTATION_TOLERANCE_RAD:
            guards.append("return target changes the recorded stable tool orientation")
        if guards:
            response.update({"success": False, "stage": "return_execution_guard", "failures": guards, "execution_attempted": False})
            return 2, response

        self._spin_for(0.2)
        latest = self._tool0_pose()
        pre_execution_drift = _norm(_subtract(latest["position_m"], current["position_m"])) if latest.get("available") else float("inf")
        controller_state = self._controller_status()
        speed = None if self.latest_speed_scaling is None else float(self.latest_speed_scaling.data)
        if (
            not latest.get("available")
            or pre_execution_drift > 0.01
            or not controller_state.get("scaled_joint_trajectory_controller_active")
            or speed is None
            or not MIN_EXECUTION_SPEED_PERCENT <= speed <= MAX_EXECUTION_SPEED_PERCENT
        ):
            response.update(
                {
                    "success": False,
                    "stage": "return_pre_execution_guard",
                    "reason": "robot/controller/speed state changed after the return plan",
                    "pre_execution_tool0_drift_m": pre_execution_drift,
                    "pre_execution_speed_slider_percent": speed,
                    "execution_attempted": False,
                }
            )
            return 2, response

        execution_result = self._execute_validated_trajectory()
        final = self._tool0_pose()
        if final.get("available"):
            final_position_error = _norm(_subtract(final["position_m"], target["position_m"]))
            final_orientation_error = _quaternion_distance_rad(
                final["orientation_quat_xyzw"], target["orientation_quat_xyzw"]
            )
            actual = {
                "final_tool0_pose": final,
                "final_tool0_position_error_m": final_position_error,
                "final_tool0_orientation_error_rad": final_orientation_error,
                "within_position_tolerance": final_position_error <= FINAL_STRAW_TARGET_TOLERANCE_M,
                "orientation_stable": final_orientation_error <= FINAL_ORIENTATION_TOLERANCE_RAD,
            }
        else:
            actual = {"available": False, "reason": "final TF base_link -> tool0 is unavailable"}
        success = bool(execution_result.get("success") and actual.get("within_position_tolerance") and actual.get("orientation_stable"))
        response.update(
            {
                "success": success,
                "stage": "return" if success else "return_verification",
                "execution_result": execution_result,
                "actual": actual,
                "execution_attempted": bool(execution_result.get("execution_attempted")),
                "execution_sent": bool(execution_result.get("execution_attempted")),
            }
        )
        return (0 if success else 2), response

    def _pre_mouth_candidates(self, mouth: list[float], camera: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, Any] | None]:
        """Return both diagnostic candidates and the selected policy target."""
        base_x = _add(mouth, _scale(PRE_MOUTH_APPROACH_AXIS_BASE_LINK, self.safe_distance_m))
        camera_position = camera.get("position_m")
        if not isinstance(camera_position, list) or len(camera_position) != 3:
            return {"base-x": base_x}, {"reason": "camera optical-frame position is unavailable"}
        camera_to_mouth = _subtract(mouth, camera_position)
        distance = _norm(camera_to_mouth)
        if distance < 1e-6:
            return {"base-x": base_x}, {"reason": "camera and detected mouth positions are coincident"}
        ray = [component / distance for component in camera_to_mouth]
        camera_ray = _subtract(mouth, [self.safe_distance_m * component for component in ray])
        camera_ray_alternative = _add(mouth, [self.safe_distance_m * component for component in ray])
        return (
            {"base-x": base_x, "camera-ray": camera_ray, "camera-ray-alternative": camera_ray_alternative},
            {"camera_to_mouth_vector_m": camera_to_mouth, "camera_to_mouth_unit_vector": ray, "camera_to_mouth_distance_m": distance},
        )

    def plan(self) -> tuple[int, dict[str, Any]]:
        snapshot = self.snapshot(mouth_sample_sec=5.0)
        failures = self.readiness_failures(snapshot, require_stable_mouth=True)
        if failures:
            return 2, {"success": False, "mode": "plan", "stage": "readiness", "failures": failures, "checks": snapshot, "execution_sent": False}

        mouth = list(snapshot["mouth_pose"]["mean_position_m"])
        straw = list(snapshot["current_straw_tip_pose"]["position_m"])
        current_tool0 = snapshot["tool0_pose"]
        candidates, camera_ray_details = self._pre_mouth_candidates(mouth, snapshot["camera_tf"])
        if camera_ray_details is not None and "reason" in camera_ray_details:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "camera_ray_input",
                "reason": camera_ray_details["reason"],
                "checks": snapshot,
                "execution_sent": False,
            }
        if self.premouth_policy not in candidates:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "pre_mouth_policy",
                "reason": f"candidate for policy {self.premouth_policy} is unavailable",
                "checks": snapshot,
                "execution_sent": False,
            }
        pre_mouth = candidates[self.premouth_policy]

        orientation = list(current_tool0["orientation_quat_xyzw"])
        straw_world_offset = _rotate_tool_vector(orientation, STRAW_TIP_OFFSET_TOOL0_M)
        target_tool0_position = _subtract(pre_mouth, straw_world_offset)
        displacement = _subtract(target_tool0_position, list(current_tool0["position_m"]))
        translation_norm = _norm(displacement)
        target_tool0_in_ur_base = self._point_in_ur_base(target_tool0_position, snapshot["ur_base_tf"])
        target_tool0_radius = _norm(target_tool0_in_ur_base)
        if target_tool0_radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "gross_reach_guard",
                "reason": "target tool0 pose exceeds the UR10e nominal 1.30 m reach from physical base",
                "detected_mouth_pose": mouth,
                "pre_mouth_pose": pre_mouth,
                "target_tool0_pose": target_tool0_position,
                "target_tool0_position_in_ur_base_m": target_tool0_in_ur_base,
                "target_tool0_radius_from_ur_base_m": target_tool0_radius,
                "maximum_radius_m": MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
                "checks": snapshot,
                "execution_sent": False,
            }

        if translation_norm > MAX_PLAN_TRANSLATION_M:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "planned_translation_guard",
                "reason": "planned tool0 translation exceeds the 0.30 m diagnostic plan-only limit",
                "premouth_policy": self.premouth_policy,
                "safe_distance_m": self.safe_distance_m,
                "detected_mouth_pose": mouth,
                "pre_mouth_candidates": candidates,
                "camera_ray_details": camera_ray_details,
                "target_tool0_pose": target_tool0_position,
                "planned_tool0_translation_m": displacement,
                "planned_tool0_translation_norm_m": translation_norm,
                "maximum_translation_m": MAX_PLAN_TRANSLATION_M,
                "checks": snapshot,
                "execution_sent": False,
            }

        target = {
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            "position_m": target_tool0_position,
            "orientation_quat_xyzw": orientation,
        }
        plan_result = self._run_plan(target)
        response = {
            "success": bool(plan_result.get("success")),
            "mode": "plan",
            "premouth_policy": self.premouth_policy,
            "safe_distance_m": self.safe_distance_m,
            "detected_mouth_pose": {"frame_id": BASE_FRAME, "position_m": mouth},
            "camera_position": snapshot["camera_tf"]["position_m"],
            "pre_mouth_candidates": {"frame_id": BASE_FRAME, "positions_m": candidates},
            "camera_ray_details": camera_ray_details,
            "pre_mouth_pose": {"frame_id": BASE_FRAME, "position_m": pre_mouth},
            "pre_mouth_offset_base_link_m": _subtract(pre_mouth, mouth),
            "current_straw_tip_pose": snapshot["current_straw_tip_pose"],
            "target_tool0_pose": target,
            "planned_tool0_translation_m": displacement,
            "planned_tool0_translation_norm_m": translation_norm,
            "maximum_plan_translation_m": MAX_PLAN_TRANSLATION_M,
            "target_tool0_position_in_ur_base_m": target_tool0_in_ur_base,
            "target_tool0_radius_from_ur_base_m": target_tool0_radius,
            "maximum_radius_m": MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
            "estimated_final_straw_tip_to_pre_mouth_m": 0.0,
            "final_straw_tip_to_detected_mouth_standoff_m": self.safe_distance_m,
            "orientation_difference_rad": 0.0,
            "orientation_preserved": True,
            "checks": snapshot,
            "plan_result": plan_result,
            "planned_trajectory_duration_sec": plan_result.get("planned_trajectory", {}).get("duration_sec"),
            "execution_sent": False,
            "execution_disabled": True,
            "mount_calibration": snapshot["mount_calibration"],
        }
        return (0 if response["success"] else 2), response

    def _execution_guards(self, prepared: dict[str, Any], *, confirm_real_motion: bool) -> list[str]:
        """Return every reason the already-planned trajectory cannot move."""
        guards: list[str] = []
        if not confirm_real_motion:
            guards.append("--confirm-real-motion is required")
        if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
            guards.append("UR10E_ALLOW_REAL_EXECUTION=1 is required")
        if not prepared.get("success"):
            guards.append("the frozen-pose MoveIt plan did not succeed")
        if self._validated_trajectory is None:
            guards.append("the validated MoveIt trajectory is unavailable")
        checks = prepared.get("checks", {})
        if not checks.get("execute_trajectory_available"):
            guards.append(f"{EXECUTE_TRAJECTORY_ACTION} action server is unavailable")
        target_radius = prepared.get("target_tool0_radius_from_ur_base_m")
        if not isinstance(target_radius, (int, float)) or float(target_radius) > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            guards.append(
                "target tool0 pose is outside the UR10e nominal reach envelope "
                f"({float(target_radius or 0.0):.4f} m > {MAX_TOOL0_RADIUS_FROM_UR_BASE_M:.2f} m)"
            )
        orientation_difference = prepared.get("orientation_difference_rad")
        if not isinstance(orientation_difference, (int, float)) or float(orientation_difference) > FINAL_ORIENTATION_TOLERANCE_RAD:
            guards.append("target tool orientation differs unexpectedly from the current tool orientation")
        standoff = prepared.get("final_straw_tip_to_detected_mouth_standoff_m")
        if not isinstance(standoff, (int, float)) or float(standoff) < MIN_FACE_CLEARANCE_M:
            guards.append("pre-mouth straw target is too close to the detected face")
        speed_percent = checks.get("speed_slider_percent")
        if speed_percent is None:
            guards.append("speed slider state is unavailable")
        elif not MIN_EXECUTION_SPEED_PERCENT <= float(speed_percent) <= MAX_EXECUTION_SPEED_PERCENT:
            guards.append(
                f"speed slider is {float(speed_percent):.1f}%, outside the required "
                f"{MIN_EXECUTION_SPEED_PERCENT:.0f}%–{MAX_EXECUTION_SPEED_PERCENT:.0f}% range"
            )
        return guards

    def _execute_validated_trajectory(self) -> dict[str, Any]:
        """Execute only the RobotTrajectory returned by the immediately prior plan."""
        if self._validated_trajectory is None:
            return {"success": False, "stage": "validated_trajectory", "reason": "no validated trajectory is cached"}
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._validated_trajectory
        future = self.execute_trajectory.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "execute_trajectory_goal",
                "reason": "MoveIt rejected the validated trajectory",
                "execution_attempted": False,
            }
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=ACTION_TIMEOUT_SEC)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "execute_trajectory_timeout",
                "reason": f"MoveIt did not finish within {ACTION_TIMEOUT_SEC:.0f} seconds; cancel requested",
                "execution_attempted": True,
            }
        result = wrapped_result.result
        return {
            "success": int(result.error_code.val) == 1,
            "stage": "execute_trajectory",
            "result_status": int(wrapped_result.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "execution_attempted": True,
            "controller_goal_type": "MoveIt ExecuteTrajectory (validated MoveIt plan; no raw FollowJointTrajectory goal)",
        }

    def execute(
        self,
        *,
        confirm_real_motion: bool,
        allow_validated_camera_ray_execute: bool,
    ) -> tuple[int, dict[str, Any]]:
        if self.premouth_policy != "camera-ray" or not allow_validated_camera_ray_execute:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "camera_ray_execution_gate",
                "reason": (
                    "execution is limited to a visually validated camera-ray target; "
                    "use --premouth-policy camera-ray --allow-validated-camera-ray-execute"
                ),
                "premouth_policy": self.premouth_policy,
                "safe_distance_m": self.safe_distance_m,
                "execution_attempted": False,
                "execution_sent": False,
                "execution_disabled": True,
            }

        # plan() performs all checks, collects a fresh five-second stable
        # observation, freezes its mean mouth pose, and computes this one
        # immutable pre-mouth target before any execution is considered.
        _, prepared = self.plan()
        prepared["mode"] = "execute"
        prepared["frozen_detected_mouth_pose"] = prepared.get("detected_mouth_pose")
        prepared["execution_disabled"] = False
        prepared["execution_sent"] = False
        prepared["frozen_target_policy"] = "The five-second stable mouth-pose mean is frozen before planning; no perception updates are consumed during motion."

        if not prepared.get("success"):
            prepared.update(
                {
                    "success": False,
                    "stage": "pre_execution_plan",
                    "reason": prepared.get("reason", "the fresh MoveIt plan failed"),
                    "execution_attempted": False,
                    "execution_sent": False,
                }
            )
            return 2, prepared

        guards = self._execution_guards(prepared, confirm_real_motion=confirm_real_motion)
        if guards:
            prepared.update(
                {
                    "success": False,
                    "stage": "execution_safety_guard",
                    "failures": guards,
                    "execution_attempted": False,
                    "execution_sent": False,
                }
            )
            return 2, prepared

        # Guard against a manual movement or controller/speed change between
        # the plan result and execution without recomputing the frozen target.
        self._spin_for(0.2)
        latest_tool0 = self._tool0_pose()
        start_tool0 = prepared.get("checks", {}).get("tool0_pose", {})
        if not latest_tool0.get("available") or not start_tool0.get("available"):
            prepared.update(
                {
                    "success": False,
                    "stage": "pre_execution_tf_guard",
                    "reason": "base_link -> tool0 TF is unavailable immediately before execution",
                    "execution_attempted": False,
                }
            )
            return 2, prepared
        start_drift = _norm(_subtract(latest_tool0["position_m"], start_tool0["position_m"]))
        controller_state = self._controller_status()
        speed_percent = None if self.latest_speed_scaling is None else float(self.latest_speed_scaling.data)
        late_guards: list[str] = []
        if start_drift > 0.01:
            late_guards.append(f"tool0 moved {start_drift:.4f} m after planning")
        if not controller_state.get("scaled_joint_trajectory_controller_active"):
            late_guards.append("scaled_joint_trajectory_controller is no longer active")
        if speed_percent is None or not MIN_EXECUTION_SPEED_PERCENT <= speed_percent <= MAX_EXECUTION_SPEED_PERCENT:
            late_guards.append("speed slider changed outside the required 5%–15% range")
        if late_guards:
            prepared.update(
                {
                    "success": False,
                    "stage": "pre_execution_state_guard",
                    "failures": late_guards,
                    "pre_execution_tool0_pose": latest_tool0,
                    "pre_execution_tool0_drift_m": start_drift,
                    "pre_execution_speed_slider_percent": speed_percent,
                    "execution_attempted": False,
                }
            )
            return 2, prepared

        execution_result = self._execute_validated_trajectory()
        final_tool0 = self._tool0_pose()
        actual: dict[str, Any]
        if final_tool0.get("available"):
            final_straw = _add(
                final_tool0["position_m"],
                _rotate_tool_vector(final_tool0["orientation_quat_xyzw"], STRAW_TIP_OFFSET_TOOL0_M),
            )
            target_pre_mouth = prepared["pre_mouth_pose"]["position_m"]
            final_error = _norm(_subtract(final_straw, target_pre_mouth))
            orientation_error = _quaternion_distance_rad(
                final_tool0["orientation_quat_xyzw"], start_tool0["orientation_quat_xyzw"]
            )
            actual = {
                "final_tool0_pose": final_tool0,
                "final_straw_tip_pose": {"frame_id": BASE_FRAME, "position_m": final_straw},
                "final_straw_tip_to_pre_mouth_error_m": final_error,
                "actual_tool0_displacement_m": _subtract(final_tool0["position_m"], start_tool0["position_m"]),
                "orientation_difference_from_start_rad": orientation_error,
                "straw_tip_within_target_tolerance": final_error <= FINAL_STRAW_TARGET_TOLERANCE_M,
                "orientation_stable": orientation_error <= FINAL_ORIENTATION_TOLERANCE_RAD,
            }
        else:
            actual = {"available": False, "reason": "final TF base_link -> tool0 is unavailable"}

        success = bool(
            execution_result.get("success")
            and actual.get("straw_tip_within_target_tolerance")
            and actual.get("orientation_stable")
        )
        prepared.update(
            {
                "success": success,
                "stage": "execute" if success else "execution_verification",
                "execution_result": execution_result,
                "actual": actual,
                "execution_attempted": bool(execution_result.get("execution_attempted")),
                "execution_sent": bool(execution_result.get("execution_attempted")),
                "automatic_retreat_sent": False,
            }
        )
        if not success:
            prepared["reason"] = "MoveIt execution failed, the tool orientation changed, or the straw missed the pre-mouth target"
        return (0 if success else 2), prepared


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "plan", "execute"), default="check")
    parser.add_argument(
        "--premouth-policy",
        choices=PREMOUTH_POLICIES,
        default=DEFAULT_PREMOUTH_POLICY,
        help="Pre-mouth target policy; camera-ray is the default for the real D435i.",
    )
    parser.add_argument(
        "--safe-distance",
        type=float,
        default=DEFAULT_SAFE_DISTANCE_M,
        help="Pre-mouth stand-off distance in metres (default: 0.05).",
    )
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required together with --mode execute and UR10E_ALLOW_REAL_EXECUTION=1.",
    )
    parser.add_argument(
        "--allow-validated-camera-ray-execute",
        action="store_true",
        help="Additional gate for one guarded camera-ray pre-mouth execution after RViz validation.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Mouth-pose collection duration for check mode (default: 1.0; use 8 for a stability report).",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help=(
            "Optional JSON report path for the diagnostic check or plan-only result."
        ),
    )
    args = parser.parse_args()
    if args.sample_seconds <= 0.0 or not math.isfinite(args.sample_seconds):
        parser.error("--sample-seconds must be a positive finite number")
    if args.safe_distance <= 0.0 or not math.isfinite(args.safe_distance):
        parser.error("--safe-distance must be a positive finite number")
    return args


def main() -> int:
    args = _parse_args()
    rclpy.init()
    node = RealPreMouthFromPerceptionPlan(
        premouth_policy=args.premouth_policy,
        safe_distance_m=args.safe_distance,
    )
    try:
        if args.mode == "check":
            snapshot = node.snapshot(mouth_sample_sec=args.sample_seconds)
            failures = node.readiness_failures(snapshot, require_stable_mouth=False)
            response = {
                "success": not failures,
                "mode": "check",
                "premouth_policy": args.premouth_policy,
                "safe_distance_m": args.safe_distance,
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }
            exit_code = 0 if not failures else 2
        elif args.mode == "plan":
            exit_code, response = node.plan()
        else:
            exit_code, response = node.execute(
                confirm_real_motion=args.confirm_real_motion,
                allow_validated_camera_ray_execute=args.allow_validated_camera_ray_execute,
            )
        report_text = json.dumps(_jsonable(response), indent=2, sort_keys=True)
        if args.report_file is not None:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(report_text + "\n", encoding="utf-8")
        print(report_text, flush=True)
        return exit_code
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
