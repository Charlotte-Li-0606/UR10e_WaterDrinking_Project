#!/usr/bin/env python3
"""Guarded real-UR10e pre-mouth probe driven by /detected_mouth_pose.

This is deliberately independent of LLM, OpenClaw, feeding execution, and raw
trajectory actions.  Planning uses MoveIt's /move_action with ``plan_only``.
Execution is restricted to an explicitly validated camera-ray or feeding-vector
target and uses MoveIt's ``/execute_trajectory`` action only after a fresh
frozen observation and a successful plan.
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

# Explicitly identify the real-robot backend.  Do not inherit the obsolete
# static DDS peer left by an earlier network setup: the real driver, MoveIt,
# and camera are all on this host/subnet and use normal subnet discovery.
# This changes only this short-lived probe process, never the running driver.
os.environ["UR10E_BACKEND"] = "real"
os.environ.pop("ROS_STATIC_PEERS", None)
os.environ.pop("ROS_LOCALHOST_ONLY", None)

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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from std_msgs.msg import Bool, Float64, String  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402


BASE_FRAME = "base_link"
UR_BASE_FRAME = "base"
TOOL_FRAME = "tool0"
CAMERA_OPTICAL_FRAME = "d435i_color_optical_frame"
CAMERA_LINK_FRAME = "d435i_link"
MOUTH_TOPIC = "/detected_mouth_pose"
MOUTH_NORMAL_TOPIC = "/detected_mouth_normal"
MOUTH_STATUS_TOPIC = "/mouth_detection/status"
MOVE_ACTION = "/move_action"
EXECUTE_TRAJECTORY_ACTION = "/execute_trajectory"
GROUP_NAME = "ur_manipulator"
SCALED_CONTROLLER = "scaled_joint_trajectory_controller"
SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
ROBOT_PROGRAM_RUNNING_TOPIC = "/io_and_status_controller/robot_program_running"
SAFETY_MODE_TOPIC = "/io_and_status_controller/safety_mode"
ROBOT_MODE_TOPIC = "/io_and_status_controller/robot_mode"
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

# Pre-mouth policies. All legacy policies remain available. ``tcp-forward`` is
# the recommended real policy after the no-motion frame diagnostic: retreat
# from the mouth opposite current tool0/TCP +X expressed in base_link.
PRE_MOUTH_APPROACH_AXIS_BASE_LINK = (1.0, 0.0, 0.0)
DEFAULT_PREMOUTH_POLICY = "camera-ray"
PREMOUTH_POLICIES = ("tcp-forward", "base-x", "camera-ray", "feeding-vector")
DEFAULT_SAFE_DISTANCE_M = 0.050
MIN_SAFE_DISTANCE_M = 0.030
MAX_SAFE_DISTANCE_M = 0.080
DEFAULT_FEEDING_VECTOR = (0.0, -1.0, 0.0)
FEEDING_VECTOR_SIGNS = ("plus", "minus")
MAX_ABS_FEEDING_VECTOR_Z = 0.30
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
MAX_PLAN_TRANSLATION_M = MAX_TOOL0_RADIUS_FROM_UR_BASE_M
MIN_EXECUTION_SPEED_PERCENT = 5.0
MAX_EXECUTION_SPEED_PERCENT = 60.0
DEFAULT_MOUTH_SAMPLE_SECONDS = 2.0
MIN_MOUTH_SAMPLE_SECONDS = 0.5
MAX_MOUTH_SAMPLE_SECONDS = 8.0
DEFAULT_TRAJECTORY_VELOCITY_SCALING = 0.10
DEFAULT_TRAJECTORY_ACCELERATION_SCALING = 0.10
MIN_TRAJECTORY_SCALING = 0.01
MAX_TRAJECTORY_SCALING = 0.20
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
# At low pendant settings, the scaled controller can take much
# longer than the nominal MoveIt trajectory duration.  Keep the client alive
# long enough to receive the controller's real terminal result.
ACTION_TIMEOUT_SEC = 180.0
JOINT_STATE_DISCOVERY_TIMEOUT_SEC = 5.0
TF_DISCOVERY_TIMEOUT_SEC = 8.0
MOVE_GROUP_DISCOVERY_TIMEOUT_SEC = 8.0
MAX_CAMERA_MOUNT_TRANSLATION_ERROR_M = 0.001
MAX_CAMERA_MOUNT_ROTATION_ERROR_RAD = math.radians(0.5)


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


def _normalize_feeding_vector(vector: tuple[float, float, float]) -> list[float]:
    """Return a finite unit base_link feeding direction or fail closed."""
    if not all(math.isfinite(float(component)) for component in vector):
        raise ValueError("feeding vector components must be finite")
    magnitude = _norm(vector)
    if magnitude < 1e-6:
        raise ValueError("feeding vector norm is too small to define an approach direction")
    return [float(component) / magnitude for component in vector]


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


def _tool_x_axis_base(orientation_xyzw: list[float]) -> list[float]:
    """Return current tool0/TCP +X as a finite unit vector in base_link."""
    axis = _rotate_tool_vector(orientation_xyzw, (1.0, 0.0, 0.0))
    magnitude = _norm(axis)
    if not math.isfinite(magnitude) or magnitude < 1e-6:
        raise RuntimeError("tool0/TCP +X direction in base_link is invalid")
    return [component / magnitude for component in axis]


def _quaternion_distance_rad(first: list[float], second: list[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _quaternion_from_rpy(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


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

    def __init__(
        self,
        *,
        premouth_policy: str = DEFAULT_PREMOUTH_POLICY,
        safe_distance_m: float = DEFAULT_SAFE_DISTANCE_M,
        maximum_plan_translation_m: float = MAX_PLAN_TRANSLATION_M,
        feeding_vector: tuple[float, float, float] = DEFAULT_FEEDING_VECTOR,
        feeding_vector_sign: str = "plus",
        mouth_sample_seconds: float = DEFAULT_MOUTH_SAMPLE_SECONDS,
        trajectory_velocity_scaling: float = DEFAULT_TRAJECTORY_VELOCITY_SCALING,
        trajectory_acceleration_scaling: float = DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    ) -> None:
        super().__init__("real_premouth_from_perception_plan")
        if premouth_policy not in PREMOUTH_POLICIES:
            raise ValueError(f"unsupported pre-mouth policy {premouth_policy}")
        if not math.isfinite(safe_distance_m) or not MIN_SAFE_DISTANCE_M <= safe_distance_m <= MAX_SAFE_DISTANCE_M:
            raise ValueError(
                f"safe_distance_m must be finite and within {MIN_SAFE_DISTANCE_M:.2f}–{MAX_SAFE_DISTANCE_M:.2f} m"
            )
        if not math.isfinite(maximum_plan_translation_m) or not 0.0 < maximum_plan_translation_m <= MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            raise ValueError(
                "maximum_plan_translation_m must be positive, finite, and no larger than "
                f"the {MAX_TOOL0_RADIUS_FROM_UR_BASE_M:.2f} m gross reach bound"
            )
        if feeding_vector_sign not in FEEDING_VECTOR_SIGNS:
            raise ValueError(f"feeding_vector_sign must be one of {FEEDING_VECTOR_SIGNS}")
        if not math.isfinite(mouth_sample_seconds) or not MIN_MOUTH_SAMPLE_SECONDS <= mouth_sample_seconds <= MAX_MOUTH_SAMPLE_SECONDS:
            raise ValueError(
                f"mouth_sample_seconds must be within {MIN_MOUTH_SAMPLE_SECONDS:.1f}–{MAX_MOUTH_SAMPLE_SECONDS:.1f} seconds"
            )
        for name, value in (
            ("trajectory_velocity_scaling", trajectory_velocity_scaling),
            ("trajectory_acceleration_scaling", trajectory_acceleration_scaling),
        ):
            if not math.isfinite(value) or not MIN_TRAJECTORY_SCALING <= value <= MAX_TRAJECTORY_SCALING:
                raise ValueError(
                    f"{name} must be within {MIN_TRAJECTORY_SCALING:.2f}–{MAX_TRAJECTORY_SCALING:.2f}"
                )
        normalized_feeding_vector = _normalize_feeding_vector(feeding_vector)
        if premouth_policy == "feeding-vector" and abs(normalized_feeding_vector[2]) > MAX_ABS_FEEDING_VECTOR_Z:
            raise ValueError(
                f"abs(normalized feeding_vector_z) must be <= {MAX_ABS_FEEDING_VECTOR_Z:.2f} for feeding-vector policy"
            )
        self.premouth_policy = premouth_policy
        self.safe_distance_m = safe_distance_m
        self.maximum_plan_translation_m = maximum_plan_translation_m
        self.feeding_vector_input = [float(component) for component in feeding_vector]
        self.feeding_vector_normalized = normalized_feeding_vector
        self.feeding_vector_sign = feeding_vector_sign
        self.mouth_sample_seconds = float(mouth_sample_seconds)
        self.trajectory_velocity_scaling = float(trajectory_velocity_scaling)
        self.trajectory_acceleration_scaling = float(trajectory_acceleration_scaling)
        self.latest_joint_state: JointState | None = None
        self.latest_speed_scaling: Float64 | None = None
        self.latest_robot_program_running: Bool | None = None
        self.latest_safety_mode: SafetyMode | None = None
        self.latest_robot_mode: RobotMode | None = None
        self.mouth_samples: list[MouthSample] = []
        self.normal_samples: list[MouthSample] = []
        self.latest_mouth_status: dict[str, Any] | None = None
        self.latest_mouth_status_received_monotonic: float | None = None
        self._validated_trajectory: Any | None = None
        latched_robot_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.create_subscription(Float64, SPEED_SCALING_TOPIC, self._speed_scaling_callback, 10)
        self.create_subscription(
            Bool,
            ROBOT_PROGRAM_RUNNING_TOPIC,
            self._robot_program_running_callback,
            latched_robot_state_qos,
        )
        self.create_subscription(
            SafetyMode,
            SAFETY_MODE_TOPIC,
            self._safety_mode_callback,
            latched_robot_state_qos,
        )
        self.create_subscription(
            RobotMode,
            ROBOT_MODE_TOPIC,
            self._robot_mode_callback,
            latched_robot_state_qos,
        )
        self.create_subscription(PoseStamped, MOUTH_TOPIC, self._mouth_callback, 20)
        self.create_subscription(Vector3Stamped, MOUTH_NORMAL_TOPIC, self._normal_callback, 20)
        self.create_subscription(String, MOUTH_STATUS_TOPIC, self._mouth_status_callback, 20)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        # Deliberately lazy.  In check and plan modes this probe must not even
        # create an ExecuteTrajectory client; those modes only use MoveIt's
        # planning action and never contact the execution action endpoint.
        self.execute_trajectory: ActionClient | None = None
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")

    def _execution_action_client(self) -> ActionClient:
        """Create the execution client only on the guarded execute path."""
        if self.execute_trajectory is None:
            self.execute_trajectory = ActionClient(self, ExecuteTrajectory, EXECUTE_TRAJECTORY_ACTION)
        return self.execute_trajectory

    def _joint_state_callback(self, message: JointState) -> None:
        self.latest_joint_state = message

    def _speed_scaling_callback(self, message: Float64) -> None:
        self.latest_speed_scaling = message

    def _robot_program_running_callback(self, message: Bool) -> None:
        self.latest_robot_program_running = message

    def _safety_mode_callback(self, message: SafetyMode) -> None:
        self.latest_safety_mode = message

    def _robot_mode_callback(self, message: RobotMode) -> None:
        self.latest_robot_mode = message

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

    def _mouth_status_callback(self, message: String) -> None:
        """Keep the detector's diagnosis so a missing pose is actionable."""
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            status = {"detected": False, "reason": "invalid_status_json", "raw": str(message.data)}
        if not isinstance(status, dict):
            status = {"detected": False, "reason": "invalid_status_payload", "raw": status}
        self.latest_mouth_status = status
        self.latest_mouth_status_received_monotonic = time.monotonic()

    @staticmethod
    def _mount_calibration() -> dict[str, Any]:
        try:
            configuration = json.loads(MOUNT_CALIBRATION_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"status": "missing", "reason": f"cannot read mount calibration config: {exc}"}
        status = configuration.get("calibration_status")
        if status not in {"provisional", "verified"}:
            return {"status": "invalid", "reason": "calibration_status must be provisional or verified"}
        transform = configuration.get("tool0_to_d435i_link")
        if not isinstance(transform, dict):
            return {"status": "invalid", "reason": "tool0_to_d435i_link is missing"}
        translation = transform.get("translation_m")
        rpy = transform.get("rpy_rad")
        values = list(translation or []) + list(rpy or [])
        if len(translation or []) != 3 or len(rpy or []) != 3 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
        ):
            return {"status": "invalid", "reason": "tool0_to_d435i_link must contain finite translation_m and rpy_rad"}
        metrics = configuration.get("calibration_metrics")
        manual_corrected = bool(
            status == "provisional"
            and isinstance(metrics, dict)
            and metrics.get("method") == "manual_physical_axis_alignment"
            and metrics.get("target_camera_optical_axes_in_base_link", {}).get("+Z_depth_forward")
            == [-1.0, 0.0, 0.0]
        )
        verified_physical = bool(
            status == "verified"
            and isinstance(metrics, dict)
            and isinstance(metrics.get("rms_residual_m"), (int, float))
            and float(metrics["rms_residual_m"]) <= 0.010
        )
        return {
            "status": status,
            "metrics": metrics,
            "tool0_to_d435i_link": {
                "translation_m": [float(value) for value in translation],
                "rpy_rad": [float(value) for value in rpy],
            },
            "corrected_physical_profile": manual_corrected or verified_physical,
            "config": str(MOUNT_CALIBRATION_CONFIG),
        }

    @staticmethod
    def _camera_mount_match(
        calibration: dict[str, Any], live_tool_to_camera_link: dict[str, Any]
    ) -> dict[str, Any]:
        if not live_tool_to_camera_link.get("available"):
            return {"matches": False, "reason": "live tool0 -> d435i_link TF is unavailable"}
        configured = calibration.get("tool0_to_d435i_link")
        if not isinstance(configured, dict):
            return {"matches": False, "reason": "corrected camera mount transform is unavailable in config"}
        translation_error = _norm(
            _subtract(live_tool_to_camera_link["position_m"], configured["translation_m"])
        )
        configured_quaternion = _quaternion_from_rpy(configured["rpy_rad"])
        rotation_error = _quaternion_distance_rad(
            live_tool_to_camera_link["orientation_quat_xyzw"], configured_quaternion
        )
        matches = bool(
            calibration.get("corrected_physical_profile")
            and translation_error <= MAX_CAMERA_MOUNT_TRANSLATION_ERROR_M
            and rotation_error <= MAX_CAMERA_MOUNT_ROTATION_ERROR_RAD
        )
        return {
            "matches": matches,
            "translation_error_m": translation_error,
            "rotation_error_rad": rotation_error,
            "rotation_error_deg": math.degrees(rotation_error),
            "maximum_translation_error_m": MAX_CAMERA_MOUNT_TRANSLATION_ERROR_M,
            "maximum_rotation_error_deg": math.degrees(MAX_CAMERA_MOUNT_ROTATION_ERROR_RAD),
            "configured": configured,
            "live": live_tool_to_camera_link,
            "reason": None if matches else "live camera mount TF does not match the corrected physical calibration",
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

    def _frame_transform(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = TF_DISCOVERY_TIMEOUT_SEC,
    ) -> dict[str, Any]:
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
        status = None
        if (
            self.latest_mouth_status is not None
            and self.latest_mouth_status_received_monotonic is not None
            and self.latest_mouth_status_received_monotonic >= start
        ):
            status = dict(self.latest_mouth_status)
            status["latest_received_age_sec"] = now - self.latest_mouth_status_received_monotonic
        if not received:
            reason = f"no {MOUTH_TOPIC} messages received in {duration_sec:.1f} seconds"
            if status is not None and status.get("reason"):
                reason += f"; {MOUTH_STATUS_TOPIC} reports {status['reason']}"
            result: dict[str, Any] = {"available": False, "reason": reason}
            if status is not None:
                result["perception_status"] = status
            return result
        wrong_frame = sorted({sample.frame_id for sample in received if sample.frame_id != BASE_FRAME})
        if wrong_frame:
            return {
                "available": False,
                "reason": f"mouth pose must use {BASE_FRAME}; received {wrong_frame}",
                "received_count": len(received),
            }
        if not valid:
            result = {
                "available": False,
                "reason": f"mouth pose is stale (over {MAX_MOUTH_POSE_AGE_SEC:.1f} s old)",
                "received_count": len(received),
            }
            if status is not None:
                result["perception_status"] = status
                if status.get("reason"):
                    result["reason"] += f"; {MOUTH_STATUS_TOPIC} reports {status['reason']}"
            return result
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

    def snapshot(self, mouth_sample_sec: float, *, inspect_controllers: bool = False) -> dict[str, Any]:
        self._wait_for_joint_state()
        self._spin_for(0.2)
        move_group_available = self.move_group.wait_for_server(timeout_sec=MOVE_GROUP_DISCOVERY_TIMEOUT_SEC)
        # Intentionally not checked in check/plan modes.  Creating or waiting
        # on an ExecuteTrajectory client is reserved exclusively for the
        # guarded execute path below.
        execute_trajectory_available = None
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
        controller_status = (
            self._controller_status()
            if inspect_controllers
            else {
                "available": None,
                "inspection": "skipped in plan mode; controller state is checked only immediately before execution",
            }
        )
        calibration = self._mount_calibration()
        tool_to_camera_link = self._frame_transform(TOOL_FRAME, CAMERA_LINK_FRAME)
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
            "tool0_to_camera_link_tf": tool_to_camera_link,
            "camera_mount_match": self._camera_mount_match(calibration, tool_to_camera_link),
            "required_nodes": {
                "move_group_exists": "/move_group" in nodes,
                "controller_manager_exists": "/controller_manager" in nodes,
            },
            "move_group_available": move_group_available,
            "execute_trajectory_available": execute_trajectory_available,
            "controllers": controller_status,
            "speed_slider_percent": None
            if self.latest_speed_scaling is None
            else float(self.latest_speed_scaling.data),
            "robot_program_running": None
            if self.latest_robot_program_running is None
            else bool(self.latest_robot_program_running.data),
            "safety_mode": None if self.latest_safety_mode is None else int(self.latest_safety_mode.mode),
            "safety_mode_normal": bool(
                self.latest_safety_mode is not None and self.latest_safety_mode.mode == SafetyMode.NORMAL
            ),
            "robot_mode": None if self.latest_robot_mode is None else int(self.latest_robot_mode.mode),
            "robot_mode_running": bool(
                self.latest_robot_mode is not None and self.latest_robot_mode.mode == RobotMode.RUNNING
            ),
            "mouth_pose": self._collect_mouth_samples(mouth_sample_sec),
            "mount_calibration": calibration,
        }

    @staticmethod
    def readiness_failures(
        snapshot: dict[str, Any], *, require_stable_mouth: bool, require_controller_inspection: bool = False
    ) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        if not snapshot["ur_base_tf"].get("available"):
            failures.append("TF base -> base_link is unavailable")
        if not snapshot["camera_tf"].get("available"):
            failures.append(f"TF base_link -> {CAMERA_OPTICAL_FRAME} is unavailable")
        if not snapshot.get("mount_calibration", {}).get("corrected_physical_profile"):
            failures.append("corrected physical D435i mount calibration profile is not loaded")
        if not snapshot.get("camera_mount_match", {}).get("matches"):
            failures.append(
                snapshot.get("camera_mount_match", {}).get("reason")
                or "live camera mount TF does not match corrected calibration"
            )
        if not snapshot.get("move_group_available"):
            failures.append(f"{MOVE_ACTION} is unavailable")
        if require_controller_inspection:
            nodes = snapshot["required_nodes"]
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
        goal.request.max_velocity_scaling_factor = self.trajectory_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.trajectory_acceleration_scaling
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
            "execute_trajectory_available": self._execution_action_client().wait_for_server(timeout_sec=2.0),
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
            failures.append(
                "speed slider is unavailable or outside the required "
                f"{MIN_EXECUTION_SPEED_PERCENT:.0f}%–{MAX_EXECUTION_SPEED_PERCENT:.0f}% range"
            )
        return failures

    def return_to_recorded_start(
        self,
        report_path: Path | None,
        *,
        confirm_real_motion: bool,
        no_execute: bool,
    ) -> tuple[int, dict[str, Any]]:
        """Return only to the recorded start of one successful prior probe run."""
        response: dict[str, Any] = {
            "mode": "return",
            "execution_sent": False,
            "execution_disabled": False,
            "automatic_retreat_sent": False,
        }
        if no_execute:
            response.update(
                {
                    "success": False,
                    "stage": "no_execute_policy",
                    "reason": "--no-execute prohibits the recorded-pose return for this invocation",
                    "execution_attempted": False,
                    "execution_disabled": True,
                }
            )
            return 2, response
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

    def _pre_mouth_candidates(
        self,
        mouth: list[float],
        camera: dict[str, Any],
        current_tool0: dict[str, Any],
    ) -> tuple[dict[str, list[float]], dict[str, Any] | None]:
        """Return both diagnostic candidates and the selected policy target."""
        base_x = _add(mouth, _scale(PRE_MOUTH_APPROACH_AXIS_BASE_LINK, self.safe_distance_m))
        tool_x_axis_base = _tool_x_axis_base(list(current_tool0["orientation_quat_xyzw"]))
        tcp_forward = _subtract(mouth, _scale(tuple(tool_x_axis_base), self.safe_distance_m))
        feeding_plus = _add(mouth, _scale(tuple(self.feeding_vector_normalized), self.safe_distance_m))
        feeding_minus = _subtract(mouth, _scale(tuple(self.feeding_vector_normalized), self.safe_distance_m))
        camera_position = camera.get("position_m")
        if not isinstance(camera_position, list) or len(camera_position) != 3:
            return {
                "base-x": base_x,
                "tcp-forward": tcp_forward,
                "feeding-vector-plus": feeding_plus,
                "feeding-vector-minus": feeding_minus,
            }, {
                "reason": "camera optical-frame position is unavailable",
                "tool_x_axis_base": tool_x_axis_base,
            }
        camera_to_mouth = _subtract(mouth, camera_position)
        distance = _norm(camera_to_mouth)
        if distance < 1e-6:
            return {
                "base-x": base_x,
                "tcp-forward": tcp_forward,
                "feeding-vector-plus": feeding_plus,
                "feeding-vector-minus": feeding_minus,
            }, {
                "reason": "camera and detected mouth positions are coincident",
                "tool_x_axis_base": tool_x_axis_base,
            }
        ray = [component / distance for component in camera_to_mouth]
        camera_ray = _subtract(mouth, [self.safe_distance_m * component for component in ray])
        camera_ray_alternative = _add(mouth, [self.safe_distance_m * component for component in ray])
        return (
            {
                "base-x": base_x,
                "tcp-forward": tcp_forward,
                "camera-ray": camera_ray,
                "camera-ray-alternative": camera_ray_alternative,
                "feeding-vector-plus": feeding_plus,
                "feeding-vector-minus": feeding_minus,
            },
            {
                "camera_to_mouth_vector_m": camera_to_mouth,
                "camera_to_mouth_unit_vector": ray,
                "camera_to_mouth_distance_m": distance,
                "tool_x_axis_base": tool_x_axis_base,
                "feeding_vector_input": self.feeding_vector_input,
                "feeding_vector_normalized": self.feeding_vector_normalized,
                "feeding_vector_sign": self.feeding_vector_sign,
            },
        )

    def plan(self) -> tuple[int, dict[str, Any]]:
        # Keep planning independent of controller-manager service calls.  The
        # plan path only needs live state, TF, perception, and /move_action;
        # controller state is rechecked later, immediately before an execute.
        snapshot = self.snapshot(mouth_sample_sec=self.mouth_sample_seconds, inspect_controllers=False)
        failures = self.readiness_failures(snapshot, require_stable_mouth=True)
        if failures:
            return 2, {"success": False, "mode": "plan", "stage": "readiness", "failures": failures, "checks": snapshot, "execution_sent": False}

        mouth = list(snapshot["mouth_pose"]["mean_position_m"])
        straw = list(snapshot["current_straw_tip_pose"]["position_m"])
        current_tool0 = snapshot["tool0_pose"]
        candidates, camera_ray_details = self._pre_mouth_candidates(
            mouth,
            snapshot["camera_tf"],
            current_tool0,
        )
        if (
            self.premouth_policy == "camera-ray"
            and camera_ray_details is not None
            and "reason" in camera_ray_details
        ):
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "camera_ray_input",
                "reason": camera_ray_details["reason"],
                "checks": snapshot,
                "execution_sent": False,
            }
        selected_candidate = (
            f"feeding-vector-{self.feeding_vector_sign}"
            if self.premouth_policy == "feeding-vector"
            else self.premouth_policy
        )
        if selected_candidate not in candidates:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "pre_mouth_policy",
                "reason": f"candidate for policy {selected_candidate} is unavailable",
                "checks": snapshot,
                "execution_sent": False,
            }
        pre_mouth = candidates[selected_candidate]
        tool_x_axis_base = (
            None if camera_ray_details is None else camera_ray_details.get("tool_x_axis_base")
        )
        if self.premouth_policy == "tcp-forward":
            if not isinstance(tool_x_axis_base, list) or len(tool_x_axis_base) != 3:
                return 2, {
                    "success": False,
                    "mode": "plan",
                    "stage": "tcp_forward_axis",
                    "reason": "tool0/TCP +X direction in base_link is unavailable",
                    "execution_sent": False,
                }
            print(f"tool_x_axis_base: {tool_x_axis_base}", file=sys.stderr, flush=True)
            print(f"safe_distance: {self.safe_distance_m:g}", file=sys.stderr, flush=True)
            print(
                f"pre_mouth - mouth: {_subtract(pre_mouth, mouth)}",
                file=sys.stderr,
                flush=True,
            )

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
                "premouth_policy": self.premouth_policy,
                "safe_distance_m": self.safe_distance_m,
                "feeding_vector_input": self.feeding_vector_input,
                "feeding_vector_normalized": self.feeding_vector_normalized,
                "feeding_vector_sign": self.feeding_vector_sign,
                "tool_x_axis_base": tool_x_axis_base,
                "selected_candidate": selected_candidate,
                "detected_mouth_pose": mouth,
                "pre_mouth_pose": pre_mouth,
                "target_tool0_pose": target_tool0_position,
                "target_tool0_orientation_quat_xyzw": orientation,
                "planned_tool0_translation_m": displacement,
                "planned_tool0_translation_norm_m": translation_norm,
                "orientation_preserved": True,
                "target_tool0_position_in_ur_base_m": target_tool0_in_ur_base,
                "target_tool0_radius_from_ur_base_m": target_tool0_radius,
                "maximum_radius_m": MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }

        if translation_norm > self.maximum_plan_translation_m:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "planned_translation_guard",
                "reason": (
                    "planned tool0 translation exceeds the configured "
                    f"{self.maximum_plan_translation_m:.2f} m diagnostic plan-only review limit"
                ),
                "premouth_policy": self.premouth_policy,
                "safe_distance_m": self.safe_distance_m,
                "feeding_vector_input": self.feeding_vector_input,
                "feeding_vector_normalized": self.feeding_vector_normalized,
                "feeding_vector_sign": self.feeding_vector_sign,
                "tool_x_axis_base": tool_x_axis_base,
                "selected_candidate": selected_candidate,
                "detected_mouth_pose": mouth,
                "pre_mouth_candidates": candidates,
                "pre_mouth_pose": pre_mouth,
                "pre_mouth_offset_base_link_m": _subtract(pre_mouth, mouth),
                "camera_ray_details": camera_ray_details,
                "target_tool0_pose": target_tool0_position,
                "target_tool0_orientation_quat_xyzw": orientation,
                "planned_tool0_translation_m": displacement,
                "planned_tool0_translation_norm_m": translation_norm,
                "maximum_translation_m": self.maximum_plan_translation_m,
                "orientation_preserved": True,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
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
            "timing_profile": {
                "mouth_sample_seconds": self.mouth_sample_seconds,
                "trajectory_velocity_scaling": self.trajectory_velocity_scaling,
                "trajectory_acceleration_scaling": self.trajectory_acceleration_scaling,
                "recommended_pendant_speed_percent": MAX_EXECUTION_SPEED_PERCENT,
            },
            "feeding_vector_input": self.feeding_vector_input,
            "feeding_vector_normalized": self.feeding_vector_normalized,
            "feeding_vector_sign": self.feeding_vector_sign,
            "tool_x_axis_base": tool_x_axis_base,
            "selected_candidate": selected_candidate,
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
            "maximum_plan_translation_m": self.maximum_plan_translation_m,
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
        if checks.get("robot_program_running") is not True:
            guards.append("UR External Control program is not Running")
        if checks.get("safety_mode_normal") is not True:
            guards.append("UR safety mode is unavailable or not NORMAL")
        if checks.get("robot_mode_running") is not True:
            guards.append("UR robot mode is unavailable or not RUNNING")
        return guards

    def _execute_validated_trajectory(self) -> dict[str, Any]:
        """Execute only the RobotTrajectory returned by the immediately prior plan."""
        if self._validated_trajectory is None:
            return {"success": False, "stage": "validated_trajectory", "reason": "no validated trajectory is cached"}
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._validated_trajectory
        client = self._execution_action_client()
        if not client.wait_for_server(timeout_sec=2.0):
            return {
                "success": False,
                "stage": "execute_trajectory_server",
                "reason": f"{EXECUTE_TRAJECTORY_ACTION} action server is unavailable",
                "execution_attempted": False,
            }
        future = client.send_goal_async(goal)
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
        allow_validated_feeding_vector_execute: bool,
        allow_validated_tcp_forward_execute: bool,
        no_execute: bool,
    ) -> tuple[int, dict[str, Any]]:
        if no_execute:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "no_execute_policy",
                "reason": "--no-execute prohibits all real motion for this invocation",
                "execution_attempted": False,
                "execution_sent": False,
                "execution_disabled": True,
            }
        policy_execution_allowed = (
            self.premouth_policy == "camera-ray" and allow_validated_camera_ray_execute
        ) or (
            self.premouth_policy == "feeding-vector" and allow_validated_feeding_vector_execute
        ) or (
            self.premouth_policy == "tcp-forward" and allow_validated_tcp_forward_execute
        )
        if not policy_execution_allowed:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "premouth_policy_execution_gate",
                "reason": (
                    "execution requires an explicit gate for the selected validated policy; use "
                    "--allow-validated-camera-ray-execute with camera-ray or "
                    "--allow-validated-feeding-vector-execute with feeding-vector or "
                    "--allow-validated-tcp-forward-execute with tcp-forward"
                ),
                "premouth_policy": self.premouth_policy,
                "safe_distance_m": self.safe_distance_m,
                "execution_attempted": False,
                "execution_sent": False,
                "execution_disabled": True,
            }

        # plan() performs all checks, collects a fresh bounded stable
        # observation, freezes its mean mouth pose, and computes this one
        # immutable pre-mouth target before any execution is considered.
        _, prepared = self.plan()
        prepared["mode"] = "execute"
        prepared["frozen_detected_mouth_pose"] = prepared.get("detected_mouth_pose")
        prepared["execution_disabled"] = False
        prepared["execution_sent"] = False
        prepared["frozen_target_policy"] = (
            f"The {self.mouth_sample_seconds:g}-second stable mouth-pose mean is frozen before planning; "
            "no perception updates are consumed during motion."
        )

        if not prepared.get("success"):
            failed_plan_stage = prepared.get("stage")
            failure_reason = prepared.get("reason")
            if not failure_reason and prepared.get("failures"):
                failure_reason = "; ".join(str(failure) for failure in prepared["failures"])
            prepared.update(
                {
                    "success": False,
                    "stage": "pre_execution_plan",
                    "pre_execution_plan_stage": failed_plan_stage,
                    "reason": failure_reason or "the fresh MoveIt plan failed",
                    "execution_attempted": False,
                    "execution_sent": False,
                }
            )
            return 2, prepared

        # This is the first point at which execute mode touches the execution
        # action endpoint.  Plan/check paths never instantiate this client.
        execution_client = self._execution_action_client()
        prepared["checks"]["execute_trajectory_available"] = execution_client.wait_for_server(timeout_sec=2.0)
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
        prepared["pre_execution_controller_state"] = controller_state
        prepared["pre_execution_speed_slider_percent"] = speed_percent
        prepared["pre_execution_robot_program_running"] = bool(
            self.latest_robot_program_running is not None and self.latest_robot_program_running.data
        )
        prepared["pre_execution_safety_mode_normal"] = bool(
            self.latest_safety_mode is not None and self.latest_safety_mode.mode == SafetyMode.NORMAL
        )
        prepared["pre_execution_robot_mode_running"] = bool(
            self.latest_robot_mode is not None and self.latest_robot_mode.mode == RobotMode.RUNNING
        )
        late_guards: list[str] = []
        if start_drift > 0.01:
            late_guards.append(f"tool0 moved {start_drift:.4f} m after planning")
        if not controller_state.get("scaled_joint_trajectory_controller_active"):
            late_guards.append("scaled_joint_trajectory_controller is no longer active")
        if speed_percent is None or not MIN_EXECUTION_SPEED_PERCENT <= speed_percent <= MAX_EXECUTION_SPEED_PERCENT:
            late_guards.append(
                "speed slider changed outside the required "
                f"{MIN_EXECUTION_SPEED_PERCENT:.0f}%–{MAX_EXECUTION_SPEED_PERCENT:.0f}% range"
            )
        if self.latest_robot_program_running is None or not self.latest_robot_program_running.data:
            late_guards.append("UR External Control program is no longer Running")
        if self.latest_safety_mode is None or self.latest_safety_mode.mode != SafetyMode.NORMAL:
            late_guards.append("UR safety mode is no longer NORMAL")
        if self.latest_robot_mode is None or self.latest_robot_mode.mode != RobotMode.RUNNING:
            late_guards.append("UR robot mode is no longer RUNNING")
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
    parser.add_argument("--mode", choices=("check", "plan", "execute", "return"), default="check")
    parser.add_argument(
        "--premouth-policy",
        choices=PREMOUTH_POLICIES,
        default=DEFAULT_PREMOUTH_POLICY,
        help=(
            "Pre-mouth target policy. camera-ray is the validated real default with the corrected D435i extrinsic; "
            "tcp-forward, legacy base-x, and configurable feeding-vector remain available."
        ),
    )
    parser.add_argument(
        "--safe-distance",
        type=float,
        default=DEFAULT_SAFE_DISTANCE_M,
        help="Pre-mouth stand-off distance in metres (default: 0.05).",
    )
    parser.add_argument(
        "--maximum-plan-translation",
        type=float,
        default=MAX_PLAN_TRANSLATION_M,
        help=(
            "Per-invocation tool0 translation cap in metres. The default is the UR10e "
            "nominal 1.30 m reach; the radial reach guard and MoveIt checks still apply."
        ),
    )
    parser.add_argument(
        "--feeding-vector-x",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[0],
        help="Base-link X component of the feeding vector (real default: 0).",
    )
    parser.add_argument(
        "--feeding-vector-y",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[1],
        help="Base-link Y component of the feeding vector (real default: -1).",
    )
    parser.add_argument(
        "--feeding-vector-z",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[2],
        help="Base-link Z component of the feeding vector (real default: 0).",
    )
    parser.add_argument(
        "--feeding-vector-sign",
        choices=FEEDING_VECTOR_SIGNS,
        default="plus",
        help="Select mouth ± safe_distance * normalized(feeding vector).",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Explicitly prohibit motion, even if --mode execute or --mode return is supplied.",
    )
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required with --mode execute/return and UR10E_ALLOW_REAL_EXECUTION=1.",
    )
    parser.add_argument(
        "--allow-validated-camera-ray-execute",
        action="store_true",
        help="Additional gate for one guarded camera-ray pre-mouth execution after RViz validation.",
    )
    parser.add_argument(
        "--allow-validated-feeding-vector-execute",
        action="store_true",
        help="Additional gate for one guarded feeding-vector pre-mouth execution after RViz validation.",
    )
    parser.add_argument(
        "--allow-validated-tcp-forward-execute",
        action="store_true",
        help="Additional gate for one guarded tcp-forward pre-mouth execution after RViz validation.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Mouth-pose collection duration for check mode (default: 1.0; use 8 for a stability report).",
    )
    parser.add_argument(
        "--mouth-sample-seconds",
        type=float,
        default=DEFAULT_MOUTH_SAMPLE_SECONDS,
        help="Stable mouth-pose collection duration used by plan/execute (default: 2.0 seconds).",
    )
    parser.add_argument(
        "--trajectory-velocity-scaling",
        type=float,
        default=DEFAULT_TRAJECTORY_VELOCITY_SCALING,
        help="MoveIt velocity scaling for the validated plan (default: 0.10; allowed: 0.01–0.20).",
    )
    parser.add_argument(
        "--trajectory-acceleration-scaling",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
        help="MoveIt acceleration scaling for the validated plan (default: 0.10; allowed: 0.01–0.20).",
    )
    parser.add_argument(
        "--return-report",
        type=Path,
        help="Successful execution report whose recorded start pose is the guarded --mode return target.",
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
    if not math.isfinite(args.mouth_sample_seconds) or not MIN_MOUTH_SAMPLE_SECONDS <= args.mouth_sample_seconds <= MAX_MOUTH_SAMPLE_SECONDS:
        parser.error(
            f"--mouth-sample-seconds must be within {MIN_MOUTH_SAMPLE_SECONDS:.1f}–{MAX_MOUTH_SAMPLE_SECONDS:.1f}"
        )
    for option, value in (
        ("--trajectory-velocity-scaling", args.trajectory_velocity_scaling),
        ("--trajectory-acceleration-scaling", args.trajectory_acceleration_scaling),
    ):
        if not math.isfinite(value) or not MIN_TRAJECTORY_SCALING <= value <= MAX_TRAJECTORY_SCALING:
            parser.error(f"{option} must be within {MIN_TRAJECTORY_SCALING:.2f}–{MAX_TRAJECTORY_SCALING:.2f}")
    if not math.isfinite(args.safe_distance) or not MIN_SAFE_DISTANCE_M <= args.safe_distance <= MAX_SAFE_DISTANCE_M:
        parser.error(f"--safe-distance must be within {MIN_SAFE_DISTANCE_M:.2f}–{MAX_SAFE_DISTANCE_M:.2f} m")
    if not math.isfinite(args.maximum_plan_translation) or not 0.0 < args.maximum_plan_translation <= MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
        parser.error(
            "--maximum-plan-translation must be positive and no larger than "
            f"{MAX_TOOL0_RADIUS_FROM_UR_BASE_M:.2f} m"
        )
    feeding_vector = (args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z)
    try:
        normalized_feeding_vector = _normalize_feeding_vector(feeding_vector)
    except ValueError as exc:
        parser.error(str(exc))
    if args.premouth_policy == "feeding-vector" and abs(normalized_feeding_vector[2]) > MAX_ABS_FEEDING_VECTOR_Z:
        parser.error(f"feeding-vector policy requires abs(normalized z) <= {MAX_ABS_FEEDING_VECTOR_Z:.2f}")
    return args


def main() -> int:
    args = _parse_args()
    if args.premouth_policy == "tcp-forward":
        print(
            f"Using tcp-forward policy: current tool0/TCP +X, safe_distance={args.safe_distance:g}",
            file=sys.stderr,
            flush=True,
        )
    if args.premouth_policy == "feeding-vector":
        normalized = _normalize_feeding_vector(
            (args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z)
        )
        vector_text = ", ".join(f"{component:g}" for component in normalized)
        print(
            f"Using feeding-vector policy: [{vector_text}], safe_distance={args.safe_distance:g}",
            file=sys.stderr,
            flush=True,
        )
    rclpy.init()
    node = RealPreMouthFromPerceptionPlan(
        premouth_policy=args.premouth_policy,
        safe_distance_m=args.safe_distance,
        maximum_plan_translation_m=args.maximum_plan_translation,
        feeding_vector=(args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z),
        feeding_vector_sign=args.feeding_vector_sign,
        mouth_sample_seconds=args.mouth_sample_seconds,
        trajectory_velocity_scaling=args.trajectory_velocity_scaling,
        trajectory_acceleration_scaling=args.trajectory_acceleration_scaling,
    )
    try:
        if args.mode == "check":
            snapshot = node.snapshot(mouth_sample_sec=args.sample_seconds, inspect_controllers=True)
            failures = node.readiness_failures(
                snapshot, require_stable_mouth=False, require_controller_inspection=True
            )
            response = {
                "success": not failures,
                "mode": "check",
                "premouth_policy": args.premouth_policy,
                "safe_distance_m": args.safe_distance,
                "feeding_vector_input": [args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z],
                "feeding_vector_normalized": _normalize_feeding_vector(
                    (args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z)
                ),
                "feeding_vector_sign": args.feeding_vector_sign,
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }
            exit_code = 0 if not failures else 2
        elif args.mode == "plan":
            exit_code, response = node.plan()
        elif args.mode == "execute":
            exit_code, response = node.execute(
                confirm_real_motion=args.confirm_real_motion,
                allow_validated_camera_ray_execute=args.allow_validated_camera_ray_execute,
                allow_validated_feeding_vector_execute=args.allow_validated_feeding_vector_execute,
                allow_validated_tcp_forward_execute=args.allow_validated_tcp_forward_execute,
                no_execute=args.no_execute,
            )
        else:
            exit_code, response = node.return_to_recorded_start(
                args.return_report,
                confirm_real_motion=args.confirm_real_motion,
                no_execute=args.no_execute,
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
