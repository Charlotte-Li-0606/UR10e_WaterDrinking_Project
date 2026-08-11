#!/usr/bin/env python3
"""Guarded real-UR10e pre-mouth probe driven by mouth candidates.

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
from geometry_msgs.msg import Pose, PoseStamped  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup  # noqa: E402
from moveit_msgs.msg import (  # noqa: E402
    BoundingVolume,
    Constraints,
    OrientationConstraint,
    PlanningSceneComponents,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import (  # noqa: E402
    GetCartesianPath,
    GetPlanningScene,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.action import ActionClient  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from std_msgs.msg import Bool, Float64, String  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402

from robot_layer.arm_ur10e.agent_tools.planning_scene_manager import (  # noqa: E402
    COMBINED_TOOL_COLLISION_OBJECT_ID,
    PlanningSceneObstacleConfig,
    PlanningSceneObstacleManager,
    combined_tool_collision_verification,
)
from robot_layer.arm_ur10e.perception.real_mouth_target_tracker import (  # noqa: E402
    RealMouthTargetTracker,
    validate_target_selection,
)


BASE_FRAME = "base_link"
UR_BASE_FRAME = "base"
MOVEIT_PLANNING_FRAME = "world"
TOOL_FRAME = "tool0"
CAMERA_OPTICAL_FRAME = "d435i_color_optical_frame"
CAMERA_LINK_FRAME = "d435i_link"
MOUTH_TOPIC = "/detected_mouth_pose"
MOUTH_CANDIDATES_TOPIC = "/detected_mouth_candidates"
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
MAX_SAFE_DISTANCE_M = 0.050
DEFAULT_FEEDING_VECTOR = (0.0, -1.0, 0.0)
FEEDING_VECTOR_SIGNS = ("plus", "minus")
MAX_ABS_FEEDING_VECTOR_Z = 0.30
MIN_FACE_CLEARANCE_M = 0.050
# This start-state check only prevents crossing through the mouth plane.  It
# must not demand the final stand-off before a move that is itself increasing
# the separation to the final 50 mm pre-mouth point.
MIN_CURRENT_SIDE_MARGIN_M = 0.005
MOUNT_CALIBRATION_CONFIG = PROJECT_ROOT / "config/ur10e_real/d435i_mount_calibration.json"

# The nominal UR10e reach is 1.30 m.  Exact reachability and collision checks
# are delegated to MoveIt, but a target beyond this gross physical radius is
# physically unreasonable and is rejected before planning.  In particular, do
# not use the simulated feeding-policy XYZ minimum/maximum here: base_link is
# rotated relative to the physical UR base on this real robot.
MAX_TOOL0_RADIUS_FROM_UR_BASE_M = 1.30
MAX_PLAN_TRANSLATION_M = MAX_TOOL0_RADIUS_FROM_UR_BASE_M
MIN_EXECUTION_SPEED_PERCENT = 5.0
MAX_EXECUTION_SPEED_PERCENT = 30.0
DEFAULT_MOUTH_SAMPLE_SECONDS = 2.0
MIN_MOUTH_SAMPLE_SECONDS = 0.5
MAX_MOUTH_SAMPLE_SECONDS = 8.0
DEFAULT_TRAJECTORY_VELOCITY_SCALING = 0.30
DEFAULT_TRAJECTORY_ACCELERATION_SCALING = 0.30
MIN_TRAJECTORY_SCALING = 0.01
MAX_TRAJECTORY_SCALING = 0.30
MAX_MOUTH_POSE_AGE_SEC = 1.0
MIN_STABLE_SAMPLES = 3
MAX_POSE_SPREAD_M = 0.025
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_RAD = 0.001
FINAL_ORIENTATION_TOLERANCE_RAD = 0.01
# The physical cup/tool must remain upright.  On this installation tool0 +Z is
# the flange's downward axis, so it must stay within 5 degrees of base_link -Z.
# The MoveIt rotation-vector constraint bounds the two tilt components.  Each
# component is limited to 5/sqrt(2) degrees so their combined magnitude cannot
# exceed the independently checked 5-degree physical limit.  Rotation about
# tool0 +Z (wrist-3-style spin) remains free.
MAX_TOOL_VERTICAL_TILT_RAD = math.radians(5.0)
VERTICAL_AXIS_COMPONENT_TOLERANCE_RAD = MAX_TOOL_VERTICAL_TILT_RAD / math.sqrt(2.0)
FREE_TOOL_AXIS_SPIN_TOLERANCE_RAD = math.pi
TOOL_VERTICAL_AXIS_TOOL0 = (0.0, 0.0, 1.0)
REQUIRED_VERTICAL_AXIS_BASE_LINK = (0.0, 0.0, -1.0)
FK_SERVICE = "/compute_fk"
FK_SERVICE_DISCOVERY_TIMEOUT_SEC = 2.0
FK_WAYPOINT_TIMEOUT_SEC = 0.5
FK_TRAJECTORY_VALIDATION_TIMEOUT_SEC = 15.0
FINAL_STRAW_TARGET_TOLERANCE_M = 0.02
RETURN_START_MATCH_TOLERANCE_M = 0.02
RETURN_START_MATCH_ORIENTATION_TOLERANCE_RAD = 0.02
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PLANNER = "LIN"
ADAPTIVE_PREMOUTH_STANDOFFS_M = (0.050, 0.070, 0.090, 0.120, 0.150, 0.180)
ADAPTIVE_PREMOUTH_YAWS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0)
IK_SERVICE = "/compute_ik"
STATE_VALIDITY_SERVICE = "/check_state_validity"
CARTESIAN_PATH_SERVICE = "/compute_cartesian_path"
PLANNING_SCENE_SERVICE = "/get_planning_scene"
IK_TIMEOUT_SEC = 0.25
CANDIDATE_SERVICE_TIMEOUT_SEC = 1.0
CARTESIAN_MAX_STEP_M = 0.01
CARTESIAN_COMPLETE_FRACTION = 0.999
CARTESIAN_MAX_REVOLUTE_JUMP_RAD = math.radians(20.0)
MONITORED_ROBOT_LINKS = ("wrist_2_link", "wrist_3_link", TOOL_FRAME)
HUMAN_OBJECT_PREFIXES = (
    "human_",
    "face_",
    "real_human_obstacle_",
)
# At low pendant settings, the scaled controller can take much
# longer than the nominal MoveIt trajectory duration.  Keep the client alive
# long enough to receive the controller's real terminal result.
ACTION_TIMEOUT_SEC = 180.0
JOINT_STATE_DISCOVERY_TIMEOUT_SEC = 5.0
TF_DISCOVERY_TIMEOUT_SEC = 8.0
MOVE_GROUP_DISCOVERY_TIMEOUT_SEC = 8.0
MAX_CAMERA_MOUNT_TRANSLATION_ERROR_M = 0.001
MAX_CAMERA_MOUNT_ROTATION_ERROR_RAD = math.radians(0.5)
MAX_PRE_EXECUTION_TARGET_DRIFT_M = 0.030
MAX_PRE_EXECUTION_OBSTACLE_DRIFT_M = 0.050


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


def _finite_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        converted = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    return converted if all(math.isfinite(component) for component in converted) else None


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


def _quaternion_multiply_xyzw(first: list[float], second: list[float]) -> list[float]:
    """Return normalized ``first * second`` without using Euler angles."""
    x1, y1, z1, w1 = (float(value) for value in first)
    x2, y2, z2, w2 = (float(value) for value in second)
    result = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
    magnitude = math.sqrt(sum(value * value for value in result))
    if not math.isfinite(magnitude) or magnitude < 1e-9:
        raise ValueError("quaternion product is not finite and nonzero")
    return [value / magnitude for value in result]


def _orientation_after_local_tool_z_rotation(
    orientation_xyzw: list[float], yaw_rad: float
) -> list[float]:
    """Spin a verified flange-down pose about tool0 local Z only."""
    if not math.isfinite(float(yaw_rad)):
        raise ValueError("tool-local yaw must be finite")
    half = float(yaw_rad) / 2.0
    return _quaternion_multiply_xyzw(
        orientation_xyzw,
        [0.0, 0.0, math.sin(half), math.cos(half)],
    )


def _adaptive_premouth_pose_candidates(
    *,
    mouth_position_m: list[float],
    approach_offset_unit: list[float],
    verified_flange_down_orientation_xyzw: list[float],
    straw_tip_offset_tool0_m: tuple[float, float, float] = STRAW_TIP_OFFSET_TOOL0_M,
    standoffs_m: tuple[float, ...] = ADAPTIVE_PREMOUTH_STANDOFFS_M,
    yaws_deg: tuple[float, ...] = ADAPTIVE_PREMOUTH_YAWS_DEG,
) -> list[dict[str, Any]]:
    """Generate calibrated straw-tip/tool0 candidates on one approach line."""
    mouth = _finite_xyz(mouth_position_m)
    approach = _finite_xyz(approach_offset_unit)
    if mouth is None or approach is None:
        raise ValueError("mouth position and approach direction must be finite XYZ values")
    approach_norm = _norm(approach)
    if approach_norm < 1e-9:
        raise ValueError("approach direction must be nonzero")
    approach = [value / approach_norm for value in approach]
    candidates: list[dict[str, Any]] = []
    for standoff_m in standoffs_m:
        if not math.isfinite(float(standoff_m)) or float(standoff_m) <= 0.0:
            raise ValueError("candidate standoffs must be positive and finite")
        straw_tip = _add(mouth, [float(standoff_m) * value for value in approach])
        for yaw_deg in yaws_deg:
            orientation = _orientation_after_local_tool_z_rotation(
                verified_flange_down_orientation_xyzw,
                math.radians(float(yaw_deg)),
            )
            rotated_straw_offset = _rotate_tool_vector(
                orientation,
                straw_tip_offset_tool0_m,
            )
            tool0_position = _subtract(straw_tip, rotated_straw_offset)
            reconstructed_straw_tip = _add(tool0_position, rotated_straw_offset)
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "standoff_m": float(standoff_m),
                    "yaw_deg": float(yaw_deg),
                    "approach_offset_unit": list(approach),
                    "straw_tip_pose": {
                        "frame_id": BASE_FRAME,
                        "position_m": straw_tip,
                    },
                    "tool0_pose": {
                        "frame_id": BASE_FRAME,
                        "link_name": TOOL_FRAME,
                        "position_m": tool0_position,
                        "orientation_quat_xyzw": orientation,
                    },
                    "reconstructed_straw_tip_position_m": reconstructed_straw_tip,
                    "straw_tip_reconstruction_error_m": _norm(
                        _subtract(reconstructed_straw_tip, straw_tip)
                    ),
                    "flange_vertical_axis_error_rad": _tool_vertical_tilt_rad(
                        orientation
                    ),
                    "flange_vertical_axis_error_deg": math.degrees(
                        _tool_vertical_tilt_rad(orientation)
                    ),
                    "wrist_3_joint_direct_command": False,
                    "orientation_source": (
                        "verified live flange-down quaternion plus tool-local-Z spin"
                    ),
                }
            )
    return candidates


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


def _execution_target_verification(
    *,
    final_tool0: dict[str, Any],
    start_tool0: dict[str, Any],
    target_tool0: dict[str, Any],
    target_straw_tip_position_m: list[float],
) -> dict[str, Any]:
    """Verify the executed pose against the planned target, not the start.

    Adaptive yaw intentionally changes the tool quaternion from its starting
    value.  The start-to-final difference remains useful diagnostic data, but
    only final-to-target error determines whether execution reached the
    orientation MoveIt planned and collision-checked.
    """
    if not final_tool0.get("available"):
        return {
            "available": False,
            "reason": "final TF base_link -> tool0 is unavailable",
        }
    final_position = _finite_xyz(final_tool0.get("position_m"))
    start_position = _finite_xyz(start_tool0.get("position_m"))
    target_straw = _finite_xyz(target_straw_tip_position_m)
    orientations: dict[str, list[float]] = {}
    for label, source in (
        ("final", final_tool0),
        ("start", start_tool0),
        ("target", target_tool0),
    ):
        raw = source.get("orientation_quat_xyzw")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return {
                "available": False,
                "reason": f"{label} tool0 orientation is unavailable",
            }
        try:
            converted = [float(value) for value in raw]
        except (TypeError, ValueError):
            return {
                "available": False,
                "reason": f"{label} tool0 orientation is invalid",
            }
        magnitude = math.sqrt(sum(value * value for value in converted))
        if (
            not all(math.isfinite(value) for value in converted)
            or not math.isfinite(magnitude)
            or magnitude < 1e-9
        ):
            return {
                "available": False,
                "reason": f"{label} tool0 orientation is invalid",
            }
        orientations[label] = [value / magnitude for value in converted]
    if final_position is None or start_position is None or target_straw is None:
        return {
            "available": False,
            "reason": "start, final, or target execution position is invalid",
        }

    final_straw = _add(
        final_position,
        _rotate_tool_vector(
            orientations["final"],
            STRAW_TIP_OFFSET_TOOL0_M,
        ),
    )
    final_error = _norm(_subtract(final_straw, target_straw))
    start_to_final = _quaternion_distance_rad(
        orientations["final"], orientations["start"]
    )
    start_to_target = _quaternion_distance_rad(
        orientations["target"], orientations["start"]
    )
    final_to_target = _quaternion_distance_rad(
        orientations["final"], orientations["target"]
    )
    orientation_matches_target = (
        final_to_target <= FINAL_ORIENTATION_TOLERANCE_RAD
    )
    return {
        "available": True,
        "final_tool0_pose": final_tool0,
        "final_straw_tip_pose": {
            "frame_id": BASE_FRAME,
            "position_m": final_straw,
        },
        "final_straw_tip_to_pre_mouth_error_m": final_error,
        "actual_tool0_displacement_m": _subtract(
            final_position, start_position
        ),
        "orientation_difference_from_start_rad": start_to_final,
        "planned_orientation_difference_from_start_rad": start_to_target,
        "final_orientation_error_from_planned_target_rad": final_to_target,
        "straw_tip_within_target_tolerance": (
            final_error <= FINAL_STRAW_TARGET_TOLERANCE_M
        ),
        "orientation_matches_planned_target": orientation_matches_target,
        # Compatibility alias for existing report consumers.  "Stable" now
        # means stable at the planned target, not unchanged from the start.
        "orientation_stable": orientation_matches_target,
        "orientation_verification_reference": "planned_target_tool0_orientation",
    }


def _tool_vertical_tilt_rad(orientation_xyzw: list[float]) -> float:
    """Angle between tool0 +Z and base_link -Z, independent of tool spin."""
    axis = _rotate_tool_vector(orientation_xyzw, TOOL_VERTICAL_AXIS_TOOL0)
    magnitude = _norm(axis)
    if not math.isfinite(magnitude) or magnitude < 1e-9:
        raise RuntimeError("tool0 vertical axis in base_link is invalid")
    dot = sum(
        float(component) * float(required)
        for component, required in zip(axis, REQUIRED_VERTICAL_AXIS_BASE_LINK)
    ) / magnitude
    return math.acos(max(-1.0, min(1.0, dot)))


def _vertical_reference_quaternion(orientation_xyzw: list[float]) -> list[float]:
    """Project an orientation to exact vertical while preserving its spin.

    The returned orientation maps tool0 +Z to base_link -Z.  Its horizontal
    tool +X heading is taken from the supplied orientation, avoiding an
    unnecessary wrist spin when the constraint becomes active.
    """
    source = [float(component) for component in orientation_xyzw]
    if len(source) != 4 or not all(math.isfinite(component) for component in source):
        raise RuntimeError("tool0 orientation quaternion is invalid")
    magnitude = math.sqrt(sum(component * component for component in source))
    if magnitude < 1e-9:
        raise RuntimeError("tool0 orientation quaternion has zero length")
    source = [component / magnitude for component in source]
    tool_x = _rotate_tool_vector(source, (1.0, 0.0, 0.0))
    horizontal = math.hypot(tool_x[0], tool_x[1])
    if horizontal >= 1e-6:
        heading = math.atan2(tool_x[1], tool_x[0])
    else:
        tool_y = _rotate_tool_vector(source, (0.0, 1.0, 0.0))
        if math.hypot(tool_y[0], tool_y[1]) < 1e-6:
            raise RuntimeError("tool0 spin cannot be recovered from the supplied orientation")
        heading = math.atan2(tool_y[1], tool_y[0]) + math.pi / 2.0
    reference = [math.cos(heading / 2.0), math.sin(heading / 2.0), 0.0, 0.0]
    if sum(a * b for a, b in zip(reference, source)) < 0.0:
        reference = [-component for component in reference]
    return reference


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
        target_selection: str = "center",
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
        self.target_selection = validate_target_selection(target_selection)
        self.target_tracker = RealMouthTargetTracker(self.target_selection, base_frame=BASE_FRAME)
        self.mouth_sample_seconds = float(mouth_sample_seconds)
        self.trajectory_velocity_scaling = float(trajectory_velocity_scaling)
        self.trajectory_acceleration_scaling = float(trajectory_acceleration_scaling)
        self.latest_joint_state: JointState | None = None
        self.latest_speed_scaling: Float64 | None = None
        self.latest_robot_program_running: Bool | None = None
        self.latest_safety_mode: SafetyMode | None = None
        self.latest_robot_mode: RobotMode | None = None
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
        self.create_subscription(String, MOUTH_CANDIDATES_TOPIC, self._mouth_candidates_callback, 20)
        self.create_subscription(String, MOUTH_STATUS_TOPIC, self._mouth_status_callback, 20)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        # Deliberately lazy.  In check and plan modes this probe must not even
        # create an ExecuteTrajectory client; those modes only use MoveIt's
        # planning action and never contact the execution action endpoint.
        self.execute_trajectory: ActionClient | None = None
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self.compute_fk = self.create_client(GetPositionFK, FK_SERVICE)
        self.compute_ik = self.create_client(GetPositionIK, IK_SERVICE)
        self.check_state_validity = self.create_client(
            GetStateValidity,
            STATE_VALIDITY_SERVICE,
        )
        self.compute_cartesian_path = self.create_client(
            GetCartesianPath,
            CARTESIAN_PATH_SERVICE,
        )
        self.get_planning_scene = self.create_client(
            GetPlanningScene,
            PLANNING_SCENE_SERVICE,
        )

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

    def _mouth_candidates_callback(self, message: String) -> None:
        """Update the guarded real target lock from all visible candidates."""
        result = self.target_tracker.update_json(message.data)
        if not result.get("success") and result.get("identity_unsafe"):
            self.get_logger().warning(str(result.get("reason") or "mouth target identity became unsafe"))

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
        """Collect and validate the explicitly selected multi-face target."""
        start = time.monotonic()
        self._spin_for(duration_sec)
        now = time.monotonic()
        result = self.target_tracker.observation(
            started_monotonic=start,
            now_monotonic=now,
            max_age_sec=MAX_MOUTH_POSE_AGE_SEC,
            minimum_samples=MIN_STABLE_SAMPLES,
            max_spread_m=MAX_POSE_SPREAD_M,
        )
        result["sample_duration_sec"] = duration_sec
        status = None
        if (
            self.latest_mouth_status is not None
            and self.latest_mouth_status_received_monotonic is not None
            and self.latest_mouth_status_received_monotonic >= start
        ):
            status = dict(self.latest_mouth_status)
            status["latest_received_age_sec"] = now - self.latest_mouth_status_received_monotonic
        if status is not None:
            result["perception_status"] = status
            if not result.get("available") and status.get("reason"):
                result["reason"] = f"{result.get('reason', 'selected target unavailable')}; {MOUTH_STATUS_TOPIC} reports {status['reason']}"
        return result

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
        vertical_axis_guard: dict[str, Any] = {
            "available": False,
            "tool_axis_tool0": list(TOOL_VERTICAL_AXIS_TOOL0),
            "required_axis_base_link": list(REQUIRED_VERTICAL_AXIS_BASE_LINK),
            "maximum_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
            "maximum_tilt_deg": math.degrees(MAX_TOOL_VERTICAL_TILT_RAD),
        }
        if tool0.get("available"):
            try:
                tilt = _tool_vertical_tilt_rad(tool0["orientation_quat_xyzw"])
                vertical_axis_guard.update(
                    {
                        "available": True,
                        "tilt_rad": tilt,
                        "tilt_deg": math.degrees(tilt),
                        "within_limit": tilt <= MAX_TOOL_VERTICAL_TILT_RAD,
                    }
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                vertical_axis_guard["reason"] = str(exc)
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
        tool_to_camera_optical = self._frame_transform(
            TOOL_FRAME,
            CAMERA_OPTICAL_FRAME,
        )
        return {
            "joint_state": self._joint_state_status(),
            "tool0_pose": tool0,
            "tool_vertical_axis_guard": vertical_axis_guard,
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
            "tool0_to_camera_optical_tf": tool_to_camera_optical,
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

    def _apply_multi_person_planning_scene(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Apply and verify deterministic collision geometry before planning."""
        mouth = snapshot.get("mouth_pose", {})
        visible = mouth.get("visible_candidates")
        selected_index = mouth.get("selected_candidate_index")
        selected_mean = mouth.get("mean_position_m")
        camera_position = snapshot.get("camera_tf", {}).get("position_m")
        if not isinstance(visible, list) or not visible:
            return {"success": False, "reason": "no fresh visible candidates are available for obstacle geometry"}
        if not isinstance(selected_index, int) or not 0 <= selected_index < len(visible):
            return {"success": False, "reason": "selected candidate index is invalid for obstacle geometry"}
        if not isinstance(selected_mean, list) or len(selected_mean) != 3:
            return {"success": False, "reason": "stable selected mouth mean is unavailable for obstacle geometry"}
        if (
            not isinstance(camera_position, list)
            or len(camera_position) != 3
            or not all(isinstance(component, (int, float)) and math.isfinite(float(component)) for component in camera_position)
        ):
            return {"success": False, "reason": "camera position is unavailable for obstacle geometry"}
        mouth_positions: list[list[float]] = []
        try:
            for candidate in visible:
                position = candidate["position_m"]
                values = [float(component) for component in position]
                if len(values) != 3 or not all(math.isfinite(component) for component in values):
                    raise ValueError
                mouth_positions.append(values)
            frozen_selected = [float(component) for component in selected_mean]
            if len(frozen_selected) != 3 or not all(math.isfinite(component) for component in frozen_selected):
                raise ValueError
            mouth_positions[selected_index] = frozen_selected
        except (KeyError, TypeError, ValueError):
            return {"success": False, "reason": "visible candidate positions are invalid"}

        manager: PlanningSceneObstacleManager | None = None
        try:
            manager = PlanningSceneObstacleManager(
                PlanningSceneObstacleConfig(
                    base_frame=BASE_FRAME,
                    mouth_topic=MOUTH_TOPIC,
                    include_table=False,
                    service_timeout_sec=5.0,
                    mouth_wait_timeout_sec=1.0,
                )
            )
            combined_tool = manager.apply_combined_tool_collision(verify=True)
            if not combined_tool.get("success"):
                result = {
                    "success": False,
                    "reason": (
                        combined_tool.get("reason")
                        or "combined camera/cup-holder/straw collision geometry could not be verified"
                    ),
                    "combined_tool_collision_geometry": combined_tool,
                }
            else:
                result = manager.apply_people(
                    mouth_positions,
                    camera_position=camera_position,
                    verify=True,
                )
                result["combined_tool_collision_geometry"] = combined_tool
        except Exception as exc:
            result = {"success": False, "reason": f"PlanningScene obstacle manager raised {exc.__class__.__name__}: {exc}"}
        finally:
            if manager is not None:
                manager.destroy_node()
        result["target_selection"] = self.target_selection
        result["selected_candidate_index"] = selected_index
        result["frozen_candidate_positions_m"] = mouth_positions
        result["selected_mouth_position_m"] = frozen_selected
        result["geometry_layer"] = "fixed_human_safety_objects"
        result["dynamic_octomap_modified"] = False
        return result

    def _apply_combined_tool_collision_geometry(self) -> dict[str, Any]:
        """Attach the rigid physical tool body before any search or route plan."""
        manager: PlanningSceneObstacleManager | None = None
        try:
            manager = PlanningSceneObstacleManager(
                PlanningSceneObstacleConfig(
                    base_frame=BASE_FRAME,
                    mouth_topic=MOUTH_TOPIC,
                    include_table=False,
                    service_timeout_sec=5.0,
                    mouth_wait_timeout_sec=1.0,
                )
            )
            return manager.apply_combined_tool_collision(verify=True)
        except Exception as exc:
            return {
                "success": False,
                "operation": "attach_combined_tool_collision",
                "reason": (
                    "PlanningScene combined-tool collision manager raised "
                    f"{exc.__class__.__name__}: {exc}"
                ),
                "execution_sent": False,
            }
        finally:
            if manager is not None:
                manager.destroy_node()

    @staticmethod
    def readiness_failures(
        snapshot: dict[str, Any], *, require_stable_mouth: bool, require_controller_inspection: bool = False
    ) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        vertical_axis = snapshot.get("tool_vertical_axis_guard", {})
        if not vertical_axis.get("available"):
            failures.append("tool vertical-axis alignment could not be verified")
        elif not vertical_axis.get("within_limit"):
            failures.append(
                "tool0 +Z is not aligned with base_link -Z: "
                f"tilt {float(vertical_axis.get('tilt_deg', float('nan'))):.2f} deg exceeds "
                f"the {math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
            )
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
            failures.append(mouth.get("reason", f"{MOUTH_CANDIDATES_TOPIC} is unavailable"))
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

    def _current_robot_state(self) -> RobotState | None:
        if self.latest_joint_state is None:
            return None
        state = RobotState()
        state.joint_state = self.latest_joint_state
        # Preserve verified attached collision bodies from the monitored scene
        # while replacing the joint values with this fresh live state.
        state.is_diff = True
        return state

    @staticmethod
    def _collision_contact_report(contact: Any) -> dict[str, Any]:
        body_1 = str(contact.contact_body_1)
        body_2 = str(contact.contact_body_2)
        type_1 = int(contact.body_type_1)
        type_2 = int(contact.body_type_2)
        human = any(
            body.startswith(HUMAN_OBJECT_PREFIXES)
            for body in (body_1, body_2)
        )
        octomap = "<octomap>" in (body_1, body_2)
        self_collision = type_1 == 0 and type_2 == 0
        tool_component = any(
            token in body.lower()
            for body in (body_1, body_2)
            for token in ("camera", "d435", "cup", "holder", "straw")
        )
        return {
            "body_1": body_1,
            "body_type_1": type_1,
            "body_2": body_2,
            "body_type_2": type_2,
            "depth_m": float(contact.depth),
            "position_m": [
                float(contact.position.x),
                float(contact.position.y),
                float(contact.position.z),
            ],
            "normal": [
                float(contact.normal.x),
                float(contact.normal.y),
                float(contact.normal.z),
            ],
            "pair": f"{body_1} <-> {body_2}",
            "self_collision": self_collision,
            "human_geometry_collision": human,
            "octomap_collision": octomap,
            "camera_cup_or_straw_collision": tool_component,
        }

    def _state_validity(
        self,
        robot_state: RobotState | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        base = {
            "available": False,
            "label": label,
            "valid": False,
            "contacts": [],
            "collision_pairs": [],
        }
        if robot_state is None:
            return {**base, "reason": "robot state is unavailable"}
        if not self.check_state_validity.wait_for_service(
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC
        ):
            return {
                **base,
                "reason": f"{STATE_VALIDITY_SERVICE} is unavailable",
            }
        request = GetStateValidity.Request()
        request.robot_state = robot_state
        # Candidate IK responses carry joint values but not the persistent
        # combined tool body.  A diff state keeps that body attached while
        # MoveIt checks the requested joints.
        request.robot_state.is_diff = True
        request.group_name = GROUP_NAME
        future = self.check_state_validity.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC,
        )
        response = future.result()
        if response is None:
            return {**base, "reason": "state-validity request timed out"}
        contacts = [
            self._collision_contact_report(contact)
            for contact in response.contacts
        ]
        pairs = sorted({contact["pair"] for contact in contacts})
        classifications: list[str] = []
        if any(contact["self_collision"] for contact in contacts):
            classifications.append("SELF_COLLISION")
        if any(contact["human_geometry_collision"] for contact in contacts):
            classifications.append("HUMAN_SAFETY_GEOMETRY_COLLISION")
        if any(contact["octomap_collision"] for contact in contacts):
            classifications.append("OCTOMAP_COLLISION")
        if any(contact["camera_cup_or_straw_collision"] for contact in contacts):
            classifications.append("CAMERA_CUP_OR_STRAW_COLLISION")
        if not response.valid and not classifications:
            classifications.append("INVALID_WITHOUT_REPORTED_CONTACT")
        if response.valid:
            classifications.append("VALID")
        return {
            **base,
            "available": True,
            "valid": bool(response.valid),
            "contacts": contacts,
            "collision_pairs": pairs,
            "classifications": classifications,
            "constraint_results": [
                {
                    "satisfied": bool(item.result),
                    "distance": float(item.distance),
                }
                for item in response.constraint_result
            ],
            "reason": None
            if response.valid
            else (
                "; ".join(pairs)
                if pairs
                else "MoveIt reported the state invalid without contact details"
            ),
        }

    def _solve_candidate_ik(self, target: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "available": False,
            "success": False,
            "avoid_collisions_during_ik": False,
            "ik_solver_modified": False,
        }
        seed = self._current_robot_state()
        if seed is None:
            return {**base, "reason": "complete IK seed state is unavailable"}
        if not self.compute_ik.wait_for_service(
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC
        ):
            return {**base, "reason": f"{IK_SERVICE} is unavailable"}
        request = GetPositionIK.Request()
        request.ik_request.group_name = GROUP_NAME
        request.ik_request.robot_state = seed
        request.ik_request.avoid_collisions = False
        request.ik_request.ik_link_name = TOOL_FRAME
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = BASE_FRAME
        request.ik_request.pose_stamped.pose.position.x = float(
            target["position_m"][0]
        )
        request.ik_request.pose_stamped.pose.position.y = float(
            target["position_m"][1]
        )
        request.ik_request.pose_stamped.pose.position.z = float(
            target["position_m"][2]
        )
        (
            request.ik_request.pose_stamped.pose.orientation.x,
            request.ik_request.pose_stamped.pose.orientation.y,
            request.ik_request.pose_stamped.pose.orientation.z,
            request.ik_request.pose_stamped.pose.orientation.w,
        ) = (float(value) for value in target["orientation_quat_xyzw"])
        request.ik_request.timeout = Duration(seconds=IK_TIMEOUT_SEC).to_msg()
        future = self.compute_ik.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC,
        )
        response = future.result()
        if response is None:
            return {**base, "reason": "IK request timed out"}
        success = int(response.error_code.val) == 1
        result: dict[str, Any] = {
            **base,
            "available": True,
            "success": success,
            "error_code": int(response.error_code.val),
            "error_message": str(response.error_code.message),
        }
        if not success:
            result["reason"] = response.error_code.message or (
                f"IK failed with error code {int(response.error_code.val)}"
            )
            return result
        result["robot_state"] = response.solution
        result["joint_state"] = {
            "names": list(response.solution.joint_state.name),
            "positions": [
                float(value) for value in response.solution.joint_state.position
            ],
        }
        return result

    def _fk_positions(
        self,
        robot_state: RobotState,
        link_names: tuple[str, ...] = MONITORED_ROBOT_LINKS,
    ) -> dict[str, Any]:
        base = {"available": False, "requested_links": list(link_names)}
        if not self.compute_fk.wait_for_service(
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC
        ):
            return {**base, "reason": f"{FK_SERVICE} is unavailable"}
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_FRAME
        request.fk_link_names = list(link_names)
        request.robot_state = robot_state
        future = self.compute_fk.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC,
        )
        response = future.result()
        if response is None:
            return {**base, "reason": "FK request timed out"}
        if int(response.error_code.val) != 1:
            return {
                **base,
                "reason": f"FK failed with error code {int(response.error_code.val)}",
                "error_code": int(response.error_code.val),
            }
        poses: dict[str, Any] = {}
        for name, stamped in zip(response.fk_link_names, response.pose_stamped):
            poses[str(name)] = {
                "position_m": [
                    float(stamped.pose.position.x),
                    float(stamped.pose.position.y),
                    float(stamped.pose.position.z),
                ],
                "orientation_quat_xyzw": [
                    float(stamped.pose.orientation.x),
                    float(stamped.pose.orientation.y),
                    float(stamped.pose.orientation.z),
                    float(stamped.pose.orientation.w),
                ],
            }
        return {
            **base,
            "available": True,
            "poses": poses,
            "missing_links": sorted(set(link_names) - set(poses)),
        }

    def _planning_scene_geometry(self) -> tuple[dict[str, Any], list[Any]]:
        base: dict[str, Any] = {
            "available": False,
            "human_collision_objects_preserved": False,
            "allowed_collision_exceptions_added": False,
        }
        if not self.get_planning_scene.wait_for_service(
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC
        ):
            return ({**base, "reason": f"{PLANNING_SCENE_SERVICE} is unavailable"}, [])
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.OCTOMAP
            | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            | PlanningSceneComponents.LINK_PADDING_AND_SCALING
        )
        future = self.get_planning_scene.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC,
        )
        response = future.result()
        if response is None:
            return ({**base, "reason": "PlanningScene request timed out"}, [])
        objects = list(response.scene.world.collision_objects)
        serialized: list[dict[str, Any]] = []
        primitive_names = {
            SolidPrimitive.BOX: "box",
            SolidPrimitive.SPHERE: "sphere",
            SolidPrimitive.CYLINDER: "cylinder",
            SolidPrimitive.CONE: "cone",
        }
        for collision in objects:
            primitives: list[dict[str, Any]] = []
            for primitive, pose in zip(collision.primitives, collision.primitive_poses):
                primitives.append(
                    {
                        "type": primitive_names.get(int(primitive.type), str(int(primitive.type))),
                        "dimensions_m": [float(value) for value in primitive.dimensions],
                        "pose": {
                            "position_m": [
                                float(pose.position.x),
                                float(pose.position.y),
                                float(pose.position.z),
                            ],
                            "orientation_quat_xyzw": [
                                float(pose.orientation.x),
                                float(pose.orientation.y),
                                float(pose.orientation.z),
                                float(pose.orientation.w),
                            ],
                        },
                    }
                )
            serialized.append(
                {
                    "id": str(collision.id),
                    "frame_id": str(collision.header.frame_id),
                    "object_pose": {
                        "position_m": [
                            float(collision.pose.position.x),
                            float(collision.pose.position.y),
                            float(collision.pose.position.z),
                        ],
                        "orientation_quat_xyzw": [
                            float(collision.pose.orientation.x),
                            float(collision.pose.orientation.y),
                            float(collision.pose.orientation.z),
                            float(collision.pose.orientation.w),
                        ],
                    },
                    "primitives": primitives,
                    "mesh_count": len(collision.meshes),
                }
            )
        human_ids = sorted(
            item["id"]
            for item in serialized
            if item["id"].startswith(HUMAN_OBJECT_PREFIXES)
        )
        attached_objects = list(
            response.scene.robot_state.attached_collision_objects
        )
        attached_serialized = [
            {
                "id": str(item.object.id),
                "link_name": str(item.link_name),
                "frame_id": str(item.object.header.frame_id),
                "touch_links": [str(value) for value in item.touch_links],
                "primitive_count": len(item.object.primitives),
                "mesh_count": len(item.object.meshes),
            }
            for item in attached_objects
        ]
        combined_tool_geometry = combined_tool_collision_verification(
            attached_objects
        )
        acm = response.scene.allowed_collision_matrix
        human_allowed_pairs: list[str] = []
        for row_index, name in enumerate(acm.entry_names):
            if row_index >= len(acm.entry_values):
                continue
            for column_index, enabled in enumerate(acm.entry_values[row_index].enabled):
                if not enabled or column_index >= len(acm.entry_names):
                    continue
                other = acm.entry_names[column_index]
                if str(name).startswith(HUMAN_OBJECT_PREFIXES) or str(other).startswith(
                    HUMAN_OBJECT_PREFIXES
                ):
                    human_allowed_pairs.append(f"{name} <-> {other}")
        octomap = response.scene.world.octomap.octomap
        report = {
            **base,
            "available": True,
            "world_collision_objects": serialized,
            "attached_collision_objects": attached_serialized,
            "combined_tool_collision_geometry": combined_tool_geometry,
            "human_object_ids": human_ids,
            "human_collision_objects_preserved": bool(human_ids),
            "human_allowed_collision_pairs": sorted(set(human_allowed_pairs)),
            "allowed_collision_exceptions_added": False,
            "octomap": {
                "frame_id": str(response.scene.world.octomap.header.frame_id),
                "binary": bool(octomap.binary),
                "id": str(octomap.id),
                "resolution_m": float(octomap.resolution),
                "serialized_byte_count": len(octomap.data),
                "present": bool(octomap.data),
            },
            "link_padding": [
                {"link_name": str(item.link_name), "padding_m": float(item.padding)}
                for item in response.scene.link_padding
            ],
            "link_scale": [
                {"link_name": str(item.link_name), "scale": float(item.scale)}
                for item in response.scene.link_scale
            ],
        }
        if human_allowed_pairs:
            report["reason"] = "human allowed-collision entries are present in the PlanningScene"
        return report, objects

    @staticmethod
    def _world_primitive_pose(collision: Any, primitive_pose: Pose) -> tuple[list[float], list[float]]:
        object_orientation = [
            float(collision.pose.orientation.x),
            float(collision.pose.orientation.y),
            float(collision.pose.orientation.z),
            float(collision.pose.orientation.w),
        ]
        if _norm(object_orientation) < 1e-9:
            object_orientation = [0.0, 0.0, 0.0, 1.0]
        local_position = [
            float(primitive_pose.position.x),
            float(primitive_pose.position.y),
            float(primitive_pose.position.z),
        ]
        world_position = _add(
            [
                float(collision.pose.position.x),
                float(collision.pose.position.y),
                float(collision.pose.position.z),
            ],
            _rotate_tool_vector(object_orientation, local_position),
        )
        primitive_orientation = [
            float(primitive_pose.orientation.x),
            float(primitive_pose.orientation.y),
            float(primitive_pose.orientation.z),
            float(primitive_pose.orientation.w),
        ]
        if _norm(primitive_orientation) < 1e-9:
            primitive_orientation = [0.0, 0.0, 0.0, 1.0]
        return world_position, _quaternion_multiply_xyzw(
            object_orientation,
            primitive_orientation,
        )

    @staticmethod
    def _point_primitive_signed_clearance(
        point: list[float],
        primitive: Any,
        center: list[float],
        orientation_xyzw: list[float],
    ) -> float | None:
        relative = _subtract(point, center)
        if int(primitive.type) == SolidPrimitive.SPHERE:
            return _norm(relative) - float(primitive.dimensions[0])
        if int(primitive.type) == SolidPrimitive.BOX:
            inverse = [
                -float(orientation_xyzw[0]),
                -float(orientation_xyzw[1]),
                -float(orientation_xyzw[2]),
                float(orientation_xyzw[3]),
            ]
            local = _rotate_tool_vector(inverse, relative)
            half = [float(value) / 2.0 for value in primitive.dimensions]
            delta = [abs(local[index]) - half[index] for index in range(3)]
            outside = _norm([max(value, 0.0) for value in delta])
            inside = min(max(delta), 0.0)
            return outside + inside
        return None

    def _candidate_feature_clearances(
        self,
        *,
        candidate: dict[str, Any],
        robot_state: RobotState,
        scene_objects: list[Any],
        snapshot: dict[str, Any],
        combined_tool_geometry: dict[str, Any],
    ) -> dict[str, Any]:
        fk = self._fk_positions(robot_state)
        features: dict[str, list[float]] = {
            "straw_tip": list(candidate["straw_tip_pose"]["position_m"]),
            "tool0": list(candidate["tool0_pose"]["position_m"]),
            # Keep the calibrated reference point in the clearance report in
            # addition to the exact attached-box collision check in MoveIt.
            "cup_holder_reference": list(candidate["tool0_pose"]["position_m"]),
        }
        if fk.get("available"):
            for link_name, pose in fk.get("poses", {}).items():
                features[link_name] = list(pose["position_m"])
        camera_local = snapshot.get("tool0_to_camera_optical_tf", {})
        if camera_local.get("available"):
            features["camera_optical_center"] = _add(
                list(candidate["tool0_pose"]["position_m"]),
                _rotate_tool_vector(
                    list(candidate["tool0_pose"]["orientation_quat_xyzw"]),
                    list(camera_local["position_m"]),
                ),
            )

        clearances: list[dict[str, Any]] = []
        for collision in scene_objects:
            object_id = str(collision.id)
            if not object_id.startswith(HUMAN_OBJECT_PREFIXES):
                continue
            for primitive_index, (primitive, primitive_pose) in enumerate(
                zip(collision.primitives, collision.primitive_poses)
            ):
                center, orientation = self._world_primitive_pose(
                    collision,
                    primitive_pose,
                )
                for feature_name, point in features.items():
                    clearance = self._point_primitive_signed_clearance(
                        point,
                        primitive,
                        center,
                        orientation,
                    )
                    if clearance is None:
                        continue
                    clearances.append(
                        {
                            "feature": feature_name,
                            "human_object": object_id,
                            "primitive_index": primitive_index,
                            "origin_or_point_to_object_surface_clearance_m": clearance,
                        }
                    )
        minimum = min(
            (
                item["origin_or_point_to_object_surface_clearance_m"]
                for item in clearances
            ),
            default=None,
        )
        combined_geometry_complete = bool(
            combined_tool_geometry.get("success")
            and combined_tool_geometry.get("real_execution_geometry_complete")
        )
        return {
            "available": bool(clearances),
            "minimum_origin_or_point_clearance_m": minimum,
            "all_origin_or_point_clearances_nonnegative": bool(
                clearances and minimum is not None and minimum >= 0.0
            ),
            "clearances": clearances,
            "feature_points_m": features,
            "fk": fk,
            "safety_margin_source": (
                "the conservative human head/torso/face keepout geometry itself; "
                "no allowed-collision exception is used"
            ),
            "tool_geometry_modeling": {
                "wrist_2_link_moveit_collision_geometry": True,
                "wrist_3_link_moveit_collision_geometry": True,
                "camera_moveit_collision_geometry": combined_geometry_complete,
                "cup_holder_moveit_collision_geometry": combined_geometry_complete,
                "straw_moveit_collision_geometry": combined_geometry_complete,
                "combined_attached_collision_object_id": COMBINED_TOOL_COLLISION_OBJECT_ID,
                "combined_attached_box_verified": combined_geometry_complete,
                "combined_attached_box_collision_checked_by_moveit": combined_geometry_complete,
                "combined_attached_box_specification": combined_tool_geometry,
                "camera_point_clearance_checked": "camera_optical_center" in features,
                "cup_holder_reference_clearance_checked": True,
                "straw_tip_point_clearance_checked": True,
                "real_execution_geometry_complete": combined_geometry_complete,
                "reason": (
                    None
                    if combined_geometry_complete
                    else combined_tool_geometry.get("reason")
                    or "combined camera/cup-holder/straw attached collision box is unavailable"
                ),
            },
        }

    def _candidate_joint_motion(self, robot_state: RobotState) -> float | None:
        if self.latest_joint_state is None:
            return None
        current = dict(
            zip(self.latest_joint_state.name, self.latest_joint_state.position)
        )
        target = dict(zip(robot_state.joint_state.name, robot_state.joint_state.position))
        if not all(name in current and name in target for name in EXPECTED_JOINTS):
            return None
        travel = 0.0
        for name in EXPECTED_JOINTS:
            difference = float(target[name]) - float(current[name])
            difference = math.atan2(math.sin(difference), math.cos(difference))
            travel += abs(difference)
        return travel

    @staticmethod
    def _verify_human_geometry(
        scene_report: dict[str, Any],
        planning_scene_application: dict[str, Any],
    ) -> dict[str, Any]:
        actual_by_id = {
            item["id"]: item
            for item in scene_report.get("world_collision_objects", [])
            if isinstance(item, dict)
        }
        config = planning_scene_application.get("config", {})
        findings: list[dict[str, Any]] = []
        expected_specs: list[tuple[str, str, list[float], list[float]]] = []
        for person in planning_scene_application.get("people", []):
            ids = person.get("object_ids", [])
            if len(ids) != 3:
                continue
            expected_specs.extend(
                [
                    (
                        str(ids[0]),
                        "sphere",
                        [float(config.get("head_radius_m", 0.0))],
                        [float(value) for value in person.get("head_center", [])],
                    ),
                    (
                        str(ids[1]),
                        "box",
                        [float(value) for value in config.get("torso_size_m", [])],
                        [float(value) for value in person.get("torso_center", [])],
                    ),
                    (
                        str(ids[2]),
                        "sphere",
                        [float(config.get("face_safety_radius_m", 0.0))],
                        [
                            float(value)
                            for value in person.get("face_safety_center", [])
                        ],
                    ),
                ]
            )
        success = bool(expected_specs)
        for object_id, shape, dimensions, center in expected_specs:
            actual = actual_by_id.get(object_id)
            primitive = (
                actual.get("primitives", [None])[0]
                if isinstance(actual, dict) and actual.get("primitives")
                else None
            )
            actual_frame = actual.get("frame_id") if actual else None
            # MoveIt canonicalizes collision objects submitted in base_link to
            # its fixed planning frame, world.  The UR description connects
            # world -> base_link by the verified identity fixed joint.
            frame_ok = bool(
                actual
                and actual_frame in (BASE_FRAME, MOVEIT_PLANNING_FRAME)
            )
            shape_ok = bool(primitive and primitive.get("type") == shape)
            dimensions_ok = bool(
                primitive
                and len(primitive.get("dimensions_m", [])) == len(dimensions)
                and all(
                    abs(float(measured) - float(expected)) <= 1e-6
                    for measured, expected in zip(
                        primitive.get("dimensions_m", []), dimensions
                    )
                )
            )
            actual_center: list[float] = []
            if primitive and actual:
                object_pose = actual.get("object_pose", {})
                object_position = object_pose.get("position_m", [])
                object_orientation = object_pose.get(
                    "orientation_quat_xyzw", []
                )
                primitive_position = primitive.get("pose", {}).get(
                    "position_m", []
                )
                if (
                    len(object_position) == 3
                    and len(object_orientation) == 4
                    and len(primitive_position) == 3
                ):
                    actual_center = _add(
                        [float(value) for value in object_position],
                        _rotate_tool_vector(
                            [float(value) for value in object_orientation],
                            [float(value) for value in primitive_position],
                        ),
                    )
            pose_ok = bool(
                len(actual_center) == len(center) == 3
                and all(
                    abs(float(measured) - float(expected)) <= 1e-6
                    for measured, expected in zip(actual_center, center)
                )
            )
            valid = frame_ok and shape_ok and dimensions_ok and pose_ok
            success = success and valid
            findings.append(
                {
                    "object_id": object_id,
                    "valid": valid,
                    "input_frame_id": BASE_FRAME,
                    "accepted_scene_frame_ids": [
                        BASE_FRAME,
                        MOVEIT_PLANNING_FRAME,
                    ],
                    "actual_frame_id": actual_frame,
                    "moveit_canonicalized_to_planning_frame": bool(
                        actual_frame == MOVEIT_PLANNING_FRAME
                    ),
                    "expected_shape": shape,
                    "actual_shape": primitive.get("type") if primitive else None,
                    "expected_dimensions_m": dimensions,
                    "actual_dimensions_m": primitive.get("dimensions_m") if primitive else None,
                    "expected_center_m": center,
                    "actual_center_m": actual_center,
                    "frame_valid": frame_ok,
                    "shape_valid": shape_ok,
                    "dimensions_valid": dimensions_ok,
                    "pose_valid": pose_ok,
                }
            )
        return {
            "success": success,
            "findings": findings,
            "configured_safety_geometry": config,
            "geometry_was_shrunk_or_deleted": False,
            "safety_margin_policy": (
                "head and face radii plus torso dimensions are retained exactly; "
                "candidate straw-tip and robot states must remain outside them"
            ),
        }

    def _select_adaptive_premouth_goal(
        self,
        *,
        mouth: list[float],
        original_pre_mouth: list[float],
        snapshot: dict[str, Any],
        planning_scene_application: dict[str, Any],
    ) -> dict[str, Any]:
        approach = _subtract(original_pre_mouth, mouth)
        approach_norm = _norm(approach)
        if approach_norm < 1e-9:
            return {
                "success": False,
                "stage": "adaptive_goal_generation",
                "reason": "validated pre-mouth approach line is degenerate",
                "candidates": [],
            }
        approach_unit = [value / approach_norm for value in approach]
        current_tool0 = snapshot["tool0_pose"]
        candidates = _adaptive_premouth_pose_candidates(
            mouth_position_m=mouth,
            approach_offset_unit=approach_unit,
            verified_flange_down_orientation_xyzw=list(
                current_tool0["orientation_quat_xyzw"]
            ),
        )
        scene_report, scene_objects = self._planning_scene_geometry()
        human_geometry_verification = self._verify_human_geometry(
            scene_report,
            planning_scene_application,
        )
        start_state_validity = self._state_validity(
            self._current_robot_state(),
            label="current_start_state",
        )
        result: dict[str, Any] = {
            "success": False,
            "stage": "adaptive_goal_selection",
            "approach_policy": self.premouth_policy,
            "approach_offset_unit": approach_unit,
            "standoff_candidates_m": list(ADAPTIVE_PREMOUTH_STANDOFFS_M),
            "yaw_candidates_deg": list(ADAPTIVE_PREMOUTH_YAWS_DEG),
            "candidate_count": len(candidates),
            "start_state_validity": start_state_validity,
            "planning_scene": scene_report,
            "human_geometry_verification": human_geometry_verification,
            "planning_scene_application": planning_scene_application,
            "candidates": [],
            "execution_sent": False,
        }
        if not scene_report.get("available"):
            result["reason"] = scene_report.get("reason")
            return result
        if not scene_report.get("human_collision_objects_preserved"):
            result["reason"] = "fixed human collision geometry is absent"
            return result
        if scene_report.get("human_allowed_collision_pairs"):
            result["reason"] = "human allowed-collision exceptions are present"
            return result
        if not human_geometry_verification.get("success"):
            result["reason"] = "fixed human collision geometry failed frame/pose/dimension verification"
            return result
        if not start_state_validity.get("available") or not start_state_validity.get(
            "valid"
        ):
            result["reason"] = (
                "current start state is invalid: "
                + str(start_state_validity.get("reason") or "unknown reason")
            )
            return result

        valid_candidates: list[tuple[tuple[float, float, float], dict[str, Any], RobotState]] = []
        current_position = list(current_tool0["position_m"])
        for candidate in candidates:
            report = dict(candidate)
            target = candidate["tool0_pose"]
            target_in_ur_base = self._point_in_ur_base(
                list(target["position_m"]),
                snapshot["ur_base_tf"],
            )
            radius = _norm(target_in_ur_base)
            path_length = _norm(_subtract(list(target["position_m"]), current_position))
            workspace_valid = bool(
                radius <= MAX_TOOL0_RADIUS_FROM_UR_BASE_M
                and path_length <= self.maximum_plan_translation_m
                and candidate["flange_vertical_axis_error_rad"]
                <= MAX_TOOL_VERTICAL_TILT_RAD
                and candidate["straw_tip_reconstruction_error_m"] <= 1e-8
            )
            report.update(
                {
                    "workspace_valid": workspace_valid,
                    "target_tool0_position_in_ur_base_m": target_in_ur_base,
                    "target_tool0_radius_from_ur_base_m": radius,
                    "path_length_m": path_length,
                }
            )
            if not workspace_valid:
                report.update(
                    {
                        "valid": False,
                        "rejection_stage": "workspace",
                        "rejection_reason": (
                            "candidate violates reach, translation, flange-down, or "
                            "straw-tip reconstruction bounds"
                        ),
                    }
                )
                result["candidates"].append(report)
                continue

            ik = self._solve_candidate_ik(target)
            robot_state = ik.get("robot_state")
            report["ik"] = {
                key: value for key, value in ik.items() if key != "robot_state"
            }
            if not ik.get("success") or not isinstance(robot_state, RobotState):
                report.update(
                    {
                        "valid": False,
                        "rejection_stage": "ik",
                        "rejection_reason": ik.get("reason") or "no IK solution",
                    }
                )
                result["candidates"].append(report)
                continue

            validity = self._state_validity(
                robot_state,
                label=(
                    f"candidate_{candidate['candidate_index']}_"
                    f"{candidate['standoff_m']:.3f}m_{candidate['yaw_deg']:+.0f}deg"
                ),
            )
            clearance = self._candidate_feature_clearances(
                candidate=candidate,
                robot_state=robot_state,
                scene_objects=scene_objects,
                snapshot=snapshot,
                combined_tool_geometry=scene_report.get(
                    "combined_tool_collision_geometry", {}
                ),
            )
            joint_motion = self._candidate_joint_motion(robot_state)
            report.update(
                {
                    "final_state_validity": validity,
                    "clearance": clearance,
                    "joint_motion_l1_rad": joint_motion,
                }
            )
            safety_clearance = clearance.get(
                "minimum_origin_or_point_clearance_m"
            )
            valid = bool(
                validity.get("available")
                and validity.get("valid")
                and clearance.get("all_origin_or_point_clearances_nonnegative")
                and isinstance(safety_clearance, (int, float))
            )
            report["valid"] = valid
            if valid:
                report["rejection_stage"] = None
                report["rejection_reason"] = None
                score = (
                    float(safety_clearance),
                    -float(joint_motion if joint_motion is not None else float("inf")),
                    -path_length,
                )
                report["score"] = {
                    "safety_clearance_m": float(safety_clearance),
                    "joint_motion_l1_rad": joint_motion,
                    "path_length_m": path_length,
                    "priority_order": [
                        "maximum safety clearance",
                        "minimum joint motion",
                        "minimum path length",
                    ],
                }
                valid_candidates.append((score, report, robot_state))
            else:
                report["rejection_stage"] = "final_state_collision_or_clearance"
                report["rejection_reason"] = validity.get("reason") or (
                    "candidate violates a fixed human safety volume"
                )
            result["candidates"].append(report)

        if not valid_candidates:
            result["reason"] = "every adaptive pre-mouth candidate is invalid"
            result["all_collision_pairs"] = sorted(
                {
                    pair
                    for candidate in result["candidates"]
                    for pair in candidate.get("final_state_validity", {}).get(
                        "collision_pairs", []
                    )
                }
            )
            return result
        _, selected, selected_state = max(valid_candidates, key=lambda item: item[0])
        self._selected_goal_robot_state = selected_state
        self._selected_goal_diagnostic = selected
        result.update(
            {
                "success": True,
                "selected_candidate_index": selected["candidate_index"],
                "selected_candidate": selected,
                "selected_standoff_m": selected["standoff_m"],
                "selected_yaw_deg": selected["yaw_deg"],
                "selected_tool0_pose": selected["tool0_pose"],
                "selected_straw_tip_pose": selected["straw_tip_pose"],
                "real_execution_geometry_complete": selected["clearance"][
                    "tool_geometry_modeling"
                ]["real_execution_geometry_complete"],
                "reason": None,
            }
        )
        return result

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

    @staticmethod
    def _vertical_axis_constraint(target_pose: Pose) -> OrientationConstraint:
        """Constrain tilt while leaving rotation about tool0 +Z free."""
        reference = _vertical_reference_quaternion(
            [
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            ]
        )
        constraint = OrientationConstraint()
        constraint.header.frame_id = BASE_FRAME
        constraint.link_name = TOOL_FRAME
        (
            constraint.orientation.x,
            constraint.orientation.y,
            constraint.orientation.z,
            constraint.orientation.w,
        ) = reference
        constraint.absolute_x_axis_tolerance = VERTICAL_AXIS_COMPONENT_TOLERANCE_RAD
        constraint.absolute_y_axis_tolerance = VERTICAL_AXIS_COMPONENT_TOLERANCE_RAD
        constraint.absolute_z_axis_tolerance = FREE_TOOL_AXIS_SPIN_TOLERANCE_RAD
        constraint.parameterization = OrientationConstraint.ROTATION_VECTOR
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
            # Retain the attached combined tool body in the scene start state.
            goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(constraints)
        goal.request.path_constraints.name = "tool_vertical_axis_with_free_spin"
        goal.request.path_constraints.orientation_constraints.append(
            self._vertical_axis_constraint(pose)
        )
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _validate_trajectory_vertical_axis(
        self,
        trajectory: Any,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Use MoveIt FK to reject any trajectory waypoint that tilts the cup."""
        validation_deadline = min(
            deadline if deadline is not None else float("inf"),
            time.monotonic() + FK_TRAJECTORY_VALIDATION_TIMEOUT_SEC,
        )
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        names = list(getattr(joint_trajectory, "joint_names", []))
        points = list(getattr(joint_trajectory, "points", []))
        base = {
            "success": False,
            "stage": "trajectory_vertical_axis_validation",
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            "tool_axis_tool0": list(TOOL_VERTICAL_AXIS_TOOL0),
            "required_axis_base_link": list(REQUIRED_VERTICAL_AXIS_BASE_LINK),
            "maximum_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
            "maximum_tilt_deg": math.degrees(MAX_TOOL_VERTICAL_TILT_RAD),
            "maximum_tool0_excursion_from_start_m": None,
            "sampled_waypoints": 0,
            "trajectory_waypoints": len(points),
        }
        if not names or not points:
            return {**base, "reason": "planned trajectory has no joint waypoints"}
        if time.monotonic() >= validation_deadline:
            return {**base, "reason": "vertical-axis validation deadline expired"}
        if not self.compute_fk.wait_for_service(
            timeout_sec=min(
                FK_SERVICE_DISCOVERY_TIMEOUT_SEC,
                max(0.0, validation_deadline - time.monotonic()),
            )
        ):
            return {**base, "reason": f"{FK_SERVICE} is unavailable; refusing unvalidated trajectory"}

        maximum_tilt = -1.0
        maximum_index: int | None = None
        first_tool_position: list[float] | None = None
        final_tool_position: list[float] | None = None
        maximum_tool_excursion = 0.0
        maximum_tool_excursion_index: int | None = None
        for index, point in enumerate(points):
            positions = [float(value) for value in point.positions]
            if len(positions) != len(names) or not all(math.isfinite(value) for value in positions):
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": f"trajectory waypoint {index} has invalid joint positions",
                }
            remaining = validation_deadline - time.monotonic()
            if remaining <= 0.0:
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": "vertical-axis validation timed out before all waypoints were checked",
                }
            request = GetPositionFK.Request()
            request.header.frame_id = BASE_FRAME
            request.fk_link_names = [TOOL_FRAME]
            request.robot_state.joint_state.name = names
            request.robot_state.joint_state.position = positions
            request.robot_state.is_diff = False
            future = self.compute_fk.call_async(request)
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=min(FK_WAYPOINT_TIMEOUT_SEC, remaining),
            )
            response = future.result()
            if response is None:
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": f"FK timed out at trajectory waypoint {index}",
                }
            if int(response.error_code.val) != 1:
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": (
                        f"FK failed at trajectory waypoint {index} with error code "
                        f"{int(response.error_code.val)}"
                    ),
                }
            try:
                pose_index = list(response.fk_link_names).index(TOOL_FRAME)
                tool_pose = response.pose_stamped[pose_index].pose
                orientation = tool_pose.orientation
                quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
                tilt = _tool_vertical_tilt_rad(quaternion)
                tool_position = [
                    float(tool_pose.position.x),
                    float(tool_pose.position.y),
                    float(tool_pose.position.z),
                ]
                if not all(math.isfinite(value) for value in tool_position):
                    raise ValueError("tool0 FK position is not finite")
            except (IndexError, RuntimeError, TypeError, ValueError) as exc:
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": f"invalid FK result at trajectory waypoint {index}: {exc}",
                }
            if first_tool_position is None:
                first_tool_position = list(tool_position)
            final_tool_position = list(tool_position)
            excursion = math.sqrt(
                sum(
                    (value - start) ** 2
                    for value, start in zip(tool_position, first_tool_position)
                )
            )
            if excursion > maximum_tool_excursion:
                maximum_tool_excursion = excursion
                maximum_tool_excursion_index = index
            if tilt > maximum_tilt:
                maximum_tilt = tilt
                maximum_index = index
            if tilt > MAX_TOOL_VERTICAL_TILT_RAD:
                return {
                    **base,
                    "sampled_waypoints": index + 1,
                    "maximum_observed_tilt_rad": tilt,
                    "maximum_observed_tilt_deg": math.degrees(tilt),
                    "maximum_tilt_waypoint_index": index,
                    "maximum_tool0_excursion_from_start_m": maximum_tool_excursion,
                    "maximum_tool0_excursion_waypoint_index": maximum_tool_excursion_index,
                    "reason": (
                        f"trajectory waypoint {index} tilts tool0 +Z by "
                        f"{math.degrees(tilt):.2f} deg from base_link -Z, above the "
                        f"{math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
                    ),
                }
        return {
            **base,
            "success": True,
            "sampled_waypoints": len(points),
            "maximum_observed_tilt_rad": maximum_tilt,
            "maximum_observed_tilt_deg": math.degrees(maximum_tilt),
            "maximum_tilt_waypoint_index": maximum_index,
            "first_tool0_position_m": first_tool_position,
            "final_tool0_position_m": final_tool_position,
            "maximum_tool0_excursion_from_start_m": maximum_tool_excursion,
            "maximum_tool0_excursion_waypoint_index": maximum_tool_excursion_index,
            "wrist_3_joint_directly_commanded": False,
            "tool_axis_spin_free": True,
        }

    def _run_cartesian_plan(self, target: dict[str, Any]) -> dict[str, Any]:
        """Plan a complete, collision-checked Cartesian path without execution."""
        self._validated_trajectory = None
        base: dict[str, Any] = {
            "success": False,
            "stage": "cartesian_plan_only",
            "planner": "moveit_compute_cartesian_path",
            "execution_sent": False,
            "avoid_collisions": True,
            "max_step_m": CARTESIAN_MAX_STEP_M,
            "maximum_revolute_joint_jump_rad": CARTESIAN_MAX_REVOLUTE_JUMP_RAD,
            "required_fraction": CARTESIAN_COMPLETE_FRACTION,
            "velocity_scaling": self.trajectory_velocity_scaling,
            "acceleration_scaling": self.trajectory_acceleration_scaling,
        }
        start_state = self._current_robot_state()
        if start_state is None:
            return {**base, "reason": "Cartesian start state is unavailable"}
        if not self.compute_cartesian_path.wait_for_service(
            timeout_sec=CANDIDATE_SERVICE_TIMEOUT_SEC
        ):
            return {**base, "reason": f"{CARTESIAN_PATH_SERVICE} is unavailable"}
        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        request.start_state = start_state
        request.group_name = GROUP_NAME
        request.link_name = TOOL_FRAME
        waypoint = Pose()
        (
            waypoint.position.x,
            waypoint.position.y,
            waypoint.position.z,
        ) = (float(value) for value in target["position_m"])
        (
            waypoint.orientation.x,
            waypoint.orientation.y,
            waypoint.orientation.z,
            waypoint.orientation.w,
        ) = (float(value) for value in target["orientation_quat_xyzw"])
        request.waypoints = [waypoint]
        request.max_step = CARTESIAN_MAX_STEP_M
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = CARTESIAN_MAX_REVOLUTE_JUMP_RAD
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.trajectory_velocity_scaling
        request.max_acceleration_scaling_factor = self.trajectory_acceleration_scaling
        request.path_constraints.name = "cartesian_tool_vertical_axis_with_free_spin"
        request.path_constraints.orientation_constraints.append(
            self._vertical_axis_constraint(waypoint)
        )
        future = self.compute_cartesian_path.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=ACTION_TIMEOUT_SEC,
        )
        response = future.result()
        if response is None:
            return {**base, "reason": "Cartesian planning request timed out"}
        summary = _trajectory_summary(response.solution)
        complete = bool(
            int(response.error_code.val) == 1
            and float(response.fraction) >= CARTESIAN_COMPLETE_FRACTION
            and int(summary.get("points", 0)) >= 2
        )
        vertical_axis_validation = None
        if complete:
            vertical_axis_validation = self._validate_trajectory_vertical_axis(
                response.solution
            )
            complete = bool(vertical_axis_validation.get("success"))
            if complete:
                self._validated_trajectory = response.solution
        return {
            **base,
            "success": complete,
            "error_code": int(response.error_code.val),
            "error_message": str(response.error_code.message),
            "fraction": float(response.fraction),
            "complete": float(response.fraction) >= CARTESIAN_COMPLETE_FRACTION,
            "planned_trajectory": summary,
            "vertical_axis_validation": vertical_axis_validation,
            "reason": None
            if complete
            else (
                vertical_axis_validation.get("reason")
                if isinstance(vertical_axis_validation, dict)
                else (
                    "Cartesian path is incomplete or collision-blocked: "
                    f"fraction={float(response.fraction):.6f}, "
                    f"error_code={int(response.error_code.val)}"
                )
            ),
        }

    def _run_plan(self, target: dict[str, Any]) -> dict[str, Any]:
        return self._run_cartesian_plan(target)

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

    def _prepare_dynamic_scene_for_goal_selection(
        self,
        *,
        mouth: list[float],
        original_pre_mouth: list[float],
        snapshot: dict[str, Any],
        planning_scene_application: dict[str, Any],
    ) -> dict[str, Any]:
        """Static-scene default; the integrated OctoMap workflow overrides it."""
        return {
            "success": True,
            "dynamic_octomap_enabled": False,
            "octomap_clear_attempted": False,
            "human_collision_objects_preserved": bool(
                planning_scene_application.get("success")
            ),
            "final_state_validity_changed_after_rebuild": None,
            "note": "No dynamic OctoMap layer is managed by this static-scene pipeline.",
        }

    def diagnose_frozen_mouth_static_scene(
        self,
        mouth_position_m: list[float],
        *,
        rebuild_dynamic_octomap: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """Replay one recorded mouth point against the live state without motion.

        This diagnostic exists for reproducing a failed real plan when the
        person is no longer visible.  It consumes the current joint state,
        current corrected camera TF, current calibrated tool transforms, and a
        caller-supplied recorded mouth point.  The default refuses to proceed
        if a dynamic OctoMap is present.  The integrated subclass may instead
        request its stationary dynamic-layer rebuild.  Neither route creates
        an execution client.
        """
        try:
            mouth = [float(value) for value in mouth_position_m]
        except (TypeError, ValueError):
            mouth = []
        if len(mouth) != 3 or not all(math.isfinite(value) for value in mouth):
            return 2, {
                "success": False,
                "mode": "diagnose-frozen",
                "stage": "diagnostic_input",
                "reason": "recorded mouth position must contain three finite base_link values",
                "execution_sent": False,
                "execution_disabled": True,
            }

        snapshot = self.snapshot(
            mouth_sample_sec=MIN_MOUTH_SAMPLE_SECONDS,
            inspect_controllers=False,
        )
        snapshot["live_mouth_pose_before_diagnostic_override"] = snapshot.get(
            "mouth_pose"
        )
        snapshot["mouth_pose"] = {
            "available": True,
            "stable": True,
            "diagnostic_recorded_input": True,
            "frame_id": BASE_FRAME,
            "mean_position_m": mouth,
            "selected_candidate_index": 0,
            "visible_candidates": [{"position_m": mouth}],
        }
        failures = self.readiness_failures(
            snapshot,
            require_stable_mouth=False,
        )
        if failures:
            return 2, {
                "success": False,
                "mode": "diagnose-frozen",
                "stage": "readiness",
                "failures": failures,
                "checks": snapshot,
                "recorded_mouth_position_m": mouth,
                "execution_sent": False,
                "execution_disabled": True,
            }

        planning_scene_application = self._apply_multi_person_planning_scene(snapshot)
        snapshot["planning_scene_obstacles"] = planning_scene_application
        if not planning_scene_application.get("success"):
            return 2, {
                "success": False,
                "mode": "diagnose-frozen",
                "stage": "planning_scene_obstacles",
                "reason": planning_scene_application.get("reason"),
                "checks": snapshot,
                "recorded_mouth_position_m": mouth,
                "execution_sent": False,
                "execution_disabled": True,
            }

        candidates, approach_details = self._pre_mouth_candidates(
            mouth,
            snapshot["camera_tf"],
            snapshot["tool0_pose"],
        )
        original_pre_mouth = candidates.get(self.premouth_policy)
        if not isinstance(original_pre_mouth, list):
            return 2, {
                "success": False,
                "mode": "diagnose-frozen",
                "stage": "validated_approach_policy",
                "reason": f"{self.premouth_policy} did not produce a target",
                "approach_details": approach_details,
                "execution_sent": False,
                "execution_disabled": True,
            }

        dynamic_scene_preparation = (
            self._prepare_dynamic_scene_for_goal_selection(
                mouth=mouth,
                original_pre_mouth=original_pre_mouth,
                snapshot=snapshot,
                planning_scene_application=planning_scene_application,
            )
            if rebuild_dynamic_octomap
            else {
                "success": True,
                "dynamic_octomap_enabled": False,
                "octomap_clear_attempted": False,
                "note": "Static diagnostic requested; dynamic scene was not modified.",
            }
        )
        if not dynamic_scene_preparation.get("success"):
            return 2, {
                "success": False,
                "mode": "diagnose-frozen",
                "stage": "dynamic_scene_preparation",
                "reason": dynamic_scene_preparation.get("reason"),
                "recorded_mouth_position_m": mouth,
                "dynamic_scene_preparation": dynamic_scene_preparation,
                "planning_scene_application": planning_scene_application,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }

        adaptive = self._select_adaptive_premouth_goal(
            mouth=mouth,
            original_pre_mouth=original_pre_mouth,
            snapshot=snapshot,
            planning_scene_application=planning_scene_application,
        )
        octomap = adaptive.get("planning_scene", {}).get("octomap", {})
        octomap_disabled = bool(
            adaptive.get("planning_scene", {}).get("available")
            and not octomap.get("present")
        )
        collision_contacts = [
            contact
            for validity in [adaptive.get("start_state_validity", {})]
            + [
                item.get("final_state_validity", {})
                for item in adaptive.get("candidates", [])
            ]
            for contact in validity.get("contacts", [])
        ]
        attribution = {
            "self_collision_pairs": sorted(
                {item["pair"] for item in collision_contacts if item.get("self_collision")}
            ),
            "human_geometry_collision_pairs": sorted(
                {
                    item["pair"]
                    for item in collision_contacts
                    if item.get("human_geometry_collision")
                }
            ),
            "camera_cup_or_straw_collision_pairs": sorted(
                {
                    item["pair"]
                    for item in collision_contacts
                    if item.get("camera_cup_or_straw_collision")
                }
            ),
            "octomap_collision_pairs": sorted(
                {item["pair"] for item in collision_contacts if item.get("octomap_collision")}
            ),
        }
        exact_pairs = sorted(
            {item["pair"] for item in collision_contacts if item.get("pair")}
        )
        response: dict[str, Any] = {
            "success": False,
            "mode": "diagnose-frozen",
            "stage": "static_scene_diagnostic",
            "recorded_mouth_position_m": mouth,
            "recorded_mouth_frame_id": BASE_FRAME,
            "recorded_input_used_instead_of_live_perception": True,
            "current_start_state_validity": adaptive.get("start_state_validity"),
            "adaptive_goal_selection": adaptive,
            "planning_scene_application": planning_scene_application,
            "human_geometry_verification": adaptive.get(
                "human_geometry_verification"
            ),
            "dynamic_octomap_disabled": octomap_disabled,
            "dynamic_octomap_rebuild_requested": rebuild_dynamic_octomap,
            "dynamic_scene_preparation": dynamic_scene_preparation,
            "exact_collision_pairs_found": exact_pairs,
            "invalidity_attribution": attribution,
            "checks": snapshot,
            "execution_sent": False,
            "execution_disabled": True,
        }
        if not rebuild_dynamic_octomap and not octomap_disabled:
            response["reason"] = (
                "dynamic OctoMap is present; restart MoveGroup with use_octomap:=false "
                "before the static-scene diagnostic"
            )
            return 2, response
        if not adaptive.get("success"):
            response["reason"] = adaptive.get("reason")
            return 2, response

        target = adaptive["selected_tool0_pose"]
        plan_result = self._run_plan(
            {
                "frame_id": BASE_FRAME,
                "link_name": TOOL_FRAME,
                "position_m": list(target["position_m"]),
                "orientation_quat_xyzw": list(target["orientation_quat_xyzw"]),
            }
        )
        response.update(
            {
                "success": bool(plan_result.get("success")),
                "stage": (
                    "rebuilt_octomap_route_plan_only"
                    if rebuild_dynamic_octomap
                    else "static_scene_cartesian_plan_only"
                ),
                "reason": plan_result.get("reason"),
                "selected_candidate": adaptive.get("selected_candidate"),
                "selected_standoff_m": adaptive.get("selected_standoff_m"),
                "selected_yaw_deg": adaptive.get("selected_yaw_deg"),
                "final_tool0_pose": adaptive.get("selected_tool0_pose"),
                "straw_tip_pose": adaptive.get("selected_straw_tip_pose"),
                "flange_vertical_axis_error_rad": adaptive.get(
                    "selected_candidate", {}
                ).get("flange_vertical_axis_error_rad"),
                "clearance": adaptive.get("selected_candidate", {}).get(
                    "clearance"
                ),
                "cartesian_plan_result": plan_result,
                "cartesian_planning_succeeded": bool(
                    plan_result.get("direct_path_accepted", plan_result.get("success"))
                ),
                "ompl_needed": bool(plan_result.get("detour_attempted")),
            }
        )
        return (0 if response["success"] else 2), response

    def plan(self) -> tuple[int, dict[str, Any]]:
        # Keep planning independent of controller-manager service calls.  The
        # plan path only needs live state, TF, perception, and /move_action;
        # controller state is rechecked later, immediately before an execute.
        snapshot = self.snapshot(mouth_sample_sec=self.mouth_sample_seconds, inspect_controllers=False)
        failures = self.readiness_failures(snapshot, require_stable_mouth=True)
        if failures:
            return 2, {"success": False, "mode": "plan", "stage": "readiness", "failures": failures, "checks": snapshot, "execution_sent": False}

        planning_scene = self._apply_multi_person_planning_scene(snapshot)
        snapshot["planning_scene_obstacles"] = planning_scene
        if not planning_scene.get("success"):
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "planning_scene_obstacles",
                "reason": str(planning_scene.get("reason") or "deterministic multi-person PlanningScene update failed"),
                "target_selection": self.target_selection,
                "checks": snapshot,
                "execution_sent": False,
            }

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

        dynamic_scene_preparation = self._prepare_dynamic_scene_for_goal_selection(
            mouth=mouth,
            original_pre_mouth=pre_mouth,
            snapshot=snapshot,
            planning_scene_application=planning_scene,
        )
        if not dynamic_scene_preparation.get("success"):
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "stationary_octomap_rebuild",
                "reason": dynamic_scene_preparation.get("reason"),
                "dynamic_scene_preparation": dynamic_scene_preparation,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }

        adaptive_goal_selection = self._select_adaptive_premouth_goal(
            mouth=mouth,
            original_pre_mouth=pre_mouth,
            snapshot=snapshot,
            planning_scene_application=planning_scene,
        )
        if not adaptive_goal_selection.get("success"):
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "adaptive_goal_selection",
                "reason": adaptive_goal_selection.get("reason"),
                "premouth_policy": self.premouth_policy,
                "selected_policy_candidate": selected_candidate,
                "detected_mouth_pose": {
                    "frame_id": BASE_FRAME,
                    "position_m": mouth,
                },
                "original_50mm_pre_mouth_pose": {
                    "frame_id": BASE_FRAME,
                    "position_m": pre_mouth,
                },
                "adaptive_goal_selection": adaptive_goal_selection,
                "checks": snapshot,
                "execution_sent": False,
                "execution_disabled": True,
            }
        selected_goal = adaptive_goal_selection["selected_candidate"]
        pre_mouth = list(selected_goal["straw_tip_pose"]["position_m"])
        orientation = list(selected_goal["tool0_pose"]["orientation_quat_xyzw"])
        target_tool0_position = list(selected_goal["tool0_pose"]["position_m"])
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
        # Keep every collision-free adaptive goal available to the dynamic
        # backend.  A highest-clearance endpoint can still have a blocked
        # intermediate route, while another validated yaw/standoff may be
        # reachable without relaxing any safety constraint.
        self._route_candidate_targets = [
            {
                "frame_id": BASE_FRAME,
                "link_name": TOOL_FRAME,
                "position_m": list(item["tool0_pose"]["position_m"]),
                "orientation_quat_xyzw": list(item["tool0_pose"]["orientation_quat_xyzw"]),
                "standoff_m": item.get("standoff_m"),
                "yaw_deg": item.get("yaw_deg"),
            }
            for item in adaptive_goal_selection.get("candidates", [])
            if item.get("valid") is True and isinstance(item.get("tool0_pose"), dict)
        ]
        plan_result = self._run_plan(target)
        response = {
            "success": bool(plan_result.get("success")),
            "mode": "plan",
            "premouth_policy": self.premouth_policy,
            "target_selection": self.target_selection,
            "selected_candidate_index": snapshot["mouth_pose"].get("selected_candidate_index"),
            "safe_distance_m": adaptive_goal_selection["selected_standoff_m"],
            "requested_preferred_safe_distance_m": self.safe_distance_m,
            "selected_tool_yaw_deg": adaptive_goal_selection["selected_yaw_deg"],
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
            "final_straw_tip_to_detected_mouth_standoff_m": adaptive_goal_selection[
                "selected_standoff_m"
            ],
            "orientation_difference_rad": _quaternion_distance_rad(
                list(current_tool0["orientation_quat_xyzw"]),
                orientation,
            ),
            "orientation_preserved": abs(
                float(adaptive_goal_selection["selected_yaw_deg"])
            ) < 1e-9,
            "adaptive_goal_selection": adaptive_goal_selection,
            "dynamic_scene_preparation": dynamic_scene_preparation,
            "flange_vertical_axis_error_rad": selected_goal[
                "flange_vertical_axis_error_rad"
            ],
            "flange_vertical_axis_error_deg": selected_goal[
                "flange_vertical_axis_error_deg"
            ],
            "selected_candidate_clearance": selected_goal["clearance"],
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
        if self.target_selection != "center":
            guards.append("guarded real execution supports only the center mouth target")
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
        vertical_error = prepared.get("flange_vertical_axis_error_rad")
        if not isinstance(vertical_error, (int, float)) or float(
            vertical_error
        ) > MAX_TOOL_VERTICAL_TILT_RAD:
            guards.append(
                "selected target does not preserve the verified flange-down axis"
            )
        modeling = prepared.get("selected_candidate_clearance", {}).get(
            "tool_geometry_modeling", {}
        )
        if not modeling.get("real_execution_geometry_complete"):
            guards.append(
                modeling.get("reason")
                or "camera, cup-holder, and straw collision geometry is incomplete"
            )
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
        vertical_axis_validation = self._validate_trajectory_vertical_axis(
            self._validated_trajectory
        )
        if not vertical_axis_validation.get("success"):
            return {
                "success": False,
                "stage": "pre_execution_vertical_axis_validation",
                "reason": vertical_axis_validation.get("reason"),
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
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
            "vertical_axis_validation": vertical_axis_validation,
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
        if self.target_selection != "center":
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "target_selection_execution_gate",
                "reason": "guarded real execution supports only the center mouth target",
                "target_selection": self.target_selection,
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
        perception_state = self.target_tracker.current_state(max_age_sec=MAX_MOUTH_POSE_AGE_SEC)
        prepared["pre_execution_perception_state"] = perception_state
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
        if not perception_state.get("available"):
            late_guards.append(
                "selected mouth target is unavailable immediately before execution: "
                f"{perception_state.get('reason', 'unknown perception failure')}"
            )
        else:
            frozen_target = _finite_xyz(
                prepared.get("frozen_detected_mouth_pose", {}).get("position_m")
                if isinstance(prepared.get("frozen_detected_mouth_pose"), dict)
                else None
            )
            current_target = _finite_xyz(perception_state.get("selected_position_m"))
            if frozen_target is None or current_target is None:
                late_guards.append("selected mouth target coordinates are invalid immediately before execution")
            else:
                target_drift = _norm(_subtract(current_target, frozen_target))
                prepared["pre_execution_target_drift_m"] = target_drift
                if target_drift > MAX_PRE_EXECUTION_TARGET_DRIFT_M:
                    late_guards.append(
                        f"selected mouth target moved {target_drift:.4f} m after planning, above the "
                        f"{MAX_PRE_EXECUTION_TARGET_DRIFT_M:.4f} m limit"
                    )

            frozen_candidates = prepared.get("checks", {}).get(
                "planning_scene_obstacles", {}
            ).get("frozen_candidate_positions_m")
            current_candidates = perception_state.get("visible_candidates")
            if not isinstance(frozen_candidates, list) or not isinstance(current_candidates, list):
                late_guards.append("multi-person obstacle coordinates are unavailable immediately before execution")
            elif len(frozen_candidates) != len(current_candidates):
                late_guards.append(
                    "visible person count changed after planning "
                    f"({len(frozen_candidates)} to {len(current_candidates)})"
                )
            else:
                frozen_positions = [_finite_xyz(position) for position in frozen_candidates]
                current_positions = [
                    _finite_xyz(candidate.get("position_m")) if isinstance(candidate, dict) else None
                    for candidate in current_candidates
                ]
                if any(position is None for position in frozen_positions + current_positions):
                    late_guards.append("multi-person obstacle coordinates are invalid immediately before execution")
                else:
                    obstacle_drifts = [
                        _norm(_subtract(current, frozen))
                        for current, frozen in zip(current_positions, frozen_positions)
                    ]
                    maximum_obstacle_drift = max(obstacle_drifts, default=0.0)
                    prepared["pre_execution_obstacle_drifts_m"] = obstacle_drifts
                    prepared["pre_execution_maximum_obstacle_drift_m"] = maximum_obstacle_drift
                    if maximum_obstacle_drift > MAX_PRE_EXECUTION_OBSTACLE_DRIFT_M:
                        late_guards.append(
                            "a visible person's collision geometry moved "
                            f"{maximum_obstacle_drift:.4f} m after planning, above the "
                            f"{MAX_PRE_EXECUTION_OBSTACLE_DRIFT_M:.4f} m limit"
                        )
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
        actual = _execution_target_verification(
            final_tool0=final_tool0,
            start_tool0=start_tool0,
            target_tool0=prepared.get("target_tool0_pose", {}),
            target_straw_tip_position_m=prepared.get("pre_mouth_pose", {}).get(
                "position_m", []
            ),
        )

        success = bool(
            execution_result.get("success")
            and actual.get("straw_tip_within_target_tolerance")
            and actual.get("orientation_matches_planned_target")
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
            failures: list[str] = []
            if not execution_result.get("success"):
                failures.append("MoveIt execution failed")
            if not actual.get("straw_tip_within_target_tolerance"):
                failures.append("the straw missed the pre-mouth target")
            if not actual.get("orientation_matches_planned_target"):
                failures.append("the final tool orientation missed the planned target")
            prepared["reason"] = "; ".join(failures) or (
                actual.get("reason") or "post-execution verification failed"
            )
        return (0 if success else 2), prepared


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("check", "plan", "execute", "return", "diagnose-frozen"),
        default="check",
    )
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
        help="Pre-mouth stand-off distance in metres (default: 0.050).",
    )
    parser.add_argument(
        "--target-selection",
        choices=("left", "center", "right"),
        default="center",
        help=(
            "Select the initially left, center, or right visible mouth and retain that 3D identity. "
            "Guarded real execution remains center-only."
        ),
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
        help="MoveIt velocity scaling for the validated plan (default: 0.30; allowed: 0.01–0.30).",
    )
    parser.add_argument(
        "--trajectory-acceleration-scaling",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
        help="MoveIt acceleration scaling for the validated plan (default: 0.30; allowed: 0.01–0.30).",
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
    parser.add_argument(
        "--diagnostic-mouth-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Recorded base_link mouth point used only by --mode diagnose-frozen. "
            "This mode is plan-only and refuses a nonempty OctoMap."
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
    if args.mode == "diagnose-frozen" and args.diagnostic_mouth_position is None:
        parser.error("--mode diagnose-frozen requires --diagnostic-mouth-position X Y Z")
    if args.mode != "diagnose-frozen" and args.diagnostic_mouth_position is not None:
        parser.error("--diagnostic-mouth-position is valid only with --mode diagnose-frozen")
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
        target_selection=args.target_selection,
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
                "target_selection": args.target_selection,
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
        elif args.mode == "return":
            exit_code, response = node.return_to_recorded_start(
                args.return_report,
                confirm_real_motion=args.confirm_real_motion,
                no_execute=args.no_execute,
            )
        else:
            exit_code, response = node.diagnose_frozen_mouth_static_scene(
                args.diagnostic_mouth_position
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
