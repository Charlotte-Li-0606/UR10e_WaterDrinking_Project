#!/usr/bin/env python3
"""Guarded real-UR10e feed_water with search, selection, and replanning.

The workflow remains one high-level operation.  It uses the physical D435i
multi-mouth stream to retain the selected person's 3D identity, performs a
bounded search only when that target is absent, and asks MoveGroup/OctoMap to
find a route to a validated pre-mouth target. The default remains the frozen
one-shot workflow. An explicit tracking mode monitors stable mouth drift,
cancels and replans MoveIt motion when needed, then uses bounded relative Servo
translation during the pre-mouth hold after MoveGroup releases the controller.
Backward/up/down search
steps are bounded Cartesian translations and left/right steps are bounded pose
rotations about tool0 local Z. Throughout search and obstacle routes, tool0 +Z
must remain aligned with base_link -Z within five degrees; MoveIt may choose
spin about that axis, including wrist_3_joint motion. The final pre-mouth goal
retains its validated full orientation. After a successful real hold, the same
guarded process plans one collision-checked return to the fixed, versioned
`initial_position`; it preserves the human scene, OctoMap, attached tool body,
and vertical-axis constraint, and rechecks every return waypoint immediately
before dispatch.

Plan mode never creates an execution request.  Execute mode retains the
existing environment, confirmation, controller, External Control, safety,
robot-mode, speed, calibration, identity, reach, collision, and final-pose
gates. The workflow never commands wrist_3_joint directly and refuses any
trajectory that has not passed waypoint FK validation.
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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy  # noqa: E402
from geometry_msgs.msg import Pose, PoseStamped  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup  # noqa: E402
from moveit_msgs.msg import (  # noqa: E402
    Constraints,
    JointConstraint,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
    ServoStatus,
)
from moveit_msgs.srv import GetPlanningScene, GetStateValidity, ServoCommandType  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Empty, SetBool  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402

from robot_layer.arm_ur10e.control.motion_backend import MotionRequest  # noqa: E402
from robot_layer.arm_ur10e.control.relative_tracking import RelativeTrackingSession  # noqa: E402
from robot_layer.arm_ur10e.control.ros_servo_backend import RosServoCommandSink  # noqa: E402
from robot_layer.arm_ur10e.control.continuous_servo_tracking import (  # noqa: E402
    ContinuousServoConfig,
    ContinuousServoController,
    MotionCommandArbiter,
    MotionCommandOwner,
    camera_ray_premouth_target,
    octomap_layer_status,
)
from robot_layer.arm_ur10e.agent_tools.planning_scene_manager import (  # noqa: E402
    PlanningSceneObstacleConfig,
    PlanningSceneObstacleManager,
)
from robot_layer.arm_ur10e.perception.continuous_mouth_tracker import (  # noqa: E402
    ContinuousMouthTarget,
    ContinuousMouthTracker,
    ContinuousTrackingState,
    InitialTargetAcquirer,
)

from scripts.real_dynamic_obstacle_avoidance_plan import (  # noqa: E402
    DETOUR_ROUTE_STRATEGY,
    DIRECT_ROUTE_STRATEGY,
    FILTERED_CLOUD_TOPIC,
    OMPL_PIPELINE,
    OMPL_PLANNER,
    RAW_CLOUD_TOPIC,
    REPLAN_ATTEMPTS,
    REPLAN_DELAY_SEC,
    RealDynamicObstacleAvoidancePlan,
)
from scripts.real_premouth_from_perception_plan import (  # noqa: E402
    ACTION_TIMEOUT_SEC,
    BASE_FRAME,
    CAMERA_OPTICAL_FRAME,
    DEFAULT_MOUTH_SAMPLE_SECONDS,
    DEFAULT_SAFE_DISTANCE_M,
    DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    DEFAULT_TRAJECTORY_VELOCITY_SCALING,
    EXPECTED_JOINTS,
    GROUP_NAME,
    MAX_MOUTH_POSE_AGE_SEC,
    MAX_PLAN_TRANSLATION_M,
    MAX_POSE_SPREAD_M,
    MAX_PRE_EXECUTION_TARGET_DRIFT_M,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MAX_TOOL_VERTICAL_TILT_RAD,
    MAX_EXECUTION_SPEED_PERCENT,
    MOUTH_STATUS_TOPIC,
    MIN_EXECUTION_SPEED_PERCENT,
    MIN_STABLE_SAMPLES,
    PILZ_PIPELINE,
    PILZ_PLANNER,
    STRAW_TIP_OFFSET_TOOL0_M,
    TOOL_FRAME,
    UR_BASE_FRAME,
    RealPreMouthFromPerceptionPlan,
    _add,
    _adaptive_premouth_pose_candidates,
    _finite_xyz,
    _jsonable,
    _norm,
    _quaternion_distance_rad,
    _rotate_tool_vector,
    _subtract,
    _trajectory_summary,
    _tool_vertical_tilt_rad,
)


SEARCH_MAX_TIME_SEC = 15.0
SEARCH_STABILITY_RESERVE_SEC = 1.25
SEARCH_PLANNING_PIPELINE = PILZ_PIPELINE
SEARCH_PLANNER = PILZ_PLANNER
SEARCH_ALLOWED_PLANNING_TIME_SEC = 2.0
SEARCH_PLAN_RESULT_TIMEOUT_SEC = 3.0
SEARCH_BACK_DISTANCE_M = 0.040
SEARCH_VERTICAL_DISTANCE_M = 0.050
SEARCH_MAX_ACTUAL_SEGMENT_M = 0.110
SEARCH_FINAL_POSITION_TOLERANCE_M = 0.010
SEARCH_FINAL_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
SEARCH_MAX_JOINT_EXCURSION_RAD = math.radians(45.0)
SEARCH_MAX_CUMULATIVE_JOINT_TRAVEL_RAD = 2.0
SEARCH_MAX_TRAJECTORY_DURATION_SEC = 3.0
SEARCH_WRIST_Z_TOTAL_SWEEP_DEG = 30.0
SEARCH_WRIST_Z_ANGLE_DEG = SEARCH_WRIST_Z_TOTAL_SWEEP_DEG / 2.0
TRACKING_MAX_TARGET_DRIFT_REPLANS = 2
TRACKING_MAX_APPROACH_SEGMENTS = 16
TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC = 3.0
TRACKING_SEGMENT_MAX_TRANSLATION_M = 0.050
TRACKING_SEGMENT_MAX_ROTATION_RAD = math.radians(15.0)
TRACKING_MAX_APPROACH_DURATION_SEC = 45.0
TRACKING_TARGET_MAX_AGE_SEC = 0.75
TRACKING_MAX_DISPLACEMENT_M = 0.06
TRACKING_MAX_LINEAR_SPEED_MPS = 0.02
TRACKING_MAX_LINEAR_ACCELERATION_MPS2 = 0.10
# Continuous recovery is a local mouth-following correction, not permission
# for OMPL to choose a remote IK branch.  These limits are checked against the
# complete planned trajectory before it can be cached or executed.
CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD = {
    "shoulder_pan_joint": math.radians(45.0),
    "shoulder_lift_joint": math.radians(45.0),
    "elbow_joint": math.radians(60.0),
    "wrist_1_joint": math.radians(60.0),
    "wrist_2_joint": math.radians(60.0),
    "wrist_3_joint": math.radians(90.0),
}
CONTINUOUS_RECOVERY_MAX_CUMULATIVE_JOINT_TRAVEL_RAD = 4.0
CONTINUOUS_RECOVERY_MAX_TRAJECTORY_DURATION_SEC = 20.0
CONTINUOUS_RECOVERY_TOOL_PATH_DETOUR_MARGIN_M = 0.15
CONTINUOUS_RECOVERY_MIN_TOOL_PATH_ENVELOPE_M = 0.15
CONTINUOUS_RECOVERY_MAX_TOOL_PATH_ENVELOPE_M = 0.50
SEARCH_WRIST_Z_CANDIDATE_ANGLES_RAD = tuple(
    math.radians(value) for value in (SEARCH_WRIST_Z_ANGLE_DEG, 10.0, 5.0)
)
SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC = 0.010
SEARCH_STATIONARY_SAMPLE_COUNT = 2
SEARCH_STATIONARY_TIMEOUT_SEC = 0.75
SEARCH_OCTOMAP_REBUILD_TIMEOUT_SEC = 1.5
GOAL_OCTOMAP_REBUILD_TIMEOUT_SEC = 4.0
OCTOMAP_REBUILD_REQUIRED_FRAMES = 3
OCTOMAP_REBUILD_MAX_POINT_COUNT_SPREAD = 0.15
OCTOMAP_REBUILD_HISTORY_LIMIT = 16
SEARCH_BACK_CANDIDATE_DISTANCES_M = (0.040, 0.030, 0.020)
SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M = (0.050, 0.040, 0.030, 0.020)
EXECUTION_MOUTH_DRIFT_CONFIRMATION_WINDOW_SEC = 1.0
EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES = 3
MAX_EXECUTION_TARGET_DRIFT_M = 0.050
INITIAL_POSITION_CONFIG = (
    PROJECT_ROOT / "config" / "ur10e_real" / "initial_position.json"
)
CONTINUOUS_TRACKING_CONFIG = PROJECT_ROOT / "config" / "continuous_mouth_tracking.yaml"
DEFAULT_PREMOUTH_HOLD_SEC = 5.0
MIN_PREMOUTH_HOLD_SEC = 2.0
MAX_PREMOUTH_HOLD_SEC = 5.0
RETURN_JOINT_GOAL_TOLERANCE_RAD = math.radians(1.0)
RETURN_PLANNING_ATTEMPTS = 3
RETURN_PLANNING_TIME_SEC = 10.0
SEARCH_OFFSETS_CAMERA_OPTICAL = (
    ("backward_wide", (0.0, 0.0, -SEARCH_BACK_DISTANCE_M)),
    (
        "scan_up",
        (0.0, -SEARCH_VERTICAL_DISTANCE_M, -SEARCH_BACK_DISTANCE_M),
    ),
    (
        "scan_down",
        (0.0, SEARCH_VERTICAL_DISTANCE_M, -SEARCH_BACK_DISTANCE_M),
    ),
)


def _load_continuous_tracking_config() -> dict[str, Any]:
    """Load only the isolated continuous-mode parameters."""
    try:
        raw = yaml.safe_load(CONTINUOUS_TRACKING_CONFIG.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot load {CONTINUOUS_TRACKING_CONFIG}: {exc}") from exc
    values = raw.get("continuous_mouth_tracking") if isinstance(raw, dict) else None
    if not isinstance(values, dict):
        raise RuntimeError("continuous_mouth_tracking configuration is missing")
    validity_period = float(values.get("state_validity_check_period_sec", 0.0))
    if not 0.05 <= validity_period <= 0.50:
        raise RuntimeError(
            "state_validity_check_period_sec must remain within [0.05, 0.50] seconds"
        )
    return dict(values)


def _active_search_cloud_gate_failures(
    *,
    use_octomap: bool,
    cloud_statuses: dict[str, dict[str, Any]],
) -> list[str]:
    """Require RGB-D clouds only when the dynamic OctoMap layer is enabled."""
    if not use_octomap:
        return []
    if not all(status.get("active") for status in cloud_statuses.values()):
        return ["raw or filtered wrist point cloud became stale before search motion"]
    return []


def _quaternion_multiply_xyzw(
    first: list[float], second: list[float]
) -> list[float]:
    """Return the normalized Hamilton product ``first * second`` in XYZW."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("quaternions must contain four values")
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


def _select_consistent_cloud_frame_window(
    items: list[dict[str, Any]],
    *,
    required_frames: int = OCTOMAP_REBUILD_REQUIRED_FRAMES,
    maximum_relative_point_count_spread: float = (
        OCTOMAP_REBUILD_MAX_POINT_COUNT_SPREAD
    ),
) -> dict[str, Any]:
    """Select the newest consecutive stationary cloud window that is stable."""
    if required_frames < 1:
        raise ValueError("required_frames must be positive")
    if not 0.0 <= maximum_relative_point_count_spread < 1.0:
        raise ValueError("maximum_relative_point_count_spread must be in [0, 1)")

    best_diagnostic: dict[str, Any] | None = None
    latest_consistent: dict[str, Any] | None = None
    for start in range(max(0, len(items) - required_frames + 1)):
        window = [dict(item) for item in items[start : start + required_frames]]
        if len(window) != required_frames:
            continue
        counts = [int(item.get("point_count", 0)) for item in window]
        frame_ids = [str(item.get("frame_id", "")) for item in window]
        receive_times = [float(item.get("received_monotonic", 0.0)) for item in window]
        source_stamps = [
            (int(item.get("stamp_sec", 0)), int(item.get("stamp_nanosec", 0)))
            for item in window
        ]
        positive_counts = min(counts, default=0) > 0
        same_nonempty_frame = bool(frame_ids[0]) and len(set(frame_ids)) == 1
        fresh_sequence = all(
            later > earlier
            for earlier, later in zip(receive_times, receive_times[1:])
        ) and len(set(source_stamps)) == required_frames
        spread = (
            (max(counts) - min(counts)) / max(counts)
            if positive_counts
            else math.inf
        )
        candidate = {
            "consistent": bool(
                positive_counts
                and same_nonempty_frame
                and fresh_sequence
                and spread <= maximum_relative_point_count_spread
            ),
            "frames": window,
            "point_counts": counts,
            "relative_point_count_spread": spread,
            "same_nonempty_frame_id": same_nonempty_frame,
            "fresh_unique_sequence": fresh_sequence,
        }
        if best_diagnostic is None or candidate[
            "relative_point_count_spread"
        ] <= best_diagnostic[
            "relative_point_count_spread"
        ]:
            best_diagnostic = candidate
        if candidate["consistent"]:
            # Iteration is oldest-to-newest, so the last accepted candidate wins.
            latest_consistent = candidate

    selected = latest_consistent or best_diagnostic
    if selected is None:
        selected = {
            "consistent": False,
            "frames": [],
            "point_counts": [],
            "relative_point_count_spread": None,
            "same_nonempty_frame_id": False,
            "fresh_unique_sequence": False,
        }
    return selected


def _compare_final_state_validity(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compare only final-state checks that produced usable robot states."""
    before_valid = before.get("valid") if before.get("available") else None
    after_valid = after.get("valid") if after.get("available") else None
    available = isinstance(before_valid, bool) and isinstance(after_valid, bool)
    return {
        "available": available,
        "before_valid": before_valid,
        "after_valid": after_valid,
        "changed": (before_valid != after_valid) if available else None,
        "reason": (
            None
            if available
            else (
                "the nominal candidate did not produce usable before/after "
                "robot states; adaptive candidate IK and collision checks "
                "must determine goal validity"
            )
        ),
    }


def _quaternion_slerp_xyzw(
    start_xyzw: list[float],
    end_xyzw: list[float],
    fraction: float,
) -> list[float]:
    """Interpolate the shortest normalized quaternion arc in XYZW order."""
    if len(start_xyzw) != 4 or len(end_xyzw) != 4:
        raise ValueError("quaternions must contain four values")
    if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("quaternion interpolation fraction must be in [0, 1]")
    start = [float(value) for value in start_xyzw]
    end = [float(value) for value in end_xyzw]
    start_norm = math.sqrt(sum(value * value for value in start))
    end_norm = math.sqrt(sum(value * value for value in end))
    if start_norm < 1e-9 or end_norm < 1e-9:
        raise ValueError("quaternions must be nonzero")
    start = [value / start_norm for value in start]
    end = [value / end_norm for value in end]
    dot = sum(left * right for left, right in zip(start, end))
    if dot < 0.0:
        end = [-value for value in end]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        mixed = [
            left + float(fraction) * (right - left)
            for left, right in zip(start, end)
        ]
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        start_weight = math.sin((1.0 - float(fraction)) * theta) / sin_theta
        end_weight = math.sin(float(fraction) * theta) / sin_theta
        mixed = [
            start_weight * left + end_weight * right
            for left, right in zip(start, end)
        ]
    magnitude = math.sqrt(sum(value * value for value in mixed))
    return [value / magnitude for value in mixed]


def _bounded_tracking_segment_target(
    *,
    current_pose: dict[str, Any],
    final_pose: dict[str, Any],
    maximum_translation_m: float = TRACKING_SEGMENT_MAX_TRANSLATION_M,
    maximum_rotation_rad: float = TRACKING_SEGMENT_MAX_ROTATION_RAD,
) -> dict[str, Any]:
    """Return one bounded Cartesian pose step toward a validated final pose."""
    if (
        not math.isfinite(float(maximum_translation_m))
        or float(maximum_translation_m) <= 0.0
        or not math.isfinite(float(maximum_rotation_rad))
        or float(maximum_rotation_rad) <= 0.0
    ):
        raise ValueError("tracking segment bounds must be finite and positive")
    current_position = _finite_xyz(current_pose.get("position_m"))
    final_position = _finite_xyz(final_pose.get("position_m"))
    current_orientation = current_pose.get("orientation_quat_xyzw")
    final_orientation = final_pose.get("orientation_quat_xyzw")
    if current_position is None or final_position is None:
        raise ValueError("tracking segment poses require finite positions")
    if not isinstance(current_orientation, list) or not isinstance(
        final_orientation, list
    ):
        raise ValueError("tracking segment poses require orientations")
    translation = _subtract(final_position, current_position)
    translation_distance = _norm(translation)
    rotation_distance = _quaternion_distance_rad(
        current_orientation,
        final_orientation,
    )
    fractions = [1.0]
    if translation_distance > 1e-9:
        fractions.append(float(maximum_translation_m) / translation_distance)
    if rotation_distance > 1e-9:
        fractions.append(float(maximum_rotation_rad) / rotation_distance)
    fraction = max(0.0, min(fractions))
    position = [
        start + fraction * delta
        for start, delta in zip(current_position, translation)
    ]
    orientation = _quaternion_slerp_xyzw(
        [float(value) for value in current_orientation],
        [float(value) for value in final_orientation],
        fraction,
    )
    return {
        "frame_id": str(final_pose.get("frame_id", BASE_FRAME)),
        "link_name": str(final_pose.get("link_name", TOOL_FRAME)),
        "position_m": position,
        "orientation_quat_xyzw": orientation,
        "fraction_of_remaining_path": fraction,
        "remaining_translation_m": translation_distance,
        "remaining_rotation_rad": rotation_distance,
        "segment_translation_m": translation_distance * fraction,
        "segment_rotation_rad": rotation_distance * fraction,
        "final_segment": fraction >= 1.0 - 1e-9,
    }


def _axis_angle_vector_to_quaternion_xyzw(
    rotation_vector_rad: list[float],
) -> list[float]:
    """Convert a UR/PolyScope rotation vector to a normalized quaternion."""
    if len(rotation_vector_rad) != 3:
        raise ValueError("rotation vector must contain three values")
    vector = [float(value) for value in rotation_vector_rad]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("rotation vector must contain finite values")
    angle = _norm(vector)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    scale = math.sin(angle / 2.0) / angle
    return [
        vector[0] * scale,
        vector[1] * scale,
        vector[2] * scale,
        math.cos(angle / 2.0),
    ]


def _load_initial_position_config(
    path: Path = INITIAL_POSITION_CONFIG,
) -> dict[str, Any]:
    """Load the one immutable return target and normalize degrees to radians."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2:
        raise ValueError("initial_position must use schema version 2")
    if raw.get("name") != "initial_position":
        raise ValueError("return configuration name must be initial_position")
    names = raw.get("joint_names")
    degrees = raw.get("joint_positions_deg")
    if names != list(EXPECTED_JOINTS):
        raise ValueError("initial_position joint_names must exactly match the MoveIt group")
    if not isinstance(degrees, list) or len(degrees) != len(EXPECTED_JOINTS):
        raise ValueError("initial_position must contain six joint positions")
    joint_degrees = [float(value) for value in degrees]
    if not all(math.isfinite(value) for value in joint_degrees):
        raise ValueError("initial_position joint positions must be finite")
    if raw.get("authoritative_target") != "joint_positions_deg":
        raise ValueError("initial_position must declare the fixed joints authoritative")
    fk_reference = raw.get("moveit_tool0_fk_reference")
    if not isinstance(fk_reference, dict):
        raise ValueError("initial_position MoveIt FK reference is missing")
    if fk_reference.get("frame_id") != BASE_FRAME:
        raise ValueError("initial-position FK reference must use base_link")
    fk_position = fk_reference.get("position_m")
    fk_orientation = fk_reference.get("orientation_quat_xyzw")
    if not isinstance(fk_position, list) or len(fk_position) != 3:
        raise ValueError("initial-position FK reference must contain three position values")
    if not isinstance(fk_orientation, list) or len(fk_orientation) != 4:
        raise ValueError("initial-position FK reference must contain a quaternion")
    fk_position_m = [float(value) for value in fk_position]
    fk_orientation_xyzw = [float(value) for value in fk_orientation]
    fk_orientation_norm = _norm(fk_orientation_xyzw)
    if not all(
        math.isfinite(value)
        for value in fk_position_m + fk_orientation_xyzw
    ) or fk_orientation_norm < 1e-12:
        raise ValueError("initial-position FK reference must be finite and nonzero")
    fk_orientation_xyzw = [
        value / fk_orientation_norm for value in fk_orientation_xyzw
    ]
    fk_source = str(fk_reference.get("source", "")).strip()
    if not fk_source:
        raise ValueError("initial-position FK reference source is missing")
    displayed = raw.get("operator_displayed_tool_pose")
    if not isinstance(displayed, dict):
        raise ValueError("operator-displayed initial tool pose is missing")
    if displayed.get("frame_id") != "unverified_polyscope_active_feature":
        raise ValueError("operator-displayed pose must retain its unverified feature frame")
    if displayed.get("rotation_convention") != "polyscope_axis_angle_vector":
        raise ValueError("operator-displayed orientation must use a PolyScope rotation vector")
    displayed_position = displayed.get("position_m")
    displayed_rotation = displayed.get("rotation_vector_rad")
    if not isinstance(displayed_position, list) or len(displayed_position) != 3:
        raise ValueError("operator-displayed tool position must contain three values")
    if not isinstance(displayed_rotation, list) or len(displayed_rotation) != 3:
        raise ValueError("operator-displayed tool rotation vector must contain three values")
    displayed_position_m = [float(value) for value in displayed_position]
    displayed_rotation_rad = [float(value) for value in displayed_rotation]
    if not all(
        math.isfinite(value)
        for value in displayed_position_m + displayed_rotation_rad
    ):
        raise ValueError("operator-displayed initial tool pose must be finite")
    displayed_delta = float(
        displayed.get("moveit_target_position_delta_m", float("nan"))
    )
    if not math.isfinite(displayed_delta) or displayed_delta <= 0.0:
        raise ValueError("operator-displayed pose must record the measured FK offset")
    if displayed.get("verification_status") != (
        "not_used_as_a_moveit_base_frame_reference"
    ):
        raise ValueError("operator-displayed pose verification status is invalid")
    verification = raw.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("initial_position verification tolerances are missing")
    required_tolerances = (
        "maximum_fk_position_error_m",
        "maximum_fk_orientation_error_deg",
        "maximum_final_joint_error_deg",
        "maximum_final_tool_position_error_m",
        "maximum_final_tool_orientation_error_deg",
        "maximum_tool_vertical_tilt_deg",
    )
    tolerances = {name: float(verification[name]) for name in required_tolerances}
    if not all(math.isfinite(value) and value > 0.0 for value in tolerances.values()):
        raise ValueError("initial_position verification tolerances must be finite and positive")
    return {
        "schema_version": 2,
        "name": "initial_position",
        "config_path": str(path),
        "joint_names": list(EXPECTED_JOINTS),
        "joint_positions_deg": joint_degrees,
        "joint_positions_rad": [math.radians(value) for value in joint_degrees],
        "authoritative_target": "joint_positions_deg",
        "moveit_tool0_fk_reference": {
            "frame_id": BASE_FRAME,
            "position_m": fk_position_m,
            "orientation_quat_xyzw": fk_orientation_xyzw,
            "source": fk_source,
        },
        "operator_displayed_tool_pose": {
            "frame_id": "unverified_polyscope_active_feature",
            "position_m": displayed_position_m,
            "rotation_vector_rad": displayed_rotation_rad,
            "orientation_quat_xyzw": _axis_angle_vector_to_quaternion_xyzw(
                displayed_rotation_rad
            ),
            "rotation_convention": "polyscope_axis_angle_vector",
            "moveit_target_position_delta_m": displayed_delta,
            "verification_status": str(displayed.get("verification_status", "")),
            "note": str(displayed.get("note", "")),
        },
        "verification": tolerances,
    }


def _recorded_tool_pose_in_base_link(
    config: dict[str, Any],
    base_from_base_link: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the versioned calibrated FK reference in MoveIt's base_link."""
    del base_from_base_link  # Kept for source compatibility with older tests/tools.
    reference = config["moveit_tool0_fk_reference"]
    return {
        "frame_id": BASE_FRAME,
        "link_name": TOOL_FRAME,
        "position_m": list(reference["position_m"]),
        "orientation_quat_xyzw": list(reference["orientation_quat_xyzw"]),
        "source": reference["source"],
    }


def _trajectory_final_joint_error(
    trajectory: Any,
    target_positions_rad: dict[str, float],
) -> dict[str, Any]:
    """Measure the final planned joints against the fixed initial target."""
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    names = list(getattr(joint_trajectory, "joint_names", []))
    points = list(getattr(joint_trajectory, "points", []))
    if not names or not points:
        return {"available": False, "reason": "trajectory has no joint waypoints"}
    final_positions = list(points[-1].positions)
    if len(final_positions) != len(names):
        return {"available": False, "reason": "trajectory final joint waypoint is incomplete"}
    final = dict(zip(names, (float(value) for value in final_positions)))
    missing = sorted(set(target_positions_rad) - set(final))
    if missing:
        return {
            "available": False,
            "reason": f"trajectory final waypoint is missing joints: {', '.join(missing)}",
            "missing_joints": missing,
        }
    errors = {
        name: abs(
            math.atan2(
                math.sin(final[name] - float(target)),
                math.cos(final[name] - float(target)),
            )
        )
        for name, target in target_positions_rad.items()
    }
    return {
        "available": True,
        "joint_errors_rad": errors,
        "maximum_joint_error_rad": max(errors.values(), default=0.0),
        "maximum_joint_error_deg": math.degrees(max(errors.values(), default=0.0)),
        "final_joint_positions_rad": {name: final[name] for name in target_positions_rad},
    }


def _validate_continuous_recovery_trajectory(
    trajectory: Any,
    *,
    requested_tool0_translation_m: float,
    tool_path_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject a collision-free recovery that leaves the local IK branch."""
    requested_translation = float(requested_tool0_translation_m)
    allowed_tool_excursion = min(
        CONTINUOUS_RECOVERY_MAX_TOOL_PATH_ENVELOPE_M,
        max(
            CONTINUOUS_RECOVERY_MIN_TOOL_PATH_ENVELOPE_M,
            requested_translation + CONTINUOUS_RECOVERY_TOOL_PATH_DETOUR_MARGIN_M,
        ),
    )
    result: dict[str, Any] = {
        "success": False,
        "stage": "continuous_recovery_motion_validation",
        "requested_tool0_translation_m": requested_translation,
        "maximum_allowed_tool0_excursion_m": allowed_tool_excursion,
        "maximum_tool0_excursion_from_start_m": None,
        "maximum_cumulative_joint_travel_rad": (
            CONTINUOUS_RECOVERY_MAX_CUMULATIVE_JOINT_TRAVEL_RAD
        ),
        "maximum_trajectory_duration_sec": (
            CONTINUOUS_RECOVERY_MAX_TRAJECTORY_DURATION_SEC
        ),
        "joint_excursion_limits_rad": dict(
            CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD
        ),
    }
    if not math.isfinite(requested_translation) or requested_translation < 0.0:
        return {**result, "reason": "requested recovery translation is invalid"}
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    names = list(getattr(joint_trajectory, "joint_names", []))
    points = list(getattr(joint_trajectory, "points", []))
    if not names or not points:
        return {**result, "reason": "recovery trajectory has no joint waypoints"}
    missing = sorted(
        set(CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD) - set(names)
    )
    if missing:
        return {
            **result,
            "reason": "recovery trajectory is missing joints: " + ", ".join(missing),
            "missing_joints": missing,
        }
    positions = [[float(value) for value in point.positions] for point in points]
    if any(
        len(values) != len(names)
        or not all(math.isfinite(value) for value in values)
        for values in positions
    ):
        return {**result, "reason": "recovery trajectory has invalid joint positions"}

    def angular_distance(first: float, second: float) -> float:
        return abs(math.atan2(math.sin(second - first), math.cos(second - first)))

    start = positions[0]
    excursions = {
        name: max(
            angular_distance(start[index], values[index]) for values in positions
        )
        for index, name in enumerate(names)
    }
    cumulative_travel = sum(
        angular_distance(previous[index], current[index])
        for previous, current in zip(positions, positions[1:])
        for index in range(len(names))
    )
    final_time = points[-1].time_from_start
    duration = float(final_time.sec) + float(final_time.nanosec) * 1.0e-9
    maximum_tool_excursion = None
    if isinstance(tool_path_validation, dict):
        raw_excursion = tool_path_validation.get(
            "maximum_tool0_excursion_from_start_m"
        )
        if raw_excursion is not None:
            try:
                maximum_tool_excursion = float(raw_excursion)
            except (TypeError, ValueError):
                maximum_tool_excursion = None
    result.update(
        {
            "joint_excursions_rad": excursions,
            "joint_excursions_deg": {
                name: math.degrees(value) for name, value in excursions.items()
            },
            "cumulative_joint_travel_rad": cumulative_travel,
            "trajectory_duration_sec": duration,
            "maximum_tool0_excursion_from_start_m": maximum_tool_excursion,
        }
    )
    failures: list[str] = []
    for name, limit in CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD.items():
        if excursions[name] > limit:
            failures.append(
                f"{name} excursion {math.degrees(excursions[name]):.1f} deg "
                f"exceeds {math.degrees(limit):.1f} deg"
            )
    if cumulative_travel > CONTINUOUS_RECOVERY_MAX_CUMULATIVE_JOINT_TRAVEL_RAD:
        failures.append("cumulative joint travel exceeds the local recovery limit")
    if not math.isfinite(duration) or duration > CONTINUOUS_RECOVERY_MAX_TRAJECTORY_DURATION_SEC:
        failures.append("trajectory duration exceeds the local recovery limit")
    if maximum_tool_excursion is None or not math.isfinite(maximum_tool_excursion):
        failures.append("tool0 path excursion was not verified by waypoint FK")
    elif maximum_tool_excursion > allowed_tool_excursion:
        failures.append(
            f"tool0 path excursion {maximum_tool_excursion:.3f} m exceeds "
            f"the local {allowed_tool_excursion:.3f} m envelope"
        )
    result["success"] = not failures
    result["reason"] = "; ".join(failures) if failures else None
    return result


def _orientation_after_local_tool_z_rotation(
    tool_orientation_xyzw: list[float], angle_rad: float
) -> list[float]:
    """Rotate a tool pose about its own +Z axis without commanding a joint."""
    if not math.isfinite(float(angle_rad)):
        raise ValueError("tool-local Z rotation must be finite")
    half = float(angle_rad) / 2.0
    local_z_rotation = [0.0, 0.0, math.sin(half), math.cos(half)]
    return _quaternion_multiply_xyzw(tool_orientation_xyzw, local_z_rotation)


class RealIntegratedFeedWater(RealDynamicObstacleAvoidancePlan):
    """One real feed_water state machine retaining target identity in-process."""

    def __init__(
        self,
        *,
        target_selection: str,
        mouth_sample_seconds: float,
        trajectory_velocity_scaling: float,
        trajectory_acceleration_scaling: float,
    ) -> None:
        super().__init__(
            premouth_policy="camera-ray",
            safe_distance_m=DEFAULT_SAFE_DISTANCE_M,
            maximum_plan_translation_m=MAX_PLAN_TRANSLATION_M,
            target_selection=target_selection,
            mouth_sample_seconds=mouth_sample_seconds,
            trajectory_velocity_scaling=trajectory_velocity_scaling,
            trajectory_acceleration_scaling=trajectory_acceleration_scaling,
        )
        self._frozen_execution_mouth_position: list[float] | None = None
        self._integrated_tracking_enabled = False
        self._latest_servo_status: ServoStatus | None = None
        self._latest_servo_status_monotonic: float | None = None
        # A fresh CLI process owns a fresh human/OctoMap scene lifecycle.  The
        # first integrated workflow must retire the previous process's managed
        # geometry before it can rebuild the current scene or plan any motion.
        self._session_collision_geometry_reset_complete = False
        self._session_collision_geometry_reset_result: dict[str, Any] | None = None
        self._state_validity_client = self.create_client(
            GetStateValidity,
            "/check_state_validity",
        )
        self._clear_octomap_client = self.create_client(
            Empty,
            "/clear_octomap",
        )
        self._servo_command_type_client = self.create_client(
            ServoCommandType,
            "/servo_node/switch_command_type",
        )
        self._servo_pause_client = self.create_client(
            SetBool,
            "/servo_node/pause_servo",
        )
        # MoveGroup and Servo use separate PlanningSceneMonitor instances on
        # this installation.  Applying geometry through MoveGroup's service is
        # therefore insufficient: mirror the verified deterministic scene on
        # the standard topic so Servo receives the same human/tool objects.
        self._planning_scene_publisher = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        # MoveIt Servo publishes a latched/event-driven status rather than a
        # periodic heartbeat.  Match its reliable + transient-local QoS so a
        # newly created workflow receives the last known status immediately.
        self._servo_status_subscription = self.create_subscription(
            ServoStatus,
            "/servo_node/status",
            self._servo_status_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        continuous = _load_continuous_tracking_config()
        self._continuous_parameters = continuous
        self._continuous_tracker = ContinuousMouthTracker(
            target_timeout_sec=float(continuous["target_timeout_sec"]),
            lost_target_timeout_sec=float(continuous["lost_target_timeout_sec"]),
            minimum_confidence=float(continuous["minimum_confidence"]),
            stable_sample_count=int(continuous["stable_sample_count"]),
            stable_max_spread_m=float(continuous["stable_max_spread_m"]),
            prediction_horizon_sec=float(continuous["prediction_horizon_sec"]),
            maximum_prediction_m=float(continuous["maximum_prediction_m"]),
        )
        self._continuous_servo_controller = ContinuousServoController(
            ContinuousServoConfig(
                final_pre_mouth_standoff_m=float(
                    continuous["final_pre_mouth_standoff_m"]
                ),
                provisional_standoff_m=float(continuous["provisional_standoff_m"]),
                servo_tracking_max_error_m=float(
                    continuous["servo_tracking_max_error_m"]
                ),
                servo_startup_max_error_m=float(
                    continuous["servo_startup_max_error_m"]
                ),
                servo_replan_enter_m=float(continuous["servo_replan_enter_m"]),
                servo_replan_exit_m=float(continuous["servo_replan_exit_m"]),
                hold_entry_tolerance_m=float(continuous["hold_entry_tolerance_m"]),
                maximum_linear_speed_mps=float(
                    continuous["maximum_linear_speed_mps"]
                ),
                provisional_linear_speed_mps=float(
                    continuous["provisional_linear_speed_mps"]
                ),
                maximum_linear_acceleration_mps2=float(
                    continuous["maximum_linear_acceleration_mps2"]
                ),
                maximum_angular_speed_rps=float(
                    continuous["maximum_angular_speed_rps"]
                ),
                orientation_correction_gain=float(
                    continuous["orientation_correction_gain"]
                ),
                control_gain=float(continuous["control_gain"]),
                maximum_tool_radius_m=float(continuous["maximum_tool_radius_m"]),
                maximum_flange_tilt_deg=float(
                    continuous["maximum_flange_tilt_deg"]
                ),
                maximum_tracking_duration_sec=float(
                    continuous["maximum_tracking_duration_sec"]
                ),
            ),
            straw_tip_offset_tool0_m=STRAW_TIP_OFFSET_TOOL0_M,
        )
        self._continuous_motion_arbiter = MotionCommandArbiter()
        self._continuous_last_observation_update: dict[str, Any] = {
            "accepted": False,
            "reason": "no_observation_received",
        }
        self._continuous_status_publisher = self.create_publisher(
            String,
            "/continuous_mouth_tracking/status",
            10,
        )
        self._continuous_target_publisher = self.create_publisher(
            PoseStamped,
            "/continuous_mouth_tracking/target_pose",
            10,
        )

    def _servo_status_callback(self, message: ServoStatus) -> None:
        self._latest_servo_status = message
        self._latest_servo_status_monotonic = time.monotonic()

    def _fresh_explicit_no_face_evidence(
        self,
        *,
        maximum_age_sec: float = 0.50,
    ) -> dict[str, Any]:
        """Require a live, fresh detector result before accepting an empty scene."""
        publisher_count = int(self.count_publishers(MOUTH_STATUS_TOPIC))
        received = self.latest_mouth_status_received_monotonic
        age = None if received is None else max(0.0, time.monotonic() - received)
        status = self.latest_mouth_status
        explicit_no_face = bool(
            isinstance(status, dict)
            and status.get("detected") is False
            and status.get("reason") == "no_face"
        )
        success = bool(
            publisher_count > 0
            and explicit_no_face
            and age is not None
            and age <= float(maximum_age_sec)
        )
        return {
            "success": success,
            "topic": MOUTH_STATUS_TOPIC,
            "publisher_count": publisher_count,
            "status": status,
            "status_age_sec": age,
            "maximum_age_sec": float(maximum_age_sec),
            "explicit_no_face": explicit_no_face,
            "reason": None if success else "fresh explicit no_face evidence is unavailable",
        }

    def _reset_previous_session_collision_geometry(self) -> dict[str, Any]:
        """Remove and verify only collision geometry owned by a prior session."""
        before, _ = self._planning_scene_geometry_with_discovery()
        report: dict[str, Any] = {
            "success": False,
            "performed": True,
            "execution_sent": False,
            "collision_bypassed": False,
            "scope": {
                "managed_human_world_objects": True,
                "dynamic_octomap": True,
                "combined_attached_tool_removed": False,
                "unmanaged_world_objects_removed": False,
            },
            "planning_scene_before": before,
        }
        if not before.get("available"):
            return {
                **report,
                "reason": before.get("reason") or "MoveIt PlanningScene is unavailable",
            }

        manager: PlanningSceneObstacleManager | None = None
        try:
            manager = PlanningSceneObstacleManager(
                PlanningSceneObstacleConfig(
                    base_frame=BASE_FRAME,
                    mouth_topic="/detected_mouth_pose",
                    include_table=False,
                    service_timeout_sec=5.0,
                    mouth_wait_timeout_sec=1.0,
                )
            )
            managed_removal = manager.remove(verify=True)
        except Exception as exc:
            managed_removal = {
                "success": False,
                "reason": (
                    "previous-session PlanningScene cleanup raised "
                    f"{exc.__class__.__name__}: {exc}"
                ),
            }
        finally:
            if manager is not None:
                manager.destroy_node()
        report["managed_world_object_removal"] = managed_removal
        if not managed_removal.get("success"):
            return {
                **report,
                "reason": (
                    managed_removal.get("reason")
                    or "previous-session managed collision objects could not be removed"
                ),
            }

        octomap_present = bool(before.get("octomap", {}).get("present"))
        octomap_clear: dict[str, Any] = {
            "success": True,
            "required": octomap_present,
            "attempted": False,
            "reason": None,
        }
        if octomap_present:
            clear_client = getattr(self, "_clear_octomap_client", None)
            if clear_client is None or not clear_client.wait_for_service(timeout_sec=1.0):
                octomap_clear.update(
                    success=False,
                    reason="previous dynamic OctoMap exists but /clear_octomap is unavailable",
                )
            else:
                future = clear_client.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                octomap_clear["attempted"] = True
                if future.result() is None:
                    octomap_clear.update(
                        success=False,
                        reason="previous dynamic OctoMap clear timed out",
                    )
        report["dynamic_octomap_clear"] = octomap_clear
        if not octomap_clear.get("success"):
            return {**report, "reason": octomap_clear.get("reason")}

        after, _ = self._planning_scene_geometry_with_discovery()
        report["planning_scene_after"] = after
        if not after.get("available"):
            return {
                **report,
                "reason": after.get("reason") or "PlanningScene cleanup could not be verified",
            }
        remaining_human_ids = list(after.get("human_object_ids", []))
        report["remaining_managed_human_object_ids"] = remaining_human_ids
        if remaining_human_ids:
            return {
                **report,
                "reason": "previous-session human collision objects remain after cleanup",
            }

        return {
            **report,
            "success": True,
            "reason": None,
            "removed_object_ids": list(managed_removal.get("object_ids", [])),
            "octomap_was_present": octomap_present,
            "octomap_clear_service_confirmed": bool(
                not octomap_present or octomap_clear.get("attempted")
            ),
            "current_scene_rebuild_required_before_motion": True,
        }

    def _planning_scene_geometry_with_discovery(
        self,
        *,
        timeout_sec: float = 4.0,
    ) -> tuple[dict[str, Any], list[Any]]:
        """Bound fresh-node discovery before declaring PlanningScene unavailable."""
        started = time.monotonic()
        deadline = started + max(0.0, float(timeout_sec))
        attempts = 0
        last_scene: dict[str, Any] = {
            "available": False,
            "reason": "PlanningScene discovery has not been attempted",
        }
        last_objects: list[Any] = []
        while True:
            attempts += 1
            last_scene, last_objects = self._planning_scene_geometry()
            if last_scene.get("available"):
                return (
                    {
                        **last_scene,
                        "discovery_attempts": attempts,
                        "discovery_elapsed_sec": time.monotonic() - started,
                    },
                    last_objects,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not rclpy.ok():
                break
            # Let Fast DDS process graph updates for this newly created node,
            # then retry the service query.  No ROS action or motion command is
            # issued during discovery.
            rclpy.spin_once(self, timeout_sec=min(0.10, remaining))
        return (
            {
                **last_scene,
                "available": False,
                "discovery_attempts": attempts,
                "discovery_elapsed_sec": time.monotonic() - started,
                "discovery_timeout_sec": max(0.0, float(timeout_sec)),
                "reason": (
                    last_scene.get("reason")
                    or "MoveIt PlanningScene remained unavailable after bounded discovery"
                ),
            },
            last_objects,
        )

    def _ensure_previous_session_collision_geometry_reset(self) -> dict[str, Any]:
        """Run the previous-session cleanup exactly once for a fresh process."""
        # Instances constructed normally always define this flag as False.
        # Treat its absence as already reset for narrow unit-test fixtures that
        # intentionally allocate the class with __new__ and no ROS services.
        if getattr(self, "_session_collision_geometry_reset_complete", True):
            cached = getattr(self, "_session_collision_geometry_reset_result", None)
            return cached or {
                "success": True,
                "performed": False,
                "reason": "previous-session collision geometry was already reset",
                "execution_sent": False,
                "collision_bypassed": False,
            }
        result = self._reset_previous_session_collision_geometry()
        self._session_collision_geometry_reset_result = result
        self._session_collision_geometry_reset_complete = bool(result.get("success"))
        return result

    def _retire_stale_human_scene_for_empty_view(self) -> dict[str, Any]:
        """Remove only prior-session human objects after fresh no-face evidence."""
        # Fast DDS graph discovery on the real stack can take just over two
        # seconds for a fresh short-lived workflow node.
        deadline = time.monotonic() + 4.0
        evidence = self._fresh_explicit_no_face_evidence()
        while (
            rclpy.ok()
            and not evidence.get("success")
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
            evidence = self._fresh_explicit_no_face_evidence()
        if not evidence.get("success"):
            return {
                "success": True,
                "retired": False,
                "reason": "current view is not a verified empty scene; preserving human objects",
                "empty_view_evidence": evidence,
                "execution_sent": False,
            }
        manager: PlanningSceneObstacleManager | None = None
        try:
            manager = PlanningSceneObstacleManager(
                PlanningSceneObstacleConfig(
                    base_frame=BASE_FRAME,
                    mouth_topic="/detected_mouth_pose",
                    include_table=False,
                    service_timeout_sec=5.0,
                    mouth_wait_timeout_sec=1.0,
                )
            )
            removal = manager.remove(verify=True)
        except Exception as exc:
            removal = {
                "success": False,
                "reason": f"stale human-scene retirement raised {exc.__class__.__name__}: {exc}",
            }
        finally:
            if manager is not None:
                manager.destroy_node()
        evidence_after = self._fresh_explicit_no_face_evidence()
        success = bool(removal.get("success") and evidence_after.get("success"))
        return {
            "success": success,
            "retired": bool(removal.get("success")),
            "reason": (
                removal.get("reason")
                if not removal.get("success")
                else None
                if evidence_after.get("success")
                else "empty-view evidence changed while stale objects were retired"
            ),
            "empty_view_evidence": evidence,
            "empty_view_evidence_after": evidence_after,
            "removal": removal,
            "execution_sent": False,
            "collision_bypassed": False,
        }

    def _restore_visible_human_scene_for_return(self) -> dict[str, Any]:
        """Rebuild missing human geometry before a return when a face is visible."""
        scene_before, _ = self._planning_scene_geometry()
        base: dict[str, Any] = {
            "success": False,
            "execution_sent": False,
            "collision_bypassed": False,
            "planning_scene_before": scene_before,
        }
        if not scene_before.get("available"):
            return {
                **base,
                "reason": (
                    scene_before.get("reason")
                    or "MoveIt PlanningScene is unavailable for human-scene reconciliation"
                ),
            }
        if scene_before.get("human_collision_objects_preserved"):
            return {
                **base,
                "success": True,
                "action": "existing_human_objects_preserved",
                "reason": None,
            }

        empty_view = self._fresh_explicit_no_face_evidence()
        base["empty_view_evidence"] = empty_view
        if empty_view.get("success"):
            return {
                **base,
                "success": True,
                "action": "verified_empty_view_kept_without_human_objects",
                "reason": None,
            }

        # A visible face after an earlier no-face cleanup is a normal scene
        # transition.  Rebuild the deterministic objects from a fresh stable
        # mouth window instead of either ignoring the person or deadlocking
        # the guarded return on missing prior-session geometry.
        snapshot = self.snapshot(
            mouth_sample_sec=max(0.50, self.mouth_sample_seconds),
            inspect_controllers=False,
        )
        base["visible_view_snapshot"] = snapshot
        mouth = snapshot.get("mouth_pose", {})
        if not mouth.get("available") or not mouth.get("stable"):
            return {
                **base,
                "reason": (
                    mouth.get("reason")
                    or "a stable visible mouth is unavailable for return-scene reconstruction"
                ),
            }
        application = self._apply_multi_person_planning_scene(snapshot)
        base["planning_scene_application"] = application
        if not application.get("success"):
            return {
                **base,
                "reason": (
                    application.get("reason")
                    or "visible human collision geometry could not be reconstructed"
                ),
            }
        scene_after, _ = self._planning_scene_geometry()
        base["planning_scene_after"] = scene_after
        success = bool(
            scene_after.get("available")
            and scene_after.get("human_collision_objects_preserved")
            and not scene_after.get("human_allowed_collision_pairs")
        )
        return {
            **base,
            "success": success,
            "action": "visible_human_objects_rebuilt" if success else None,
            "reason": (
                None
                if success
                else "reconstructed human collision geometry was not verified in MoveIt"
            ),
        }

    def _synchronize_servo_planning_scene(self) -> dict[str, Any]:
        """Publish MoveGroup's verified fixed scene to Servo's scene monitor."""
        base: dict[str, Any] = {
            "success": False,
            "topic": "/planning_scene",
            "required_subscribers": 2,
            "execution_sent": False,
        }
        deadline = time.monotonic() + 2.0
        subscriber_count = int(self.count_subscribers("/planning_scene"))
        while rclpy.ok() and subscriber_count < 2 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            subscriber_count = int(self.count_subscribers("/planning_scene"))
        base["subscriber_count"] = subscriber_count
        if subscriber_count < 2:
            return {
                **base,
                "reason": (
                    "MoveGroup and Servo PlanningScene subscribers are not both available"
                ),
            }
        if not self.get_planning_scene.wait_for_service(timeout_sec=2.0):
            return {**base, "reason": "/get_planning_scene is unavailable"}
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self.get_planning_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None:
            return {**base, "reason": "PlanningScene synchronization request timed out"}
        scene = response.scene
        scene.is_diff = True
        scene.robot_state.is_diff = True
        # Only attached bodies are synchronized.  Never overwrite either
        # monitor's live joint state with the service response snapshot.
        scene.robot_state.joint_state = JointState()
        world_ids = sorted(item.id for item in scene.world.collision_objects)
        attached_ids = sorted(
            item.object.id for item in scene.robot_state.attached_collision_objects
        )
        required_human_present = any(
            item.startswith("real_human_obstacle_") for item in world_ids
        )
        required_tool_present = (
            "combined_camera_cup_holder_straw_collision" in attached_ids
        )
        if not required_human_present or not required_tool_present:
            return {
                **base,
                "reason": "verified human/tool geometry is incomplete before Servo synchronization",
                "world_object_ids": world_ids,
                "attached_object_ids": attached_ids,
            }
        self._planning_scene_publisher.publish(scene)
        rclpy.spin_once(self, timeout_sec=0.10)
        return {
            **base,
            "success": True,
            "reason": None,
            "world_object_ids": world_ids,
            "attached_object_ids": attached_ids,
            "movegroup_and_servo_scene_shared": True,
        }

    def _continuous_servo_status_snapshot(self, *, now: float) -> dict[str, Any]:
        """Read the event-driven Servo status and verify publisher liveness."""
        publisher_count = int(self.count_publishers("/servo_node/status"))
        if publisher_count < 1:
            return {
                "success": False,
                "reason": "servo_status_publisher_unavailable",
                "publisher_count": publisher_count,
            }
        if self._latest_servo_status is None:
            return {
                "success": False,
                "reason": "servo_status_unavailable",
                "publisher_count": publisher_count,
            }
        last_event_age = (
            None
            if self._latest_servo_status_monotonic is None
            else max(0.0, float(now) - self._latest_servo_status_monotonic)
        )
        return {
            "success": True,
            "reason": None,
            "publisher_count": publisher_count,
            "code": int(self._latest_servo_status.code),
            "message": str(self._latest_servo_status.message),
            "last_event_age_sec": last_event_age,
            "delivery_policy": "reliable_transient_local_event_driven",
        }

    def _mouth_candidates_callback(self, message: String) -> None:
        """Feed the identity-locked candidate into the continuous filter."""
        super()._mouth_candidates_callback(message)
        now = time.monotonic()
        state = self.target_tracker.current_state(max_age_sec=1.0, now_monotonic=now)
        if not state.get("available"):
            self._continuous_last_observation_update = {
                "accepted": False,
                "reason": str(state.get("reason") or "selected_target_unavailable"),
            }
            return
        try:
            payload = json.loads(message.data)
            source_stamp = float(payload["stamp_sec"])
            selected_index = int(state["selected_candidate_index"])
            visible = state["visible_candidates"]
            depth = float(visible[selected_index]["depth_m"])
            position = [float(value) for value in state["selected_position_m"]]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._continuous_last_observation_update = {
                "accepted": False,
                "reason": f"continuous_observation_decode_failed:{exc}",
            }
            return
        accepted, reason = self._continuous_tracker.add_observation(
            position,
            source_timestamp_sec=source_stamp,
            received_monotonic_sec=now,
            depth_m=depth,
            confidence=1.0,
            target_id=self.target_selection,
        )
        self._continuous_last_observation_update = {
            "accepted": bool(accepted),
            "reason": reason,
            "source_timestamp_sec": source_stamp,
            "depth_m": depth,
            "confidence": 1.0,
            "confidence_source": "accepted_mediapipe_rgbd_observation",
        }

    @staticmethod
    def _search_waypoints_from_offsets(
        origin: list[float],
        tool_orientation_xyzw: list[float],
        camera_orientation_xyzw: list[float],
        offsets: tuple[tuple[str, tuple[float, float, float]], ...],
    ) -> list[dict[str, Any]]:
        if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
            raise ValueError("search origin must contain three finite values")
        for label, orientation in (
            ("tool0", tool_orientation_xyzw),
            (CAMERA_OPTICAL_FRAME, camera_orientation_xyzw),
        ):
            if len(orientation) != 4 or not all(
                math.isfinite(float(value)) for value in orientation
            ):
                raise ValueError(f"{label} orientation must contain four finite values")

        inverse_tool_orientation = [
            -float(tool_orientation_xyzw[0]),
            -float(tool_orientation_xyzw[1]),
            -float(tool_orientation_xyzw[2]),
            float(tool_orientation_xyzw[3]),
        ]
        waypoints: list[dict[str, Any]] = []
        for name, camera_offset in offsets:
            base_offset = _rotate_tool_vector(
                camera_orientation_xyzw,
                camera_offset,
            )
            tool_offset = _rotate_tool_vector(
                inverse_tool_orientation,
                base_offset,
            )
            waypoints.append(
                {
                    "name": name,
                    "search_motion_type": "cartesian_translation",
                    "direction_reference": (
                        f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
                    ),
                    "camera_extrinsic_applied": True,
                    "offset_camera_optical_m": [
                        float(value) for value in camera_offset
                    ],
                    "offset_initial_tool0_m": tool_offset,
                    "offset_from_origin_m": base_offset,
                    "target_tool0_position_m": _add(origin, base_offset),
                    "target_tool0_orientation_quat_xyzw": list(
                        tool_orientation_xyzw
                    ),
                    "tool_local_z_rotation_rad": 0.0,
                }
            )
        return waypoints

    @staticmethod
    def _rotation_search_waypoint(
        origin: list[float],
        tool_orientation_xyzw: list[float],
        *,
        name: str,
        angle_rad: float,
    ) -> dict[str, Any]:
        """Describe a local-tool-Z pose goal; no joint is selected directly."""
        return {
            "name": name,
            "search_motion_type": "tool_local_z_rotation",
            "direction_reference": f"{TOOL_FRAME} local +Z axis",
            "camera_extrinsic_applied": False,
            "offset_camera_optical_m": [0.0, 0.0, 0.0],
            "offset_initial_tool0_m": [0.0, 0.0, 0.0],
            "offset_from_origin_m": [0.0, 0.0, 0.0],
            "target_tool0_position_m": list(origin),
            "target_tool0_orientation_quat_xyzw": (
                _orientation_after_local_tool_z_rotation(
                    tool_orientation_xyzw,
                    angle_rad,
                )
            ),
            "tool_local_z_rotation_rad": float(angle_rad),
            "tool_local_z_rotation_deg": math.degrees(float(angle_rad)),
            "wrist_3_direct_command": False,
            "joint_selection": "MoveIt pose-goal IK and planning",
        }

    @staticmethod
    def search_waypoints(
        origin: list[float],
        tool_orientation_xyzw: list[float],
        camera_orientation_xyzw: list[float],
    ) -> list[dict[str, Any]]:
        """Return camera-corrected directions frozen at the initial flange pose.

        Optical -Z moves backward for a wider frame and optical -/+Y scans
        image up/down. Left/right are absolute +/-15 degree rotations from the
        frozen initial orientation about tool0 local Z. Those rotations are
        pose goals, so MoveIt—not this workflow—selects wrist_3_joint motion.
        """
        translations = RealIntegratedFeedWater._search_waypoints_from_offsets(
            origin,
            tool_orientation_xyzw,
            camera_orientation_xyzw,
            SEARCH_OFFSETS_CAMERA_OPTICAL,
        )
        by_name = {item["name"]: item for item in translations}
        left = RealIntegratedFeedWater._rotation_search_waypoint(
            origin,
            tool_orientation_xyzw,
            name="scan_left",
            angle_rad=math.radians(SEARCH_WRIST_Z_ANGLE_DEG),
        )
        right = RealIntegratedFeedWater._rotation_search_waypoint(
            origin,
            tool_orientation_xyzw,
            name="scan_right",
            angle_rad=-math.radians(SEARCH_WRIST_Z_ANGLE_DEG),
        )
        return [
            by_name["backward_wide"],
            left,
            right,
            by_name["scan_up"],
            by_name["scan_down"],
        ]

    @staticmethod
    def search_waypoint_variants(
        origin: list[float],
        tool_orientation_xyzw: list[float],
        camera_orientation_xyzw: list[float],
        *,
        name: str,
        back_distance_m: float,
    ) -> list[dict[str, Any]]:
        """Return nominal-to-small bounded alternatives for one search direction."""
        if name == "backward_wide":
            offsets = tuple(
                (name, (0.0, 0.0, -distance))
                for distance in SEARCH_BACK_CANDIDATE_DISTANCES_M
            )
        elif name in ("scan_left", "scan_right"):
            sign = 1.0 if name == "scan_left" else -1.0
            variants = [
                RealIntegratedFeedWater._rotation_search_waypoint(
                    origin,
                    tool_orientation_xyzw,
                    name=name,
                    angle_rad=sign * angle,
                )
                for angle in SEARCH_WRIST_Z_CANDIDATE_ANGLES_RAD
            ]
            for index, variant in enumerate(variants):
                variant["adaptive_candidate_index"] = index
                variant["adaptive_scale_applied"] = index > 0
            return variants
        else:
            axes = {
                "scan_up": (0.0, -1.0),
                "scan_down": (0.0, 1.0),
            }
            if name not in axes:
                raise ValueError(f"unsupported search waypoint {name!r}")
            x_sign, y_sign = axes[name]
            offsets = tuple(
                (
                    name,
                    (
                        x_sign * distance,
                        y_sign * distance,
                        -float(back_distance_m),
                    ),
                )
                for distance in SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M
            )
        variants = RealIntegratedFeedWater._search_waypoints_from_offsets(
            origin,
            tool_orientation_xyzw,
            camera_orientation_xyzw,
            offsets,
        )
        for index, variant in enumerate(variants):
            variant["adaptive_candidate_index"] = index
            variant["adaptive_scale_applied"] = index > 0
        return variants

    @staticmethod
    def _base_readiness_failures(snapshot: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("real /joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("real TF base_link -> tool0 is unavailable")
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
            failures.append("real TF base -> base_link is unavailable")
        if not snapshot["camera_tf"].get("available"):
            failures.append(f"real TF base_link -> {CAMERA_OPTICAL_FRAME} is unavailable")
        if not snapshot.get("mount_calibration", {}).get("corrected_physical_profile"):
            failures.append("corrected physical D435i mount calibration is not loaded")
        if not snapshot.get("camera_mount_match", {}).get("matches"):
            failures.append(
                snapshot.get("camera_mount_match", {}).get("reason")
                or "live D435i mount TF does not match the corrected calibration"
            )
        if not snapshot.get("move_group_available"):
            failures.append("the real-UR10e MoveGroup action is unavailable")
        return failures

    def _execution_state_failures(self, *, confirm_real_motion: bool) -> list[str]:
        failures: list[str] = []
        if self.target_selection != "center":
            failures.append("guarded physical feed_water execution remains center-target only")
        if not confirm_real_motion:
            failures.append("--confirm-real-motion is required")
        if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
            failures.append("UR10E_ALLOW_REAL_EXECUTION=1 is required")
        controllers = self._controller_status()
        if not controllers.get("scaled_joint_trajectory_controller_active"):
            failures.append("scaled_joint_trajectory_controller is not active")
        speed = None if self.latest_speed_scaling is None else float(self.latest_speed_scaling.data)
        if speed is None or not MIN_EXECUTION_SPEED_PERCENT <= speed <= MAX_EXECUTION_SPEED_PERCENT:
            failures.append(
                "speed slider is unavailable or outside the required "
                f"{MIN_EXECUTION_SPEED_PERCENT:.0f}%–{MAX_EXECUTION_SPEED_PERCENT:.0f}% range"
            )
        if self.latest_robot_program_running is None or not self.latest_robot_program_running.data:
            failures.append("UR External Control program is not Running")
        if not bool(
            self.latest_safety_mode is not None
            and int(self.latest_safety_mode.mode) == int(SafetyMode.NORMAL)
        ):
            failures.append("UR safety mode is not NORMAL")
        if not bool(
            self.latest_robot_mode is not None
            and int(self.latest_robot_mode.mode) == int(RobotMode.RUNNING)
        ):
            failures.append("UR robot mode is not RUNNING")
        return failures

    def _live_ur_execution_state_failure(self) -> str | None:
        """Return the first lightweight in-motion UR state failure."""
        if (
            self.latest_robot_program_running is None
            or not self.latest_robot_program_running.data
        ):
            return "UR External Control program stopped during execution"
        if not bool(
            self.latest_safety_mode is not None
            and int(self.latest_safety_mode.mode) == int(SafetyMode.NORMAL)
        ):
            return "UR safety mode left NORMAL during execution"
        if not bool(
            self.latest_robot_mode is not None
            and int(self.latest_robot_mode.mode) == int(RobotMode.RUNNING)
        ):
            return "UR robot mode left RUNNING during execution"
        return None

    @staticmethod
    def _fixed_initial_robot_state(config: dict[str, Any]) -> RobotState:
        state = RobotState()
        state.joint_state = JointState()
        state.joint_state.name = list(config["joint_names"])
        state.joint_state.position = list(config["joint_positions_rad"])
        # Keep the monitored PlanningScene's attached collision geometry while
        # replacing the manipulator joints with this fixed candidate state.
        state.is_diff = True
        return state

    def _joint_goal_for_initial_position(
        self,
        config: dict[str, Any],
        target_pose: dict[str, Any],
    ) -> MoveGroup.Goal:
        constraints = Constraints()
        constraints.name = "fixed_initial_position_joint_goal"
        for name, position in zip(
            config["joint_names"], config["joint_positions_rad"]
        ):
            joint = JointConstraint()
            joint.joint_name = str(name)
            joint.position = float(position)
            joint.tolerance_above = RETURN_JOINT_GOAL_TOLERANCE_RAD
            joint.tolerance_below = RETURN_JOINT_GOAL_TOLERANCE_RAD
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target_pose["position_m"]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = target_pose["orientation_quat_xyzw"]
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.pipeline_id = OMPL_PIPELINE
        goal.request.planner_id = OMPL_PLANNER
        goal.request.num_planning_attempts = RETURN_PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = RETURN_PLANNING_TIME_SEC
        goal.request.max_velocity_scaling_factor = self.trajectory_velocity_scaling
        goal.request.max_acceleration_scaling_factor = (
            self.trajectory_acceleration_scaling
        )
        if self.latest_joint_state is not None:
            goal.request.start_state.joint_state = self.latest_joint_state
            goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(constraints)
        goal.request.path_constraints.name = (
            "return_tool_vertical_axis_with_free_spin"
        )
        goal.request.path_constraints.orientation_constraints.append(
            self._vertical_axis_constraint(pose)
        )
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _prepare_initial_position_target(
        self,
        *,
        require_octomap: bool = True,
    ) -> dict[str, Any]:
        """Verify the immutable joint target, FK reference, scene, and states."""
        result: dict[str, Any] = {
            "success": False,
            "stage": "initial_position_validation",
            "execution_sent": False,
            "wrist_3_direct_command": False,
            "collision_checking_required": True,
            "vertical_axis_constraint_active": True,
        }
        try:
            config = _load_initial_position_config()
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {**result, "reason": f"initial position configuration is invalid: {exc}"}
        self._wait_for_joint_state()
        self._spin_for(0.2)
        current_state = self._current_robot_state()
        base_tf = self._frame_transform(UR_BASE_FRAME, BASE_FRAME)
        target_state = self._fixed_initial_robot_state(config)
        fk = self._fk_positions(target_state, (TOOL_FRAME,))
        scene, _ = self._planning_scene_geometry()
        current_validity = self._state_validity(
            current_state,
            label="return_start_state",
        )
        target_validity = self._state_validity(
            target_state,
            label="fixed_initial_position_state",
        )
        result.update(
            {
                "config": config,
                "ur_base_tf": base_tf,
                "target_fk": fk,
                "planning_scene": scene,
                "start_state_validity": current_validity,
                "target_state_validity": target_validity,
            }
        )
        failures: list[str] = []
        if not base_tf.get("available"):
            failures.append("base <- base_link transform is unavailable")
        if not fk.get("available") or TOOL_FRAME not in fk.get("poses", {}):
            failures.append(fk.get("reason") or "fixed initial-position FK is unavailable")
        if not scene.get("available"):
            failures.append(scene.get("reason") or "MoveIt PlanningScene is unavailable")
        else:
            if not scene.get("human_collision_objects_preserved"):
                empty_view = self._fresh_explicit_no_face_evidence()
                result["empty_human_scene_evidence"] = empty_view
                if not empty_view.get("success"):
                    failures.append(
                        "fixed human head/torso/face objects are absent without fresh no-face evidence"
                    )
            if scene.get("human_allowed_collision_pairs"):
                failures.append("human allowed-collision entries are present")
            if not scene.get("combined_tool_collision_geometry", {}).get("success"):
                failures.append("combined camera/cup-holder/straw geometry is not verified")
            if require_octomap and not scene.get("octomap", {}).get("present"):
                failures.append("the current dynamic OctoMap is absent")
        if not current_validity.get("valid"):
            failures.append(
                current_validity.get("reason") or "return start state is invalid"
            )
        if not target_validity.get("valid"):
            failures.append(
                target_validity.get("reason") or "fixed initial-position state is invalid"
            )
        if failures:
            return {**result, "failures": failures, "reason": "; ".join(failures)}

        target_pose = {
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            **fk["poses"][TOOL_FRAME],
        }
        try:
            recorded_pose = _recorded_tool_pose_in_base_link(config)
            fk_position_error = _norm(
                _subtract(target_pose["position_m"], recorded_pose["position_m"])
            )
            fk_orientation_error = _quaternion_distance_rad(
                target_pose["orientation_quat_xyzw"],
                recorded_pose["orientation_quat_xyzw"],
            )
            target_tilt = _tool_vertical_tilt_rad(
                target_pose["orientation_quat_xyzw"]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return {**result, "reason": f"initial-position pose verification failed: {exc}"}
        target_in_ur_base = self._point_in_ur_base(
            target_pose["position_m"], base_tf
        )
        target_radius = _norm(target_in_ur_base)
        limits = config["verification"]
        result.update(
            {
                "target_tool0_pose": target_pose,
                "moveit_tool0_fk_reference": recorded_pose,
                "operator_displayed_tool_pose": config[
                    "operator_displayed_tool_pose"
                ],
                "fk_reference_position_error_m": fk_position_error,
                "fk_reference_orientation_error_rad": fk_orientation_error,
                "fk_reference_orientation_error_deg": math.degrees(
                    fk_orientation_error
                ),
                "target_tool_vertical_tilt_rad": target_tilt,
                "target_tool_vertical_tilt_deg": math.degrees(target_tilt),
                "target_tool0_position_in_ur_base_m": target_in_ur_base,
                "target_tool0_radius_from_ur_base_m": target_radius,
            }
        )
        if fk_position_error > limits["maximum_fk_position_error_m"]:
            failures.append(
                "configured joints disagree with the versioned MoveIt FK position: "
                f"{fk_position_error:.4f} m > "
                f"{limits['maximum_fk_position_error_m']:.4f} m"
            )
        if fk_orientation_error > math.radians(
            limits["maximum_fk_orientation_error_deg"]
        ):
            failures.append(
                "configured joints disagree with the versioned MoveIt FK orientation: "
                f"{math.degrees(fk_orientation_error):.2f} deg > "
                f"{limits['maximum_fk_orientation_error_deg']:.2f} deg"
            )
        maximum_tilt = min(
            MAX_TOOL_VERTICAL_TILT_RAD,
            math.radians(limits["maximum_tool_vertical_tilt_deg"]),
        )
        if target_tilt > maximum_tilt:
            failures.append(
                f"initial-position tool tilt is {math.degrees(target_tilt):.2f} deg, "
                f"above the {math.degrees(maximum_tilt):.2f} deg limit"
            )
        if target_radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            failures.append(
                "initial-position target is outside the UR10e reach envelope"
            )
        if failures:
            return {**result, "failures": failures, "reason": "; ".join(failures)}
        result.update(
            {
                "success": True,
                "reason": None,
                "fixed_target_verified": True,
            }
        )
        self._initial_target_robot_state = target_state
        return result

    def _current_initial_position_status(self) -> dict[str, Any]:
        """Compare fresh live UR joints with the immutable initial target."""
        base: dict[str, Any] = {
            "available": False,
            "at_initial_position": False,
            "stage": "current_initial_position_check",
        }
        try:
            config = _load_initial_position_config()
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {**base, "reason": f"initial position configuration is invalid: {exc}"}
        self._wait_for_joint_state()
        self._spin_for(0.1)
        state = self.latest_joint_state
        if state is None:
            return {**base, "reason": "live joint state is unavailable"}
        positions = {
            str(name): float(position)
            for name, position in zip(state.name, state.position)
        }
        missing = [name for name in config["joint_names"] if name not in positions]
        if missing:
            return {
                **base,
                "reason": "live joint state is missing: " + ", ".join(missing),
                "missing_joint_names": missing,
            }
        errors = {
            name: abs(
                math.atan2(
                    math.sin(positions[name] - float(target)),
                    math.cos(positions[name] - float(target)),
                )
            )
            for name, target in zip(
                config["joint_names"],
                config["joint_positions_rad"],
            )
        }
        maximum_error = max(errors.values(), default=float("inf"))
        tolerance = math.radians(
            config["verification"]["maximum_final_joint_error_deg"]
        )
        at_initial = maximum_error <= tolerance
        return {
            **base,
            "available": True,
            "at_initial_position": at_initial,
            "joint_errors_rad": errors,
            "maximum_joint_error_rad": maximum_error,
            "maximum_joint_error_deg": math.degrees(maximum_error),
            "maximum_allowed_joint_error_rad": tolerance,
            "maximum_allowed_joint_error_deg": math.degrees(tolerance),
            "reason": None if at_initial else "live joints are not at initial_position",
        }

    def _attempt_failure_recovery_return(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        motion_sent: bool,
        use_octomap: bool = True,
    ) -> dict[str, Any]:
        """Use the guarded return after a partially executed failed workflow."""
        report: dict[str, Any] = {
            "success": False,
            "attempted": False,
            "motion_was_sent_before_failure": bool(motion_sent),
            "execution_sent": False,
        }
        if not execute or not motion_sent:
            report.update(
                {
                    "reason": (
                        "no real motion preceded the failure"
                        if execute
                        else "plan-only mode cannot execute recovery"
                    ),
                    "final_state": "unchanged",
                }
            )
            return report
        stationary = self._wait_for_tracking_replan_stationary()
        report["stationary_wait"] = stationary
        if not stationary.get("success"):
            report.update(
                {
                    "reason": "UR10e did not become stationary before recovery return",
                    "final_state": "stopped_after_failure",
                }
            )
            return report
        return_kwargs = {
            "execute": True,
            "confirm_real_motion": confirm_real_motion,
        }
        if not use_octomap:
            return_kwargs["use_octomap"] = False
        code, return_result = self.return_to_initial_position(**return_kwargs)
        verification = self._current_initial_position_status()
        success = bool(
            code == 0
            and return_result.get("success")
            and verification.get("at_initial_position")
        )
        report.update(
            {
                "success": success,
                "attempted": True,
                "return_result": return_result,
                "post_return_initial_position": verification,
                "execution_sent": bool(return_result.get("execution_sent")),
                "reason": None
                if success
                else (
                    return_result.get("reason")
                    or verification.get("reason")
                    or "guarded failure-recovery return was not verified"
                ),
                "final_state": (
                    "initial_position" if success else "stopped_after_failure"
                ),
            }
        )
        return report

    def _validate_trajectory_collision_states(
        self,
        trajectory: Any,
    ) -> dict[str, Any]:
        """Recheck every cached return waypoint against the latest MoveIt scene."""
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        names = list(getattr(joint_trajectory, "joint_names", []))
        points = list(getattr(joint_trajectory, "points", []))
        base: dict[str, Any] = {
            "success": False,
            "stage": "return_trajectory_collision_validation",
            "trajectory_waypoints": len(points),
            "sampled_waypoints": 0,
            "collision_checking_required": True,
        }
        if not names or not points:
            return {**base, "reason": "return trajectory has no joint waypoints"}
        for index, point in enumerate(points):
            positions = [float(value) for value in point.positions]
            if len(positions) != len(names) or not all(
                math.isfinite(value) for value in positions
            ):
                return {
                    **base,
                    "sampled_waypoints": index,
                    "reason": f"return trajectory waypoint {index} is invalid",
                }
            state = RobotState()
            state.joint_state = JointState()
            state.joint_state.name = names
            state.joint_state.position = positions
            state.is_diff = True
            validity = self._state_validity(
                state,
                label=f"return_trajectory_waypoint_{index}",
            )
            if not validity.get("available") or not validity.get("valid"):
                return {
                    **base,
                    "sampled_waypoints": index + 1,
                    "rejected_waypoint_index": index,
                    "rejected_state_validity": validity,
                    "collision_pairs": list(validity.get("collision_pairs", [])),
                    "reason": (
                        validity.get("reason")
                        or f"return trajectory waypoint {index} is not collision-free"
                    ),
                }
        return {
            **base,
            "success": True,
            "sampled_waypoints": len(points),
            "collision_pairs": [],
            "reason": None,
        }

    def _plan_return_to_initial_position(
        self,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        """Prefer a complete Cartesian route, then a constrained OMPL route."""
        base: dict[str, Any] = {
            "success": False,
            "stage": "return_plan",
            "execution_sent": False,
            "collision_checking_required": True,
            "vertical_axis_constraint_active": True,
            "tool_axis_spin_free": True,
            "wrist_3_direct_command": False,
        }
        if not prepared.get("success"):
            return {**base, "reason": "fixed initial-position target is not valid"}
        config = prepared["config"]
        target_pose = prepared["target_tool0_pose"]
        targets = dict(
            zip(config["joint_names"], config["joint_positions_rad"])
        )
        tolerance = math.radians(
            config["verification"]["maximum_final_joint_error_deg"]
        )
        cartesian = self._run_cartesian_plan(target_pose)
        cartesian_trajectory = self._validated_trajectory
        cartesian_joint_error = _trajectory_final_joint_error(
            cartesian_trajectory,
            targets,
        ) if cartesian_trajectory is not None else {
            "available": False,
            "reason": "no validated Cartesian trajectory",
        }
        cartesian_exact = bool(
            cartesian.get("success")
            and cartesian_joint_error.get("available")
            and float(cartesian_joint_error["maximum_joint_error_rad"]) <= tolerance
        )
        if cartesian_exact:
            return {
                **base,
                "success": True,
                "reason": None,
                "route_strategy": "complete_collision_checked_cartesian_return",
                "planner": "moveit_compute_cartesian_path",
                "cartesian_plan": cartesian,
                "cartesian_final_joint_error": cartesian_joint_error,
                "ompl_needed": False,
                "validated_trajectory": _trajectory_summary(
                    self._validated_trajectory
                ),
            }

        # Do not execute a Cartesian IK branch that reaches the pose with the
        # wrong configured joints. Ask MoveIt for the exact fixed joint goal.
        self._validated_trajectory = None
        ompl = self._run_goal(
            self._joint_goal_for_initial_position(config, target_pose)
        )
        ompl_trajectory = self._validated_trajectory
        ompl_joint_error = _trajectory_final_joint_error(
            ompl_trajectory,
            targets,
        ) if ompl_trajectory is not None else {
            "available": False,
            "reason": "no validated OMPL trajectory",
        }
        ompl_exact = bool(
            ompl.get("success")
            and ompl_joint_error.get("available")
            and float(ompl_joint_error["maximum_joint_error_rad"]) <= tolerance
        )
        if not ompl_exact:
            self._validated_trajectory = None
        return {
            **base,
            "success": ompl_exact,
            "reason": None
            if ompl_exact
            else "MoveIt could not produce a collision-free, flange-down route to the fixed initial joints",
            "route_strategy": (
                "collision_checked_ompl_joint_return" if ompl_exact else None
            ),
            "planner": f"{OMPL_PIPELINE}/{OMPL_PLANNER}",
            "cartesian_plan": cartesian,
            "cartesian_final_joint_error": cartesian_joint_error,
            "ompl_plan": ompl,
            "ompl_final_joint_error": ompl_joint_error,
            "ompl_needed": True,
            "validated_trajectory": _trajectory_summary(
                self._validated_trajectory
            ) if self._validated_trajectory is not None else None,
        }

    def return_to_initial_position(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        use_octomap: bool = True,
    ) -> tuple[int, dict[str, Any]]:
        """Validate, plan, and optionally execute the one configured return."""
        response: dict[str, Any] = {
            "success": False,
            "stage": "return_to_initial_position",
            "mode": "execute" if execute else "plan",
            "execution_attempted": False,
            "execution_sent": False,
            "automatic_retreat_sent": False,
            "collision_checking_required": True,
            "vertical_axis_constraint_active": True,
            "wrist_3_direct_command": False,
        }
        stale_scene = (
            self._retire_stale_human_scene_for_empty_view()
            if execute
            else {
                "success": True,
                "retired": False,
                "reason": "plan-only mode does not mutate stale scene objects",
                "execution_sent": False,
            }
        )
        response["stale_human_scene_lifecycle"] = stale_scene
        if not stale_scene.get("success"):
            response.update(
                stage="stale_human_scene_retirement_refused",
                reason=stale_scene.get("reason"),
            )
            return 2, response
        combined = self._apply_combined_tool_collision_geometry()
        human_scene = (
            self._restore_visible_human_scene_for_return()
            if execute
            else {
                "success": True,
                "action": "plan_only_scene_mutation_deferred",
                "execution_sent": False,
                "collision_bypassed": False,
                "reason": None,
            }
        )
        response["human_scene_reconciliation"] = human_scene
        if not human_scene.get("success"):
            response.update(
                stage="return_human_scene_reconciliation_refused",
                reason=(
                    human_scene.get("reason")
                    or "human collision scene could not be reconciled for return"
                ),
            )
            return 2, response
        dynamic = (
            self.dynamic_readiness(execution_mode=True if execute else None)
            if use_octomap
            else {
                "success": True,
                "use_octomap": False,
                "dynamic_obstacle_layer_active": False,
                "degraded": False,
                "status": "dynamic_obstacle_layer_disabled",
                "failures": [],
            }
        )
        prepared = (
            self._prepare_initial_position_target()
            if use_octomap
            else self._prepare_initial_position_target(require_octomap=False)
        )
        response.update(
            {
                "combined_tool_collision_geometry": combined,
                "dynamic_octomap_readiness": dynamic,
                "target_validation": prepared,
            }
        )
        failures: list[str] = []
        if not combined.get("success"):
            failures.append(
                combined.get("reason") or "attached tool collision geometry is invalid"
            )
        if not dynamic.get("success"):
            failures.extend(str(item) for item in dynamic.get("failures", []))
        if not prepared.get("success"):
            failures.append(
                prepared.get("reason") or "fixed initial-position target is invalid"
            )
        if failures:
            response.update(
                {
                    "stage": "return_target_or_scene_refused",
                    "failures": failures,
                    "reason": "; ".join(failures),
                }
            )
            return 2, response
        if not execute:
            response.update(
                {
                    "success": True,
                    "stage": "return_target_validated",
                    "reason": None,
                    "route_planning_deferred": True,
                    "route_planning_deferred_reason": (
                        "the sequential return must be planned from the actual post-hold state"
                    ),
                }
            )
            return 0, response

        execution_failures = self._execution_state_failures(
            confirm_real_motion=confirm_real_motion
        )
        if execution_failures:
            response.update(
                {
                    "stage": "return_execution_readiness",
                    "failures": execution_failures,
                    "reason": "; ".join(execution_failures),
                }
            )
            return 2, response
        plan = self._plan_return_to_initial_position(prepared)
        response["plan_result"] = plan
        if not plan.get("success") or self._validated_trajectory is None:
            response.update(
                {
                    "stage": "return_planning_refused",
                    "reason": plan.get("reason") or "validated return trajectory is unavailable",
                }
            )
            return 2, response

        # Refresh every live execution gate and both endpoint states after the
        # plan. Refuse if the scene or controller changed before dispatch.
        self._spin_for(0.2)
        pre_execution_failures = self._execution_state_failures(
            confirm_real_motion=confirm_real_motion
        )
        latest_dynamic = (
            self.dynamic_readiness(execution_mode=True)
            if use_octomap
            else {
                "success": True,
                "use_octomap": False,
                "dynamic_obstacle_layer_active": False,
                "degraded": False,
                "status": "dynamic_obstacle_layer_disabled",
                "failures": [],
            }
        )
        latest_scene, _ = self._planning_scene_geometry()
        latest_start_validity = self._state_validity(
            self._current_robot_state(),
            label="return_pre_execution_start_state",
        )
        latest_target_validity = self._state_validity(
            getattr(self, "_initial_target_robot_state", None),
            label="return_pre_execution_target_state",
        )
        response.update(
            {
                "pre_execution_dynamic_octomap_readiness": latest_dynamic,
                "pre_execution_planning_scene": latest_scene,
                "pre_execution_start_state_validity": latest_start_validity,
                "pre_execution_target_state_validity": latest_target_validity,
            }
        )
        if not latest_dynamic.get("success"):
            pre_execution_failures.extend(
                str(item) for item in latest_dynamic.get("failures", [])
            )
        if not latest_scene.get("available"):
            pre_execution_failures.append(
                latest_scene.get("reason")
                or "MoveIt PlanningScene is unavailable before return execution"
            )
        else:
            if not latest_scene.get("human_collision_objects_preserved"):
                latest_empty_view = self._fresh_explicit_no_face_evidence()
                response["pre_execution_empty_human_scene_evidence"] = (
                    latest_empty_view
                )
                if not latest_empty_view.get("success"):
                    pre_execution_failures.append(
                        "human objects are absent and fresh no-face evidence disappeared before return execution"
                    )
            if latest_scene.get("human_allowed_collision_pairs"):
                pre_execution_failures.append(
                    "human allowed-collision entries appeared before return execution"
                )
            if not latest_scene.get("combined_tool_collision_geometry", {}).get(
                "success"
            ):
                pre_execution_failures.append(
                    "combined camera/cup-holder/straw geometry is invalid before return execution"
                )
            if use_octomap and not latest_scene.get("octomap", {}).get("present"):
                pre_execution_failures.append(
                    "the dynamic OctoMap is absent before return execution"
                )
        if not latest_start_validity.get("valid"):
            pre_execution_failures.append(
                latest_start_validity.get("reason")
                or "return pre-execution start state is invalid"
            )
        if not latest_target_validity.get("valid"):
            pre_execution_failures.append(
                latest_target_validity.get("reason")
                or "return pre-execution target state is invalid"
            )
        if pre_execution_failures:
            response.update(
                {
                    "stage": "return_pre_execution_refused",
                    "failures": pre_execution_failures,
                    "reason": "; ".join(pre_execution_failures),
                }
            )
            return 2, response

        trajectory_collision_validation = self._validate_trajectory_collision_states(
            self._validated_trajectory
        )
        response["pre_execution_trajectory_collision_validation"] = (
            trajectory_collision_validation
        )
        if not trajectory_collision_validation.get("success"):
            response.update(
                {
                    "stage": "return_pre_execution_trajectory_refused",
                    "reason": (
                        trajectory_collision_validation.get("reason")
                        or "the cached return trajectory is invalid in the latest scene"
                    ),
                }
            )
            return 2, response

        execution = RealPreMouthFromPerceptionPlan._execute_validated_trajectory(
            self
        )
        self._spin_for(0.2)
        actual_pose = self._tool0_pose()
        target_pose = prepared["target_tool0_pose"]
        config = prepared["config"]
        target_joints = dict(
            zip(config["joint_names"], config["joint_positions_rad"])
        )
        current_joints = {}
        if self.latest_joint_state is not None:
            current_joints = dict(
                zip(self.latest_joint_state.name, self.latest_joint_state.position)
            )
        joint_errors = {
            name: abs(
                math.atan2(
                    math.sin(float(current_joints[name]) - float(target)),
                    math.cos(float(current_joints[name]) - float(target)),
                )
            )
            for name, target in target_joints.items()
            if name in current_joints
        }
        final_joint_error = (
            max(joint_errors.values())
            if len(joint_errors) == len(target_joints)
            else float("inf")
        )
        if actual_pose.get("available"):
            final_position_error = _norm(
                _subtract(actual_pose["position_m"], target_pose["position_m"])
            )
            final_orientation_error = _quaternion_distance_rad(
                actual_pose["orientation_quat_xyzw"],
                target_pose["orientation_quat_xyzw"],
            )
            try:
                final_tilt = _tool_vertical_tilt_rad(
                    actual_pose["orientation_quat_xyzw"]
                )
            except (RuntimeError, TypeError, ValueError):
                final_tilt = float("inf")
        else:
            final_position_error = float("inf")
            final_orientation_error = float("inf")
            final_tilt = float("inf")
        limits = config["verification"]
        verified = bool(
            execution.get("success")
            and final_joint_error
            <= math.radians(limits["maximum_final_joint_error_deg"])
            and final_position_error
            <= limits["maximum_final_tool_position_error_m"]
            and final_orientation_error
            <= math.radians(limits["maximum_final_tool_orientation_error_deg"])
            and final_tilt <= MAX_TOOL_VERTICAL_TILT_RAD
        )
        response.update(
            {
                "success": verified,
                "stage": (
                    "returned_initial_position"
                    if verified
                    else "return_execution_or_verification_failed"
                ),
                "reason": None
                if verified
                else (
                    execution.get("reason")
                    or "return execution did not finish within the fixed target tolerances"
                ),
                "execution_result": execution,
                "execution_attempted": bool(execution.get("execution_attempted")),
                "execution_sent": bool(execution.get("execution_attempted")),
                "automatic_retreat_sent": bool(
                    execution.get("execution_attempted")
                ),
                "actual": {
                    "tool0_pose": actual_pose,
                    "joint_errors_rad": joint_errors,
                    "maximum_joint_error_rad": final_joint_error,
                    "maximum_joint_error_deg": math.degrees(final_joint_error),
                    "tool0_position_error_m": final_position_error,
                    "tool0_orientation_error_rad": final_orientation_error,
                    "tool0_orientation_error_deg": math.degrees(
                        final_orientation_error
                    ),
                    "tool_vertical_tilt_rad": final_tilt,
                    "tool_vertical_tilt_deg": math.degrees(final_tilt),
                },
            }
        )
        return (0 if verified else 2), response

    def _explicit_no_face(self) -> bool:
        status = self.latest_mouth_status
        return bool(
            isinstance(status, dict)
            and status.get("detected") is False
            and status.get("reason") == "no_face"
        )

    def _candidate_visible(self) -> bool:
        status = self.latest_mouth_status
        return bool(isinstance(status, dict) and status.get("detected") is True)

    def _selected_observation(self, started: float) -> dict[str, Any]:
        return self.target_tracker.observation(
            started_monotonic=started,
            now_monotonic=time.monotonic(),
            max_age_sec=MAX_MOUTH_POSE_AGE_SEC,
            minimum_samples=MIN_STABLE_SAMPLES,
            max_spread_m=MAX_POSE_SPREAD_M,
        )

    def _wait_for_selected_stability(self, started: float, deadline: float) -> dict[str, Any]:
        result = self._selected_observation(started)
        while rclpy.ok() and time.monotonic() < deadline:
            if result.get("available") and result.get("stable"):
                return result
            if result.get("identity_unsafe"):
                return result
            rclpy.spin_once(
                self,
                timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())),
            )
            result = self._selected_observation(started)
        return result

    def _search_goal_for_target(self, target: dict[str, Any]):
        """Build a translation or tool-local-Z rotation active-search goal."""
        goal = RealPreMouthFromPerceptionPlan._goal_for_target(self, target)
        # Active-search moves are short, bounded Cartesian pose changes.  Use
        # Pilz LIN here so a background OMPL solve cannot consume the search
        # deadline or preempt the next adaptive candidate.  OMPL remains the
        # alternate-path planner for the later dynamic-obstacle stage.
        goal.request.pipeline_id = SEARCH_PLANNING_PIPELINE
        goal.request.planner_id = SEARCH_PLANNER
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = SEARCH_ALLOWED_PLANNING_TIME_SEC
        rotation_goal = target.get("search_motion_type") == "tool_local_z_rotation"
        for constraints in goal.request.goal_constraints:
            if rotation_goal:
                constraints.name = (
                    "active_search_tool_local_z_pose_goal_with_vertical_path"
                )
            else:
                source = constraints.orientation_constraints[0]
                pose = Pose()
                pose.orientation = source.orientation
                constraints.orientation_constraints.clear()
                constraints.orientation_constraints.append(
                    self._vertical_axis_constraint(pose)
                )
                constraints.name = (
                    "active_search_position_and_vertical_axis"
                )
        goal.request.path_constraints.name = (
            "vertical_axis_intermediate_active_search"
        )
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    @staticmethod
    def _validate_search_joint_motion(trajectory: Any) -> dict[str, Any]:
        """Reject a small visual-search goal that hides a large joint route."""
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        names = list(getattr(joint_trajectory, "joint_names", []))
        points = list(getattr(joint_trajectory, "points", []))
        result: dict[str, Any] = {
            "success": False,
            "maximum_joint_excursion_rad": None,
            "maximum_allowed_joint_excursion_rad": (
                SEARCH_MAX_JOINT_EXCURSION_RAD
            ),
            "cumulative_joint_travel_rad": None,
            "maximum_cumulative_joint_travel_rad": (
                SEARCH_MAX_CUMULATIVE_JOINT_TRAVEL_RAD
            ),
            "trajectory_duration_sec": None,
            "maximum_trajectory_duration_sec": (
                SEARCH_MAX_TRAJECTORY_DURATION_SEC
            ),
        }
        if not names or not points:
            result["reason"] = "planned search trajectory has no joint waypoints"
            return result
        positions = [
            [float(value) for value in point.positions]
            for point in points
        ]
        if any(
            len(values) != len(names)
            or not all(math.isfinite(value) for value in values)
            for values in positions
        ):
            result["reason"] = "planned search trajectory has invalid joint positions"
            return result
        start = positions[0]
        excursions = {
            name: max(abs(values[index] - start[index]) for values in positions)
            for index, name in enumerate(names)
        }
        maximum_excursion = max(excursions.values(), default=0.0)
        cumulative_travel = sum(
            abs(current[index] - previous[index])
            for previous, current in zip(positions, positions[1:])
            for index in range(len(names))
        )
        final_time = points[-1].time_from_start
        duration = float(final_time.sec) + float(final_time.nanosec) * 1e-9
        result.update(
            {
                "maximum_joint_excursion_rad": maximum_excursion,
                "joint_excursions_rad": excursions,
                "cumulative_joint_travel_rad": cumulative_travel,
                "trajectory_duration_sec": duration,
            }
        )
        failures: list[str] = []
        if maximum_excursion > SEARCH_MAX_JOINT_EXCURSION_RAD:
            failures.append(
                "joint excursion exceeds the bounded active-search limit"
            )
        if cumulative_travel > SEARCH_MAX_CUMULATIVE_JOINT_TRAVEL_RAD:
            failures.append(
                "cumulative joint travel exceeds the bounded active-search limit"
            )
        if duration > SEARCH_MAX_TRAJECTORY_DURATION_SEC:
            failures.append(
                "trajectory duration is excessive for a local active-search step"
            )
        result["success"] = not failures
        result["reason"] = "; ".join(failures) if failures else None
        return result

    def _search_plan(
        self,
        target: dict[str, Any],
        *,
        deadline: float,
        stationary_verified: bool = False,
    ) -> tuple[dict[str, Any], Any | None]:
        """Plan one bounded Pilz translation or tool-local-Z rotation."""
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return {
                "success": False,
                "stage": "search_plan_deadline",
                "reason": "search deadline expired before planning the next segment",
                "execution_sent": False,
            }, None
        goal = self._search_goal_for_target(target)
        if stationary_verified:
            start_joint_state = goal.request.start_state.joint_state
            start_joint_state.velocity = [0.0] * len(start_joint_state.name)
        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=min(2.0, remaining),
        )
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "search_move_group_goal",
                "reason": "MoveGroup rejected the plan-only search goal",
                "execution_sent": False,
            }, None
        result_future = handle.get_result_async()
        remaining = deadline - time.monotonic()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=max(
                0.0,
                min(SEARCH_PLAN_RESULT_TIMEOUT_SEC, remaining),
            ),
        )
        wrapped = result_future.result()
        if wrapped is None:
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "search_plan_timeout",
                "reason": "MoveGroup did not return the search plan before timeout",
                "execution_sent": False,
            }, None
        result = wrapped.result
        success = int(result.error_code.val) == 1
        vertical_axis_validation = None
        joint_motion_validation = None
        trajectory = None
        if success:
            vertical_axis_validation = self._validate_trajectory_vertical_axis(
                result.planned_trajectory,
                deadline=deadline,
            )
            success = bool(vertical_axis_validation.get("success"))
        if success:
            joint_motion_validation = self._validate_search_joint_motion(
                result.planned_trajectory
            )
            success = bool(joint_motion_validation.get("success"))
        if success:
            trajectory = result.planned_trajectory
        rotation_goal = target.get("search_motion_type") == "tool_local_z_rotation"
        validation_reason = None
        if vertical_axis_validation is not None and not vertical_axis_validation.get(
            "success"
        ):
            validation_reason = vertical_axis_validation.get("reason")
        elif joint_motion_validation is not None and not joint_motion_validation.get(
            "success"
        ):
            validation_reason = joint_motion_validation.get("reason")
        return {
            "success": success,
            "stage": "search_plan_only",
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "planner": f"{SEARCH_PLANNING_PIPELINE}/{SEARCH_PLANNER}",
            "position_goal_constraint": True,
            "goal_orientation_constraint": rotation_goal,
            "goal_vertical_axis_constraint": not rotation_goal,
            "orientation_path_constraint": True,
            "maximum_tool_vertical_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
            "tool_axis_spin_free": True,
            "intermediate_flange_orientation_unconstrained": False,
            "vertical_axis_validation": vertical_axis_validation,
            "joint_motion_validation": joint_motion_validation,
            "reason": validation_reason,
            "search_motion_type": target.get(
                "search_motion_type", "cartesian_translation"
            ),
            "tool_local_z_rotation_rad": target.get(
                "tool_local_z_rotation_rad", 0.0
            ),
            "wrist_3_direct_command": False,
            "execution_sent": False,
        }, trajectory

    def _target_ik_diagnostic(
        self,
        target: dict[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        """Describe a failed collision-checked active-search pose plan."""
        rotation_goal = target.get("search_motion_type") == "tool_local_z_rotation"
        return {
            "classification": "SEARCH_PLANNING_FAILED",
            "reason": (
                "Pilz LIN could not generate a collision-free bounded route to the "
                "bounded search goal before the deadline; this may be a "
                "start-state collision, IK failure, or unavailable route"
            ),
            "target": target,
            "planner": f"{SEARCH_PLANNING_PIPELINE}/{SEARCH_PLANNER}",
            "position_goal_constraint": True,
            "goal_orientation_constraint": rotation_goal,
            "goal_vertical_axis_constraint": not rotation_goal,
            "orientation_path_constraint": True,
            "maximum_tool_vertical_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
            "tool_axis_spin_free": True,
            "intermediate_flange_orientation_unconstrained": False,
            "search_motion_type": target.get(
                "search_motion_type", "cartesian_translation"
            ),
            "wrist_3_direct_command": False,
            "deadline_remaining_sec": max(0.0, deadline - time.monotonic()),
        }

    def _search_start_state_validity(self, deadline: float) -> dict[str, Any]:
        """Ask MoveIt whether the live robot state collides with its scene."""
        client = getattr(self, "_state_validity_client", None)
        latest_joint_state = getattr(self, "latest_joint_state", None)
        if client is None or latest_joint_state is None:
            return {
                "available": False,
                "valid": None,
                "reason": "state-validity diagnostic is unavailable",
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not client.wait_for_service(
            timeout_sec=min(0.25, max(0.0, remaining))
        ):
            return {
                "available": False,
                "valid": None,
                "reason": "/check_state_validity is unavailable",
            }
        request = GetStateValidity.Request()
        request.robot_state.joint_state = latest_joint_state
        # Preserve the verified camera/cup-holder/straw attached body while
        # checking the fresh live joints at every search step.
        request.robot_state.is_diff = True
        request.group_name = GROUP_NAME
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=min(0.75, max(0.0, deadline - time.monotonic())),
        )
        result = future.result()
        if result is None:
            return {
                "available": False,
                "valid": None,
                "reason": "/check_state_validity timed out",
            }
        contacts = [
            {
                "body_1": str(contact.contact_body_1),
                "body_2": str(contact.contact_body_2),
                "depth_m": float(contact.depth),
                "position_m": [
                    float(contact.position.x),
                    float(contact.position.y),
                    float(contact.position.z),
                ],
            }
            for contact in result.contacts
        ]
        octomap_only = bool(contacts) and all(
            "<octomap>" in (contact["body_1"], contact["body_2"])
            for contact in contacts
        )
        return {
            "available": True,
            "valid": bool(result.valid),
            "contacts": contacts,
            "octomap_only_collision": octomap_only,
            "classification": (
                "START_STATE_VALID"
                if result.valid
                else (
                    "START_STATE_IN_OCTOMAP_COLLISION"
                    if octomap_only
                    else "START_STATE_COLLISION"
                )
            ),
        }

    def _ensure_valid_search_start_state(self, deadline: float) -> dict[str, Any]:
        """Clear a stale robot-overlap map once, rebuild it, and revalidate.

        Collision checking is never bypassed: recovery is permitted only when
        every reported contact is with the OctoMap, and planning resumes only
        after newer raw and MoveIt-filtered clouds arrive and the rebuilt scene
        reports the live robot state valid.
        """
        before = self._search_start_state_validity(deadline)
        result: dict[str, Any] = {
            "before": before,
            "octomap_clear_attempted": False,
            "octomap_rebuilt_from_fresh_clouds": False,
            "recovered": False,
        }
        if not before.get("available") or before.get("valid"):
            return result
        if not before.get("octomap_only_collision"):
            return result

        clear_client = getattr(self, "_clear_octomap_client", None)
        if clear_client is None or not clear_client.wait_for_service(
            timeout_sec=min(0.25, max(0.0, deadline - time.monotonic()))
        ):
            result["recovery_reason"] = "/clear_octomap is unavailable"
            return result
        clear_future = clear_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(
            self,
            clear_future,
            timeout_sec=min(0.75, max(0.0, deadline - time.monotonic())),
        )
        result["octomap_clear_attempted"] = True
        if clear_future.result() is None:
            result["recovery_reason"] = "/clear_octomap timed out"
            return result

        cleared_at = time.monotonic()
        rebuild_deadline = min(
            deadline,
            cleared_at + SEARCH_OCTOMAP_REBUILD_TIMEOUT_SEC,
        )
        fresh_topics: set[str] = set()
        while rclpy.ok() and time.monotonic() < rebuild_deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(
                    0.05,
                    max(0.0, rebuild_deadline - time.monotonic()),
                ),
            )
            for topic in (RAW_CLOUD_TOPIC, FILTERED_CLOUD_TOPIC):
                record = self._clouds.get(topic)
                if record is not None and float(
                    record["received_monotonic"]
                ) > cleared_at:
                    fresh_topics.add(topic)
            if len(fresh_topics) == 2:
                break
        result["fresh_cloud_topics"] = sorted(fresh_topics)
        result["octomap_rebuilt_from_fresh_clouds"] = len(fresh_topics) == 2
        if len(fresh_topics) != 2:
            result["recovery_reason"] = (
                "fresh raw and filtered clouds did not arrive after OctoMap clear"
            )
            return result

        after = self._search_start_state_validity(deadline)
        result["after"] = after
        result["recovered"] = bool(after.get("available") and after.get("valid"))
        if not result["recovered"]:
            result["recovery_reason"] = (
                "rebuilt OctoMap still collides with the current robot state"
            )
        return result

    def _prepare_dynamic_scene_for_goal_selection(
        self,
        *,
        mouth: list[float],
        original_pre_mouth: list[float],
        snapshot: dict[str, Any],
        planning_scene_application: dict[str, Any],
    ) -> dict[str, Any]:
        """Rebuild only the dynamic OctoMap from stationary camera frames."""
        deadline = time.monotonic() + 4.0
        report: dict[str, Any] = {
            "success": False,
            "stage": "stationary_octomap_rebuild",
            "dynamic_octomap_enabled": True,
            "octomap_clear_attempted": False,
            "octomap_clear_scope": "dynamic_octomap_only",
            "fixed_human_collision_objects_removed": False,
            "allowed_collision_exceptions_added": False,
            "required_consistent_frames_per_topic": OCTOMAP_REBUILD_REQUIRED_FRAMES,
            "post_clear_settling_timeout_sec": GOAL_OCTOMAP_REBUILD_TIMEOUT_SEC,
            "dynamic_point_cloud_filtering": {
                "moveit_filtered_cloud_topic": FILTERED_CLOUD_TOPIC,
                "ur_chain_self_filtering_available": True,
                "camera_filtered_as_robot_geometry": False,
                "cup_holder_filtered_as_robot_geometry": False,
                "straw_filtered_as_robot_geometry": False,
                "reason": (
                    "the MoveIt self-filter can mask only collision geometry in the "
                    "current robot model; camera, cup holder, and straw are not modeled"
                ),
            },
        }
        stationary = self._wait_for_search_stationary(deadline)
        report["camera_stationary_guard"] = stationary
        if not stationary.get("success"):
            report["reason"] = "camera/robot was not stationary before OctoMap rebuild"
            return report

        before_scene, _ = self._planning_scene_geometry()
        report["planning_scene_before_clear"] = before_scene
        expected_human_ids = sorted(
            planning_scene_application.get("object_ids", [])
        )
        before_human_ids = sorted(before_scene.get("human_object_ids", []))
        if not expected_human_ids or not set(expected_human_ids).issubset(
            before_human_ids
        ):
            report["reason"] = "fixed human collision objects are missing before OctoMap clear"
            return report

        approach = _subtract(original_pre_mouth, mouth)
        approach_norm = _norm(approach)
        if approach_norm < 1e-9:
            report["reason"] = "validated approach line is degenerate"
            return report
        original_candidate = _adaptive_premouth_pose_candidates(
            mouth_position_m=mouth,
            approach_offset_unit=[value / approach_norm for value in approach],
            verified_flange_down_orientation_xyzw=list(
                snapshot["tool0_pose"]["orientation_quat_xyzw"]
            ),
            standoffs_m=(0.050,),
            yaws_deg=(0.0,),
        )[0]

        before_ik = self._solve_candidate_ik(original_candidate["tool0_pose"])
        before_robot_state = before_ik.get("robot_state")
        before_validity = self._state_validity(
            before_robot_state if isinstance(before_robot_state, RobotState) else None,
            label="original_50mm_yaw_0_before_octomap_rebuild",
        )
        report["original_goal_before_rebuild"] = {
            "candidate": original_candidate,
            "ik": {
                key: value for key, value in before_ik.items() if key != "robot_state"
            },
            "state_validity": before_validity,
        }

        clear_client = getattr(self, "_clear_octomap_client", None)
        if clear_client is None or not clear_client.wait_for_service(
            timeout_sec=min(0.5, max(0.0, deadline - time.monotonic()))
        ):
            report["reason"] = "/clear_octomap is unavailable"
            return report
        clear_future = clear_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(
            self,
            clear_future,
            timeout_sec=min(1.0, max(0.0, deadline - time.monotonic())),
        )
        report["octomap_clear_attempted"] = True
        if clear_future.result() is None:
            report["reason"] = "/clear_octomap timed out"
            return report

        cleared_at = time.monotonic()
        rebuild_deadline = cleared_at + GOAL_OCTOMAP_REBUILD_TIMEOUT_SEC
        frames: dict[str, list[dict[str, Any]]] = {
            RAW_CLOUD_TOPIC: [],
            FILTERED_CLOUD_TOPIC: [],
        }
        selected_windows: dict[str, dict[str, Any]] = {}
        while rclpy.ok() and time.monotonic() < rebuild_deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(
                    0.05,
                    max(0.0, rebuild_deadline - time.monotonic()),
                ),
            )
            for topic in frames:
                history = getattr(self, "_cloud_history", {}).get(topic, [])
                frames[topic] = [
                    dict(item)
                    for item in history
                    if float(item["received_monotonic"]) > cleared_at
                ][-OCTOMAP_REBUILD_HISTORY_LIMIT:]
                selected_windows[topic] = _select_consistent_cloud_frame_window(
                    frames[topic]
                )
            if all(
                selected_windows.get(topic, {}).get("consistent", False)
                for topic in frames
            ):
                break
        consistency: dict[str, Any] = {}
        consistent = True
        for topic, items in frames.items():
            window = selected_windows.get(topic) or (
                _select_consistent_cloud_frame_window(items)
            )
            frames_consistent = bool(window["consistent"])
            consistency[topic] = {
                **window,
                "sampled_frame_count": len(items),
                "sampled_point_counts": [
                    int(item.get("point_count", 0)) for item in items
                ],
                "maximum_relative_point_count_spread": (
                    OCTOMAP_REBUILD_MAX_POINT_COUNT_SPREAD
                ),
            }
            consistent = consistent and frames_consistent
        report["stationary_rebuild_frames"] = consistency
        report["octomap_rebuilt_from_consistent_stationary_frames"] = consistent
        if not consistent:
            failed_topics = [
                topic
                for topic, details in consistency.items()
                if not details["consistent"]
            ]
            report["reason"] = (
                "three consecutive consistent RGB-D frames were not received for: "
                + ", ".join(failed_topics)
            )
            return report

        after_scene, _ = self._planning_scene_geometry()
        report["planning_scene_after_rebuild"] = after_scene
        after_human_ids = sorted(after_scene.get("human_object_ids", []))
        human_preserved = set(expected_human_ids).issubset(after_human_ids)
        report["human_collision_objects_preserved"] = human_preserved
        if not human_preserved:
            report["reason"] = "fixed human collision objects changed during OctoMap rebuild"
            return report

        after_ik = self._solve_candidate_ik(original_candidate["tool0_pose"])
        after_robot_state = after_ik.get("robot_state")
        after_validity = self._state_validity(
            after_robot_state if isinstance(after_robot_state, RobotState) else None,
            label="original_50mm_yaw_0_after_octomap_rebuild",
        )
        report["original_goal_after_rebuild"] = {
            "candidate": original_candidate,
            "ik": {
                key: value for key, value in after_ik.items() if key != "robot_state"
            },
            "state_validity": after_validity,
        }
        comparison = _compare_final_state_validity(
            before_validity,
            after_validity,
        )
        report["original_goal_validity_comparison"] = comparison
        report["final_state_validity_changed_after_rebuild"] = comparison[
            "changed"
        ]
        rebuilt_octomap_available = bool(
            after_scene.get("available")
            and after_scene.get("octomap", {}).get("present")
        )
        report["rebuilt_planning_scene_available"] = bool(
            after_scene.get("available")
        )
        report["rebuilt_octomap_available"] = rebuilt_octomap_available
        report["success"] = bool(
            after_scene.get("available")
            and human_preserved
            and rebuilt_octomap_available
        )
        report["reason"] = None if report["success"] else (
            "rebuilt planning scene or dynamic OctoMap is unavailable"
        )
        return report

    def _wait_for_search_stationary(
        self,
        deadline: float,
        *,
        timeout_sec: float = SEARCH_STATIONARY_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Require fresh stationary joint samples before each OMPL search plan.

        The UR controller can report one final nonzero velocity sample after a
        trajectory result succeeds.  Wait for two new stationary samples
        before representing the verified start state with exact zero
        velocities in the planning goal.
        """
        if not math.isfinite(float(timeout_sec)) or float(timeout_sec) <= 0.0:
            raise ValueError("stationary wait timeout must be finite and positive")
        settle_deadline = min(
            deadline,
            time.monotonic() + float(timeout_sec),
        )
        consecutive = 0
        last_state: Any | None = None
        latest_max_speed: float | None = None
        latest_reason = "no fresh joint-state sample received"
        while rclpy.ok() and time.monotonic() < settle_deadline:
            state = self.latest_joint_state
            if state is not None and state is not last_state:
                last_state = state
                names = list(state.name)
                velocities = list(state.velocity)
                if len(names) != len(velocities):
                    consecutive = 0
                    latest_reason = "joint-state velocity vector is incomplete"
                else:
                    by_name = dict(zip(names, velocities))
                    if not all(name in by_name for name in EXPECTED_JOINTS):
                        consecutive = 0
                        latest_reason = "joint-state velocity vector is missing a UR10e joint"
                    else:
                        expected_velocities = [
                            float(by_name[name]) for name in EXPECTED_JOINTS
                        ]
                        if not all(math.isfinite(value) for value in expected_velocities):
                            consecutive = 0
                            latest_reason = "joint-state velocity vector contains a non-finite value"
                        else:
                            latest_max_speed = max(
                                (abs(value) for value in expected_velocities),
                                default=0.0,
                            )
                            if (
                                latest_max_speed
                                <= SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC
                            ):
                                consecutive += 1
                                latest_reason = "stationary"
                            else:
                                consecutive = 0
                                latest_reason = "UR10e joints are still settling"
                            if consecutive >= SEARCH_STATIONARY_SAMPLE_COUNT:
                                return {
                                    "success": True,
                                    "timeout_sec": float(timeout_sec),
                                    "maximum_joint_speed_rad_sec": latest_max_speed,
                                    "maximum_allowed_joint_speed_rad_sec": (
                                        SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC
                                    ),
                                    "stationary_sample_count": consecutive,
                                }
            rclpy.spin_once(
                self,
                timeout_sec=min(
                    0.02,
                    max(0.0, settle_deadline - time.monotonic()),
                ),
            )
        return {
            "success": False,
            "reason": latest_reason,
            "timeout_sec": float(timeout_sec),
            "maximum_joint_speed_rad_sec": latest_max_speed,
            "maximum_allowed_joint_speed_rad_sec": (
                SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC
            ),
            "stationary_sample_count": consecutive,
        }

    def _wait_for_tracking_replan_stationary(self) -> dict[str, Any]:
        """Wait for controlled post-cancel deceleration before scene rebuild."""
        return self._wait_for_search_stationary(
            time.monotonic() + TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC,
            timeout_sec=TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC,
        )

    def _cancel_goal(self, handle: Any) -> None:
        cancel = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel, timeout_sec=5.0)

    def _execute_search_trajectory(
        self,
        trajectory: Any,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        """Execute one validated segment and cancel when any face appears."""
        vertical_axis_validation = self._validate_trajectory_vertical_axis(
            trajectory,
            deadline=deadline,
        )
        if not vertical_axis_validation.get("success"):
            return {
                "success": False,
                "stage": "search_pre_execution_vertical_axis_validation",
                "reason": vertical_axis_validation.get("reason"),
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        joint_motion_validation = self._validate_search_joint_motion(trajectory)
        if not joint_motion_validation.get("success"):
            return {
                "success": False,
                "stage": "search_pre_execution_joint_motion_validation",
                "reason": joint_motion_validation.get("reason"),
                "vertical_axis_validation": vertical_axis_validation,
                "joint_motion_validation": joint_motion_validation,
                "execution_attempted": False,
            }
        current_pose = self._tool0_pose()
        if not current_pose.get("available"):
            return {
                "success": False,
                "stage": "search_pre_execution_vertical_axis_guard",
                "reason": "live tool0 pose is unavailable immediately before execution",
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        try:
            current_tilt = _tool_vertical_tilt_rad(
                current_pose["orientation_quat_xyzw"]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "stage": "search_pre_execution_vertical_axis_guard",
                "reason": f"live tool vertical-axis check failed: {exc}",
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        if current_tilt > MAX_TOOL_VERTICAL_TILT_RAD:
            return {
                "success": False,
                "stage": "search_pre_execution_vertical_axis_guard",
                "reason": (
                    f"live tool tilt {math.degrees(current_tilt):.2f} deg exceeds "
                    f"the {math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
                ),
                "live_tool_tilt_rad": current_tilt,
                "vertical_axis_validation": vertical_axis_validation,
                "joint_motion_validation": joint_motion_validation,
                "execution_attempted": False,
            }
        client = self._execution_action_client()
        if not client.wait_for_server(timeout_sec=2.0):
            return {
                "success": False,
                "stage": "search_execute_server",
                "reason": "/execute_trajectory is unavailable",
                "execution_attempted": False,
            }
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return {
                "success": False,
                "stage": "search_execute_deadline",
                "reason": "search deadline expired before trajectory submission",
                "execution_attempted": False,
            }
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=min(2.0, remaining),
        )
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "search_execute_goal",
                "reason": "MoveIt rejected the validated search trajectory",
                "execution_attempted": False,
            }
        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            live_ur_failure = self._live_ur_execution_state_failure()
            if live_ur_failure is not None:
                self._cancel_goal(handle)
                return {
                    "success": False,
                    "stage": "search_cancelled_for_ur_execution_state",
                    "reason": live_ur_failure,
                    "vertical_axis_validation": vertical_axis_validation,
                    "joint_motion_validation": joint_motion_validation,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
                }
            live_pose = self._tool0_pose()
            if not live_pose.get("available"):
                self._cancel_goal(handle)
                return {
                    "success": False,
                    "stage": "search_cancelled_for_vertical_axis_guard",
                    "reason": "live tool0 pose became unavailable during search",
                    "vertical_axis_validation": vertical_axis_validation,
                    "joint_motion_validation": joint_motion_validation,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
                }
            try:
                live_tilt = _tool_vertical_tilt_rad(
                    live_pose["orientation_quat_xyzw"]
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                self._cancel_goal(handle)
                return {
                    "success": False,
                    "stage": "search_cancelled_for_vertical_axis_guard",
                    "reason": f"live tool vertical-axis check failed: {exc}",
                    "vertical_axis_validation": vertical_axis_validation,
                    "joint_motion_validation": joint_motion_validation,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
                }
            if live_tilt > MAX_TOOL_VERTICAL_TILT_RAD:
                self._cancel_goal(handle)
                return {
                    "success": False,
                    "stage": "search_cancelled_for_vertical_axis_guard",
                    "reason": (
                        f"live tool tilt {math.degrees(live_tilt):.2f} deg exceeded "
                        f"the {math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
                    ),
                    "live_tool_tilt_rad": live_tilt,
                    "vertical_axis_validation": vertical_axis_validation,
                    "joint_motion_validation": joint_motion_validation,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
                }
            if self._candidate_visible():
                self._cancel_goal(handle)
                return {
                    "success": True,
                    "stage": "search_cancelled_for_candidate",
                    "candidate_detected": True,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
                    "vertical_axis_validation": vertical_axis_validation,
                    "joint_motion_validation": joint_motion_validation,
                }
        wrapped = result_future.result() if result_future.done() else None
        if wrapped is None:
            self._cancel_goal(handle)
            return {
                "success": False,
                "stage": "search_execute_timeout",
                "reason": "search trajectory exceeded the bounded search deadline; cancel requested",
                "execution_attempted": True,
            }
        result = wrapped.result
        error_code = int(result.error_code.val)
        success = error_code == 1
        controller_failure = None
        if error_code == -4:
            controller_state = self._controller_status()
            controller_failure = {
                "classification": "CONTROL_FAILED",
                "moveit_error_code": error_code,
                "reason": (
                    "MoveIt lost or aborted trajectory control after goal "
                    "acceptance; check the UR reverse interface and External "
                    "Control program"
                ),
                "scaled_joint_trajectory_controller_active": (
                    controller_state.get(
                        "scaled_joint_trajectory_controller_active"
                    )
                ),
                "external_control_program_running": bool(
                    self.latest_robot_program_running is not None
                    and self.latest_robot_program_running.data
                ),
                "safety_mode": (
                    None
                    if self.latest_safety_mode is None
                    else int(self.latest_safety_mode.mode)
                ),
                "robot_mode": (
                    None
                    if self.latest_robot_mode is None
                    else int(self.latest_robot_mode.mode)
                ),
            }
        return {
            "success": success,
            "stage": "search_execute",
            "error_code": error_code,
            "error_message": result.error_code.message,
            "reason": (
                None
                if success
                else (
                    controller_failure["reason"]
                    if controller_failure is not None
                    else result.error_code.message
                    or f"MoveIt execution error code {error_code}"
                )
            ),
            "failure_diagnostic": controller_failure,
            "result_status": int(wrapped.status),
            "candidate_detected": self._candidate_visible(),
            "execution_attempted": True,
            "vertical_axis_validation": vertical_axis_validation,
            "joint_motion_validation": joint_motion_validation,
        }

    def active_search(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        use_octomap: bool = True,
    ) -> dict[str, Any]:
        """Acquire the selected mouth with at most 15 seconds of fixed search."""
        started = time.monotonic()
        deadline = started + SEARCH_MAX_TIME_SEC
        snapshot = self.snapshot(
            mouth_sample_sec=min(self.mouth_sample_seconds, SEARCH_MAX_TIME_SEC),
            inspect_controllers=execute,
        )
        response: dict[str, Any] = {
            "success": False,
            "stage": "active_search",
            "execute": execute,
            "target_selection": self.target_selection,
            "maximum_time_sec": SEARCH_MAX_TIME_SEC,
            "planner": f"{SEARCH_PLANNING_PIPELINE}/{SEARCH_PLANNER}",
            "ompl_active_search_enabled": False,
            "dynamic_octomap_required": bool(use_octomap),
            "translation_only": False,
            "position_only_search_goals": False,
            "vertical_axis_constraint_active": True,
            "maximum_tool_vertical_tilt_deg": math.degrees(
                MAX_TOOL_VERTICAL_TILT_RAD
            ),
            "tool_axis_spin_free": True,
            "intermediate_flange_orientation_unconstrained": False,
            "fk_waypoint_vertical_axis_validation": True,
            "maximum_joint_excursion_rad": SEARCH_MAX_JOINT_EXCURSION_RAD,
            "maximum_cumulative_joint_travel_rad": (
                SEARCH_MAX_CUMULATIVE_JOINT_TRAVEL_RAD
            ),
            "maximum_search_trajectory_duration_sec": (
                SEARCH_MAX_TRAJECTORY_DURATION_SEC
            ),
            "rotation_search_enabled": True,
            "left_right_search_strategy": "MoveIt tool0-local-Z pose rotations",
            "left_right_rotation_each_side_deg": SEARCH_WRIST_Z_ANGLE_DEG,
            "left_right_total_sweep_deg": SEARCH_WRIST_Z_TOTAL_SWEEP_DEG,
            "wrist_3_direct_command": False,
            "search_direction_reference": {
                "backward_up_down": (
                    f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
                ),
                "left_right": f"{TOOL_FRAME} local Z rotation",
            },
            "camera_extrinsic_applied": True,
            "search_order": [
                "backward_wide",
                "scan_left",
                "scan_right",
                "scan_up",
                "scan_down",
            ],
            "adaptive_distance_policy_m": {
                "backward": list(SEARCH_BACK_CANDIDATE_DISTANCES_M),
                "up_down": list(SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M),
                "skip_if_all_bounded_candidates_fail": True,
            },
            "adaptive_left_right_angles_deg": [
                math.degrees(value)
                for value in SEARCH_WRIST_Z_CANDIDATE_ANGLES_RAD
            ],
            "trajectory_sent": False,
            "checks": snapshot,
            "search_steps": [],
            "skipped_search_waypoints": [],
            "start_state_checks": [],
        }
        failures = self._base_readiness_failures(snapshot)
        if execute:
            failures.extend(
                self._execution_state_failures(confirm_real_motion=confirm_real_motion)
            )
        if failures:
            response.update({"stage": "active_search_readiness", "failures": failures})
            return response

        initial = snapshot["mouth_pose"]
        if initial.get("available") and initial.get("stable"):
            response.update(
                {
                    "success": True,
                    "stage": "mouth_found_without_search_motion",
                    "found_without_motion": True,
                    "selected_mouth": initial,
                    "elapsed_sec": time.monotonic() - started,
                }
            )
            return response
        if self._candidate_visible():
            stable = self._wait_for_selected_stability(started, deadline)
            response.update(
                {
                    "success": bool(stable.get("available") and stable.get("stable")),
                    "stage": "mouth_stability_wait",
                    "candidate_detected": True,
                    "stopped_for_stability": True,
                    "selected_mouth": stable,
                    "elapsed_sec": time.monotonic() - started,
                }
            )
            if not response["success"]:
                response["reason"] = str(
                    stable.get("reason")
                    or "visible mouth did not become a stable selected identity; search motion was withheld"
                )
            return response
        if not self._explicit_no_face():
            response.update(
                {
                    "stage": "active_search_perception_gate",
                    "reason": (
                        "mouth perception did not explicitly report no_face; search motion was withheld"
                    ),
                    "elapsed_sec": time.monotonic() - started,
                }
            )
            return response

        origin_tool0 = [float(value) for value in snapshot["tool0_pose"]["position_m"]]
        orientation = [
            float(value) for value in snapshot["tool0_pose"]["orientation_quat_xyzw"]
        ]
        camera_orientation = [
            float(value)
            for value in snapshot["camera_tf"]["orientation_quat_xyzw"]
        ]
        origin_straw = _add(
            origin_tool0,
            _rotate_tool_vector(orientation, STRAW_TIP_OFFSET_TOOL0_M),
        )
        waypoints = self.search_waypoints(
            origin_tool0,
            orientation,
            camera_orientation,
        )
        active_back_distance = SEARCH_BACK_DISTANCE_M
        motion_deadline = deadline - SEARCH_STABILITY_RESERVE_SEC
        for requested_waypoint in waypoints:
            if time.monotonic() >= motion_deadline:
                break
            if self._candidate_visible():
                stable = self._wait_for_selected_stability(started, deadline)
                response.update(
                    {
                        "success": bool(stable.get("available") and stable.get("stable")),
                        "stage": "mouth_found_during_search",
                        "candidate_detected": True,
                        "stopped_for_stability": True,
                        "selected_mouth": stable,
                        "trajectory_sent": bool(response["trajectory_sent"]),
                        "elapsed_sec": time.monotonic() - started,
                    }
                )
                if not response["success"]:
                    response["reason"] = str(
                        stable.get("reason")
                        or "mouth candidate appeared but did not become a stable selected identity"
                    )
                return response

            stationary: dict[str, Any] | None = None
            if execute:
                stationary = self._wait_for_search_stationary(motion_deadline)
                if not stationary.get("success"):
                    response.update(
                        {
                            "stage": "active_search_stationary_guard",
                            "reason": (
                                "UR10e did not report a stationary joint state before "
                                "the next bounded search plan"
                            ),
                            "stationary_joint_state": stationary,
                        }
                    )
                    return response

            current = self._tool0_pose()
            if not current.get("available"):
                response.update(
                    {
                        "stage": "active_search_tf",
                        "reason": "base_link -> tool0 became unavailable during search",
                    }
                )
                return response
            current_position = [float(value) for value in current["position_m"]]
            current_orientation = [
                float(value) for value in current["orientation_quat_xyzw"]
            ]

            state_check = self._ensure_valid_search_start_state(motion_deadline)
            state_check["before_search_waypoint"] = requested_waypoint["name"]
            response["start_state_checks"].append(state_check)
            before_state = state_check.get("before", {})
            start_state_valid = bool(
                before_state.get("valid") or state_check.get("recovered")
            )
            if before_state.get("available") and not start_state_valid:
                contacts = (
                    state_check.get("after", {}).get("contacts")
                    or before_state.get("contacts")
                    or []
                )
                contact_pairs = ", ".join(
                    f"{item['body_1']} - {item['body_2']}"
                    for item in contacts
                )
                response.update(
                    {
                        "stage": "active_search_start_state_collision",
                        "reason": (
                            "MoveIt reports the current robot state in "
                            "collision"
                            + (f": {contact_pairs}" if contact_pairs else "")
                            + "; no search trajectory was requested"
                        ),
                        "failure_diagnostic": {
                            "classification": (
                                state_check.get("after", {}).get(
                                    "classification"
                                )
                                or before_state.get("classification")
                            ),
                            "contacts": contacts,
                            "stale_octomap_recovery": state_check,
                        },
                    }
                )
                return response

            variants = self.search_waypoint_variants(
                origin_tool0,
                orientation,
                camera_orientation,
                name=str(requested_waypoint["name"]),
                back_distance_m=active_back_distance,
            )
            planning_attempts: list[dict[str, Any]] = []
            waypoint: dict[str, Any] | None = None
            step: dict[str, Any] | None = None
            trajectory: Any | None = None
            for variant in variants:
                if time.monotonic() >= motion_deadline:
                    break
                variant = dict(variant)
                rotation_goal = (
                    variant.get("search_motion_type") == "tool_local_z_rotation"
                )
                if rotation_goal:
                    # Hold the live tool position while requesting an absolute
                    # local-Z scan orientation from the frozen initial pose.
                    variant["target_tool0_position_m"] = list(current_position)
                segment = _norm(
                    _subtract(variant["target_tool0_position_m"], current_position)
                )
                angular_segment = _quaternion_distance_rad(
                    current_orientation,
                    variant["target_tool0_orientation_quat_xyzw"],
                )
                target_in_ur_base = self._point_in_ur_base(
                    variant["target_tool0_position_m"],
                    snapshot["ur_base_tf"],
                )
                radius = _norm(target_in_ur_base)
                attempt: dict[str, Any] = {
                    **variant,
                    "segment_distance_m": segment,
                    "segment_orientation_distance_rad": angular_segment,
                    "target_tool0_radius_from_ur_base_m": radius,
                    "position_goal_constraint": True,
                    "goal_orientation_constraint": rotation_goal,
                    "goal_vertical_axis_constraint": not rotation_goal,
                    "orientation_path_constraint": True,
                    "maximum_tool_vertical_tilt_rad": (
                        MAX_TOOL_VERTICAL_TILT_RAD
                    ),
                    "tool_axis_spin_free": True,
                    "intermediate_flange_orientation_unconstrained": False,
                    "wrist_3_direct_command": False,
                    "plan_result": None,
                    "failure_diagnostic": None,
                }
                if segment > SEARCH_MAX_ACTUAL_SEGMENT_M:
                    attempt["failure_diagnostic"] = {
                        "classification": "SEGMENT_LIMIT_EXCEEDED",
                        "reason": (
                            "actual robot pose is too far from this bounded "
                            "search candidate"
                        ),
                    }
                    planning_attempts.append(attempt)
                    continue
                if radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
                    attempt["failure_diagnostic"] = {
                        "classification": "NOMINAL_REACH_EXCEEDED",
                        "reason": "search candidate exceeds the UR10e nominal reach envelope",
                    }
                    planning_attempts.append(attempt)
                    continue
                target = {
                    "frame_id": BASE_FRAME,
                    "link_name": TOOL_FRAME,
                    "position_m": variant["target_tool0_position_m"],
                    "orientation_quat_xyzw": (
                        variant["target_tool0_orientation_quat_xyzw"]
                        if rotation_goal
                        else current_orientation
                    ),
                    "search_motion_type": variant["search_motion_type"],
                    "tool_local_z_rotation_rad": variant[
                        "tool_local_z_rotation_rad"
                    ],
                }
                plan, candidate_trajectory = self._search_plan(
                    target,
                    deadline=motion_deadline,
                    stationary_verified=bool(
                        stationary is not None and stationary.get("success")
                    ),
                )
                attempt["plan_result"] = plan
                if not plan.get("success") or candidate_trajectory is None:
                    attempt["failure_diagnostic"] = self._target_ik_diagnostic(
                        target,
                        deadline=motion_deadline,
                    )
                    planning_attempts.append(attempt)
                    continue
                planning_attempts.append(dict(attempt))
                waypoint = variant
                trajectory = candidate_trajectory
                step = {
                    **attempt,
                    "execution_result": None,
                    "stationary_joint_state": stationary,
                    "planning_attempts": planning_attempts,
                }
                break

            if step is None or waypoint is None or trajectory is None:
                last_diagnostic = (
                    planning_attempts[-1].get("failure_diagnostic")
                    if planning_attempts
                    else {
                        "classification": "SEARCH_DEADLINE_EXPIRED",
                        "reason": "search deadline expired before a candidate could be planned",
                    }
                )
                skipped = {
                    **requested_waypoint,
                    "skipped": True,
                    "planning_attempts": planning_attempts,
                    "failure_diagnostic": last_diagnostic,
                }
                response["search_steps"].append(skipped)
                response["skipped_search_waypoints"].append(
                    {
                        "name": requested_waypoint["name"],
                        "classification": last_diagnostic.get("classification"),
                        "reason": last_diagnostic.get("reason"),
                    }
                )
                if requested_waypoint["name"] == "backward_wide":
                    active_back_distance = 0.0
                continue
            if not execute:
                response["search_steps"].append(step)
                response.update(
                    {
                        "stage": "active_search_plan_only",
                        "reason": "selected mouth is absent; first real search waypoint planned without motion",
                        "planning_success": True,
                        "requires_search_execution": True,
                        "search_origin_straw_tip": origin_straw,
                        "next_search_waypoint": step,
                        "elapsed_sec": time.monotonic() - started,
                    }
                )
                return response

            late_failures = self._execution_state_failures(
                confirm_real_motion=confirm_real_motion
            )
            clouds = (
                {
                    RAW_CLOUD_TOPIC: self._cloud_status(RAW_CLOUD_TOPIC),
                    FILTERED_CLOUD_TOPIC: self._cloud_status(FILTERED_CLOUD_TOPIC),
                }
                if use_octomap
                else {}
            )
            late_failures.extend(
                _active_search_cloud_gate_failures(
                    use_octomap=use_octomap,
                    cloud_statuses=clouds,
                )
            )
            step["dynamic_octomap_required"] = bool(use_octomap)
            step["point_cloud_status"] = clouds
            if late_failures:
                step["pre_execution_failures"] = late_failures
                response["search_steps"].append(step)
                response.update(
                    {
                        "stage": "active_search_pre_execution_guard",
                        "failures": late_failures,
                    }
                )
                return response
            execution = self._execute_search_trajectory(
                trajectory,
                deadline=motion_deadline,
            )
            step["execution_result"] = execution
            response["search_steps"].append(step)
            response["trajectory_sent"] = bool(
                response["trajectory_sent"] or execution.get("execution_attempted")
            )
            if not execution.get("success"):
                detail = (
                    execution.get("reason")
                    or execution.get("error_message")
                    or f"MoveIt execution error code {execution.get('error_code')}"
                )
                response.update(
                    {
                        "stage": "active_search_execution",
                        "reason": f"bounded search segment failed: {detail}",
                        "failure_diagnostic": {
                            "classification": (
                                execution.get("failure_diagnostic", {}).get(
                                    "classification"
                                )
                                if isinstance(
                                    execution.get("failure_diagnostic"), dict
                                )
                                else None
                            )
                            or "SEARCH_EXECUTION_FAILED",
                            "reason": detail,
                            "execution_stage": execution.get("stage"),
                            "error_code": execution.get("error_code"),
                            "error_message": execution.get("error_message"),
                            "controller_diagnostic": execution.get(
                                "failure_diagnostic"
                            ),
                        },
                    }
                )
                return response
            if execution.get("candidate_detected") or self._candidate_visible():
                stable = self._wait_for_selected_stability(started, deadline)
                response.update(
                    {
                        "success": bool(stable.get("available") and stable.get("stable")),
                        "stage": "mouth_found_during_search",
                        "candidate_detected": True,
                        "stopped_for_stability": True,
                        "selected_mouth": stable,
                        "elapsed_sec": time.monotonic() - started,
                    }
                )
                if not response["success"]:
                    response["reason"] = str(
                        stable.get("reason")
                        or "mouth candidate appeared but did not become a stable selected identity"
                    )
                return response

            final = self._tool0_pose()
            if not final.get("available"):
                response.update(
                    {
                        "stage": "active_search_verification",
                        "reason": "final tool0 TF is unavailable after search segment",
                    }
                )
                return response
            final_error = _norm(
                _subtract(final["position_m"], waypoint["target_tool0_position_m"])
            )
            step["final_tool0_position_error_m"] = final_error
            step["final_tool0_orientation_quat_xyzw"] = [
                float(value) for value in final["orientation_quat_xyzw"]
            ]
            if waypoint.get("search_motion_type") == "tool_local_z_rotation":
                final_orientation_error = _quaternion_distance_rad(
                    step["final_tool0_orientation_quat_xyzw"],
                    waypoint["target_tool0_orientation_quat_xyzw"],
                )
                step["final_tool0_orientation_error_rad"] = (
                    final_orientation_error
                )
                if (
                    final_orientation_error
                    > SEARCH_FINAL_ORIENTATION_TOLERANCE_RAD
                ):
                    response.update(
                        {
                            "stage": "active_search_verification",
                            "reason": (
                                "tool-local-Z search rotation missed its "
                                "bounded pose target"
                            ),
                            "final_orientation_error_rad": (
                                final_orientation_error
                            ),
                        }
                    )
                    return response
            if final_error > SEARCH_FINAL_POSITION_TOLERANCE_M:
                response.update(
                    {
                        "stage": "active_search_verification",
                        "reason": "search segment missed its bounded target",
                        "final_position_error_m": final_error,
                    }
                )
                return response
            if waypoint["name"] == "backward_wide":
                active_back_distance = abs(
                    float(waypoint["offset_camera_optical_m"][2])
                )

        stable = self._wait_for_selected_stability(started, deadline)
        response.update(
            {
                "success": bool(stable.get("available") and stable.get("stable")),
                "stage": "mouth_found_after_search" if stable.get("stable") else "active_search_timeout",
                "selected_mouth": stable,
                "elapsed_sec": time.monotonic() - started,
                "trajectory_sent": bool(response["trajectory_sent"]),
            }
        )
        if not response["success"]:
            skipped_summary = "; ".join(
                f"{item['name']}: {item.get('classification')}"
                for item in response["skipped_search_waypoints"]
            )
            response["reason"] = "selected mouth was not found within the bounded 15-second search"
            if skipped_summary:
                response["reason"] += f"; skipped unreachable candidates: {skipped_summary}"
        return response

    def plan(self) -> tuple[int, dict[str, Any]]:
        code, response = super().plan()
        plan_result = response.get("plan_result", {})
        if isinstance(plan_result, dict):
            if "planner" in plan_result:
                response["planner"] = plan_result["planner"]
            if "route_strategy" in plan_result:
                response["route_strategy"] = plan_result["route_strategy"]
        if not response.get("success") and not response.get("stage"):
            response["stage"] = "dynamic_route_plan_only"
            response["reason"] = (
                "the direct path and vertical-axis OMPL detour both failed "
                "for the frozen real pre-mouth target; the tool constraint "
                "was not relaxed"
            )
            response["planning_error_code"] = plan_result.get("error_code")
            response["failure_diagnostic"] = {
                "classification": "DIRECT_AND_DETOUR_PLANNING_FAILED",
                "reason": response["reason"],
                "error_code": plan_result.get("error_code"),
                "error_message": plan_result.get("error_message"),
                "direct_path_plan_result": plan_result.get(
                    "direct_path_plan_result"
                ),
                "obstacle_layer_attribution": "combined_scene_only",
            }
        detected = response.get("detected_mouth_pose")
        if response.get("success") and isinstance(detected, dict):
            position = detected.get("position_m")
            if isinstance(position, list) and len(position) == 3:
                self._frozen_execution_mouth_position = [float(value) for value in position]
        return code, response

    def _goal_for_selected_dynamic_route(self, target: dict[str, Any]):
        """Rebuild the validated route profile for guarded plan-and-execute."""
        strategy = getattr(self, "_selected_dynamic_route_strategy", None)
        if strategy == DIRECT_ROUTE_STRATEGY:
            return self._direct_goal_for_target(target)
        if strategy == DETOUR_ROUTE_STRATEGY:
            return self._goal_for_target(target)
        raise RuntimeError("no validated dynamic route strategy is available")

    @staticmethod
    def _execution_mouth_drift_confirmation(
        observation: dict[str, Any],
        frozen_position: list[float] | None,
    ) -> dict[str, Any]:
        """Confirm genuine mouth motion from a stable multi-frame mean.

        Wrist-camera motion can produce an isolated depth or landmark shift
        even though every sample is transformed at its image timestamp.  Do
        not compare that one latest sample directly with the frozen target.
        Require a fresh stable window and compare its mean in ``base_link``.
        """
        report: dict[str, Any] = {
            "available": bool(observation.get("available")),
            "stable": bool(observation.get("stable")),
            "sample_count": int(observation.get("sample_count", 0) or 0),
            "required_samples": EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES,
            "window_sec": EXECUTION_MOUTH_DRIFT_CONFIRMATION_WINDOW_SEC,
            "maximum_spread_m": MAX_POSE_SPREAD_M,
            "threshold_m": MAX_EXECUTION_TARGET_DRIFT_M,
            "confirmed": False,
        }
        frozen = _finite_xyz(frozen_position)
        mean = _finite_xyz(observation.get("mean_position_m"))
        if frozen is None:
            report["reason"] = "frozen mouth target is unavailable"
            return report
        if not report["available"]:
            report["reason"] = str(
                observation.get("reason") or "no fresh mouth observations"
            )
            return report
        if (
            not report["stable"]
            or report["sample_count"] < EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES
        ):
            report["reason"] = "mouth drift is not confirmed by a stable sample window"
            return report
        if mean is None:
            report["reason"] = "stable mouth mean is invalid"
            return report
        drift = _norm(_subtract(mean, frozen))
        report.update(
            {
                "mean_position_m": mean,
                "frozen_position_m": frozen,
                "drift_m": drift,
                "confirmed": drift > MAX_EXECUTION_TARGET_DRIFT_M,
                "reason": None,
            }
        )
        return report

    @staticmethod
    def _pre_execution_target_drift_requires_replan(
        result: dict[str, Any],
    ) -> bool:
        """Recognize only fresh-target drift guards, never other safety failures."""
        if result.get("stage") != "pre_execution_state_guard":
            return False
        failures = result.get("failures")
        if not isinstance(failures, list) or not failures:
            return False
        allowed_prefixes = (
            "selected mouth target moved ",
            "a visible person's collision geometry moved ",
        )
        return all(
            isinstance(failure, str)
            and failure.startswith(allowed_prefixes)
            for failure in failures
        )

    def _execute_direct_with_tracking_monitor(
        self,
        *,
        vertical_axis_validation: dict[str, Any],
        readiness: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the cached Cartesian trajectory while watching mouth drift.

        MoveGroup owns the controller during this phase.  Significant stable
        target motion cancels execution and asks the caller to plan again from
        the stopped state; Servo is never published concurrently.
        """
        final_target = getattr(self, "_frozen_dynamic_target", None)
        current_pose = self._tool0_pose()
        if not isinstance(final_target, dict) or not current_pose.get("available"):
            return {
                "success": False,
                "stage": "tracking_segment_target",
                "reason": "current or final tool pose is unavailable",
                "execution_attempted": False,
            }
        try:
            tracking_segment = _bounded_tracking_segment_target(
                current_pose=current_pose,
                final_pose=final_target,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "stage": "tracking_segment_target",
                "reason": f"could not construct bounded tracking segment: {exc}",
                "execution_attempted": False,
            }

        segment_plan: dict[str, Any] | None = None
        if not tracking_segment["final_segment"]:
            segment_plan = self._run_cartesian_plan(tracking_segment)
            if not segment_plan.get("success") or self._validated_trajectory is None:
                return {
                    "success": False,
                    "stage": "tracking_segment_plan",
                    "reason": segment_plan.get("reason")
                    or "bounded Cartesian tracking segment could not be planned",
                    "tracking_segment": tracking_segment,
                    "segment_plan": segment_plan,
                    "execution_attempted": False,
                }
            vertical_axis_validation = segment_plan.get(
                "vertical_axis_validation",
                vertical_axis_validation,
            )

        collision_validation = self._validate_trajectory_collision_states(
            self._validated_trajectory
        )
        if not collision_validation.get("success"):
            return {
                "success": False,
                "stage": "tracking_segment_collision_validation",
                "reason": collision_validation.get("reason")
                or "bounded tracking segment became collision-invalid",
                "tracking_segment": tracking_segment,
                "segment_plan": segment_plan,
                "segment_collision_validation": collision_validation,
                "execution_attempted": False,
            }

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._validated_trajectory
        client = self._execution_action_client()
        if not client.wait_for_server(timeout_sec=2.0):
            return {
                "success": False,
                "stage": "tracked_execute_trajectory_server",
                "reason": "/execute_trajectory is unavailable",
                "execution_attempted": False,
            }
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "tracked_execute_trajectory_goal",
                "reason": "MoveIt rejected the tracked Cartesian trajectory",
                "execution_attempted": False,
            }
        result_future = handle.get_result_async()
        watch_started = time.monotonic()
        deadline = watch_started + ACTION_TIMEOUT_SEC
        cancellation_reason = None
        mouth_confirmation: dict[str, Any] = {
            "available": False,
            "confirmed": False,
            "reason": "waiting for stable in-motion mouth samples",
        }
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            live_failure = self._live_ur_execution_state_failure()
            if live_failure is not None:
                cancellation_reason = live_failure
                break
            perception = self.target_tracker.current_state(
                max_age_sec=TRACKING_TARGET_MAX_AGE_SEC
            )
            if perception.get("identity_unsafe"):
                cancellation_reason = "selected mouth identity became unsafe during execution"
                break
            if not perception.get("available"):
                cancellation_reason = str(
                    perception.get("reason") or "selected mouth became stale during execution"
                )
                break
            now = time.monotonic()
            observation = self.target_tracker.observation(
                started_monotonic=max(
                    watch_started,
                    now - EXECUTION_MOUTH_DRIFT_CONFIRMATION_WINDOW_SEC,
                ),
                now_monotonic=now,
                max_age_sec=TRACKING_TARGET_MAX_AGE_SEC,
                minimum_samples=EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES,
                max_spread_m=MAX_POSE_SPREAD_M,
            )
            mouth_confirmation = self._execution_mouth_drift_confirmation(
                observation,
                self._frozen_execution_mouth_position,
            )
            if mouth_confirmation.get("confirmed"):
                cancellation_reason = (
                    "stable mouth displacement requires a fresh pre-mouth plan"
                )
                break
        if cancellation_reason is not None or not result_future.done():
            self._cancel_goal(handle)
            return {
                "success": False,
                "stage": "tracked_cartesian_execution_cancelled",
                "reason": cancellation_reason or "tracked Cartesian execution timed out",
                "execution_attempted": True,
                "execution_sent": True,
                "cancel_requested": True,
                "tracking_replan_required": bool(
                    mouth_confirmation.get("confirmed")
                ),
                "tracking_replan_reason": (
                    "target_drift"
                    if mouth_confirmation.get("confirmed")
                    else None
                ),
                "mouth_drift_confirmation": mouth_confirmation,
                "tracking_segment": tracking_segment,
                "segment_plan": segment_plan,
                "segment_collision_validation": collision_validation,
                "vertical_axis_validation": vertical_axis_validation,
                "dynamic_octomap_readiness": readiness,
                "route_strategy": DIRECT_ROUTE_STRATEGY,
                "planner": "moveit_compute_cartesian_path",
            }
        wrapped = result_future.result()
        if wrapped is None:
            return {
                "success": False,
                "stage": "tracked_cartesian_execution_result",
                "reason": "MoveIt returned no execution result",
                "execution_attempted": True,
            }
        result = wrapped.result
        execution_succeeded = int(result.error_code.val) == 1
        segment_boundary = bool(
            execution_succeeded and not tracking_segment["final_segment"]
        )
        return {
            "success": execution_succeeded,
            "stage": (
                "tracked_cartesian_segment_complete"
                if segment_boundary
                else "tracked_cartesian_execute_trajectory"
            ),
            "result_status": int(wrapped.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "execution_attempted": True,
            "execution_sent": True,
            "tracking_replan_required": segment_boundary,
            "tracking_replan_reason": (
                "segment_boundary" if segment_boundary else None
            ),
            "mouth_drift_confirmation": mouth_confirmation,
            "tracking_segment": tracking_segment,
            "segment_plan": segment_plan,
            "segment_collision_validation": collision_validation,
            "vertical_axis_validation": vertical_axis_validation,
            "dynamic_octomap_readiness": readiness,
            "route_strategy": DIRECT_ROUTE_STRATEGY,
            "planner": "moveit_compute_cartesian_path",
        }

    def _servo_track_during_premouth_hold(self, duration_sec: float) -> dict[str, Any]:
        """Track small relative mouth motion after MoveGroup releases control."""
        report: dict[str, Any] = {
            "success": False,
            "stage": "premouth_servo_tracking",
            "duration_sec": float(duration_sec),
            "reference_locked": False,
            "servo_command_count": 0,
            "maximum_command_speed_mps": 0.0,
            "maximum_mouth_displacement_m": 0.0,
            "maximum_actual_tool_displacement_m": 0.0,
            "maximum_orientation_error_rad": 0.0,
            "stop_reason": None,
        }
        perception = self.target_tracker.current_state(
            max_age_sec=TRACKING_TARGET_MAX_AGE_SEC
        )
        mouth = _finite_xyz(perception.get("selected_position_m"))
        tool = self._tool0_pose()
        if not perception.get("available") or mouth is None:
            report["stop_reason"] = str(
                perception.get("reason") or "fresh selected mouth is unavailable"
            )
            return report
        if not tool.get("available"):
            report["stop_reason"] = "live tool0 pose is unavailable"
            return report
        session = RelativeTrackingSession(
            max_target_displacement_m=TRACKING_MAX_DISPLACEMENT_M,
            max_tool_radius_m=MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
            max_linear_speed_mps=TRACKING_MAX_LINEAR_SPEED_MPS,
            max_linear_acceleration_mps2=TRACKING_MAX_LINEAR_ACCELERATION_MPS2,
        )
        session.lock(mouth, tool["position_m"])
        reference_tool_position = list(tool["position_m"])
        reference_tool_orientation = list(tool["orientation_quat_xyzw"])
        report["reference_locked"] = True
        self._latest_servo_status = None
        sink = RosServoCommandSink(
            self,
            max_linear_mps=TRACKING_MAX_LINEAR_SPEED_MPS,
            max_angular_rps=0.0,
            armed=True,
        )
        started = time.monotonic()
        last_update = started
        try:
            while rclpy.ok() and time.monotonic() - started < float(duration_sec):
                rclpy.spin_once(self, timeout_sec=0.05)
                live_failure = self._live_ur_execution_state_failure()
                if live_failure is not None:
                    report["stop_reason"] = live_failure
                    break
                if (
                    self._latest_servo_status is not None
                    and int(self._latest_servo_status.code) != ServoStatus.NO_WARNING
                ):
                    report["stop_reason"] = (
                        f"servo_status:{int(self._latest_servo_status.code)}:"
                        f"{self._latest_servo_status.message}"
                    )
                    break
                perception = self.target_tracker.current_state(
                    max_age_sec=TRACKING_TARGET_MAX_AGE_SEC
                )
                mouth = _finite_xyz(perception.get("selected_position_m"))
                if not perception.get("available") or mouth is None:
                    report["stop_reason"] = str(
                        perception.get("reason") or "tracked mouth became stale"
                    )
                    break
                if perception.get("identity_unsafe"):
                    report["stop_reason"] = "selected mouth identity became unsafe"
                    break
                tool = self._tool0_pose()
                if not tool.get("available"):
                    report["stop_reason"] = "live tool0 pose became unavailable"
                    break
                report["maximum_actual_tool_displacement_m"] = max(
                    float(report["maximum_actual_tool_displacement_m"]),
                    _norm(_subtract(tool["position_m"], reference_tool_position)),
                )
                report["maximum_orientation_error_rad"] = max(
                    float(report["maximum_orientation_error_rad"]),
                    _quaternion_distance_rad(
                        tool["orientation_quat_xyzw"],
                        reference_tool_orientation,
                    ),
                )
                now = time.monotonic()
                command = session.update(
                    mouth,
                    tool["position_m"],
                    dt_sec=now - last_update,
                )
                last_update = now
                if not command.allowed:
                    report["stop_reason"] = command.reason
                    break
                request = MotionRequest(
                    target_position=command.desired_tool_position_m,
                    target_orientation_xyzw=tuple(
                        float(value) for value in tool["orientation_quat_xyzw"]
                    ),
                    plan_only=False,
                    reason="integrated_premouth_tracking",
                    linear_velocity_mps=command.linear_velocity_mps,
                    angular_velocity_rps=(0.0, 0.0, 0.0),
                    preserve_orientation=True,
                )
                sink(request)
                report["servo_command_count"] += 1
                report["maximum_command_speed_mps"] = max(
                    float(report["maximum_command_speed_mps"]),
                    _norm(list(command.linear_velocity_mps)),
                )
                report["maximum_mouth_displacement_m"] = max(
                    float(report["maximum_mouth_displacement_m"]),
                    float(command.mouth_displacement_m),
                )
            if report["stop_reason"] is None:
                report["success"] = True
                report["stop_reason"] = "hold_duration_complete"
        finally:
            sink.halt()
        final_tool = self._tool0_pose()
        report["final_tool0_pose"] = final_tool
        return report

    def _execute_validated_trajectory(self) -> dict[str, Any]:
        """MoveGroup plan-and-execute with same-target scene-change replanning."""
        target = getattr(self, "_frozen_dynamic_target", None)
        if not isinstance(target, dict):
            return {
                "success": False,
                "stage": "dynamic_target",
                "reason": "no validated frozen dynamic target is available",
                "execution_attempted": False,
            }
        if self._validated_trajectory is None:
            return {
                "success": False,
                "stage": "dynamic_validated_trajectory",
                "reason": "no prevalidated dynamic trajectory is cached",
                "execution_attempted": False,
            }
        vertical_axis_validation = self._validate_trajectory_vertical_axis(
            self._validated_trajectory
        )
        if not vertical_axis_validation.get("success"):
            return {
                "success": False,
                "stage": "dynamic_pre_execution_vertical_axis_validation",
                "reason": vertical_axis_validation.get("reason"),
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        current_pose = self._tool0_pose()
        if not current_pose.get("available"):
            return {
                "success": False,
                "stage": "dynamic_pre_execution_vertical_axis_guard",
                "reason": "live tool0 pose is unavailable immediately before execution",
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        try:
            current_tilt = _tool_vertical_tilt_rad(
                current_pose["orientation_quat_xyzw"]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "success": False,
                "stage": "dynamic_pre_execution_vertical_axis_guard",
                "reason": f"live tool vertical-axis check failed: {exc}",
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        if current_tilt > MAX_TOOL_VERTICAL_TILT_RAD:
            return {
                "success": False,
                "stage": "dynamic_pre_execution_vertical_axis_guard",
                "reason": (
                    f"live tool tilt {math.degrees(current_tilt):.2f} deg exceeds "
                    f"the {math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
                ),
                "live_tool_tilt_rad": current_tilt,
                "vertical_axis_validation": vertical_axis_validation,
                "execution_attempted": False,
            }
        route_strategy = getattr(self, "_selected_dynamic_route_strategy", None)
        if route_strategy not in (DIRECT_ROUTE_STRATEGY, DETOUR_ROUTE_STRATEGY):
            return {
                "success": False,
                "stage": "dynamic_route_strategy",
                "reason": "no validated direct or detour route strategy is available",
                "route_strategy": route_strategy,
                "execution_attempted": False,
            }
        if route_strategy == DIRECT_ROUTE_STRATEGY:
            readiness = self.dynamic_readiness(execution_mode=True)
            if not readiness.get("success"):
                return {
                    "success": False,
                    "stage": "cartesian_execution_readiness",
                    "reason": "; ".join(readiness.get("failures", [])),
                    "dynamic_octomap_readiness": readiness,
                    "route_strategy": route_strategy,
                    "planner": "moveit_compute_cartesian_path",
                    "execution_attempted": False,
                }
            if self._integrated_tracking_enabled:
                result = self._execute_direct_with_tracking_monitor(
                    vertical_axis_validation=vertical_axis_validation,
                    readiness=readiness,
                )
            else:
                result = RealPreMouthFromPerceptionPlan._execute_validated_trajectory(
                    self
                )
            result.update(
                {
                    "route_strategy": route_strategy,
                    "planner": "moveit_compute_cartesian_path",
                    "same_target_replanning": False,
                    "cartesian_path_complete": True,
                    "cartesian_collision_checking": True,
                    "dynamic_octomap_readiness": readiness,
                }
            )
            return result
        planner = (
            f"{OMPL_PIPELINE}/{OMPL_PLANNER}"
        )
        readiness = self.dynamic_readiness(execution_mode=True)
        if not readiness.get("success"):
            return {
                "success": False,
                "stage": "dynamic_execution_readiness",
                "reason": "; ".join(readiness.get("failures", [])),
                "dynamic_octomap_readiness": readiness,
                "route_strategy": route_strategy,
                "planner": planner,
                "execution_attempted": False,
            }
        goal = self._goal_for_selected_dynamic_route(target)
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = REPLAN_ATTEMPTS
        goal.planning_options.replan_delay = REPLAN_DELAY_SEC
        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "dynamic_move_group_goal",
                "reason": "MoveGroup rejected the guarded same-target plan-and-execute goal",
                "route_strategy": route_strategy,
                "planner": planner,
                "execution_attempted": False,
            }
        result_future = handle.get_result_async()
        execution_watch_started = time.monotonic()
        deadline = execution_watch_started + ACTION_TIMEOUT_SEC
        cancel_reason: str | None = None
        mouth_drift_confirmation: dict[str, Any] = {
            "available": False,
            "confirmed": False,
            "reason": "waiting for fresh in-motion mouth samples",
            "required_samples": EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES,
            "window_sec": EXECUTION_MOUTH_DRIFT_CONFIRMATION_WINDOW_SEC,
            "threshold_m": MAX_EXECUTION_TARGET_DRIFT_M,
        }
        live_vertical_axis_guard: dict[str, Any] = {
            "maximum_allowed_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
            "maximum_observed_tilt_rad": 0.0,
            "active": True,
        }
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            live_ur_failure = self._live_ur_execution_state_failure()
            if live_ur_failure is not None:
                cancel_reason = live_ur_failure
                break
            live_pose = self._tool0_pose()
            if not live_pose.get("available"):
                cancel_reason = "live tool0 pose became unavailable during execution"
                break
            try:
                live_tilt = _tool_vertical_tilt_rad(
                    live_pose["orientation_quat_xyzw"]
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                cancel_reason = f"live tool vertical-axis check failed: {exc}"
                break
            live_vertical_axis_guard["maximum_observed_tilt_rad"] = max(
                float(live_vertical_axis_guard["maximum_observed_tilt_rad"]),
                live_tilt,
            )
            if live_tilt > MAX_TOOL_VERTICAL_TILT_RAD:
                cancel_reason = (
                    f"live tool tilt {math.degrees(live_tilt):.2f} deg exceeded "
                    f"the {math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg limit"
                )
                break
            raw = self._cloud_status(RAW_CLOUD_TOPIC)
            filtered = self._cloud_status(FILTERED_CLOUD_TOPIC)
            if not raw.get("active") or not filtered.get("active"):
                cancel_reason = "raw or filtered wrist point cloud became stale during execution"
                break
            perception = self.target_tracker.current_state(max_age_sec=MAX_MOUTH_POSE_AGE_SEC)
            if perception.get("identity_unsafe"):
                cancel_reason = "selected mouth identity became ambiguous during execution"
                break
            now = time.monotonic()
            observation = self.target_tracker.observation(
                started_monotonic=max(
                    execution_watch_started,
                    now - EXECUTION_MOUTH_DRIFT_CONFIRMATION_WINDOW_SEC,
                ),
                now_monotonic=now,
                max_age_sec=MAX_MOUTH_POSE_AGE_SEC,
                minimum_samples=EXECUTION_MOUTH_DRIFT_CONFIRMATION_MIN_SAMPLES,
                max_spread_m=MAX_POSE_SPREAD_M,
            )
            if observation.get("identity_unsafe"):
                cancel_reason = "selected mouth identity became ambiguous during execution"
                break
            mouth_drift_confirmation = self._execution_mouth_drift_confirmation(
                observation,
                self._frozen_execution_mouth_position,
            )
            if mouth_drift_confirmation.get("confirmed"):
                drift = float(mouth_drift_confirmation["drift_m"])
                cancel_reason = (
                    f"confirmed selected mouth motion is {drift:.4f} m during execution, "
                    f"above the {MAX_EXECUTION_TARGET_DRIFT_M:.4f} m limit across "
                    f"{mouth_drift_confirmation['sample_count']} stable samples"
                )
                break
        if cancel_reason is not None or not result_future.done():
            self._cancel_goal(handle)
            return {
                "success": False,
                "stage": "dynamic_execution_cancelled",
                "reason": cancel_reason or "MoveGroup execution exceeded the bounded timeout",
                "execution_attempted": True,
                "execution_sent": True,
                "cancel_requested": True,
                "same_target_replanning": True,
                "route_strategy": route_strategy,
                "planner": planner,
                "mouth_drift_confirmation": mouth_drift_confirmation,
                "tracking_replan_required": bool(
                    mouth_drift_confirmation.get("confirmed")
                ),
                "tracking_replan_reason": (
                    "target_drift"
                    if mouth_drift_confirmation.get("confirmed")
                    else None
                ),
                "vertical_axis_validation": vertical_axis_validation,
                "live_vertical_axis_guard": live_vertical_axis_guard,
            }
        wrapped = result_future.result()
        if wrapped is None:
            return {
                "success": False,
                "stage": "dynamic_execution_result",
                "reason": "MoveGroup returned no dynamic execution result",
                "route_strategy": route_strategy,
                "planner": planner,
                "execution_attempted": True,
                "execution_sent": True,
            }
        result = wrapped.result
        success = int(result.error_code.val) == 1
        response = {
            "success": success,
            "stage": "dynamic_move_group_plan_and_execute",
            "result_status": int(wrapped.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "execution_attempted": True,
            "execution_sent": True,
            "controller_goal_type": "MoveGroup plan-and-execute with scene monitoring",
            "route_strategy": route_strategy,
            "planner": planner,
            "same_target_replanning": True,
            "maximum_replan_attempts": REPLAN_ATTEMPTS,
            "replan_delay_sec": REPLAN_DELAY_SEC,
            "wait_for_clear": False,
            "mouth_drift_confirmation": mouth_drift_confirmation,
            "vertical_axis_validation": vertical_axis_validation,
            "live_vertical_axis_guard": live_vertical_axis_guard,
        }
        if not success:
            detail = result.error_code.message or (
                f"MoveGroup plan-and-execute failed with error code "
                f"{int(result.error_code.val)}"
            )
            response["reason"] = detail
            response["failure_diagnostic"] = {
                "classification": (
                    "CONTROL_FAILED"
                    if int(result.error_code.val) == -4
                    else "DYNAMIC_PLAN_OR_EXECUTION_FAILED"
                ),
                "reason": detail,
                "error_code": int(result.error_code.val),
                "error_message": result.error_code.message,
            }
            if int(result.error_code.val) == -4:
                response["reason"] = (
                    "MoveIt CONTROL_FAILED (-4) after dispatch; check the UR "
                    "reverse interface, External Control program, and scaled "
                    "joint trajectory controller"
                )
                response["failure_diagnostic"]["reason"] = response["reason"]
        return response

    @staticmethod
    def _continuous_target_report(target: ContinuousMouthTarget) -> dict[str, Any]:
        """Convert the filter state to JSON-safe diagnostics."""
        return {
            "available": bool(target.available),
            "frame_id": BASE_FRAME,
            "position_m": [float(value) for value in target.position_m],
            "predicted_position_m": [
                float(value) for value in target.predicted_position_m
            ],
            "velocity_mps": [float(value) for value in target.velocity_mps],
            "source_timestamp_sec": float(target.source_timestamp_sec),
            "target_age_sec": float(target.age_sec),
            "confidence": float(target.confidence),
            "target_id": target.target_id,
            "tracking_state": target.state.value,
            "provisional": bool(target.provisional),
            "stable": bool(target.stable),
            "sample_count": int(target.sample_count),
            "spread_m": float(target.spread_m),
            "prediction_m": float(target.prediction_m),
            "reason": target.reason,
        }

    def _publish_continuous_diagnostics(
        self,
        target: ContinuousMouthTarget,
        *,
        state: str | None = None,
        servo_status: dict[str, Any] | None = None,
        target_error_m: float | None = None,
        target_displacement_m: float | None = None,
        fallback_reason: str | None = None,
        safety_stop_reason: str | None = None,
        octomap: dict[str, Any] | None = None,
    ) -> None:
        """Publish the latest local tracking state without network calls."""
        report = self._continuous_target_report(target)
        report.update(
            {
                "state": state or target.state.value,
                "servo_status": servo_status,
                "target_error_m": target_error_m,
                "target_displacement_m": target_displacement_m,
                "fallback_reason": fallback_reason,
                "safety_stop_reason": safety_stop_reason,
                "octomap": octomap,
            }
        )
        status = String()
        status.data = json.dumps(_jsonable(report), sort_keys=True)
        self._continuous_status_publisher.publish(status)
        if target.available:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = BASE_FRAME
            pose.pose.position.x = float(target.predicted_position_m[0])
            pose.pose.position.y = float(target.predicted_position_m[1])
            pose.pose.position.z = float(target.predicted_position_m[2])
            pose.pose.orientation.w = 1.0
            self._continuous_target_publisher.publish(pose)

    def _acquire_continuous_target(self) -> dict[str, Any]:
        """Acquire stable data for three seconds, or return a provisional target."""
        self._continuous_tracker.reset(searching=True)
        started = time.monotonic()
        acquirer = InitialTargetAcquirer(
            started_monotonic_sec=started,
            timeout_sec=float(
                self._continuous_parameters["initial_target_acquisition_timeout_sec"]
            ),
        )
        decision = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            target = self._continuous_tracker.target(now_monotonic_sec=now)
            decision = acquirer.evaluate(target, now_monotonic_sec=now)
            self._publish_continuous_diagnostics(
                target,
                state=decision.state.value,
            )
            if decision.complete:
                break
        assert decision is not None
        return {
            "success": bool(decision.target.available),
            "state": decision.state.value,
            "reason": decision.reason,
            "active_search_required": bool(decision.active_search_required),
            "elapsed_sec": time.monotonic() - started,
            "target": decision.target,
            "target_report": self._continuous_target_report(decision.target),
            "last_observation_update": dict(
                self._continuous_last_observation_update
            ),
        }

    def _continuous_scene_snapshot(
        self,
        target: ContinuousMouthTarget,
        *,
        execute: bool,
    ) -> dict[str, Any]:
        """Preserve all fixed scene objects around the filtered selected target."""
        snapshot = self.snapshot(
            mouth_sample_sec=min(0.25, self.mouth_sample_seconds),
            inspect_controllers=execute,
        )
        identity = self.target_tracker.current_state(max_age_sec=1.0)
        visible = identity.get("visible_candidates")
        selected_index = identity.get("selected_candidate_index")
        if not isinstance(visible, list) or not visible:
            visible = [
                {
                    "position_m": [float(value) for value in target.position_m],
                    "image_x": 0.0,
                    "image_y": 0.0,
                    "depth_m": None,
                    "surface_normal": None,
                }
            ]
            selected_index = 0
        snapshot["mouth_pose"] = {
            "available": True,
            "stable": bool(target.stable),
            "provisional": bool(target.provisional),
            "frame_id": BASE_FRAME,
            "target_selection": self.target_selection,
            "identity_locked": True,
            "identity_unsafe": False,
            "sample_count": int(target.sample_count),
            "mean_position_m": [float(value) for value in target.position_m],
            "latest_position_m": [float(value) for value in target.position_m],
            "selected_candidate_index": int(selected_index),
            "visible_candidates": visible,
        }
        return snapshot

    def _continuous_pose_target(
        self,
        target: ContinuousMouthTarget,
        *,
        standoff_m: float,
    ) -> dict[str, Any]:
        """Build tool0 from the validated camera ray and measured straw offset."""
        tool = self._tool0_pose()
        camera = self._frame_transform(BASE_FRAME, CAMERA_OPTICAL_FRAME)
        if not tool.get("available") or not camera.get("available"):
            return {
                "success": False,
                "reason": "live tool0 or D435i optical TF is unavailable",
            }
        try:
            straw, tool0 = camera_ray_premouth_target(
                mouth_position_m=target.predicted_position_m,
                camera_position_m=camera["position_m"],
                tool_orientation_xyzw=tool["orientation_quat_xyzw"],
                straw_tip_offset_tool0_m=STRAW_TIP_OFFSET_TOOL0_M,
                standoff_m=float(standoff_m),
            )
            tilt = _tool_vertical_tilt_rad(tool["orientation_quat_xyzw"])
        except (TypeError, ValueError, RuntimeError) as exc:
            return {"success": False, "reason": f"continuous target invalid: {exc}"}
        if tilt > MAX_TOOL_VERTICAL_TILT_RAD:
            return {
                "success": False,
                "reason": (
                    f"live flange tilt {math.degrees(tilt):.2f} deg exceeds "
                    f"{math.degrees(MAX_TOOL_VERTICAL_TILT_RAD):.1f} deg"
                ),
            }
        target_in_ur_base = self._point_in_ur_base(
            [float(value) for value in tool0],
            self._frame_transform(UR_BASE_FRAME, BASE_FRAME),
        )
        if _norm(target_in_ur_base) > float(
            self._continuous_parameters["maximum_tool_radius_m"]
        ):
            return {"success": False, "reason": "continuous target is outside workspace"}
        return {
            "success": True,
            "frame_id": BASE_FRAME,
            "position_m": [float(value) for value in tool0],
            "orientation_quat_xyzw": [
                float(value) for value in tool["orientation_quat_xyzw"]
            ],
            "straw_tip_position_m": [float(value) for value in straw],
            "mouth_position_m": [float(value) for value in target.position_m],
            "standoff_m": float(standoff_m),
            "flange_vertical_axis_error_rad": float(tilt),
        }

    def _continuous_ompl_goal_for_target(
        self,
        pose_target: dict[str, Any],
    ) -> MoveGroup.Goal:
        """Keep an OMPL recovery on the current, locally reachable IK branch."""
        goal = RealDynamicObstacleAvoidancePlan._goal_for_target(
            self, pose_target
        )
        state = self.latest_joint_state
        names = list(getattr(state, "name", [])) if state is not None else []
        positions = list(getattr(state, "position", [])) if state is not None else []
        if len(names) != len(positions):
            raise RuntimeError("live joint state is incomplete for constrained OMPL recovery")
        current = {
            str(name): float(position) for name, position in zip(names, positions)
        }
        missing = sorted(
            set(CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD) - set(current)
        )
        if missing or not all(math.isfinite(value) for value in current.values()):
            raise RuntimeError(
                "live joint state is invalid for constrained OMPL recovery"
                + (": " + ", ".join(missing) if missing else "")
            )
        goal.request.path_constraints.name = (
            "continuous_local_ik_branch_with_vertical_tool_axis"
        )
        for name, limit in CONTINUOUS_RECOVERY_JOINT_EXCURSION_LIMITS_RAD.items():
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = current[name]
            constraint.tolerance_above = limit
            constraint.tolerance_below = limit
            constraint.weight = 1.0
            goal.request.path_constraints.joint_constraints.append(constraint)
        return goal

    def _continuous_plan_with_fallback(
        self,
        pose_target: dict[str, Any],
        *,
        local_recovery: bool = False,
    ) -> dict[str, Any]:
        """Plan in backend order; constrain only post-staging local recovery."""
        current_tool = self._tool0_pose()
        if not current_tool.get("available"):
            return {
                "success": False,
                "reason": "live tool0 pose is unavailable for local recovery validation",
                "attempts": [],
            }
        requested_translation = _norm(
            _subtract(pose_target["position_m"], current_tool["position_m"])
        )
        if not self._continuous_motion_arbiter.acquire(MotionCommandOwner.PLANNER):
            return {
                "success": False,
                "reason": "motion_command_owner_conflict",
                "attempts": [],
            }
        attempts: list[dict[str, Any]] = []
        selected = None
        try:
            planners = (
                ("cartesian", lambda: self._run_cartesian_plan(pose_target)),
                (
                    "pilz",
                    lambda: self._run_goal(
                        RealPreMouthFromPerceptionPlan._goal_for_target(
                            self, pose_target
                        )
                    ),
                ),
                (
                    "ompl",
                    lambda: self._run_goal(
                        self._continuous_ompl_goal_for_target(pose_target)
                        if local_recovery
                        else RealDynamicObstacleAvoidancePlan._goal_for_target(
                            self, pose_target
                        )
                    ),
                ),
            )
            for backend, planner in planners:
                self._validated_trajectory = None
                try:
                    plan = planner()
                except (RuntimeError, TypeError, ValueError) as exc:
                    plan = {
                        "success": False,
                        "stage": "continuous_recovery_planner_setup",
                        "reason": f"{backend} recovery setup failed: {exc}",
                        "execution_sent": False,
                    }
                trajectory = self._validated_trajectory
                collision = (
                    self._validate_trajectory_collision_states(trajectory)
                    if plan.get("success") and trajectory is not None
                    else None
                )
                recovery_motion = (
                    _validate_continuous_recovery_trajectory(
                        trajectory,
                        requested_tool0_translation_m=requested_translation,
                        tool_path_validation=plan.get(
                            "vertical_axis_validation"
                        ),
                    )
                    if local_recovery
                    and plan.get("success")
                    and trajectory is not None
                    else None
                )
                success = bool(
                    plan.get("success")
                    and trajectory is not None
                    and isinstance(collision, dict)
                    and collision.get("success")
                    and (
                        not local_recovery
                        or (
                            isinstance(recovery_motion, dict)
                            and recovery_motion.get("success")
                        )
                    )
                )
                attempts.append(
                    {
                        "backend": backend,
                        "success": success,
                        "plan": plan,
                        "trajectory_collision_validation": collision,
                        "recovery_motion_validation": recovery_motion,
                    }
                )
                if success:
                    selected = backend
                    break
                self._validated_trajectory = None
        finally:
            self._continuous_motion_arbiter.release(MotionCommandOwner.PLANNER)
        return {
            "success": selected is not None,
            "selected_backend": selected,
            "attempted_backends": [item["backend"] for item in attempts],
            "ompl_needed": selected == "ompl",
            "requested_tool0_translation_m": requested_translation,
            "local_recovery_constraints_active": bool(local_recovery),
            "attempts": attempts,
            "reason": None
            if selected is not None
            else next(
                (
                    item["recovery_motion_validation"]["reason"]
                    for item in reversed(attempts)
                    if isinstance(item.get("recovery_motion_validation"), dict)
                    and item["recovery_motion_validation"].get("reason")
                ),
                (
                    "Cartesian, Pilz, and constrained OMPL could not produce a validated local route"
                    if local_recovery
                    else "Cartesian, Pilz, and OMPL could not produce a validated route"
                ),
            ),
        }

    def _execute_continuous_planned_trajectory(
        self,
        *,
        confirm_real_motion: bool,
    ) -> dict[str, Any]:
        """Dispatch one cached MoveIt trajectory while Servo is excluded."""
        failures = self._execution_state_failures(
            confirm_real_motion=confirm_real_motion
        )
        if failures:
            return {
                "success": False,
                "execution_attempted": False,
                "execution_sent": False,
                "reason": "; ".join(failures),
                "failures": failures,
            }
        if self._validated_trajectory is None:
            return {
                "success": False,
                "execution_attempted": False,
                "execution_sent": False,
                "reason": "no validated recovery trajectory is cached",
            }
        collision = self._validate_trajectory_collision_states(
            self._validated_trajectory
        )
        if not collision.get("success"):
            return {
                "success": False,
                "execution_attempted": False,
                "execution_sent": False,
                "reason": collision.get("reason"),
                "trajectory_collision_validation": collision,
            }
        if not self._continuous_motion_arbiter.acquire(MotionCommandOwner.PLANNER):
            return {
                "success": False,
                "execution_attempted": False,
                "execution_sent": False,
                "reason": "Servo still owns motion during planned execution",
            }
        try:
            result = RealPreMouthFromPerceptionPlan._execute_validated_trajectory(
                self
            )
        finally:
            self._continuous_motion_arbiter.release(MotionCommandOwner.PLANNER)
        result["execution_sent"] = bool(result.get("execution_attempted"))
        result["trajectory_collision_validation"] = collision
        return result

    def _set_continuous_servo_active(self, active: bool) -> dict[str, Any]:
        """Select Twist commands and pause/unpause the configured Servo node."""
        if not self._servo_pause_client.wait_for_service(timeout_sec=1.0):
            return {"success": False, "reason": "/servo_node/pause_servo unavailable"}
        if active:
            if not self._servo_command_type_client.wait_for_service(timeout_sec=1.0):
                return {
                    "success": False,
                    "reason": "/servo_node/switch_command_type unavailable",
                }
            command_request = ServoCommandType.Request()
            command_request.command_type = ServoCommandType.Request.TWIST
            command_future = self._servo_command_type_client.call_async(command_request)
            rclpy.spin_until_future_complete(self, command_future, timeout_sec=2.0)
            command_response = command_future.result()
            if command_response is None or not command_response.success:
                return {"success": False, "reason": "Servo rejected Twist command mode"}
        pause_request = SetBool.Request()
        pause_request.data = not active
        pause_future = self._servo_pause_client.call_async(pause_request)
        rclpy.spin_until_future_complete(self, pause_future, timeout_sec=2.0)
        pause_response = pause_future.result()
        return {
            "success": bool(pause_response is not None and pause_response.success),
            "active": bool(active),
            "reason": None
            if pause_response is not None and pause_response.success
            else "Servo pause service rejected the requested state",
        }

    def _continuous_servo_approach_and_hold(
        self,
        *,
        hold_duration_sec: float,
        confirm_real_motion: bool,
        octomap: dict[str, Any],
    ) -> dict[str, Any]:
        """Continuously Servo toward the latest target, recovering only as needed."""
        report: dict[str, Any] = {
            "success": False,
            "stage": "continuous_servo_approach_and_hold",
            "servo_command_count": 0,
            "recovery_attempts": [],
            "maximum_target_error_m": 0.0,
            "maximum_target_displacement_m": 0.0,
            "hold_duration_sec": float(hold_duration_sec),
            "stop_reason": None,
            "octomap": octomap,
        }
        # Planning and executing the coarse staging trajectory can take several
        # seconds while this node is waiting on MoveIt.  Do not arm Servo from
        # the acquisition window captured before that motion: reacquire from
        # the now-stationary wrist camera so the first Servo cycle starts with
        # a fresh, filtered target in the staging camera view.
        reacquisition = self._acquire_continuous_target()
        report["post_staging_target_acquisition"] = {
            key: value for key, value in reacquisition.items() if key != "target"
        }
        if not reacquisition.get("success"):
            report["stop_reason"] = (
                reacquisition.get("reason")
                or "post_staging_target_acquisition_failed"
            )
            return report
        failures = self._execution_state_failures(
            confirm_real_motion=confirm_real_motion
        )
        if failures:
            report["stop_reason"] = "; ".join(failures)
            return report
        initial_validity = self._search_start_state_validity(
            time.monotonic() + 0.75
        )
        report["initial_state_validity"] = initial_validity
        if not initial_validity.get("available"):
            report["stop_reason"] = (
                initial_validity.get("reason")
                or "continuous state-validity service unavailable"
            )
            return report
        if not initial_validity.get("valid"):
            report["stop_reason"] = "continuous_start_state_collision"
            return report
        activation = self._set_continuous_servo_active(True)
        report["servo_activation"] = activation
        if not activation.get("success"):
            report["stop_reason"] = activation.get("reason")
            return report
        if not self._continuous_motion_arbiter.acquire(MotionCommandOwner.SERVO):
            self._set_continuous_servo_active(False)
            report["stop_reason"] = "planner still owns motion"
            return report
        self._continuous_servo_controller.reset()
        sink = RosServoCommandSink(
            self,
            topic=str(self._continuous_parameters["servo_twist_topic"]),
            max_linear_mps=float(
                self._continuous_parameters["maximum_linear_speed_mps"]
            ),
            max_angular_rps=float(
                self._continuous_parameters["maximum_angular_speed_rps"]
            ),
            armed=True,
        )
        started = time.monotonic()
        last_update = started
        hold_started: float | None = None
        diagnostics: list[dict[str, Any]] = []
        validity_period_sec = float(
            self._continuous_parameters["state_validity_check_period_sec"]
        )
        next_validity_check = started
        latest_state_validity = initial_validity
        report["state_validity_check_period_sec"] = validity_period_sec
        report["state_validity_check_count"] = 1
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                now = time.monotonic()
                live_failure = self._live_ur_execution_state_failure()
                if live_failure is not None:
                    report["stop_reason"] = live_failure
                    break
                servo_status = self._continuous_servo_status_snapshot(now=now)
                if not servo_status.get("success"):
                    report["stop_reason"] = servo_status.get("reason")
                    break
                if now >= next_validity_check:
                    latest_state_validity = self._search_start_state_validity(
                        time.monotonic() + max(0.20, validity_period_sec)
                    )
                    report["state_validity_check_count"] += 1
                    next_validity_check = time.monotonic() + validity_period_sec
                    if not latest_state_validity.get("available"):
                        report["stop_reason"] = (
                            latest_state_validity.get("reason")
                            or "continuous state-validity watchdog unavailable"
                        )
                        break
                    if not latest_state_validity.get("valid"):
                        report["stop_reason"] = "continuous_state_collision"
                        report["collision_state_validity"] = latest_state_validity
                        break
                target = self._continuous_tracker.target(now_monotonic_sec=now)
                tool = self._tool0_pose()
                camera = self._frame_transform(BASE_FRAME, CAMERA_OPTICAL_FRAME)
                if not tool.get("available") or not camera.get("available"):
                    report["stop_reason"] = "live tool0 or camera TF became unavailable"
                    break
                decision = self._continuous_servo_controller.update(
                    target,
                    current_tool0_position_m=tool["position_m"],
                    current_tool0_orientation_xyzw=tool["orientation_quat_xyzw"],
                    camera_position_m=camera["position_m"],
                    servo_status_code=int(servo_status["code"]),
                    elapsed_sec=now - started,
                    dt_sec=now - last_update,
                )
                last_update = now
                report["maximum_target_error_m"] = max(
                    float(report["maximum_target_error_m"]),
                    0.0
                    if not math.isfinite(decision.target_error_m)
                    else float(decision.target_error_m),
                )
                report["maximum_target_displacement_m"] = max(
                    float(report["maximum_target_displacement_m"]),
                    float(decision.target_displacement_m),
                )
                self._publish_continuous_diagnostics(
                    target,
                    state=decision.state,
                    servo_status=servo_status,
                    target_error_m=decision.target_error_m,
                    target_displacement_m=decision.target_displacement_m,
                    fallback_reason=decision.fallback_reason,
                    safety_stop_reason=decision.safety_stop_reason,
                    octomap=octomap,
                )
                diagnostics.append(
                    {
                        "elapsed_sec": now - started,
                        "state": decision.state,
                        "target_age_sec": target.age_sec,
                        "target_confidence": target.confidence,
                        "target_error_m": decision.target_error_m,
                        "target_displacement_m": decision.target_displacement_m,
                        "servo_status": servo_status,
                        "state_validity": latest_state_validity,
                        "fallback_reason": decision.fallback_reason,
                        "safety_stop_reason": decision.safety_stop_reason,
                    }
                )
                if len(diagnostics) > 100:
                    diagnostics.pop(0)
                if decision.safety_stop_reason is not None:
                    report["stop_reason"] = decision.safety_stop_reason
                    break
                if decision.recovery_required:
                    sink.halt()
                    self._set_continuous_servo_active(False)
                    self._continuous_motion_arbiter.release(MotionCommandOwner.SERVO)
                    recovery_target = {
                        "frame_id": BASE_FRAME,
                        "position_m": list(decision.desired_tool0_position_m),
                        "orientation_quat_xyzw": list(
                            tool["orientation_quat_xyzw"]
                        ),
                    }
                    recovery_plan = self._continuous_plan_with_fallback(
                        recovery_target,
                        local_recovery=True,
                    )
                    recovery_execution = None
                    if recovery_plan.get("success"):
                        recovery_execution = self._execute_continuous_planned_trajectory(
                            confirm_real_motion=confirm_real_motion
                        )
                    recovery = {
                        "reason": decision.fallback_reason,
                        "plan": recovery_plan,
                        "execution": recovery_execution,
                    }
                    report["recovery_attempts"].append(recovery)
                    if (
                        not recovery_plan.get("success")
                        or not isinstance(recovery_execution, dict)
                        or not recovery_execution.get("success")
                    ):
                        report["stop_reason"] = (
                            recovery_plan.get("reason")
                            or (recovery_execution or {}).get("reason")
                            or "continuous recovery failed"
                        )
                        break
                    current_target = self._continuous_tracker.target(
                        now_monotonic_sec=time.monotonic()
                    )
                    if not current_target.available:
                        report["stop_reason"] = (
                            current_target.reason
                            or "target became stale during recovery"
                        )
                        break
                    if len(report["recovery_attempts"]) >= 3:
                        report["stop_reason"] = "continuous recovery attempt limit"
                        break
                    activation = self._set_continuous_servo_active(True)
                    if not activation.get("success"):
                        report["stop_reason"] = activation.get("reason")
                        break
                    if not self._continuous_motion_arbiter.acquire(
                        MotionCommandOwner.SERVO
                    ):
                        report["stop_reason"] = "could not reacquire Servo ownership"
                        break
                    sink.armed = True
                    self._continuous_servo_controller.reset()
                    last_update = time.monotonic()
                    hold_started = None
                    continue
                request = MotionRequest(
                    target_position=decision.desired_tool0_position_m,
                    target_orientation_xyzw=tuple(
                        float(value) for value in tool["orientation_quat_xyzw"]
                    ),
                    plan_only=False,
                    reason="continuous_camera_ray_premouth_tracking",
                    linear_velocity_mps=decision.linear_velocity_mps,
                    angular_velocity_rps=decision.angular_velocity_rps,
                    preserve_orientation=True,
                )
                sink(request)
                report["servo_command_count"] += 1
                if decision.hold_ready:
                    if hold_started is None:
                        hold_started = now
                    if now - hold_started >= float(hold_duration_sec):
                        report["success"] = True
                        report["stop_reason"] = "hold_duration_complete"
                        break
                else:
                    hold_started = None
        finally:
            sink.halt()
            self._set_continuous_servo_active(False)
            if self._continuous_motion_arbiter.owner == MotionCommandOwner.SERVO:
                self._continuous_motion_arbiter.release(MotionCommandOwner.SERVO)
        report["recent_diagnostics"] = diagnostics
        report["elapsed_sec"] = time.monotonic() - started
        report["hold_elapsed_sec"] = (
            0.0
            if hold_started is None
            else max(0.0, time.monotonic() - hold_started)
        )
        report["final_tool0_pose"] = self._tool0_pose()
        return report

    def run_continuous_integrated(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        allow_validated_camera_ray_execute: bool,
        no_execute: bool,
        hold_duration_sec: float,
        use_octomap: bool,
    ) -> tuple[int, dict[str, Any]]:
        """Run the separate continuous-Servo feed-water workflow."""
        report: dict[str, Any] = {
            "success": False,
            "mode": "execute" if execute else "plan",
            "stage": "continuous_tracking_initialization",
            "execution_attempted": False,
            "execution_sent": False,
            "continuous_mouth_tracking": True,
            "existing_one_shot_path_modified": False,
            "existing_segmented_tracking_path_modified": False,
            "parameters": dict(self._continuous_parameters),
            "backend_priority": ["servo", "cartesian", "pilz", "ompl"],
            "servo_command_interface": str(
                self._continuous_parameters["servo_twist_topic"]
            ),
            "integrated_real_feed_water": {
                "mouth_tracking_during_execution": True,
                "tracking_policy": "continuous_moveit_servo_approach_and_hold",
                "segmented_approach_enabled": False,
                "multi_target_identity_lock": True,
                "active_search_vertical_axis_constraint": True,
                "tool_axis_spin_free": True,
                "vertical_axis_detour_after_direct_rejection": True,
                "same_target_replanning": True,
            },
        }
        if execute and no_execute:
            report.update(stage="no_execute_policy", reason="--no-execute prohibits motion")
            return 2, report
        if execute and not allow_validated_camera_ray_execute:
            report.update(
                stage="premouth_policy_execution_gate",
                reason="--allow-validated-camera-ray-execute is required",
            )
            return 2, report
        if not MIN_PREMOUTH_HOLD_SEC <= float(hold_duration_sec) <= MAX_PREMOUTH_HOLD_SEC:
            report.update(stage="hold_duration_validation", reason="hold duration is invalid")
            return 2, report

        combined = self._apply_combined_tool_collision_geometry()
        report["combined_tool_collision_geometry"] = combined
        if not combined.get("success"):
            report.update(
                stage="combined_tool_collision_geometry",
                reason=combined.get("reason"),
            )
            return 2, report
        initial_before = self._current_initial_position_status()
        report["initial_position_before"] = initial_before
        if not initial_before.get("available"):
            report.update(stage="initial_position_entry_check", reason=initial_before.get("reason"))
            return 2, report
        if not initial_before.get("at_initial_position"):
            code, preflight_return = self.return_to_initial_position(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
                use_octomap=use_octomap,
            )
            report["preflight_return_to_initial_position"] = preflight_return
            if not execute:
                report.update(
                    stage="initial_position_required_plan_only",
                    reason="plan-only mode cannot move to initial_position",
                )
                return 2, report
            initial_after = self._current_initial_position_status()
            if code != 0 or not initial_after.get("at_initial_position"):
                report.update(
                    stage="preflight_return_to_initial_position_refused",
                    reason=preflight_return.get("reason") or initial_after.get("reason"),
                    execution_attempted=bool(preflight_return.get("execution_attempted")),
                    execution_sent=bool(preflight_return.get("execution_sent")),
                )
                return 2, report

        acquisition = self._acquire_continuous_target()
        report["initial_target_acquisition"] = {
            key: value for key, value in acquisition.items() if key != "target"
        }
        if not acquisition.get("success"):
            search = self.active_search(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
                use_octomap=use_octomap,
            )
            report["active_search"] = search
            report["execution_attempted"] = bool(search.get("trajectory_sent"))
            report["execution_sent"] = bool(search.get("trajectory_sent"))
            if not search.get("success"):
                report.update(
                    stage="continuous_initial_target_unavailable",
                    reason=search.get("reason") or acquisition.get("reason"),
                )
                return 2, report
            acquisition = self._acquire_continuous_target()
            report["post_search_target_acquisition"] = {
                key: value for key, value in acquisition.items() if key != "target"
            }
            if not acquisition.get("success"):
                report.update(
                    stage="continuous_post_search_target_unavailable",
                    reason=acquisition.get("reason"),
                )
                return 2, report
        target = acquisition["target"]
        report["detected_mouth_pose"] = {
            "frame_id": BASE_FRAME,
            "position_m": [float(value) for value in target.position_m],
            "tracking": self._continuous_target_report(target),
        }
        snapshot = self._continuous_scene_snapshot(target, execute=execute)
        readiness = self._base_readiness_failures(snapshot)
        if readiness:
            report.update(
                stage="continuous_readiness",
                reason="; ".join(readiness),
                failures=readiness,
            )
            return 2, report
        scene = self._apply_multi_person_planning_scene(snapshot)
        report["fixed_planning_scene"] = scene
        if not scene.get("success"):
            report.update(stage="fixed_planning_scene", reason=scene.get("reason"))
            return 2, report
        scene_sync = self._synchronize_servo_planning_scene()
        report["servo_planning_scene_synchronization"] = scene_sync
        if not scene_sync.get("success"):
            report.update(
                stage="servo_planning_scene_synchronization",
                reason=scene_sync.get("reason"),
            )
            return 2, report
        monitored_scene, _ = self._planning_scene_geometry()
        report["monitored_planning_scene"] = monitored_scene
        if not use_octomap and monitored_scene.get("octomap", {}).get("present"):
            report["octomap"] = {
                **octomap_layer_status(
                    use_octomap=False,
                    rebuild_succeeded=None,
                    occupancy_present=True,
                ),
            }
            report.update(
                stage="octomap_configuration_mismatch",
                reason=(
                    "continuous mode requested use_octomap=false but the running "
                    "MoveGroup still contains occupancy; restart MoveIt with "
                    "use_octomap:=false before planning"
                ),
            )
            return 2, report

        final_target = self._continuous_pose_target(
            target,
            standoff_m=float(self._continuous_parameters["final_pre_mouth_standoff_m"]),
        )
        if not final_target.get("success"):
            report.update(stage="continuous_target_generation", reason=final_target.get("reason"))
            return 2, report
        report["target_tool0_pose"] = {
            "frame_id": BASE_FRAME,
            "position_m": list(final_target["position_m"]),
            "orientation_quat_xyzw": list(
                final_target["orientation_quat_xyzw"]
            ),
        }
        report["pre_mouth_pose"] = {
            "frame_id": BASE_FRAME,
            "position_m": list(final_target["straw_tip_position_m"]),
            "standoff_m": float(final_target["standoff_m"]),
        }
        octomap_rebuild = None
        if use_octomap:
            dynamic = self.dynamic_readiness(execution_mode=True if execute else None)
            if dynamic.get("success"):
                octomap_rebuild = self._prepare_dynamic_scene_for_goal_selection(
                    mouth=[float(value) for value in target.position_m],
                    original_pre_mouth=list(final_target["straw_tip_position_m"]),
                    snapshot=snapshot,
                    planning_scene_application=scene,
                )
            octomap = octomap_layer_status(
                use_octomap=True,
                rebuild_succeeded=bool(
                    dynamic.get("success")
                    and isinstance(octomap_rebuild, dict)
                    and octomap_rebuild.get("success")
                ),
            )
            octomap["readiness"] = dynamic
            octomap["rebuild"] = octomap_rebuild
            if not octomap["dynamic_obstacle_layer_active"]:
                report["octomap"] = octomap
                report.update(
                    stage="dynamic_obstacle_layer_unavailable",
                    reason=(
                        "the explicitly enabled stationary OctoMap rebuild failed; "
                        "fixed PlanningScene objects remain active, but outbound "
                        "motion is withheld rather than using stale occupancy"
                    ),
                )
                return 2, report
        else:
            octomap = octomap_layer_status(
                use_octomap=False,
                rebuild_succeeded=None,
            )
        report["octomap"] = octomap

        staging = self._continuous_pose_target(
            target,
            standoff_m=float(self._continuous_parameters["coarse_staging_standoff_m"]),
        )
        report["coarse_staging_target"] = staging
        if not staging.get("success"):
            report.update(stage="coarse_staging_target", reason=staging.get("reason"))
            return 2, report
        staging_plan = self._continuous_plan_with_fallback(staging)
        report["coarse_staging_plan"] = staging_plan
        if not staging_plan.get("success"):
            report.update(stage="coarse_staging_plan", reason=staging_plan.get("reason"))
            return 2, report
        if not execute:
            return_validation_code, return_validation = self.return_to_initial_position(
                execute=False,
                confirm_real_motion=False,
                use_octomap=use_octomap,
            )
            report.update(
                success=return_validation_code == 0,
                stage="continuous_tracking_plan_only_validated",
                reason=return_validation.get("reason"),
                return_to_initial_position=return_validation,
                execution_disabled=True,
                servo_commands_sent=0,
                pre_mouth_hold={
                    "completed": False,
                    "plan_only": True,
                    "motion_command_sent": False,
                },
            )
            return (0 if report["success"] else 2), report

        staging_execution = self._execute_continuous_planned_trajectory(
            confirm_real_motion=confirm_real_motion
        )
        report["coarse_staging_execution"] = staging_execution
        report["execution_attempted"] = bool(staging_execution.get("execution_attempted"))
        report["execution_sent"] = bool(staging_execution.get("execution_sent"))
        if not staging_execution.get("success"):
            recovery = self._attempt_failure_recovery_return(
                execute=True,
                confirm_real_motion=confirm_real_motion,
                motion_sent=bool(staging_execution.get("execution_sent")),
                use_octomap=use_octomap,
            )
            report.update(
                stage="coarse_staging_execution",
                reason=staging_execution.get("reason"),
                failure_recovery_return=recovery,
            )
            return 2, report
        try:
            servo = self._continuous_servo_approach_and_hold(
                hold_duration_sec=float(hold_duration_sec),
                confirm_real_motion=confirm_real_motion,
                octomap=octomap,
            )
        except Exception as exc:
            # Preserve the single guarded recovery path even if the tracking
            # controller raises unexpectedly after staging motion was sent.
            servo = {
                "success": False,
                "stage": "continuous_servo_approach_and_hold",
                "servo_command_count": 0,
                "stop_reason": (
                    "continuous_tracking_exception:"
                    f"{exc.__class__.__name__}:{exc}"
                ),
                "exception_caught": True,
            }
        report["continuous_servo"] = servo
        report["pre_mouth_hold"] = {
            "completed": bool(servo.get("success")),
            "duration_sec": float(hold_duration_sec),
            "motion_command_sent": bool(servo.get("servo_command_count", 0)),
            "tracking": servo,
        }
        report["execution_sent"] = bool(
            report["execution_sent"] or servo.get("servo_command_count", 0)
        )
        if not servo.get("success"):
            recovery = self._attempt_failure_recovery_return(
                execute=True,
                confirm_real_motion=confirm_real_motion,
                motion_sent=bool(report["execution_sent"]),
                use_octomap=use_octomap,
            )
            report.update(
                stage="continuous_servo_stopped",
                reason=servo.get("stop_reason"),
                failure_recovery_return=recovery,
            )
            return 2, report
        return_code, return_result = self.return_to_initial_position(
            execute=True,
            confirm_real_motion=confirm_real_motion,
            use_octomap=use_octomap,
        )
        report["return_to_initial_position"] = return_result
        if return_code != 0 or not return_result.get("success"):
            report.update(
                stage="return_to_initial_position_refused",
                reason=return_result.get("reason"),
            )
            return 2, report
        report.update(
            success=True,
            stage="continuous_tracking_returned_initial_position",
            reason=None,
            final_state="initial_position",
        )
        return 0, report

    def run_integrated(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        allow_validated_camera_ray_execute: bool,
        no_execute: bool,
        track_mouth_during_execution: bool = False,
        continuous_mouth_tracking: bool = False,
        use_octomap: bool = False,
        hold_duration_sec: float = DEFAULT_PREMOUTH_HOLD_SEC,
    ) -> tuple[int, dict[str, Any]]:
        session_scene_reset = (
            self._ensure_previous_session_collision_geometry_reset()
        )
        if not session_scene_reset.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "previous_session_collision_geometry_reset",
                "reason": (
                    session_scene_reset.get("reason")
                    or "previous-session collision geometry could not be reset"
                ),
                "previous_session_collision_geometry_reset": session_scene_reset,
                "execution_attempted": False,
                "execution_sent": False,
            }
        if continuous_mouth_tracking:
            if track_mouth_during_execution:
                return 2, {
                    "success": False,
                    "stage": "tracking_mode_selection",
                    "reason": (
                        "continuous and segmented tracking modes are mutually exclusive"
                    ),
                    "previous_session_collision_geometry_reset": session_scene_reset,
                    "execution_attempted": False,
                    "execution_sent": False,
                }
            code, result = self.run_continuous_integrated(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
                allow_validated_camera_ray_execute=(
                    allow_validated_camera_ray_execute
                ),
                no_execute=no_execute,
                hold_duration_sec=hold_duration_sec,
                use_octomap=use_octomap,
            )
            result["previous_session_collision_geometry_reset"] = session_scene_reset
            return code, result
        contract = {
            "previous_session_collision_geometry_reset": session_scene_reset,
            "multi_target_identity_lock": True,
            "target_selection": self.target_selection,
            "active_search": True,
            "active_search_planner": (
                f"{SEARCH_PLANNING_PIPELINE}/{SEARCH_PLANNER}"
            ),
            "ompl_active_search_enabled": False,
            "ompl_dynamic_obstacle_detour_enabled": True,
            "translation_only_search": False,
            "position_only_active_search_goals": False,
            "active_search_vertical_axis_constraint": True,
            "maximum_tool_vertical_tilt_deg": math.degrees(
                MAX_TOOL_VERTICAL_TILT_RAD
            ),
            "tool_axis_spin_free": True,
            "active_search_intermediate_flange_orientation_unconstrained": False,
            "fk_waypoint_vertical_axis_validation": True,
            "maximum_joint_excursion_rad": SEARCH_MAX_JOINT_EXCURSION_RAD,
            "maximum_cumulative_joint_travel_rad": (
                SEARCH_MAX_CUMULATIVE_JOINT_TRAVEL_RAD
            ),
            "maximum_search_trajectory_duration_sec": (
                SEARCH_MAX_TRAJECTORY_DURATION_SEC
            ),
            "rotation_search_enabled": True,
            "left_right_search_strategy": "MoveIt tool0-local-Z pose rotations",
            "left_right_rotation_each_side_deg": SEARCH_WRIST_Z_ANGLE_DEG,
            "left_right_total_sweep_deg": SEARCH_WRIST_Z_TOTAL_SWEEP_DEG,
            "wrist_3_direct_command": False,
            "active_search_direction_reference": {
                "backward_up_down": (
                    f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
                ),
                "left_right": f"{TOOL_FRAME} local Z rotation",
            },
            "active_search_camera_extrinsic_applied": True,
            "active_search_order": [
                "backward_wide",
                "scan_left",
                "scan_right",
                "scan_up",
                "scan_down",
            ],
            "adaptive_search_distances_m": {
                "backward": list(SEARCH_BACK_CANDIDATE_DISTANCES_M),
                "up_down": list(SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M),
                "skip_unreachable_direction": True,
            },
            "adaptive_left_right_angles_deg": [
                math.degrees(value)
                for value in SEARCH_WRIST_Z_CANDIDATE_ANGLES_RAD
            ],
            "dynamic_obstacle_avoidance": True,
            "direct_clear_path_first": True,
            "vertical_axis_detour_after_direct_rejection": True,
            "detour_final_goal_orientation_constraint": True,
            "combined_static_and_dynamic_scene_checks": True,
            "same_target_replanning": True,
            "wait_for_clear": False,
            "automatic_return_to_initial_position": True,
            "initial_position_required_before_workflow": True,
            "failure_recovery_return_enabled": True,
            "return_target": "initial_position",
            "return_collision_checking": True,
            "return_vertical_axis_constraint": True,
            "return_wrist_3_direct_command": False,
            "pre_mouth_hold_duration_sec": float(hold_duration_sec),
            "mouth_tracking_during_execution": bool(
                track_mouth_during_execution
            ),
            "tracking_policy": (
                "bounded_cartesian_segments_with_fresh_target_replanning_then_"
                "relative_servo_at_premouth"
                if track_mouth_during_execution
                else "disabled_one_shot_frozen_target"
            ),
            "tracking_maximum_target_drift_replans": (
                TRACKING_MAX_TARGET_DRIFT_REPLANS
            ),
            "tracking_maximum_approach_segments": TRACKING_MAX_APPROACH_SEGMENTS,
            "tracking_segment_maximum_translation_m": (
                TRACKING_SEGMENT_MAX_TRANSLATION_M
            ),
            "tracking_segment_maximum_rotation_deg": math.degrees(
                TRACKING_SEGMENT_MAX_ROTATION_RAD
            ),
            "tracking_maximum_approach_duration_sec": (
                TRACKING_MAX_APPROACH_DURATION_SEC
            ),
            "tracking_maximum_displacement_m": TRACKING_MAX_DISPLACEMENT_M,
            "tracking_maximum_linear_speed_mps": TRACKING_MAX_LINEAR_SPEED_MPS,
            "tracking_maximum_linear_acceleration_mps2": (
                TRACKING_MAX_LINEAR_ACCELERATION_MPS2
            ),
        }
        if (
            isinstance(hold_duration_sec, bool)
            or not math.isfinite(float(hold_duration_sec))
            or not MIN_PREMOUTH_HOLD_SEC
            <= float(hold_duration_sec)
            <= MAX_PREMOUTH_HOLD_SEC
        ):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "hold_duration_validation",
                "reason": (
                    "pre-mouth hold must be finite and between "
                    f"{MIN_PREMOUTH_HOLD_SEC:.0f} and {MAX_PREMOUTH_HOLD_SEC:.0f} seconds"
                ),
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        if execute and no_execute:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "no_execute_policy",
                "reason": "--no-execute prohibits all real motion",
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        if execute and not allow_validated_camera_ray_execute:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "premouth_policy_execution_gate",
                "reason": "--allow-validated-camera-ray-execute is required",
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        combined_tool_geometry = self._apply_combined_tool_collision_geometry()
        contract["combined_camera_cup_holder_straw_collision_geometry"] = {
            "required": True,
            "verified": bool(combined_tool_geometry.get("success")),
            "object_id": combined_tool_geometry.get("object_id"),
            "link_name": combined_tool_geometry.get("link_name"),
            "dimensions_m": combined_tool_geometry.get("dimensions_m"),
            "center_tool0_m": combined_tool_geometry.get("center_tool0_m"),
            "follows_tool0": combined_tool_geometry.get("follows_tool0"),
        }
        if not combined_tool_geometry.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "combined_tool_collision_geometry",
                "reason": (
                    combined_tool_geometry.get("reason")
                    or "combined camera/cup-holder/straw collision geometry could not be verified"
                ),
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        initial_before = self._current_initial_position_status()
        preflight_return: dict[str, Any] = {
            "required": False,
            "attempted": False,
            "success": bool(initial_before.get("at_initial_position")),
            "execution_sent": False,
            "initial_position_before": initial_before,
        }
        if not initial_before.get("available"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "initial_position_entry_check",
                "reason": initial_before.get("reason")
                or "current initial-position status is unavailable",
                "preflight_return_to_initial_position": preflight_return,
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        if not initial_before.get("at_initial_position"):
            preflight_return["required"] = True
            return_code, return_result = self.return_to_initial_position(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
            )
            preflight_return.update(
                {
                    "attempted": bool(execute),
                    "return_result": return_result,
                    "execution_sent": bool(return_result.get("execution_sent")),
                }
            )
            if not execute:
                preflight_return.update(
                    {
                        "success": False,
                        "reason": (
                            "plan-only mode cannot move the UR10e to initial_position"
                        ),
                    }
                )
                return 2, {
                    "success": False,
                    "mode": "plan",
                    "stage": "initial_position_required_plan_only",
                    "reason": preflight_return["reason"],
                    "preflight_return_to_initial_position": preflight_return,
                    "combined_tool_collision_geometry": combined_tool_geometry,
                    "execution_attempted": False,
                    "execution_sent": False,
                    "integrated_real_feed_water": contract,
                }
            initial_after = self._current_initial_position_status()
            preflight_return["initial_position_after"] = initial_after
            preflight_return["success"] = bool(
                return_code == 0
                and return_result.get("success")
                and initial_after.get("at_initial_position")
            )
            preflight_return["reason"] = (
                None
                if preflight_return["success"]
                else (
                    return_result.get("reason")
                    or initial_after.get("reason")
                    or "preflight return to initial_position was not verified"
                )
            )
            if not preflight_return["success"]:
                return 2, {
                    "success": False,
                    "mode": "execute",
                    "stage": "preflight_return_to_initial_position_refused",
                    "reason": preflight_return["reason"],
                    "preflight_return_to_initial_position": preflight_return,
                    "combined_tool_collision_geometry": combined_tool_geometry,
                    "execution_attempted": bool(
                        return_result.get("execution_attempted")
                    ),
                    "execution_sent": bool(return_result.get("execution_sent")),
                    "integrated_real_feed_water": contract,
                }
        else:
            preflight_return["reason"] = None
            preflight_return["initial_position_after"] = initial_before
        # Initial-position recovery is a prerequisite for every outbound
        # feed-water operation.  Evaluate the outbound dynamic layer only
        # after that prerequisite has been checked and, when authorized,
        # guardedly restored.
        dynamic = self.dynamic_readiness(execution_mode=True if execute else None)
        if not dynamic.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "dynamic_octomap_readiness",
                "reason": "; ".join(dynamic.get("failures", [])),
                "dynamic_octomap_readiness": dynamic,
                "preflight_return_to_initial_position": preflight_return,
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": bool(preflight_return.get("execution_sent")),
                "execution_sent": bool(preflight_return.get("execution_sent")),
                "integrated_real_feed_water": contract,
            }
        search = self.active_search(
            execute=execute,
            confirm_real_motion=confirm_real_motion,
        )
        if not search.get("success"):
            search_motion_sent = bool(search.get("trajectory_sent"))
            recovery = self._attempt_failure_recovery_return(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
                motion_sent=search_motion_sent,
            )
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "active_search",
                "reason": search.get("reason") or "active search did not recover the selected mouth",
                "active_search": search,
                "preflight_return_to_initial_position": preflight_return,
                "failure_recovery_return": recovery,
                "dynamic_octomap_readiness": dynamic,
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": bool(
                    preflight_return.get("execution_sent")
                    or recovery.get("execution_sent")
                    or any(
                        isinstance(step, dict)
                        and isinstance(step.get("execution_result"), dict)
                        and step["execution_result"].get("execution_attempted")
                        for step in search.get("search_steps", [])
                    )
                ),
                "execution_sent": bool(
                    preflight_return.get("execution_sent")
                    or search_motion_sent
                    or recovery.get("execution_sent")
                ),
                "return_execution_attempted": bool(recovery.get("attempted")),
                "return_execution_sent": bool(recovery.get("execution_sent")),
                "final_state": recovery.get("final_state"),
                "integrated_real_feed_water": contract,
            }
        self._integrated_tracking_enabled = bool(track_mouth_during_execution)
        tracking_replans: list[dict[str, Any]] = []
        any_execution_attempted = False
        any_execution_sent = False
        target_drift_replans = 0
        approach_started = time.monotonic()
        if execute:
            for attempt in range(TRACKING_MAX_APPROACH_SEGMENTS):
                if (
                    track_mouth_during_execution
                    and time.monotonic() - approach_started
                    > TRACKING_MAX_APPROACH_DURATION_SEC
                ):
                    code = 2
                    result = {
                        "success": False,
                        "mode": "execute",
                        "stage": "tracking_approach_duration",
                        "reason": "segmented tracking approach exceeded its duration limit",
                        "execution_attempted": any_execution_attempted,
                        "execution_sent": any_execution_sent,
                    }
                    break
                if attempt > 0:
                    stationary = self._wait_for_tracking_replan_stationary()
                    tracking_replans[-1]["pre_replan_stationary_wait"] = stationary
                    if not stationary.get("success"):
                        code = 2
                        result = {
                            "success": False,
                            "mode": "execute",
                            "stage": "tracking_replan_stationary_wait",
                            "reason": (
                                "UR10e did not become stationary after tracked "
                                "trajectory cancellation"
                            ),
                            "pre_replan_stationary_wait": stationary,
                            "execution_attempted": any_execution_attempted,
                            "execution_sent": any_execution_sent,
                        }
                        break
                code, result = super().execute(
                    confirm_real_motion=confirm_real_motion,
                    allow_validated_camera_ray_execute=allow_validated_camera_ray_execute,
                    allow_validated_feeding_vector_execute=False,
                    allow_validated_tcp_forward_execute=False,
                    no_execute=False,
                )
                execution_result = result.get("execution_result", {})
                if isinstance(execution_result, dict):
                    any_execution_attempted = bool(
                        any_execution_attempted
                        or execution_result.get("execution_attempted")
                    )
                    any_execution_sent = bool(
                        any_execution_sent
                        or execution_result.get("execution_sent")
                    )
                pre_execution_drift_replan = (
                    track_mouth_during_execution
                    and self._pre_execution_target_drift_requires_replan(result)
                )
                execution_replan = bool(
                    isinstance(execution_result, dict)
                    and execution_result.get("tracking_replan_required")
                )
                replan_required = bool(
                    pre_execution_drift_replan or execution_replan
                )
                replan_reason = (
                    "pre_execution_target_drift"
                    if pre_execution_drift_replan
                    else (
                        execution_result.get("tracking_replan_reason")
                        if isinstance(execution_result, dict)
                        else None
                    )
                )
                if replan_reason in (
                    "target_drift",
                    "pre_execution_target_drift",
                ):
                    target_drift_replans += 1
                tracking_replans.append(
                    {
                        "attempt": attempt + 1,
                        "success": bool(result.get("success")),
                        "stage": result.get("stage"),
                        "execution_stage": execution_result.get("stage")
                        if isinstance(execution_result, dict)
                        else None,
                        "reason": result.get("reason"),
                        "replan_required": replan_required,
                        "replan_reason": replan_reason,
                        "tracking_segment": (
                            execution_result.get("tracking_segment")
                            if isinstance(execution_result, dict)
                            else None
                        ),
                        "mouth_drift_confirmation": (
                            execution_result.get("mouth_drift_confirmation")
                            if isinstance(execution_result, dict)
                            else None
                        ),
                        "pre_execution_target_drift_m": result.get(
                            "pre_execution_target_drift_m"
                        ),
                    }
                )
                if code == 0 and result.get("success"):
                    break
                if (
                    target_drift_replans > TRACKING_MAX_TARGET_DRIFT_REPLANS
                ):
                    code = 2
                    result.update(
                        {
                            "success": False,
                            "stage": "tracking_target_drift_replan_limit",
                            "reason": (
                                "tracked target exceeded the bounded drift-replan limit"
                            ),
                        }
                    )
                    break
                if not (
                    track_mouth_during_execution
                    and replan_required
                    and attempt + 1 < TRACKING_MAX_APPROACH_SEGMENTS
                ):
                    break
            else:
                code = 2
                result = {
                    "success": False,
                    "mode": "execute",
                    "stage": "tracking_approach_segment_limit",
                    "reason": "segmented tracking approach exhausted its segment limit",
                    "execution_attempted": any_execution_attempted,
                    "execution_sent": any_execution_sent,
                }
        else:
            code, result = self.plan()
        workflow_motion_sent = bool(
            search.get("trajectory_sent") or any_execution_sent
        )
        failure_recovery: dict[str, Any] | None = None
        if code != 0 or not result.get("success"):
            failure_recovery = self._attempt_failure_recovery_return(
                execute=execute,
                confirm_real_motion=confirm_real_motion,
                motion_sent=workflow_motion_sent,
            )
        if execute:
            result["execution_attempted"] = bool(
                result.get("execution_attempted")
                or preflight_return.get("execution_sent")
                or search.get("trajectory_sent")
                or any_execution_attempted
                or (
                    failure_recovery is not None
                    and failure_recovery.get("execution_sent")
                )
            )
            result["execution_sent"] = bool(
                result.get("execution_sent")
                or preflight_return.get("execution_sent")
                or workflow_motion_sent
                or (
                    failure_recovery is not None
                    and failure_recovery.get("execution_sent")
                )
            )
        result["tracking_replan_attempts"] = tracking_replans
        result["active_search"] = search
        result["preflight_return_to_initial_position"] = preflight_return
        if failure_recovery is not None:
            result["failure_recovery_return"] = failure_recovery
            result["return_execution_attempted"] = bool(
                failure_recovery.get("attempted")
            )
            result["return_execution_sent"] = bool(
                failure_recovery.get("execution_sent")
            )
            result["final_state"] = failure_recovery.get("final_state")
        result["dynamic_octomap_readiness"] = dynamic
        result["combined_tool_collision_geometry"] = combined_tool_geometry
        result["integrated_real_feed_water"] = contract
        if code != 0 or not result.get("success"):
            return code, result

        hold_report = {
            "success": True,
            "duration_sec": float(hold_duration_sec),
            "motion_command_sent": False,
            "completed": False,
        }
        if execute:
            if track_mouth_during_execution:
                tracking_hold = self._servo_track_during_premouth_hold(
                    float(hold_duration_sec)
                )
                hold_report.update(
                    {
                        "completed": bool(tracking_hold.get("success")),
                        "motion_command_sent": bool(
                            tracking_hold.get("servo_command_count", 0)
                        ),
                        "tracking": tracking_hold,
                    }
                )
            else:
                # Keep ROS state and the wrist RGB-D scene fresh during this
                # deliberate no-command dwell at the validated pre-mouth pose.
                self._spin_for(float(hold_duration_sec))
                hold_report["completed"] = True
        else:
            hold_report.update(
                {
                    "completed": False,
                    "plan_only": True,
                    "reason": "no dwell is performed in plan-only mode",
                }
            )
        return_code, return_result = self.return_to_initial_position(
            execute=execute,
            confirm_real_motion=confirm_real_motion,
        )
        result["pre_mouth_hold"] = hold_report
        result["return_to_initial_position"] = return_result
        result["automatic_retreat_sent"] = bool(
            return_result.get("automatic_retreat_sent")
        )
        result["return_execution_attempted"] = bool(
            return_result.get("execution_attempted")
        )
        result["return_execution_sent"] = bool(
            return_result.get("execution_sent")
        )
        tracking_hold_failed = bool(
            execute
            and track_mouth_during_execution
            and not hold_report.get("completed")
        )
        if return_code != 0 or not return_result.get("success"):
            result.update(
                {
                    "success": False,
                    "stage": "return_to_initial_position_refused",
                    "reason": (
                        return_result.get("reason")
                        or "guarded return to initial_position was refused"
                    ),
                    "final_state": "holding_pre_mouth",
                }
            )
            return 2, result
        if tracking_hold_failed:
            result.update(
                {
                    "success": False,
                    "stage": "premouth_tracking_stopped",
                    "reason": hold_report.get("tracking", {}).get(
                        "stop_reason", "pre-mouth tracking stopped"
                    ),
                    "final_state": "initial_position" if execute else "validated",
                }
            )
            return 2, result
        result.update(
            {
                "stage": (
                    "returned_initial_position"
                    if execute
                    else "pre_mouth_and_return_target_validated"
                ),
                "final_state": (
                    "initial_position"
                    if execute
                    else "pre_mouth_and_return_target_validated"
                ),
                "reason": None,
            }
        )
        return 0, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--validate-initial-position",
        action="store_true",
        help=(
            "No-motion validation of the fixed return configuration, FK, "
            "state validity, attached tool geometry, human objects, and OctoMap."
        ),
    )
    mode.add_argument(
        "--diagnose-frozen-mouth",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "No-motion diagnostic using a recorded base_link mouth point; "
            "rebuilds only the dynamic OctoMap from stationary frames."
        ),
    )
    parser.add_argument("--confirm-real-motion", action="store_true")
    parser.add_argument("--allow-validated-camera-ray-execute", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument(
        "--track-mouth-during-execution",
        action="store_true",
        help=(
            "Opt in to bounded mouth-drift cancel/replan during MoveIt motion "
            "and relative Servo corrections during the pre-mouth hold."
        ),
    )
    parser.add_argument(
        "--continuous-mouth-tracking",
        action="store_true",
        help=(
            "Use the isolated continuous MoveIt Servo approach/hold mode; "
            "the existing one-shot and segmented modes remain unchanged."
        ),
    )
    parser.add_argument(
        "--use-octomap",
        action="store_true",
        help=(
            "Enable the experimental dynamic OctoMap layer for continuous mode; "
            "disabled by default while fixed PlanningScene objects remain active."
        ),
    )
    parser.add_argument("--target-selection", choices=("left", "center", "right"), default="center")
    parser.add_argument("--mouth-sample-seconds", type=float, default=DEFAULT_MOUTH_SAMPLE_SECONDS)
    parser.add_argument(
        "--trajectory-velocity-scaling",
        type=float,
        default=DEFAULT_TRAJECTORY_VELOCITY_SCALING,
    )
    parser.add_argument(
        "--trajectory-acceleration-scaling",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=DEFAULT_PREMOUTH_HOLD_SEC,
        help=(
            "Motionless pre-mouth dwell before the guarded return "
            f"({MIN_PREMOUTH_HOLD_SEC:.0f}-{MAX_PREMOUTH_HOLD_SEC:.0f} seconds)."
        ),
    )
    parser.add_argument("--report-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init()
    node = RealIntegratedFeedWater(
        target_selection=args.target_selection,
        mouth_sample_seconds=args.mouth_sample_seconds,
        trajectory_velocity_scaling=args.trajectory_velocity_scaling,
        trajectory_acceleration_scaling=args.trajectory_acceleration_scaling,
    )
    try:
        if args.validate_initial_position:
            code, result = node.return_to_initial_position(
                execute=False,
                confirm_real_motion=False,
            )
            result["execution_disabled"] = True
            result["diagnostic"] = "validate_initial_position"
        elif args.diagnose_frozen_mouth is not None:
            dynamic = node.dynamic_readiness(execution_mode=None)
            if dynamic.get("success"):
                code, result = node.diagnose_frozen_mouth_static_scene(
                    list(args.diagnose_frozen_mouth),
                    rebuild_dynamic_octomap=True,
                )
            else:
                code = 2
                result = {
                    "success": False,
                    "mode": "diagnose-frozen",
                    "stage": "dynamic_octomap_readiness",
                    "reason": "; ".join(dynamic.get("failures", [])),
                    "execution_sent": False,
                    "execution_disabled": True,
                }
            result["dynamic_octomap_readiness"] = dynamic
        else:
            code, result = node.run_integrated(
                execute=bool(args.execute),
                confirm_real_motion=bool(args.confirm_real_motion),
                allow_validated_camera_ray_execute=bool(
                    args.allow_validated_camera_ray_execute
                ),
                no_execute=bool(args.no_execute),
                track_mouth_during_execution=bool(
                    args.track_mouth_during_execution
                ),
                continuous_mouth_tracking=bool(args.continuous_mouth_tracking),
                use_octomap=bool(args.use_octomap),
                hold_duration_sec=float(args.hold_duration),
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    report = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    print(report, flush=True)
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(report + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
