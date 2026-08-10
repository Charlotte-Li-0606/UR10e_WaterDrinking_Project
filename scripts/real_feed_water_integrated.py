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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy  # noqa: E402
from geometry_msgs.msg import Pose  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup  # noqa: E402
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, ServoStatus  # noqa: E402
from moveit_msgs.srv import GetStateValidity  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import Empty  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402

from robot_layer.arm_ur10e.control.motion_backend import MotionRequest  # noqa: E402
from robot_layer.arm_ur10e.control.relative_tracking import RelativeTrackingSession  # noqa: E402
from robot_layer.arm_ur10e.control.ros_servo_backend import RosServoCommandSink  # noqa: E402

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
TRACKING_MAX_REPLAN_ATTEMPTS = 2
TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC = 3.0
TRACKING_TARGET_MAX_AGE_SEC = 0.75
TRACKING_MAX_DISPLACEMENT_M = 0.06
TRACKING_MAX_LINEAR_SPEED_MPS = 0.02
TRACKING_MAX_LINEAR_ACCELERATION_MPS2 = 0.10
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
        self._state_validity_client = self.create_client(
            GetStateValidity,
            "/check_state_validity",
        )
        self._clear_octomap_client = self.create_client(
            Empty,
            "/clear_octomap",
        )
        self.create_subscription(
            ServoStatus,
            "/servo_node/status",
            self._servo_status_callback,
            10,
        )

    def _servo_status_callback(self, message: ServoStatus) -> None:
        self._latest_servo_status = message

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

    def _prepare_initial_position_target(self) -> dict[str, Any]:
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
                failures.append("fixed human head/torso/face objects are absent")
            if scene.get("human_allowed_collision_pairs"):
                failures.append("human allowed-collision entries are present")
            if not scene.get("combined_tool_collision_geometry", {}).get("success"):
                failures.append("combined camera/cup-holder/straw geometry is not verified")
            if not scene.get("octomap", {}).get("present"):
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
        combined = self._apply_combined_tool_collision_geometry()
        dynamic = self.dynamic_readiness(execution_mode=True if execute else None)
        prepared = self._prepare_initial_position_target()
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
        latest_dynamic = self.dynamic_readiness(execution_mode=True)
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
                pre_execution_failures.append(
                    "fixed human head/torso/face objects disappeared before return execution"
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
            if not latest_scene.get("octomap", {}).get("present"):
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
        before_value = before_validity.get("valid")
        after_value = after_validity.get("valid")
        report["final_state_validity_changed_after_rebuild"] = (
            isinstance(before_value, bool)
            and isinstance(after_value, bool)
            and before_value != after_value
        )
        report["success"] = bool(
            after_scene.get("available")
            and human_preserved
            and after_validity.get("available")
        )
        report["reason"] = None if report["success"] else (
            "rebuilt scene or final-state diagnostic is unavailable"
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
            clouds = {
                RAW_CLOUD_TOPIC: self._cloud_status(RAW_CLOUD_TOPIC),
                FILTERED_CLOUD_TOPIC: self._cloud_status(FILTERED_CLOUD_TOPIC),
            }
            if not all(status.get("active") for status in clouds.values()):
                late_failures.append("raw or filtered wrist point cloud became stale before search motion")
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
                "mouth_drift_confirmation": mouth_confirmation,
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
        return {
            "success": int(result.error_code.val) == 1,
            "stage": "tracked_cartesian_execute_trajectory",
            "result_status": int(wrapped.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "execution_attempted": True,
            "execution_sent": True,
            "tracking_replan_required": False,
            "mouth_drift_confirmation": mouth_confirmation,
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

    def run_integrated(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        allow_validated_camera_ray_execute: bool,
        no_execute: bool,
        track_mouth_during_execution: bool = False,
        hold_duration_sec: float = DEFAULT_PREMOUTH_HOLD_SEC,
    ) -> tuple[int, dict[str, Any]]:
        contract = {
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
            "return_target": "initial_position",
            "return_collision_checking": True,
            "return_vertical_axis_constraint": True,
            "return_wrist_3_direct_command": False,
            "pre_mouth_hold_duration_sec": float(hold_duration_sec),
            "mouth_tracking_during_execution": bool(
                track_mouth_during_execution
            ),
            "tracking_policy": (
                "cancel_and_replan_during_moveit_then_relative_servo_at_premouth"
                if track_mouth_during_execution
                else "disabled_one_shot_frozen_target"
            ),
            "tracking_maximum_replan_attempts": TRACKING_MAX_REPLAN_ATTEMPTS,
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
        dynamic = self.dynamic_readiness(execution_mode=True if execute else None)
        if not dynamic.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "dynamic_octomap_readiness",
                "reason": "; ".join(dynamic.get("failures", [])),
                "dynamic_octomap_readiness": dynamic,
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": False,
                "execution_sent": False,
                "integrated_real_feed_water": contract,
            }
        search = self.active_search(
            execute=execute,
            confirm_real_motion=confirm_real_motion,
        )
        if not search.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "active_search",
                "reason": search.get("reason") or "active search did not recover the selected mouth",
                "active_search": search,
                "dynamic_octomap_readiness": dynamic,
                "combined_tool_collision_geometry": combined_tool_geometry,
                "execution_attempted": bool(
                    any(
                        isinstance(step, dict)
                        and isinstance(step.get("execution_result"), dict)
                        and step["execution_result"].get("execution_attempted")
                        for step in search.get("search_steps", [])
                    )
                ),
                "execution_sent": bool(search.get("trajectory_sent")),
                "integrated_real_feed_water": contract,
            }
        self._integrated_tracking_enabled = bool(track_mouth_during_execution)
        tracking_replans: list[dict[str, Any]] = []
        any_execution_attempted = False
        any_execution_sent = False
        if execute:
            for attempt in range(TRACKING_MAX_REPLAN_ATTEMPTS + 1):
                if attempt > 0:
                    stationary = self._wait_for_tracking_replan_stationary()
                    tracking_replans[-1]["post_cancel_stationary_wait"] = stationary
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
                            "post_cancel_stationary_wait": stationary,
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
                tracking_replans.append(
                    {
                        "attempt": attempt + 1,
                        "success": bool(result.get("success")),
                        "stage": result.get("stage"),
                        "execution_stage": execution_result.get("stage")
                        if isinstance(execution_result, dict)
                        else None,
                        "reason": result.get("reason"),
                        "replan_required": bool(
                            isinstance(execution_result, dict)
                            and execution_result.get("tracking_replan_required")
                        ),
                        "mouth_drift_confirmation": (
                            execution_result.get("mouth_drift_confirmation")
                            if isinstance(execution_result, dict)
                            else None
                        ),
                    }
                )
                if code == 0 and result.get("success"):
                    break
                if not (
                    track_mouth_during_execution
                    and isinstance(execution_result, dict)
                    and execution_result.get("tracking_replan_required")
                    and attempt < TRACKING_MAX_REPLAN_ATTEMPTS
                ):
                    break
        else:
            code, result = self.plan()
        if execute:
            result["execution_attempted"] = bool(
                result.get("execution_attempted") or any_execution_attempted
            )
            result["execution_sent"] = bool(
                result.get("execution_sent") or any_execution_sent
            )
        result["tracking_replan_attempts"] = tracking_replans
        result["active_search"] = search
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
