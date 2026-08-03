#!/usr/bin/env python3
"""Guarded real-UR10e feed_water with search, selection, and replanning.

The workflow remains one high-level operation.  It uses the physical D435i
multi-mouth stream to retain the selected person's 3D identity, performs a
bounded translation-only search only when that target is absent, freezes the
recovered 80 mm pre-mouth target, and asks MoveGroup/OctoMap to find and
monitor an alternate route to that same coordinate.

Plan mode never creates an execution request.  Execute mode retains the
existing environment, confirmation, controller, External Control, safety,
robot-mode, speed, calibration, identity, reach, collision, and final-pose
gates.  Search is limited to predefined offsets and never rotates the tool.
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
from geometry_msgs.msg import PoseStamped  # noqa: E402
from moveit_msgs.action import ExecuteTrajectory  # noqa: E402
from moveit_msgs.srv import GetPositionIK  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402

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
    FINAL_ORIENTATION_TOLERANCE_RAD,
    GROUP_NAME,
    MAX_MOUTH_POSE_AGE_SEC,
    MAX_PLAN_TRANSLATION_M,
    MAX_POSE_SPREAD_M,
    MAX_PRE_EXECUTION_TARGET_DRIFT_M,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MAX_EXECUTION_SPEED_PERCENT,
    MIN_EXECUTION_SPEED_PERCENT,
    MIN_STABLE_SAMPLES,
    PILZ_PIPELINE,
    PILZ_PLANNER,
    STRAW_TIP_OFFSET_TOOL0_M,
    TOOL_FRAME,
    RealPreMouthFromPerceptionPlan,
    _add,
    _jsonable,
    _norm,
    _quaternion_distance_rad,
    _rotate_tool_vector,
    _subtract,
    _trajectory_summary,
)


SEARCH_MAX_TIME_SEC = 15.0
SEARCH_STABILITY_RESERVE_SEC = 1.25
SEARCH_BACK_DISTANCE_M = 0.040
SEARCH_LATERAL_DISTANCE_M = 0.050
SEARCH_VERTICAL_DISTANCE_M = 0.050
SEARCH_MAX_ACTUAL_SEGMENT_M = 0.110
SEARCH_FINAL_POSITION_TOLERANCE_M = 0.010
SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC = 0.010
SEARCH_STATIONARY_SAMPLE_COUNT = 2
SEARCH_STATIONARY_TIMEOUT_SEC = 0.75
SEARCH_BACK_CANDIDATE_DISTANCES_M = (0.040, 0.030, 0.020)
SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M = (0.050, 0.040, 0.030, 0.020)
SEARCH_OFFSETS_CAMERA_OPTICAL = (
    ("backward_wide", (0.0, 0.0, -SEARCH_BACK_DISTANCE_M)),
    (
        "scan_left",
        (-SEARCH_LATERAL_DISTANCE_M, 0.0, -SEARCH_BACK_DISTANCE_M),
    ),
    (
        "scan_right",
        (SEARCH_LATERAL_DISTANCE_M, 0.0, -SEARCH_BACK_DISTANCE_M),
    ),
    (
        "scan_up",
        (0.0, -SEARCH_VERTICAL_DISTANCE_M, -SEARCH_BACK_DISTANCE_M),
    ),
    (
        "scan_down",
        (0.0, SEARCH_VERTICAL_DISTANCE_M, -SEARCH_BACK_DISTANCE_M),
    ),
)


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
        self._search_ik_client = self.create_client(GetPositionIK, "/compute_ik")

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
                }
            )
        return waypoints

    @staticmethod
    def search_waypoints(
        origin: list[float],
        tool_orientation_xyzw: list[float],
        camera_orientation_xyzw: list[float],
    ) -> list[dict[str, Any]]:
        """Return camera-corrected directions frozen at the initial flange pose.

        Search semantics follow the camera view rigidly attached to ``tool0``:
        optical -Z moves backward for a wider frame, optical -/+X scans image
        left/right, and optical -/+Y scans image up/down.  Transforming those
        vectors with the live camera orientation applies the calibrated camera
        extrinsic instead of incorrectly adding fixed ``base_link`` offsets.
        """
        return RealIntegratedFeedWater._search_waypoints_from_offsets(
            origin,
            tool_orientation_xyzw,
            camera_orientation_xyzw,
            SEARCH_OFFSETS_CAMERA_OPTICAL,
        )

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
        else:
            axes = {
                "scan_left": (-1.0, 0.0),
                "scan_right": (1.0, 0.0),
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

    def _search_plan(
        self,
        target: dict[str, Any],
        *,
        deadline: float,
        stationary_verified: bool = False,
    ) -> tuple[dict[str, Any], Any | None]:
        """Plan one Pilz-LIN search segment while preserving orientation."""
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return {
                "success": False,
                "stage": "search_plan_deadline",
                "reason": "search deadline expired before planning the next segment",
                "execution_sent": False,
            }, None
        goal = RealPreMouthFromPerceptionPlan._goal_for_target(self, target)
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
            timeout_sec=max(0.0, min(3.0, remaining)),
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
        trajectory = result.planned_trajectory if success else None
        return {
            "success": success,
            "stage": "search_plan_only",
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "execution_sent": False,
        }, trajectory

    def _target_ik_diagnostic(
        self,
        target: dict[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        """Classify a generic MoveGroup search-plan failure without motion."""
        diagnostic: dict[str, Any] = {
            "classification": "IK_DIAGNOSTIC_UNAVAILABLE",
            "reason": "MoveIt IK diagnostic did not complete",
            "target": target,
            "checks": {},
        }
        if self.latest_joint_state is None:
            diagnostic["reason"] = "fresh joint state is unavailable for IK diagnosis"
            return diagnostic
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not self._search_ik_client.wait_for_service(
            timeout_sec=min(0.25, remaining)
        ):
            diagnostic["reason"] = "/compute_ik is unavailable within the search deadline"
            return diagnostic

        def solve(*, avoid_collisions: bool) -> dict[str, Any]:
            request = GetPositionIK.Request()
            request.ik_request.group_name = GROUP_NAME
            request.ik_request.ik_link_name = TOOL_FRAME
            request.ik_request.pose_stamped = PoseStamped()
            request.ik_request.pose_stamped.header.frame_id = BASE_FRAME
            pose = request.ik_request.pose_stamped.pose
            pose.position.x, pose.position.y, pose.position.z = target["position_m"]
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = target["orientation_quat_xyzw"]
            request.ik_request.avoid_collisions = avoid_collisions
            request.ik_request.timeout.nanosec = 250_000_000
            request.ik_request.robot_state.joint_state = self.latest_joint_state
            request.ik_request.robot_state.is_diff = False
            future = self._search_ik_client.call_async(request)
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=max(0.0, min(0.5, deadline - time.monotonic())),
            )
            response = future.result()
            if response is None:
                return {"response_received": False, "success": False}
            code = int(response.error_code.val)
            return {
                "response_received": True,
                "success": code == 1,
                "error_code": code,
            }

        collision_disabled = solve(avoid_collisions=False)
        diagnostic["checks"]["collision_disabled"] = collision_disabled
        if not collision_disabled.get("response_received"):
            diagnostic["reason"] = (
                "/compute_ik did not return the collision-disabled diagnostic "
                "before the bounded search deadline"
            )
            return diagnostic
        if not collision_disabled.get("success"):
            diagnostic.update(
                {
                    "classification": "NO_IK_SOLUTION",
                    "reason": (
                        "MoveIt found no IK solution for the fixed-orientation "
                        "tool0 search target, even with collision checking disabled"
                    ),
                }
            )
            return diagnostic

        collision_enabled = solve(avoid_collisions=True)
        diagnostic["checks"]["collision_enabled"] = collision_enabled
        if not collision_enabled.get("response_received"):
            diagnostic["reason"] = (
                "/compute_ik did not return the collision-enabled diagnostic "
                "before the bounded search deadline"
            )
            return diagnostic
        if not collision_enabled.get("success"):
            diagnostic.update(
                {
                    "classification": "GOAL_IN_COLLISION",
                    "reason": (
                        "the tool0 search target has IK but no collision-free IK "
                        "solution in the current PlanningScene"
                    ),
                }
            )
            return diagnostic

        diagnostic.update(
            {
                "classification": "PILZ_CARTESIAN_PATH_FAILED",
                "reason": (
                    "the target has collision-free IK, but Pilz could not generate "
                    "the fixed-orientation Cartesian path; an intermediate IK or "
                    "joint dynamic limit was rejected"
                ),
            }
        )
        return diagnostic

    def _wait_for_search_stationary(self, deadline: float) -> dict[str, Any]:
        """Require fresh stationary joint samples before each Pilz LIN plan.

        The UR controller can report one final nonzero velocity sample after a
        trajectory result succeeds. Pilz rejects any nonzero start velocity,
        so wait for two new stationary samples before representing the
        verified start state with exact zero velocities in the planning goal.
        """
        settle_deadline = min(
            deadline,
            time.monotonic() + SEARCH_STATIONARY_TIMEOUT_SEC,
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
            "maximum_joint_speed_rad_sec": latest_max_speed,
            "maximum_allowed_joint_speed_rad_sec": (
                SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC
            ),
            "stationary_sample_count": consecutive,
        }

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
            if self._candidate_visible():
                self._cancel_goal(handle)
                return {
                    "success": True,
                    "stage": "search_cancelled_for_candidate",
                    "candidate_detected": True,
                    "execution_attempted": True,
                    "trajectory_cancel_requested": True,
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
        return {
            "success": int(result.error_code.val) == 1,
            "stage": "search_execute",
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "result_status": int(wrapped.status),
            "candidate_detected": self._candidate_visible(),
            "execution_attempted": True,
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
            "translation_only": True,
            "rotation_search_enabled": False,
            "search_direction_reference": (
                f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
            ),
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
                "directional": list(SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M),
                "skip_if_all_bounded_candidates_fail": True,
            },
            "trajectory_sent": False,
            "checks": snapshot,
            "search_steps": [],
            "skipped_search_waypoints": [],
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
                response["reason"] = "visible mouth did not become a stable selected identity; search motion was withheld"
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
            orientation_error = _quaternion_distance_rad(current_orientation, orientation)
            if orientation_error > FINAL_ORIENTATION_TOLERANCE_RAD:
                response.update(
                    {
                        "stage": "active_search_orientation_guard",
                        "reason": "tool orientation changed during translation-only search",
                        "orientation_error_rad": orientation_error,
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
                segment = _norm(
                    _subtract(variant["target_tool0_position_m"], current_position)
                )
                target_in_ur_base = self._point_in_ur_base(
                    variant["target_tool0_position_m"],
                    snapshot["ur_base_tf"],
                )
                radius = _norm(target_in_ur_base)
                attempt: dict[str, Any] = {
                    **variant,
                    "segment_distance_m": segment,
                    "orientation_error_rad": orientation_error,
                    "target_tool0_radius_from_ur_base_m": radius,
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
                    "orientation_quat_xyzw": orientation,
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
                            "classification": "SEARCH_EXECUTION_FAILED",
                            "reason": detail,
                            "execution_stage": execution.get("stage"),
                            "error_code": execution.get("error_code"),
                            "error_message": execution.get("error_message"),
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
                    response["reason"] = "mouth candidate appeared but did not become a stable selected identity"
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
                "the direct fixed-orientation path and the constrained OMPL "
                "detour both failed for the frozen real pre-mouth target"
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
        route_strategy = getattr(self, "_selected_dynamic_route_strategy", None)
        if route_strategy not in (DIRECT_ROUTE_STRATEGY, DETOUR_ROUTE_STRATEGY):
            return {
                "success": False,
                "stage": "dynamic_route_strategy",
                "reason": "no validated direct or detour route strategy is available",
                "route_strategy": route_strategy,
                "execution_attempted": False,
            }
        planner = (
            f"{PILZ_PIPELINE}/{PILZ_PLANNER}"
            if route_strategy == DIRECT_ROUTE_STRATEGY
            else f"{OMPL_PIPELINE}/{OMPL_PLANNER}"
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
        deadline = time.monotonic() + ACTION_TIMEOUT_SEC
        cancel_reason: str | None = None
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            raw = self._cloud_status(RAW_CLOUD_TOPIC)
            filtered = self._cloud_status(FILTERED_CLOUD_TOPIC)
            if not raw.get("active") or not filtered.get("active"):
                cancel_reason = "raw or filtered wrist point cloud became stale during execution"
                break
            perception = self.target_tracker.current_state(max_age_sec=MAX_MOUTH_POSE_AGE_SEC)
            if perception.get("identity_unsafe"):
                cancel_reason = "selected mouth identity became ambiguous during execution"
                break
            if perception.get("available") and self._frozen_execution_mouth_position is not None:
                selected = perception.get("selected_position_m")
                if isinstance(selected, list) and len(selected) == 3:
                    drift = _norm(_subtract(selected, self._frozen_execution_mouth_position))
                    if drift > MAX_PRE_EXECUTION_TARGET_DRIFT_M:
                        cancel_reason = (
                            f"selected mouth moved {drift:.4f} m during execution, above the "
                            f"{MAX_PRE_EXECUTION_TARGET_DRIFT_M:.4f} m limit"
                        )
                        break
        if cancel_reason is not None or not result_future.done():
            self._cancel_goal(handle)
            return {
                "success": False,
                "stage": "dynamic_execution_cancelled",
                "reason": cancel_reason or "MoveGroup execution exceeded the bounded timeout",
                "execution_attempted": True,
                "cancel_requested": True,
                "same_target_replanning": True,
                "route_strategy": route_strategy,
                "planner": planner,
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
            "controller_goal_type": "MoveGroup plan-and-execute with scene monitoring",
            "route_strategy": route_strategy,
            "planner": planner,
            "same_target_replanning": True,
            "maximum_replan_attempts": REPLAN_ATTEMPTS,
            "replan_delay_sec": REPLAN_DELAY_SEC,
            "wait_for_clear": False,
        }
        if not success:
            detail = result.error_code.message or (
                f"MoveGroup plan-and-execute failed with error code "
                f"{int(result.error_code.val)}"
            )
            response["reason"] = detail
            response["failure_diagnostic"] = {
                "classification": "DYNAMIC_PLAN_OR_EXECUTION_FAILED",
                "reason": detail,
                "error_code": int(result.error_code.val),
                "error_message": result.error_code.message,
            }
        return response

    def run_integrated(
        self,
        *,
        execute: bool,
        confirm_real_motion: bool,
        allow_validated_camera_ray_execute: bool,
        no_execute: bool,
    ) -> tuple[int, dict[str, Any]]:
        contract = {
            "multi_target_identity_lock": True,
            "target_selection": self.target_selection,
            "active_search": True,
            "translation_only_search": True,
            "rotation_search_enabled": False,
            "active_search_direction_reference": (
                f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
            ),
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
                "directional": list(SEARCH_DIRECTIONAL_CANDIDATE_DISTANCES_M),
                "skip_unreachable_direction": True,
            },
            "dynamic_obstacle_avoidance": True,
            "direct_clear_path_first": True,
            "constrained_detour_only_after_direct_rejection": True,
            "combined_static_and_dynamic_scene_checks": True,
            "same_target_replanning": True,
            "wait_for_clear": False,
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
        dynamic = self.dynamic_readiness(execution_mode=True if execute else None)
        if not dynamic.get("success"):
            return 2, {
                "success": False,
                "mode": "execute" if execute else "plan",
                "stage": "dynamic_octomap_readiness",
                "reason": "; ".join(dynamic.get("failures", [])),
                "dynamic_octomap_readiness": dynamic,
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
        if execute:
            code, result = super().execute(
                confirm_real_motion=confirm_real_motion,
                allow_validated_camera_ray_execute=allow_validated_camera_ray_execute,
                allow_validated_feeding_vector_execute=False,
                allow_validated_tcp_forward_execute=False,
                no_execute=False,
            )
        else:
            code, result = self.plan()
        result["active_search"] = search
        result["dynamic_octomap_readiness"] = dynamic
        result["integrated_real_feed_water"] = contract
        return code, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-real-motion", action="store_true")
    parser.add_argument("--allow-validated-camera-ray-execute", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
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
        code, result = node.run_integrated(
            execute=bool(args.execute),
            confirm_real_motion=bool(args.confirm_real_motion),
            allow_validated_camera_ray_execute=bool(
                args.allow_validated_camera_ray_execute
            ),
            no_execute=bool(args.no_execute),
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
