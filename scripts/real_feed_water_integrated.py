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
from moveit_msgs.action import ExecuteTrajectory  # noqa: E402
from ur_dashboard_msgs.msg import RobotMode, SafetyMode  # noqa: E402

from scripts.real_dynamic_obstacle_avoidance_plan import (  # noqa: E402
    FILTERED_CLOUD_TOPIC,
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
    EXPECTED_JOINTS,
    FINAL_ORIENTATION_TOLERANCE_RAD,
    MAX_MOUTH_POSE_AGE_SEC,
    MAX_PLAN_TRANSLATION_M,
    MAX_POSE_SPREAD_M,
    MAX_PRE_EXECUTION_TARGET_DRIFT_M,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MAX_EXECUTION_SPEED_PERCENT,
    MIN_EXECUTION_SPEED_PERCENT,
    MIN_STABLE_SAMPLES,
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
SEARCH_BACK_STEP_M = 0.030
SEARCH_LATERAL_STEP_M = 0.020
SEARCH_VERTICAL_STEP_M = 0.020
SEARCH_MAX_ACTUAL_SEGMENT_M = 0.035
SEARCH_FINAL_POSITION_TOLERANCE_M = 0.010
SEARCH_MAX_STATIONARY_JOINT_SPEED_RAD_SEC = 0.010
SEARCH_STATIONARY_SAMPLE_COUNT = 2
SEARCH_STATIONARY_TIMEOUT_SEC = 0.75
SEARCH_OFFSETS = (
    ("retreat_1", (SEARCH_BACK_STEP_M, 0.0, 0.0)),
    ("retreat_2", (2.0 * SEARCH_BACK_STEP_M, 0.0, 0.0)),
    ("retreat_3", (3.0 * SEARCH_BACK_STEP_M, 0.0, 0.0)),
    ("scan_up", (3.0 * SEARCH_BACK_STEP_M, 0.0, SEARCH_VERTICAL_STEP_M)),
    (
        "scan_upper_left",
        (3.0 * SEARCH_BACK_STEP_M, SEARCH_LATERAL_STEP_M, SEARCH_VERTICAL_STEP_M),
    ),
    ("scan_left", (3.0 * SEARCH_BACK_STEP_M, SEARCH_LATERAL_STEP_M, 0.0)),
    (
        "scan_lower_left",
        (3.0 * SEARCH_BACK_STEP_M, SEARCH_LATERAL_STEP_M, -SEARCH_VERTICAL_STEP_M),
    ),
    ("scan_down", (3.0 * SEARCH_BACK_STEP_M, 0.0, -SEARCH_VERTICAL_STEP_M)),
    (
        "scan_lower_right",
        (3.0 * SEARCH_BACK_STEP_M, -SEARCH_LATERAL_STEP_M, -SEARCH_VERTICAL_STEP_M),
    ),
    ("scan_right", (3.0 * SEARCH_BACK_STEP_M, -SEARCH_LATERAL_STEP_M, 0.0)),
    (
        "scan_upper_right",
        (3.0 * SEARCH_BACK_STEP_M, -SEARCH_LATERAL_STEP_M, SEARCH_VERTICAL_STEP_M),
    ),
    ("scan_center", (3.0 * SEARCH_BACK_STEP_M, 0.0, 0.0)),
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

    @staticmethod
    def search_waypoints(origin: list[float]) -> list[dict[str, Any]]:
        """Return the fixed absolute translation-only scan from one origin."""
        if len(origin) != 3 or not all(math.isfinite(float(value)) for value in origin):
            raise ValueError("search origin must contain three finite values")
        return [
            {
                "name": name,
                "offset_from_origin_m": [float(value) for value in offset],
                "target_tool0_position_m": _add(origin, list(offset)),
            }
            for name, offset in SEARCH_OFFSETS
        ]

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
            "trajectory_sent": False,
            "checks": snapshot,
            "search_steps": [],
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
        origin_straw = _add(
            origin_tool0,
            _rotate_tool_vector(orientation, STRAW_TIP_OFFSET_TOOL0_M),
        )
        waypoints = self.search_waypoints(origin_tool0)
        motion_deadline = deadline - SEARCH_STABILITY_RESERVE_SEC
        for waypoint in waypoints:
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
                        "trajectory_sent": bool(response["search_steps"]),
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
            segment = _norm(
                _subtract(waypoint["target_tool0_position_m"], current_position)
            )
            orientation_error = _quaternion_distance_rad(current_orientation, orientation)
            if segment > SEARCH_MAX_ACTUAL_SEGMENT_M:
                response.update(
                    {
                        "stage": "active_search_segment_guard",
                        "reason": "actual robot pose is too far from the next bounded search waypoint",
                        "segment_distance_m": segment,
                    }
                )
                return response
            if orientation_error > FINAL_ORIENTATION_TOLERANCE_RAD:
                response.update(
                    {
                        "stage": "active_search_orientation_guard",
                        "reason": "tool orientation changed during translation-only search",
                        "orientation_error_rad": orientation_error,
                    }
                )
                return response

            target_in_ur_base = self._point_in_ur_base(
                waypoint["target_tool0_position_m"],
                snapshot["ur_base_tf"],
            )
            radius = _norm(target_in_ur_base)
            if radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
                response.update(
                    {
                        "stage": "active_search_reach_guard",
                        "reason": "search waypoint exceeds the UR10e nominal reach envelope",
                        "target_tool0_radius_from_ur_base_m": radius,
                    }
                )
                return response
            target = {
                "frame_id": BASE_FRAME,
                "link_name": TOOL_FRAME,
                "position_m": waypoint["target_tool0_position_m"],
                "orientation_quat_xyzw": orientation,
            }
            plan, trajectory = self._search_plan(
                target,
                deadline=motion_deadline,
                stationary_verified=bool(
                    stationary is not None and stationary.get("success")
                ),
            )
            step = {
                **waypoint,
                "segment_distance_m": segment,
                "orientation_error_rad": orientation_error,
                "target_tool0_radius_from_ur_base_m": radius,
                "plan_result": plan,
                "execution_result": None,
                "stationary_joint_state": stationary,
            }
            if not plan.get("success") or trajectory is None:
                response["search_steps"].append(step)
                response.update(
                    {
                        "stage": "active_search_plan",
                        "reason": "MoveIt could not plan the next bounded search waypoint",
                    }
                )
                return response
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
            response["trajectory_sent"] = bool(execution.get("execution_attempted"))
            if not execution.get("success"):
                response.update(
                    {
                        "stage": "active_search_execution",
                        "reason": "bounded search segment did not complete safely",
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

        stable = self._wait_for_selected_stability(started, deadline)
        response.update(
            {
                "success": bool(stable.get("available") and stable.get("stable")),
                "stage": "mouth_found_after_search" if stable.get("stable") else "active_search_timeout",
                "selected_mouth": stable,
                "elapsed_sec": time.monotonic() - started,
                "trajectory_sent": bool(response["search_steps"]),
            }
        )
        if not response["success"]:
            response["reason"] = "selected mouth was not found within the bounded 15-second search"
        return response

    def plan(self) -> tuple[int, dict[str, Any]]:
        code, response = super().plan()
        if not response.get("success") and not response.get("stage"):
            plan_result = response.get("plan_result", {})
            response["stage"] = "dynamic_ompl_plan_only"
            response["reason"] = (
                "OMPL could not find a constrained collision-free route to the "
                "frozen real pre-mouth target"
            )
            response["planning_error_code"] = plan_result.get("error_code")
        detected = response.get("detected_mouth_pose")
        if response.get("success") and isinstance(detected, dict):
            position = detected.get("position_m")
            if isinstance(position, list) and len(position) == 3:
                self._frozen_execution_mouth_position = [float(value) for value in position]
        return code, response

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
        readiness = self.dynamic_readiness(execution_mode=True)
        if not readiness.get("success"):
            return {
                "success": False,
                "stage": "dynamic_execution_readiness",
                "reason": "; ".join(readiness.get("failures", [])),
                "dynamic_octomap_readiness": readiness,
                "execution_attempted": False,
            }
        goal = self._goal_for_target(target)
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
            }
        wrapped = result_future.result()
        if wrapped is None:
            return {
                "success": False,
                "stage": "dynamic_execution_result",
                "reason": "MoveGroup returned no dynamic execution result",
                "execution_attempted": True,
            }
        result = wrapped.result
        return {
            "success": int(result.error_code.val) == 1,
            "stage": "dynamic_move_group_plan_and_execute",
            "result_status": int(wrapped.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "execution_attempted": True,
            "controller_goal_type": "MoveGroup plan-and-execute with scene monitoring",
            "same_target_replanning": True,
            "maximum_replan_attempts": REPLAN_ATTEMPTS,
            "replan_delay_sec": REPLAN_DELAY_SEC,
            "wait_for_clear": False,
        }

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
            "dynamic_obstacle_avoidance": True,
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
    parser.add_argument("--trajectory-velocity-scaling", type=float, default=0.10)
    parser.add_argument("--trajectory-acceleration-scaling", type=float, default=0.10)
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
