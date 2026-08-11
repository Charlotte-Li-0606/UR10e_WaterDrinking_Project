#!/usr/bin/env python3
"""Independent no-motion direct-first/OctoMap plan for the calibrated UR10e.

This wrapper deliberately reuses the proven real pre-mouth perception,
coordinate, reach, final-orientation, and combined PlanningScene guards.  It
first checks a complete collision-checked Cartesian route.  Only when that
route is incomplete or rejected does it ask OMPL RRTConnect for a non-linear
route to the same frozen collision-free pre-mouth target.  Every route keeps
tool0 +Z aligned with base_link -Z while allowing spin about that axis.  It has
no execution mode and never creates an ExecuteTrajectory client.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy  # noqa: E402
from rcl_interfaces.srv import GetParameters  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402

from scripts.real_premouth_from_perception_plan import (  # noqa: E402
    ACTION_TIMEOUT_SEC,
    DEFAULT_FEEDING_VECTOR,
    DEFAULT_MOUTH_SAMPLE_SECONDS,
    DEFAULT_SAFE_DISTANCE_M,
    DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    DEFAULT_TRAJECTORY_VELOCITY_SCALING,
    FEEDING_VECTOR_SIGNS,
    FREE_TOOL_AXIS_SPIN_TOLERANCE_RAD,
    MAX_MOUTH_SAMPLE_SECONDS,
    MAX_TOOL_VERTICAL_TILT_RAD,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MIN_MOUTH_SAMPLE_SECONDS,
    ORIENTATION_TOLERANCE_RAD,
    PILZ_PIPELINE,
    PILZ_PLANNER,
    PREMOUTH_POLICIES,
    RealPreMouthFromPerceptionPlan,
    VERTICAL_AXIS_COMPONENT_TOLERANCE_RAD,
    _jsonable,
    _trajectory_summary,
)


OMPL_PIPELINE = "ompl"
OMPL_PLANNER = "RRTConnect"
RAW_CLOUD_TOPIC = "/wrist_rgbd/points"
FILTERED_CLOUD_TOPIC = "/wrist_rgbd/filtered_cloud"
CLOUD_MAX_AGE_SEC = 0.75
CLOUD_SETTLE_SEC = 4.0
REPLAN_ATTEMPTS = 3
REPLAN_DELAY_SEC = 0.0
CARTESIAN_ROUTE_STRATEGY = "complete_collision_checked_cartesian_path"
DIRECT_ROUTE_STRATEGY = CARTESIAN_ROUTE_STRATEGY
DETOUR_ROUTE_STRATEGY = "ompl_vertical_axis_detour_after_direct_path_rejected"
COMBINED_SCENE_DESCRIPTION = "static_collision_objects_plus_live_octomap"


class RealDynamicObstacleAvoidancePlan(RealPreMouthFromPerceptionPlan):
    """Real calibrated target pipeline with a plan-only OMPL request profile."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clouds: dict[str, dict[str, Any] | None] = {
            RAW_CLOUD_TOPIC: None,
            FILTERED_CLOUD_TOPIC: None,
        }
        self._cloud_sequences: dict[str, int] = {
            RAW_CLOUD_TOPIC: 0,
            FILTERED_CLOUD_TOPIC: 0,
        }
        self._cloud_history: dict[str, list[dict[str, Any]]] = {
            RAW_CLOUD_TOPIC: [],
            FILTERED_CLOUD_TOPIC: [],
        }
        self.create_subscription(
            PointCloud2,
            RAW_CLOUD_TOPIC,
            lambda message: self._record_cloud(RAW_CLOUD_TOPIC, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            FILTERED_CLOUD_TOPIC,
            lambda message: self._record_cloud(FILTERED_CLOUD_TOPIC, message),
            qos_profile_sensor_data,
        )
        self._move_group_parameters = self.create_client(
            GetParameters,
            "/move_group/get_parameters",
        )

    def _record_cloud(self, topic: str, message: PointCloud2) -> None:
        self._cloud_sequences[topic] += 1
        record = {
            "frame_id": str(message.header.frame_id),
            "height": int(message.height),
            "point_count": int(message.width) * int(message.height),
            "received_monotonic": time.monotonic(),
            "sequence": self._cloud_sequences[topic],
            "stamp_sec": int(message.header.stamp.sec),
            "stamp_nanosec": int(message.header.stamp.nanosec),
            "width": int(message.width),
        }
        self._clouds[topic] = record
        history = self._cloud_history[topic]
        history.append(dict(record))
        del history[:-20]

    def _cloud_status(self, topic: str) -> dict[str, Any]:
        record = self._clouds[topic]
        if record is None:
            return {
                "active": False,
                "topic": topic,
                "reason": "no PointCloud2 received",
            }
        age = time.monotonic() - float(record["received_monotonic"])
        active = bool(
            record["frame_id"]
            and int(record["point_count"]) > 0
            and age <= CLOUD_MAX_AGE_SEC
        )
        return {
            "active": active,
            "age_sec": age,
            "frame_id": record["frame_id"],
            "height": record["height"],
            "maximum_age_sec": CLOUD_MAX_AGE_SEC,
            "point_count": record["point_count"],
            "topic": topic,
            "width": record["width"],
        }

    def dynamic_readiness(self, *, execution_mode: bool | None = False) -> dict[str, Any]:
        """Require current occupancy input and the requested MoveGroup mode.

        ``False`` preserves this standalone script's original hard plan-only
        requirement. ``True`` is used only by the guarded real ``feed_water``
        state machine and requires MoveGroup execution to be enabled. ``None``
        validates a plan-only request without requiring the surrounding
        MoveGroup process to be globally execution-disabled.
        """
        deadline = time.monotonic() + CLOUD_SETTLE_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            if all(self._clouds[topic] is not None for topic in self._clouds):
                break
            rclpy.spin_once(
                self,
                timeout_sec=min(0.10, max(0.0, deadline - time.monotonic())),
            )
        raw = self._cloud_status(RAW_CLOUD_TOPIC)
        filtered = self._cloud_status(FILTERED_CLOUD_TOPIC)
        parameters: dict[str, Any] = {
            "available": False,
            "allow_trajectory_execution": None,
            "octomap_resolution_m": None,
            "sensors": [],
        }
        if self._move_group_parameters.wait_for_service(timeout_sec=2.0):
            request = GetParameters.Request()
            request.names = ["sensors", "octomap_resolution", "allow_trajectory_execution"]
            future = self._move_group_parameters.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            if response is not None and len(response.values) == 3:
                parameters = {
                    "available": True,
                    "allow_trajectory_execution": bool(response.values[2].bool_value),
                    "octomap_resolution_m": float(response.values[1].double_value),
                    "sensors": list(response.values[0].string_array_value),
                }
        failures: list[str] = []
        if not parameters["available"]:
            failures.append("MoveGroup parameters are unavailable")
        if "wrist_rgbd_pointcloud" not in parameters["sensors"]:
            failures.append("MoveGroup has not loaded wrist_rgbd_pointcloud")
        resolution = parameters["octomap_resolution_m"]
        if not isinstance(resolution, (int, float)) or not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
            failures.append("MoveGroup OctoMap resolution is unavailable or invalid")
        if execution_mode is False and parameters["allow_trajectory_execution"] is not False:
            failures.append("MoveGroup trajectory execution is not explicitly disabled")
        if execution_mode is True and parameters["allow_trajectory_execution"] is not True:
            failures.append("MoveGroup trajectory execution is not enabled for guarded feed_water")
        if not raw["active"]:
            failures.append("raw wrist PointCloud2 is missing, empty, or stale")
        if not filtered["active"]:
            failures.append("MoveIt-filtered PointCloud2 is missing, empty, or stale")
        return {
            "success": not failures,
            "failures": failures,
            "execution_mode": execution_mode,
            "move_group_parameters": parameters,
            "raw_cloud": raw,
            "filtered_cloud": filtered,
        }

    def _goal_for_target(self, target: dict[str, Any]):
        """Build an OMPL detour with a vertical-axis path and exact final pose."""
        goal = super()._goal_for_target(target)
        goal.request.pipeline_id = OMPL_PIPELINE
        goal.request.planner_id = OMPL_PLANNER
        goal.request.num_planning_attempts = REPLAN_ATTEMPTS
        goal.request.path_constraints.name = "vertical_axis_intermediate_dynamic_detour"
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = REPLAN_ATTEMPTS
        goal.planning_options.replan_delay = REPLAN_DELAY_SEC
        return goal

    def _direct_goal_for_target(self, target: dict[str, Any]):
        """Build the direct vertical-axis Pilz request without dispatch."""
        return RealPreMouthFromPerceptionPlan._goal_for_target(self, target)

    def _run_goal(self, goal: Any) -> dict[str, Any]:
        """Run one plan-only MoveGroup goal against the current combined scene."""
        self._validated_trajectory = None
        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "move_group_goal",
                "reason": "MoveGroup rejected the plan-only goal",
                "execution_sent": False,
            }
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=ACTION_TIMEOUT_SEC,
        )
        wrapped_result = result_future.result()
        if wrapped_result is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "move_group_timeout",
                "reason": (
                    "MoveGroup did not return a plan within "
                    f"{ACTION_TIMEOUT_SEC:.0f} seconds; cancel requested"
                ),
                "execution_sent": False,
            }
        result = wrapped_result.result
        success = int(result.error_code.val) == 1
        vertical_axis_validation = None
        if success:
            vertical_axis_validation = self._validate_trajectory_vertical_axis(
                result.planned_trajectory
            )
            success = bool(vertical_axis_validation.get("success"))
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
            "vertical_axis_validation": vertical_axis_validation,
            "reason": None
            if success or vertical_axis_validation is None
            else vertical_axis_validation.get("reason"),
            "execution_sent": False,
        }

    def _run_plan(self, target: dict[str, Any]) -> dict[str, Any]:
        candidates = list(getattr(self, "_route_candidate_targets", []) or [])
        if not candidates:
            candidates = [target]
        candidates.sort(key=lambda item: (
            0 if item.get("position_m") == target.get("position_m") else 1,
            -float(item.get("standoff_m") or 0.0),
            -abs(float(item.get("yaw_deg") or 0.0)),
        ))
        candidates = candidates[:12]
        route_attempts: list[dict[str, Any]] = []
        last_direct: dict[str, Any] = {}
        last_result: dict[str, Any] = {}
        def summary(value: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value.get(key)
                for key in ("success", "stage", "error_code", "error_message", "fraction", "points", "reason")
                if key in value
            }
        for candidate in candidates:
            frozen = {
                "frame_id": str(candidate["frame_id"]),
                "link_name": str(candidate["link_name"]),
                "position_m": [float(value) for value in candidate["position_m"]],
                "orientation_quat_xyzw": [float(value) for value in candidate["orientation_quat_xyzw"]],
            }
            self._frozen_dynamic_target = frozen
            direct_result = self._run_cartesian_plan(frozen)
            last_direct = direct_result
            attempt = {"standoff_m": candidate.get("standoff_m"), "yaw_deg": candidate.get("yaw_deg"), "direct": summary(direct_result)}
            if direct_result.get("success"):
                self._selected_dynamic_route_strategy = DIRECT_ROUTE_STRATEGY
                direct_result.update({
                    "route_strategy": DIRECT_ROUTE_STRATEGY,
                    "planner": "moveit_compute_cartesian_path",
                    "combined_planning_scene_checked": True,
                    "planning_scene": COMBINED_SCENE_DESCRIPTION,
                    "direct_path_accepted": True,
                    "detour_attempted": False,
                    "cartesian_path_complete": True,
                    "cartesian_collision_checking": True,
                    "same_target_replanning": True,
                    "orientation_path_constraint": True,
                    "maximum_tool_vertical_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
                    "tool_axis_spin_free": True,
                    "execution_sent": False,
                    "route_attempts": route_attempts + [attempt],
                })
                return direct_result
            result = self._run_goal(self._goal_for_target(frozen))
            attempt["ompl"] = summary(result)
            route_attempts.append(attempt)
            last_result = result
            if result.get("success"):
                result.update({
                    "route_strategy": DETOUR_ROUTE_STRATEGY,
                    "planner": f"{OMPL_PIPELINE}/{OMPL_PLANNER}",
                    "combined_planning_scene_checked": True,
                    "planning_scene": COMBINED_SCENE_DESCRIPTION,
                    "direct_path_accepted": False,
                    "direct_path_plan_result": direct_result,
                    "detour_attempted": True,
                    "cartesian_path_complete": False,
                    "cartesian_collision_checking": True,
                    "same_target_replanning": True,
                    "orientation_path_constraint": True,
                    "maximum_tool_vertical_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
                    "tool_axis_spin_free": True,
                    "final_goal_orientation_constraint": True,
                    "final_goal_orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
                    "intermediate_flange_orientation_unconstrained": False,
                    "execution_sent": False,
                    "route_attempts": route_attempts,
                })
                return result

        result = dict(last_result or last_direct)
        result.update(
            {
                "route_strategy": None,
                "planner": f"{OMPL_PIPELINE}/{OMPL_PLANNER}",
                "combined_planning_scene_checked": True,
                "planning_scene": COMBINED_SCENE_DESCRIPTION,
                "direct_path_accepted": False,
                "direct_path_plan_result": last_direct,
                "detour_attempted": True,
                "cartesian_path_complete": False,
                "cartesian_collision_checking": True,
                "same_target_replanning": True,
                "maximum_replan_attempts": REPLAN_ATTEMPTS,
                "replan_delay_sec": REPLAN_DELAY_SEC,
                "wait_for_clear": False,
                "route_attempts": route_attempts,
                "orientation_path_constraint": True,
                "maximum_tool_vertical_tilt_rad": MAX_TOOL_VERTICAL_TILT_RAD,
                "tool_axis_spin_free": True,
                "final_goal_orientation_constraint": True,
                "final_goal_orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
                "intermediate_flange_orientation_unconstrained": False,
                "execution_sent": False,
            }
        )
        if not result.get("success"):
            result["failure_diagnostic"] = {
                "classification": "DIRECT_AND_DETOUR_PLANNING_FAILED",
                "reason": (
                    "the combined static-and-dynamic PlanningScene rejected or "
                    "could not plan either the direct route or the OMPL detour "
                    "while enforcing the vertical-tool axis constraint"
                ),
                "direct_error_code": last_direct.get("error_code"),
                "direct_error_message": last_direct.get("error_message"),
                "detour_error_code": result.get("error_code"),
                "detour_error_message": result.get("error_message"),
                "obstacle_layer_attribution": "combined_scene_only",
            }
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--premouth-policy",
        choices=PREMOUTH_POLICIES,
        default="camera-ray",
        help="Reuse one existing calibrated real pre-mouth target policy.",
    )
    parser.add_argument(
        "--mouth-sample-seconds",
        type=float,
        default=DEFAULT_MOUTH_SAMPLE_SECONDS,
    )
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.mouth_sample_seconds) or not (
        MIN_MOUTH_SAMPLE_SECONDS
        <= args.mouth_sample_seconds
        <= MAX_MOUTH_SAMPLE_SECONDS
    ):
        parser.error(
            f"--mouth-sample-seconds must be within "
            f"{MIN_MOUTH_SAMPLE_SECONDS:.1f}–{MAX_MOUTH_SAMPLE_SECONDS:.1f}"
        )
    return args


def main() -> int:
    args = _parse_args()
    rclpy.init()
    node = RealDynamicObstacleAvoidancePlan(
        premouth_policy=args.premouth_policy,
        safe_distance_m=DEFAULT_SAFE_DISTANCE_M,
        maximum_plan_translation_m=MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
        feeding_vector=DEFAULT_FEEDING_VECTOR,
        feeding_vector_sign=FEEDING_VECTOR_SIGNS[0],
        target_selection="center",
        mouth_sample_seconds=args.mouth_sample_seconds,
        trajectory_velocity_scaling=DEFAULT_TRAJECTORY_VELOCITY_SCALING,
        trajectory_acceleration_scaling=DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    )
    try:
        readiness = node.dynamic_readiness()
        if readiness["success"]:
            exit_code, response = node.plan()
            if not response.get("success") and not response.get("stage"):
                plan_result = response.get("plan_result", {})
                response["stage"] = "dynamic_route_plan_only"
                response["reason"] = (
                    "neither the direct route nor the vertical-axis OMPL detour "
                    "reached the frozen real pre-mouth target"
                )
                response["planning_error_code"] = plan_result.get("error_code")
        else:
            exit_code = 2
            response = {
                "success": False,
                "mode": "dynamic_obstacle_avoidance_plan_only",
                "stage": "dynamic_octomap_readiness",
                "reason": "; ".join(readiness["failures"]),
                "execution_sent": False,
            }
        response.update(
            {
                "mode": "dynamic_obstacle_avoidance_plan_only",
                "dynamic_octomap_readiness": readiness,
                "execution_disabled": True,
                "execution_sent": False,
                "same_target_replanning": True,
                "wait_for_clear": False,
                "real_execution_supported": False,
            }
        )
        plan_result = response.get("plan_result", {})
        response.setdefault(
            "planner",
            plan_result.get("planner", f"{OMPL_PIPELINE}/{OMPL_PLANNER}"),
        )
        if isinstance(plan_result, dict) and "route_strategy" in plan_result:
            response.setdefault("route_strategy", plan_result["route_strategy"])
        report = json.dumps(_jsonable(response), indent=2, sort_keys=True)
        if args.report_file is not None:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)
        return exit_code
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
