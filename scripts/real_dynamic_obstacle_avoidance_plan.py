#!/usr/bin/env python3
"""Independent no-motion OMPL/OctoMap plan for the calibrated real UR10e.

This wrapper deliberately reuses the proven real pre-mouth perception,
coordinate, reach, orientation, and PlanningScene guards.  It changes only the
plan-only MoveGroup planner profile from Pilz LIN to OMPL RRTConnect so a
non-linear route to the same frozen 80 mm pre-mouth target can be evaluated.
It has no execution mode and never creates an ExecuteTrajectory client.
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
    DEFAULT_FEEDING_VECTOR,
    DEFAULT_MOUTH_SAMPLE_SECONDS,
    DEFAULT_SAFE_DISTANCE_M,
    DEFAULT_TRAJECTORY_ACCELERATION_SCALING,
    DEFAULT_TRAJECTORY_VELOCITY_SCALING,
    FEEDING_VECTOR_SIGNS,
    MAX_MOUTH_SAMPLE_SECONDS,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MIN_MOUTH_SAMPLE_SECONDS,
    PREMOUTH_POLICIES,
    RealPreMouthFromPerceptionPlan,
    _jsonable,
)


OMPL_PIPELINE = "ompl"
OMPL_PLANNER = "RRTConnectkConfigDefault"
RAW_CLOUD_TOPIC = "/wrist_rgbd/points"
FILTERED_CLOUD_TOPIC = "/wrist_rgbd/filtered_cloud"
CLOUD_MAX_AGE_SEC = 0.75
CLOUD_SETTLE_SEC = 4.0
REPLAN_ATTEMPTS = 3
REPLAN_DELAY_SEC = 0.0
# OMPL could not sample the real six-joint constraint manifold when all three
# path-orientation axes were fixed to the Pilz-only 0.001 rad tolerance.  Keep
# the *final* goal at that original tolerance, but permit at most 0.05 rad
# (2.86 degrees) on any tool-orientation axis along a detour.  This is a small
# bounded deviation, not a search rotation or an arbitrary wrist pose.
MAX_PATH_ORIENTATION_DEVIATION_RAD = 0.05


class RealDynamicObstacleAvoidancePlan(RealPreMouthFromPerceptionPlan):
    """Real calibrated target pipeline with a plan-only OMPL request profile."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clouds: dict[str, dict[str, Any] | None] = {
            RAW_CLOUD_TOPIC: None,
            FILTERED_CLOUD_TOPIC: None,
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
        self._clouds[topic] = {
            "frame_id": str(message.header.frame_id),
            "height": int(message.height),
            "point_count": int(message.width) * int(message.height),
            "received_monotonic": time.monotonic(),
            "width": int(message.width),
        }

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
        """Reuse the real constraints but choose the bounded OMPL profile."""
        goal = super()._goal_for_target(target)
        goal.request.pipeline_id = OMPL_PIPELINE
        goal.request.planner_id = OMPL_PLANNER
        goal.request.num_planning_attempts = REPLAN_ATTEMPTS
        path_orientation = goal.request.path_constraints.orientation_constraints[0]
        path_orientation.absolute_x_axis_tolerance = MAX_PATH_ORIENTATION_DEVIATION_RAD
        path_orientation.absolute_y_axis_tolerance = MAX_PATH_ORIENTATION_DEVIATION_RAD
        path_orientation.absolute_z_axis_tolerance = MAX_PATH_ORIENTATION_DEVIATION_RAD
        path_orientation.parameterization = path_orientation.ROTATION_VECTOR
        goal.request.path_constraints.name = "bounded_real_dynamic_detour_orientation"
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = REPLAN_ATTEMPTS
        goal.planning_options.replan_delay = REPLAN_DELAY_SEC
        return goal

    def _run_plan(self, target: dict[str, Any]) -> dict[str, Any]:
        self._frozen_dynamic_target = {
            "frame_id": str(target["frame_id"]),
            "link_name": str(target["link_name"]),
            "position_m": [float(value) for value in target["position_m"]],
            "orientation_quat_xyzw": [
                float(value) for value in target["orientation_quat_xyzw"]
            ],
        }
        result = super()._run_plan(target)
        result.update(
            {
                "planner": f"{OMPL_PIPELINE}/{OMPL_PLANNER}",
                "same_target_replanning": True,
                "maximum_replan_attempts": REPLAN_ATTEMPTS,
                "replan_delay_sec": REPLAN_DELAY_SEC,
                "wait_for_clear": False,
                "orientation_path_constraint": True,
                "maximum_path_orientation_deviation_rad": MAX_PATH_ORIENTATION_DEVIATION_RAD,
                "execution_sent": False,
            }
        )
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
                response["stage"] = "dynamic_ompl_plan_only"
                response["reason"] = (
                    "OMPL could not find a constrained collision-free route to the frozen real pre-mouth target"
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
                "planner": f"{OMPL_PIPELINE}/{OMPL_PLANNER}",
                "same_target_replanning": True,
                "wait_for_clear": False,
                "real_execution_supported": False,
            }
        )
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
