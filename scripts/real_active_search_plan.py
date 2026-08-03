#!/usr/bin/env python3
"""Plan the first bounded active-search step on the connected real UR10e.

This is an independent, execution-disabled real-hardware validation entry
point.  It consumes the physical robot joint/TF state and the real D435i
mouth-perception stream.  A stable visible mouth ends the search without a
plan.  When the detector explicitly reports ``no_face``, it asks MoveIt to
plan exactly one 80 mm camera-backward translation while preserving the
current tool0 orientation.  The camera direction is transformed through the
live calibrated flange-to-camera extrinsic instead of being treated as a
fixed ``base_link`` direction.  The script has no execution mode and never
creates an ExecuteTrajectory client.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy  # noqa: E402
from rcl_interfaces.srv import GetParameters  # noqa: E402

from scripts.real_premouth_from_perception_plan import (  # noqa: E402
    BASE_FRAME,
    CAMERA_OPTICAL_FRAME,
    MAX_TOOL0_RADIUS_FROM_UR_BASE_M,
    MOUTH_STATUS_TOPIC,
    STRAW_TIP_OFFSET_TOOL0_M,
    TOOL_FRAME,
    RealPreMouthFromPerceptionPlan,
    _add,
    _jsonable,
    _norm,
    _rotate_tool_vector,
)


SEARCH_BACK_DISTANCE_M = 0.080
SEARCH_BACK_OFFSET_CAMERA_OPTICAL = (0.0, 0.0, -SEARCH_BACK_DISTANCE_M)
MOUTH_SAMPLE_SECONDS = 1.0
MOVE_GROUP_PARAMETER_SERVICE = "/move_group/get_parameters"


class RealActiveSearchPlan(RealPreMouthFromPerceptionPlan):
    """Real-state, no-execution planner for one bounded search waypoint."""

    def __init__(self) -> None:
        super().__init__(
            maximum_plan_translation_m=SEARCH_BACK_DISTANCE_M,
            mouth_sample_seconds=MOUTH_SAMPLE_SECONDS,
            trajectory_velocity_scaling=0.05,
            trajectory_acceleration_scaling=0.05,
        )
        self._move_group_parameters = self.create_client(
            GetParameters,
            MOVE_GROUP_PARAMETER_SERVICE,
        )

    def _execution_disabled_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "available": False,
            "allow_trajectory_execution": None,
            "service": MOVE_GROUP_PARAMETER_SERVICE,
        }
        if not self._move_group_parameters.wait_for_service(timeout_sec=2.0):
            status["reason"] = "MoveGroup parameter service is unavailable"
            return status
        request = GetParameters.Request()
        request.names = ["allow_trajectory_execution"]
        future = self._move_group_parameters.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None or len(response.values) != 1:
            status["reason"] = "MoveGroup did not return allow_trajectory_execution"
            return status
        status.update(
            {
                "available": True,
                "allow_trajectory_execution": bool(response.values[0].bool_value),
            }
        )
        return status

    @staticmethod
    def _readiness_failures(
        snapshot: dict[str, Any],
        execution_status: dict[str, Any],
    ) -> list[str]:
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
                or "live D435i mount TF does not match the calibrated real profile"
            )
        if not snapshot.get("move_group_available"):
            failures.append("the real-UR10e MoveGroup action is unavailable")
        if not execution_status.get("available"):
            failures.append("MoveGroup execution-disable state could not be verified")
        elif execution_status.get("allow_trajectory_execution") is not False:
            failures.append("MoveGroup trajectory execution is not explicitly disabled")
        return failures

    @staticmethod
    def _is_explicit_no_face(mouth: dict[str, Any]) -> bool:
        status = mouth.get("perception_status")
        return bool(
            isinstance(status, dict)
            and status.get("detected") is False
            and status.get("reason") == "no_face"
        )

    def run(self) -> tuple[int, dict[str, Any]]:
        snapshot = self.snapshot(
            mouth_sample_sec=MOUTH_SAMPLE_SECONDS,
            inspect_controllers=True,
        )
        execution_status = self._execution_disabled_status()
        response: dict[str, Any] = {
            "mode": "real_active_search_plan_only",
            "physical_robot_state_used": True,
            "execution_disabled": True,
            "execution_sent": False,
            "trajectory_sent": False,
            "translation_only": True,
            "rotation_search_enabled": False,
            "search_direction_reference": (
                f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
            ),
            "camera_extrinsic_applied": True,
            "checks": snapshot,
            "move_group_execution": execution_status,
        }
        failures = self._readiness_failures(snapshot, execution_status)
        if failures:
            response.update(
                {
                    "success": False,
                    "stage": "real_hardware_readiness",
                    "failures": failures,
                }
            )
            return 2, response

        mouth = snapshot["mouth_pose"]
        if mouth.get("available") and mouth.get("stable"):
            response.update(
                {
                    "success": True,
                    "stage": "mouth_found_without_motion",
                    "found_without_motion": True,
                    "reason": "a stable mouth is already visible; active-search motion is unnecessary",
                    "search_steps": [],
                }
            )
            return 0, response
        if not self._is_explicit_no_face(mouth):
            response.update(
                {
                    "success": False,
                    "stage": "perception_gate",
                    "reason": (
                        "the mouth stream is unavailable or unstable but does not explicitly "
                        f"report no_face on {MOUTH_STATUS_TOPIC}; withholding the search plan"
                    ),
                }
            )
            return 2, response

        tool0 = snapshot["tool0_pose"]
        current_tool0 = [float(value) for value in tool0["position_m"]]
        orientation = [float(value) for value in tool0["orientation_quat_xyzw"]]
        camera_orientation = [
            float(value)
            for value in snapshot["camera_tf"]["orientation_quat_xyzw"]
        ]
        offset = _rotate_tool_vector(
            camera_orientation,
            SEARCH_BACK_OFFSET_CAMERA_OPTICAL,
        )
        inverse_tool_orientation = [
            -orientation[0],
            -orientation[1],
            -orientation[2],
            orientation[3],
        ]
        offset_initial_tool0 = _rotate_tool_vector(
            inverse_tool_orientation,
            offset,
        )
        target_tool0 = _add(current_tool0, offset)
        current_straw = _add(
            current_tool0,
            _rotate_tool_vector(orientation, STRAW_TIP_OFFSET_TOOL0_M),
        )
        target_straw = _add(current_straw, offset)
        target_in_ur_base = self._point_in_ur_base(
            target_tool0,
            snapshot["ur_base_tf"],
        )
        radius = _norm(target_in_ur_base)
        if not math.isfinite(radius) or radius > MAX_TOOL0_RADIUS_FROM_UR_BASE_M:
            response.update(
                {
                    "success": False,
                    "stage": "gross_reach_guard",
                    "reason": "first search waypoint exceeds the UR10e nominal reach envelope",
                    "target_tool0_radius_from_ur_base_m": radius,
                }
            )
            return 2, response

        target = {
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            "position_m": target_tool0,
            "orientation_quat_xyzw": orientation,
        }
        plan = self._run_plan(target)
        response.update(
            {
                "success": bool(plan.get("success")),
                "stage": "first_search_waypoint_plan",
                "reason": None if plan.get("success") else "MoveIt could not plan the first real search waypoint",
                "found_without_motion": False,
                "search_origin_straw_tip": {
                    "frame_id": BASE_FRAME,
                    "position_m": current_straw,
                },
                "next_search_waypoint": {
                    "name": "backward_wide",
                    "direction_reference": (
                        f"{CAMERA_OPTICAL_FRAME} frozen at initial {TOOL_FRAME} pose"
                    ),
                    "camera_extrinsic_applied": True,
                    "offset_camera_optical_m": list(
                        SEARCH_BACK_OFFSET_CAMERA_OPTICAL
                    ),
                    "offset_initial_tool0_m": offset_initial_tool0,
                    "offset_from_origin_m": offset,
                    "target_straw_tip_position_m": target_straw,
                    "target_tool0_pose": target,
                    "segment_distance_m": _norm(offset),
                    "orientation_preserved": True,
                },
                "target_tool0_radius_from_ur_base_m": radius,
                "plan_result": plan,
                "search_steps": [],
            }
        )
        return (0 if response["success"] else 2), response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Optional path for the JSON plan-only report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init()
    node = RealActiveSearchPlan()
    try:
        code, result = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    serialized = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    print(serialized)
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(serialized + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
