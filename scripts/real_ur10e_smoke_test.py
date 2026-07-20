#!/usr/bin/env python3
"""Conservative first-motion smoke test for the physical UR10e.

Modes:
  check   Read joint state, TF, controller, and action-server availability.
  plan    Ask MoveIt to plan a fixed 2 cm base-Z tool0 translation only.
  execute Execute that same plan only after all explicit safety gates pass.

The script never starts the UR driver or MoveIt.  It creates no arbitrary
joint target, keeps the current tool0 orientation as a strict path constraint,
and never calls the feeding, LLM, perception, or OpenClaw paths.
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

# This smoke test is always for the physical backend.  The value is set before
# importing/constructing the canonical SDK, so it cannot accidentally select
# the Gazebo endpoint.
os.environ["UR10E_BACKEND"] = "real"

import rclpy  # noqa: E402
import rclpy.time  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from controller_manager_msgs.srv import ListControllers  # noqa: E402
from moveit_msgs.action import MoveGroup  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

from robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk import UR10eRobotEnv  # noqa: E402


BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
SCALED_CONTROLLER = "scaled_joint_trajectory_controller"
SCALED_ACTION = f"/{SCALED_CONTROLLER}/follow_joint_trajectory"
MOVE_ACTION = "/move_action"
EXPECTED_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Fixed by design. This script deliberately exposes no CLI pose or offset.
SMOKE_OFFSET_BASE_Z_M = 0.02
STRICT_ORIENTATION_TOLERANCE_RAD = 0.001


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float):
        return round(value, 9)
    return value


class _ReadOnlyProbe(Node):
    """A ROS node used only for subscriptions, TF lookup, and service reads."""

    def __init__(self) -> None:
        super().__init__("real_ur10e_smoke_test_probe")
        self.latest_joint_state: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.scaled_trajectory = ActionClient(self, FollowJointTrajectory, SCALED_ACTION)

    def _joint_callback(self, message: JointState) -> None:
        self.latest_joint_state = message


def _joint_state_summary(message: JointState | None) -> dict[str, Any]:
    if message is None:
        return {"received": False, "reason": "no /joint_states message received"}
    index = {name: number for number, name in enumerate(message.name)}
    positions = {
        name: float(message.position[index[name]])
        for name in EXPECTED_JOINTS
        if name in index and index[name] < len(message.position)
    }
    velocities = {
        name: float(message.velocity[index[name]])
        for name in EXPECTED_JOINTS
        if name in index and index[name] < len(message.velocity)
    }
    return {
        "received": True,
        "expected_joint_count": len(EXPECTED_JOINTS),
        "received_expected_joint_count": len(positions),
        "positions_rad": positions,
        "velocities_rad_s": velocities,
        "frame_id": message.header.frame_id,
    }


def _read_only_checks(timeout_sec: float = 5.0) -> dict[str, Any]:
    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    node = _ReadOnlyProbe()
    try:
        deadline = time.monotonic() + timeout_sec
        while node.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        joint_state = _joint_state_summary(node.latest_joint_state)
        tf_result: dict[str, Any]
        try:
            transform = node.tf_buffer.lookup_transform(
                BASE_FRAME,
                TOOL_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            tf_result = {
                "available": True,
                "parent_frame": BASE_FRAME,
                "child_frame": TOOL_FRAME,
                "position_m": [translation.x, translation.y, translation.z],
                "orientation_quat_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
            }
        except Exception as exc:
            tf_result = {"available": False, "reason": str(exc)}

        controller_result: dict[str, Any]
        if not node.controllers.wait_for_service(timeout_sec=1.0):
            controller_result = {"available": False, "reason": "/controller_manager/list_controllers is unavailable"}
        else:
            future = node.controllers.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
            response = future.result()
            if response is None:
                controller_result = {"available": False, "reason": "controller manager returned no response"}
            else:
                active = sorted(controller.name for controller in response.controller if controller.state == "active")
                controller_result = {
                    "available": True,
                    "active": active,
                    "joint_state_broadcaster_active": "joint_state_broadcaster" in active,
                    "scaled_joint_trajectory_controller_active": SCALED_CONTROLLER in active,
                }

        action_servers = {
            "move_group_available": node.move_group.wait_for_server(timeout_sec=1.0),
            "scaled_follow_joint_trajectory_available": node.scaled_trajectory.wait_for_server(timeout_sec=1.0),
        }
        base_checks_ok = bool(
            joint_state.get("received")
            and joint_state.get("received_expected_joint_count") == len(EXPECTED_JOINTS)
            and tf_result.get("available")
            and controller_result.get("joint_state_broadcaster_active")
            and controller_result.get("scaled_joint_trajectory_controller_active")
        )
        return {
            "success": base_checks_ok,
            "backend": "real",
            "real_execution_environment": os.environ.get("UR10E_ALLOW_REAL_EXECUTION", "0"),
            "joint_state": joint_state,
            "tool0_pose": tf_result,
            "controller_status": controller_result,
            "action_servers": action_servers,
            "note": "Read-only checks only; no action goal or controller command was sent.",
        }
    finally:
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _build_target(current_pose: dict[str, Any]) -> dict[str, Any]:
    position = current_pose.get("position")
    orientation = current_pose.get("orientation_quat")
    if not isinstance(position, list) or len(position) != 3:
        raise RuntimeError("current tool0 pose does not contain a three-value position")
    if not isinstance(orientation, list) or len(orientation) != 4:
        raise RuntimeError("current tool0 pose does not contain a quaternion")
    target_position = [float(position[0]), float(position[1]), float(position[2]) + SMOKE_OFFSET_BASE_Z_M]
    if not all(math.isfinite(value) for value in target_position + [float(value) for value in orientation]):
        raise RuntimeError("current tool0 pose contains non-finite values")
    return {
        "frame_id": BASE_FRAME,
        "position_m": target_position,
        "orientation_quat_xyzw": [float(value) for value in orientation],
        "translation_offset_m": [0.0, 0.0, SMOKE_OFFSET_BASE_Z_M],
        "orientation_preserved": True,
        "strict_orientation_tolerance_rad": STRICT_ORIENTATION_TOLERANCE_RAD,
    }


def _guard_execute(arguments: argparse.Namespace) -> str | None:
    if arguments.mode != "execute":
        return None
    if not arguments.confirm_real_motion:
        return "execution requires --confirm-real-motion"
    if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
        return "execution requires UR10E_ALLOW_REAL_EXECUTION=1"
    return None


def _plan_or_execute(arguments: argparse.Namespace, checks: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    blocked = _guard_execute(arguments)
    if blocked is not None:
        return 2, {"success": False, "mode": arguments.mode, "stage": "execution_gate", "reason": blocked, "checks": checks}
    if not checks.get("success"):
        return 2, {"success": False, "mode": arguments.mode, "stage": "read_only_checks", "reason": "required read-only checks did not pass", "checks": checks}
    if not checks.get("action_servers", {}).get("move_group_available"):
        return 2, {
            "success": False,
            "mode": arguments.mode,
            "stage": "moveit_availability",
            "reason": "/move_action is unavailable; start a reviewed MoveIt instance separately before plan-only testing",
            "checks": checks,
        }

    env: UR10eRobotEnv | None = None
    try:
        env = UR10eRobotEnv()
        current = _jsonable(env.get_end_effector_pose())
        target = _build_target(current)
        target_endpose = [*target["position_m"], *target["orientation_quat_xyzw"]]
        plan_result = _jsonable(
            env.move_to_pose(target_endpose, plan_only=True, strict_orientation=True)
        )
        response: dict[str, Any] = {
            "success": bool(plan_result.get("success")),
            "mode": arguments.mode,
            "checks": checks,
            "current_tool0_pose": current,
            "target_tool0_pose": target,
            "plan_result": plan_result,
            "execution_sent": False,
        }
        if not plan_result.get("success"):
            response["stage"] = "plan"
            response["reason"] = "MoveIt did not produce the strict-orientation 2 cm plan; no execution was sent"
            return 2, response
        if arguments.mode == "plan":
            response["note"] = "Plan-only MoveIt request completed; no trajectory was executed."
            return 0, response

        execution_result = _jsonable(
            env.move_to_pose(target_endpose, plan_only=False, strict_orientation=True)
        )
        response["execution_result"] = execution_result
        response["execution_sent"] = True
        response["success"] = bool(execution_result.get("success"))
        response["stage"] = "execute"
        response["reason"] = None if response["success"] else "MoveIt execution did not complete"
        return (0 if response["success"] else 2), response
    except Exception as exc:
        return 2, {
            "success": False,
            "mode": arguments.mode,
            "stage": "sdk_or_moveit",
            "reason": f"{exc.__class__.__name__}: {exc}",
            "checks": checks,
            "execution_sent": False,
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "plan", "execute"), default="check")
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required with --mode execute; does nothing in check/plan modes.",
    )
    arguments = parser.parse_args()
    checks = _read_only_checks()
    if arguments.mode == "check":
        print(json.dumps(_jsonable({"mode": "check", **checks}), sort_keys=True))
        return 0 if checks.get("success") else 2
    status, result = _plan_or_execute(arguments, checks)
    print(json.dumps(_jsonable(result), sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
