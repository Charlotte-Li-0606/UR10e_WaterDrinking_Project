#!/usr/bin/env python3
"""Isolated, conservative MoveIt execution probe for the physical UR10e.

This utility deliberately does not import the feeding, LLM, OpenClaw, camera,
or perception code.  It answers one commissioning question only: can the real
UR10e execute a tiny MoveIt-managed trajectory?

Modes:
  check   Read-only ROS, controller, dashboard, TF, and action checks.
  plan    Plan the fixed tool0 probe translation only.
  execute Plan first, then execute that same fixed MoveIt target only after
          both explicit real-motion gates are supplied.
  debug   Read-only diagnosis for the common "plans but does not move" layers.

No mode sends a raw FollowJointTrajectory goal.  The sole executable request
is a MoveGroup action goal, so MoveIt owns controller selection and execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This is an isolated physical-robot utility.  Keep ROS discovery on this
# ThinkPad so a Wi-Fi simulator cannot offer a competing TF tree or action.
os.environ["UR10E_BACKEND"] = "real"
os.environ.pop("ROS_STATIC_PEERS", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["ROS_LOCALHOST_ONLY"] = "0"

import rclpy  # noqa: E402
import rclpy.time  # noqa: E402
import tf2_ros  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from controller_manager_msgs.srv import ListControllers  # noqa: E402
from geometry_msgs.msg import Pose  # noqa: E402
from moveit_msgs.action import MoveGroup  # noqa: E402
from moveit_msgs.msg import BoundingVolume, Constraints, OrientationConstraint, PositionConstraint  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter_client import AsyncParameterClient  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402
from ur_dashboard_msgs.srv import (  # noqa: E402
    GetProgramState,
    GetRobotMode,
    GetSafetyMode,
    IsInRemoteControl,
    IsProgramRunning,
)


BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
GROUP_NAME = "ur_manipulator"
MOVE_ACTION = "/move_action"
SCALED_CONTROLLER = "scaled_joint_trajectory_controller"
SCALED_ACTION = f"/{SCALED_CONTROLLER}/follow_joint_trajectory"
SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
EXPECTED_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Fixed commissioning motions.  There is intentionally no generic CLI pose,
# joint, or offset input that could widen the scope of this physical test.
DEFAULT_PROBE_OFFSET_BASE_Z_M = 0.01
DOWNWARD_2CM_PROBE_OFFSET_BASE_Z_M = -0.02
MOTION_VERIFICATION_TOLERANCE_M = 0.005
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_RAD = 0.001
MAX_SCALING = 0.02
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PLANNER = "LIN"
EXECUTION_TIMEOUT_SEC = 45.0

ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}
SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED_SAFETY_MODE",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}


def _jsonable(value: Any) -> Any:
    """Convert ROS values and Parameters into concise JSON-safe data."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and value.__class__.__module__.startswith("rclpy"):
        return _jsonable(value.value)
    if isinstance(value, float):
        return round(value, 8)
    return value


def _parameter_value_to_python(value: Any) -> Any:
    """Decode an rcl_interfaces/ParameterValue without depending on a node API."""

    parameter_type = int(getattr(value, "type", 0))
    fields = {
        1: "bool_value",
        2: "integer_value",
        3: "double_value",
        4: "string_value",
        5: "byte_array_value",
        6: "bool_array_value",
        7: "integer_array_value",
        8: "double_array_value",
        9: "string_array_value",
    }
    field = fields.get(parameter_type)
    if field is None:
        return None
    return _jsonable(getattr(value, field))


def _duration_seconds(trajectory: Any) -> float | None:
    points = getattr(getattr(trajectory, "joint_trajectory", None), "points", [])
    if not points:
        return None
    duration = points[-1].time_from_start
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _trajectory_summary(trajectory: Any) -> dict[str, Any]:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    points = list(getattr(joint_trajectory, "points", []))
    return {
        "joint_names": list(getattr(joint_trajectory, "joint_names", [])),
        "points": len(points),
        "duration_sec": _duration_seconds(trajectory),
    }


def _quaternion_distance_rad(before: list[float], after: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(before, after)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class RealMoveItExecutionProbe(Node):
    """Read-only diagnostics plus one tightly bounded MoveGroup motion path."""

    def __init__(self, probe_offset_base_z_m: float = DEFAULT_PROBE_OFFSET_BASE_Z_M) -> None:
        super().__init__("real_moveit_execution_probe")
        self.probe_offset_base_z_m = probe_offset_base_z_m
        self.latest_joint_state: JointState | None = None
        self.latest_speed_scaling: Float64 | None = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.create_subscription(Float64, SPEED_SCALING_TOPIC, self._speed_scaling_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.scaled_trajectory = ActionClient(self, FollowJointTrajectory, SCALED_ACTION)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")

        # Dashboard calls are all read-only.  The UR driver exposes these
        # names when launch_dashboard_client is enabled (the Jazzy default).
        self.dashboard_remote = self.create_client(IsInRemoteControl, "/dashboard_client/is_in_remote_control")
        self.dashboard_program_running = self.create_client(IsProgramRunning, "/dashboard_client/program_running")
        self.dashboard_robot_mode = self.create_client(GetRobotMode, "/dashboard_client/get_robot_mode")
        self.dashboard_safety_mode = self.create_client(GetSafetyMode, "/dashboard_client/get_safety_mode")
        self.dashboard_program_state = self.create_client(GetProgramState, "/dashboard_client/program_state")

    def _joint_state_callback(self, message: JointState) -> None:
        self.latest_joint_state = message

    def _speed_scaling_callback(self, message: Float64) -> None:
        self.latest_speed_scaling = message

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))

    def _graph_nodes(self) -> list[str]:
        names: list[str] = []
        for name, namespace in self.get_node_names_and_namespaces():
            if namespace == "/":
                names.append(f"/{name}")
            else:
                names.append(f"{namespace.rstrip('/')}/{name}")
        return sorted(names)

    def _joint_state_status(self) -> dict[str, Any]:
        message = self.latest_joint_state
        if message is None:
            return {"received": False, "reason": "no /joint_states message received"}
        index = {name: position for position, name in enumerate(message.name)}
        present = [name for name in EXPECTED_JOINTS if index.get(name) is not None and index[name] < len(message.position)]
        return {
            "received": True,
            "frame_id": message.header.frame_id,
            "received_expected_joint_count": len(present),
            "expected_joint_count": len(EXPECTED_JOINTS),
            "complete": len(present) == len(EXPECTED_JOINTS),
            "positions_rad": {name: float(message.position[index[name]]) for name in present},
            "velocities_rad_s": {
                name: float(message.velocity[index[name]])
                for name in present
                if index[name] < len(message.velocity)
            },
        }

    def _tool_pose(self, timeout_sec: float = 3.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    TOOL_FRAME,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.0),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                return {
                    "available": True,
                    "frame_id": BASE_FRAME,
                    "link_name": TOOL_FRAME,
                    "position_m": [translation.x, translation.y, translation.z],
                    "orientation_quat_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
                }
            except Exception as exc:  # TF error text is useful diagnostic context.
                last_error = exc
        return {"available": False, "reason": str(last_error) if last_error else "TF lookup timed out"}

    def _controller_status(self) -> dict[str, Any]:
        if not self.controllers.wait_for_service(timeout_sec=1.0):
            return {"available": False, "reason": "/controller_manager/list_controllers is unavailable"}
        future = self.controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        response = future.result()
        if response is None:
            return {"available": False, "reason": "controller manager returned no response"}
        controllers = [
            {"name": controller.name, "state": controller.state, "type": controller.type}
            for controller in response.controller
        ]
        active = sorted(item["name"] for item in controllers if item["state"] == "active")
        return {
            "available": True,
            "active": active,
            "controllers": controllers,
            "joint_state_broadcaster_active": "joint_state_broadcaster" in active,
            "scaled_joint_trajectory_controller_active": SCALED_CONTROLLER in active,
        }

    def _call_dashboard(
        self,
        label: str,
        client: Any,
        service_type: Any,
        transform: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        if not client.wait_for_service(timeout_sec=0.5):
            return {"available": False, "reason": f"{label} service is unavailable"}
        future = client.call_async(service_type.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None:
            return {"available": False, "reason": f"{label} returned no response"}
        data = transform(response)
        data["available"] = True
        return data

    def _dashboard_status(self) -> dict[str, Any]:
        remote = self._call_dashboard(
            "is_in_remote_control",
            self.dashboard_remote,
            IsInRemoteControl,
            lambda response: {
                "success": bool(response.success),
                "remote_control": bool(response.remote_control),
                "answer": response.answer,
            },
        )
        running = self._call_dashboard(
            "program_running",
            self.dashboard_program_running,
            IsProgramRunning,
            lambda response: {
                "success": bool(response.success),
                "program_running": bool(response.program_running),
                "answer": response.answer,
            },
        )
        robot_mode = self._call_dashboard(
            "get_robot_mode",
            self.dashboard_robot_mode,
            GetRobotMode,
            lambda response: {
                "success": bool(response.success),
                "mode": int(response.robot_mode.mode),
                "mode_name": ROBOT_MODE_NAMES.get(int(response.robot_mode.mode), "UNKNOWN"),
                "answer": response.answer,
            },
        )
        safety_mode = self._call_dashboard(
            "get_safety_mode",
            self.dashboard_safety_mode,
            GetSafetyMode,
            lambda response: {
                "success": bool(response.success),
                "mode": int(response.safety_mode.mode),
                "mode_name": SAFETY_MODE_NAMES.get(int(response.safety_mode.mode), "UNKNOWN"),
                "answer": response.answer,
            },
        )
        program_state = self._call_dashboard(
            "program_state",
            self.dashboard_program_state,
            GetProgramState,
            lambda response: {
                "success": bool(response.success),
                "state": response.state.state,
                "program_name": response.program_name,
                "answer": response.answer,
            },
        )
        return {
            "remote_control": remote,
            "program_running": running,
            "robot_mode": robot_mode,
            "safety_mode": safety_mode,
            "program_state": program_state,
            "note": (
                "The dashboard exposes whether a program is playing, not its internal URCap node label. "
                "A playing program plus Remote Control and an active scaled controller is the available "
                "ROS-side indication that External Control is likely running."
            ),
        }

    def _parameter_values(self, node_name: str, names: list[str]) -> dict[str, Any]:
        client = AsyncParameterClient(self, node_name)
        try:
            if not client.wait_for_services(timeout_sec=0.5):
                return {"available": False, "reason": f"parameter service unavailable for {node_name}"}
            future = client.get_parameters(names)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            if response is None:
                return {"available": False, "reason": f"no parameter response from {node_name}"}
            values = getattr(response, "values", response)
            return {
                "available": True,
                "values": {
                    name: _parameter_value_to_python(parameter)
                    for name, parameter in zip(names, values)
                },
            }
        except Exception as exc:
            return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}

    def _moveit_controller_mapping(self) -> dict[str, Any]:
        names = [
            "moveit_simple_controller_manager.controller_names",
            f"moveit_simple_controller_manager.{SCALED_CONTROLLER}.default",
            f"moveit_simple_controller_manager.{SCALED_CONTROLLER}.action_ns",
            f"moveit_simple_controller_manager.{SCALED_CONTROLLER}.type",
        ]
        result = self._parameter_values("/move_group", names)
        values = result.get("values", {}) if result.get("available") else {}
        controller_names = values.get(names[0], []) if isinstance(values, dict) else []
        result["scaled_controller_advertised"] = SCALED_CONTROLLER in controller_names
        result["expected_controller"] = SCALED_CONTROLLER
        return result

    @staticmethod
    def _sim_processes() -> list[str]:
        try:
            output = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"process inspection unavailable: {exc}"]
        patterns = ("gazebo", "gz sim", "ros_gz", "ur_simulation", "ur_sim_control")
        return [line.strip() for line in output.splitlines() if any(pattern in line.lower() for pattern in patterns)]

    @staticmethod
    def _recent_driver_log_alerts() -> list[dict[str, str]]:
        log_root = Path.home() / ".ros" / "log"
        if not log_root.is_dir():
            return []
        pattern = re.compile(
            r"path tolerance|goal tolerance|controller timeout|trajectory.*abort|goal.*reject|"
            r"follow_joint_trajectory|reverse interface|external control|protective stop|safety mode",
            re.IGNORECASE,
        )
        matches: list[dict[str, str]] = []
        for log_path in sorted(log_root.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)[:40]:
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
            except OSError:
                continue
            for line in lines:
                if pattern.search(line):
                    matches.append({"log": log_path.name, "line": line[-500:]})
                    if len(matches) >= 20:
                        return matches
        return matches

    def snapshot(self, *, wait_sec: float = 1.5) -> dict[str, Any]:
        self._spin_for(wait_sec)
        nodes = self._graph_nodes()
        controller_status = self._controller_status()
        dashboard = self._dashboard_status()
        action_servers = {
            "move_group_available": self.move_group.wait_for_server(timeout_sec=2.0),
            "scaled_follow_joint_trajectory_available": self.scaled_trajectory.wait_for_server(timeout_sec=2.0),
        }
        exact_counts = {node: nodes.count(node) for node in ("/move_group", "/controller_manager")}
        sim_nodes = [
            node
            for node in nodes
            if any(token in node.lower() for token in ("gazebo", "ros_gz", "ur_sim", "/gz"))
        ]
        return {
            "backend": "real",
            "joint_state": self._joint_state_status(),
            "tool0_pose": self._tool_pose(),
            "nodes": nodes,
            "required_nodes": {
                "move_group_exists": "/move_group" in nodes,
                "controller_manager_exists": "/controller_manager" in nodes,
                "duplicate_counts": exact_counts,
            },
            "controllers": controller_status,
            "action_servers": action_servers,
            "dashboard": dashboard,
            "speed_scaling_factor": None if self.latest_speed_scaling is None else float(self.latest_speed_scaling.data),
            "moveit_controller_mapping": self._moveit_controller_mapping(),
            "possible_sim_nodes": sim_nodes,
            "possible_sim_processes": self._sim_processes(),
        }

    @staticmethod
    def readiness_failures(snapshot: dict[str, Any], *, require_moveit: bool = True) -> list[str]:
        failures: list[str] = []
        joint_state = snapshot["joint_state"]
        tool_pose = snapshot["tool0_pose"]
        nodes = snapshot["required_nodes"]
        controllers = snapshot["controllers"]
        actions = snapshot["action_servers"]
        dashboard = snapshot["dashboard"]
        mapping = snapshot["moveit_controller_mapping"]
        if not joint_state.get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not tool_pose.get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        if not nodes.get("controller_manager_exists"):
            failures.append("/controller_manager node is unavailable")
        if not controllers.get("joint_state_broadcaster_active"):
            failures.append("joint_state_broadcaster is not active")
        if not controllers.get("scaled_joint_trajectory_controller_active"):
            failures.append("scaled_joint_trajectory_controller is not active")
        if not actions.get("scaled_follow_joint_trajectory_available"):
            failures.append(f"{SCALED_ACTION} action server is unavailable")
        if require_moveit and not nodes.get("move_group_exists"):
            failures.append("/move_group node is unavailable")
        if require_moveit and not actions.get("move_group_available"):
            failures.append(f"{MOVE_ACTION} action server is unavailable")
        if require_moveit and not mapping.get("scaled_controller_advertised"):
            failures.append("MoveIt does not advertise scaled_joint_trajectory_controller")
        if nodes.get("duplicate_counts", {}).get("/move_group", 0) > 1:
            failures.append("multiple /move_group nodes are visible")
        if snapshot.get("possible_sim_nodes") or snapshot.get("possible_sim_processes"):
            failures.append("possible Gazebo/simulation process or node is visible")
        remote = dashboard["remote_control"]
        program = dashboard["program_running"]
        robot = dashboard["robot_mode"]
        safety = dashboard["safety_mode"]
        if remote.get("available") and (not remote.get("success") or not remote.get("remote_control")):
            failures.append("teach pendant is not in Remote Control mode")
        if program.get("available") and (not program.get("success") or not program.get("program_running")):
            failures.append("the UR program containing External Control is not playing")
        if robot.get("available") and (not robot.get("success") or robot.get("mode_name") != "RUNNING"):
            failures.append(f"robot mode is {robot.get('mode_name', 'unknown')}, not RUNNING")
        if safety.get("available") and (not safety.get("success") or safety.get("mode_name") not in {"NORMAL", "REDUCED"}):
            failures.append(f"safety mode is {safety.get('mode_name', 'unknown')}, not NORMAL/REDUCED")
        return failures

    def _goal_for_target(self, target: dict[str, Any], *, plan_only: bool) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.pipeline_id = PILZ_PIPELINE
        goal.request.planner_id = PILZ_PLANNER
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = MAX_SCALING
        goal.request.max_acceleration_scaling_factor = MAX_SCALING
        if self.latest_joint_state is not None:
            goal.request.start_state.joint_state = self.latest_joint_state
            goal.request.start_state.is_diff = False

        target_pose = Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = target["position_m"]
        target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = target[
            "orientation_quat_xyzw"
        ]

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = BASE_FRAME
        position_constraint.link_name = TOOL_FRAME
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE_M]
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(target_pose)
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = BASE_FRAME
        orientation_constraint.link_name = TOOL_FRAME
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = "real_moveit_execution_probe_fixed_tool0_goal"
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _run_move_group(self, target: dict[str, Any], *, plan_only: bool) -> dict[str, Any]:
        if not self.move_group.wait_for_server(timeout_sec=5.0):
            return {
                "success": False,
                "stage": "move_group_availability",
                "reason": f"{MOVE_ACTION} action server is unavailable",
                "goal_accepted": False,
            }
        send_future = self.move_group.send_goal_async(self._goal_for_target(target, plan_only=plan_only))
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return {
                "success": False,
                "stage": "move_group_goal",
                "reason": "MoveGroup rejected the goal",
                "goal_accepted": False,
            }
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=EXECUTION_TIMEOUT_SEC)
        if result_future.result() is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "move_group_timeout",
                "reason": f"MoveGroup did not return within {EXECUTION_TIMEOUT_SEC:.0f} seconds; cancel requested",
                "goal_accepted": True,
                "cancel_requested": True,
            }
        result = result_future.result().result
        planned = _trajectory_summary(result.planned_trajectory)
        executed = _trajectory_summary(result.executed_trajectory)
        return {
            "success": result.error_code.val == 1,
            "stage": "move_group_plan_only" if plan_only else "move_group_execute",
            "goal_accepted": True,
            "result_status": int(result_future.result().status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": planned,
            "executed_trajectory": executed,
            "execution_goal_sent_to_move_group": not plan_only,
        }

    def build_target(self, current: dict[str, Any]) -> dict[str, Any]:
        position = current.get("position_m")
        orientation = current.get("orientation_quat_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise RuntimeError("current tool0 pose does not contain a three-value position")
        if not isinstance(orientation, list) or len(orientation) != 4:
            raise RuntimeError("current tool0 pose does not contain a quaternion")
        target_position = [float(position[0]), float(position[1]), float(position[2]) + self.probe_offset_base_z_m]
        if not all(math.isfinite(value) for value in target_position + [float(value) for value in orientation]):
            raise RuntimeError("current tool0 pose contains non-finite values")
        return {
            "frame_id": BASE_FRAME,
            "tool_link": TOOL_FRAME,
            "position_m": target_position,
            "orientation_quat_xyzw": [float(value) for value in orientation],
            "translation_offset_m": [0.0, 0.0, self.probe_offset_base_z_m],
            "orientation_preserved": True,
            "planner": f"{PILZ_PIPELINE}/{PILZ_PLANNER}",
            "max_velocity_scaling": MAX_SCALING,
            "max_acceleration_scaling": MAX_SCALING,
        }

    def plan(self, snapshot: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        snapshot = snapshot or self.snapshot()
        failures = self.readiness_failures(snapshot)
        if failures:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "readiness",
                "reason": "MoveIt plan was not requested because readiness checks failed",
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
            }
        current = self._tool_pose()
        target = self.build_target(current)
        result = self._run_move_group(target, plan_only=True)
        return (0 if result.get("success") else 2), {
            "success": bool(result.get("success")),
            "mode": "plan",
            "checks": snapshot,
            "current_tool0_pose": current,
            "target_tool0_pose": target,
            "plan_result": result,
            "planned_trajectory_duration_sec": result.get("planned_trajectory", {}).get("duration_sec"),
            "execution_sent": False,
        }

    def debug(self, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        snapshot = snapshot or self.snapshot()
        failures = self.readiness_failures(snapshot)
        if not failures:
            likely_cause = "All ROS-visible readiness checks pass. Run --mode plan, then use the gated --mode execute only after operator confirmation."
            next_command = "python3 scripts/real_moveit_execution_probe.py --mode plan"
        elif any("Remote Control" in item for item in failures):
            likely_cause = "The UR controller is not in Remote Control mode."
            next_command = "On the pendant, enable Remote Control, then rerun: python3 scripts/real_moveit_execution_probe.py --mode check"
        elif any("External Control" in item for item in failures):
            likely_cause = "The UR program containing External Control is not playing."
            next_command = "On the pendant, load/play the External Control program, then rerun: python3 scripts/real_moveit_execution_probe.py --mode check"
        elif any("safety mode" in item or "robot mode" in item for item in failures):
            likely_cause = "Robot mode or safety mode blocks trajectory execution."
            next_command = "Restore the robot to RUNNING with NORMAL/REDUCED safety mode on the pendant, then rerun --mode check."
        elif any("scaled_joint_trajectory_controller" in item for item in failures):
            likely_cause = "The real scaled trajectory controller is unavailable or inactive."
            next_command = "Verify External Control is playing, then run: ros2 control list_controllers"
        elif any("MoveIt" in item or MOVE_ACTION in item or "/move_group" in item for item in failures):
            likely_cause = "MoveIt is absent or is not advertising the real scaled controller mapping."
            next_command = "Start only the project real MoveIt wrapper: ./scripts/start_ur10e_real_moveit.sh"
        elif any("Gazebo" in item or "simulation" in item for item in failures):
            likely_cause = "A simulator is visible in the local ROS graph and can confuse controller or TF selection."
            next_command = "Stop only the identified simulator process/node, then rerun: python3 scripts/real_moveit_execution_probe.py --mode check"
        else:
            likely_cause = "A required driver or ROS endpoint is unavailable."
            next_command = "Start the UR driver, verify the pendant program is playing, then rerun --mode check."
        return {
            "success": not failures,
            "mode": "debug",
            "checks": snapshot,
            "failures": failures,
            "likely_cause": likely_cause,
            "next_command": next_command,
            "recent_driver_log_alerts": self._recent_driver_log_alerts(),
            "note": (
                "Debug mode is read-only. It reports current endpoint availability and recent driver alerts, "
                "but does not send a test trajectory or inspect a past action goal from another process."
            ),
        }

    def execute(self) -> tuple[int, dict[str, Any]]:
        if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "execution_gate",
                "reason": "execution requires UR10E_ALLOW_REAL_EXECUTION=1",
                "execution_sent": False,
            }
        snapshot = self.snapshot()
        failures = self.readiness_failures(snapshot)
        if failures:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "readiness",
                "reason": "execution was blocked because readiness checks failed",
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
            }
        before = self._tool_pose()
        target = self.build_target(before)
        plan_result = self._run_move_group(target, plan_only=True)
        if not plan_result.get("success"):
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "pre_execution_plan",
                "reason": "plan-only preflight failed; no execution goal was sent",
                "checks": snapshot,
                "before_tool0_pose": before,
                "target_tool0_pose": target,
                "plan_result": plan_result,
                "execution_sent": False,
            }
        # Recheck the physical readiness immediately before the sole execute
        # request.  A pendant program can stop between planning and execution.
        pre_execution = self.snapshot(wait_sec=0.2)
        late_failures = self.readiness_failures(pre_execution)
        if late_failures:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "pre_execution_readiness",
                "reason": "state changed after planning; execution was blocked",
                "failures": late_failures,
                "checks": pre_execution,
                "before_tool0_pose": before,
                "target_tool0_pose": target,
                "plan_result": plan_result,
                "execution_sent": False,
            }
        execution_result = self._run_move_group(target, plan_only=False)
        after = self._tool_pose(timeout_sec=8.0)
        actual: dict[str, Any]
        if before.get("available") and after.get("available"):
            delta = [after_value - before_value for after_value, before_value in zip(after["position_m"], before["position_m"])]
            orientation_error = _quaternion_distance_rad(before["orientation_quat_xyzw"], after["orientation_quat_xyzw"])
            actual = {
                "translation_delta_m": delta,
                "translation_norm_m": math.sqrt(sum(component * component for component in delta)),
                "orientation_delta_rad": orientation_error,
                "expected_z_delta_m": self.probe_offset_base_z_m,
                "z_delta_within_expected_probe_range": abs(delta[2] - self.probe_offset_base_z_m)
                <= MOTION_VERIFICATION_TOLERANCE_M,
                "orientation_preserved": orientation_error <= 0.01,
            }
        else:
            actual = {"available": False, "reason": "before/after TF pose was unavailable"}
        moved = bool(actual.get("z_delta_within_expected_probe_range") and actual.get("orientation_preserved"))
        success = bool(execution_result.get("success") and moved)
        response = {
            "success": success,
            "mode": "execute",
            "checks": snapshot,
            "before_tool0_pose": before,
            "target_tool0_pose": target,
            "plan_result": plan_result,
            "execution_result": execution_result,
            "after_tool0_pose": after,
            "actual_motion": actual,
            "execution_sent": bool(execution_result.get("goal_accepted")),
            "note": "The only action goal was sent to MoveIt /move_action; no raw controller goal was used.",
        }
        if not success:
            response["stage"] = "execution_verification"
            response["reason"] = "MoveIt execution failed or tool0 did not move by the expected base-Z probe distance"
            response["debug"] = self.debug(self.snapshot(wait_sec=0.2))
            return 2, response
        response["stage"] = "execute"
        response["reason"] = None
        return 0, response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "plan", "execute", "debug"), default="check")
    parser.add_argument(
        "--downward-2cm",
        action="store_true",
        help="Use the only alternate probe: fixed tool0 -2 cm in base_link Z, with orientation preserved.",
    )
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required together with UR10E_ALLOW_REAL_EXECUTION=1 for --mode execute.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.mode == "execute" and not arguments.confirm_real_motion:
        print(
            json.dumps(
                {
                    "success": False,
                    "mode": "execute",
                    "stage": "execution_gate",
                    "reason": "execution requires --confirm-real-motion",
                    "execution_sent": False,
                },
                sort_keys=True,
            )
        )
        return 2
    probe_offset = DOWNWARD_2CM_PROBE_OFFSET_BASE_Z_M if arguments.downward_2cm else DEFAULT_PROBE_OFFSET_BASE_Z_M
    rclpy.init(args=None)
    probe = RealMoveItExecutionProbe(probe_offset)
    try:
        if arguments.mode == "check":
            snapshot = probe.snapshot()
            failures = probe.readiness_failures(snapshot)
            result = {
                "success": not failures,
                "mode": "check",
                "checks": snapshot,
                "failures": failures,
                "execution_sent": False,
                "note": "Read-only checks only; no MoveIt or controller action goal was sent.",
            }
            code = 0 if not failures else 2
        elif arguments.mode == "plan":
            code, result = probe.plan()
        elif arguments.mode == "debug":
            result = probe.debug()
            code = 0 if result["success"] else 2
        else:
            code, result = probe.execute()
        print(json.dumps(_jsonable(result), sort_keys=True))
        return code
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "mode": arguments.mode,
                    "stage": "probe_exception",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "execution_sent": False,
                },
                sort_keys=True,
            )
        )
        return 2
    finally:
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
