#!/usr/bin/env python3
"""Isolated real-UR10e mock pre-mouth test using TF and MoveIt only.

This utility has no LLM, OpenClaw, feeding, camera, or perception imports.  It
defines a mock target relative to the current straw-tip pose, plans a stable
tool pose with MoveIt, and optionally executes that one plan through
``/move_action`` after explicit real-motion gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this physical-robot utility on the ThinkPad's local ROS graph.  This
# prevents a simulator discovered through another interface from supplying TF
# or a competing MoveIt/controller endpoint.
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
    GetRobotMode,
    GetSafetyMode,
    IsInRemoteControl,
    IsProgramRunning,
)


# MoveIt plans in the ROS model's base_link frame.  Semantic workcell
# directions are expressed in UR's physical ``base`` frame and transformed
# into base_link at runtime.  On this robot they differ by a 180-degree Z
# rotation, so treating their X axes as interchangeable is unsafe.
BASE_FRAME = "base_link"
UR_BASE_FRAME = "base"
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

# Geometry is expressed in the tool0/flange coordinate frame.  The camera
# value is recorded and reported for the known-tool-geometry check; this mock
# target is deliberately defined by the straw tip alone.
STRAW_TIP_OFFSET_TOOL0_M = (0.110, 0.0, 0.0)
CAMERA_OPTICAL_CENTER_OFFSET_TOOL0_M = (0.070, 0.0, 0.015)
MOCK_PRE_MOUTH_STRAW_DISPLACEMENT_UR_BASE_M = (0.050, 0.0, 0.0)

POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_RAD = 0.001
FINAL_STRAW_TARGET_TOLERANCE_M = 0.005
FINAL_ORIENTATION_TOLERANCE_RAD = 0.01
MAX_TOOL0_TRANSLATION_M = 0.10
MAX_VELOCITY_SCALING = 0.05
PILZ_PIPELINE = "pilz_industrial_motion_planner"
PILZ_PLANNER = "LIN"
ACTION_TIMEOUT_SEC = 60.0

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
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def _parameter_value_to_python(value: Any) -> Any:
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
    field = fields.get(int(getattr(value, "type", 0)))
    return None if field is None else _jsonable(getattr(value, field))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [left_value - right_value for left_value, right_value in zip(left, right)]


def _add(left: list[float], right: tuple[float, float, float] | list[float]) -> list[float]:
    return [left_value + right_value for left_value, right_value in zip(left, right)]


def _quaternion_distance_rad(first: list[float], second: list[float]) -> float:
    dot = abs(sum(left * right for left, right in zip(first, second)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _rotate_tool_vector(orientation_xyzw: list[float], vector_tool: tuple[float, float, float]) -> list[float]:
    """Return a tool-frame vector expressed in base_link using a unit quaternion."""

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


class RealPreMouthMockTest(Node):
    """Read-only checks plus one tightly bounded MoveGroup execution path."""

    def __init__(self) -> None:
        super().__init__("real_premouth_mock_test")
        self.latest_joint_state: JointState | None = None
        self.latest_speed_scaling: Float64 | None = None
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.create_subscription(Float64, SPEED_SCALING_TOPIC, self._speed_scaling_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        # This client is only used to inspect server availability.  It never
        # sends a raw FollowJointTrajectory goal.
        self.scaled_trajectory = ActionClient(self, FollowJointTrajectory, SCALED_ACTION)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self.dashboard_remote = self.create_client(IsInRemoteControl, "/dashboard_client/is_in_remote_control")
        self.dashboard_program = self.create_client(IsProgramRunning, "/dashboard_client/program_running")
        self.dashboard_robot_mode = self.create_client(GetRobotMode, "/dashboard_client/get_robot_mode")
        self.dashboard_safety_mode = self.create_client(GetSafetyMode, "/dashboard_client/get_safety_mode")

    def _joint_state_callback(self, message: JointState) -> None:
        self.latest_joint_state = message

    def _speed_scaling_callback(self, message: Float64) -> None:
        self.latest_speed_scaling = message

    def _wait_for_data(self, wait_sec: float = 0.7) -> None:
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _tool0_pose(self, timeout_sec: float = 1.0) -> dict[str, Any]:
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                TOOL_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except Exception as exc:
            return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "available": True,
            "frame_id": BASE_FRAME,
            "link_name": TOOL_FRAME,
            "position_m": [float(translation.x), float(translation.y), float(translation.z)],
            "orientation_quat_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
        }

    def _ur_base_pose_in_base_link(self, timeout_sec: float = 1.0) -> dict[str, Any]:
        """Return UR ``base`` expressed in the MoveIt planning frame."""

        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                UR_BASE_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except Exception as exc:
            return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "available": True,
            "parent_frame": BASE_FRAME,
            "child_frame": UR_BASE_FRAME,
            "position_m": [float(translation.x), float(translation.y), float(translation.z)],
            "orientation_quat_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
        }

    def _joint_state_status(self) -> dict[str, Any]:
        message = self.latest_joint_state
        if message is None:
            return {"received": False, "complete": False}
        positions = dict(zip(message.name, message.position))
        return {
            "received": True,
            "complete": all(name in positions for name in EXPECTED_JOINTS),
            "received_expected_joint_count": sum(name in positions for name in EXPECTED_JOINTS),
            "expected_joint_count": len(EXPECTED_JOINTS),
            "positions_rad": {name: float(positions[name]) for name in EXPECTED_JOINTS if name in positions},
        }

    def _controller_status(self) -> dict[str, Any]:
        if not self.controllers.wait_for_service(timeout_sec=0.5):
            return {"available": False, "reason": "/controller_manager/list_controllers is unavailable"}
        future = self.controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None:
            return {"available": False, "reason": "controller manager did not return a response"}
        listed = [{"name": item.name, "state": item.state, "type": item.type} for item in response.controller]
        active = {item["name"] for item in listed if item["state"] == "active"}
        return {
            "available": True,
            "controllers": listed,
            "joint_state_broadcaster_active": "joint_state_broadcaster" in active,
            "scaled_joint_trajectory_controller_active": SCALED_CONTROLLER in active,
        }

    def _call_dashboard(self, client: Any, service_type: Any, decode: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        if not client.wait_for_service(timeout_sec=0.4):
            return {"available": False, "reason": "dashboard service unavailable"}
        future = client.call_async(service_type.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        response = future.result()
        if response is None:
            return {"available": False, "reason": "dashboard service did not return"}
        return {"available": True, **decode(response)}

    def _dashboard_status(self) -> dict[str, Any]:
        return {
            "remote_control": self._call_dashboard(
                self.dashboard_remote,
                IsInRemoteControl,
                lambda response: {"success": bool(response.success), "remote_control": bool(response.remote_control), "answer": response.answer},
            ),
            "program_running": self._call_dashboard(
                self.dashboard_program,
                IsProgramRunning,
                lambda response: {"success": bool(response.success), "program_running": bool(response.program_running), "answer": response.answer},
            ),
            "robot_mode": self._call_dashboard(
                self.dashboard_robot_mode,
                GetRobotMode,
                lambda response: {
                    "success": bool(response.success),
                    "mode": int(response.robot_mode.mode),
                    "mode_name": ROBOT_MODE_NAMES.get(int(response.robot_mode.mode), "UNKNOWN"),
                },
            ),
            "safety_mode": self._call_dashboard(
                self.dashboard_safety_mode,
                GetSafetyMode,
                lambda response: {
                    "success": bool(response.success),
                    "mode": int(response.safety_mode.mode),
                    "mode_name": SAFETY_MODE_NAMES.get(int(response.safety_mode.mode), "UNKNOWN"),
                },
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
            return {
                "available": True,
                "values": {
                    name: _parameter_value_to_python(parameter)
                    for name, parameter in zip(names, getattr(response, "values", response))
                },
            }
        except Exception as exc:
            return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}

    def _moveit_controller_mapping(self) -> dict[str, Any]:
        names = [
            "moveit_simple_controller_manager.controller_names",
            f"moveit_simple_controller_manager.{SCALED_CONTROLLER}.default",
            f"moveit_simple_controller_manager.{SCALED_CONTROLLER}.action_ns",
        ]
        result = self._parameter_values("/move_group", names)
        values = result.get("values", {}) if result.get("available") else {}
        advertised = values.get(names[0], []) if isinstance(values, dict) else []
        result["scaled_controller_advertised"] = SCALED_CONTROLLER in advertised
        result["expected_controller"] = SCALED_CONTROLLER
        return result

    @staticmethod
    def _possible_sim_processes() -> list[str]:
        try:
            output = subprocess.run(
                ["ps", "-eo", "pid=,args="], check=False, capture_output=True, text=True, timeout=2.0
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"process inspection unavailable: {exc}"]
        patterns = ("gazebo", "gz sim", "ros_gz", "ur_simulation", "ur_sim_control")
        return [line.strip() for line in output.splitlines() if any(pattern in line.lower() for pattern in patterns)]

    def snapshot(self, wait_sec: float = 0.7) -> dict[str, Any]:
        self._wait_for_data(wait_sec)
        # A new rclpy process can receive the graph cache after its first TF
        # message.  Discover action servers before sampling node names so a
        # healthy local MoveIt process is not reported as absent transiently.
        move_group_available = self.move_group.wait_for_server(timeout_sec=2.0)
        scaled_action_available = self.scaled_trajectory.wait_for_server(timeout_sec=1.0)
        nodes = [
            f"{namespace.rstrip('/')}/{name}".replace("//", "/")
            for name, namespace in self.get_node_names_and_namespaces()
        ]
        controller_status = self._controller_status()
        tool0 = self._tool0_pose()
        ur_base_pose = self._ur_base_pose_in_base_link()
        return {
            "backend": "real",
            "joint_state": self._joint_state_status(),
            "tool0_pose": tool0,
            "ur_base_pose_in_base_link": ur_base_pose,
            "known_tool_geometry": {
                "tool0_to_straw_tip_m": list(STRAW_TIP_OFFSET_TOOL0_M),
                "tool0_to_camera_optical_center_m": list(CAMERA_OPTICAL_CENTER_OFFSET_TOOL0_M),
            },
            "required_nodes": {
                "move_group_exists": "/move_group" in nodes,
                "controller_manager_exists": "/controller_manager" in nodes,
                "duplicate_counts": {
                    "/move_group": nodes.count("/move_group"),
                    "/controller_manager": nodes.count("/controller_manager"),
                },
            },
            "controllers": controller_status,
            "action_servers": {
                "move_group_available": move_group_available,
                "scaled_follow_joint_trajectory_available": scaled_action_available,
            },
            "dashboard": self._dashboard_status(),
            "speed_scaling_percent": None if self.latest_speed_scaling is None else float(self.latest_speed_scaling.data),
            "moveit_controller_mapping": self._moveit_controller_mapping(),
            "possible_sim_nodes": [name for name in nodes if any(tag in name.lower() for tag in ("gazebo", "ur_sim", "ros_gz"))],
            "possible_sim_processes": self._possible_sim_processes(),
        }

    @staticmethod
    def readiness_failures(snapshot: dict[str, Any]) -> list[str]:
        failures: list[str] = []
        if not snapshot["joint_state"].get("complete"):
            failures.append("/joint_states is missing or incomplete")
        if not snapshot["tool0_pose"].get("available"):
            failures.append("TF base_link -> tool0 is unavailable")
        if not snapshot["ur_base_pose_in_base_link"].get("available"):
            failures.append("TF base_link -> UR base is unavailable")
        nodes = snapshot["required_nodes"]
        if not nodes.get("move_group_exists"):
            failures.append("/move_group node is unavailable")
        if not nodes.get("controller_manager_exists"):
            failures.append("/controller_manager node is unavailable")
        if nodes.get("duplicate_counts", {}).get("/move_group", 0) > 1:
            failures.append("multiple /move_group nodes are visible")
        controllers = snapshot["controllers"]
        if not controllers.get("joint_state_broadcaster_active"):
            failures.append("joint_state_broadcaster is not active")
        if not controllers.get("scaled_joint_trajectory_controller_active"):
            failures.append("scaled_joint_trajectory_controller is not active")
        actions = snapshot["action_servers"]
        if not actions.get("move_group_available"):
            failures.append(f"{MOVE_ACTION} action server is unavailable")
        if not actions.get("scaled_follow_joint_trajectory_available"):
            failures.append(f"{SCALED_ACTION} action server is unavailable")
        if not snapshot["moveit_controller_mapping"].get("scaled_controller_advertised"):
            failures.append("MoveIt does not advertise scaled_joint_trajectory_controller")
        if snapshot["possible_sim_nodes"] or snapshot["possible_sim_processes"]:
            failures.append("a possible simulator is visible in the local ROS graph")
        dashboard = snapshot["dashboard"]
        remote = dashboard["remote_control"]
        program = dashboard["program_running"]
        robot = dashboard["robot_mode"]
        safety = dashboard["safety_mode"]
        if remote.get("available") and (not remote.get("success") or not remote.get("remote_control")):
            failures.append("teach pendant is not in Remote Control mode")
        if program.get("available") and (not program.get("success") or not program.get("program_running")):
            failures.append("the UR External Control program is not playing")
        if robot.get("available") and (not robot.get("success") or robot.get("mode_name") != "RUNNING"):
            failures.append(f"robot mode is {robot.get('mode_name', 'unknown')}, not RUNNING")
        if safety.get("available") and (not safety.get("success") or safety.get("mode_name") not in {"NORMAL", "REDUCED"}):
            failures.append(f"safety mode is {safety.get('mode_name', 'unknown')}, not NORMAL/REDUCED")
        return failures

    @staticmethod
    def geometry_for_tool0(current_tool0: dict[str, Any], ur_base_pose_in_base_link: dict[str, Any]) -> dict[str, Any]:
        position = current_tool0.get("position_m")
        orientation = current_tool0.get("orientation_quat_xyzw")
        if not isinstance(position, list) or len(position) != 3 or not isinstance(orientation, list) or len(orientation) != 4:
            raise RuntimeError("current tool0 TF does not contain a complete pose")
        position = [float(value) for value in position]
        orientation = [float(value) for value in orientation]
        base_orientation = ur_base_pose_in_base_link.get("orientation_quat_xyzw")
        if not isinstance(base_orientation, list) or len(base_orientation) != 4:
            raise RuntimeError("UR base -> base_link TF does not contain an orientation")
        base_forward_displacement_in_base_link = _rotate_tool_vector(
            [float(value) for value in base_orientation], MOCK_PRE_MOUTH_STRAW_DISPLACEMENT_UR_BASE_M
        )
        straw_world_offset = _rotate_tool_vector(orientation, STRAW_TIP_OFFSET_TOOL0_M)
        camera_world_offset = _rotate_tool_vector(orientation, CAMERA_OPTICAL_CENTER_OFFSET_TOOL0_M)
        current_straw_position = _add(position, straw_world_offset)
        current_camera_position = _add(position, camera_world_offset)
        mock_target_position = _add(current_straw_position, base_forward_displacement_in_base_link)
        target_tool0_position = _subtract(mock_target_position, straw_world_offset)
        target_displacement = _subtract(target_tool0_position, position)
        return {
            "current_tool0_pose": current_tool0,
            "current_straw_tip_pose": {
                "frame_id": BASE_FRAME,
                "position_m": current_straw_position,
                "orientation_quat_xyzw": orientation,
            },
            "current_camera_optical_center_pose": {
                "frame_id": BASE_FRAME,
                "position_m": current_camera_position,
                "orientation_quat_xyzw": orientation,
            },
            "mock_pre_mouth_target": {
                "frame_id": BASE_FRAME,
                "position_m": mock_target_position,
                "orientation_quat_xyzw": orientation,
                "defined_relative_to": "current straw tip + [0.050, 0.0, 0.0] m in UR base, transformed into base_link",
            },
            "target_tool0_pose": {
                "frame_id": BASE_FRAME,
                "link_name": TOOL_FRAME,
                "position_m": target_tool0_position,
                "orientation_quat_xyzw": orientation,
            },
            "planned_tool0_displacement_m": target_displacement,
            "planned_tool0_translation_norm_m": _norm(target_displacement),
            "expected_straw_tip_displacement_ur_base_m": list(MOCK_PRE_MOUTH_STRAW_DISPLACEMENT_UR_BASE_M),
            "expected_straw_tip_displacement_base_link_m": base_forward_displacement_in_base_link,
            "orientation_difference_rad": 0.0,
            "orientation_preserved_by_target": True,
            "tool0_to_straw_tip_world_offset_m": straw_world_offset,
        }

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

    def _goal_for_target(self, target: dict[str, Any], *, plan_only: bool) -> MoveGroup.Goal:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target["position_m"]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = target["orientation_quat_xyzw"]
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = BASE_FRAME
        position_constraint.link_name = TOOL_FRAME
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE_M]
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose)
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0
        goal_constraints = Constraints()
        goal_constraints.name = "real_premouth_mock_straw_tip_goal"
        goal_constraints.position_constraints.append(position_constraint)
        goal_constraints.orientation_constraints.append(self._orientation_constraint(pose))

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.pipeline_id = PILZ_PIPELINE
        goal.request.planner_id = PILZ_PLANNER
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        goal.request.max_acceleration_scaling_factor = MAX_VELOCITY_SCALING
        if self.latest_joint_state is not None:
            goal.request.start_state.joint_state = self.latest_joint_state
            goal.request.start_state.is_diff = False
        goal.request.goal_constraints.append(goal_constraints)
        # A fixed end orientation and this path constraint prohibit an
        # unexpected wrist/tool reorientation during the Cartesian LIN motion.
        goal.request.path_constraints.orientation_constraints.append(self._orientation_constraint(pose))
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def _run_move_group(self, target: dict[str, Any], *, plan_only: bool) -> dict[str, Any]:
        if not self.move_group.wait_for_server(timeout_sec=5.0):
            return {"success": False, "stage": "move_group_availability", "reason": "MoveGroup action server unavailable"}
        future = self.move_group.send_goal_async(self._goal_for_target(target, plan_only=plan_only))
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return {"success": False, "stage": "move_group_goal", "reason": "MoveGroup rejected the goal", "goal_accepted": False}
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=ACTION_TIMEOUT_SEC)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return {
                "success": False,
                "stage": "move_group_timeout",
                "reason": f"MoveGroup did not respond within {ACTION_TIMEOUT_SEC:.0f} seconds; cancel requested",
                "goal_accepted": True,
                "cancel_requested": True,
            }
        result = wrapped_result.result
        return {
            "success": result.error_code.val == 1,
            "stage": "move_group_plan_only" if plan_only else "move_group_execute",
            "goal_accepted": True,
            "result_status": int(wrapped_result.status),
            "error_code": int(result.error_code.val),
            "error_message": result.error_code.message,
            "planning_time_sec": float(result.planning_time),
            "planned_trajectory": _trajectory_summary(result.planned_trajectory),
            "executed_trajectory": _trajectory_summary(result.executed_trajectory),
            "execution_goal_sent_to_move_group": not plan_only,
        }

    @staticmethod
    def _execution_guards(snapshot: dict[str, Any], geometry: dict[str, Any]) -> list[str]:
        guards: list[str] = []
        if geometry["planned_tool0_translation_norm_m"] > MAX_TOOL0_TRANSLATION_M:
            guards.append(
                f"planned tool0 translation is {geometry['planned_tool0_translation_norm_m']:.4f} m, over {MAX_TOOL0_TRANSLATION_M:.2f} m"
            )
        if geometry["orientation_difference_rad"] > FINAL_ORIENTATION_TOLERANCE_RAD:
            guards.append("target tool orientation differs unexpectedly from current orientation")
        speed = snapshot.get("speed_scaling_percent")
        if speed is None:
            guards.append("speed scaling is unavailable; execution requires a visible 5%-10% setting")
        elif not 5.0 <= float(speed) <= 10.0:
            guards.append(f"speed scaling is {float(speed):.2f}%, outside required 5%-10% range")
        return guards

    def plan(self, snapshot: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        snapshot = snapshot or self.snapshot()
        failures = self.readiness_failures(snapshot)
        if failures:
            return 2, {
                "success": False,
                "mode": "plan",
                "stage": "readiness",
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
            }
        geometry = self.geometry_for_tool0(snapshot["tool0_pose"], snapshot["ur_base_pose_in_base_link"])
        result = self._run_move_group(geometry["target_tool0_pose"], plan_only=True)
        return (0 if result.get("success") else 2), {
            "success": bool(result.get("success")),
            "mode": "plan",
            "checks": snapshot,
            **geometry,
            "plan_result": result,
            "planned_trajectory_duration_sec": result.get("planned_trajectory", {}).get("duration_sec"),
            "execution_sent": False,
        }

    def execute(self) -> tuple[int, dict[str, Any]]:
        snapshot = self.snapshot()
        failures = self.readiness_failures(snapshot)
        if failures:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "readiness",
                "failures": failures,
                "checks": snapshot,
                "execution_sent": False,
            }
        geometry = self.geometry_for_tool0(snapshot["tool0_pose"], snapshot["ur_base_pose_in_base_link"])
        guards = self._execution_guards(snapshot, geometry)
        if guards:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "execution_safety_guard",
                "failures": guards,
                "checks": snapshot,
                **geometry,
                "execution_sent": False,
            }
        plan_result = self._run_move_group(geometry["target_tool0_pose"], plan_only=True)
        if not plan_result.get("success"):
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "pre_execution_plan",
                "reason": "plan-only preflight failed; no execution goal was sent",
                "checks": snapshot,
                **geometry,
                "plan_result": plan_result,
                "execution_sent": False,
            }
        latest = self.snapshot(wait_sec=0.2)
        late_failures = self.readiness_failures(latest) + self._execution_guards(latest, geometry)
        if late_failures:
            return 2, {
                "success": False,
                "mode": "execute",
                "stage": "pre_execution_readiness",
                "failures": late_failures,
                "checks": latest,
                **geometry,
                "plan_result": plan_result,
                "execution_sent": False,
            }
        execution_result = self._run_move_group(geometry["target_tool0_pose"], plan_only=False)
        final_tool0 = self._tool0_pose(timeout_sec=8.0)
        actual: dict[str, Any]
        if final_tool0.get("available"):
            final_geometry = self.geometry_for_tool0(final_tool0, latest["ur_base_pose_in_base_link"])
            final_straw = final_geometry["current_straw_tip_pose"]
            target_straw = geometry["mock_pre_mouth_target"]
            final_error = _norm(_subtract(final_straw["position_m"], target_straw["position_m"]))
            actual = {
                "final_tool0_pose": final_tool0,
                "final_straw_tip_pose": final_straw,
                "final_distance_straw_tip_to_mock_target_m": final_error,
                "actual_tool0_displacement_m": _subtract(final_tool0["position_m"], geometry["current_tool0_pose"]["position_m"]),
                "actual_straw_tip_displacement_m": _subtract(
                    final_straw["position_m"], geometry["current_straw_tip_pose"]["position_m"]
                ),
                "orientation_difference_rad": _quaternion_distance_rad(
                    final_tool0["orientation_quat_xyzw"], geometry["current_tool0_pose"]["orientation_quat_xyzw"]
                ),
            }
            actual["orientation_stable"] = actual["orientation_difference_rad"] <= FINAL_ORIENTATION_TOLERANCE_RAD
            actual["straw_tip_reached_mock_target"] = final_error <= FINAL_STRAW_TARGET_TOLERANCE_M
        else:
            actual = {"available": False, "reason": "final TF base_link -> tool0 is unavailable"}
        success = bool(execution_result.get("success") and actual.get("straw_tip_reached_mock_target") and actual.get("orientation_stable"))
        response = {
            "success": success,
            "mode": "execute",
            "checks": snapshot,
            **geometry,
            "plan_result": plan_result,
            "execution_result": execution_result,
            "actual": actual,
            "execution_sent": bool(execution_result.get("goal_accepted")),
            "note": "Only MoveIt /move_action was used; no raw FollowJointTrajectory goal was sent.",
        }
        if not success:
            response["stage"] = "execution_verification"
            response["reason"] = "MoveIt execution failed, orientation changed, or straw tip missed the mock target"
            return 2, response
        response["stage"] = "execute"
        response["reason"] = None
        return 0, response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "plan", "execute"), default="check")
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required with UR10E_ALLOW_REAL_EXECUTION=1 for --mode execute.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.mode == "execute" and not arguments.confirm_real_motion:
        print(json.dumps({"success": False, "mode": "execute", "stage": "execution_gate", "reason": "execution requires --confirm-real-motion", "execution_sent": False}))
        return 2
    if arguments.mode == "execute" and os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
        print(json.dumps({"success": False, "mode": "execute", "stage": "execution_gate", "reason": "execution requires UR10E_ALLOW_REAL_EXECUTION=1", "execution_sent": False}))
        return 2
    rclpy.init(args=None)
    test = RealPreMouthMockTest()
    try:
        if arguments.mode == "check":
            snapshot = test.snapshot()
            result = {
                "success": not test.readiness_failures(snapshot),
                "mode": "check",
                "checks": snapshot,
                "current_tool_geometry": (
                    test.geometry_for_tool0(snapshot["tool0_pose"], snapshot["ur_base_pose_in_base_link"])
                    if snapshot["tool0_pose"].get("available") and snapshot["ur_base_pose_in_base_link"].get("available")
                    else None
                ),
                "failures": test.readiness_failures(snapshot),
                "execution_sent": False,
                "note": "Read-only checks only; no MoveIt or controller goal was sent.",
            }
            code = 0 if result["success"] else 2
        elif arguments.mode == "plan":
            code, result = test.plan()
        else:
            code, result = test.execute()
        print(json.dumps(_jsonable(result), sort_keys=True))
        return code
    except Exception as exc:
        print(json.dumps({"success": False, "mode": arguments.mode, "stage": "probe_exception", "reason": f"{exc.__class__.__name__}: {exc}", "execution_sent": False}, sort_keys=True))
        return 2
    finally:
        test.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
