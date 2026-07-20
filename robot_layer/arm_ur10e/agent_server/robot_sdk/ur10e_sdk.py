#!/usr/bin/env python3
"""
UR10e ROS2 SDK for ABot-Claw.

This module mirrors the high-level shape of the Piper SDK, but the backend is
ROS2 Jazzy + MoveIt2 + ros2_control. It is designed for the current setup:

    5090 host/container: runs this SDK and ABot-Claw agent code
    ThinkPad: runs UR10e simulation, MoveIt, RViz, and controllers

Main public APIs:
    - move_joints()
    - move_to_pose()
    - move_straw_tip_to_pre_mouth()
    - move_straw_tip_to_mouth()
    - retreat()
    - move_relative()
    - reset()
    - get_robot_state()
    - get_robot_end_pose()
    - get_observation()

Unsupported for bare UR10e simulation for now:
    - set_gripper()
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

import rclpy
import rclpy.time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    CollisionObject,
    Constraints,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
import tf2_ros

try:  # Works for normal package imports.
    from .backend import (
        UR10eBackendSettings,
        require_real_execution_authorized,
        resolve_ur10e_backend_settings,
    )
except ImportError:  # ``feeding_tools`` loads this module directly by path.
    from robot_layer.arm_ur10e.agent_server.robot_sdk.backend import (  # type: ignore[no-redef]
        UR10eBackendSettings,
        require_real_execution_authorized,
        resolve_ur10e_backend_settings,
    )


_ROBOT_SDK_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PROJECT_CONFIG = _PROJECT_ROOT / "config" / "ur10e_sdk_config.yaml"
# The root project config is the canonical source for feeding geometry and
# backend policy.  The package-local file remains only as a compatibility
# fallback for out-of-tree users of this SDK.
_DEFAULT_CONFIG = str(_PROJECT_CONFIG if _PROJECT_CONFIG.is_file() else Path(_ROBOT_SDK_DIR) / "config.yaml")


def _load_config() -> dict:
    cfg_path = os.environ.get("UR10E_SDK_CONFIG", _DEFAULT_CONFIG)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@dataclass
class CameraFrame:
    device_id: str
    stream_type: str
    frame: bytes | np.ndarray | None
    timestamp: float
    width: int = 0
    height: int = 0
    depth_scale: Optional[float] = None
    encoding: str = ""
    frame_id: str = ""


class _UR10eRosNode(Node):
    def __init__(self, cfg: dict):
        super().__init__("ur10e_robot_env")
        ros_cfg = cfg.get("ros", {})
        moveit_cfg = cfg.get("moveit", {})
        controller_cfg = cfg.get("controller", {})
        observation_cfg = cfg.get("observation", {})

        self.base_frame = ros_cfg.get("base_frame_id", "base_link")
        self.tool_frame = ros_cfg.get("tool_frame_id", "tool0")
        self.joint_state_topic = ros_cfg.get("joint_state_topic", "/joint_states")
        self.group_name = moveit_cfg.get("group_name", "ur_manipulator")
        self.trajectory_action = controller_cfg.get(
            "trajectory_action",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )
        self.rgb_topic = observation_cfg.get("rgb_topic", "/wrist_rgbd/image")
        self.depth_topic = observation_cfg.get("depth_topic", "/wrist_rgbd/depth_image")
        self.camera_info_topic = observation_cfg.get(
            "camera_info_topic", "/wrist_rgbd/camera_info"
        )
        self.camera_frame_id = observation_cfg.get(
            "camera_frame_id", "wrist_rgbd_camera_optical_frame"
        )
        self.straw_tip_frame_id = observation_cfg.get(
            "straw_tip_frame_id", "feeding_straw_tip_marker"
        )

        self.latest_joint_state: Optional[JointState] = None
        self.latest_rgb_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_cb,
            10,
        )
        self.create_subscription(Image, self.rgb_topic, self._rgb_image_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.depth_topic, self._depth_image_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            qos_profile_sensor_data,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.apply_planning_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action,
        )
        self.move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self.list_controllers_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )

    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def _rgb_image_cb(self, msg: Image):
        self.latest_rgb_image = msg

    def _depth_image_cb(self, msg: Image):
        self.latest_depth_image = msg

    def _camera_info_cb(self, msg: CameraInfo):
        self.latest_camera_info = msg


class UR10eRobotEnv:
    """High-level UR10e robot environment used by ABot-Claw agent code."""

    def __init__(
        self,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        init_ros_node: bool = True,
        config: Optional[dict] = None,
    ):
        self.cfg = copy.deepcopy(config) if config is not None else _load_config()
        self.backend: UR10eBackendSettings = resolve_ur10e_backend_settings(self.cfg)
        # There is one SDK implementation.  Backends only select the ROS
        # controller endpoint used by that same MoveIt-based implementation.
        controller_cfg = dict(self.cfg.get("controller", {}))
        controller_cfg["trajectory_action"] = self.backend.trajectory_action
        self.cfg["controller"] = controller_cfg
        ur_cfg = self.cfg.get("ur10e", {})
        self.NUM_JOINTS = int(ur_cfg.get("num_joints", 6))
        self.JOINT_NAMES = list(
            ur_cfg.get(
                "joint_names",
                [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            )
        )
        self.RESET_JOINTS = list(
            ur_cfg.get("reset_joints", [0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        )
        self.max_velocity = float(
            max_velocity if max_velocity is not None else ur_cfg.get("max_velocity", 0.2)
        )
        self.max_acceleration = float(
            max_acceleration
            if max_acceleration is not None
            else ur_cfg.get("max_acceleration", 0.2)
        )
        if self.backend.max_velocity_limit is not None:
            self.max_velocity = min(self.max_velocity, self.backend.max_velocity_limit)
        if self.backend.max_acceleration_limit is not None:
            self.max_acceleration = min(self.max_acceleration, self.backend.max_acceleration_limit)
        self.default_duration = float(ur_cfg.get("default_duration", 7.0))
        self.cartesian_max_step = float(ur_cfg.get("cartesian_max_step", 0.01))
        self.min_cartesian_fraction = float(ur_cfg.get("min_cartesian_fraction", 0.99))
        self.avoid_collisions = bool(ur_cfg.get("avoid_collisions", True))
        observation_cfg = self.cfg.get("observation", {})
        self.camera_frame_id = observation_cfg.get(
            "camera_frame_id", "wrist_rgbd_camera_optical_frame"
        )
        self.straw_tip_frame_id = observation_cfg.get(
            "straw_tip_frame_id", "feeding_straw_tip_marker"
        )
        feeding_cfg = self.cfg.get("feeding", {})
        self.flange_to_camera_optical_center = list(
            feeding_cfg.get("flange_to_camera_optical_center", [0.07, 0.0, -0.015])
        )
        self.flange_to_straw_tip = list(
            feeding_cfg.get("flange_to_straw_tip", [0.11, 0.0, 0.0])
        )
        self.ready_straw_tip_position = list(
            feeding_cfg.get("ready_straw_tip_position", [0.25, 0.25, 0.85])
        )
        self.pre_mouth_safe_position = list(
            feeding_cfg.get("pre_mouth_safe_position", [0.357, 0.860, 1.708])
        )
        self.mouth_target_position = list(
            feeding_cfg.get("mouth_target_position", [0.357, 0.940, 1.708])
        )
        self.flange_down_rpy = list(
            feeding_cfg.get("flange_down_rpy", [math.pi, 0.0, 0.0])
        )

        if init_ros_node and not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.node = _UR10eRosNode(self.cfg)
        self._wait_until_ready()
        self._spin_until_joint_state(timeout=5.0)
        print("UR10eRobotEnv (ROS2 + MoveIt2) initialized")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def close(self):
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _spin(self, timeout_sec: float = 0.1):
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def _wait_until_ready(self):
        # Planning and read-only state access must remain possible on the
        # real backend before a motion controller is armed.  Controller and
        # execution permission checks therefore happen immediately before an
        # executing goal, not during SDK construction.
        checks = [("/move_action", self.node.move_group_client.wait_for_server)]
        for name, wait_fn in checks:
            if not wait_fn(timeout_sec=30.0):
                raise RuntimeError(f"{name} is not available")

    def _spin_until_joint_state(self, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.node.latest_joint_state is not None:
                return True
            self._spin(0.1)
        return False

    def _has_complete_joint_state(self) -> bool:
        msg = self.node.latest_joint_state
        if msg is None:
            return False
        index = {name: i for i, name in enumerate(msg.name)}
        return all(index.get(name) is not None and index[name] < len(msg.position) for name in self.JOINT_NAMES)

    def _active_controller_names(self, timeout_sec: float = 3.0) -> list[str]:
        """Read controller state without publishing a command."""
        if not self.node.list_controllers_client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/controller_manager/list_controllers is not available")
        future = self.node.list_controllers_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        response = future.result()
        if response is None:
            raise RuntimeError("/controller_manager/list_controllers returned no response")
        return [controller.name for controller in response.controller if controller.state == "active"]

    def _ensure_execution_ready(self) -> None:
        """Perform safety prerequisites immediately before sending a motion goal."""
        require_real_execution_authorized(self.backend)
        if not self.node.trajectory_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"{self.node.trajectory_action} is not available")
        if not self._spin_until_joint_state(timeout=2.0) or not self._has_complete_joint_state():
            raise RuntimeError("a complete /joint_states message is required before execution")
        # This also makes a missing base_link -> tool0 transform a hard stop.
        self._current_pose_msg()
        if self.backend.is_real:
            active = self._active_controller_names()
            if self.backend.expected_controller not in active:
                raise RuntimeError(
                    "real UR10e execution requires active controller "
                    f"{self.backend.expected_controller!r}; active controllers: {active}"
                )

    def _current_pose_msg(self) -> Pose:
        deadline = time.time() + 30.0
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            self._spin(0.1)
            try:
                tf = self.node.tf_buffer.lookup_transform(
                    self.node.base_frame,
                    self.node.tool_frame,
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=1.0),
                )
                pose = Pose()
                pose.position.x = tf.transform.translation.x
                pose.position.y = tf.transform.translation.y
                pose.position.z = tf.transform.translation.z
                pose.orientation = tf.transform.rotation
                return pose
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not read TF {self.node.base_frame}->{self.node.tool_frame}: {last_error}"
        )

    def _frame_pose(self, frame_id: str, timeout: float = 2.0) -> Dict[str, object]:
        """Return a named frame pose in the configured base frame."""
        deadline = time.time() + timeout
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            self._spin(0.05)
            try:
                tf = self.node.tf_buffer.lookup_transform(
                    self.node.base_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=0.2),
                )
                q = tf.transform.rotation
                return {
                    "position": [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ],
                    "orientation_quat": [q.x, q.y, q.z, q.w],
                    "orientation_euler": list(euler_from_quaternion(q.x, q.y, q.z, q.w)),
                    "timestamp": time.time(),
                    "frame_id": self.node.base_frame,
                    "link_name": frame_id,
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not read TF {self.node.base_frame}->{frame_id}: {last_error}"
        )

    def _convert_endpose(self, endpose: Sequence[float]) -> Pose:
        if len(endpose) == 6:
            x, y, z, roll, pitch, yaw = [float(v) for v in endpose]
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        elif len(endpose) == 7:
            x, y, z, qx, qy, qz, qw = [float(v) for v in endpose]
        else:
            raise ValueError("endpose must have 6 Euler values or 7 quaternion values")

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def _solve_ik(self, pose: Pose):
        if not self.node.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                "/compute_ik is not available. Start MoveIt move_group and check ROS2 "
                "communication before using endpose control."
            )

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.node.group_name
        req.ik_request.ik_link_name = self.node.tool_frame
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = self.node.base_frame
        req.ik_request.pose_stamped.pose = pose
        req.ik_request.avoid_collisions = self.avoid_collisions
        req.ik_request.timeout.sec = 2
        if self.node.latest_joint_state is None:
            self._spin_until_joint_state(timeout=2.0)
        if self.node.latest_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.node.latest_joint_state
            req.ik_request.robot_state.is_diff = False

        future = self.node.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=15.0)
        if future.result() is None:
            raise RuntimeError("/compute_ik call failed")
        return future.result()

    def _call_ik(self, pose: Pose):
        return self._solve_ik(pose).error_code

    def _joint_positions_from_ik_solution(self, ik_response) -> list[float]:
        joint_state = ik_response.solution.joint_state
        index = {name: i for i, name in enumerate(joint_state.name)}
        out = []
        for name in self.JOINT_NAMES:
            i = index.get(name)
            if i is None or i >= len(joint_state.position):
                raise RuntimeError(f"IK solution did not include joint {name}")
            out.append(float(joint_state.position[i]))
        return out

    def _plan_cartesian_path(self, pose: Pose):
        if not self.node.cartesian_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                "/compute_cartesian_path is not available. Start the MoveIt move_group "
                "server before using endpose control."
            )

        req = GetCartesianPath.Request()
        req.header.frame_id = self.node.base_frame
        req.group_name = self.node.group_name
        req.link_name = self.node.tool_frame
        req.waypoints = [pose]
        req.max_step = self.cartesian_max_step
        req.jump_threshold = 0.0
        req.avoid_collisions = self.avoid_collisions
        if self.node.latest_joint_state is None:
            self._spin_until_joint_state(timeout=2.0)
        if self.node.latest_joint_state is not None:
            req.start_state.joint_state = self.node.latest_joint_state
            req.start_state.is_diff = False
        else:
            req.start_state.is_diff = True

        future = self.node.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=15.0)
        if future.result() is None:
            raise RuntimeError("/compute_cartesian_path call failed")
        return future.result()

    def _move_group_to_pose(
        self,
        pose: Pose,
        duration: Optional[float] = None,
        plan_only: bool = False,
        *,
        orientation_tolerance_rad: float = 0.05,
        enforce_path_orientation: bool = False,
    ) -> Dict[str, object]:
        if not self.node.move_group_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("/move_action is not available. Start MoveIt move_group first.")
        if not plan_only:
            self._ensure_execution_ready()
        try:
            orientation_tolerance = float(orientation_tolerance_rad)
        except (TypeError, ValueError) as exc:
            raise ValueError("orientation_tolerance_rad must be a finite value in (0, 0.05]") from exc
        if not math.isfinite(orientation_tolerance) or not 0.0 < orientation_tolerance <= 0.05:
            raise ValueError("orientation_tolerance_rad must be a finite value in (0, 0.05]")

        if self.node.latest_joint_state is None:
            self._spin_until_joint_state(timeout=2.0)

        goal = MoveGroup.Goal()
        goal.request.group_name = self.node.group_name
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = min(max(self.max_velocity, 0.01), 1.0)
        goal.request.max_acceleration_scaling_factor = min(max(self.max_acceleration, 0.01), 1.0)
        if self.node.latest_joint_state is not None:
            goal.request.start_state.joint_state = self.node.latest_joint_state
            goal.request.start_state.is_diff = False

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.node.base_frame
        position_constraint.link_name = self.node.tool_frame
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]
        region_pose = Pose()
        region_pose.position = pose.position
        region_pose.orientation.w = 1.0
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(region_pose)
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.node.base_frame
        orientation_constraint.link_name = self.node.tool_frame
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance
        orientation_constraint.weight = 1.0

        goal_constraint = Constraints()
        goal_constraint.name = "tool0_pose_goal"
        goal_constraint.position_constraints.append(position_constraint)
        goal_constraint.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints.append(goal_constraint)
        if enforce_path_orientation:
            path_constraint = Constraints()
            path_constraint.name = "tool0_orientation_path_constraint"
            path_constraint.orientation_constraints.append(copy.deepcopy(orientation_constraint))
            goal.request.path_constraints = path_constraint

        goal.planning_options.plan_only = bool(plan_only)
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay = 0.2

        send_future = self.node.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveGroup rejected the goal")

        result_future = handle.get_result_async()
        timeout = max(30.0, (duration or self.default_duration) + 25.0)
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout)
        if result_future.result() is None:
            raise RuntimeError("MoveGroup planning/execution timed out")
        result = result_future.result().result
        points = len(result.planned_trajectory.joint_trajectory.points)
        return {
            "success": result.error_code.val == 1,
            "stage": "move_group_plan_execute" if not plan_only else "move_group_plan_only",
            "error_code": result.error_code.val,
            "points": points,
            "planning_time": result.planning_time,
        }

    def _retime_trajectory(self, robot_trajectory, duration: float):
        """Scale MoveIt's existing timing without invalidating its derivatives.

        ``/compute_cartesian_path`` already returns a time-parameterized
        trajectory with joint velocities and accelerations.  Replacing only
        ``time_from_start`` makes those derivatives physically inconsistent,
        which causes the Gazebo position controller to visibly jerk.  A
        uniform time scale preserves the collision-checked joint path and its
        velocity/acceleration profile while stretching it to the requested
        gentle duration.
        """
        points = robot_trajectory.joint_trajectory.points
        duration = max(float(duration), 1.0)
        if len(points) == 0:
            raise RuntimeError("Trajectory has no points")
        if len(points) == 1:
            whole = int(duration)
            points[0].time_from_start = Duration(
                sec=whole,
                nanosec=int((duration - whole) * 1_000_000_000),
            )
            return

        source_times = [
            float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
            for point in points
        ]
        source_duration = source_times[-1]
        if source_duration <= 0.0 or any(
            later < earlier for earlier, later in zip(source_times, source_times[1:])
        ):
            raise RuntimeError("MoveIt returned an invalid trajectory timing")
        time_scale = duration / source_duration
        for point, source_seconds in zip(points, source_times):
            seconds = source_seconds * time_scale
            whole = int(seconds)
            point.time_from_start = Duration(
                sec=whole,
                nanosec=int((seconds - whole) * 1_000_000_000),
            )
            if point.velocities:
                point.velocities = [float(value) / time_scale for value in point.velocities]
            if point.accelerations:
                point.accelerations = [float(value) / (time_scale * time_scale) for value in point.accelerations]

    def apply_human_keepout(self) -> Dict[str, object]:
        """Add the configured human safety volume to MoveIt's planning scene.

        Gazebo's DART backend is unstable when a static contact proxy touches
        this fixed-base UR model, so the proxy is deliberately visual-only in
        Gazebo.  This method restores the same conservative torso/head/mouth
        exclusion volume where it belongs: MoveIt's collision checker.
        """
        scene_cfg = self.cfg.get("planning_scene", {}).get("human_keepout", {})
        if not scene_cfg.get("enabled", False):
            return {"success": True, "enabled": False, "objects": []}

        objects = scene_cfg.get("objects", [])
        if not isinstance(objects, list) or not objects:
            raise ValueError("planning_scene.human_keepout.objects must be a non-empty list")
        if not self.node.apply_planning_scene_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/apply_planning_scene is not available")

        scene = PlanningScene()
        scene.is_diff = True
        added_ids: list[str] = []
        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                raise ValueError("Each human keepout object must be a mapping")
            object_id = str(item.get("id", "")).strip()
            shape = str(item.get("shape", "")).strip().lower()
            center = item.get("center")
            dimensions = item.get("dimensions")
            if not object_id or not isinstance(center, list) or len(center) != 3:
                raise ValueError("Each human keepout object needs id and a three-value center")
            if not all(math.isfinite(float(value)) for value in center):
                raise ValueError(f"Human keepout {object_id} has non-finite center values")

            collision = CollisionObject()
            collision.header.frame_id = self.node.base_frame
            collision.id = object_id
            collision.operation = CollisionObject.ADD
            primitive = SolidPrimitive()
            if shape == "box":
                if not isinstance(dimensions, list) or len(dimensions) != 3:
                    raise ValueError(f"Box keepout {object_id} needs three dimensions")
                values = [float(value) for value in dimensions]
                if not all(math.isfinite(value) and value > 0.0 for value in values):
                    raise ValueError(f"Box keepout {object_id} dimensions must be positive")
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = values
            elif shape == "sphere":
                if not isinstance(dimensions, list) or len(dimensions) != 1:
                    raise ValueError(f"Sphere keepout {object_id} needs one radius")
                radius = float(dimensions[0])
                if not math.isfinite(radius) or radius <= 0.0:
                    raise ValueError(f"Sphere keepout {object_id} radius must be positive")
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [radius]
            else:
                raise ValueError(f"Unsupported human keepout shape {shape!r} at index {index}")

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(value) for value in center]
            pose.orientation.w = 1.0
            collision.primitives.append(primitive)
            collision.primitive_poses.append(pose)
            scene.world.collision_objects.append(collision)
            added_ids.append(object_id)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.node.apply_planning_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("MoveIt rejected the human keepout planning-scene update")
        return {"success": True, "enabled": True, "objects": added_ids}

    def _execute_trajectory(self, robot_trajectory, duration: Optional[float] = None):
        if self.backend.is_real:
            raise RuntimeError(
                "the real UR10e backend does not send direct trajectory-controller goals; "
                "use the reviewed MoveGroup pose path"
            )
        self._ensure_execution_ready()
        execution_duration = float(duration or self.default_duration)
        self._retime_trajectory(robot_trajectory, execution_duration)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory

        send_future = self.node.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory controller rejected the goal")

        result_future = handle.get_result_async()
        # Gazebo with GUI/RGB-D rendering can run substantially slower than
        # real time. Allow a bounded wall-clock margin, but cancel explicitly
        # on expiry so a caller never leaves an uncontrolled active goal.
        timeout = max(30.0, execution_duration * 3.0 + 15.0)
        rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=timeout,
        )
        if result_future.result() is None:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=5.0)
            raise RuntimeError(
                "Trajectory execution timed out after %.1f s; cancel requested" % timeout
            )
        return result_future.result().result

    # ------------------------------------------------------------------ #
    # Public API, Piper-compatible shape
    # ------------------------------------------------------------------ #

    def move_joints(
        self,
        joint_states: Sequence[float],
        gripper=None,
        max_velocity=None,
        max_acceleration=None,
        duration: Optional[float] = None,
    ) -> Dict[str, object]:
        """Move UR10e to target joint positions via FollowJointTrajectory."""
        if self.backend.is_real:
            raise RuntimeError(
                "the real UR10e backend rejects direct joint commands; use a reviewed MoveIt pose goal"
            )
        if len(joint_states) != self.NUM_JOINTS:
            raise ValueError(f"joint_states must have {self.NUM_JOINTS} values")
        if gripper is not None:
            print("UR10e SDK: gripper argument ignored; no gripper is configured yet")

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        trajectory = JointTrajectory()
        trajectory.joint_names = self.JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joint_states]
        trajectory.points = [point]

        class _RobotTrajectory:
            pass

        robot_trajectory = _RobotTrajectory()
        robot_trajectory.joint_trajectory = trajectory
        result = self._execute_trajectory(robot_trajectory, duration=duration)
        return {"success": result.error_code == 0, "error_code": result.error_code}

    def move_to_pose(
        self,
        endpose: Sequence[float],
        max_velocity=None,
        max_acceleration=None,
        duration: Optional[float] = None,
        plan_only: bool = False,
        *,
        strict_orientation: bool = False,
    ) -> Dict[str, object]:
        """Move tool0 to an absolute pose in base_link.

        endpose formats:
            [x, y, z, roll, pitch, yaw]
            [x, y, z, qx, qy, qz, qw]
        """
        target = self._convert_endpose(endpose)
        if self.backend.is_real:
            # The physical backend always gives pose goals to MoveIt.  It
            # never publishes a raw joint trajectory from this SDK.
            return self._move_group_to_pose(
                target,
                duration=duration,
                plan_only=plan_only,
                orientation_tolerance_rad=0.001 if strict_orientation else 0.05,
                enforce_path_orientation=bool(strict_orientation),
            )
        try:
            current = self._current_pose_msg()
            position_error = math.sqrt(
                (current.position.x - target.position.x) ** 2
                + (current.position.y - target.position.y) ** 2
                + (current.position.z - target.position.z) ** 2
            )
            orientation_dot = abs(
                current.orientation.x * target.orientation.x
                + current.orientation.y * target.orientation.y
                + current.orientation.z * target.orientation.z
                + current.orientation.w * target.orientation.w
            )
            if position_error <= 0.005 and orientation_dot >= 0.999:
                return {
                    "success": True,
                    "stage": "already_at_target",
                    "position_error_m": position_error,
                    "orientation_dot": orientation_dot,
                    "points": 0,
                }
        except RuntimeError:
            pass

        path = self._plan_cartesian_path(target)
        points = len(path.solution.joint_trajectory.points)
        if path.fraction < self.min_cartesian_fraction or points < 2:
            return {
                "success": False,
                "stage": "cartesian_path",
                "fraction": path.fraction,
                "points": points,
                "error_code": path.error_code.val,
            }

        if plan_only:
            return {
                "success": True,
                "stage": "plan_only",
                "fraction": path.fraction,
                "points": points,
            }

        result = self._execute_trajectory(path.solution, duration=duration)
        end_pose = self.get_robot_end_pose()
        return {
            "success": result.error_code == 0,
            "stage": "execute",
            "error_code": result.error_code,
            "error_string": result.error_string,
            "fraction": path.fraction,
            "points": points,
            "end_pose": end_pose,
        }

    def move_to_pose_ik_joint_target(
        self,
        endpose: Sequence[float],
        duration: Optional[float] = None,
        plan_only: bool = False,
    ) -> Dict[str, object]:
        """Move to an absolute tool0 pose by solving IK then sending joints.

        This is useful as a setup/ready motion when the current tool orientation
        is not already aligned with the target orientation. The feeding motion
        itself should still use Cartesian planning once the flange is down.
        """
        target = self._convert_endpose(endpose)
        if self.backend.is_real:
            return self._move_group_to_pose(target, duration=duration, plan_only=plan_only)
        ik_response = self._solve_ik(target)
        if ik_response.error_code.val != 1:
            return {"success": False, "stage": "ik", "error_code": ik_response.error_code.val}
        joint_positions = self._joint_positions_from_ik_solution(ik_response)
        if plan_only:
            return {
                "success": True,
                "stage": "ik_joint_target_plan_only",
                "joint_positions": joint_positions,
            }
        result = self.move_joints(joint_positions, duration=duration)
        return {
            "success": bool(result.get("success")),
            "stage": "ik_joint_target_execute",
            "joint_positions": joint_positions,
            "move_result": result,
        }

    def plan_straw_tip_to_pose(
        self,
        straw_tip_position: Sequence[float],
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
        label: str = "straw_tip_target",
    ) -> Dict[str, object]:
        """Compute the tool0 pose that puts the straw tip at a target.

        MoveIt controls the UR10e `tool0` frame, but the feeding task target is
        the straw tip mounted below/near the flange. This helper converts:

            desired straw-tip position + fixed flange-down orientation

        into:

            desired tool0/flange pose

        The returned pose is in the configured MoveIt base frame.
        """
        target_straw = np.asarray(straw_tip_position, dtype=float)
        tool_offset = np.asarray(
            flange_to_straw_tip or self.flange_to_straw_tip,
            dtype=float,
        )
        rpy = list(flange_down_rpy or self.flange_down_rpy)
        qx, qy, qz, qw = quaternion_from_euler(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        rotation = rotation_matrix_from_quaternion(qx, qy, qz, qw)
        target_tool0 = target_straw - rotation @ tool_offset
        return {
            "base_frame": self.node.base_frame,
            "tool_frame": self.node.tool_frame,
            "target_label": label,
            "target_straw_tip_position": target_straw.tolist(),
            "flange_to_straw_tip": tool_offset.tolist(),
            "flange_down_rpy": [float(x) for x in rpy],
            "tool0_target_position": target_tool0.tolist(),
            "tool0_target_orientation_quat": [qx, qy, qz, qw],
            "tool0_target_endpose": [
                float(target_tool0[0]),
                float(target_tool0[1]),
                float(target_tool0[2]),
                float(qx),
                float(qy),
                float(qz),
                float(qw),
            ],
        }

    def plan_straw_tip_to_pre_mouth_pose(
        self,
        pre_mouth_safe_position: Optional[Sequence[float]] = None,
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Compute the tool0 pose that puts the straw tip at pre-mouth."""
        target = pre_mouth_safe_position or self.pre_mouth_safe_position
        plan = self.plan_straw_tip_to_pose(
            target,
            flange_to_straw_tip=flange_to_straw_tip,
            flange_down_rpy=flange_down_rpy,
            label="pre_mouth_safe_position",
        )
        plan["pre_mouth_safe_position"] = plan["target_straw_tip_position"]
        return plan

    def move_straw_tip_to_position(
        self,
        straw_tip_position: Sequence[float],
        *,
        label: str = "straw_tip_target",
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
        plan_only: bool = False,
        planning_mode: str = "move_group",
    ) -> Dict[str, object]:
        """Move the flange-mounted straw tip to a target while keeping flange down."""
        plan = self.plan_straw_tip_to_pose(
            straw_tip_position,
            flange_to_straw_tip=flange_to_straw_tip,
            flange_down_rpy=flange_down_rpy,
            label=label,
        )
        if self.backend.is_real:
            # The real backend has no direct controller/joint command path;
            # retain the same constrained target but let MoveIt execute it.
            planning_mode = "move_group"
        if planning_mode == "move_group":
            move_result = self._move_group_to_pose(
                self._convert_endpose(plan["tool0_target_endpose"]),
                duration=duration,
                plan_only=plan_only,
            )
            planner_name = "moveit_move_group"
        elif planning_mode == "cartesian":
            move_result = self.move_to_pose(
                plan["tool0_target_endpose"],
                duration=duration,
                plan_only=plan_only,
            )
            planner_name = "moveit_cartesian_path"
        elif planning_mode == "ik_joint_target":
            move_result = self.move_to_pose_ik_joint_target(
                plan["tool0_target_endpose"],
                duration=duration,
                plan_only=plan_only,
            )
            planner_name = "moveit_ik_joint_target"
        else:
            raise ValueError("planning_mode must be 'move_group', 'cartesian', or 'ik_joint_target'")
        result = {
            "success": bool(move_result.get("success")),
            "stage": move_result.get("stage"),
            "planner": planner_name,
            "keep_flange_down": True,
            "plan": plan,
            "move_result": move_result,
        }
        if not plan_only and move_result.get("success"):
            result["end_pose"] = self.get_robot_end_pose()
        return result

    def move_straw_tip_to_pre_mouth(
        self,
        pre_mouth_safe_position: Optional[Sequence[float]] = None,
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
        plan_only: bool = False,
        planning_mode: str = "move_group",
    ) -> Dict[str, object]:
        """Move the flange-mounted straw tip to the pre-mouth target.

        This is the planner-backed feeding primitive that should replace the
        old greedy FSK search. The flange orientation is fixed by
        `flange_down_rpy`, so the cup/straw holder stays downward while MoveIt
        plans the actual robot motion.
        """
        target = pre_mouth_safe_position or self.pre_mouth_safe_position
        result = self.move_straw_tip_to_position(
            target,
            label="pre_mouth_safe_position",
            flange_to_straw_tip=flange_to_straw_tip,
            flange_down_rpy=flange_down_rpy,
            duration=duration,
            plan_only=plan_only,
            planning_mode=planning_mode,
        )
        result["plan"]["pre_mouth_safe_position"] = result["plan"]["target_straw_tip_position"]
        return result

    def move_straw_tip_to_mouth(
        self,
        mouth_target_position: Optional[Sequence[float]] = None,
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
        plan_only: bool = False,
    ) -> Dict[str, object]:
        """Move the flange-mounted straw tip to the configured mouth target."""
        target = mouth_target_position or self.mouth_target_position
        result = self.move_straw_tip_to_position(
            target,
            label="mouth_target_position",
            flange_to_straw_tip=flange_to_straw_tip,
            flange_down_rpy=flange_down_rpy,
            duration=duration,
            plan_only=plan_only,
        )
        result["plan"]["mouth_target_position"] = result["plan"]["target_straw_tip_position"]
        return result

    def retreat(
        self,
        ready_straw_tip_position: Optional[Sequence[float]] = None,
        flange_to_straw_tip: Optional[Sequence[float]] = None,
        flange_down_rpy: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
        plan_only: bool = False,
    ) -> Dict[str, object]:
        """Return the straw tip to the configured ready position."""
        target = ready_straw_tip_position or self.ready_straw_tip_position
        return self.move_straw_tip_to_position(
            target,
            label="ready_straw_tip_position",
            flange_to_straw_tip=flange_to_straw_tip,
            flange_down_rpy=flange_down_rpy,
            duration=duration,
            plan_only=plan_only,
        )

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        duration: Optional[float] = None,
        plan_only: bool = False,
    ) -> Dict[str, object]:
        """Move tool0 by a relative Cartesian offset in base_link."""
        current = self._current_pose_msg()
        target = Pose()
        target.position.x = current.position.x + float(dx)
        target.position.y = current.position.y + float(dy)
        target.position.z = current.position.z + float(dz)
        target.orientation = current.orientation
        return self.move_to_pose(
            [
                target.position.x,
                target.position.y,
                target.position.z,
                target.orientation.x,
                target.orientation.y,
                target.orientation.z,
                target.orientation.w,
            ],
            duration=duration,
            plan_only=plan_only,
        )

    def set_gripper(self, position, max_velocity=None, max_acceleration=None):
        """UR10e has no configured gripper in the current simulation."""
        return {"success": False, "unsupported": True, "reason": "no gripper configured"}

    def reset(self, max_velocity=None, max_acceleration=None, duration: Optional[float] = None):
        """Move to configured reset joint positions."""
        return self.move_joints(self.RESET_JOINTS, duration=duration or self.default_duration)

    @staticmethod
    def _camera_frame_from_msg(msg: Optional[Image], stream_type: str) -> Optional[CameraFrame]:
        if msg is None:
            return None
        stamp = msg.header.stamp
        return CameraFrame(
            device_id="wrist_rgbd",
            stream_type=stream_type,
            frame=bytes(msg.data),
            timestamp=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            width=int(msg.width),
            height=int(msg.height),
            depth_scale=1.0 if stream_type == "depth" else None,
            encoding=msg.encoding,
            frame_id=msg.header.frame_id,
        )

    def read_cameras(self) -> tuple[Optional[CameraFrame], Optional[CameraFrame]]:
        """Return the latest raw RGB and depth frames from the wrist camera."""
        self._spin(0.0)
        return (
            self._camera_frame_from_msg(self.node.latest_rgb_image, "rgb"),
            self._camera_frame_from_msg(self.node.latest_depth_image, "depth"),
        )

    def get_camera_info(self) -> dict:
        """Return latest wrist-camera calibration without an image payload."""
        self._spin(0.0)
        msg = self.node.latest_camera_info
        if msg is None:
            return {
                "available": False,
                "topic": self.node.camera_info_topic,
                "frame_id": self.camera_frame_id,
            }
        stamp = msg.header.stamp
        return {
            "available": True,
            "topic": self.node.camera_info_topic,
            "frame_id": msg.header.frame_id,
            "timestamp": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": msg.distortion_model,
            "d": list(msg.d),
            "k": list(msg.k),
            "r": list(msg.r),
            "p": list(msg.p),
        }

    def get_robot_state(self):
        """Return current joint state in a Piper-compatible dictionary."""
        # ``latest_joint_state`` is a callback-owned cache.  Spin before every
        # read so callers that poll for a completed trajectory do not keep
        # observing the velocity sample that arrived just before settling.
        self._spin(0.05)
        if self.node.latest_joint_state is None:
            self._spin_until_joint_state(timeout=2.0)
        msg = self.node.latest_joint_state
        if msg is None:
            return {
                "joint_positions": np.array(self.RESET_JOINTS, dtype=float),
                "joint_velocities": np.zeros(self.NUM_JOINTS),
                "gripper_position": np.array([]),
            }

        index = {name: i for i, name in enumerate(msg.name)}
        positions = []
        velocities = []
        for name in self.JOINT_NAMES:
            i = index.get(name)
            positions.append(float(msg.position[i]) if i is not None and i < len(msg.position) else 0.0)
            velocities.append(float(msg.velocity[i]) if i is not None and i < len(msg.velocity) else 0.0)

        return {
            "joint_positions": np.array(positions, dtype=float),
            "joint_velocities": np.array(velocities, dtype=float),
            "gripper_position": np.array([]),
        }

    def get_joint_state(self) -> Dict[str, object]:
        """Return a read-only, named view of the latest UR joint state."""
        self._spin(0.05)
        if self.node.latest_joint_state is None:
            self._spin_until_joint_state(timeout=2.0)
        msg = self.node.latest_joint_state
        if msg is None:
            return {
                "available": False,
                "topic": self.node.joint_state_topic,
                "joint_names": list(self.JOINT_NAMES),
                "reason": "no /joint_states message was received",
            }
        index = {name: i for i, name in enumerate(msg.name)}
        positions = {
            name: float(msg.position[i])
            for name, i in index.items()
            if name in self.JOINT_NAMES and i < len(msg.position)
        }
        velocities = {
            name: float(msg.velocity[i])
            for name, i in index.items()
            if name in self.JOINT_NAMES and i < len(msg.velocity)
        }
        return {
            "available": self._has_complete_joint_state(),
            "topic": self.node.joint_state_topic,
            "joint_names": list(self.JOINT_NAMES),
            "positions": positions,
            "velocities": velocities,
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
        }

    def get_robot_end_pose(self):
        """Return current tool0 pose from TF base_link -> tool0."""
        pose = self._current_pose_msg()
        q = pose.orientation
        return {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation_quat": [q.x, q.y, q.z, q.w],
            "orientation_euler": list(euler_from_quaternion(q.x, q.y, q.z, q.w)),
            "timestamp": time.time(),
            "frame_id": self.node.base_frame,
            "link_name": self.node.tool_frame,
        }

    def get_end_effector_pose(self):
        """Piper-style alias for the read-only ``base_link -> tool0`` pose."""
        return self.get_robot_end_pose()

    def get_straw_tip_pose(self):
        """Return the live TF pose of the tool-mounted straw-tip marker."""
        return self._frame_pose(self.straw_tip_frame_id)

    def get_tool_pose(self, tool: str = "straw_tip"):
        """Return only the reviewed fixed feeding tool frame."""
        if tool != "straw_tip":
            raise ValueError("the only configured tool is straw_tip")
        return self.get_straw_tip_pose()

    def get_backend_status(self) -> Dict[str, object]:
        """Read-only backend diagnostics used by physical-robot setup checks."""
        status: Dict[str, object] = dict(self.backend.status())
        status["joint_state"] = self.get_joint_state()
        try:
            self._current_pose_msg()
            status["base_to_tool0_tf_available"] = True
        except Exception as exc:
            status["base_to_tool0_tf_available"] = False
            status["tf_reason"] = str(exc)
        try:
            status["active_controllers"] = self._active_controller_names(timeout_sec=1.0)
        except Exception as exc:
            status["active_controllers"] = []
            status["controller_reason"] = str(exc)
        return status

    def get_mouth_target_pose(self):
        """Return the configured fixed mouth target in the base frame."""
        return {
            "position": list(self.mouth_target_position),
            "frame_id": self.node.base_frame,
            "link_name": "feeding_mouth_target",
            "timestamp": time.time(),
        }

    def get_observation(self):
        rgb_frame, depth_frame = self.read_cameras()
        return {
            "robot_state": self.get_robot_state(),
            "tool0_pose": self.get_robot_end_pose(),
            "straw_tip_pose": self.get_straw_tip_pose(),
            "mouth_target_pose": self.get_mouth_target_pose(),
            "camera": {
                "rgb": rgb_frame,
                "depth": depth_frame,
                "camera_info": self.get_camera_info(),
            },
            "timestamp": {
                "robot_state": time.time(),
                "tool0_pose": time.time(),
                "straw_tip_pose": time.time(),
                "mouth_target_pose": time.time(),
                "cameras": time.time(),
            },
        }

    def get_latest_decoded_frame(self, stream_type: str, device_id: str | None = None):
        rgb_frame, depth_frame = self.read_cameras()
        if stream_type in {"rgb", "color"}:
            return rgb_frame.frame if rgb_frame is not None else None
        if stream_type == "depth":
            return depth_frame.frame if depth_frame is not None else None
        raise ValueError("stream_type must be 'rgb', 'color', or 'depth'")

    def get_all_frames(self):
        rgb_frame, depth_frame = self.read_cameras()
        return {
            "rgb": rgb_frame,
            "depth": depth_frame,
            "camera_info": self.get_camera_info(),
        }

    def get_cameras(self):
        return ["wrist_rgbd"]

    def get_state(self):
        return {
            "robot_state": self.get_robot_state(),
            "tool0_pose": self.get_robot_end_pose(),
            "straw_tip_pose": self.get_straw_tip_pose(),
            "mouth_target_pose": self.get_mouth_target_pose(),
            "cameras": self.get_all_frames(),
            "timestamp": time.time(),
        }


# Backward-friendly alias if ABot-Claw code expects a robot env class name.
RobotEnv = UR10eRobotEnv
UR10eSDK = UR10eRobotEnv


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rotation_matrix_from_quaternion(qx: float, qy: float, qz: float, qw: float):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


if __name__ == "__main__":
    env = UR10eRobotEnv()
    try:
        print("[1] get_robot_state()")
        print(env.get_robot_state())
        print("[2] get_robot_end_pose()")
        try:
            print(env.get_robot_end_pose())
        except RuntimeError as exc:
            print({"success": False, "stage": "tf_lookup", "error": str(exc)})
        print("[3] plan-only relative move dz=0.05")
        try:
            print(env.move_relative(dz=0.05, plan_only=True))
        except RuntimeError as exc:
            print({"success": False, "stage": "moveit_planning", "error": str(exc)})
    finally:
        env.close()
