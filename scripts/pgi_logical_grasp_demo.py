#!/usr/bin/env python3
"""Plan and optionally execute one isolated PGI logical-grasp simulation.

The default mode is plan-only. Simulation execution requires both
``--execute-sim`` and ``--confirm-simulation`` and is refused on ROS domain 0.
The script never starts a driver and never calls a real-robot controller.

Stage 4 deliberately uses kinematic cup following plus a MoveIt attached
collision object. Contact forces, friction, and drop behaviour remain outside
this logical ownership test.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from builtin_interfaces.msg import Duration as DurationMessage
from control_msgs.action import FollowJointTrajectory, GripperCommand
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    MotionPlanResponse,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Limits come from the expanded UR10e description used by this simulation.
# MoveIt's IK service may return an equivalent angle shifted by 2*pi.  Keeping
# that representation would make a short Cartesian change look like a full
# joint revolution to a joint-goal planner.
ARM_JOINT_LIMITS = {
    "shoulder_pan_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "shoulder_lift_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "elbow_joint": (-math.pi, math.pi),
    "wrist_1_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "wrist_2_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "wrist_3_joint": (-2.0 * math.pi, 2.0 * math.pi),
}
LEFT_JAW = "pgi_left_finger_joint"
RIGHT_JAW = "pgi_right_finger_joint"
ARM_CONTROLLER = "joint_trajectory_controller"
GRIPPER_CONTROLLER = "pgi_gripper_controller"
SUCCESS = MoveItErrorCodes.SUCCESS


def duration_message(seconds: float) -> DurationMessage:
    whole = int(seconds)
    return DurationMessage(sec=whole, nanosec=int((seconds - whole) * 1e9))


def identity_pose() -> Pose:
    pose = Pose()
    pose.orientation.w = 1.0
    return pose


def pose_matrix(pose: Pose) -> np.ndarray:
    matrix = np.eye(4)
    quaternion = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    if np.linalg.norm(quaternion) < 1e-9:
        quaternion = [0.0, 0.0, 0.0, 1.0]
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def matrix_pose(matrix: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[:3, 3]
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = quaternion
    return pose


def stamped_pose(frame_id: str, position: np.ndarray, quaternion: np.ndarray) -> PoseStamped:
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.pose.position.x, message.pose.position.y, message.pose.position.z = position
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = quaternion
    return message


class LogicalGraspDemo(Node):
    def __init__(
        self,
        node_name: str = "pgi_logical_grasp_demo",
        status_topic: str = "/pgi/logical_grasp/status",
    ) -> None:
        super().__init__(node_name)
        defaults = {
            "cup_grasp_pose_topic": "/pgi/perception/cup_grasp_pose",
            "cup_model_pose_topic": "/model/pgi_staging_cup/pose",
            "planning_frame": "base_link",
            "grasp_link": "pgi_grasp_center",
            "cup_id": "pgi_staging_cup",
            "cup_model_name": "pgi_staging_cup",
            "gazebo_set_pose_service": "/world/pgi_140_80_camera/set_pose",
            "approach_down_angle_deg": 15.0,
            "pregrasp_backoff_m": 0.070,
            "grasp_backoff_m": 0.040,
            "staging_lift_m": 0.080,
            "side_ready_backoff_m": 0.120,
            "side_ready_height_m": 0.300,
            "cartesian_transfer_height_m": 0.450,
            # Simulation-only option. When enabled, the cup-clear transfer
            # legs use collision-checked Pilz joint PTP motion, so no Cartesian
            # waypoint locks the flange orientation during free-space travel.
            # Exact grasp orientation remains mandatory from side-ready onward.
            "relax_transit_flange_orientation": False,
            # Optional bounded spin about the gripper's local +Z at the high
            # transfer point. MoveIt chooses the joint motion; wrist_3 is never
            # commanded directly. A local-Z spin preserves the approach axis.
            "transit_flange_spin_deg": 0.0,
            "side_ready_joints_rad": [
                -1.2903761544653078,
                -0.8867852166149196,
                -2.6924239017277234,
                -3.8426568578501117,
                -0.2890242292195187,
                1.122349596657656,
            ],
            "lift_height_m": 0.120,
            "release_height_m": 0.0,
            "jaw_open_m": 0.040,
            "jaw_logical_hold_m": 0.0393,
            "jaw_max_effort_n": 20.0,
            "arm_velocity_scaling": 0.10,
            "arm_acceleration_scaling": 0.10,
            "cartesian_max_step_m": 0.005,
            "cartesian_revolute_jump_rad": 0.35,
            "target_max_age_s": 2.0,
            "target_model_xy_tolerance_m": 0.030,
            "joint_final_tolerance_rad": 0.015,
            "cup_follow_rate_hz": 20.0,
            "hold_duration_s": 2.0,
            "execution_timeout_scale": 3.0,
            "execution_timeout_margin_s": 10.0,
            "touch_links": [
                "pgi_left_finger",
                "pgi_right_finger",
                "pgi_body",
                "pgi_grasp_center",
            ],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.planning_frame = str(self.get_parameter("planning_frame").value)
        self.grasp_link = str(self.get_parameter("grasp_link").value)
        self.cup_id = str(self.get_parameter("cup_id").value)
        self.cup_model_name = str(self.get_parameter("cup_model_name").value)
        self.touch_links = list(self.get_parameter("touch_links").value)

        self.joint_state: JointState | None = None
        self.first_clock_ns: int | None = None
        self.clock_advanced = False
        self.cup_target: PoseStamped | None = None
        self.cup_target_wall_time = 0.0
        self.cup_model_pose: Pose | None = None
        self.cup_model_pose_wall_time = 0.0
        self.cup_follow_relative: np.ndarray | None = None
        self.last_followed_cup_pose: Pose | None = None

        self.create_subscription(JointState, "/joint_states", self._joint_callback, 20)
        self.create_subscription(Clock, "/clock", self._clock_callback, 20)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("cup_grasp_pose_topic").value),
            self._cup_target_callback,
            10,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("cup_model_pose_topic").value),
            self._cup_model_pose_callback,
            10,
        )
        self.status_publisher = self.create_publisher(String, status_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.parameter_client = AsyncParameterClient(self, "/move_group")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.motion_plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path"
        )
        self.validity_client = self.create_client(
            GetStateValidity, "/check_state_validity"
        )
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.list_controllers_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.switch_controller_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self.set_pose_client = self.create_client(
            SetEntityPose,
            str(self.get_parameter("gazebo_set_pose_service").value),
        )
        self.arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            f"/{ARM_CONTROLLER}/follow_joint_trajectory",
        )
        self.gripper_action = ActionClient(
            self,
            GripperCommand,
            f"/{GRIPPER_CONTROLLER}/gripper_cmd",
        )

    def _joint_callback(self, message: JointState) -> None:
        required = set(ARM_JOINTS + [LEFT_JAW, RIGHT_JAW])
        if required <= set(message.name):
            self.joint_state = message

    def _clock_callback(self, message: Clock) -> None:
        clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self.first_clock_ns is None:
            self.first_clock_ns = clock_ns
        elif clock_ns > self.first_clock_ns:
            self.clock_advanced = True

    def _cup_target_callback(self, message: PoseStamped) -> None:
        self.cup_target = message
        self.cup_target_wall_time = time.monotonic()

    def _cup_model_pose_callback(self, message: Pose) -> None:
        self.cup_model_pose = message
        self.cup_model_pose_wall_time = time.monotonic()

    def publish_status(self, stage: str, **details) -> None:
        payload = {"stage": stage, **details}
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.status_publisher.publish(message)
        self.get_logger().info(message.data)

    def wait_future(self, future, timeout: float, label: str):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Timed out waiting for {label}")
        return future.result()

    def call(self, client, request, timeout: float = 10.0):
        return self.wait_future(client.call_async(request), timeout, client.srv_name)

    def wait_for_inputs(self, execute: bool, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            target_fresh = (
                self.cup_target is not None
                and time.monotonic() - self.cup_target_wall_time
                <= float(self.get_parameter("target_max_age_s").value)
            )
            model_ready = self.cup_model_pose is not None or not execute
            if (
                self.joint_state is not None
                and self.clock_advanced
                and target_fresh
                and model_ready
            ):
                return
        missing = {
            "joint_state": self.joint_state is None,
            "advancing_clock": not self.clock_advanced,
            "fresh_cup_target": self.cup_target is None
            or time.monotonic() - self.cup_target_wall_time
            > float(self.get_parameter("target_max_age_s").value),
            "cup_model_pose": execute and self.cup_model_pose is None,
        }
        raise RuntimeError(f"Simulation inputs unavailable: {missing}")

    def wait_for_services(self, execute: bool) -> None:
        services = [
            (self.ik_client, "IK"),
            (self.cartesian_client, "Cartesian planning"),
            (self.motion_plan_client, "MoveIt motion planning"),
            (self.validity_client, "state validity"),
            (self.scene_client, "planning scene read"),
            (self.apply_scene_client, "planning scene write"),
            (self.list_controllers_client, "controller list"),
        ]
        if execute:
            services.extend(
                [
                    (self.switch_controller_client, "controller switch"),
                    (self.set_pose_client, "Gazebo set pose"),
                ]
            )
        for client, label in services:
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"{label} service is unavailable")
        if not self.parameter_client.wait_for_services(timeout_sec=10.0):
            raise RuntimeError("MoveIt parameter service is unavailable")
        if execute:
            if not self.arm_action.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("Simulation arm action is unavailable")
            if not self.gripper_action.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("Simulation gripper action is unavailable")

    def controller_states(self) -> dict[str, dict[str, str]]:
        response = self.call(self.list_controllers_client, ListControllers.Request())
        return {
            controller.name: {"state": controller.state, "type": controller.type}
            for controller in response.controller
        }

    def verify_guards(self, execute: bool) -> dict:
        domain = os.environ.get("ROS_DOMAIN_ID", "0")
        try:
            domain_id = int(domain)
        except ValueError as error:
            raise RuntimeError(f"Invalid ROS_DOMAIN_ID={domain!r}") from error
        if domain_id == 0:
            raise RuntimeError("Refusing ROS domain 0; this runner is simulation-only")
        if not self.clock_advanced:
            raise RuntimeError("Gazebo clock is not advancing")

        future = self.parameter_client.get_parameters(["allow_trajectory_execution"])
        response = self.wait_future(future, 10.0, "MoveIt parameters")
        allow_execution = parameter_value_to_python(response.values[0])
        if allow_execution is not False:
            raise RuntimeError(
                "MoveGroup must remain plan-only; this runner owns the isolated sim action"
            )

        controllers = self.controller_states()
        expected_types = {
            ARM_CONTROLLER: "joint_trajectory_controller/JointTrajectoryController",
            GRIPPER_CONTROLLER: "position_controllers/GripperActionController",
        }
        for name, expected_type in expected_types.items():
            actual = controllers.get(name)
            if actual is None or actual["type"] != expected_type:
                raise RuntimeError(f"Unexpected simulation controller {name}: {actual}")
        if controllers[GRIPPER_CONTROLLER]["state"] != "active":
            raise RuntimeError("PGI gripper controller is not active")

        world_to_base = self.lookup_matrix("world", self.planning_frame)
        translation_error = float(np.linalg.norm(world_to_base[:3, 3]))
        rotation_error = float(Rotation.from_matrix(world_to_base[:3, :3]).magnitude())
        if translation_error > 1e-3 or rotation_error > math.radians(0.1):
            raise RuntimeError(
                "Gazebo world and MoveIt base_link are not aligned; refusing pose following"
            )
        return {
            "ros_domain_id": domain_id,
            "move_group_plan_only": True,
            "controllers": controllers,
            "world_base_translation_error_m": translation_error,
            "world_base_rotation_error_deg": math.degrees(rotation_error),
            "execution_requested": execute,
        }

    def lookup_matrix(self, target: str, source: str, timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                transform = self.tf_buffer.lookup_transform(
                    target, source, Time(), timeout=Duration(seconds=0.1)
                )
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation = transform.transform.rotation
                return pose_matrix(pose)
            except TransformException as error:
                last_error = str(error)
        raise RuntimeError(f"TF {target} <- {source} unavailable: {last_error}")

    def live_state(self) -> RobotState:
        if self.joint_state is None:
            raise RuntimeError("Complete joint state is unavailable")
        state = RobotState()
        state.joint_state = deepcopy(self.joint_state)
        state.is_diff = False
        return state

    def get_cup_object(self) -> CollisionObject:
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        scene = self.call(self.scene_client, request).scene
        attached_ids = {
            item.object.id for item in scene.robot_state.attached_collision_objects
        }
        if self.cup_id in attached_ids:
            raise RuntimeError("Cup is already attached before this workflow")
        cup = next(
            (item for item in scene.world.collision_objects if item.id == self.cup_id),
            None,
        )
        if cup is None:
            raise RuntimeError("Cup collision object is missing from MoveIt")
        return deepcopy(cup)

    def state_validity(self, state: RobotState) -> dict:
        request = GetStateValidity.Request()
        request.robot_state = state
        request.group_name = "ur_manipulator"
        response = self.call(self.validity_client, request)
        contacts = sorted(
            {
                tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
                for contact in response.contacts
            }
        )
        return {"valid": bool(response.valid), "contacts": contacts}

    def begin_grasp_contact_planning(self) -> None:
        """Hook for contact-aware simulations; Stage 4 changes no ACM state."""

    def end_grasp_contact_planning(self) -> None:
        """Undo any contact-aware planning override; Stage 4 is a no-op."""

    def ik_pose(
        self, pose: PoseStamped, seed: RobotState, avoid_collisions: bool
    ) -> tuple[RobotState | None, int]:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "ur_manipulator"
        ik.robot_state = deepcopy(seed)
        ik.avoid_collisions = avoid_collisions
        ik.ik_link_name = self.grasp_link
        ik.pose_stamped = pose
        ik.timeout = duration_message(2.0)
        response = self.call(self.ik_client, request)
        if response.error_code.val != SUCCESS:
            return None, response.error_code.val
        return self.normalize_ik_solution(response.solution, seed), response.error_code.val

    def normalize_ik_solution(
        self, solution: RobotState, seed: RobotState
    ) -> RobotState:
        """Choose the in-limit 2*pi equivalent nearest to the IK seed."""
        normalized = deepcopy(solution)
        seed_map = dict(
            zip(seed.joint_state.name, seed.joint_state.position, strict=True)
        )
        positions = list(normalized.joint_state.position)
        changes = {}
        for index, name in enumerate(normalized.joint_state.name):
            if name not in ARM_JOINT_LIMITS or name not in seed_map:
                continue
            lower, upper = ARM_JOINT_LIMITS[name]
            raw = positions[index]
            candidates = [
                raw + turns * 2.0 * math.pi
                for turns in range(-2, 3)
                if lower - 1e-9 <= raw + turns * 2.0 * math.pi <= upper + 1e-9
            ]
            if not candidates:
                raise RuntimeError(
                    f"IK returned out-of-limit {name}={raw:.6f} rad"
                )
            selected = min(candidates, key=lambda value: abs(value - seed_map[name]))
            positions[index] = selected
            if abs(selected - raw) > 1e-6:
                changes[name] = {"raw": raw, "normalized": selected}
        normalized.joint_state.position = positions
        if changes:
            self.publish_status("ik_angle_normalized", joints=changes)
        return normalized

    def state_after_trajectory(self, start: RobotState, trajectory) -> RobotState:
        points = trajectory.joint_trajectory.points
        if not points:
            raise RuntimeError("Planned trajectory contains no points")
        final_map = dict(
            zip(
                trajectory.joint_trajectory.joint_names,
                points[-1].positions,
                strict=True,
            )
        )
        result = deepcopy(start)
        positions = list(result.joint_state.position)
        for index, name in enumerate(result.joint_state.name):
            if name in final_map:
                positions[index] = final_map[name]
        result.joint_state.position = positions
        result.joint_state.velocity = []
        result.joint_state.effort = []
        return result

    def state_with_arm_positions(
        self, seed: RobotState, positions: list[float]
    ) -> RobotState:
        if len(positions) != len(ARM_JOINTS):
            raise RuntimeError(
                "side_ready_joints_rad must contain exactly six values"
            )
        target = dict(zip(ARM_JOINTS, positions, strict=True))
        result = deepcopy(seed)
        values = list(result.joint_state.position)
        present = set(result.joint_state.name)
        missing = set(ARM_JOINTS) - present
        if missing:
            raise RuntimeError(f"Arm state is missing joints: {sorted(missing)}")
        for index, name in enumerate(result.joint_state.name):
            if name in target:
                values[index] = target[name]
        result.joint_state.position = values
        result.joint_state.velocity = []
        result.joint_state.effort = []
        return result

    def plan_pilz_ptp(self, start: RobotState, goal: RobotState):
        goal_map = dict(
            zip(goal.joint_state.name, goal.joint_state.position, strict=True)
        )
        constraints = Constraints()
        for name in ARM_JOINTS:
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(goal_map[name])
            joint.tolerance_above = 1e-4
            joint.tolerance_below = 1e-4
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)

        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "ur_manipulator"
        motion.pipeline_id = "pilz_industrial_motion_planner"
        motion.planner_id = "PTP"
        motion.start_state = deepcopy(start)
        motion.goal_constraints = [constraints]
        motion.allowed_planning_time = 5.0
        motion.num_planning_attempts = 1
        motion.max_velocity_scaling_factor = float(
            self.get_parameter("arm_velocity_scaling").value
        )
        motion.max_acceleration_scaling_factor = float(
            self.get_parameter("arm_acceleration_scaling").value
        )
        response = self.call(self.motion_plan_client, request, timeout=10.0)
        plan = response.motion_plan_response
        if (
            plan.error_code.val != SUCCESS
            or not plan.trajectory.joint_trajectory.points
        ):
            raise RuntimeError(
                "Pilz PTP planning failed: "
                f"code={plan.error_code.val}, message={plan.error_code.message}"
            )
        return plan

    def plan_cartesian(self, start: RobotState, waypoints: list[PoseStamped]):
        if not waypoints:
            raise RuntimeError("Cartesian plan requires at least one waypoint")
        for waypoint in waypoints:
            if waypoint.header.frame_id != self.planning_frame:
                raise RuntimeError(
                    f"Cartesian waypoint must be in {self.planning_frame}"
                )
        request = GetCartesianPath.Request()
        request.header.frame_id = self.planning_frame
        request.start_state = deepcopy(start)
        request.group_name = "ur_manipulator"
        request.link_name = self.grasp_link
        request.waypoints = [deepcopy(item.pose) for item in waypoints]
        request.max_step = float(
            self.get_parameter("cartesian_max_step_m").value
        )
        request.jump_threshold = 0.0
        request.prismatic_jump_threshold = 0.05
        request.revolute_jump_threshold = float(
            self.get_parameter("cartesian_revolute_jump_rad").value
        )
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = float(
            self.get_parameter("arm_velocity_scaling").value
        )
        request.max_acceleration_scaling_factor = float(
            self.get_parameter("arm_acceleration_scaling").value
        )
        started = time.monotonic()
        response = self.call(self.cartesian_client, request, timeout=20.0)
        if (
            response.error_code.val != SUCCESS
            or response.fraction < 0.999
            or not response.solution.joint_trajectory.points
        ):
            raise RuntimeError(
                "Cartesian planning incomplete: "
                f"code={response.error_code.val}, fraction={response.fraction:.6f}"
            )
        plan = MotionPlanResponse()
        plan.trajectory = response.solution
        plan.planning_time = time.monotonic() - started
        plan.error_code = response.error_code
        return plan

    def candidate_poses(self, cup_target: PoseStamped) -> dict:
        if cup_target.header.frame_id != self.planning_frame:
            raise RuntimeError(
                f"Cup target must be in {self.planning_frame}, got {cup_target.header.frame_id}"
            )
        cup = np.array(
            [
                cup_target.pose.position.x,
                cup_target.pose.position.y,
                cup_target.pose.position.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(cup)) or np.linalg.norm(cup[:2]) < 0.1:
            raise RuntimeError(f"Invalid cup target: {cup.tolist()}")

        azimuth = math.atan2(cup[1], cup[0])
        elevation = math.radians(
            float(self.get_parameter("approach_down_angle_deg").value)
        )
        radial = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        local_x = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        local_z = np.array(
            [
                math.cos(elevation) * radial[0],
                math.cos(elevation) * radial[1],
                -math.sin(elevation),
            ]
        )
        local_y = np.cross(local_z, local_x)
        quaternion = Rotation.from_matrix(
            np.column_stack((local_x, local_y, local_z))
        ).as_quat()

        pregrasp = cup - float(self.get_parameter("pregrasp_backoff_m").value) * local_z
        grasp = cup - float(self.get_parameter("grasp_backoff_m").value) * local_z
        staging = pregrasp + np.array(
            [0.0, 0.0, float(self.get_parameter("staging_lift_m").value)]
        )
        side_ready = np.array(
            [
                cup[0]
                - float(self.get_parameter("side_ready_backoff_m").value)
                * local_z[0],
                cup[1]
                - float(self.get_parameter("side_ready_backoff_m").value)
                * local_z[1],
                float(self.get_parameter("side_ready_height_m").value),
            ]
        )
        transfer = side_ready.copy()
        transfer[2] = float(
            self.get_parameter("cartesian_transfer_height_m").value
        )
        lift = grasp + np.array(
            [0.0, 0.0, float(self.get_parameter("lift_height_m").value)]
        )
        release = grasp + np.array(
            [0.0, 0.0, float(self.get_parameter("release_height_m").value)]
        )
        return {
            "cup": cup,
            "axis": local_z,
            "quaternion": quaternion,
            "side_ready": stamped_pose(
                self.planning_frame, side_ready, quaternion
            ),
            "transfer": stamped_pose(
                self.planning_frame, transfer, quaternion
            ),
            "staging": stamped_pose(self.planning_frame, staging, quaternion),
            "pregrasp": stamped_pose(self.planning_frame, pregrasp, quaternion),
            "grasp": stamped_pose(self.planning_frame, grasp, quaternion),
            "lift": stamped_pose(self.planning_frame, lift, quaternion),
            "release": stamped_pose(self.planning_frame, release, quaternion),
            "approach_azimuth_deg": math.degrees(azimuth),
        }

    def evaluate_top_grasp(self, seed: RobotState, cup_target: PoseStamped) -> dict:
        current_grasp = self.lookup_matrix(self.planning_frame, self.grasp_link)
        quaternion = Rotation.from_matrix(current_grasp[:3, :3]).as_quat()
        position = np.array(
            [
                cup_target.pose.position.x,
                cup_target.pose.position.y,
                cup_target.pose.position.z,
            ]
        )
        pose = stamped_pose(self.planning_frame, position, quaternion)
        solution, code = self.ik_pose(pose, seed, avoid_collisions=False)
        validity = self.state_validity(solution) if solution is not None else None
        return {
            "feasible": False,
            "rim_diameter_m": 0.120,
            "maximum_opening_m": 0.080,
            "rim_exceeds_opening_m": 0.040,
            "ik_without_collision_check": code == SUCCESS,
            "target_state": validity,
            "reason": (
                "The rim cannot pass through the jaws and the lower-band pose "
                "collides with the cup/tool body."
            ),
        }

    def attached_object(
        self, world_object: CollisionObject, base_to_grasp: np.ndarray
    ) -> AttachedCollisionObject:
        attached = AttachedCollisionObject()
        attached.link_name = self.grasp_link
        attached.touch_links = self.touch_links
        attached.weight = 0.15
        attached.object = deepcopy(world_object)
        attached.object.header.frame_id = self.grasp_link
        attached.object.pose = identity_pose()
        grasp_to_base = np.linalg.inv(base_to_grasp)
        attached.object.primitive_poses = [
            matrix_pose(grasp_to_base @ pose_matrix(pose))
            for pose in world_object.primitive_poses
        ]
        attached.object.mesh_poses = [
            matrix_pose(grasp_to_base @ pose_matrix(pose))
            for pose in world_object.mesh_poses
        ]
        attached.object.operation = CollisionObject.ADD
        return attached

    def apply_attach(
        self, world_object: CollisionObject, base_to_grasp: np.ndarray
    ) -> AttachedCollisionObject:
        attached = self.attached_object(world_object, base_to_grasp)
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)
        request = ApplyPlanningScene.Request()
        request.scene = scene
        if not self.call(self.apply_scene_client, request).success:
            raise RuntimeError("MoveIt rejected cup attach")
        return attached

    def relocated_world_object(
        self,
        original: CollisionObject,
        initial_model_pose: Pose,
        final_model_pose: Pose,
    ) -> CollisionObject:
        relocated = deepcopy(original)
        initial_inverse = np.linalg.inv(pose_matrix(initial_model_pose))
        final_matrix = pose_matrix(final_model_pose)
        relocated.header.frame_id = self.planning_frame
        relocated.pose = identity_pose()
        relocated.primitive_poses = [
            matrix_pose(final_matrix @ initial_inverse @ pose_matrix(pose))
            for pose in original.primitive_poses
        ]
        relocated.mesh_poses = [
            matrix_pose(final_matrix @ initial_inverse @ pose_matrix(pose))
            for pose in original.mesh_poses
        ]
        relocated.operation = CollisionObject.ADD
        return relocated

    def apply_detach(self, world_object: CollisionObject) -> None:
        remove = AttachedCollisionObject()
        remove.link_name = self.grasp_link
        remove.object.id = self.cup_id
        remove.object.operation = CollisionObject.REMOVE

        detach_scene = PlanningScene()
        detach_scene.is_diff = True
        detach_scene.robot_state.is_diff = True
        detach_scene.robot_state.attached_collision_objects.append(remove)
        detach_request = ApplyPlanningScene.Request()
        detach_request.scene = detach_scene
        if not self.call(self.apply_scene_client, detach_request).success:
            raise RuntimeError("MoveIt rejected cup detach")

        world_object = deepcopy(world_object)
        world_object.operation = CollisionObject.ADD
        world_scene = PlanningScene()
        world_scene.is_diff = True
        world_scene.robot_state.is_diff = True
        world_scene.world.collision_objects.append(world_object)
        world_request = ApplyPlanningScene.Request()
        world_request.scene = world_scene
        if not self.call(self.apply_scene_client, world_request).success:
            raise RuntimeError("MoveIt rejected cup world restore after detach")

    def verify_scene_ownership(self, attached_expected: bool) -> None:
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        scene = self.call(self.scene_client, request).scene
        in_world = any(item.id == self.cup_id for item in scene.world.collision_objects)
        attached = any(
            item.object.id == self.cup_id
            for item in scene.robot_state.attached_collision_objects
        )
        if attached != attached_expected or in_world == attached_expected:
            raise RuntimeError(
                f"Unexpected cup ownership: attached={attached}, in_world={in_world}"
            )

    def preflight(self, cup_target: PoseStamped, cup_object: CollisionObject) -> dict:
        poses = self.candidate_poses(cup_target)
        live = self.live_state()
        top = self.evaluate_top_grasp(live, cup_target)
        relax_transit_flange = bool(
            self.get_parameter("relax_transit_flange_orientation").value
        )
        transit_flange_spin_deg = float(
            self.get_parameter("transit_flange_spin_deg").value
        )
        if not math.isfinite(transit_flange_spin_deg):
            raise RuntimeError("transit_flange_spin_deg must be finite")
        if abs(transit_flange_spin_deg) > 45.0:
            raise RuntimeError(
                "transit_flange_spin_deg exceeds the simulation-only 45 deg bound"
            )
        if not relax_transit_flange and abs(transit_flange_spin_deg) > 1e-9:
            raise RuntimeError(
                "transit_flange_spin_deg requires relax_transit_flange_orientation"
            )

        if abs(transit_flange_spin_deg) > 1e-9:
            transfer_orientation = poses["transfer"].pose.orientation
            transfer_rotation = Rotation.from_quat(
                [
                    transfer_orientation.x,
                    transfer_orientation.y,
                    transfer_orientation.z,
                    transfer_orientation.w,
                ]
            ) * Rotation.from_euler("z", transit_flange_spin_deg, degrees=True)
            (
                transfer_orientation.x,
                transfer_orientation.y,
                transfer_orientation.z,
                transfer_orientation.w,
            ) = transfer_rotation.as_quat()

        actual_start_matrix = self.lookup_matrix(
            self.planning_frame, self.grasp_link
        )
        side_branch = self.state_with_arm_positions(
            live,
            list(self.get_parameter("side_ready_joints_rad").value),
        )
        side_branch_validity = self.state_validity(side_branch)
        if not side_branch_validity["valid"]:
            raise RuntimeError(
                "Configured side-ready IK branch is in collision: "
                f"{side_branch_validity['contacts']}"
            )

        # Build the high transfer goal by moving upward from the already
        # validated low-side IK branch. Pilz PTP then changes branches at this
        # high, cup-clear pose; all task-space approach motion stays Cartesian.
        self.publish_status("deriving_transfer_on_side_branch")
        side_to_transfer_seed = self.plan_cartesian(
            side_branch, [poses["transfer"]]
        )
        transfer_branch = self.state_after_trajectory(
            side_branch, side_to_transfer_seed.trajectory
        )

        self.publish_status("planning_camera_ready_to_transfer_ptp")
        transfer_plan = self.plan_pilz_ptp(live, transfer_branch)
        transfer_state = self.state_after_trajectory(
            live, transfer_plan.trajectory
        )

        if relax_transit_flange:
            self.publish_status(
                "planning_transfer_to_side_ready_ptp",
                transit_flange_orientation_relaxed=True,
            )
            ready_plan = self.plan_pilz_ptp(transfer_state, side_branch)
        else:
            self.publish_status("planning_transfer_to_side_ready")
            ready_plan = self.plan_cartesian(
                transfer_state, [poses["side_ready"]]
            )
        ready_state = self.state_after_trajectory(
            transfer_state, ready_plan.trajectory
        )

        self.publish_status("planning_side_ready_to_staging")
        staging_plan = self.plan_cartesian(ready_state, [poses["staging"]])
        staging_state = self.state_after_trajectory(
            ready_state, staging_plan.trajectory
        )

        self.publish_status("planning_staging_to_pregrasp")
        pre_plan = self.plan_cartesian(staging_state, [poses["pregrasp"]])
        pre_state = self.state_after_trajectory(
            staging_state, pre_plan.trajectory
        )

        self.publish_status("planning_pregrasp_to_grasp")
        self.begin_grasp_contact_planning()
        try:
            grasp_plan = self.plan_cartesian(pre_state, [poses["grasp"]])
        finally:
            self.end_grasp_contact_planning()
        grasp_state = self.state_after_trajectory(pre_state, grasp_plan.trajectory)

        grasp_matrix = pose_matrix(poses["grasp"].pose)
        attached = False
        try:
            attached_object = self.apply_attach(cup_object, grasp_matrix)
            attached = True
            self.verify_scene_ownership(True)
            self.publish_status("planning_attached_lift_place")
            grasp_attached_state = deepcopy(grasp_state)
            grasp_attached_state.attached_collision_objects = [attached_object]
            lift_plan = self.plan_cartesian(
                grasp_attached_state, [poses["lift"]]
            )
            lift_state = self.state_after_trajectory(
                grasp_attached_state, lift_plan.trajectory
            )
            place_plan = self.plan_cartesian(lift_state, [poses["release"]])
            release_state = self.state_after_trajectory(
                lift_state, place_plan.trajectory
            )
            release_detached_state = deepcopy(release_state)
            release_detached_state.attached_collision_objects = []
        finally:
            if attached:
                self.apply_detach(cup_object)
                self.verify_scene_ownership(False)

        self.publish_status("planning_retreat_and_return")
        retreat_plan = self.plan_cartesian(
            release_detached_state, [poses["pregrasp"]]
        )
        retreat_state = self.state_after_trajectory(
            release_detached_state, retreat_plan.trajectory
        )
        unstage_plan = self.plan_cartesian(retreat_state, [poses["staging"]])
        unstage_state = self.state_after_trajectory(
            retreat_state, unstage_plan.trajectory
        )
        ready_return_plan = self.plan_cartesian(
            unstage_state, [poses["side_ready"]]
        )
        ready_return_state = self.state_after_trajectory(
            unstage_state, ready_return_plan.trajectory
        )
        if relax_transit_flange:
            transfer_return_plan = self.plan_pilz_ptp(
                ready_return_state, transfer_branch
            )
        else:
            transfer_return_plan = self.plan_cartesian(
                ready_return_state, [poses["transfer"]]
            )
        transfer_return_state = self.state_after_trajectory(
            ready_return_state, transfer_return_plan.trajectory
        )
        return_plan = self.plan_pilz_ptp(transfer_return_state, live)

        def plan_summary(response, backend: str) -> dict:
            points = response.trajectory.joint_trajectory.points
            final = points[-1].time_from_start
            duration = final.sec + final.nanosec / 1e9
            summary = {
                "backend": backend,
                "planning_time_s": response.planning_time,
                "trajectory_points": len(points),
                "trajectory_duration_s": duration,
            }
            names = list(response.trajectory.joint_trajectory.joint_names)
            if "wrist_3_joint" in names and points:
                index = names.index("wrist_3_joint")
                start_position = points[0].positions[index]
                wrist_delta = points[-1].positions[index] - start_position
                wrist_excursion = max(
                    abs(point.positions[index] - start_position) for point in points
                )
                summary["wrist_3_delta_deg"] = math.degrees(wrist_delta)
                summary["wrist_3_max_excursion_deg"] = math.degrees(
                    wrist_excursion
                )
            return summary

        return {
            "top_grasp": top,
            "selected_strategy": {
                "name": "oblique_radial_side_grasp",
                "approach_down_angle_deg": float(
                    self.get_parameter("approach_down_angle_deg").value
                ),
                "approach_azimuth_deg": poses["approach_azimuth_deg"],
                "approach_axis_base": poses["axis"].tolist(),
                "transit_flange_orientation_relaxed": relax_transit_flange,
                "transit_flange_spin_deg": transit_flange_spin_deg,
                "grasp_orientation_locked": True,
                "camera_ready_xyz": actual_start_matrix[:3, 3].tolist(),
                "side_ready_xyz": [
                    poses["side_ready"].pose.position.x,
                    poses["side_ready"].pose.position.y,
                    poses["side_ready"].pose.position.z,
                ],
                "transfer_xyz": [
                    poses["transfer"].pose.position.x,
                    poses["transfer"].pose.position.y,
                    poses["transfer"].pose.position.z,
                ],
                "staging_xyz": [
                    poses["staging"].pose.position.x,
                    poses["staging"].pose.position.y,
                    poses["staging"].pose.position.z,
                ],
                "pregrasp_xyz": [
                    poses["pregrasp"].pose.position.x,
                    poses["pregrasp"].pose.position.y,
                    poses["pregrasp"].pose.position.z,
                ],
                "grasp_xyz": [
                    poses["grasp"].pose.position.x,
                    poses["grasp"].pose.position.y,
                    poses["grasp"].pose.position.z,
                ],
                "lift_xyz": [
                    poses["lift"].pose.position.x,
                    poses["lift"].pose.position.y,
                    poses["lift"].pose.position.z,
                ],
                "release_xyz": [
                    poses["release"].pose.position.x,
                    poses["release"].pose.position.y,
                    poses["release"].pose.position.z,
                ],
            },
            "plans": {
                "side_branch_to_transfer_seed": plan_summary(
                    side_to_transfer_seed, "cartesian_validation_only"
                ),
                "camera_ready_to_transfer": plan_summary(
                    transfer_plan, "pilz_ptp"
                ),
                "transfer_to_side_ready": plan_summary(
                    ready_plan,
                    "pilz_ptp_flange_relaxed"
                    if relax_transit_flange
                    else "cartesian",
                ),
                "side_ready_to_staging": plan_summary(
                    staging_plan, "cartesian"
                ),
                "staging_to_pregrasp": plan_summary(pre_plan, "cartesian"),
                "pregrasp_to_grasp": plan_summary(grasp_plan, "cartesian"),
                "attached_lift": plan_summary(lift_plan, "cartesian"),
                "attached_place": plan_summary(place_plan, "cartesian"),
                "retreat": plan_summary(retreat_plan, "cartesian"),
                "unstage": plan_summary(unstage_plan, "cartesian"),
                "staging_to_side_ready": plan_summary(
                    ready_return_plan, "cartesian"
                ),
                "side_ready_to_transfer": plan_summary(
                    transfer_return_plan,
                    "pilz_ptp_flange_relaxed"
                    if relax_transit_flange
                    else "cartesian",
                ),
                "transfer_to_camera_ready": plan_summary(
                    return_plan, "pilz_ptp"
                ),
            },
            "poses": poses,
            "transfer_plan": transfer_plan,
            "ready_plan": ready_plan,
            "staging_plan": staging_plan,
            "pre_plan": pre_plan,
            "grasp_plan": grasp_plan,
            "lift_plan": lift_plan,
            "place_plan": place_plan,
            "retreat_plan": retreat_plan,
            "unstage_plan": unstage_plan,
            "ready_return_plan": ready_return_plan,
            "transfer_return_plan": transfer_return_plan,
            "return_plan": return_plan,
            "start_state": live,
        }

    def switch_arm(self, activate: bool) -> None:
        request = SwitchController.Request()
        request.activate_controllers = [ARM_CONTROLLER] if activate else []
        request.deactivate_controllers = [] if activate else [ARM_CONTROLLER]
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = duration_message(3.0)
        response = self.call(self.switch_controller_client, request, timeout=5.0)
        if not response.ok:
            raise RuntimeError(f"Arm controller switch failed: {response.message}")
        expected = "active" if activate else "inactive"
        if self.controller_states()[ARM_CONTROLLER]["state"] != expected:
            raise RuntimeError(f"Arm controller did not become {expected}")

    def execute_gripper(self, position: float) -> dict:
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = float(self.get_parameter("jaw_max_effort_n").value)
        handle = self.wait_future(
            self.gripper_action.send_goal_async(goal), 5.0, "gripper goal acceptance"
        )
        if not handle.accepted:
            raise RuntimeError("Simulation gripper rejected the goal")
        wrapped = self.wait_future(
            handle.get_result_async(), 8.0, "gripper action result"
        )
        result = wrapped.result
        if not result.reached_goal:
            raise RuntimeError(
                f"Gripper did not reach {position:.4f} m; stalled={result.stalled}"
            )
        return {
            "command_m_per_jaw": position,
            "measured_m_per_jaw": result.position,
            "stalled": bool(result.stalled),
            "reached_goal": bool(result.reached_goal),
        }

    def set_cup_pose(self, pose: Pose) -> None:
        request = SetEntityPose.Request()
        request.entity.name = self.cup_model_name
        request.entity.type = Entity.MODEL
        request.pose = pose
        if not self.call(self.set_pose_client, request, timeout=2.0).success:
            raise RuntimeError("Gazebo rejected the logical cup pose")
        self.last_followed_cup_pose = deepcopy(pose)

    def follow_cup_once(self) -> None:
        if self.cup_follow_relative is None:
            return
        base_to_grasp = self.lookup_matrix(
            self.planning_frame, self.grasp_link, timeout=0.5
        )
        cup_pose = matrix_pose(base_to_grasp @ self.cup_follow_relative)
        self.set_cup_pose(cup_pose)

    def execute_arm(self, trajectory, follow_cup: bool = False) -> dict:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = deepcopy(trajectory.joint_trajectory)
        goal.trajectory.header.stamp = Time().to_msg()
        goal.goal_time_tolerance = duration_message(3.0)
        handle = self.wait_future(
            self.arm_action.send_goal_async(goal), 5.0, "arm goal acceptance"
        )
        if not handle.accepted:
            raise RuntimeError("Simulation arm controller rejected the trajectory")
        result_future = handle.get_result_async()
        final_time = goal.trajectory.points[-1].time_from_start
        expected_duration = final_time.sec + final_time.nanosec / 1e9
        # Gazebo may deliberately run below real time when its GUI, RGB-D
        # renderer, MoveIt, and RQT are active together.  Keep this a bounded
        # wall-clock wait, but scale it from the trajectory's simulation-time
        # duration so a slow real-time factor is not mistaken for a failure.
        deadline = time.monotonic() + (
            expected_duration
            * float(self.get_parameter("execution_timeout_scale").value)
            + float(self.get_parameter("execution_timeout_margin_s").value)
        )
        follow_period = 1.0 / float(self.get_parameter("cup_follow_rate_hz").value)
        next_follow = 0.0
        while not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if follow_cup and time.monotonic() >= next_follow:
                self.follow_cup_once()
                next_follow = time.monotonic() + follow_period
        if not result_future.done() or result_future.result() is None:
            handle.cancel_goal_async()
            raise RuntimeError("Simulation arm trajectory timed out")
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Simulation arm trajectory failed: {result.error_code} {result.error_string}"
            )
        if follow_cup:
            self.follow_cup_once()
        return {
            "trajectory_points": len(goal.trajectory.points),
            "expected_duration_s": expected_duration,
            "controller_error_code": result.error_code,
        }

    def verify_joint_target(self, trajectory) -> float:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.joint_state is None:
            raise RuntimeError("Joint state disappeared after execution")
        actual = dict(
            zip(self.joint_state.name, self.joint_state.position, strict=True)
        )
        final = trajectory.joint_trajectory.points[-1]
        errors = [
            abs(actual[name] - expected)
            for name, expected in zip(
                trajectory.joint_trajectory.joint_names,
                final.positions,
                strict=True,
            )
        ]
        maximum = max(errors, default=0.0)
        if maximum > float(self.get_parameter("joint_final_tolerance_rad").value):
            raise RuntimeError(f"Arm final-joint error is {maximum:.6f} rad")
        return maximum

    def execute_workflow(
        self,
        preflight: dict,
        cup_object: CollisionObject,
        initial_model_pose: Pose,
    ) -> dict:
        reports: dict[str, object] = {}
        controller_active = False
        cup_attached = False
        try:
            self.switch_arm(True)
            controller_active = True
            reports["open"] = self.execute_gripper(
                float(self.get_parameter("jaw_open_m").value)
            )

            self.publish_status("moving_camera_ready_to_transfer")
            reports["camera_ready_to_transfer"] = self.execute_arm(
                preflight["transfer_plan"].trajectory
            )
            reports["camera_ready_to_transfer"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["transfer_plan"].trajectory)
            )

            self.publish_status("moving_transfer_to_side_ready")
            reports["transfer_to_side_ready"] = self.execute_arm(
                preflight["ready_plan"].trajectory
            )
            reports["transfer_to_side_ready"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["ready_plan"].trajectory)
            )

            self.publish_status("moving_side_ready_to_staging")
            reports["side_ready_to_staging"] = self.execute_arm(
                preflight["staging_plan"].trajectory
            )
            reports["side_ready_to_staging"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["staging_plan"].trajectory)
            )

            self.publish_status("descending_to_pregrasp")
            reports["staging_to_pregrasp"] = self.execute_arm(
                preflight["pre_plan"].trajectory
            )
            reports["staging_to_pregrasp"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["pre_plan"].trajectory)
            )

            self.publish_status("oblique_side_approach")
            reports["pregrasp_to_grasp"] = self.execute_arm(
                preflight["grasp_plan"].trajectory
            )
            reports["pregrasp_to_grasp"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["grasp_plan"].trajectory)
            )
            reports["logical_close"] = self.execute_gripper(
                float(self.get_parameter("jaw_logical_hold_m").value)
            )

            base_to_grasp = self.lookup_matrix(self.planning_frame, self.grasp_link)
            current_model_pose = deepcopy(self.cup_model_pose or initial_model_pose)
            self.cup_follow_relative = np.linalg.inv(base_to_grasp) @ pose_matrix(
                current_model_pose
            )
            self.apply_attach(cup_object, base_to_grasp)
            cup_attached = True
            self.verify_scene_ownership(True)
            self.follow_cup_once()
            self.publish_status("cup_attached")

            reports["attached_lift"] = self.execute_arm(
                preflight["lift_plan"].trajectory, follow_cup=True
            )
            reports["attached_lift"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["lift_plan"].trajectory)
            )

            hold_deadline = time.monotonic() + float(
                self.get_parameter("hold_duration_s").value
            )
            while time.monotonic() < hold_deadline:
                self.follow_cup_once()
                rclpy.spin_once(self, timeout_sec=0.05)
            reports["hold_duration_s"] = float(
                self.get_parameter("hold_duration_s").value
            )

            reports["attached_place"] = self.execute_arm(
                preflight["place_plan"].trajectory, follow_cup=True
            )
            reports["attached_place"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["place_plan"].trajectory)
            )

            final_model_pose = deepcopy(
                self.last_followed_cup_pose or initial_model_pose
            )
            relocated = self.relocated_world_object(
                cup_object, initial_model_pose, final_model_pose
            )
            self.apply_detach(relocated)
            cup_attached = False
            self.cup_follow_relative = None
            self.verify_scene_ownership(False)
            self.publish_status("cup_detached")
            reports["release_open"] = self.execute_gripper(
                float(self.get_parameter("jaw_open_m").value)
            )

            reports["retreat"] = self.execute_arm(
                preflight["retreat_plan"].trajectory
            )
            reports["retreat"]["max_joint_error_rad"] = self.verify_joint_target(
                preflight["retreat_plan"].trajectory
            )
            reports["unstage"] = self.execute_arm(
                preflight["unstage_plan"].trajectory
            )
            reports["unstage"]["max_joint_error_rad"] = self.verify_joint_target(
                preflight["unstage_plan"].trajectory
            )
            reports["staging_to_side_ready"] = self.execute_arm(
                preflight["ready_return_plan"].trajectory
            )
            reports["staging_to_side_ready"]["max_joint_error_rad"] = (
                self.verify_joint_target(
                    preflight["ready_return_plan"].trajectory
                )
            )
            reports["side_ready_to_transfer"] = self.execute_arm(
                preflight["transfer_return_plan"].trajectory
            )
            reports["side_ready_to_transfer"]["max_joint_error_rad"] = (
                self.verify_joint_target(
                    preflight["transfer_return_plan"].trajectory
                )
            )
            reports["transfer_to_camera_ready"] = self.execute_arm(
                preflight["return_plan"].trajectory
            )
            reports["transfer_to_camera_ready"]["max_joint_error_rad"] = (
                self.verify_joint_target(preflight["return_plan"].trajectory)
            )
            self.switch_arm(False)
            controller_active = False
            self.publish_status("complete", cup_attached=False)
            reports["success"] = True
            reports["cup_attached_at_end"] = False
            reports["arm_controller_active_at_end"] = False
            return reports
        except Exception:
            self.publish_status("stopped_on_error", cup_attached=cup_attached)
            raise
        finally:
            if controller_active:
                try:
                    self.switch_arm(False)
                except Exception as error:  # simulation-only best-effort stop
                    self.get_logger().error(f"Could not deactivate sim arm: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-sim",
        action="store_true",
        help="Execute the verified trajectories on isolated Gazebo controllers.",
    )
    parser.add_argument(
        "--confirm-simulation",
        action="store_true",
        help="Required with --execute-sim; confirms this is the isolated simulation.",
    )
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def json_safe_report(preflight: dict) -> dict:
    return {
        "top_grasp": preflight["top_grasp"],
        "selected_strategy": preflight["selected_strategy"],
        "plans": preflight["plans"],
    }


def main() -> int:
    args = parse_args()
    if args.execute_sim and not args.confirm_simulation:
        print(
            "--execute-sim requires --confirm-simulation",
            file=sys.stderr,
        )
        return 2
    rclpy.init(args=sys.argv)
    node = LogicalGraspDemo()
    attached_during_failure = False
    try:
        node.wait_for_inputs(args.execute_sim)
        node.wait_for_services(args.execute_sim)
        guards = node.verify_guards(args.execute_sim)
        frozen_target = deepcopy(node.cup_target)
        if frozen_target is None:
            raise RuntimeError("Cup target disappeared")
        cup_object = node.get_cup_object()

        model_pose = deepcopy(node.cup_model_pose)
        if args.execute_sim:
            if model_pose is None:
                raise RuntimeError("Gazebo cup model pose is unavailable")
            target_xy = np.array(
                [frozen_target.pose.position.x, frozen_target.pose.position.y]
            )
            model_xy = np.array([model_pose.position.x, model_pose.position.y])
            target_model_error = float(np.linalg.norm(target_xy - model_xy))
            if target_model_error > float(
                node.get_parameter("target_model_xy_tolerance_m").value
            ):
                raise RuntimeError(
                    f"Camera/Gazebo cup mismatch is {target_model_error:.4f} m"
                )
        else:
            target_model_error = None

        node.publish_status("preflight_started", execution=args.execute_sim)
        preflight = node.preflight(frozen_target, cup_object)
        report = {
            "mode": "execute_sim" if args.execute_sim else "plan_only",
            "guards": guards,
            "camera_model_xy_error_m": target_model_error,
            **json_safe_report(preflight),
        }
        if args.execute_sim:
            if model_pose is None:
                raise RuntimeError("Cup model pose disappeared")
            report["execution"] = node.execute_workflow(
                preflight, cup_object, model_pose
            )
        else:
            report["execution"] = {
                "attempted": False,
                "trajectory_sent": False,
                "controller_switched": False,
            }
            node.publish_status("plan_only_complete", success=True)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        # Do not auto-detach on a mid-air failure. Keeping the simulated cup
        # frozen/attached is safer and makes the stopped ownership state clear.
        attached_during_failure = node.cup_follow_relative is not None
        failure = {
            "success": False,
            "error": str(error),
            "cup_may_remain_attached": attached_during_failure,
            "real_robot_command_sent": False,
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
