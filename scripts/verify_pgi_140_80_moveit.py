#!/usr/bin/env python3
"""Verify the PGI MoveIt model using plan-only services.

The script refuses ROS domain 0, requires advancing Gazebo time, and never
calls MoveIt's trajectory execution action or any controller action.
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
import xml.etree.ElementTree as ET

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import (
    GetMotionPlan,
    GetPlanningScene,
    GetPositionIK,
    GetStateValidity,
)
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
LEFT_JOINT = "pgi_left_finger_joint"
RIGHT_JOINT = "pgi_right_finger_joint"
MOVE_GROUP_NODE = "/move_group"
RRTCONNECT = "RRTConnect"
JAW_LIMIT = 0.040
NUMERICAL_LIMIT_EPSILON = 1e-9
CUP_ID = "pgi_staging_cup"
CUP_DIMENSIONS = [0.205, 0.050]
STRAW_DIMENSIONS = [0.070, 0.004]


class MoveItVerifier(Node):
    def __init__(self) -> None:
        super().__init__("pgi_140_80_moveit_verifier")
        self.joint_state: JointState | None = None
        self.first_clock_ns: int | None = None
        self.clock_advanced = False
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_subscription(Clock, "/clock", self._clock_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.parameter_client = AsyncParameterClient(self, MOVE_GROUP_NODE)
        self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")

    def _joint_callback(self, message: JointState) -> None:
        names = set(message.name)
        if set(ARM_JOINTS + [LEFT_JOINT, RIGHT_JOINT]) <= names:
            self.joint_state = message

    def _clock_callback(self, message: Clock) -> None:
        clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self.first_clock_ns is None:
            self.first_clock_ns = clock_ns
        elif clock_ns > self.first_clock_ns:
            self.clock_advanced = True

    def wait_for_inputs(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.clock_advanced and self.joint_state is not None:
                return
        raise RuntimeError("No advancing Gazebo clock and complete joint state received")

    def wait_for_services(self, timeout: float) -> None:
        if not self.parameter_client.wait_for_services(timeout_sec=timeout):
            raise RuntimeError("MoveIt parameters service is unavailable")
        clients = {
            "state validity": self.validity_client,
            "IK": self.ik_client,
            "planning scene": self.scene_client,
            "motion plan": self.plan_client,
        }
        for name, client in clients.items():
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"MoveIt {name} service is unavailable")

    def call(self, client, request, timeout: float = 10.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Service call timed out: {client.srv_name}")
        return future.result()

    def live_robot_state(self) -> RobotState:
        state = RobotState()
        state.joint_state = deepcopy(self.joint_state)
        # Gazebo can publish 0.040000000000001 m at the exact 40 mm stop.
        # Normalize only floating-point noise; never mask a physical overrun.
        for index, name in enumerate(state.joint_state.name):
            if name not in (LEFT_JOINT, RIGHT_JOINT):
                continue
            position = state.joint_state.position[index]
            if JAW_LIMIT < position <= JAW_LIMIT + NUMERICAL_LIMIT_EPSILON:
                state.joint_state.position[index] = JAW_LIMIT
            elif -NUMERICAL_LIMIT_EPSILON <= position < 0.0:
                state.joint_state.position[index] = 0.0
        state.is_diff = False
        return state

    def check_parameters(self) -> dict:
        names = [
            "allow_trajectory_execution",
            "moveit_manage_controllers",
            "moveit_simple_controller_manager.controller_names",
            "moveit_simple_controller_manager.pgi_gripper_controller.type",
            "moveit_simple_controller_manager.pgi_gripper_controller.action_ns",
            "moveit_simple_controller_manager.pgi_gripper_controller.command_joint",
            "robot_description_semantic",
        ]
        future = self.parameter_client.get_parameters(names)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("Could not read move_group parameters")
        response = future.result()
        values = {
            name: parameter_value_to_python(value)
            for name, value in zip(names, response.values)
        }

        if values["allow_trajectory_execution"] is not False:
            raise RuntimeError("MoveIt Stage 3 is not plan-only")
        if values["moveit_manage_controllers"] is not False:
            raise RuntimeError("MoveIt must not switch simulation controllers in Stage 3")
        if values["moveit_simple_controller_manager.controller_names"] != [
            "joint_trajectory_controller",
            "pgi_gripper_controller",
        ]:
            raise RuntimeError("Unexpected MoveIt controller list")
        if values["moveit_simple_controller_manager.pgi_gripper_controller.type"] != (
            "GripperCommand"
        ):
            raise RuntimeError("PGI MoveIt controller is not GripperCommand")
        if values["moveit_simple_controller_manager.pgi_gripper_controller.action_ns"] != (
            "gripper_cmd"
        ):
            raise RuntimeError("Unexpected PGI gripper action namespace")
        if values[
            "moveit_simple_controller_manager.pgi_gripper_controller.command_joint"
        ] != LEFT_JOINT:
            raise RuntimeError("MoveIt does not command the PGI master jaw")

        semantic = ET.fromstring(values["robot_description_semantic"])
        groups = {group.attrib["name"] for group in semantic.findall("group")}
        if groups != {"ur_manipulator", "pgi_gripper", "ur10e_pgi"}:
            raise RuntimeError(f"Unexpected MoveIt groups: {sorted(groups)}")
        end_effectors = {
            item.attrib["name"]: item.attrib for item in semantic.findall("end_effector")
        }
        expected_ee = {
            "name": "pgi_140_80",
            "parent_link": "pgi_body",
            "group": "pgi_gripper",
            "parent_group": "ur_manipulator",
        }
        if end_effectors.get("pgi_140_80") != expected_ee:
            raise RuntimeError("PGI end-effector semantics are incorrect")
        return {
            "plan_only": True,
            "moveit_manages_controllers": False,
            "groups": sorted(groups),
            "end_effector": expected_ee,
            "controllers": values[
                "moveit_simple_controller_manager.controller_names"
            ],
        }

    def check_state_validity(self, group_name: str) -> dict:
        request = GetStateValidity.Request()
        request.robot_state = self.live_robot_state()
        request.group_name = group_name
        response = self.call(self.validity_client, request)
        contacts = [
            sorted((contact.contact_body_1, contact.contact_body_2))
            for contact in response.contacts
        ]
        return {"valid": bool(response.valid), "contacts": contacts}

    def check_scene(self, require_cup: bool) -> dict:
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            | PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )
        scene = self.call(self.scene_client, request).scene
        matrix = scene.allowed_collision_matrix
        indices = {name: index for index, name in enumerate(matrix.entry_names)}

        def allowed(first: str, second: str) -> bool:
            if first not in indices or second not in indices:
                return False
            return bool(matrix.entry_values[indices[first]].enabled[indices[second]])

        required_pairs = [
            ("pgi_body", "pgi_left_finger"),
            ("pgi_body", "pgi_right_finger"),
            ("pgi_left_finger", "pgi_right_finger"),
            ("pgi_camera_interposer", "d435i_mount"),
        ]
        allowed_pairs = {
            f"{first}<->{second}": allowed(first, second)
            for first, second in required_pairs
        }
        cup = next(
            (item for item in scene.world.collision_objects if item.id == CUP_ID),
            None,
        )
        cup_dimensions_match = bool(
            cup is not None
            and len(cup.primitives) == 2
            and all(
                abs(actual - expected) <= 1e-9
                for actual, expected in zip(
                    cup.primitives[0].dimensions, CUP_DIMENSIONS
                )
            )
            and all(
                abs(actual - expected) <= 1e-9
                for actual, expected in zip(
                    cup.primitives[1].dimensions, STRAW_DIMENSIONS
                )
            )
        )
        return {
            "required_internal_pairs_allowed": allowed_pairs,
            "all_required_pairs_allowed": all(allowed_pairs.values()),
            "cup_required": require_cup,
            "cup_present": cup is not None,
            "cup_dimensions_match": cup_dimensions_match,
            "cup_is_attached": any(
                item.object.id == CUP_ID
                for item in scene.robot_state.attached_collision_objects
            ),
        }

    def check_grasp_center_ik(self) -> dict:
        deadline = time.monotonic() + 5.0
        transform = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_link", "pgi_grasp_center", Time()
                )
                break
            except Exception:  # tf2 exposes several transient lookup exceptions
                continue
        if transform is None:
            raise RuntimeError("TF base_link -> pgi_grasp_center is unavailable")

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation

        request = GetPositionIK.Request()
        request.ik_request.group_name = "ur_manipulator"
        request.ik_request.robot_state = self.live_robot_state()
        request.ik_request.avoid_collisions = True
        request.ik_request.ik_link_name = "pgi_grasp_center"
        request.ik_request.pose_stamped = pose
        request.ik_request.timeout = Duration(sec=2)
        response = self.call(self.ik_client, request)
        return {
            "success": response.error_code.val == MoveItErrorCodes.SUCCESS,
            "error_code": response.error_code.val,
            "ik_link": "pgi_grasp_center",
            "solution_joint_count": len(response.solution.joint_state.name),
        }

    def plan_joint_goal(self, group_name: str, targets: dict[str, float]) -> dict:
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = group_name
        motion.start_state = self.live_robot_state()
        motion.pipeline_id = "ompl"
        motion.planner_id = RRTCONNECT
        motion.num_planning_attempts = 1
        motion.allowed_planning_time = 5.0
        motion.max_velocity_scaling_factor = 0.1
        motion.max_acceleration_scaling_factor = 0.1

        goal = Constraints()
        for joint_name, position in targets.items():
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = position
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            goal.joint_constraints.append(constraint)
        motion.goal_constraints.append(goal)

        response = self.call(self.plan_client, request).motion_plan_response
        return {
            "success": response.error_code.val == MoveItErrorCodes.SUCCESS,
            "error_code": response.error_code.val,
            "trajectory_points": len(response.trajectory.joint_trajectory.points),
            "planning_time_s": response.planning_time,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cup",
        action="store_true",
        help="Require the independent Stage-3 cup and its exact primitive dimensions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        numeric_domain = int(domain_id)
    except ValueError:
        print(f"Refusing invalid ROS_DOMAIN_ID={domain_id!r}", file=sys.stderr)
        return 2
    if numeric_domain == 0:
        print("Refusing ROS domain 0; use the isolated PGI simulation domain", file=sys.stderr)
        return 2

    rclpy.init()
    verifier = MoveItVerifier()
    try:
        verifier.wait_for_inputs(10.0)
        verifier.wait_for_services(10.0)
        parameters = verifier.check_parameters()
        arm_validity = verifier.check_state_validity("ur_manipulator")
        gripper_validity = verifier.check_state_validity("pgi_gripper")
        scene = verifier.check_scene(args.require_cup)
        ik = verifier.check_grasp_center_ik()

        positions = dict(zip(verifier.joint_state.name, verifier.joint_state.position))
        arm_targets = {name: positions[name] for name in ARM_JOINTS}
        arm_targets["shoulder_pan_joint"] += 0.01
        arm_plan = verifier.plan_joint_goal("ur_manipulator", arm_targets)
        gripper_plan = verifier.plan_joint_goal("pgi_gripper", {LEFT_JOINT: 0.020})

        success = all(
            [
                arm_validity["valid"],
                gripper_validity["valid"],
                scene["all_required_pairs_allowed"],
                not args.require_cup
                or (scene["cup_present"] and scene["cup_dimensions_match"]),
                not scene["cup_is_attached"],
                ik["success"],
                arm_plan["success"],
                gripper_plan["success"],
            ]
        )
        report = {
            "simulation_domain": numeric_domain,
            "parameters": parameters,
            "state_validity": {
                "ur_manipulator": arm_validity,
                "pgi_gripper": gripper_validity,
            },
            "planning_scene": scene,
            "grasp_center_ik": ik,
            "plan_only_motion_plans": {
                "ur_manipulator_0.01_rad": arm_plan,
                "pgi_gripper_to_0.020_m": gripper_plan,
            },
            "trajectory_execution_called": False,
            "success": success,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if success else 1
    except RuntimeError as error:
        print(f"PGI MoveIt verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
