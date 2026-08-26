#!/usr/bin/env python3
"""Run one guarded Stage-5 native-contact grasp in isolated Gazebo.

MoveIt still owns all arm planning and collision checking.  Unlike Stage 4,
this runner never teleports the cup and never uses kinematic pose following.
The MoveIt AttachedCollisionObject is planning-scene ownership only; whether
the Gazebo cup actually lifts is measured from its dynamic model pose.
"""

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from scipy.spatial.transform import Rotation

from pgi_logical_grasp_demo import (
    LogicalGraspDemo,
    json_safe_report,
    parse_args,
    pose_matrix,
)


class PhysicalGraspDemo(LogicalGraspDemo):
    def __init__(self) -> None:
        super().__init__(
            node_name="pgi_physical_grasp_demo",
            status_topic="/pgi/physical_grasp/status",
        )
        defaults = {
            "physical_close_m": 0.030,
            "minimum_closure_m": 0.0005,
            "contact_settle_s": 1.0,
            "release_settle_s": 1.5,
            "minimum_lift_m": 0.015,
            "maximum_lift_xy_drift_m": 0.030,
            "maximum_hold_drift_m": 0.010,
            "maximum_cup_tilt_deg": 15.0,
            "maximum_release_height_m": 0.015,
            "minimum_release_drop_m": 0.010,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.physical_moveit_attached = False
        self.saved_allowed_collision_matrix = None

    @staticmethod
    def set_acm_pair(matrix, first: str, second: str, allowed: bool) -> None:
        """Set one symmetric ACM pair while preserving every other entry."""
        for name in (first, second):
            if name in matrix.entry_names:
                continue
            for row in matrix.entry_values:
                row.enabled.append(False)
            matrix.entry_names.append(name)
            row = AllowedCollisionEntry()
            row.enabled = [False] * len(matrix.entry_names)
            matrix.entry_values.append(row)
        first_index = matrix.entry_names.index(first)
        second_index = matrix.entry_names.index(second)
        matrix.entry_values[first_index].enabled[second_index] = allowed
        matrix.entry_values[second_index].enabled[first_index] = allowed

    def apply_allowed_collision_matrix(self, matrix) -> None:
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = deepcopy(matrix)
        request = ApplyPlanningScene.Request()
        request.scene = scene
        if not self.call(self.apply_scene_client, request).success:
            raise RuntimeError("MoveIt rejected the grasp-contact ACM update")

    def begin_grasp_contact_planning(self) -> None:
        if self.saved_allowed_collision_matrix is not None:
            raise RuntimeError("Grasp-contact ACM override is already active")
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        original = deepcopy(self.call(self.scene_client, request).scene.allowed_collision_matrix)
        modified = deepcopy(original)
        for fingertip in ("pgi_left_finger", "pgi_right_finger"):
            self.set_acm_pair(modified, fingertip, self.cup_id, True)
        self.apply_allowed_collision_matrix(modified)
        self.saved_allowed_collision_matrix = original
        self.publish_status(
            "grasp_contact_acm_enabled",
            allowed_pairs=[
                ["pgi_left_finger", self.cup_id],
                ["pgi_right_finger", self.cup_id],
            ],
        )

    def end_grasp_contact_planning(self) -> None:
        if self.saved_allowed_collision_matrix is None:
            return
        original = self.saved_allowed_collision_matrix
        self.apply_allowed_collision_matrix(original)
        self.saved_allowed_collision_matrix = None
        self.publish_status("grasp_contact_acm_restored")

    def plan_cartesian(self, start, waypoints):
        """Add exact full-goal contacts when contact-aware planning fails."""
        contact_override_active = self.saved_allowed_collision_matrix is not None
        try:
            return super().plan_cartesian(start, waypoints)
        except RuntimeError as original_error:
            if not contact_override_active or not waypoints:
                raise
            solution, code = self.ik_pose(
                waypoints[-1], start, avoid_collisions=False
            )
            validity = self.state_validity(solution) if solution is not None else None
            raise RuntimeError(
                f"{original_error}; full grasp IK code={code}, "
                f"state_validity={validity}"
            ) from original_error

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def current_cup_pose(self):
        self.spin_for(0.10)
        if self.cup_model_pose is None:
            raise RuntimeError("Dynamic Gazebo cup pose is unavailable")
        return deepcopy(self.cup_model_pose)

    @staticmethod
    def cup_xyz(pose) -> np.ndarray:
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float
        )

    @staticmethod
    def cup_tilt_deg(pose) -> float:
        quaternion = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )
        if np.linalg.norm(quaternion) < 1e-9:
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        local_up = Rotation.from_quat(quaternion).apply([0.0, 0.0, 1.0])
        cosine = float(np.clip(np.dot(local_up, [0.0, 0.0, 1.0]), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def pose_report(self, pose) -> dict:
        return {
            "xyz_m": self.cup_xyz(pose).tolist(),
            "tilt_deg": self.cup_tilt_deg(pose),
        }

    def require_upright(self, pose, stage: str) -> None:
        tilt = self.cup_tilt_deg(pose)
        limit = float(self.get_parameter("maximum_cup_tilt_deg").value)
        if tilt > limit:
            raise RuntimeError(
                f"Cup tilt at {stage} is {tilt:.2f} deg, above {limit:.2f} deg"
            )

    def execute_contact_close(self) -> dict:
        if self.joint_state is None:
            raise RuntimeError("Jaw state unavailable before physical close")
        before = dict(
            zip(self.joint_state.name, self.joint_state.position, strict=True)
        ).get("pgi_left_finger_joint")
        if before is None:
            raise RuntimeError("Left jaw state unavailable before physical close")

        goal = GripperCommand.Goal()
        goal.command.position = float(self.get_parameter("physical_close_m").value)
        goal.command.max_effort = float(self.get_parameter("jaw_max_effort_n").value)
        handle = self.wait_future(
            self.gripper_action.send_goal_async(goal),
            5.0,
            "physical gripper goal acceptance",
        )
        if not handle.accepted:
            raise RuntimeError("Simulation gripper rejected the physical close")
        wrapped = self.wait_future(
            handle.get_result_async(), 12.0, "physical gripper result"
        )
        result = wrapped.result
        closure = float(before - result.position)
        minimum = float(self.get_parameter("minimum_closure_m").value)
        if closure < minimum:
            raise RuntimeError(
                f"Physical jaw closure was only {closure:.6f} m; no contact evidence"
            )
        if not result.stalled and not result.reached_goal:
            raise RuntimeError("Physical close neither reached its goal nor stalled")
        return {
            "command_m_per_jaw": goal.command.position,
            "start_m_per_jaw": before,
            "measured_m_per_jaw": float(result.position),
            "closure_m_per_jaw": closure,
            "stalled": bool(result.stalled),
            "reached_goal": bool(result.reached_goal),
            "action_status": int(wrapped.status),
            "commanded_max_effort_n": goal.command.max_effort,
        }

    def run_arm_segment(self, name: str, plan, reports: dict) -> None:
        reports[name] = self.execute_arm(plan.trajectory)
        reports[name]["max_joint_error_rad"] = self.verify_joint_target(
            plan.trajectory
        )

    def execute_physical_workflow(
        self, preflight: dict, cup_object, initial_model_pose
    ) -> dict:
        reports: dict[str, object] = {}
        controller_active = False
        try:
            self.switch_arm(True)
            controller_active = True
            reports["open"] = self.execute_gripper(
                float(self.get_parameter("jaw_open_m").value)
            )

            self.publish_status("moving_camera_ready_to_transfer")
            self.run_arm_segment(
                "camera_ready_to_transfer", preflight["transfer_plan"], reports
            )
            self.publish_status("moving_transfer_to_side_ready")
            self.run_arm_segment(
                "transfer_to_side_ready", preflight["ready_plan"], reports
            )
            self.publish_status("moving_side_ready_to_staging")
            self.run_arm_segment(
                "side_ready_to_staging", preflight["staging_plan"], reports
            )
            self.publish_status("descending_to_pregrasp")
            self.run_arm_segment(
                "staging_to_pregrasp", preflight["pre_plan"], reports
            )
            self.publish_status("oblique_side_approach")
            self.run_arm_segment(
                "pregrasp_to_grasp", preflight["grasp_plan"], reports
            )

            before_close = self.current_cup_pose()
            self.require_upright(before_close, "before_close")
            reports["cup_before_close"] = self.pose_report(before_close)
            reports["physical_close"] = self.execute_contact_close()
            self.spin_for(float(self.get_parameter("contact_settle_s").value))
            contact_pose = self.current_cup_pose()
            self.require_upright(contact_pose, "after_contact")
            reports["cup_after_contact"] = self.pose_report(contact_pose)
            contact_shift = float(
                np.linalg.norm(self.cup_xyz(contact_pose)[:2] - self.cup_xyz(before_close)[:2])
            )
            reports["contact_xy_shift_m"] = contact_shift
            if contact_shift > float(
                self.get_parameter("maximum_lift_xy_drift_m").value
            ):
                raise RuntimeError(
                    f"Cup moved {contact_shift:.4f} m while closing; pinch is unstable"
                )

            # MoveIt ownership is needed for collision checking the carried
            # geometry, but Gazebo remains fully dynamic and is never set_pose'd.
            current_world_object = self.relocated_world_object(
                cup_object, initial_model_pose, contact_pose
            )
            base_to_grasp = self.lookup_matrix(self.planning_frame, self.grasp_link)
            self.apply_attach(current_world_object, base_to_grasp)
            self.physical_moveit_attached = True
            self.verify_scene_ownership(True)
            self.publish_status("physical_contact_ready")

            self.run_arm_segment("physical_lift", preflight["lift_plan"], reports)
            self.spin_for(0.5)
            lifted_pose = self.current_cup_pose()
            self.require_upright(lifted_pose, "after_lift")
            lift_delta = self.cup_xyz(lifted_pose) - self.cup_xyz(contact_pose)
            reports["cup_after_lift"] = self.pose_report(lifted_pose)
            reports["measured_lift_delta_m"] = lift_delta.tolist()
            if lift_delta[2] < float(self.get_parameter("minimum_lift_m").value):
                raise RuntimeError(
                    f"Cup rose only {lift_delta[2]:.4f} m; native grasp did not lift it"
                )
            lift_xy = float(np.linalg.norm(lift_delta[:2]))
            if lift_xy > float(
                self.get_parameter("maximum_lift_xy_drift_m").value
            ):
                raise RuntimeError(
                    f"Cup XY drift during lift is {lift_xy:.4f} m"
                )
            self.publish_status("physical_lift_verified", lift_m=float(lift_delta[2]))

            hold_seconds = float(self.get_parameter("hold_duration_s").value)
            self.spin_for(hold_seconds)
            held_pose = self.current_cup_pose()
            self.require_upright(held_pose, "after_hold")
            hold_drift = float(
                np.linalg.norm(self.cup_xyz(held_pose) - self.cup_xyz(lifted_pose))
            )
            reports["hold_duration_s"] = hold_seconds
            reports["cup_after_hold"] = self.pose_report(held_pose)
            reports["hold_drift_m"] = hold_drift
            if hold_drift > float(
                self.get_parameter("maximum_hold_drift_m").value
            ):
                raise RuntimeError(f"Cup drift during hold is {hold_drift:.4f} m")

            self.run_arm_segment("physical_place", preflight["place_plan"], reports)
            self.spin_for(0.5)
            placed_pose = self.current_cup_pose()
            reports["cup_before_release"] = self.pose_report(placed_pose)

            placed_world_object = self.relocated_world_object(
                cup_object, initial_model_pose, placed_pose
            )
            self.apply_detach(placed_world_object)
            self.physical_moveit_attached = False
            self.verify_scene_ownership(False)
            reports["release_open"] = self.execute_gripper(
                float(self.get_parameter("jaw_open_m").value)
            )
            self.spin_for(float(self.get_parameter("release_settle_s").value))
            released_pose = self.current_cup_pose()
            self.require_upright(released_pose, "after_release")
            reports["cup_after_release"] = self.pose_report(released_pose)
            release_drop = placed_pose.position.z - released_pose.position.z
            reports["measured_release_drop_m"] = float(release_drop)
            configured_release_height = float(
                self.get_parameter("release_height_m").value
            )
            if configured_release_height > 0.0 and release_drop < float(
                self.get_parameter("minimum_release_drop_m").value
            ):
                raise RuntimeError(
                    f"Cup dropped only {release_drop:.4f} m during release test"
                )
            maximum_height = float(
                self.get_parameter("maximum_release_height_m").value
            )
            if released_pose.position.z > maximum_height:
                raise RuntimeError(
                    f"Released cup base is still {released_pose.position.z:.4f} m high"
                )
            self.publish_status("physical_release_verified")

            self.run_arm_segment("retreat", preflight["retreat_plan"], reports)
            self.run_arm_segment("unstage", preflight["unstage_plan"], reports)
            self.run_arm_segment(
                "staging_to_side_ready", preflight["ready_return_plan"], reports
            )
            self.run_arm_segment(
                "side_ready_to_transfer",
                preflight["transfer_return_plan"],
                reports,
            )
            self.run_arm_segment(
                "transfer_to_camera_ready", preflight["return_plan"], reports
            )
            self.switch_arm(False)
            controller_active = False
            reports["success"] = True
            reports["native_contact_used"] = True
            reports["gazebo_pose_following_used"] = False
            reports["cup_attached_at_end"] = False
            reports["arm_controller_active_at_end"] = False
            self.publish_status("complete", native_contact=True)
            return reports
        except Exception:
            self.publish_status(
                "stopped_on_error",
                moveit_cup_attached=self.physical_moveit_attached,
            )
            raise
        finally:
            # Planning-scene ownership must not leak into the next test after
            # a failed native grasp. This does not move the Gazebo cup; it only
            # restores MoveIt's world/attached bookkeeping at the measured
            # dynamic pose.
            if self.physical_moveit_attached:
                try:
                    failure_pose = self.current_cup_pose()
                    failure_world_object = self.relocated_world_object(
                        cup_object, initial_model_pose, failure_pose
                    )
                    self.apply_detach(failure_world_object)
                    self.physical_moveit_attached = False
                    self.verify_scene_ownership(False)
                except Exception as error:
                    self.get_logger().error(
                        f"Failed to restore MoveIt cup ownership: {error}"
                    )
            if controller_active:
                try:
                    self.switch_arm(False)
                except Exception as error:
                    self.get_logger().error(
                        f"Failed to deactivate simulation arm controller: {error}"
                    )


def main() -> int:
    args = parse_args()
    if args.execute_sim and not args.confirm_simulation:
        print("--execute-sim requires --confirm-simulation", file=sys.stderr)
        return 2
    rclpy.init(args=sys.argv)
    node = PhysicalGraspDemo()
    try:
        node.wait_for_inputs(args.execute_sim)
        node.wait_for_services(args.execute_sim)
        guards = node.verify_guards(args.execute_sim)
        frozen_target = deepcopy(node.cup_target)
        model_pose = deepcopy(node.cup_model_pose)
        if frozen_target is None:
            raise RuntimeError("Cup target disappeared")
        if args.execute_sim and model_pose is None:
            raise RuntimeError("Dynamic Gazebo cup pose is unavailable")
        if model_pose is not None:
            node.require_upright(model_pose, "initial")
            if model_pose.position.z > float(
                node.get_parameter("maximum_release_height_m").value
            ):
                raise RuntimeError(
                    f"Dynamic cup is not on the ground: z={model_pose.position.z:.4f} m"
                )

        cup_object = node.get_cup_object()
        if args.execute_sim:
            target_xy = np.array(
                [frozen_target.pose.position.x, frozen_target.pose.position.y]
            )
            model_xy = node.cup_xyz(model_pose)[:2]
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
            "stage": 5,
            "guards": guards,
            "camera_model_xy_error_m": target_model_error,
            "physics_assumptions": {
                "cup_mass_kg": 0.15,
                "cup_friction": 1.2,
                "finger_friction": 1.5,
                "jaw_effort_limit_n_per_jaw": 40.0,
                "cup_inertia": "provisional cylinder envelope",
            },
            **json_safe_report(preflight),
        }
        if args.execute_sim:
            report["execution"] = node.execute_physical_workflow(
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
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(error),
                    "moveit_cup_may_remain_attached": node.physical_moveit_attached,
                    "real_robot_command_sent": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
