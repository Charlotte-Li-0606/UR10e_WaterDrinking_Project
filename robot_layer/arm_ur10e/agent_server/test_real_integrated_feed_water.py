"""No-motion checks for the integrated real feed_water state machine."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    RobotState,
    RobotTrajectory,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_layer.arm_ur10e.agent_server import real_feed_water_backend as backend
from scripts.real_dynamic_obstacle_avoidance_plan import (
    DETOUR_ROUTE_STRATEGY,
    DIRECT_ROUTE_STRATEGY,
    RealDynamicObstacleAvoidancePlan,
)
from scripts.real_active_search_plan import RealActiveSearchPlan
from scripts.real_feed_water_integrated import (
    DEFAULT_PREMOUTH_HOLD_SEC,
    MAX_EXECUTION_TARGET_DRIFT_M,
    SEARCH_ALLOWED_PLANNING_TIME_SEC,
    SEARCH_PLANNER,
    SEARCH_PLANNING_PIPELINE,
    SEARCH_WRIST_Z_ANGLE_DEG,
    TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC,
    TRACKING_SEGMENT_MAX_ROTATION_RAD,
    TRACKING_SEGMENT_MAX_TRANSLATION_M,
    RealIntegratedFeedWater,
    _bounded_tracking_segment_target,
    _active_search_cloud_gate_failures,
    _compare_final_state_validity,
    _load_initial_position_config,
    _orientation_after_local_tool_z_rotation,
    _recorded_tool_pose_in_base_link,
    _select_consistent_cloud_frame_window,
    _trajectory_final_joint_error,
)
from scripts.real_premouth_from_perception_plan import (
    ADAPTIVE_PREMOUTH_STANDOFFS_M,
    ADAPTIVE_PREMOUTH_YAWS_DEG,
    RealPreMouthFromPerceptionPlan,
    _adaptive_premouth_pose_candidates,
    _execution_target_verification,
    _tool_vertical_tilt_rad,
)


class RealIntegratedFeedWaterTest(unittest.TestCase):
    @staticmethod
    def _cloud_frame(sequence: int, point_count: int) -> dict[str, object]:
        return {
            "frame_id": "d435i_color_optical_frame",
            "point_count": point_count,
            "received_monotonic": 100.0 + sequence,
            "stamp_sec": 200 + sequence,
            "stamp_nanosec": 0,
        }

    def test_octomap_rebuild_waits_past_unstable_initial_clouds(self) -> None:
        frames = [
            self._cloud_frame(0, 3803),
            self._cloud_frame(1, 3428),
            self._cloud_frame(2, 3063),
            self._cloud_frame(3, 3040),
            self._cloud_frame(4, 3025),
        ]

        selected = _select_consistent_cloud_frame_window(frames)

        self.assertTrue(selected["consistent"])
        self.assertEqual([3063, 3040, 3025], selected["point_counts"])
        self.assertLessEqual(selected["relative_point_count_spread"], 0.15)

    def test_octomap_rebuild_rejects_unstable_clouds_after_full_window(self) -> None:
        frames = [
            self._cloud_frame(0, 4000),
            self._cloud_frame(1, 3000),
            self._cloud_frame(2, 2200),
            self._cloud_frame(3, 1600),
        ]

        selected = _select_consistent_cloud_frame_window(frames)

        self.assertFalse(selected["consistent"])

    def test_octomap_rebuild_does_not_fail_when_nominal_candidate_has_no_ik(self) -> None:
        comparison = _compare_final_state_validity(
            {
                "available": False,
                "valid": False,
                "reason": "robot state is unavailable",
            },
            {
                "available": False,
                "valid": False,
                "reason": "robot state is unavailable",
            },
        )

        self.assertFalse(comparison["available"])
        self.assertIsNone(comparison["changed"])
        self.assertIn("adaptive candidate", comparison["reason"])

    def test_active_search_skips_cloud_gate_when_octomap_is_disabled(self) -> None:
        failures = _active_search_cloud_gate_failures(
            use_octomap=False,
            cloud_statuses={
                "/wrist_rgbd/points": {"active": False},
                "/wrist_rgbd/filtered_cloud": {"active": False},
            },
        )

        self.assertEqual([], failures)

    def test_active_search_keeps_cloud_gate_when_octomap_is_enabled(self) -> None:
        failures = _active_search_cloud_gate_failures(
            use_octomap=True,
            cloud_statuses={
                "/wrist_rgbd/points": {"active": True},
                "/wrist_rgbd/filtered_cloud": {"active": False},
            },
        )

        self.assertEqual(
            ["raw or filtered wrist point cloud became stale before search motion"],
            failures,
        )

    def test_octomap_rebuild_reports_actual_final_state_validity_change(self) -> None:
        comparison = _compare_final_state_validity(
            {"available": True, "valid": False},
            {"available": True, "valid": True},
        )

        self.assertTrue(comparison["available"])
        self.assertTrue(comparison["changed"])
        self.assertIsNone(comparison["reason"])

    def test_tracking_segment_bounds_translation_and_rotation(self) -> None:
        target = _bounded_tracking_segment_target(
            current_pose={
                "position_m": [0.0, 0.0, 0.0],
                "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            final_pose={
                "frame_id": "base_link",
                "link_name": "tool0",
                "position_m": [0.12, 0.0, 0.0],
                "orientation_quat_xyzw": [
                    0.0,
                    0.0,
                    math.sin(math.radians(30.0) / 2.0),
                    math.cos(math.radians(30.0) / 2.0),
                ],
            },
        )

        self.assertFalse(target["final_segment"])
        self.assertLessEqual(
            target["segment_translation_m"],
            TRACKING_SEGMENT_MAX_TRANSLATION_M,
        )
        self.assertLessEqual(
            target["segment_rotation_rad"],
            TRACKING_SEGMENT_MAX_ROTATION_RAD,
        )

    def test_tracking_segment_uses_complete_final_pose_when_already_bounded(self) -> None:
        final_pose = {
            "frame_id": "base_link",
            "link_name": "tool0",
            "position_m": [0.02, -0.01, 0.0],
            "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

        target = _bounded_tracking_segment_target(
            current_pose={
                "position_m": [0.0, 0.0, 0.0],
                "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            final_pose=final_pose,
        )

        self.assertTrue(target["final_segment"])
        self.assertEqual(final_pose["position_m"], target["position_m"])

    def test_initial_position_config_preserves_operator_joint_target(self) -> None:
        config = _load_initial_position_config()

        self.assertEqual("initial_position", config["name"])
        self.assertEqual(
            [3.23, -56.38, -100.43, -112.69, 91.03, 5.54],
            config["joint_positions_deg"],
        )
        self.assertEqual("joint_positions_deg", config["authoritative_target"])
        self.assertEqual("base_link", config["moveit_tool0_fk_reference"]["frame_id"])
        self.assertEqual(
            "unverified_polyscope_active_feature",
            config["operator_displayed_tool_pose"]["frame_id"],
        )

    def test_live_initial_position_check_wraps_revolute_joint_angles(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        config = _load_initial_position_config()
        state = JointState()
        state.name = list(config["joint_names"])
        state.position = list(config["joint_positions_rad"])
        state.position[0] += 2.0 * math.pi
        node.latest_joint_state = state
        node._wait_for_joint_state = Mock()
        node._spin_for = Mock()

        result = node._current_initial_position_status()

        self.assertTrue(result["available"])
        self.assertTrue(result["at_initial_position"])
        self.assertLess(result["maximum_joint_error_rad"], 1e-9)
        self.assertEqual(
            "polyscope_axis_angle_vector",
            config["operator_displayed_tool_pose"]["rotation_convention"],
        )

    def test_calibrated_moveit_fk_reference_is_not_the_unverified_display_pose(self) -> None:
        config = _load_initial_position_config()
        transformed = _recorded_tool_pose_in_base_link(config)

        for measured, expected in zip(
            transformed["position_m"], [-0.31619894, 0.15448432, 0.80049741]
        ):
            self.assertAlmostEqual(expected, measured, places=6)
        self.assertAlmostEqual(
            1.144934,
            math.degrees(
                _tool_vertical_tilt_rad(transformed["orientation_quat_xyzw"])
            ),
            places=5,
        )
        self.assertEqual(
            [0.32366, -0.13817, 0.40047],
            config["operator_displayed_tool_pose"]["position_m"],
        )

    def test_return_trajectory_final_joint_error_wraps_revolute_angles(self) -> None:
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["shoulder_pan_joint"]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=[-math.pi + 0.01])
        ]

        result = _trajectory_final_joint_error(
            trajectory,
            {"shoulder_pan_joint": math.pi - 0.01},
        )

        self.assertTrue(result["available"])
        self.assertAlmostEqual(0.02, result["maximum_joint_error_rad"], places=6)

    def test_fixed_initial_joint_goal_keeps_vertical_path_constraint(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.trajectory_velocity_scaling = 0.3
        node.trajectory_acceleration_scaling = 0.3
        node.latest_joint_state = None
        config = _load_initial_position_config()

        goal = node._joint_goal_for_initial_position(
            config,
            {
                "position_m": [-0.32366, 0.13817, 0.40047],
                "orientation_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
            },
        )

        self.assertEqual(6, len(goal.request.goal_constraints[0].joint_constraints))
        self.assertEqual("ompl", goal.request.pipeline_id)
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))
        self.assertTrue(goal.planning_options.plan_only)

    def test_return_collision_recheck_samples_every_trajectory_waypoint(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._state_validity = Mock(
            side_effect=[
                {"available": True, "valid": True, "collision_pairs": []},
                {"available": True, "valid": True, "collision_pairs": []},
            ]
        )
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["shoulder_pan_joint"]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=[0.0]),
            JointTrajectoryPoint(positions=[0.1]),
        ]

        result = node._validate_trajectory_collision_states(trajectory)

        self.assertTrue(result["success"])
        self.assertEqual(2, result["sampled_waypoints"])
        self.assertEqual(2, node._state_validity.call_count)

    def test_return_collision_recheck_reports_rejected_pair(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._state_validity = Mock(
            return_value={
                "available": True,
                "valid": False,
                "collision_pairs": ["wrist_3_link <-> octomap"],
                "reason": "wrist_3_link <-> octomap",
            }
        )
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["shoulder_pan_joint"]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=[0.0])
        ]

        result = node._validate_trajectory_collision_states(trajectory)

        self.assertFalse(result["success"])
        self.assertEqual(0, result["rejected_waypoint_index"])
        self.assertEqual(["wrist_3_link <-> octomap"], result["collision_pairs"])

    def test_fixed_initial_target_passes_fk_scene_and_state_validation(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        config = _load_initial_position_config()
        node._wait_for_joint_state = Mock()
        node._spin_for = Mock()
        node._current_robot_state = Mock(return_value=RobotState())
        node._frame_transform = Mock(
            return_value={
                "available": True,
                "position_m": [0.0, 0.0, 0.0],
                "orientation_quat_xyzw": [0.0, 0.0, 1.0, 0.0],
            }
        )
        node._fk_positions = Mock(
            return_value={
                "available": True,
                "poses": {
                    "tool0": {
                        "position_m": list(
                            config["moveit_tool0_fk_reference"]["position_m"]
                        ),
                        "orientation_quat_xyzw": list(
                            config["moveit_tool0_fk_reference"][
                                "orientation_quat_xyzw"
                            ]
                        ),
                    }
                },
            }
        )
        node._planning_scene_geometry = Mock(
            return_value=(
                {
                    "available": True,
                    "human_collision_objects_preserved": True,
                    "human_allowed_collision_pairs": [],
                    "combined_tool_collision_geometry": {"success": True},
                    "octomap": {"present": True},
                },
                None,
            )
        )
        node._state_validity = Mock(
            return_value={
                "available": True,
                "valid": True,
                "collision_pairs": [],
                "reason": None,
            }
        )
        node._point_in_ur_base = Mock(return_value=[0.3162, -0.1545, 0.8005])

        result = node._prepare_initial_position_target()

        self.assertTrue(result["success"])
        self.assertAlmostEqual(0.0, result["fk_reference_position_error_m"])
        self.assertLess(result["target_tool_vertical_tilt_deg"], 5.0)
        self.assertEqual([], result["target_state_validity"]["collision_pairs"])

    def test_execution_verification_accepts_planned_adaptive_yaw(self) -> None:
        actual = _execution_target_verification(
            start_tool0={
                "available": True,
                "position_m": [-0.19486736, 0.16107203, 0.85729495],
                "orientation_quat_xyzw": [
                    0.63927172,
                    0.76898093,
                    -0.00002742,
                    -0.0000234,
                ],
            },
            target_tool0={
                "position_m": [-0.57143579, 0.27594522, 0.72439055],
                "orientation_quat_xyzw": [
                    0.41846215,
                    0.90823423,
                    -0.00002043,
                    -0.0000297,
                ],
            },
            final_tool0={
                "available": True,
                "position_m": [-0.57158251, 0.275892, 0.72437274],
                "orientation_quat_xyzw": [
                    0.41830535,
                    0.90830646,
                    -0.00003931,
                    -0.00003945,
                ],
            },
            target_straw_tip_position_m=[
                -0.64291147,
                0.35955878,
                0.72439461,
            ],
        )

        self.assertTrue(actual["straw_tip_within_target_tolerance"])
        self.assertTrue(actual["orientation_matches_planned_target"])
        self.assertTrue(actual["orientation_stable"])
        self.assertAlmostEqual(
            math.radians(30.0),
            actual["planned_orientation_difference_from_start_rad"],
            places=6,
        )
        self.assertLess(
            actual["final_orientation_error_from_planned_target_rad"],
            math.radians(0.02),
        )
        self.assertGreater(
            actual["orientation_difference_from_start_rad"],
            math.radians(29.0),
        )

    def test_execution_verification_rejects_start_pose_when_target_has_yaw(self) -> None:
        start = {
            "available": True,
            "position_m": [0.0, 0.0, 0.0],
            "orientation_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        }
        target = {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_quat_xyzw": [
                math.cos(math.radians(15.0)),
                math.sin(math.radians(15.0)),
                0.0,
                0.0,
            ],
        }

        actual = _execution_target_verification(
            start_tool0=start,
            target_tool0=target,
            final_tool0=start,
            target_straw_tip_position_m=[0.11, 0.0, 0.0],
        )

        self.assertFalse(actual["orientation_matches_planned_target"])
        self.assertAlmostEqual(
            0.0,
            actual["orientation_difference_from_start_rad"],
        )

    def test_adaptive_goal_generator_builds_all_standoff_yaw_candidates(self) -> None:
        candidates = _adaptive_premouth_pose_candidates(
            mouth_position_m=[-0.9, 0.03, 0.65],
            approach_offset_unit=[1.0, 0.0, 0.0],
            verified_flange_down_orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(
            len(ADAPTIVE_PREMOUTH_STANDOFFS_M)
            * len(ADAPTIVE_PREMOUTH_YAWS_DEG),
            len(candidates),
        )
        self.assertEqual(0.050, candidates[0]["standoff_m"])
        self.assertEqual(0.0, candidates[0]["yaw_deg"])
        self.assertEqual(
            ADAPTIVE_PREMOUTH_STANDOFFS_M[-1],
            candidates[-1]["standoff_m"],
        )
        self.assertEqual(-60.0, candidates[-1]["yaw_deg"])

    def test_candidate_yaw_moves_tool0_but_keeps_straw_tip_on_approach_line(self) -> None:
        candidates = _adaptive_premouth_pose_candidates(
            mouth_position_m=[0.0, 0.0, 0.7],
            approach_offset_unit=[-1.0, 0.0, 0.0],
            verified_flange_down_orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
            standoffs_m=(0.12,),
            yaws_deg=(0.0, 60.0),
        )

        self.assertEqual(
            candidates[0]["straw_tip_pose"]["position_m"],
            candidates[1]["straw_tip_pose"]["position_m"],
        )
        self.assertNotEqual(
            candidates[0]["tool0_pose"]["position_m"],
            candidates[1]["tool0_pose"]["position_m"],
        )
        for candidate in candidates:
            self.assertLess(candidate["straw_tip_reconstruction_error_m"], 1e-9)
            self.assertAlmostEqual(0.0, candidate["flange_vertical_axis_error_rad"])
            self.assertFalse(candidate["wrist_3_joint_direct_command"])

    def test_collision_contact_diagnostic_reports_exact_pair_and_layer(self) -> None:
        contact = SimpleNamespace(
            contact_body_1="wrist_3_link",
            contact_body_2="real_human_obstacle_0_face_safety",
            body_type_1=0,
            body_type_2=1,
            depth=0.012,
            position=SimpleNamespace(x=-0.8, y=0.1, z=0.7),
            normal=SimpleNamespace(x=1.0, y=0.0, z=0.0),
        )

        report = RealPreMouthFromPerceptionPlan._collision_contact_report(contact)

        self.assertEqual(
            "wrist_3_link <-> real_human_obstacle_0_face_safety",
            report["pair"],
        )
        self.assertTrue(report["human_geometry_collision"])
        self.assertFalse(report["octomap_collision"])
        self.assertFalse(report["self_collision"])

    def test_cartesian_request_enforces_scaling_collision_and_jump_bounds(self) -> None:
        node = RealPreMouthFromPerceptionPlan.__new__(
            RealPreMouthFromPerceptionPlan
        )
        node.trajectory_velocity_scaling = 0.3
        node.trajectory_acceleration_scaling = 0.3
        node._validated_trajectory = None
        node._current_robot_state = Mock(return_value=RobotState())
        node._validate_trajectory_vertical_axis = Mock(
            return_value={"success": True}
        )
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["shoulder_pan_joint"]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=[0.0]),
            JointTrajectoryPoint(positions=[0.01]),
        ]
        response = SimpleNamespace(
            error_code=SimpleNamespace(val=1, message=""),
            fraction=1.0,
            solution=trajectory,
        )
        future = Mock()
        future.result.return_value = response
        node.compute_cartesian_path = Mock()
        node.compute_cartesian_path.wait_for_service.return_value = True
        node.compute_cartesian_path.call_async.return_value = future

        with patch(
            "scripts.real_premouth_from_perception_plan.rclpy.spin_until_future_complete"
        ):
            result = node._run_cartesian_plan(
                {
                    "position_m": [0.1, 0.2, 0.3],
                    "orientation_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
                }
            )

        request = node.compute_cartesian_path.call_async.call_args.args[0]
        self.assertTrue(result["success"])
        self.assertTrue(request.avoid_collisions)
        self.assertAlmostEqual(0.3, request.max_velocity_scaling_factor)
        self.assertAlmostEqual(0.3, request.max_acceleration_scaling_factor)
        self.assertAlmostEqual(
            result["maximum_revolute_joint_jump_rad"],
            request.revolute_jump_threshold,
        )

    @staticmethod
    def _snapshot(mouth: dict[str, object]) -> dict[str, object]:
        return {
            "joint_state": {"complete": True},
            "tool0_pose": {
                "available": True,
                "position_m": [-0.25, -0.04, 0.61],
                "orientation_quat_xyzw": [0.6, 0.8, 0.0, 0.0],
            },
            "tool_vertical_axis_guard": {
                "available": True,
                "within_limit": True,
                "tilt_deg": 0.0,
            },
            "ur_base_tf": {
                "available": True,
                "position_m": [0.0, 0.0, 0.0],
                "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "camera_tf": {
                "available": True,
                "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "mount_calibration": {"corrected_physical_profile": True},
            "camera_mount_match": {"matches": True},
            "move_group_available": True,
            "mouth_pose": mouth,
        }

    @staticmethod
    def _policy_only_node(mouth: dict[str, object]) -> RealIntegratedFeedWater:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node.mouth_sample_seconds = 1.0
        node.latest_mouth_status = {"detected": False, "reason": "no_face"}
        snapshot = RealIntegratedFeedWaterTest._snapshot(mouth)
        node.snapshot = Mock(return_value=snapshot)
        node._tool0_pose = Mock(return_value=snapshot["tool0_pose"])
        node._search_plan = Mock(
            return_value=(
                {
                    "success": True,
                    "stage": "search_plan_only",
                    "execution_sent": False,
                },
                object(),
            )
        )
        return node

    def test_search_waypoints_mix_translations_and_tool_z_rotations(self) -> None:
        origin = [-0.25, -0.04, 0.61]
        identity = [0.0, 0.0, 0.0, 1.0]
        waypoints = RealIntegratedFeedWater.search_waypoints(
            origin,
            identity,
            identity,
        )

        self.assertEqual(5, len(waypoints))
        self.assertEqual(
            ["backward_wide", "scan_left", "scan_right", "scan_up", "scan_down"],
            [item["name"] for item in waypoints],
        )
        self.assertEqual([0.0, 0.0, -0.04], waypoints[0]["offset_from_origin_m"])
        self.assertEqual(
            [
                "cartesian_translation",
                "tool_local_z_rotation",
                "tool_local_z_rotation",
                "cartesian_translation",
                "cartesian_translation",
            ],
            [item["search_motion_type"] for item in waypoints],
        )
        self.assertAlmostEqual(
            SEARCH_WRIST_Z_ANGLE_DEG,
            waypoints[1]["tool_local_z_rotation_deg"],
        )
        self.assertAlmostEqual(
            -SEARCH_WRIST_Z_ANGLE_DEG,
            waypoints[2]["tool_local_z_rotation_deg"],
        )
        self.assertTrue(waypoints[0]["camera_extrinsic_applied"])
        self.assertFalse(waypoints[1]["camera_extrinsic_applied"])
        self.assertFalse(waypoints[1]["wrist_3_direct_command"])

    def test_translation_search_goal_constrains_vertical_axis_with_free_spin(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.trajectory_velocity_scaling = 0.30
        node.trajectory_acceleration_scaling = 0.30
        node.latest_joint_state = None
        target = {
            "position_m": [-0.20, 0.10, 0.55],
            "orientation_quat_xyzw": [0.60, 0.80, 0.0, 0.0],
            "search_motion_type": "cartesian_translation",
        }

        goal = node._search_goal_for_target(target)

        self.assertEqual("pilz_industrial_motion_planner", goal.request.pipeline_id)
        self.assertEqual("LIN", goal.request.planner_id)
        self.assertEqual(SEARCH_PLANNING_PIPELINE, goal.request.pipeline_id)
        self.assertEqual(SEARCH_PLANNER, goal.request.planner_id)
        self.assertEqual(1, goal.request.num_planning_attempts)
        self.assertEqual(
            SEARCH_ALLOWED_PLANNING_TIME_SEC,
            goal.request.allowed_planning_time,
        )
        self.assertEqual(1, len(goal.request.goal_constraints))
        self.assertEqual(1, len(goal.request.goal_constraints[0].position_constraints))
        self.assertEqual(1, len(goal.request.goal_constraints[0].orientation_constraints))
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))
        self.assertEqual(
            "vertical_axis_intermediate_active_search",
            goal.request.path_constraints.name,
        )

    def test_left_right_search_uses_pose_goal_about_tool_local_z(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.trajectory_velocity_scaling = 0.30
        node.trajectory_acceleration_scaling = 0.30
        node.latest_joint_state = None
        initial = [0.60, 0.80, 0.0, 0.0]
        target = {
            "position_m": [-0.20, 0.10, 0.55],
            "orientation_quat_xyzw": _orientation_after_local_tool_z_rotation(
                initial,
                0.25,
            ),
            "search_motion_type": "tool_local_z_rotation",
        }

        goal = node._search_goal_for_target(target)

        constraints = goal.request.goal_constraints[0]
        self.assertEqual(
            "active_search_tool_local_z_pose_goal_with_vertical_path",
            constraints.name,
        )
        self.assertEqual(1, len(constraints.position_constraints))
        self.assertEqual(1, len(constraints.orientation_constraints))
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))

    def test_independent_search_plan_uses_the_same_vertical_profile(self) -> None:
        node = RealActiveSearchPlan.__new__(RealActiveSearchPlan)
        node.trajectory_velocity_scaling = 0.30
        node.trajectory_acceleration_scaling = 0.30
        node.latest_joint_state = None
        target = {
            "position_m": [-0.20, 0.10, 0.55],
            "orientation_quat_xyzw": [0.60, 0.80, 0.0, 0.0],
        }

        goal = node._goal_for_target(target)

        self.assertEqual("pilz_industrial_motion_planner", goal.request.pipeline_id)
        self.assertEqual("LIN", goal.request.planner_id)
        self.assertEqual(1, len(goal.request.goal_constraints[0].orientation_constraints))
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))

    def test_active_search_readiness_rejects_unsafe_flange_tilt(self) -> None:
        snapshot = self._snapshot(
            {"available": False, "stable": False, "reason": "no face"}
        )
        snapshot["tool_vertical_axis_guard"] = {
            "available": True,
            "within_limit": False,
            "tilt_deg": 97.6,
        }

        failures = RealIntegratedFeedWater._base_readiness_failures(snapshot)

        self.assertTrue(any("tilt" in failure.lower() for failure in failures))

    def test_search_waypoints_apply_camera_extrinsic_instead_of_base_axes(self) -> None:
        origin = [0.0, 0.0, 0.0]
        identity = [0.0, 0.0, 0.0, 1.0]
        quarter_turn_about_y = [0.0, 2.0**-0.5, 0.0, 2.0**-0.5]

        waypoints = RealIntegratedFeedWater.search_waypoints(
            origin,
            identity,
            quarter_turn_about_y,
        )

        backward = waypoints[0]
        self.assertAlmostEqual(-0.04, backward["offset_from_origin_m"][0])
        self.assertAlmostEqual(0.0, backward["offset_from_origin_m"][1])
        self.assertAlmostEqual(0.0, backward["offset_from_origin_m"][2])
        self.assertEqual(
            backward["offset_from_origin_m"],
            backward["offset_initial_tool0_m"],
        )

    def test_left_right_search_variants_shrink_rotation_angle(self) -> None:
        origin = [-0.25, -0.04, 0.61]
        identity = [0.0, 0.0, 0.0, 1.0]

        variants = RealIntegratedFeedWater.search_waypoint_variants(
            origin,
            identity,
            identity,
            name="scan_left",
            back_distance_m=0.04,
        )

        self.assertEqual(
            [15.0, 10.0, 5.0],
            [round(item["tool_local_z_rotation_deg"], 6) for item in variants],
        )
        self.assertEqual(
            [False, True, True],
            [item["adaptive_scale_applied"] for item in variants],
        )

    def test_local_search_joint_motion_validator_accepts_small_route(self) -> None:
        trajectory = SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=["shoulder_pan_joint", "wrist_3_joint"],
                points=[
                    SimpleNamespace(
                        positions=[0.0, 0.0],
                        time_from_start=SimpleNamespace(sec=0, nanosec=0),
                    ),
                    SimpleNamespace(
                        positions=[0.10, 0.25],
                        time_from_start=SimpleNamespace(sec=1, nanosec=0),
                    ),
                ],
            )
        )

        result = RealIntegratedFeedWater._validate_search_joint_motion(trajectory)

        self.assertTrue(result["success"])

    def test_local_search_joint_motion_validator_rejects_long_route(self) -> None:
        trajectory = SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=["shoulder_pan_joint", "wrist_3_joint"],
                points=[
                    SimpleNamespace(
                        positions=[0.0, 0.0],
                        time_from_start=SimpleNamespace(sec=0, nanosec=0),
                    ),
                    SimpleNamespace(
                        positions=[1.20, -1.20],
                        time_from_start=SimpleNamespace(sec=4, nanosec=710000000),
                    ),
                ],
            )
        )

        result = RealIntegratedFeedWater._validate_search_joint_motion(trajectory)

        self.assertFalse(result["success"])
        self.assertIn("joint excursion", result["reason"])
        self.assertIn("trajectory duration", result["reason"])

    def test_stable_selected_mouth_skips_search_motion(self) -> None:
        mouth = {
            "available": True,
            "stable": True,
            "selected_candidate_index": 1,
            "candidate_count": 3,
        }
        node = self._policy_only_node(mouth)

        result = node.active_search(execute=False, confirm_real_motion=False)

        self.assertTrue(result["success"])
        self.assertTrue(result["found_without_motion"])
        self.assertFalse(result["trajectory_sent"])
        self.assertEqual(
            "pilz_industrial_motion_planner/LIN",
            result["planner"],
        )
        self.assertFalse(result["ompl_active_search_enabled"])
        node._search_plan.assert_not_called()

    def test_no_face_plan_only_validates_first_real_search_waypoint(self) -> None:
        mouth = {
            "available": False,
            "stable": False,
            "reason": "no face",
        }
        node = self._policy_only_node(mouth)

        result = node.active_search(execute=False, confirm_real_motion=False)

        self.assertFalse(result["success"])
        self.assertTrue(result["planning_success"])
        self.assertTrue(result["requires_search_execution"])
        self.assertEqual("active_search_plan_only", result["stage"])
        self.assertFalse(result["trajectory_sent"])
        self.assertEqual(1, len(result["search_steps"]))
        node._search_plan.assert_called_once()

    def test_octomap_start_collision_is_reported_before_planning(self) -> None:
        mouth = {"available": False, "stable": False, "reason": "no face"}
        node = self._policy_only_node(mouth)
        node._ensure_valid_search_start_state = Mock(
            return_value={
                "before": {
                    "available": True,
                    "valid": False,
                    "classification": "START_STATE_IN_OCTOMAP_COLLISION",
                    "contacts": [
                        {
                            "body_1": "<octomap>",
                            "body_2": "forearm_link",
                            "depth_m": 0.02,
                        }
                    ],
                },
                "octomap_clear_attempted": True,
                "recovered": False,
            }
        )

        result = node.active_search(execute=False, confirm_real_motion=False)

        self.assertEqual("active_search_start_state_collision", result["stage"])
        self.assertIn("<octomap> - forearm_link", result["reason"])
        self.assertEqual(
            "START_STATE_IN_OCTOMAP_COLLISION",
            result["failure_diagnostic"]["classification"],
        )
        self.assertFalse(result["trajectory_sent"])
        node._search_plan.assert_not_called()

    def test_unreachable_direction_is_reported_and_remaining_search_continues(self) -> None:
        mouth = {"available": False, "stable": False, "reason": "no face"}
        node = self._policy_only_node(mouth)
        node._search_plan = Mock(
            return_value=(
                {
                    "success": False,
                    "stage": "search_plan_only",
                    "error_code": -31,
                    "error_message": "",
                    "execution_sent": False,
                },
                None,
            )
        )
        node._target_ik_diagnostic = Mock(
            return_value={
                "classification": "NO_IK_SOLUTION",
                "reason": "fixed-orientation target has no IK solution",
            }
        )
        node._wait_for_selected_stability = Mock(
            return_value={"available": False, "stable": False}
        )

        result = node.active_search(execute=False, confirm_real_motion=False)

        self.assertEqual("active_search_timeout", result["stage"])
        self.assertFalse(result["trajectory_sent"])
        self.assertEqual(5, len(result["skipped_search_waypoints"]))
        self.assertEqual(17, node._search_plan.call_count)
        self.assertIn("NO_IK_SOLUTION", result["reason"])

    def test_smaller_search_candidate_is_selected_after_nominal_ik_failure(self) -> None:
        mouth = {"available": False, "stable": False, "reason": "no face"}
        node = self._policy_only_node(mouth)
        failed = {
            "success": False,
            "stage": "search_plan_only",
            "error_code": -31,
            "error_message": "",
            "execution_sent": False,
        }
        succeeded = {
            "success": True,
            "stage": "search_plan_only",
            "error_code": 1,
            "error_message": "",
            "execution_sent": False,
        }
        node._search_plan = Mock(side_effect=[(failed, None), (succeeded, object())])
        node._target_ik_diagnostic = Mock(
            return_value={
                "classification": "NO_IK_SOLUTION",
                "reason": "fixed-orientation target has no IK solution",
            }
        )

        result = node.active_search(execute=False, confirm_real_motion=False)

        self.assertEqual("active_search_plan_only", result["stage"])
        self.assertTrue(result["next_search_waypoint"]["adaptive_scale_applied"])
        self.assertEqual(
            0.03,
            result["next_search_waypoint"]["offset_camera_optical_m"][2] * -1,
        )
        self.assertEqual(2, len(result["next_search_waypoint"]["planning_attempts"]))

    def test_canonical_execute_command_selects_integrated_real_pipeline(self) -> None:
        command = backend._pipeline_command(
            execute=True,
            report_path=backend.REPORT_DIR / "test.json",
            target_selection="center",
            hold_duration_sec=DEFAULT_PREMOUTH_HOLD_SEC,
        )

        self.assertEqual(str(backend.REAL_FEED_WATER_SCRIPT), command[1])
        self.assertIn("--execute", command)
        self.assertIn("--confirm-real-motion", command)
        self.assertIn("--allow-validated-camera-ray-execute", command)
        self.assertIn("--hold-duration", command)
        hold_argument = command.index("--hold-duration") + 1
        self.assertEqual("5.0", command[hold_argument])
        self.assertNotIn("--no-execute", command)

    def test_tracked_execute_command_is_explicit_opt_in(self) -> None:
        default_command = backend._pipeline_command(
            execute=True,
            report_path=backend.REPORT_DIR / "default.json",
            target_selection="center",
            hold_duration_sec=DEFAULT_PREMOUTH_HOLD_SEC,
        )
        tracked_command = backend._pipeline_command(
            execute=True,
            report_path=backend.REPORT_DIR / "tracked.json",
            target_selection="center",
            hold_duration_sec=DEFAULT_PREMOUTH_HOLD_SEC,
            track_mouth_during_execution=True,
        )

        self.assertNotIn("--track-mouth-during-execution", default_command)
        self.assertIn("--track-mouth-during-execution", tracked_command)

    def test_continuous_execute_command_and_octomap_are_explicit_opt_ins(self) -> None:
        default_command = backend._pipeline_command(
            execute=True,
            report_path=backend.REPORT_DIR / "default.json",
            target_selection="center",
            hold_duration_sec=DEFAULT_PREMOUTH_HOLD_SEC,
        )
        continuous_command = backend._pipeline_command(
            execute=True,
            report_path=backend.REPORT_DIR / "continuous.json",
            target_selection="center",
            hold_duration_sec=DEFAULT_PREMOUTH_HOLD_SEC,
            continuous_mouth_tracking=True,
            use_octomap=True,
        )

        self.assertNotIn("--continuous-mouth-tracking", default_command)
        self.assertNotIn("--use-octomap", default_command)
        self.assertIn("--continuous-mouth-tracking", continuous_command)
        self.assertIn("--use-octomap", continuous_command)

    def test_continuous_mode_dispatches_before_legacy_workflow(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.run_continuous_integrated = Mock(return_value=(0, {"success": True}))

        code, result = node.run_integrated(
            execute=False,
            confirm_real_motion=False,
            allow_validated_camera_ray_execute=False,
            no_execute=False,
            continuous_mouth_tracking=True,
            use_octomap=False,
            hold_duration_sec=5.0,
        )

        self.assertEqual(0, code)
        self.assertTrue(result["success"])
        node.run_continuous_integrated.assert_called_once_with(
            execute=False,
            confirm_real_motion=False,
            allow_validated_camera_ray_execute=False,
            no_execute=False,
            hold_duration_sec=5.0,
            use_octomap=False,
        )

    def test_continuous_servo_requires_fresh_post_staging_target(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._acquire_continuous_target = Mock(
            return_value={
                "success": False,
                "state": "NO_TARGET",
                "reason": "no_valid_observation_after_acquisition_timeout",
                "target": object(),
            }
        )
        node._execution_state_failures = Mock(return_value=[])

        result = node._continuous_servo_approach_and_hold(
            hold_duration_sec=5.0,
            confirm_real_motion=True,
            octomap={"use_octomap": False},
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            "no_valid_observation_after_acquisition_timeout",
            result["stop_reason"],
        )
        self.assertEqual(
            "NO_TARGET",
            result["post_staging_target_acquisition"]["state"],
        )
        node._execution_state_failures.assert_not_called()

    def test_continuous_servo_checks_execution_only_after_fresh_reacquisition(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._acquire_continuous_target = Mock(
            return_value={
                "success": True,
                "state": "STABLE_TARGET",
                "reason": "stable_target_acquired",
                "target": object(),
            }
        )
        node._execution_state_failures = Mock(
            return_value=["controller readiness changed"]
        )

        result = node._continuous_servo_approach_and_hold(
            hold_duration_sec=5.0,
            confirm_real_motion=True,
            octomap={"use_octomap": False},
        )

        self.assertFalse(result["success"])
        self.assertEqual("controller readiness changed", result["stop_reason"])
        self.assertEqual(
            "STABLE_TARGET",
            result["post_staging_target_acquisition"]["state"],
        )
        node._execution_state_failures.assert_called_once_with(
            confirm_real_motion=True
        )

    def test_event_driven_servo_status_does_not_expire_by_message_age(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_publishers = Mock(return_value=1)
        node._latest_servo_status = SimpleNamespace(code=0, message="No warnings")
        node._latest_servo_status_monotonic = 1.0

        result = node._continuous_servo_status_snapshot(now=101.0)

        self.assertTrue(result["success"])
        self.assertEqual(100.0, result["last_event_age_sec"])
        self.assertEqual(
            "reliable_transient_local_event_driven",
            result["delivery_policy"],
        )

    def test_event_driven_servo_status_requires_live_publisher(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_publishers = Mock(return_value=0)
        node._latest_servo_status = SimpleNamespace(code=0, message="No warnings")
        node._latest_servo_status_monotonic = 1.0

        result = node._continuous_servo_status_snapshot(now=2.0)

        self.assertFalse(result["success"])
        self.assertEqual("servo_status_publisher_unavailable", result["reason"])

    @patch("scripts.real_feed_water_integrated.time.monotonic", return_value=10.0)
    def test_fresh_explicit_no_face_can_authorize_empty_human_scene(
        self,
        _monotonic: Mock,
    ) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_publishers = Mock(return_value=1)
        node.latest_mouth_status = {"detected": False, "reason": "no_face"}
        node.latest_mouth_status_received_monotonic = 9.8

        result = node._fresh_explicit_no_face_evidence()

        self.assertTrue(result["success"])
        self.assertTrue(result["explicit_no_face"])
        self.assertAlmostEqual(0.2, result["status_age_sec"])

    @patch("scripts.real_feed_water_integrated.time.monotonic", return_value=10.0)
    def test_stale_no_face_does_not_authorize_empty_human_scene(
        self,
        _monotonic: Mock,
    ) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_publishers = Mock(return_value=1)
        node.latest_mouth_status = {"detected": False, "reason": "no_face"}
        node.latest_mouth_status_received_monotonic = 9.0

        result = node._fresh_explicit_no_face_evidence()

        self.assertFalse(result["success"])

    @patch("scripts.real_feed_water_integrated.PlanningSceneObstacleManager")
    def test_verified_empty_view_retires_only_managed_stale_human_scene(
        self,
        manager_type: Mock,
    ) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._spin_for = Mock()
        evidence = {
            "success": True,
            "explicit_no_face": True,
            "status_age_sec": 0.01,
        }
        node._fresh_explicit_no_face_evidence = Mock(return_value=evidence)
        manager = manager_type.return_value
        manager.remove.return_value = {
            "success": True,
            "operation": "remove",
            "object_ids": ["real_human_obstacle_0_torso"],
        }

        result = node._retire_stale_human_scene_for_empty_view()

        self.assertTrue(result["success"])
        self.assertTrue(result["retired"])
        self.assertFalse(result["collision_bypassed"])
        manager.remove.assert_called_once_with(verify=True)
        manager.destroy_node.assert_called_once_with()

    @patch("scripts.real_feed_water_integrated.rclpy.spin_once")
    @patch("scripts.real_feed_water_integrated.rclpy.spin_until_future_complete")
    def test_continuous_scene_is_shared_with_movegroup_and_servo(
        self,
        _spin_until_complete: Mock,
        _spin_once: Mock,
    ) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_subscribers = Mock(return_value=2)
        scene = PlanningScene()
        human = CollisionObject()
        human.id = "real_human_obstacle_0_torso"
        scene.world.collision_objects.append(human)
        tool = AttachedCollisionObject()
        tool.object.id = "combined_camera_cup_holder_straw_collision"
        scene.robot_state.attached_collision_objects.append(tool)
        future = Mock()
        future.result.return_value = SimpleNamespace(scene=scene)
        node.get_planning_scene = Mock()
        node.get_planning_scene.wait_for_service.return_value = True
        node.get_planning_scene.call_async.return_value = future
        node._planning_scene_publisher = Mock()

        result = node._synchronize_servo_planning_scene()

        self.assertTrue(result["success"])
        self.assertTrue(result["movegroup_and_servo_scene_shared"])
        node._planning_scene_publisher.publish.assert_called_once_with(scene)
        self.assertTrue(scene.is_diff)
        self.assertTrue(scene.robot_state.is_diff)
        self.assertEqual([], list(scene.robot_state.joint_state.name))

    def test_continuous_scene_sync_refuses_without_both_monitors(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.count_subscribers = Mock(return_value=1)
        node.get_planning_scene = Mock()

        with patch(
            "scripts.real_feed_water_integrated.rclpy.ok", return_value=False
        ):
            result = node._synchronize_servo_planning_scene()

        self.assertFalse(result["success"])
        self.assertEqual(1, result["subscriber_count"])
        node.get_planning_scene.wait_for_service.assert_not_called()

    def test_integrated_plan_sequences_search_before_dynamic_target_plan(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={
                "success": True,
                "object_id": "combined_camera_cup_holder_straw_collision",
                "link_name": "tool0",
                "dimensions_m": [0.1, 0.1, 0.3],
                "center_tool0_m": [0.0, 0.0, 0.15],
                "follows_tool0": True,
            }
        )
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node._current_initial_position_status = Mock(
            return_value={"available": True, "at_initial_position": True}
        )
        node.active_search = Mock(
            return_value={
                "success": True,
                "stage": "mouth_found_without_search_motion",
                "trajectory_sent": False,
            }
        )
        node.plan = Mock(return_value=(0, {"success": True, "stage": "move_group_plan_only"}))
        node.return_to_initial_position = Mock(
            return_value=(
                0,
                {
                    "success": True,
                    "stage": "return_target_validated",
                    "execution_attempted": False,
                    "execution_sent": False,
                    "automatic_retreat_sent": False,
                },
            )
        )

        code, result = node.run_integrated(
            execute=False,
            confirm_real_motion=False,
            allow_validated_camera_ray_execute=False,
            no_execute=True,
        )

        self.assertEqual(0, code)
        self.assertTrue(result["success"])
        self.assertTrue(result["integrated_real_feed_water"]["multi_target_identity_lock"])
        self.assertTrue(result["integrated_real_feed_water"]["active_search"])
        self.assertTrue(
            result["integrated_real_feed_water"][
                "combined_camera_cup_holder_straw_collision_geometry"
            ]["verified"]
        )
        self.assertEqual(
            "pilz_industrial_motion_planner/LIN",
            result["integrated_real_feed_water"]["active_search_planner"],
        )
        self.assertFalse(
            result["integrated_real_feed_water"]["ompl_active_search_enabled"]
        )
        self.assertTrue(
            result["integrated_real_feed_water"][
                "ompl_dynamic_obstacle_detour_enabled"
            ]
        )
        self.assertTrue(result["integrated_real_feed_water"]["dynamic_obstacle_avoidance"])
        self.assertFalse(
            result["integrated_real_feed_water"][
                "mouth_tracking_during_execution"
            ]
        )
        self.assertEqual(
            "disabled_one_shot_frozen_target",
            result["integrated_real_feed_water"]["tracking_policy"],
        )
        self.assertFalse(result["integrated_real_feed_water"]["translation_only_search"])
        self.assertFalse(
            result["integrated_real_feed_water"][
                "active_search_intermediate_flange_orientation_unconstrained"
            ]
        )
        self.assertTrue(
            result["integrated_real_feed_water"][
                "vertical_axis_detour_after_direct_rejection"
            ]
        )
        self.assertTrue(
            result["integrated_real_feed_water"]["active_search_vertical_axis_constraint"]
        )
        node.dynamic_readiness.assert_called_once_with(execution_mode=None)
        node.active_search.assert_called_once_with(execute=False, confirm_real_motion=False)
        node.plan.assert_called_once_with()
        node.return_to_initial_position.assert_called_once_with(
            execute=False,
            confirm_real_motion=False,
        )
        self.assertEqual(
            "pre_mouth_and_return_target_validated", result["final_state"]
        )

    def test_integrated_pipeline_refuses_before_search_without_attached_tool_box(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={
                "success": False,
                "reason": "attached combined-tool collision geometry is unavailable",
            }
        )
        node.dynamic_readiness = Mock()
        node.active_search = Mock()

        code, result = node.run_integrated(
            execute=False,
            confirm_real_motion=False,
            allow_validated_camera_ray_execute=False,
            no_execute=True,
        )

        self.assertEqual(2, code)
        self.assertEqual("combined_tool_collision_geometry", result["stage"])
        self.assertFalse(result["execution_sent"])
        node.dynamic_readiness.assert_not_called()
        node.active_search.assert_not_called()

    def test_preflight_return_must_be_verified_before_active_search(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={"success": True}
        )
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node._current_initial_position_status = Mock(
            side_effect=[
                {
                    "available": True,
                    "at_initial_position": False,
                    "reason": "live joints are not at initial_position",
                },
                {
                    "available": True,
                    "at_initial_position": False,
                    "reason": "live joints are not at initial_position",
                },
            ]
        )
        node.return_to_initial_position = Mock(
            return_value=(
                2,
                {
                    "success": False,
                    "reason": "return verification failed",
                    "execution_attempted": True,
                    "execution_sent": True,
                },
            )
        )
        node.active_search = Mock()

        code, result = node.run_integrated(
            execute=True,
            confirm_real_motion=True,
            allow_validated_camera_ray_execute=True,
            no_execute=False,
        )

        self.assertEqual(2, code)
        self.assertEqual("preflight_return_to_initial_position_refused", result["stage"])
        node.return_to_initial_position.assert_called_once_with(
            execute=True,
            confirm_real_motion=True,
        )
        node.dynamic_readiness.assert_not_called()
        node.active_search.assert_not_called()

    def test_plan_only_reports_required_initial_return_without_motion(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={"success": True}
        )
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node._current_initial_position_status = Mock(
            return_value={
                "available": True,
                "at_initial_position": False,
                "reason": "live joints are not at initial_position",
            }
        )
        node.return_to_initial_position = Mock(
            return_value=(
                0,
                {
                    "success": True,
                    "stage": "return_target_validated",
                    "execution_attempted": False,
                    "execution_sent": False,
                },
            )
        )
        node.active_search = Mock()

        code, result = node.run_integrated(
            execute=False,
            confirm_real_motion=False,
            allow_validated_camera_ray_execute=False,
            no_execute=True,
        )

        self.assertEqual(2, code)
        self.assertEqual("initial_position_required_plan_only", result["stage"])
        self.assertFalse(result["execution_sent"])
        node.return_to_initial_position.assert_called_once_with(
            execute=False,
            confirm_real_motion=False,
        )
        node.dynamic_readiness.assert_not_called()
        node.active_search.assert_not_called()

    def test_failure_recovery_reuses_guarded_return_after_motion(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._wait_for_tracking_replan_stationary = Mock(
            return_value={"success": True}
        )
        node.return_to_initial_position = Mock(
            return_value=(
                0,
                {
                    "success": True,
                    "execution_attempted": True,
                    "execution_sent": True,
                },
            )
        )
        node._current_initial_position_status = Mock(
            return_value={"available": True, "at_initial_position": True}
        )

        result = node._attempt_failure_recovery_return(
            execute=True,
            confirm_real_motion=True,
            motion_sent=True,
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["attempted"])
        self.assertEqual("initial_position", result["final_state"])
        node.return_to_initial_position.assert_called_once_with(
            execute=True,
            confirm_real_motion=True,
        )

    def test_active_search_failure_requests_recovery_when_motion_was_sent(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={"success": True}
        )
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node._current_initial_position_status = Mock(
            return_value={"available": True, "at_initial_position": True}
        )
        node.active_search = Mock(
            return_value={
                "success": False,
                "reason": "bounded search segment failed",
                "trajectory_sent": True,
                "search_steps": [],
            }
        )
        node._attempt_failure_recovery_return = Mock(
            return_value={
                "success": True,
                "attempted": True,
                "execution_sent": True,
                "final_state": "initial_position",
            }
        )

        code, result = node.run_integrated(
            execute=True,
            confirm_real_motion=True,
            allow_validated_camera_ray_execute=True,
            no_execute=False,
        )

        self.assertEqual(2, code)
        self.assertEqual("active_search", result["stage"])
        self.assertEqual("initial_position", result["final_state"])
        node._attempt_failure_recovery_return.assert_called_once_with(
            execute=True,
            confirm_real_motion=True,
            motion_sent=True,
        )

    def test_tracked_integrated_execution_refreshes_after_segment_boundary(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node._apply_combined_tool_collision_geometry = Mock(
            return_value={"success": True}
        )
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node._current_initial_position_status = Mock(
            return_value={"available": True, "at_initial_position": True}
        )
        node.active_search = Mock(
            return_value={
                "success": True,
                "stage": "mouth_found_without_search_motion",
                "trajectory_sent": False,
            }
        )
        node._wait_for_tracking_replan_stationary = Mock(
            return_value={
                "success": True,
                "maximum_joint_speed_rad_sec": 0.0,
                "maximum_allowed_joint_speed_rad_sec": 0.01,
            }
        )
        node._servo_track_during_premouth_hold = Mock(
            return_value={
                "success": True,
                "servo_command_count": 0,
                "stop_reason": "hold_duration_complete",
            }
        )
        node.return_to_initial_position = Mock(
            return_value=(
                0,
                {
                    "success": True,
                    "execution_attempted": True,
                    "execution_sent": True,
                    "automatic_retreat_sent": True,
                },
            )
        )
        first_segment = {
            "success": False,
            "stage": "execution_verification",
            "reason": "the straw missed the final pre-mouth target",
            "execution_result": {
                "success": True,
                "stage": "tracked_cartesian_segment_complete",
                "execution_attempted": True,
                "execution_sent": True,
                "tracking_replan_required": True,
                "tracking_replan_reason": "segment_boundary",
                "tracking_segment": {
                    "segment_translation_m": 0.05,
                    "final_segment": False,
                },
            },
        }
        final_segment = {
            "success": True,
            "stage": "execute",
            "execution_result": {
                "success": True,
                "stage": "tracked_cartesian_execute_trajectory",
                "execution_attempted": True,
                "execution_sent": True,
                "tracking_replan_required": False,
            },
        }

        with patch.object(
            RealDynamicObstacleAvoidancePlan,
            "execute",
            side_effect=[(2, first_segment), (0, final_segment)],
        ) as parent_execute:
            code, result = node.run_integrated(
                execute=True,
                confirm_real_motion=True,
                allow_validated_camera_ray_execute=True,
                no_execute=False,
                track_mouth_during_execution=True,
                hold_duration_sec=5.0,
            )

        self.assertEqual(0, code)
        self.assertEqual(2, parent_execute.call_count)
        node._wait_for_tracking_replan_stationary.assert_called_once_with()
        self.assertEqual(
            "segment_boundary",
            result["tracking_replan_attempts"][0]["replan_reason"],
        )
        self.assertEqual("initial_position", result["final_state"])

    def test_dynamic_plan_failure_reports_collision_free_route_stage(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        parent_result = {
            "success": False,
            "plan_result": {"error_code": 99999},
        }

        with patch.object(
            RealDynamicObstacleAvoidancePlan,
            "plan",
            return_value=(2, parent_result),
        ):
            code, result = node.plan()

        self.assertEqual(2, code)
        self.assertEqual("dynamic_route_plan_only", result["stage"])
        self.assertEqual(99999, result["planning_error_code"])
        self.assertIn("vertical-axis OMPL detour", result["reason"])
        self.assertEqual(
            "DIRECT_AND_DETOUR_PLANNING_FAILED",
            result["failure_diagnostic"]["classification"],
        )

    def test_execution_reuses_the_selected_dynamic_route_profile(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        direct_goal = object()
        detour_goal = object()
        node._direct_goal_for_target = Mock(return_value=direct_goal)
        node._goal_for_target = Mock(return_value=detour_goal)
        target = {"position_m": [0.0, 0.0, 0.0]}

        node._selected_dynamic_route_strategy = DIRECT_ROUTE_STRATEGY
        self.assertIs(direct_goal, node._goal_for_selected_dynamic_route(target))
        node._direct_goal_for_target.assert_called_once_with(target)
        node._goal_for_target.assert_not_called()

        node._selected_dynamic_route_strategy = DETOUR_ROUTE_STRATEGY
        self.assertIs(detour_goal, node._goal_for_selected_dynamic_route(target))
        node._goal_for_target.assert_called_once_with(target)

    def test_camera_approach_drift_below_execution_threshold_is_not_confirmed(self) -> None:
        result = RealIntegratedFeedWater._execution_mouth_drift_confirmation(
            {
                "available": True,
                "stable": True,
                "sample_count": 3,
                "mean_position_m": [0.0338, 0.0, 0.0],
            },
            [0.0, 0.0, 0.0],
        )

        self.assertEqual(0.050, MAX_EXECUTION_TARGET_DRIFT_M)
        self.assertAlmostEqual(0.0338, result["drift_m"])
        self.assertFalse(result["confirmed"])

    def test_sustained_large_mouth_drift_is_still_confirmed(self) -> None:
        result = RealIntegratedFeedWater._execution_mouth_drift_confirmation(
            {
                "available": True,
                "stable": True,
                "sample_count": 3,
                "mean_position_m": [0.060, 0.0, 0.0],
            },
            [0.0, 0.0, 0.0],
        )

        self.assertAlmostEqual(0.060, result["drift_m"])
        self.assertTrue(result["confirmed"])

    def test_tracking_replan_waits_longer_without_raising_stationary_limit(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node._wait_for_search_stationary = Mock(
            return_value={
                "success": True,
                "maximum_joint_speed_rad_sec": 0.009,
                "maximum_allowed_joint_speed_rad_sec": 0.010,
            }
        )

        with patch(
            "scripts.real_feed_water_integrated.time.monotonic",
            return_value=100.0,
        ):
            result = node._wait_for_tracking_replan_stationary()

        self.assertTrue(result["success"])
        self.assertEqual(3.0, TRACKING_POST_CANCEL_SETTLE_TIMEOUT_SEC)
        node._wait_for_search_stationary.assert_called_once_with(
            103.0,
            timeout_sec=3.0,
        )
        self.assertEqual(0.010, result["maximum_allowed_joint_speed_rad_sec"])

    def test_tracking_replans_pre_execution_target_drift_only(self) -> None:
        self.assertTrue(
            RealIntegratedFeedWater._pre_execution_target_drift_requires_replan(
                {
                    "stage": "pre_execution_state_guard",
                    "failures": [
                        "selected mouth target moved 0.0400 m after planning, "
                        "above the 0.0300 m limit",
                        "a visible person's collision geometry moved 0.0600 m "
                        "after planning, above the 0.0500 m limit",
                    ],
                }
            )
        )

    def test_tracking_does_not_replan_past_controller_safety_failure(self) -> None:
        self.assertFalse(
            RealIntegratedFeedWater._pre_execution_target_drift_requires_replan(
                {
                    "stage": "pre_execution_state_guard",
                    "failures": [
                        "selected mouth target moved 0.0400 m after planning, "
                        "above the 0.0300 m limit",
                        "scaled_joint_trajectory_controller is no longer active",
                    ],
                }
            )
        )

    def test_unstable_single_frame_drift_is_not_confirmed(self) -> None:
        result = RealIntegratedFeedWater._execution_mouth_drift_confirmation(
            {
                "available": True,
                "stable": False,
                "sample_count": 1,
                "mean_position_m": [0.20, 0.0, 0.0],
            },
            [0.0, 0.0, 0.0],
        )

        self.assertFalse(result["confirmed"])
        self.assertIn("stable sample window", result["reason"])

    def test_verified_stationary_search_goal_uses_exact_zero_start_velocity(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.move_group = Mock()
        node.move_group.send_goal_async.return_value = Mock()
        fake_goal = SimpleNamespace(
            request=SimpleNamespace(
                start_state=SimpleNamespace(
                    joint_state=SimpleNamespace(
                        name=["joint_1", "joint_2"],
                        velocity=[0.004, -0.003],
                    )
                )
            )
        )

        with patch.object(
            node,
            "_search_goal_for_target",
            return_value=fake_goal,
        ), patch(
            "scripts.real_feed_water_integrated.rclpy.spin_until_future_complete"
        ):
            node.move_group.send_goal_async.return_value.result.return_value = None
            node._search_plan(
                {},
                deadline=10_000_000_000.0,
                stationary_verified=True,
            )

        self.assertEqual([0.0, 0.0], fake_goal.request.start_state.joint_state.velocity)


if __name__ == "__main__":
    unittest.main()
