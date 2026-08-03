"""No-motion checks for the integrated real feed_water state machine."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robot_layer.arm_ur10e.agent_server import real_feed_water_backend as backend
from scripts.real_dynamic_obstacle_avoidance_plan import (
    DETOUR_ROUTE_STRATEGY,
    DIRECT_ROUTE_STRATEGY,
    RealDynamicObstacleAvoidancePlan,
)
from scripts.real_feed_water_integrated import RealIntegratedFeedWater


class RealIntegratedFeedWaterTest(unittest.TestCase):
    @staticmethod
    def _snapshot(mouth: dict[str, object]) -> dict[str, object]:
        return {
            "joint_state": {"complete": True},
            "tool0_pose": {
                "available": True,
                "position_m": [-0.25, -0.04, 0.61],
                "orientation_quat_xyzw": [0.6, 0.8, 0.0, 0.0],
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

    def test_search_waypoints_match_the_bounded_translation_only_policy(self) -> None:
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
        positions = [origin] + [item["target_tool0_position_m"] for item in waypoints]
        segment_lengths = [
            sum((float(a) - float(b)) ** 2 for a, b in zip(current, previous)) ** 0.5
            for previous, current in zip(positions, positions[1:])
        ]
        self.assertLessEqual(max(segment_lengths), 0.10 + 1e-9)
        self.assertLessEqual(
            max(abs(item["offset_camera_optical_m"][0]) for item in waypoints),
            0.05,
        )
        self.assertLessEqual(
            max(abs(item["offset_camera_optical_m"][1]) for item in waypoints),
            0.05,
        )
        self.assertTrue(all(item["camera_extrinsic_applied"] for item in waypoints))

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

    def test_search_waypoint_variants_shrink_without_changing_direction(self) -> None:
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
            [-0.05, -0.04, -0.03, -0.02],
            [item["offset_camera_optical_m"][0] for item in variants],
        )
        self.assertTrue(
            all(item["offset_camera_optical_m"][2] == -0.04 for item in variants)
        )
        self.assertEqual(
            [False, True, True, True],
            [item["adaptive_scale_applied"] for item in variants],
        )

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
        self.assertEqual(19, node._search_plan.call_count)
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
        )

        self.assertEqual(str(backend.REAL_FEED_WATER_SCRIPT), command[1])
        self.assertIn("--execute", command)
        self.assertIn("--confirm-real-motion", command)
        self.assertIn("--allow-validated-camera-ray-execute", command)
        self.assertNotIn("--no-execute", command)

    def test_integrated_plan_sequences_search_before_dynamic_target_plan(self) -> None:
        node = RealIntegratedFeedWater.__new__(RealIntegratedFeedWater)
        node.target_selection = "center"
        node.dynamic_readiness = Mock(return_value={"success": True, "failures": []})
        node.active_search = Mock(
            return_value={
                "success": True,
                "stage": "mouth_found_without_search_motion",
                "trajectory_sent": False,
            }
        )
        node.plan = Mock(return_value=(0, {"success": True, "stage": "move_group_plan_only"}))

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
        self.assertTrue(result["integrated_real_feed_water"]["dynamic_obstacle_avoidance"])
        node.dynamic_readiness.assert_called_once_with(execution_mode=None)
        node.active_search.assert_called_once_with(execute=False, confirm_real_motion=False)
        node.plan.assert_called_once_with()

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
        self.assertIn("direct fixed-orientation path", result["reason"])
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

        with patch(
            "scripts.real_feed_water_integrated.RealPreMouthFromPerceptionPlan._goal_for_target",
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
