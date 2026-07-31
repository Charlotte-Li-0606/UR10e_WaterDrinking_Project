"""Hardware-free checks for same-target dynamic obstacle replanning."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

from robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk import UR10eRobotEnv
from robot_layer.arm_ur10e.agent_tools.feeding_tools import (
    FeedingSafetyConfig,
    FeedingSkillLibrary,
    SAFE_FEEDING_TOOL_NAMES,
)


FROZEN_TARGET = [0.42, 0.73, 1.08]
FROZEN_MOUTH = {"frame_id": "base_link", "position": [0.42, 0.81, 1.08]}
FLANGE_DOWN_RPY = [3.14159265, 0.0, 0.2]


class _FakeBackend:
    def __init__(self, *, is_real: bool) -> None:
        self.is_real = is_real

    def status(self):
        return {"name": "real" if self.is_real else "sim"}


class _FakeEnvironment:
    def __init__(self, *, is_real: bool = False, plan_success: bool = True) -> None:
        self.backend = _FakeBackend(is_real=is_real)
        self.plan_success = plan_success
        self.calls: list[dict[str, object]] = []

    def move_straw_tip_to_pre_mouth(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "success": self.plan_success,
            "move_result": {
                "dynamic_obstacle_replanning": True,
                "replan_enabled": True,
                "replan_attempts": 3,
                "replan_delay_sec": 0.0,
                "replanning_observed": not kwargs["plan_only"],
            },
        }

    @staticmethod
    def get_straw_tip_pose():
        return {"position": list(FROZEN_TARGET)}


class DynamicObstacleAvoidancePolicyTest(unittest.TestCase):
    @staticmethod
    def _library(*, is_real: bool = False, plan_success: bool = True) -> FeedingSkillLibrary:
        library = FeedingSkillLibrary.__new__(FeedingSkillLibrary)
        library.config = FeedingSafetyConfig()
        library._env = _FakeEnvironment(is_real=is_real, plan_success=plan_success)
        library._octomap_status = lambda: {
            "verified": True,
            "enabled": True,
            "sensors": ["wrist_rgbd_pointcloud"],
            "octomap_resolution_m": 0.03,
        }
        library._dynamic_obstacle_cloud_status = lambda: {
            "active": True,
            "age_sec": 0.05,
            "topic": "/wrist_rgbd/points",
        }
        library._move_straw_tip_to_pre_mouth_impl = lambda _mouth, execute: {
            "success": True,
            "mouth_pose": dict(FROZEN_MOUTH),
            "pre_mouth_target": list(FROZEN_TARGET),
            "selected_standoff_m": 0.08,
            "execute": execute,
        }
        library.compute_pre_mouth_target = lambda _mouth, standoff_m: {
            "success": True,
            "pre_mouth_target": list(FROZEN_TARGET),
            "pre_mouth_standoff_m": standoff_m,
            "planner_target": {"flange_down_rpy": list(FLANGE_DOWN_RPY)},
        }
        return library

    def test_plan_only_freezes_target_and_sends_no_execution(self) -> None:
        library = self._library()

        result = library.move_straw_tip_to_pre_mouth_with_dynamic_avoidance(execute=False)

        self.assertTrue(result["success"])
        self.assertTrue(result["plan_only"])
        self.assertFalse(result["execution_sent"])
        self.assertEqual(FROZEN_TARGET, result["frozen_pre_mouth_target"])
        self.assertEqual("ompl/RRTConnectkConfigDefault", result["planner"])
        self.assertFalse(result["wait_for_clear"])
        self.assertEqual(3, result["maximum_replan_attempts"])
        self.assertEqual(1, len(library._env.calls))
        call = library._env.calls[0]
        self.assertTrue(call["plan_only"])
        self.assertTrue(call["dynamic_obstacle_replanning"])
        self.assertEqual(FROZEN_TARGET, call["pre_mouth_safe_position"])
        self.assertEqual(FLANGE_DOWN_RPY, call["flange_down_rpy"])

    def test_function_is_independent_and_not_agent_dispatchable(self) -> None:
        self.assertNotIn("dynamic_obstacle_avoidance", SAFE_FEEDING_TOOL_NAMES)
        self.assertTrue(
            callable(FeedingSkillLibrary.move_straw_tip_to_pre_mouth_with_dynamic_avoidance)
        )

    def test_real_octomap_launcher_disables_movegroup_execution(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        launcher = (
            project_root / "scripts" / "start_ur10e_real_moveit_octomap_plan_only.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("use_octomap:=true", launcher)
        self.assertIn("allow_trajectory_execution:=false", launcher)

    def test_sim_execution_uses_identical_target_for_preflight_and_motion(self) -> None:
        library = self._library()

        result = library.move_straw_tip_to_pre_mouth_with_dynamic_avoidance(execute=True)

        self.assertTrue(result["success"])
        self.assertTrue(result["execution_sent"])
        self.assertTrue(result["target_reached"])
        self.assertEqual(0.0, result["final_target_error_m"])
        self.assertEqual(2, len(library._env.calls))
        preflight, execution = library._env.calls
        self.assertTrue(preflight["plan_only"])
        self.assertFalse(execution["plan_only"])
        self.assertEqual(preflight["pre_mouth_safe_position"], execution["pre_mouth_safe_position"])
        self.assertEqual(preflight["flange_down_rpy"], execution["flange_down_rpy"])
        self.assertTrue(execution["dynamic_obstacle_replanning"])

    def test_real_execution_is_blocked_before_any_plan_or_action(self) -> None:
        library = self._library(is_real=True)

        result = library.move_straw_tip_to_pre_mouth_with_dynamic_avoidance(execute=True)

        self.assertFalse(result["success"])
        self.assertTrue(result["real_execution_blocked"])
        self.assertFalse(result["execution_sent"])
        self.assertEqual([], library._env.calls)

    def test_missing_octomap_or_stale_cloud_fails_closed(self) -> None:
        library = self._library()
        library._dynamic_obstacle_cloud_status = lambda: {
            "active": False,
            "reason": "stale cloud",
        }

        result = library.move_straw_tip_to_pre_mouth_with_dynamic_avoidance(execute=False)

        self.assertFalse(result["success"])
        self.assertFalse(result["execution_sent"])
        self.assertEqual([], library._env.calls)

    def test_cloud_gate_requires_fresh_raw_and_moveit_filtered_streams(self) -> None:
        library = FeedingSkillLibrary.__new__(FeedingSkillLibrary)
        library.config = FeedingSafetyConfig()
        library._latest_obstacle_cloud_received_monotonic = float("-inf")
        library._latest_obstacle_cloud = None
        library._latest_filtered_obstacle_cloud_received_monotonic = float("-inf")
        library._latest_filtered_obstacle_cloud = None
        raw = PointCloud2()
        raw.header.frame_id = "wrist_rgbd_camera_optical_frame"
        raw.height = 1
        raw.width = 100
        filtered = PointCloud2()
        filtered.header.frame_id = "base_link"
        filtered.height = 1
        filtered.width = 80

        with patch(
            "robot_layer.arm_ur10e.agent_tools.feeding_tools.time.monotonic",
            return_value=10.0,
        ):
            library._obstacle_cloud_callback(raw)
            self.assertFalse(library._dynamic_obstacle_cloud_status()["active"])
            library._filtered_obstacle_cloud_callback(filtered)
            self.assertTrue(library._dynamic_obstacle_cloud_status()["active"])

        with patch(
            "robot_layer.arm_ur10e.agent_tools.feeding_tools.time.monotonic",
            return_value=11.0,
        ):
            self.assertFalse(library._dynamic_obstacle_cloud_status()["active"])

    def test_no_alternate_route_returns_failure_without_execution(self) -> None:
        library = self._library(plan_success=False)

        result = library.move_straw_tip_to_pre_mouth_with_dynamic_avoidance(execute=True)

        self.assertFalse(result["success"])
        self.assertFalse(result["execution_sent"])
        self.assertEqual(1, len(library._env.calls))
        self.assertTrue(library._env.calls[0]["plan_only"])


class _ImmediateFuture:
    def __init__(self, value) -> None:
        self._value = value

    def result(self):
        return self._value


class _FakeMoveGroupClient:
    def __init__(self) -> None:
        self.goal = None

    @staticmethod
    def wait_for_server(timeout_sec: float) -> bool:
        return timeout_sec > 0.0

    def send_goal_async(self, goal, feedback_callback):
        self.goal = goal
        for state in ("PLANNING", "MONITORING", "PLANNING", "MONITORING"):
            feedback_callback(SimpleNamespace(feedback=SimpleNamespace(state=state)))
        trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(points=[object(), object()]))
        result = SimpleNamespace(
            error_code=SimpleNamespace(val=1),
            planned_trajectory=trajectory,
            planning_time=0.25,
        )
        wrapped = SimpleNamespace(status=4, result=result)
        handle = SimpleNamespace(accepted=True, get_result_async=lambda: _ImmediateFuture(wrapped))
        return _ImmediateFuture(handle)


class _PendingFuture:
    @staticmethod
    def done() -> bool:
        return False

    @staticmethod
    def result():
        return None


class _StalledMoveGroupClient:
    def __init__(self) -> None:
        self.goal = None
        self.cancel_requested = False

    @staticmethod
    def wait_for_server(timeout_sec: float) -> bool:
        return timeout_sec > 0.0

    def send_goal_async(self, goal, feedback_callback):
        self.goal = goal
        feedback_callback(SimpleNamespace(feedback=SimpleNamespace(state="MONITORING")))

        def cancel_goal_async():
            self.cancel_requested = True
            return _ImmediateFuture(SimpleNamespace())

        handle = SimpleNamespace(
            accepted=True,
            get_result_async=lambda: _PendingFuture(),
            cancel_goal_async=cancel_goal_async,
        )
        return _ImmediateFuture(handle)


class MoveGroupDynamicProfileTest(unittest.TestCase):
    @staticmethod
    def _environment(*, is_real: bool) -> UR10eRobotEnv:
        environment = UR10eRobotEnv.__new__(UR10eRobotEnv)
        environment.backend = SimpleNamespace(is_real=is_real)
        environment.max_velocity = 0.10
        environment.max_acceleration = 0.10
        environment.default_duration = 7.0
        environment.node = SimpleNamespace(
            base_frame="base_link",
            tool_frame="tool0",
            group_name="ur_manipulator",
            latest_joint_state=None,
            move_group_client=_FakeMoveGroupClient(),
        )
        environment._spin_until_joint_state = lambda timeout: None
        return environment

    def test_dynamic_profile_is_ompl_same_goal_and_immediate_bounded_replan(self) -> None:
        environment = self._environment(is_real=False)
        pose = Pose()
        pose.position.x = 0.42
        pose.position.y = 0.73
        pose.position.z = 1.08
        pose.orientation.w = 1.0

        with patch(
            "robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk.rclpy.spin_until_future_complete"
        ):
            result = environment._move_group_to_pose(
                pose,
                plan_only=True,
                orientation_tolerance_rad=0.01,
                dynamic_obstacle_replanning=True,
            )

        goal = environment.node.move_group_client.goal
        self.assertEqual("ompl", goal.request.pipeline_id)
        self.assertEqual("RRTConnectkConfigDefault", goal.request.planner_id)
        self.assertEqual(1, len(goal.request.goal_constraints))
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))
        self.assertTrue(goal.planning_options.plan_only)
        self.assertTrue(goal.planning_options.replan)
        self.assertEqual(3, goal.planning_options.replan_attempts)
        self.assertEqual(0.0, goal.planning_options.replan_delay)
        self.assertTrue(result["success"])
        self.assertTrue(result["replanning_observed"])
        self.assertEqual("ompl", result["planning_pipeline"])

    def test_sdk_also_blocks_dynamic_real_execution(self) -> None:
        environment = self._environment(is_real=True)
        pose = Pose()
        pose.orientation.w = 1.0

        with self.assertRaisesRegex(RuntimeError, "simulation/plan-only"):
            environment._move_group_to_pose(
                pose,
                plan_only=False,
                dynamic_obstacle_replanning=True,
            )
        self.assertIsNone(environment.node.move_group_client.goal)

    def test_stale_filtered_cloud_cancels_dynamic_sim_execution(self) -> None:
        environment = self._environment(is_real=False)
        client = _StalledMoveGroupClient()
        environment.node.move_group_client = client
        environment.node.latest_filtered_obstacle_cloud_monotonic = None
        environment._ensure_execution_ready = lambda: None
        pose = Pose()
        pose.orientation.w = 1.0

        with patch(
            "robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk.rclpy.spin_until_future_complete"
        ), patch(
            "robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk.rclpy.spin_once"
        ), patch(
            "robot_layer.arm_ur10e.agent_server.robot_sdk.ur10e_sdk.time.monotonic",
            side_effect=(0.0, 0.0, 0.10, 1.0),
        ):
            result = environment._move_group_to_pose(
                pose,
                plan_only=False,
                dynamic_obstacle_replanning=True,
            )

        self.assertFalse(result["success"])
        self.assertEqual("dynamic_obstacle_cloud_watchdog", result["stage"])
        self.assertTrue(result["execution_cancel_requested"])
        self.assertTrue(client.cancel_requested)


class RealPlanOnlyDynamicProfileTest(unittest.TestCase):
    def test_real_wrapper_reuses_constraints_and_changes_only_planner_profile(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            module = importlib.import_module(
                "scripts.real_dynamic_obstacle_avoidance_plan"
            )
        planner = module.RealDynamicObstacleAvoidancePlan.__new__(
            module.RealDynamicObstacleAvoidancePlan
        )
        planner.trajectory_velocity_scaling = 0.10
        planner.trajectory_acceleration_scaling = 0.10
        planner.latest_joint_state = None
        target = {
            "position_m": [-0.55, -0.10, 0.40],
            "orientation_quat_xyzw": [0.60, 0.79, 0.02, 0.04],
        }

        goal = planner._goal_for_target(target)

        self.assertEqual(module.OMPL_PIPELINE, goal.request.pipeline_id)
        self.assertEqual(module.OMPL_PLANNER, goal.request.planner_id)
        self.assertEqual(module.REPLAN_ATTEMPTS, goal.request.num_planning_attempts)
        self.assertEqual(1, len(goal.request.goal_constraints))
        self.assertEqual(1, len(goal.request.path_constraints.orientation_constraints))
        path_orientation = goal.request.path_constraints.orientation_constraints[0]
        self.assertEqual(
            module.MAX_PATH_ORIENTATION_DEVIATION_RAD,
            path_orientation.absolute_x_axis_tolerance,
        )
        self.assertEqual(
            module.MAX_PATH_ORIENTATION_DEVIATION_RAD,
            path_orientation.absolute_y_axis_tolerance,
        )
        self.assertEqual(
            module.MAX_PATH_ORIENTATION_DEVIATION_RAD,
            path_orientation.absolute_z_axis_tolerance,
        )
        goal_orientation = goal.request.goal_constraints[0].orientation_constraints[0]
        self.assertLess(
            goal_orientation.absolute_x_axis_tolerance,
            path_orientation.absolute_x_axis_tolerance,
        )
        self.assertTrue(goal.planning_options.plan_only)
        self.assertTrue(goal.planning_options.replan)
        self.assertEqual(module.REPLAN_ATTEMPTS, goal.planning_options.replan_attempts)
        self.assertEqual(0.0, goal.planning_options.replan_delay)


if __name__ == "__main__":
    unittest.main()
