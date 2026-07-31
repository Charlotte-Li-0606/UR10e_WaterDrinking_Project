"""Hardware-free checks for the bounded translation-only active search."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from std_msgs.msg import String

from robot_layer.arm_ur10e.agent_tools.feeding_tools import (
    FeedingSafetyConfig,
    FeedingSkillLibrary,
    validate_safe_feeding_tool_call,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, duration_sec: float) -> None:
        self.now += max(0.0, float(duration_sec))


class ActiveSearchPolicyTest(unittest.TestCase):
    @staticmethod
    def _policy_only_library() -> FeedingSkillLibrary:
        library = FeedingSkillLibrary.__new__(FeedingSkillLibrary)
        library.config = FeedingSafetyConfig()
        return library

    @staticmethod
    def _search_library(clock: _FakeClock) -> FeedingSkillLibrary:
        library = FeedingSkillLibrary.__new__(FeedingSkillLibrary)
        library.config = FeedingSafetyConfig()
        library.select_active_target = lambda _selection: {"success": True, "reason": None}
        library._stable_mouth_result = lambda: {
            "success": False,
            "reason": "not enough recent mouth pose samples",
        }
        library._spin_for = clock.advance
        library._fatal_search_status = lambda: None
        library._search_fallback = None
        library._env = SimpleNamespace(
            get_straw_tip_pose=lambda: {"position": [0.30, 0.60, 0.90]},
        )
        return library

    def test_default_budget_is_fifteen_seconds_but_legacy_thirty_is_accepted(self) -> None:
        self.assertEqual(15.0, FeedingSafetyConfig().search_max_time_sec)
        default_call = validate_safe_feeding_tool_call("active_search", {}, cli_execute=False)
        legacy_call = validate_safe_feeding_tool_call(
            "active_search",
            {"max_time_sec": 30.0},
            cli_execute=False,
        )
        self.assertEqual(15.0, default_call["args"]["max_time_sec"])
        self.assertEqual(30.0, legacy_call["args"]["max_time_sec"])

    def test_waypoints_are_absolute_translation_only_and_each_segment_is_small(self) -> None:
        library = self._policy_only_library()
        origin = np.asarray([0.30, 0.60, 0.90], dtype=np.float64)

        waypoints = library._search_waypoints(origin)

        self.assertEqual(12, len(waypoints))
        self.assertEqual(
            ["retreat_1", "retreat_2", "retreat_3"],
            [waypoint["waypoint"] for waypoint in waypoints[:3]],
        )
        self.assertFalse(
            any(
                token in waypoint["waypoint"]
                for waypoint in waypoints
                for token in ("rotate", "rotation", "yaw", "wrist")
            )
        )
        targets = [origin] + [
            np.asarray(waypoint["target_straw_tip"], dtype=np.float64)
            for waypoint in waypoints
        ]
        segment_lengths = [
            float(np.linalg.norm(targets[index + 1] - targets[index]))
            for index in range(len(targets) - 1)
        ]
        self.assertLessEqual(max(segment_lengths), 0.03 + 1e-9)
        offsets = np.asarray(
            [waypoint["offset_from_origin_m"] for waypoint in waypoints],
            dtype=np.float64,
        )
        self.assertAlmostEqual(0.09, float(np.max(offsets[:, 0])))
        self.assertLessEqual(float(np.max(np.abs(offsets[:, 1]))), 0.02 + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(offsets[:, 2]))), 0.02 + 1e-9)

    def test_fatal_perception_fault_blocks_search_but_no_face_remains_recoverable(self) -> None:
        library = self._policy_only_library()
        library._latest_mouth_status = None
        library._latest_mouth_status_received_monotonic = float("-inf")
        message = String()
        message.data = '{"detected": false, "reason": "no_face"}'
        library._mouth_status_callback(message)
        self.assertIsNone(library._fatal_search_status())

        message.data = '{"detected": false, "reason": "tf_unavailable"}'
        library._mouth_status_callback(message)
        failure = library._fatal_search_status()
        self.assertIsNotNone(failure)
        self.assertFalse(failure["success"])
        self.assertTrue(failure["motion_withheld"])

    def test_plan_only_clamps_legacy_request_and_preflights_one_waypoint(self) -> None:
        clock = _FakeClock()
        library = self._search_library(clock)
        library._active_targets = SimpleNamespace(
            get_active_latest_pose=lambda: {"success": False, "reason": "mouth pose is missing"},
        )
        planned: list[dict[str, object]] = []

        def preflight(waypoint):
            planned.append(dict(waypoint))
            return {"success": True, "waypoint": waypoint["waypoint"], "target_straw_tip": waypoint["target_straw_tip"]}

        library._preflight_search_waypoint = preflight
        with patch(
            "robot_layer.arm_ur10e.agent_tools.feeding_tools.time.monotonic",
            side_effect=clock.monotonic,
        ):
            result = library.search_for_mouth(max_time_sec=30.0, execute=False)

        self.assertFalse(result["success"])
        self.assertTrue(result["plan_only"])
        self.assertTrue(result["planning_success"])
        self.assertEqual(30.0, result["requested_max_time_sec"])
        self.assertEqual(15.0, result["effective_max_time_sec"])
        self.assertFalse(result["trajectory_sent"])
        self.assertEqual(1, len(planned))
        self.assertEqual("retreat_1", planned[0]["waypoint"])

    def test_first_candidate_withholds_next_trajectory_and_waits_for_stability(self) -> None:
        clock = _FakeClock()
        library = self._search_library(clock)
        latest_results = iter(
            (
                {"success": False, "reason": "mouth pose is missing"},
                {"success": True, "pose": {"position": [0.4, 0.7, 1.1]}},
            )
        )
        library._active_targets = SimpleNamespace(
            get_active_latest_pose=lambda: next(latest_results),
        )
        library._preflight_search_waypoint = lambda waypoint: {
            "success": True,
            "waypoint": waypoint["waypoint"],
            "target_straw_tip": waypoint["target_straw_tip"],
        }
        library._wait_for_search_stability = lambda _deadline: {
            "success": True,
            "mouth_pose": {"position": [0.4, 0.7, 1.1], "frame_id": "base_link"},
            "active_target_label": "center",
            "active_target_id": "center",
        }
        library._execute_search_waypoint = lambda _preflight: self.fail(
            "candidate detection must withhold the next trajectory"
        )

        with patch(
            "robot_layer.arm_ur10e.agent_tools.feeding_tools.time.monotonic",
            side_effect=clock.monotonic,
        ):
            result = library.search_for_mouth(max_time_sec=15.0, execute=True)

        self.assertTrue(result["success"])
        self.assertTrue(result["candidate_detected"])
        self.assertTrue(result["stopped_for_stability"])
        self.assertFalse(result["trajectory_sent"])
        self.assertEqual([], result["search_steps"])


if __name__ == "__main__":
    unittest.main()
