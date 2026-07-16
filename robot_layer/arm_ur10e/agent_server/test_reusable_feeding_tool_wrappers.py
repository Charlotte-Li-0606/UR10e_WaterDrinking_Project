"""ROS-free smoke checks for reusable tool result shapes and composition."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from robot_layer.arm_ur10e.agent_tools.feeding_tools import FeedingSkillLibrary


class ReusableFeedingToolWrapperTest(unittest.TestCase):
    @staticmethod
    def _library() -> FeedingSkillLibrary:
        """Construct only enough local state to exercise no-motion wrappers."""
        library = FeedingSkillLibrary.__new__(FeedingSkillLibrary)
        library._last_tool_results = {}
        library._last_feeding_stage = "idle"
        library._last_failure = None
        library.config = SimpleNamespace(
            mouth_topic="/detected_mouth_pose",
            base_frame="base_link",
            search_max_time_sec=30.0,
        )
        return library

    def test_get_observation_exposes_general_status(self) -> None:
        library = self._library()
        library.get_feeding_observation = lambda: {
            "success": True,
            "tool": "get_feeding_observation",
            "reason": None,
            "robot_state": {"ready": True},
            "camera_info": {"frame_id": "camera"},
            "mouth_detected": True,
            "detected_mouth_pose": {"position": [0.4, 0.7, 1.1], "frame_id": "base_link"},
            "mouth_stable": True,
            "stable_mouth_pose": {"position": [0.4, 0.7, 1.1], "frame_id": "base_link"},
            "straw_tip_pose": {"position": [0.2, 0.3, 0.8]},
            "planning_scene": {"applied": True},
            "planning_scene_enabled": True,
            "octomap": {"enabled": False},
            "feeding_stage": "idle",
        }

        result = library.get_observation()

        self.assertTrue(result["success"])
        self.assertEqual("get_observation", result["tool"])
        self.assertTrue(result["camera_status"]["available"])
        self.assertEqual("mediapipe", result["perception_status"]["detector"])
        self.assertEqual(result["stable_mouth_pose"], result["mouth_pose"])
        self.assertIn("octomap", result["obstacle_status"])

    def test_detect_target_returns_standard_detection_fields(self) -> None:
        library = self._library()
        library.detect_mouth = lambda: {
            "success": True,
            "mouth_detected": True,
            "mouth_pose": {
                "position": [0.4, 0.7, 1.1],
                "frame_id": "base_link",
                "confidence": 0.9,
                "source_stamp_sec": 123.0,
            },
        }

        result = library.detect_target(target_type="mouth", detector="mediapipe")

        self.assertTrue(result["success"])
        self.assertTrue(result["detected"])
        self.assertEqual("mouth", result["target_type"])
        self.assertEqual("mediapipe", result["detector"])
        self.assertEqual(0.9, result["confidence"])
        self.assertEqual(123.0, result["timestamp"])

    def test_check_progress_exposes_reusable_rule_based_fields(self) -> None:
        library = self._library()
        library.check_feeding_progress = lambda: {
            "success": True,
            "tool": "check_feeding_progress",
            "mouth_detected": True,
            "mouth_stable": True,
            "target_selected": "center",
            "pre_mouth_target_available": True,
            "distance_to_pre_mouth": 0.01,
            "planning_success": True,
            "obstacle_blocked": False,
            "reached_pre_mouth": True,
            "holding": False,
            "failed_step": None,
            "reason": None,
        }

        result = library.check_progress(task="feed_water", critic="rule_based")

        self.assertTrue(result["success"])
        self.assertEqual("check_progress", result["tool"])
        self.assertEqual("rule_based", result["critic"])
        self.assertEqual(0.01, result["distance_to_target"])
        self.assertTrue(result["reached_target"])

    def test_feed_water_plan_only_composes_reusable_tools(self) -> None:
        library = self._library()
        library.get_observation = lambda: {"success": True, "mouth_stable": True}
        library.detect_target = lambda **_kwargs: {"success": True, "detected": True}
        library.select_target = lambda **_kwargs: {"success": True, "strategy": "center"}
        library.wait_for_stable_mouth_pose = lambda **_kwargs: {
            "success": True,
            "mouth_pose": {"position": [0.4, 0.7, 1.1], "frame_id": "base_link"},
        }
        library.move_tool_to_target = lambda **_kwargs: {
            "success": True,
            "execute": False,
            "planning_scene": {"applied": True},
        }
        library.check_progress = lambda **_kwargs: {"success": True, "critic": "rule_based"}
        library.hold = lambda **_kwargs: {"success": True, "holding": True, "plan_only": True}

        result = library.feed_water(execute=False)

        self.assertTrue(result["success"])
        self.assertEqual("pre_mouth_plan_validated", result["final_state"])
        self.assertEqual(
            [
                "get_observation",
                "detect_target",
                "active_search",
                "select_target",
                "wait_for_stable_mouth_pose",
                "move_tool_to_target",
                "adjust_cup_vertical",
                "check_progress",
                "hold",
                "retreat",
            ],
            [step["tool"] for step in result["steps"]],
        )


if __name__ == "__main__":
    unittest.main()
