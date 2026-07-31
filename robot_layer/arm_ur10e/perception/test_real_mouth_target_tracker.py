"""Hardware-free checks for guarded real multi-mouth target tracking."""

from __future__ import annotations

import json
import unittest

from robot_layer.arm_ur10e.perception.real_mouth_target_tracker import RealMouthTargetTracker


def _payload(
    candidates: list[dict[str, object]],
    *,
    frame_id: str = "base_link",
) -> str:
    return json.dumps(
        {
            "frame_id": frame_id,
            "stamp_sec": 123.0,
            "image_center_x": 320.0,
            "candidates": candidates,
        }
    )


def _candidate(x: float, image_x: float, *, normal: list[float] | None = None) -> dict[str, object]:
    return {
        "position": [x, 0.2, 1.0],
        "image_x": image_x,
        "image_y": 200.0,
        "depth_m": 0.8,
        "surface_normal": normal,
    }


class RealMouthTargetTrackerTest(unittest.TestCase):
    def test_initial_left_center_and_right_selection_are_explicit(self) -> None:
        candidates = [_candidate(0.0, 100.0), _candidate(0.2, 310.0), _candidate(0.4, 500.0)]
        expected = {"left": 0, "center": 1, "right": 2}
        for selection, selected_index in expected.items():
            with self.subTest(selection=selection):
                tracker = RealMouthTargetTracker(selection)
                result = tracker.update_json(_payload(candidates), received_monotonic=10.0)
                self.assertTrue(result["success"])
                self.assertEqual(selected_index, result["selected_candidate_index"])

    def test_non_center_request_does_not_fall_back_to_only_visible_person(self) -> None:
        tracker = RealMouthTargetTracker("left")
        result = tracker.update_json(_payload([_candidate(0.0, 300.0)]), received_monotonic=10.0)
        self.assertFalse(result["success"])
        self.assertIn("not visible", result["reason"])

    def test_one_invalid_candidate_rejects_the_complete_obstacle_snapshot(self) -> None:
        tracker = RealMouthTargetTracker("center")
        result = tracker.update_json(
            _payload([_candidate(0.0, 300.0), {"image_x": 400.0}]),
            received_monotonic=10.0,
        )
        self.assertFalse(result["success"])
        self.assertIn("invalid candidate", result["reason"])

    def test_identity_lock_follows_3d_person_when_image_order_changes(self) -> None:
        tracker = RealMouthTargetTracker("left")
        tracker.update_json(
            _payload([_candidate(0.0, 100.0), _candidate(0.4, 500.0)]),
            received_monotonic=10.0,
        )
        result = tracker.update_json(
            _payload([_candidate(0.4, 100.0), _candidate(0.01, 500.0)]),
            received_monotonic=10.1,
        )
        self.assertTrue(result["success"])
        self.assertEqual("locked_3d_nearest", result["selection_method"])
        self.assertEqual(1, result["selected_candidate_index"])
        self.assertEqual([0.01, 0.2, 1.0], result["selected_position_m"])

    def test_ambiguous_identity_match_invalidates_observation(self) -> None:
        tracker = RealMouthTargetTracker("center")
        tracker.update_json(_payload([_candidate(0.0, 320.0)]), received_monotonic=10.0)
        rejected = tracker.update_json(
            _payload([_candidate(0.02, 250.0), _candidate(0.04, 390.0)]),
            received_monotonic=10.1,
        )
        observation = tracker.observation(
            started_monotonic=9.9,
            now_monotonic=10.2,
            minimum_samples=1,
        )
        self.assertFalse(rejected["success"])
        self.assertTrue(rejected["identity_unsafe"])
        self.assertFalse(observation["available"])
        self.assertTrue(observation["identity_unsafe"])

    def test_latest_rejection_must_be_resolved_by_a_new_valid_sample(self) -> None:
        tracker = RealMouthTargetTracker("center")
        valid = _payload([_candidate(0.0, 320.0)])
        tracker.update_json(valid, received_monotonic=10.0)
        tracker.update_json("not json", received_monotonic=10.1)
        rejected = tracker.observation(
            started_monotonic=9.9,
            now_monotonic=10.2,
            minimum_samples=1,
        )
        self.assertFalse(rejected["available"])
        self.assertFalse(tracker.current_state(now_monotonic=10.2)["available"])

        tracker.update_json(valid, received_monotonic=10.3)
        recovered = tracker.observation(
            started_monotonic=9.9,
            now_monotonic=10.4,
            minimum_samples=1,
        )
        self.assertTrue(recovered["available"])
        self.assertTrue(recovered["stable"])

    def test_surface_normal_uses_configured_base_frame(self) -> None:
        tracker = RealMouthTargetTracker("center", base_frame="robot_base")
        tracker.update_json(
            _payload(
                [_candidate(0.0, 320.0, normal=[1.0, 0.0, 0.0])],
                frame_id="robot_base",
            ),
            received_monotonic=10.0,
        )
        observation = tracker.observation(
            started_monotonic=9.9,
            now_monotonic=10.1,
            minimum_samples=1,
        )
        self.assertEqual("robot_base", observation["surface_normal"]["frame_id"])


if __name__ == "__main__":
    unittest.main()
