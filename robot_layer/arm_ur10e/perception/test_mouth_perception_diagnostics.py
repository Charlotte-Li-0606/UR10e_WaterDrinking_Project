"""No-motion tests for the mouth-perception debug overlay calculations."""

from __future__ import annotations

import unittest

import numpy as np

from .mouth_perception_node import (
    MouthPerceptionNode,
    _candidate_image_labels,
    _displacement_scale_diagnostic,
    _guarded_return_clears_operator_warning,
    _operator_warning_from_tracking_status,
    _updated_operator_warning_lines,
)


class MouthPerceptionDiagnosticTests(unittest.TestCase):
    def test_camera_order_labels_every_visible_mouth_candidate(self) -> None:
        self.assertEqual((), _candidate_image_labels(0))
        self.assertEqual(("C",), _candidate_image_labels(1))
        self.assertEqual(("L", "R"), _candidate_image_labels(2))

    def test_collision_warning_is_rendered_as_a_safety_stop(self) -> None:
        lines = _operator_warning_from_tracking_status(
            '{"state": "SAFETY_STOPPED", "collision_warning": true, '
            '"operator_warning": "Collision may happen"}'
        )

        self.assertEqual(
            ("Collision may happen", "HOLD not reached - guarded return only"),
            lines,
        )

    def test_nonwarning_or_malformed_status_clears_overlay(self) -> None:
        self.assertEqual(
            (),
            _operator_warning_from_tracking_status(
                '{"collision_warning": false, "operator_warning": "ignored"}'
            ),
        )
        self.assertEqual((), _operator_warning_from_tracking_status("not-json"))

    def test_servo_collision_warning_is_latched_until_verified_return(self) -> None:
        code_four = (
            '{"state": "TRACKING", "collision_warning": true, '
            '"operator_warning": "Collision may happen"}'
        )
        ordinary_update = (
            '{"state": "TRACKING", "collision_warning": false, '
            '"operator_warning": null}'
        )
        guarded_return = (
            '{"state": "GUARDED_RETURN_COMPLETE", '
            '"guarded_return_verified": true, "collision_warning": false}'
        )

        warning = _updated_operator_warning_lines((), code_four)
        self.assertEqual(("Collision may happen",), warning)
        self.assertEqual(
            warning,
            _updated_operator_warning_lines(warning, ordinary_update),
        )
        self.assertTrue(_guarded_return_clears_operator_warning(guarded_return))
        self.assertEqual(
            (),
            _updated_operator_warning_lines(warning, guarded_return),
        )

    def test_recorded_close_range_ratio_is_reproduced(self) -> None:
        result = _displacement_scale_diagnostic(
            camera_translation_m=0.5103177365,
            camera_forward_m=0.4802941692,
            expected_camera_to_mouth_m=0.09418335,
            detected_mouth_jump_m=1.4646,
        )

        self.assertAlmostEqual(
            0.0643065, result["expected_range_over_jump"], places=6
        )
        self.assertAlmostEqual(15.5505, result["jump_over_expected_range"], places=4)
        self.assertAlmostEqual(
            0.348435, result["camera_translation_over_jump"], places=6
        )

    def test_zero_jump_has_no_scale_ratio(self) -> None:
        result = _displacement_scale_diagnostic(
            camera_translation_m=0.0,
            camera_forward_m=0.0,
            expected_camera_to_mouth_m=0.4,
            detected_mouth_jump_m=0.0,
        )

        self.assertIsNone(result["expected_range_over_jump"])
        self.assertIsNone(result["jump_over_expected_range"])

    def test_negative_magnitude_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _displacement_scale_diagnostic(
                camera_translation_m=-0.01,
                camera_forward_m=0.0,
                expected_camera_to_mouth_m=0.4,
                detected_mouth_jump_m=0.1,
            )

    def test_coordinate_row_is_explicit_about_frame(self) -> None:
        self.assertEqual(
            "tool0 @ base_link: (+0.100, -0.200, +0.300) m",
            MouthPerceptionNode._format_xyz(
                "tool0 @ base_link", (0.1, -0.2, 0.3)
            ),
        )

    def test_depth_patch_expands_over_a_small_close_range_hole(self) -> None:
        depth = np.zeros((31, 31), dtype=np.float32)
        depth[10:21, 10:21] = 0.30
        depth[12:19, 12:19] = 0.0

        value, radius, count = MouthPerceptionNode._valid_depth_patch(
            depth, 15.0, 15.0, 3, 0.15, 1.30
        )

        self.assertAlmostEqual(0.30, value, places=6)
        self.assertEqual(7, radius)
        self.assertGreater(count, 0)

    def test_depth_patch_does_not_invent_depth(self) -> None:
        value, radius, count = MouthPerceptionNode._valid_depth_patch(
            np.zeros((31, 31), dtype=np.float32),
            15.0,
            15.0,
            3,
            0.15,
            1.30,
        )

        self.assertIsNone(value)
        self.assertIsNone(radius)
        self.assertEqual(0, count)

    def test_depth_patch_accepts_finite_close_range_sample(self) -> None:
        depth = np.full((15, 15), 0.12, dtype=np.float32)

        value, radius, count = MouthPerceptionNode._valid_depth_patch(
            depth,
            7.0,
            7.0,
            3,
            0.05,
            1.30,
        )

        self.assertAlmostEqual(0.12, value, places=6)
        self.assertEqual(3, radius)
        self.assertEqual(49, count)


if __name__ == "__main__":
    unittest.main()
