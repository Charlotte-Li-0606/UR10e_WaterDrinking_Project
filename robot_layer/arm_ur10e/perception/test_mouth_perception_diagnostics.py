"""No-motion tests for the mouth-perception debug overlay calculations."""

from __future__ import annotations

import unittest

import numpy as np

from .mouth_perception_node import (
    MouthPerceptionNode,
    _displacement_scale_diagnostic,
)


class MouthPerceptionDiagnosticTests(unittest.TestCase):
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
