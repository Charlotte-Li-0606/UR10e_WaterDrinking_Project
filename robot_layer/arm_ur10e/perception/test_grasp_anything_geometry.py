#!/usr/bin/env python3
"""Unit tests for observation-only Grasp-Anything depth reconstruction."""

import unittest

import numpy as np

from robot_layer.arm_ur10e.perception.grasp_anything_geometry import (
    GraspProposal2D,
    GraspReconstructionError,
    reconstruct_grasp_from_depth,
)


class GraspAnythingGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = 700.0
        self.fy = 700.0
        self.cx = 50.0
        self.cy = 40.0

    def _proposal(self, **overrides) -> GraspProposal2D:
        values = {
            "u": self.cx,
            "v": self.cy,
            "angle_rad": 0.0,
            "opening_px": 50.0,
            "score": 0.8,
        }
        values.update(overrides)
        return GraspProposal2D(**values)

    def test_front_surface_recovers_metric_opening_and_axes(self) -> None:
        depth = np.full((80, 100), 0.70, dtype=np.float32)
        grasp = reconstruct_grasp_from_depth(
            depth,
            self._proposal(),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        np.testing.assert_allclose(grasp.position_camera_m, [0.0, 0.0, 0.70])
        np.testing.assert_allclose(grasp.closing_axis_camera, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(grasp.approach_axis_camera, [0.0, 0.0, 1.0])
        self.assertAlmostEqual(grasp.opening_m, 0.05, places=5)
        self.assertGreater(grasp.depth_support_ratio, 0.99)

    def test_tilted_surface_changes_approach_axis(self) -> None:
        columns = np.arange(100, dtype=np.float32)[None, :]
        depth = np.repeat(0.70 + 0.0008 * (columns - self.cx), 80, axis=0)
        grasp = reconstruct_grasp_from_depth(
            depth,
            self._proposal(angle_rad=np.pi / 2.0),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        self.assertGreater(abs(float(grasp.approach_axis_camera[0])), 0.3)
        self.assertGreater(float(grasp.approach_axis_camera[2]), 0.7)
        self.assertAlmostEqual(
            float(np.dot(grasp.closing_axis_camera, grasp.approach_axis_camera)),
            0.0,
            places=6,
        )

    def test_sparse_occluded_depth_is_rejected(self) -> None:
        depth = np.full((80, 100), np.nan, dtype=np.float32)
        depth[38:42, 47:52] = 0.70
        with self.assertRaisesRegex(GraspReconstructionError, "need 60"):
            reconstruct_grasp_from_depth(
                depth,
                self._proposal(),
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
            )

    def test_opening_larger_than_pgi_stroke_is_rejected(self) -> None:
        depth = np.full((80, 100), 0.70, dtype=np.float32)
        with self.assertRaisesRegex(GraspReconstructionError, "outside"):
            reconstruct_grasp_from_depth(
                depth,
                self._proposal(opening_px=100.0),
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
            )


if __name__ == "__main__":
    unittest.main()
