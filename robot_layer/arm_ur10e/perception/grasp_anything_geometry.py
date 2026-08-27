"""Geometry gates for turning a 2-D Grasp-Anything proposal into 6-D.

The neural model only proposes a pixel, in-plane closing angle, opening, and
score.  This module uses registered depth to recover a visible surface point,
local normal, closing direction, and metric opening.  It has no ROS, MoveIt,
or controller dependency and never decides that a trajectory is executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


class GraspReconstructionError(ValueError):
    """Raised when image/depth evidence is insufficient for a safe candidate."""


@dataclass(frozen=True)
class GraspProposal2D:
    u: float
    v: float
    angle_rad: float
    opening_px: float
    score: float


@dataclass(frozen=True)
class ReconstructedGrasp:
    position_camera_m: np.ndarray
    quaternion_camera_xyzw: np.ndarray
    closing_axis_camera: np.ndarray
    approach_axis_camera: np.ndarray
    opening_m: float
    score: float
    valid_depth_points: int
    depth_support_ratio: float
    surface_residual_ratio: float


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise GraspReconstructionError(f"{label} is degenerate")
    return vector / norm


def reconstruct_grasp_from_depth(
    depth_m: np.ndarray,
    proposal: GraspProposal2D,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    patch_radius_px: int = 10,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
    local_depth_tolerance_m: float = 0.04,
    min_depth_points: int = 60,
    min_depth_support_ratio: float = 0.30,
    max_surface_residual_ratio: float = 0.25,
    min_opening_m: float = 0.005,
    max_opening_m: float = 0.080,
) -> ReconstructedGrasp:
    """Reconstruct one camera-frame grasp from registered depth.

    Orientation convention: local X is the jaw closing axis, local Z is the
    approach direction from the visible camera side into the object, and local
    Y completes a right-handed frame.
    """
    if depth_m.ndim != 2:
        raise GraspReconstructionError("depth image must be HxW")
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)):
        raise GraspReconstructionError("camera focal lengths are invalid")
    if patch_radius_px < 2:
        raise GraspReconstructionError("patch radius must be at least 2 pixels")

    height, width = depth_m.shape
    center_u = int(round(proposal.u))
    center_v = int(round(proposal.v))
    if not (0 <= center_u < width and 0 <= center_v < height):
        raise GraspReconstructionError("proposal center is outside the depth image")

    u0 = max(0, center_u - patch_radius_px)
    u1 = min(width, center_u + patch_radius_px + 1)
    v0 = max(0, center_v - patch_radius_px)
    v1 = min(height, center_v + patch_radius_px + 1)
    patch = np.asarray(depth_m[v0:v1, u0:u1], dtype=float)
    finite = np.isfinite(patch)
    finite &= patch >= min_depth_m
    finite &= patch <= max_depth_m
    finite_values = patch[finite]
    if finite_values.size == 0:
        raise GraspReconstructionError("proposal has no valid registered depth")
    center_depth = float(np.median(finite_values))
    local = finite & (np.abs(patch - center_depth) <= local_depth_tolerance_m)
    valid_count = int(np.count_nonzero(local))
    support_ratio = valid_count / float(patch.size)
    if valid_count < min_depth_points:
        raise GraspReconstructionError(
            f"only {valid_count} local depth points; need {min_depth_points}"
        )
    if support_ratio < min_depth_support_ratio:
        raise GraspReconstructionError(
            f"depth support {support_ratio:.3f} is below {min_depth_support_ratio:.3f}"
        )

    rows, columns = np.nonzero(local)
    image_u = columns.astype(float) + u0
    image_v = rows.astype(float) + v0
    z = patch[local]
    points = np.column_stack(
        (
            (image_u - cx) * z / fx,
            (image_v - cy) * z / fy,
            z,
        )
    )
    centroid = points.mean(axis=0)
    _u, singular_values, vh = np.linalg.svd(points - centroid, full_matrices=False)
    if singular_values.size < 3 or singular_values[1] < 1e-9:
        raise GraspReconstructionError("local depth patch cannot define a surface")
    residual_ratio = float(singular_values[2] / singular_values[1])
    if residual_ratio > max_surface_residual_ratio:
        raise GraspReconstructionError(
            f"surface residual {residual_ratio:.3f} exceeds {max_surface_residual_ratio:.3f}"
        )

    visible_normal = _unit(vh[2], "surface normal")
    # A visible outward normal should face back toward the optical origin.
    if visible_normal[2] > 0.0:
        visible_normal = -visible_normal
    approach = -visible_normal

    # Grasp-Anything angle is counter-clockwise from image horizontal while
    # image row coordinates grow downward.
    du = math.cos(proposal.angle_rad)
    dv = -math.sin(proposal.angle_rad)
    image_tangent = np.array(
        [du * center_depth / fx, dv * center_depth / fy, 0.0], dtype=float
    )
    closing = image_tangent - np.dot(image_tangent, visible_normal) * visible_normal
    closing = _unit(closing, "closing axis")
    lateral = _unit(np.cross(approach, closing), "lateral axis")
    closing = _unit(np.cross(lateral, approach), "orthogonal closing axis")

    rotation = np.column_stack((closing, lateral, approach))
    quaternion = Rotation.from_matrix(rotation).as_quat()
    surface_point = np.array(
        [
            (proposal.u - cx) * center_depth / fx,
            (proposal.v - cy) * center_depth / fy,
            center_depth,
        ],
        dtype=float,
    )
    metres_per_pixel = math.hypot(
        du * center_depth / fx,
        dv * center_depth / fy,
    )
    opening_m = abs(float(proposal.opening_px)) * metres_per_pixel
    if not min_opening_m <= opening_m <= max_opening_m:
        raise GraspReconstructionError(
            f"predicted opening {opening_m:.4f} m is outside "
            f"[{min_opening_m:.4f}, {max_opening_m:.4f}] m"
        )

    return ReconstructedGrasp(
        position_camera_m=surface_point,
        quaternion_camera_xyzw=quaternion,
        closing_axis_camera=closing,
        approach_axis_camera=approach,
        opening_m=opening_m,
        score=float(proposal.score),
        valid_depth_points=valid_count,
        depth_support_ratio=support_ratio,
        surface_residual_ratio=residual_ratio,
    )
