#!/usr/bin/env python3
"""Plan and optionally execute a Grasp-Anything-driven Gazebo grasp cycle.

The runner is simulation-only. It freezes one timestamp-matched 2-D learned
proposal, registered-depth pose, and visible-object collision box, converts
them to provisional PGI grasp/pre-grasp poses, and delegates every arm path to
MoveIt. It never starts a robot driver and refuses ROS domain 0.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

from pgi_logical_grasp_demo import (
    json_safe_report,
    parse_args,
    stamped_pose,
    write_report_json,
)
from pgi_physical_grasp_demo import PhysicalGraspDemo


EXPECTED_MODEL_SHA256 = (
    "65984ef3364790c1ece107f22bcbeb67dc8fba21784087bb3d8ff183a3582e0a"
)


class GraspAnythingMoveItDemo(PhysicalGraspDemo):
    def __init__(self) -> None:
        super().__init__(
            node_name="pgi_grasp_anything_moveit_demo",
            status_topic="/pgi/grasp_anything/moveit_status",
        )
        defaults = {
            "grasp_anything_metadata_topic": "/pgi/grasp_anything/candidate",
            "metadata_max_age_s": 3.0,
            "surface_inset_m": 0.006,
            "contact_overlap_m": 0.002,
            "minimum_object_width_m": 0.010,
            "maximum_object_extent_m": 0.500,
            "ground_support_tolerance_m": 0.040,
            "model_origin_box_tolerance_m": 0.050,
            "support_surface_id": "ground_plane",
            "ground_supported_planar_approach": True,
            "ground_supported_approach_down_deg": 15.0,
            "preserve_depth_surface_approach": True,
            "ground_supported_axis_aligned_approach": False,
            "refine_ground_supported_candidate": False,
            "maximum_refined_object_width_m": 0.076,
            "maximum_candidate_axial_shift_m": 0.080,
            "minimum_candidate_axial_shift_m": 0.010,
            "use_radial_transit_orientation": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.grasp_metadata: dict | None = None
        self.grasp_metadata_wall_time = 0.0
        self.active_metadata: dict | None = None
        self.initial_object_rotation: Rotation | None = None
        self.initial_model_z: float | None = None
        self.contact_close_m: float | None = None
        self.saved_support_allowed_collision_matrix = None
        self.planar_axes_contract: dict | None = None
        self.create_subscription(
            String,
            str(self.get_parameter("grasp_anything_metadata_topic").value),
            self._metadata_callback,
            10,
        )

    def _metadata_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict) and payload.get("accepted") is True:
            self.grasp_metadata = payload
            self.grasp_metadata_wall_time = time.monotonic()

    @staticmethod
    def _stamp_tuple(pose: PoseStamped) -> tuple[int, int]:
        return (pose.header.stamp.sec, pose.header.stamp.nanosec)

    @staticmethod
    def _metadata_stamp(metadata: dict) -> tuple[int, int] | None:
        stamp = metadata.get("source_stamp")
        if not isinstance(stamp, dict):
            return None
        try:
            return (int(stamp["sec"]), int(stamp["nanosec"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _metadata_matches_pose(self) -> bool:
        return bool(
            self.cup_target is not None
            and self.grasp_metadata is not None
            and self._metadata_stamp(self.grasp_metadata)
            == self._stamp_tuple(self.cup_target)
            and time.monotonic() - self.grasp_metadata_wall_time
            <= float(self.get_parameter("metadata_max_age_s").value)
        )

    def candidate_options(
        self, primary_pose: PoseStamped, metadata: dict
    ) -> list[tuple[PoseStamped, dict]]:
        candidates = metadata.get("depth_validated_candidates")
        if not isinstance(candidates, list) or not candidates:
            return [(deepcopy(primary_pose), deepcopy(metadata))]
        if len(candidates) > 20:
            raise RuntimeError("Perception published too many grasp candidates")
        options = []
        copied_fields = (
            "rank",
            "score",
            "opening_m",
            "pixel",
            "angle_rad",
            "depth_support_ratio",
            "surface_residual_ratio",
            "valid_depth_points",
            "opening_evidence",
        )
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RuntimeError(f"Candidate option {index} is invalid")
            pose_data = candidate.get("pose")
            if not isinstance(pose_data, dict):
                raise RuntimeError(f"Candidate option {index} has no pose")
            position = self._finite_vector(
                pose_data.get("position_xyz_m"),
                3,
                f"candidate option {index} position",
            )
            quaternion = self._finite_vector(
                pose_data.get("orientation_xyzw"),
                4,
                f"candidate option {index} orientation",
            )
            option_pose = stamped_pose(
                self.planning_frame, position, quaternion
            )
            option_pose.header.stamp = deepcopy(primary_pose.header.stamp)
            option_metadata = deepcopy(metadata)
            for field in copied_fields:
                if field not in candidate:
                    raise RuntimeError(
                        f"Candidate option {index} is missing {field}"
                    )
                option_metadata[field] = deepcopy(candidate[field])
            option_metadata["candidate_variant"] = "network_surface"
            if bool(
                self.get_parameter("refine_ground_supported_candidate").value
            ):
                refined = self.refine_candidate_along_object_axis(
                    option_pose, option_metadata
                )
                if refined is not None:
                    options.append(refined)
            options.append((option_pose, option_metadata))
        return options

    def wait_for_inputs(self, execute: bool, timeout: float = 20.0) -> None:
        super().wait_for_inputs(execute, timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if self._metadata_matches_pose():
                return
        raise RuntimeError(
            "No fresh timestamp-matched Grasp-Anything pose and metadata"
        )

    @staticmethod
    def _finite_vector(values, length: int, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (length,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Invalid {label}: {values!r}")
        return vector

    def segmented_box_data(self, geometry: dict) -> list[dict]:
        boxes = geometry.get("boxes")
        if not isinstance(boxes, list) or len(boxes) < 2:
            raise RuntimeError("Axis refinement requires segmented depth boxes")
        padding = max(0.0, float(geometry.get("padding_m", 0.0)))
        parsed = []
        reference_axis = None
        for index, box in enumerate(boxes):
            center = self._finite_vector(
                box.get("center_xyz_m"), 3, f"collision box {index} center"
            )
            padded_size = self._finite_vector(
                box.get("size_xyz_m"), 3, f"collision box {index} size"
            )
            quaternion = self._finite_vector(
                box.get("orientation_xyzw"),
                4,
                f"collision box {index} orientation",
            )
            rotation = Rotation.from_quat(quaternion).as_matrix()
            axis = rotation[:, 0]
            if reference_axis is None:
                reference_axis = axis
            elif abs(float(np.dot(reference_axis, axis))) < 0.98:
                raise RuntimeError("Segmented collision boxes disagree on object axis")
            parsed.append(
                {
                    "index": index,
                    "center": center,
                    "padded_size": padded_size,
                    "size": np.maximum(padded_size - 2.0 * padding, 0.001),
                    "rotation": rotation,
                }
            )
        return parsed

    @staticmethod
    def closing_axis_perpendicular_to_object(
        raw_closing: np.ndarray, object_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        axis_xy = object_axis.copy()
        axis_xy[2] = 0.0
        if np.linalg.norm(axis_xy) < 0.50:
            raise RuntimeError("Detected object axis is too vertical for axial grasp")
        axis_xy /= np.linalg.norm(axis_xy)
        closing = raw_closing.copy()
        closing[2] = 0.0
        closing -= np.dot(closing, axis_xy) * axis_xy
        if np.linalg.norm(closing) < 0.10:
            closing = np.array([-axis_xy[1], axis_xy[0], 0.0])
        closing /= np.linalg.norm(closing)
        if np.dot(closing, raw_closing) < 0.0:
            closing = -closing
        return closing, axis_xy

    def refine_candidate_along_object_axis(
        self, pose: PoseStamped, metadata: dict
    ) -> tuple[PoseStamped, dict] | None:
        """Slide an end grasp toward a feasible depth-derived body band.

        Grasp-Anything still selects the surface grasp and closing direction.
        The registered boxes only move that point along a ground-supported,
        elongated object's principal axis, while respecting the 80 mm stroke.
        """
        geometry = metadata["object_region"]["collision_geometry"]
        boxes = self.segmented_box_data(geometry)
        candidate = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=float,
        )
        quaternion = self._finite_vector(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            4,
            "candidate quaternion",
        )
        raw_closing = Rotation.from_quat(quaternion).as_matrix()[:, 0]
        object_axis = boxes[0]["rotation"][:, 0]
        closing, _axis_xy = self.closing_axis_perpendicular_to_object(
            raw_closing, object_axis
        )

        for box in boxes:
            local = box["rotation"].T @ (candidate - box["center"])
            box["candidate_distance"] = float(
                np.linalg.norm(
                    local / np.maximum(box["padded_size"] / 2.0, 1e-6)
                )
            )
            box["closing_width"] = float(
                np.sum(np.abs(box["rotation"].T @ closing) * box["size"])
            )
        source = min(boxes, key=lambda item: item["candidate_distance"])
        object_center = self._finite_vector(
            geometry.get("center_xyz_m"), 3, "collision box center"
        )
        toward_center = float(
            np.dot(object_center - source["center"], object_axis)
        )
        if abs(toward_center) < 1e-6:
            return None
        direction = math.copysign(1.0, toward_center)
        maximum_width = float(
            self.get_parameter("maximum_refined_object_width_m").value
        )
        jaw_span = 2.0 * float(self.get_parameter("jaw_open_m").value)
        if not 0.0 < maximum_width < jaw_span:
            raise RuntimeError(
                "maximum_refined_object_width_m must be below the PGI stroke"
            )
        maximum_shift = float(
            self.get_parameter("maximum_candidate_axial_shift_m").value
        )
        minimum_shift = float(
            self.get_parameter("minimum_candidate_axial_shift_m").value
        )
        toward_boxes = []
        for box in boxes:
            shift = float(
                np.dot(box["center"] - source["center"], object_axis)
            )
            if direction * shift < minimum_shift or abs(shift) > maximum_shift:
                continue
            if direction * shift > abs(toward_center) + 1e-6:
                continue
            toward_boxes.append((direction * shift, box, shift))
        toward_boxes.sort(key=lambda item: item[0])

        selections = []
        previous_shift = 0.0
        previous_width = float(source["closing_width"])
        for _progress, box, shift in toward_boxes:
            width = float(box["closing_width"])
            if width <= maximum_width:
                selections.append(
                    (abs(toward_center - shift), box, shift, width, False)
                )
                previous_shift, previous_width = shift, width
                continue
            if previous_width < maximum_width and width > previous_width:
                ratio = (maximum_width - previous_width) / (
                    width - previous_width
                )
                interpolated_shift = previous_shift + ratio * (
                    shift - previous_shift
                )
                selections.append(
                    (
                        abs(toward_center - interpolated_shift),
                        box,
                        interpolated_shift,
                        maximum_width,
                        True,
                    )
                )
            break
        if not selections:
            return None
        (
            _distance_to_center,
            target,
            axial_shift,
            target_width,
            interpolated,
        ) = min(
            selections, key=lambda item: item[0]
        )
        refined_position = candidate + axial_shift * object_axis
        refined_pose = deepcopy(pose)
        (
            refined_pose.pose.position.x,
            refined_pose.pose.position.y,
            refined_pose.pose.position.z,
        ) = refined_position
        refined_metadata = deepcopy(metadata)
        refined_metadata["candidate_variant"] = "depth_axis_refined"
        refined_metadata["depth_box_refinement"] = {
            "source_box_index": int(source["index"]),
            "target_box_index": int(target["index"]),
            "axial_shift_m": float(axial_shift),
            "interpolated_between_boxes": bool(interpolated),
            "object_axis_base": object_axis.tolist(),
            "closing_axis_base": closing.tolist(),
            "target_local_width_m": float(target_width),
            "network_surface_position_xyz_m": candidate.tolist(),
            "refined_surface_position_xyz_m": refined_position.tolist(),
        }
        return refined_pose, refined_metadata

    def grasp_axes(
        self, pose: PoseStamped, metadata: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quaternion = self._finite_vector(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            4,
            "candidate quaternion",
        )
        quaternion /= np.linalg.norm(quaternion)
        raw_rotation = Rotation.from_quat(quaternion).as_matrix()
        raw_closing = raw_rotation[:, 0]
        raw_approach = raw_rotation[:, 2]
        if not bool(
            self.get_parameter("ground_supported_planar_approach").value
        ):
            return quaternion, raw_closing, raw_approach

        geometry = metadata["object_region"]["collision_geometry"]
        center = self._finite_vector(
            geometry["center_xyz_m"], 3, "collision box center"
        )
        candidate = np.array(
            [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ],
            dtype=float,
        )
        toward_object = center - candidate
        toward_object[2] = 0.0
        axis_aligned = bool(
            self.get_parameter("ground_supported_axis_aligned_approach").value
        )
        if axis_aligned:
            boxes = self.segmented_box_data(geometry)
            closing, approach_xy = self.closing_axis_perpendicular_to_object(
                raw_closing, boxes[0]["rotation"][:, 0]
            )
        else:
            approach_xy = raw_approach.copy()
            approach_xy[2] = 0.0
            if np.linalg.norm(approach_xy) < 0.10:
                approach_xy = toward_object.copy()
            if np.linalg.norm(approach_xy) < 0.10:
                raise RuntimeError(
                    "Cannot derive a horizontal approach from the "
                    "candidate/object geometry"
                )
            approach_xy /= np.linalg.norm(approach_xy)
            closing = raw_closing.copy()
            closing[2] = 0.0
            closing -= np.dot(closing, approach_xy) * approach_xy
            if np.linalg.norm(closing) < 0.10:
                closing = np.array([-approach_xy[1], approach_xy[0], 0.0])
            closing /= np.linalg.norm(closing)
            if np.dot(closing, raw_closing) < 0.0:
                closing = -closing
        if np.dot(approach_xy, toward_object) < 0.0:
            approach_xy = -approach_xy
        preserve_surface_approach = bool(
            self.get_parameter("preserve_depth_surface_approach").value
        )
        if preserve_surface_approach:
            approach = raw_approach / np.linalg.norm(raw_approach)
            if approach[2] > -0.20 or np.dot(approach, toward_object) <= 0.0:
                raise RuntimeError(
                    "Depth surface approach is not a downward, inward approach"
                )
        else:
            down_angle_deg = float(
                self.get_parameter("ground_supported_approach_down_deg").value
            )
            if (
                not math.isfinite(down_angle_deg)
                or not 0.0 <= down_angle_deg <= 85.0
            ):
                raise RuntimeError(
                    "ground_supported_approach_down_deg must be in [0, 85]"
                )
            down_angle = math.radians(down_angle_deg)
            approach = np.array(
                [
                    math.cos(down_angle) * approach_xy[0],
                    math.cos(down_angle) * approach_xy[1],
                    -math.sin(down_angle),
                ]
            )
        local_y = np.cross(approach, closing)
        local_y /= np.linalg.norm(local_y)
        # A parallel-jaw closing line is undirected. Choose its equivalent
        # 180-degree spin about the approach axis with local +Y upward; this
        # keeps the camera/interposer on the reachable side of a low object.
        if local_y[2] < 0.0:
            closing = -closing
            local_y = -local_y
        planar_quaternion = Rotation.from_matrix(
            np.column_stack((closing, local_y, approach))
        ).as_quat()
        return planar_quaternion, closing, approach

    def local_depth_width(
        self, candidate: np.ndarray, closing_axis: np.ndarray, geometry: dict
    ) -> tuple[float, int]:
        boxes = geometry.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            return math.inf, -1
        ranked = []
        for index, box in enumerate(boxes):
            center = self._finite_vector(
                box.get("center_xyz_m"), 3, f"collision box {index} center"
            )
            size = self._finite_vector(
                box.get("size_xyz_m"), 3, f"collision box {index} size"
            )
            quaternion = self._finite_vector(
                box.get("orientation_xyzw"),
                4,
                f"collision box {index} orientation",
            )
            rotation = Rotation.from_quat(quaternion).as_matrix()
            local = rotation.T @ (candidate - center)
            normalized_distance = float(
                np.linalg.norm(local / np.maximum(size / 2.0, 1e-6))
            )
            ranked.append((normalized_distance, index, size, rotation))
        _, selected_index, padded_size, rotation = min(ranked, key=lambda item: item[0])
        padding = max(0.0, float(geometry.get("padding_m", 0.0)))
        unpadded_size = np.maximum(padded_size - 2.0 * padding, 0.001)
        width = float(
            np.sum(np.abs(rotation.T @ closing_axis) * unpadded_size)
        )
        return width, selected_index

    def plan_cartesian(self, start, waypoints):
        """Add endpoint IK evidence to rejected image-derived segments."""
        try:
            return super().plan_cartesian(start, waypoints)
        except RuntimeError as original_error:
            if not waypoints:
                raise
            collision_solution, collision_code = self.ik_pose(
                waypoints[-1], start, avoid_collisions=True
            )
            free_solution, free_code = self.ik_pose(
                waypoints[-1], start, avoid_collisions=False
            )
            free_validity = (
                self.state_validity(free_solution)
                if free_solution is not None
                else None
            )
            raise RuntimeError(
                f"{original_error}; endpoint collision-aware IK code="
                f"{collision_code}, unconstrained code={free_code}, "
                f"unconstrained state={free_validity}"
            ) from original_error

    def activate_candidate(self, pose: PoseStamped, metadata: dict) -> dict:
        if pose.header.frame_id != self.planning_frame:
            raise RuntimeError(
                f"Candidate must be in {self.planning_frame}, got "
                f"{pose.header.frame_id}"
            )
        if metadata.get("observation_only") is not True:
            raise RuntimeError("Grasp-Anything metadata lost observation-only marker")
        if metadata.get("model_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("Grasp-Anything model checksum is not the pinned model")
        if self._metadata_stamp(metadata) != self._stamp_tuple(pose):
            raise RuntimeError("Candidate pose and metadata timestamps do not match")

        opening_m = float(metadata.get("opening_m", math.nan))
        jaw_open_m = float(self.get_parameter("jaw_open_m").value)
        if not math.isfinite(opening_m) or not 0.0 < opening_m <= 2.0 * jaw_open_m:
            raise RuntimeError(
                f"Candidate opening {opening_m!r} exceeds the PGI jaw range"
            )
        evidence = metadata.get("opening_evidence")
        if not isinstance(evidence, dict):
            raise RuntimeError("Candidate has no physical opening evidence")
        clearance_m = float(evidence.get("clearance_m", math.nan))
        if not math.isfinite(clearance_m) or not 0.0 <= clearance_m < opening_m:
            raise RuntimeError("Candidate opening clearance is invalid")
        object_width_m = opening_m - clearance_m
        minimum_width = float(self.get_parameter("minimum_object_width_m").value)
        if object_width_m < minimum_width:
            raise RuntimeError(
                f"Object width {object_width_m:.4f} m is below {minimum_width:.4f} m"
            )
        region = metadata.get("object_region")
        geometry = region.get("collision_geometry") if isinstance(region, dict) else None
        if not isinstance(geometry, dict) or geometry.get("type") not in {
            "box",
            "multi_box",
        }:
            raise RuntimeError("Candidate has no registered-depth collision box")
        if geometry.get("frame_id") != self.planning_frame:
            raise RuntimeError("Collision box is not in the MoveIt planning frame")
        center = self._finite_vector(
            geometry.get("center_xyz_m"), 3, "collision box center"
        )
        size = self._finite_vector(
            geometry.get("size_xyz_m"), 3, "collision box size"
        )
        maximum_extent = float(self.get_parameter("maximum_object_extent_m").value)
        if np.any(size <= 0.0) or np.any(size > maximum_extent):
            raise RuntimeError(f"Collision box size is outside bounds: {size.tolist()}")

        candidate_position = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=float,
        )
        padding = float(geometry.get("padding_m", 0.0))
        half_size = size / 2.0 + max(0.0, padding)
        if np.any(np.abs(candidate_position - center) > half_size + 1e-6):
            raise RuntimeError("Candidate point is outside its depth collision box")

        quaternion = self._finite_vector(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            4,
            "candidate quaternion",
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 0.99 or quaternion_norm > 1.01:
            raise RuntimeError(
                f"Candidate quaternion norm {quaternion_norm:.6f} is invalid"
            )
        inset = float(self.get_parameter("surface_inset_m").value)
        if not 0.0 <= inset <= 0.025:
            raise RuntimeError("surface_inset_m must be in [0, 0.025]")

        grasp_quaternion, closing_axis, approach_axis = self.grasp_axes(
            pose, metadata
        )
        local_width_m, local_box_index = self.local_depth_width(
            candidate_position, closing_axis, geometry
        )
        refinement = metadata.get("depth_box_refinement")
        if isinstance(refinement, dict):
            expected_width = float(
                refinement.get("target_local_width_m", math.nan)
            )
            if not math.isfinite(local_width_m) or not math.isfinite(expected_width):
                raise RuntimeError("Refined candidate lost its local depth width")
            interpolated = bool(
                refinement.get("interpolated_between_boxes", False)
            )
            if not interpolated and abs(local_width_m - expected_width) > 0.015:
                raise RuntimeError(
                    "Refined candidate resolved to an inconsistent depth band"
                )
            object_width_m = expected_width if interpolated else local_width_m
        elif math.isfinite(local_width_m):
            object_width_m = min(object_width_m, local_width_m)
        if object_width_m < minimum_width:
            raise RuntimeError(
                f"Local object width {object_width_m:.4f} m is below "
                f"{minimum_width:.4f} m"
            )
        overlap = float(self.get_parameter("contact_overlap_m").value)
        close_m = object_width_m / 2.0 - overlap
        if not 0.0 <= close_m < jaw_open_m:
            raise RuntimeError(
                f"Derived per-jaw close target {close_m:.4f} m is invalid"
            )

        self.active_metadata = deepcopy(metadata)
        self.contact_close_m = close_m
        self.planar_axes_contract = {
            "ground_supported_planar_approach": bool(
                self.get_parameter("ground_supported_planar_approach").value
            ),
            "ground_supported_axis_aligned_approach": bool(
                self.get_parameter(
                    "ground_supported_axis_aligned_approach"
                ).value
            ),
            "candidate_variant": metadata.get(
                "candidate_variant", "network_surface"
            ),
            "depth_box_refinement": deepcopy(refinement),
            "approach_down_deg": math.degrees(
                math.asin(float(np.clip(-approach_axis[2], -1.0, 1.0)))
            ),
            "depth_surface_approach_preserved": bool(
                self.get_parameter("preserve_depth_surface_approach").value
            ),
            "closing_axis_base": closing_axis.tolist(),
            "approach_axis_base": approach_axis.tolist(),
            "grasp_orientation_xyzw": grasp_quaternion.tolist(),
            "local_depth_box_index": local_box_index,
            "local_depth_width_m": (
                local_width_m if math.isfinite(local_width_m) else None
            ),
        }
        return {
            "source": "grasp_anything_registered_depth",
            "opening_m": opening_m,
            "object_width_m": object_width_m,
            "derived_close_m_per_jaw": close_m,
            "surface_inset_m": inset,
            "score": float(metadata.get("score", math.nan)),
            "collision_box_center_xyz_m": center.tolist(),
            "collision_box_size_xyz_m": size.tolist(),
            "single_view_collision_geometry": True,
            "axis_completion": deepcopy(self.planar_axes_contract),
        }

    def candidate_poses(self, cup_target: PoseStamped) -> dict:
        if self.active_metadata is None:
            raise RuntimeError("No frozen Grasp-Anything metadata")
        surface = np.array(
            [
                cup_target.pose.position.x,
                cup_target.pose.position.y,
                cup_target.pose.position.z,
            ],
            dtype=float,
        )
        quaternion, closing_axis, approach_axis = self.grasp_axes(
            cup_target, self.active_metadata
        )
        rotation = Rotation.from_quat(quaternion).as_matrix()
        if not np.allclose(rotation[:, 0], closing_axis, atol=1e-6) or not np.allclose(
            rotation[:, 2], approach_axis, atol=1e-6
        ):
            raise RuntimeError("Completed grasp axes are internally inconsistent")
        inset = float(self.get_parameter("surface_inset_m").value)
        grasp = surface + inset * approach_axis
        pregrasp = grasp - float(
            self.get_parameter("pregrasp_backoff_m").value
        ) * approach_axis
        staging = pregrasp + np.array(
            [0.0, 0.0, float(self.get_parameter("staging_lift_m").value)]
        )
        ready_extra = max(
            0.0,
            float(self.get_parameter("side_ready_backoff_m").value)
            - float(self.get_parameter("pregrasp_backoff_m").value),
        )
        side_ready = pregrasp - ready_extra * approach_axis
        side_ready[2] = max(
            side_ready[2],
            float(self.get_parameter("side_ready_height_m").value),
        )
        transfer = side_ready.copy()
        transfer[2] = max(
            transfer[2],
            float(self.get_parameter("cartesian_transfer_height_m").value),
        )
        lift = grasp + np.array(
            [0.0, 0.0, float(self.get_parameter("lift_height_m").value)]
        )
        release = grasp + np.array(
            [0.0, 0.0, float(self.get_parameter("release_height_m").value)]
        )
        poses = {
            "cup": surface,
            "surface": surface,
            "axis": approach_axis,
            "closing_axis": closing_axis,
            "quaternion": quaternion,
            "side_ready": stamped_pose(
                self.planning_frame, side_ready, quaternion
            ),
            "transfer": stamped_pose(self.planning_frame, transfer, quaternion),
            "staging": stamped_pose(self.planning_frame, staging, quaternion),
            "pregrasp": stamped_pose(self.planning_frame, pregrasp, quaternion),
            "grasp": stamped_pose(self.planning_frame, grasp, quaternion),
            "lift": stamped_pose(self.planning_frame, lift, quaternion),
            "release": stamped_pose(self.planning_frame, release, quaternion),
            "approach_azimuth_deg": math.degrees(
                math.atan2(approach_axis[1], approach_axis[0])
            ),
            "approach_down_angle_deg": math.degrees(
                math.asin(float(np.clip(-approach_axis[2], -1.0, 1.0)))
            ),
        }
        if bool(self.get_parameter("use_radial_transit_orientation").value):
            radial_azimuth = math.atan2(grasp[1], grasp[0])
            radial = np.array(
                [math.cos(radial_azimuth), math.sin(radial_azimuth), 0.0]
            )
            transit_closing = np.array(
                [-math.sin(radial_azimuth), math.cos(radial_azimuth), 0.0]
            )
            down_angle = math.radians(
                float(
                    self.get_parameter(
                        "ground_supported_approach_down_deg"
                    ).value
                )
            )
            transit_approach = np.array(
                [
                    math.cos(down_angle) * radial[0],
                    math.cos(down_angle) * radial[1],
                    -math.sin(down_angle),
                ]
            )
            transit_y = np.cross(transit_approach, transit_closing)
            transit_quaternion = Rotation.from_matrix(
                np.column_stack(
                    (transit_closing, transit_y, transit_approach)
                )
            ).as_quat()
            poses["transfer"] = stamped_pose(
                self.planning_frame, transfer, transit_quaternion
            )
            poses["side_ready"] = stamped_pose(
                self.planning_frame, side_ready, transit_quaternion
            )
            poses["transit_orientation_source"] = (
                "validated_radial_side_branch"
            )
        else:
            poses["transit_orientation_source"] = "grasp_anything"
        return poses

    def evaluate_top_grasp(self, seed, cup_target: PoseStamped) -> dict:
        del seed, cup_target
        return {
            "feasible": True,
            "source": "grasp_anything_candidate",
            "reason": (
                "The learned closing direction and registered-depth approach "
                "replace the fixed upright-cup top/side heuristic."
            ),
        }

    def depth_collision_object(self, metadata: dict) -> CollisionObject:
        geometry = metadata["object_region"]["collision_geometry"]
        center = self._finite_vector(
            geometry["center_xyz_m"], 3, "collision box center"
        )
        size = self._finite_vector(geometry["size_xyz_m"], 3, "collision box size")
        quaternion = self._finite_vector(
            geometry["orientation_xyzw"], 4, "collision box orientation"
        )
        world_object = CollisionObject()
        world_object.header.frame_id = self.planning_frame
        world_object.id = self.cup_id
        world_object.operation = CollisionObject.ADD
        boxes = geometry.get("boxes")
        if geometry.get("type") == "box":
            boxes = [
                {
                    "center_xyz_m": center.tolist(),
                    "size_xyz_m": size.tolist(),
                    "orientation_xyzw": quaternion.tolist(),
                }
            ]
        if not isinstance(boxes, list) or not boxes:
            raise RuntimeError("Depth collision geometry has no boxes")
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise RuntimeError(f"Collision box {index} is invalid")
            box_center = self._finite_vector(
                box.get("center_xyz_m"), 3, f"collision box {index} center"
            )
            box_size = self._finite_vector(
                box.get("size_xyz_m"), 3, f"collision box {index} size"
            )
            box_quaternion = self._finite_vector(
                box.get("orientation_xyzw"),
                4,
                f"collision box {index} orientation",
            )
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = box_size.tolist()
            primitive_pose = Pose()
            (
                primitive_pose.position.x,
                primitive_pose.position.y,
                primitive_pose.position.z,
            ) = box_center
            (
                primitive_pose.orientation.x,
                primitive_pose.orientation.y,
                primitive_pose.orientation.z,
                primitive_pose.orientation.w,
            ) = box_quaternion
            world_object.primitives.append(primitive)
            world_object.primitive_poses.append(primitive_pose)
        return world_object

    def replace_moveit_object(self, world_object: CollisionObject) -> CollisionObject:
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [deepcopy(world_object)]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        if not self.call(self.apply_scene_client, request).success:
            raise RuntimeError("MoveIt rejected the depth-derived collision object")
        replaced = self.get_cup_object()
        if not replaced.primitives:
            raise RuntimeError("MoveIt did not retain the depth collision boxes")
        return replaced

    def set_initial_object_pose(self, model_pose: Pose) -> None:
        quaternion = np.array(
            [
                model_pose.orientation.x,
                model_pose.orientation.y,
                model_pose.orientation.z,
                model_pose.orientation.w,
            ],
            dtype=float,
        )
        if np.linalg.norm(quaternion) < 1e-9:
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.initial_object_rotation = Rotation.from_quat(quaternion)
        self.initial_model_z = float(model_pose.position.z)

    def require_upright(self, pose, stage: str) -> None:
        if self.initial_object_rotation is None:
            raise RuntimeError("Initial object orientation was not frozen")
        quaternion = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )
        if np.linalg.norm(quaternion) < 1e-9:
            raise RuntimeError(f"Object quaternion disappeared at {stage}")
        current = Rotation.from_quat(quaternion)
        drift = math.degrees(
            (self.initial_object_rotation.inv() * current).magnitude()
        )
        limit = float(self.get_parameter("maximum_cup_tilt_deg").value)
        if drift > limit:
            raise RuntimeError(
                f"Object orientation drift at {stage} is {drift:.2f} deg, "
                f"above {limit:.2f} deg"
            )

    def maximum_release_model_z(self, initial_model_pose) -> float:
        return float(initial_model_pose.position.z) + float(
            self.get_parameter("maximum_release_height_m").value
        )

    def begin_support_contact_planning(self) -> None:
        """Temporarily permit only the detected object/support contact pair.

        A physically supported object is already touching the Gazebo ground at
        the start of a lift. MoveIt must permit that one initial contact to
        produce a Cartesian lift, while robot/ground and every other collision
        pair remain checked.
        """
        if self.saved_support_allowed_collision_matrix is not None:
            raise RuntimeError("Support-contact ACM override is already active")
        support = str(self.get_parameter("support_surface_id").value)
        if not support:
            raise RuntimeError("support_surface_id must not be empty")
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        )
        original = deepcopy(
            self.call(self.scene_client, request).scene.allowed_collision_matrix
        )
        modified = deepcopy(original)
        self.set_acm_pair(modified, self.cup_id, support, True)
        self.apply_allowed_collision_matrix(modified)
        self.saved_support_allowed_collision_matrix = original
        self.publish_status(
            "support_contact_acm_enabled",
            allowed_pairs=[[self.cup_id, support]],
        )

    def end_support_contact_planning(self) -> None:
        if self.saved_support_allowed_collision_matrix is None:
            return
        original = self.saved_support_allowed_collision_matrix
        self.apply_allowed_collision_matrix(original)
        self.saved_support_allowed_collision_matrix = None
        self.publish_status("support_contact_acm_restored")

    def execute_contact_close(self) -> dict:
        if self.contact_close_m is None:
            raise RuntimeError("No Grasp-Anything-derived close target")
        original = float(self.get_parameter("physical_close_m").value)
        result = self.set_parameters(
            [Parameter("physical_close_m", value=self.contact_close_m)]
        )[0]
        if not result.successful:
            raise RuntimeError(f"Could not set derived close target: {result.reason}")
        try:
            report = super().execute_contact_close()
            report["source"] = "registered_depth_object_width"
            return report
        finally:
            self.set_parameters([Parameter("physical_close_m", value=original)])

    def validate_ground_support(self, metadata: dict, model_pose: Pose) -> dict:
        geometry = metadata["object_region"]["collision_geometry"]
        center = np.asarray(geometry["center_xyz_m"], dtype=float)
        size = np.asarray(geometry["size_xyz_m"], dtype=float)
        visible_box_min_z = float(center[2] - size[2] / 2.0)
        tolerance = float(self.get_parameter("ground_support_tolerance_m").value)
        if abs(visible_box_min_z) > tolerance:
            raise RuntimeError(
                f"Depth object is not ground-supported: min_z={visible_box_min_z:.4f} m"
            )
        model_origin = np.array(
            [model_pose.position.x, model_pose.position.y, model_pose.position.z]
        )
        origin_tolerance = float(
            self.get_parameter("model_origin_box_tolerance_m").value
        )
        inside = np.abs(model_origin - center) <= size / 2.0 + origin_tolerance
        if not bool(np.all(inside)):
            raise RuntimeError(
                "Gazebo model origin is inconsistent with the depth collision box"
            )
        return {
            "visible_box_min_z_m": visible_box_min_z,
            "gazebo_model_origin_consistent": True,
        }


def main() -> int:
    args = parse_args()
    if args.execute_sim and not args.confirm_simulation:
        print("--execute-sim requires --confirm-simulation", file=sys.stderr)
        return 2
    rclpy.init(args=sys.argv)
    node = GraspAnythingMoveItDemo()
    try:
        node.wait_for_inputs(args.execute_sim)
        node.wait_for_services(args.execute_sim)
        guards = node.verify_guards(args.execute_sim)
        frozen_target = deepcopy(node.cup_target)
        frozen_metadata = deepcopy(node.grasp_metadata)
        model_pose = deepcopy(node.cup_model_pose)
        if frozen_target is None or frozen_metadata is None:
            raise RuntimeError("Grasp-Anything candidate disappeared")
        if args.execute_sim and model_pose is None:
            raise RuntimeError("Dynamic Gazebo object pose is unavailable")
        if model_pose is not None:
            node.set_initial_object_pose(model_pose)
            node.require_upright(model_pose, "initial")
            support = node.validate_ground_support(frozen_metadata, model_pose)
        else:
            support = None

        world_object = node.depth_collision_object(frozen_metadata)
        cup_object = node.replace_moveit_object(world_object)
        candidate_failures = []
        candidate_contract = None
        preflight = None
        selected_metadata = None
        options = node.candidate_options(frozen_target, frozen_metadata)
        for option_pose, option_metadata in options:
            rank = int(option_metadata.get("rank", -1))
            node.publish_status(
                "grasp_anything_candidate_preflight_started",
                execution=args.execute_sim,
                rank=rank,
                score=option_metadata.get("score"),
            )
            try:
                option_contract = node.activate_candidate(
                    option_pose, option_metadata
                )
                option_preflight = node.preflight(option_pose, cup_object)
            except RuntimeError as error:
                candidate_failures.append(
                    {"rank": rank, "reason": str(error)}
                )
                node.publish_status(
                    "grasp_anything_candidate_rejected_by_moveit",
                    rank=rank,
                    reason=str(error),
                )
                continue
            candidate_contract = option_contract
            preflight = option_preflight
            selected_metadata = option_metadata
            node.publish_status(
                "grasp_anything_candidate_selected_by_moveit",
                rank=rank,
                attempted_candidates=len(candidate_failures) + 1,
            )
            break
        if candidate_contract is None or preflight is None or selected_metadata is None:
            raise RuntimeError(
                "No depth-valid Grasp-Anything candidate passed MoveIt: "
                f"{candidate_failures}"
            )
        candidate_contract["moveit_selection"] = {
            "selected_rank": int(selected_metadata.get("rank", -1)),
            "candidate_count": len(options),
            "rejected_before_selection": candidate_failures,
        }
        report = {
            "success": True,
            "mode": "execute_sim" if args.execute_sim else "plan_only",
            "stage": "grasp_anything_moveit_bridge",
            "simulation_only": True,
            "real_robot_command_sent": False,
            "guards": guards,
            "candidate_contract": candidate_contract,
            "ground_support": support,
            "limitations": [
                "single_view_depth_collision_box",
                "unknown hidden-surface geometry",
                "simulation cup mass remains 0.15 kg",
                "candidate is not approved for real execution",
            ],
            **json_safe_report(preflight),
        }
        if args.execute_sim:
            assert model_pose is not None
            report["execution"] = node.execute_physical_workflow(
                preflight, cup_object, model_pose
            )
        else:
            report["execution"] = {
                "attempted": False,
                "trajectory_sent": False,
                "controller_switched": False,
            }
            node.publish_status("grasp_anything_plan_only_complete", success=True)
        write_report_json(args.report_json, report)
        if not args.suppress_console_report:
            print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        failure = {
            "success": False,
            "error": str(error),
            "moveit_object_may_remain_attached": node.physical_moveit_attached,
            "simulation_only": True,
            "real_robot_command_sent": False,
        }
        if node.partial_execution_report:
            failure["execution_partial"] = node.partial_execution_report
        write_report_json(args.report_json, failure)
        if not args.suppress_console_report:
            print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
