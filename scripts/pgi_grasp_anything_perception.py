#!/usr/bin/env python3
"""Observation-only ROS adapter for the official Grasp-Anything RGB model.

The neural network runs in an isolated local HTTP service because its PyTorch
environment is intentionally separate from ROS 2.  This node synchronizes the
registered RGB, depth, and CameraInfo streams, asks for 2-D grasp proposals,
and reconstructs candidates in 3-D from depth.  It never plans or commands
motion and publishes on topics separate from the existing cup detector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
from geometry_msgs.msg import PoseStamped
import message_filters
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.perception.grasp_anything_geometry import (  # noqa: E402
    GraspProposal2D,
    GraspReconstructionError,
    ReconstructedGrasp,
    reconstruct_grasp_from_depth,
)


class GraspAnythingPerception(Node):
    def __init__(self) -> None:
        super().__init__("pgi_grasp_anything_perception")
        defaults = {
            "rgb_topic": "/pgi_d435i/image",
            "depth_topic": "/pgi_d435i/depth_image",
            "camera_info_topic": "/pgi_d435i/camera_info",
            "candidate_pose_topic": "/pgi/grasp_anything/candidate_pose",
            "candidate_metadata_topic": "/pgi/grasp_anything/candidate",
            "status_topic": "/pgi/grasp_anything/status",
            "debug_image_topic": "/pgi/grasp_anything/debug_image",
            "target_frame": "base_link",
            "service_url": "http://127.0.0.1:8765/v1/grasp",
            "max_rate_hz": 0.5,
            "service_timeout_s": 4.0,
            "sync_queue_size": 20,
            "sync_slop_sec": 0.08,
            "depth_scale_16uc1": 0.001,
            "min_depth_m": 0.05,
            "max_depth_m": 3.0,
            "patch_radius_px": 4,
            "local_depth_tolerance_m": 0.04,
            "min_depth_points": 30,
            "min_depth_support_ratio": 0.30,
            "max_surface_residual_ratio": 0.25,
            "min_opening_m": 0.005,
            "max_opening_m": 0.080,
            "minimum_model_score": 0.20,
            "self_mask_right_fraction": 0.73,
            "object_crop_enabled": True,
            "object_min_height_m": 0.008,
            "object_max_height_m": 0.35,
            "object_min_component_px": 100,
            "object_crop_size_px": 112,
            "object_crop_margin_px": 20,
            "object_mask_kernel_px": 5,
            "opening_clearance_m": 0.004,
            "collision_padding_m": 0.010,
            "collision_box_count": 8,
            "maximum_published_candidates": 5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._target_frame = str(self.get_parameter("target_frame").value)
        self._service_url = str(self.get_parameter("service_url").value)
        self._max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self._service_timeout_s = float(
            self.get_parameter("service_timeout_s").value
        )
        self._depth_scale = float(
            self.get_parameter("depth_scale_16uc1").value
        )
        self._minimum_score = float(
            self.get_parameter("minimum_model_score").value
        )
        self._self_mask_right_fraction = float(
            self.get_parameter("self_mask_right_fraction").value
        )
        self._object_crop_enabled = bool(
            self.get_parameter("object_crop_enabled").value
        )
        self._object_min_height = float(
            self.get_parameter("object_min_height_m").value
        )
        self._object_max_height = float(
            self.get_parameter("object_max_height_m").value
        )
        self._object_min_component_px = int(
            self.get_parameter("object_min_component_px").value
        )
        self._object_crop_size_px = int(
            self.get_parameter("object_crop_size_px").value
        )
        self._object_crop_margin_px = int(
            self.get_parameter("object_crop_margin_px").value
        )
        self._object_mask_kernel_px = int(
            self.get_parameter("object_mask_kernel_px").value
        )
        self._opening_clearance_m = float(
            self.get_parameter("opening_clearance_m").value
        )
        self._collision_padding_m = float(
            self.get_parameter("collision_padding_m").value
        )
        self._collision_box_count = int(
            self.get_parameter("collision_box_count").value
        )
        self._maximum_published_candidates = int(
            self.get_parameter("maximum_published_candidates").value
        )
        if not self._service_url.startswith("http://127.0.0.1:"):
            raise ValueError("service_url must use the loopback IPv4 address")
        if self._object_crop_enabled and self._target_frame != "base_link":
            raise ValueError(
                "object_crop_enabled requires target_frame=base_link because "
                "object height is measured from the base-frame ground plane"
            )
        if self._max_rate_hz <= 0.0:
            raise ValueError("max_rate_hz must be positive")
        if self._service_timeout_s <= 0.0:
            raise ValueError("service_timeout_s must be positive")
        if not 0.0 < self._self_mask_right_fraction <= 1.0:
            raise ValueError("self_mask_right_fraction must be in (0, 1]")
        if not self._object_min_height < self._object_max_height:
            raise ValueError("object height range is invalid")
        if self._object_min_component_px < 1:
            raise ValueError("object_min_component_px must be positive")
        if self._object_crop_size_px < 32:
            raise ValueError("object_crop_size_px must be at least 32")
        if self._object_crop_margin_px < 0:
            raise ValueError("object_crop_margin_px cannot be negative")
        if (
            self._object_mask_kernel_px < 1
            or self._object_mask_kernel_px % 2 == 0
        ):
            raise ValueError("object_mask_kernel_px must be a positive odd integer")
        if self._opening_clearance_m < 0.0:
            raise ValueError("opening_clearance_m cannot be negative")
        if not 0.0 <= self._collision_padding_m <= 0.050:
            raise ValueError("collision_padding_m must be in [0, 0.050]")
        if not 2 <= self._collision_box_count <= 16:
            raise ValueError("collision_box_count must be in [2, 16]")
        if not 1 <= self._maximum_published_candidates <= 20:
            raise ValueError("maximum_published_candidates must be in [1, 20]")

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        callback_group = MutuallyExclusiveCallbackGroup()
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("candidate_pose_topic").value),
            reliable_qos,
        )
        self._metadata_publisher = self.create_publisher(
            String,
            str(self.get_parameter("candidate_metadata_topic").value),
            reliable_qos,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            reliable_qos,
        )
        self._debug_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            image_qos,
        )
        self._rgb_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("rgb_topic").value),
            qos_profile=sensor_qos,
            callback_group=callback_group,
        )
        self._depth_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("depth_topic").value),
            qos_profile=sensor_qos,
            callback_group=callback_group,
        )
        self._info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile=sensor_qos,
            callback_group=callback_group,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._sync.registerCallback(self._on_images)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._last_process_time = 0.0
        self.get_logger().info(
            "Grasp-Anything RGB-D adapter ready; output is observation-only "
            "and cannot command motion"
        )

    def _publish_json(self, publisher, payload: dict) -> None:
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        publisher.publish(message)

    def _status(self, detected: bool, reason: str, **details) -> None:
        self._publish_json(
            self._status_publisher,
            {
                "detected": detected,
                "reason": reason,
                "observation_only": True,
                **details,
            },
        )

    @staticmethod
    def _rgb_array(message: Image) -> np.ndarray | None:
        encoding = message.encoding.lower()
        if encoding not in {"rgb8", "bgr8"}:
            return None
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step
        )
        image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
        return image[:, :, ::-1].copy() if encoding == "bgr8" else image.copy()

    def _depth_array(self, message: Image) -> np.ndarray | None:
        encoding = message.encoding.upper()
        if encoding == "32FC1":
            columns = message.step // np.dtype(np.float32).itemsize
            rows = np.frombuffer(message.data, dtype=np.float32).reshape(
                message.height, columns
            )
            return rows[:, : message.width]
        if encoding == "16UC1":
            columns = message.step // np.dtype(np.uint16).itemsize
            rows = np.frombuffer(message.data, dtype=np.uint16).reshape(
                message.height, columns
            )
            return rows[:, : message.width].astype(np.float32) * self._depth_scale
        return None

    def _request_proposals(self, rgb: np.ndarray) -> dict:
        success, encoded = cv2.imencode(
            ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RuntimeError("png_encoding_failed")
        request = Request(
            self._service_url,
            data=encoded.tobytes(),
            method="POST",
            headers={"Content-Type": "image/png"},
        )
        with urlopen(request, timeout=self._service_timeout_s) as response:
            if response.status != 200:
                raise RuntimeError(f"http_status_{response.status}")
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("success") is not True:
            raise RuntimeError(str(payload.get("reason", "service_refused")))
        if payload.get("protocol_version") != 1:
            raise RuntimeError("unsupported_protocol_version")
        if not isinstance(payload.get("candidates"), list):
            raise RuntimeError("invalid_candidate_list")
        return payload

    @staticmethod
    def _proposal_in_depth(
        proposal_rgb: GraspProposal2D,
        *,
        rgb_width: int,
        rgb_height: int,
        depth_width: int,
        depth_height: int,
    ) -> GraspProposal2D:
        values = (
            proposal_rgb.u,
            proposal_rgb.v,
            proposal_rgb.angle_rad,
            proposal_rgb.opening_px,
            proposal_rgb.score,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate contains a non-finite value")
        if not 0.0 <= proposal_rgb.u < rgb_width:
            raise ValueError("candidate u is outside the RGB image")
        if not 0.0 <= proposal_rgb.v < rgb_height:
            raise ValueError("candidate v is outside the RGB image")
        scale_x = depth_width / float(rgb_width)
        scale_y = depth_height / float(rgb_height)
        du_depth = math.cos(proposal_rgb.angle_rad) * scale_x
        dv_depth = -math.sin(proposal_rgb.angle_rad) * scale_y
        vector_scale = math.hypot(du_depth, dv_depth)
        if vector_scale < 1e-9:
            raise ValueError("candidate direction is degenerate")
        return GraspProposal2D(
            u=proposal_rgb.u * scale_x,
            v=proposal_rgb.v * scale_y,
            angle_rad=math.atan2(-dv_depth, du_depth),
            opening_px=proposal_rgb.opening_px * vector_scale,
            score=proposal_rgb.score,
        )

    @staticmethod
    def _depth_opening_in_rgb(
        proposal_rgb: GraspProposal2D,
        proposal_depth: GraspProposal2D,
        *,
        rgb_width: int,
        rgb_height: int,
        depth_width: int,
        depth_height: int,
    ) -> GraspProposal2D:
        scale_x = depth_width / float(rgb_width)
        scale_y = depth_height / float(rgb_height)
        vector_scale = math.hypot(
            math.cos(proposal_rgb.angle_rad) * scale_x,
            -math.sin(proposal_rgb.angle_rad) * scale_y,
        )
        if vector_scale < 1e-9:
            raise GraspReconstructionError("candidate image scale is degenerate")
        return GraspProposal2D(
            u=proposal_rgb.u,
            v=proposal_rgb.v,
            angle_rad=proposal_rgb.angle_rad,
            opening_px=proposal_depth.opening_px / vector_scale,
            score=proposal_rgb.score,
        )

    def _reconstruct(
        self,
        depth: np.ndarray,
        proposal: GraspProposal2D,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> ReconstructedGrasp:
        return reconstruct_grasp_from_depth(
            depth,
            proposal,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            patch_radius_px=int(self.get_parameter("patch_radius_px").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
            local_depth_tolerance_m=float(
                self.get_parameter("local_depth_tolerance_m").value
            ),
            min_depth_points=int(self.get_parameter("min_depth_points").value),
            min_depth_support_ratio=float(
                self.get_parameter("min_depth_support_ratio").value
            ),
            max_surface_residual_ratio=float(
                self.get_parameter("max_surface_residual_ratio").value
            ),
            min_opening_m=float(self.get_parameter("min_opening_m").value),
            max_opening_m=float(self.get_parameter("max_opening_m").value),
        )

    def _object_region(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        transform,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, int, int, np.ndarray, dict]:
        """Find the largest generic object above the base-frame ground plane."""
        valid = np.isfinite(depth)
        valid &= depth >= float(self.get_parameter("min_depth_m").value)
        valid &= depth <= float(self.get_parameter("max_depth_m").value)
        rows, columns = np.indices(depth.shape, dtype=np.float32)
        camera_x = (columns - cx) * depth / fx
        camera_y = (rows - cy) * depth / fy
        rotation_message = transform.transform.rotation
        rotation = Rotation.from_quat(
            [
                rotation_message.x,
                rotation_message.y,
                rotation_message.z,
                rotation_message.w,
            ]
        ).as_matrix()
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=float,
        )
        base_height = (
            rotation[2, 0] * camera_x
            + rotation[2, 1] * camera_y
            + rotation[2, 2] * depth
            + translation[2]
        )
        mask = valid
        mask &= base_height >= self._object_min_height
        mask &= base_height <= self._object_max_height
        self_mask_column = int(
            round(depth.shape[1] * self._self_mask_right_fraction)
        )
        mask[:, self_mask_column:] = False
        kernel = np.ones(
            (self._object_mask_kernel_px, self._object_mask_kernel_px),
            dtype=np.uint8,
        )
        cleaned = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        if label_count <= 1:
            raise GraspReconstructionError("no object above the ground plane")
        areas = stats[1:, cv2.CC_STAT_AREA]
        component_label = 1 + int(np.argmax(areas))
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        if area < self._object_min_component_px:
            raise GraspReconstructionError(
                f"largest depth object has {area} pixels; need "
                f"{self._object_min_component_px}"
            )
        component_depth = labels == component_label

        component_camera = np.column_stack(
            (
                camera_x[component_depth],
                camera_y[component_depth],
                depth[component_depth],
            )
        )
        component_target = component_camera @ rotation.T + translation
        visible_minimum = component_target.min(axis=0)
        visible_maximum = component_target.max(axis=0)
        collision_minimum = visible_minimum - self._collision_padding_m
        collision_maximum = visible_maximum + self._collision_padding_m
        collision_size = collision_maximum - collision_minimum
        collision_center = (collision_minimum + collision_maximum) / 2.0

        point_centroid = component_target.mean(axis=0)
        _u, _singular_values, principal_rows = np.linalg.svd(
            component_target - point_centroid, full_matrices=False
        )
        principal_axes = principal_rows.T
        if np.linalg.det(principal_axes) < 0.0:
            principal_axes[:, 2] *= -1.0
        principal_coordinates = (
            component_target - point_centroid
        ) @ principal_axes
        axis_minimum = float(principal_coordinates[:, 0].min())
        axis_maximum = float(principal_coordinates[:, 0].max())
        bin_edges = np.linspace(
            axis_minimum, axis_maximum, self._collision_box_count + 1
        )
        box_quaternion = Rotation.from_matrix(principal_axes).as_quat()
        collision_boxes = []
        for box_index in range(self._collision_box_count):
            if box_index + 1 == self._collision_box_count:
                selected_points = (
                    (principal_coordinates[:, 0] >= bin_edges[box_index])
                    & (principal_coordinates[:, 0] <= bin_edges[box_index + 1])
                )
            else:
                selected_points = (
                    (principal_coordinates[:, 0] >= bin_edges[box_index])
                    & (principal_coordinates[:, 0] < bin_edges[box_index + 1])
                )
            if int(np.count_nonzero(selected_points)) < 10:
                continue
            local_points = principal_coordinates[selected_points]
            local_minimum = local_points.min(axis=0) - self._collision_padding_m
            local_maximum = local_points.max(axis=0) + self._collision_padding_m
            local_center = (local_minimum + local_maximum) / 2.0
            local_size = local_maximum - local_minimum
            world_center = point_centroid + principal_axes @ local_center
            collision_boxes.append(
                {
                    "center_xyz_m": [
                        round(float(value), 6) for value in world_center
                    ],
                    "size_xyz_m": [
                        round(float(value), 6) for value in local_size
                    ],
                    "orientation_xyzw": [
                        round(float(value), 8) for value in box_quaternion
                    ],
                }
            )
        if len(collision_boxes) < 2:
            raise GraspReconstructionError(
                "depth object cannot define segmented collision geometry"
            )

        left_depth = int(stats[component_label, cv2.CC_STAT_LEFT])
        top_depth = int(stats[component_label, cv2.CC_STAT_TOP])
        width_depth = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height_depth = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        depth_to_rgb_x = rgb.shape[1] / float(depth.shape[1])
        depth_to_rgb_y = rgb.shape[0] / float(depth.shape[0])
        left_rgb = left_depth * depth_to_rgb_x
        top_rgb = top_depth * depth_to_rgb_y
        width_rgb = width_depth * depth_to_rgb_x
        height_rgb = height_depth * depth_to_rgb_y
        crop_size = max(
            self._object_crop_size_px,
            int(math.ceil(max(width_rgb, height_rgb)))
            + 2 * self._object_crop_margin_px,
        )
        crop_size = min(crop_size, rgb.shape[0], rgb.shape[1])
        center_u = left_rgb + width_rgb / 2.0
        center_v = top_rgb + height_rgb / 2.0
        crop_x = int(round(center_u - crop_size / 2.0))
        crop_y = int(round(center_v - crop_size / 2.0))
        crop_x = min(max(0, crop_x), rgb.shape[1] - crop_size)
        crop_y = min(max(0, crop_y), rgb.shape[0] - crop_size)
        cropped = rgb[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        details = {
            "component_pixels": area,
            "component_bbox_depth": [
                left_depth,
                top_depth,
                width_depth,
                height_depth,
            ],
            "rgb_crop": [crop_x, crop_y, crop_size, crop_size],
            "collision_geometry": {
                "type": "multi_box",
                "frame_id": self._target_frame,
                "center_xyz_m": [round(float(value), 6) for value in collision_center],
                "size_xyz_m": [round(float(value), 6) for value in collision_size],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "padding_m": self._collision_padding_m,
                "source": "registered_depth_segmented_pca_boxes",
                "single_view_incomplete": True,
                "boxes": collision_boxes,
            },
        }
        return cropped, crop_x, crop_y, component_depth, details

    def _depth_object_opening(
        self,
        proposal: GraspProposal2D,
        depth: np.ndarray,
        object_mask: np.ndarray,
        *,
        fx: float,
        fy: float,
    ) -> tuple[GraspProposal2D, dict]:
        """Measure the physical opening from the registered depth silhouette."""
        center_u = int(round(proposal.u))
        center_v = int(round(proposal.v))
        if not (
            0 <= center_u < object_mask.shape[1]
            and 0 <= center_v < object_mask.shape[0]
        ):
            raise GraspReconstructionError("candidate is outside the object mask")
        membership_kernel = np.ones(
            (self._object_mask_kernel_px, self._object_mask_kernel_px),
            dtype=np.uint8,
        )
        membership = cv2.dilate(
            object_mask.astype(np.uint8), membership_kernel
        ).astype(bool)
        if not membership[center_v, center_u]:
            raise GraspReconstructionError(
                "candidate center is not on the selected depth object"
            )

        du = math.cos(proposal.angle_rad)
        dv = -math.sin(proposal.angle_rad)

        def scan(sign: float) -> tuple[int, bool]:
            last_inside = 0
            maximum = int(math.hypot(*object_mask.shape))
            for distance in range(1, maximum + 1):
                u = int(round(proposal.u + sign * distance * du))
                v = int(round(proposal.v + sign * distance * dv))
                if not (0 <= u < object_mask.shape[1] and 0 <= v < object_mask.shape[0]):
                    return last_inside, True
                if not object_mask[v, u]:
                    return last_inside, False
                last_inside = distance
            return last_inside, True

        negative, negative_edge = scan(-1.0)
        positive, positive_edge = scan(1.0)
        if negative_edge or positive_edge:
            raise GraspReconstructionError(
                "selected object touches the image boundary along the closing axis"
            )
        if negative < 2 or positive < 2:
            raise GraspReconstructionError(
                "candidate does not have object support on both jaw sides"
            )
        silhouette_opening_px = float(negative + positive + 1)
        local_depth = float(depth[center_v, center_u])
        if not math.isfinite(local_depth) or local_depth <= 0.0:
            raise GraspReconstructionError("candidate centre depth is invalid")
        metres_per_pixel = math.hypot(
            du * local_depth / fx,
            dv * local_depth / fy,
        )
        if metres_per_pixel <= 0.0:
            raise GraspReconstructionError("candidate pixel scale is invalid")
        clearance_px = self._opening_clearance_m / metres_per_pixel
        physical_lower_bound_px = silhouette_opening_px + clearance_px
        # The network opening is trained in normalized/cropped image space and
        # is retained as diagnostic evidence.  Metric feasibility comes from
        # registered depth at the network-selected centre and direction.
        depth_opening_px = physical_lower_bound_px
        return (
            GraspProposal2D(
                u=proposal.u,
                v=proposal.v,
                angle_rad=proposal.angle_rad,
                opening_px=depth_opening_px,
                score=proposal.score,
            ),
            {
                "model_opening_px": round(abs(proposal.opening_px), 3),
                "silhouette_opening_px": round(silhouette_opening_px, 3),
                "clearance_m": self._opening_clearance_m,
                "depth_opening_px": round(depth_opening_px, 3),
                "model_to_silhouette_ratio": round(
                    abs(proposal.opening_px) / silhouette_opening_px, 3
                ),
            },
        )

    def _publish_debug(
        self,
        source: Image,
        rgb: np.ndarray,
        proposal: GraspProposal2D | None,
        *,
        accepted: bool,
        label: str,
    ) -> None:
        debug = rgb.copy()
        color = (70, 255, 70) if accepted else (255, 210, 40)
        if proposal is not None:
            center = (int(round(proposal.u)), int(round(proposal.v)))
            half = max(4.0, proposal.opening_px / 2.0)
            du = math.cos(proposal.angle_rad)
            dv = -math.sin(proposal.angle_rad)
            endpoint_a = (
                int(round(proposal.u - half * du)),
                int(round(proposal.v - half * dv)),
            )
            endpoint_b = (
                int(round(proposal.u + half * du)),
                int(round(proposal.v + half * dv)),
            )
            cv2.line(debug, endpoint_a, endpoint_b, color, 3, cv2.LINE_AA)
            cv2.drawMarker(
                debug,
                center,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
        cv2.rectangle(debug, (0, 0), (debug.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(
            debug,
            label[:100],
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        message = Image()
        message.header = source.header
        message.height, message.width = debug.shape[:2]
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = debug.tobytes()
        self._debug_publisher.publish(message)

    def _on_images(
        self, rgb_message: Image, depth_message: Image, info: CameraInfo
    ) -> None:
        now = time.monotonic()
        if now - self._last_process_time < 1.0 / self._max_rate_hz:
            return
        self._last_process_time = now
        rgb = self._rgb_array(rgb_message)
        depth = self._depth_array(depth_message)
        if rgb is None or depth is None:
            self._status(
                False,
                "unsupported_encoding",
                rgb_encoding=rgb_message.encoding,
                depth_encoding=depth_message.encoding,
            )
            return
        if info.k[0] <= 0.0 or info.k[4] <= 0.0 or info.width <= 0 or info.height <= 0:
            self._status(False, "invalid_camera_intrinsics")
            self._publish_debug(
                rgb_message, rgb, None, accepted=False, label="GA INVALID INTRINSICS"
            )
            return
        scale_x = depth.shape[1] / float(info.width)
        scale_y = depth.shape[0] / float(info.height)
        fx = float(info.k[0]) * scale_x
        fy = float(info.k[4]) * scale_y
        cx = float(info.k[2]) * scale_x
        cy = float(info.k[5]) * scale_y
        source_frame = info.header.frame_id or depth_message.header.frame_id
        if not source_frame:
            self._status(False, "missing_camera_frame")
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                Time.from_msg(depth_message.header.stamp),
                timeout=Duration(seconds=0.20),
            )
        except TransformException as error:
            self._status(False, "tf_unavailable", detail=str(error)[:240])
            self._publish_debug(
                rgb_message, rgb, None, accepted=False, label="GA TF UNAVAILABLE"
            )
            return

        inference_rgb = rgb
        crop_x = 0
        crop_y = 0
        object_mask: np.ndarray | None = None
        region_details: dict = {"object_crop_enabled": False}
        if self._object_crop_enabled:
            try:
                (
                    inference_rgb,
                    crop_x,
                    crop_y,
                    object_mask,
                    region_details,
                ) = self._object_region(
                    rgb,
                    depth,
                    transform,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
                region_details["object_crop_enabled"] = True
            except GraspReconstructionError as error:
                self._status(False, "no_depth_object_region", detail=str(error))
                self._publish_debug(
                    rgb_message,
                    rgb,
                    None,
                    accepted=False,
                    label="GA NO DEPTH OBJECT",
                )
                return
        try:
            result = self._request_proposals(inference_rgb)
        except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as error:
            self._status(
                False,
                "inference_service_unavailable",
                detail=f"{error.__class__.__name__}:{error}",
            )
            self._publish_debug(
                rgb_message, rgb, None, accepted=False, label="GA SERVICE UNAVAILABLE"
            )
            return
        if (
            int(result.get("input_width", -1)) != inference_rgb.shape[1]
            or int(result.get("input_height", -1)) != inference_rgb.shape[0]
        ):
            self._status(False, "inference_image_size_mismatch")
            return

        accepted_candidates: list[tuple[
            int, GraspProposal2D, GraspProposal2D, ReconstructedGrasp, dict
        ]] = []
        rejected: list[dict] = []
        first_proposal_rgb: GraspProposal2D | None = None
        for index, candidate in enumerate(result["candidates"]):
            proposal_rgb: GraspProposal2D | None = None
            opening_details: dict = {}
            try:
                proposal_rgb = GraspProposal2D(
                    u=float(candidate["u"]) + crop_x,
                    v=float(candidate["v"]) + crop_y,
                    angle_rad=float(candidate["angle_rad"]),
                    opening_px=float(candidate["opening_px"]),
                    score=float(candidate["score"]),
                )
                if first_proposal_rgb is None:
                    first_proposal_rgb = proposal_rgb
                if proposal_rgb.u >= (
                    rgb.shape[1] * self._self_mask_right_fraction
                ):
                    raise GraspReconstructionError(
                        "candidate lies in the calibrated camera self-mask"
                    )
                proposal_depth = self._proposal_in_depth(
                    proposal_rgb,
                    rgb_width=rgb.shape[1],
                    rgb_height=rgb.shape[0],
                    depth_width=depth.shape[1],
                    depth_height=depth.shape[0],
                )
                if proposal_depth.score < self._minimum_score:
                    raise GraspReconstructionError(
                        f"score {proposal_depth.score:.3f} is below "
                        f"{self._minimum_score:.3f}"
                    )
                opening_details = {
                    "model_opening_px": round(abs(proposal_depth.opening_px), 3)
                }
                if object_mask is not None:
                    proposal_depth, opening_details = (
                        self._depth_object_opening(
                            proposal_depth,
                            depth,
                            object_mask,
                            fx=fx,
                            fy=fy,
                        )
                    )
                    proposal_rgb = self._depth_opening_in_rgb(
                        proposal_rgb,
                        proposal_depth,
                        rgb_width=rgb.shape[1],
                        rgb_height=rgb.shape[0],
                        depth_width=depth.shape[1],
                        depth_height=depth.shape[0],
                    )
                    if index == 0:
                        first_proposal_rgb = proposal_rgb
                reconstructed = self._reconstruct(
                    depth, proposal_depth, fx=fx, fy=fy, cx=cx, cy=cy
                )
            except (KeyError, TypeError, ValueError, GraspReconstructionError) as error:
                rejected.append(
                    {
                        "rank": index,
                        "reason": str(error)[:160],
                        "pixel": (
                            [round(proposal_rgb.u, 2), round(proposal_rgb.v, 2)]
                            if proposal_rgb is not None
                            else None
                        ),
                        "opening_evidence": opening_details,
                    }
                )
                continue
            accepted_candidates.append(
                (
                    index,
                    proposal_rgb,
                    proposal_depth,
                    reconstructed,
                    opening_details,
                )
            )
            if len(accepted_candidates) >= self._maximum_published_candidates:
                break

        if not accepted_candidates:
            reason = "no_model_candidate" if not result["candidates"] else "all_candidates_rejected"
            self._status(
                False,
                reason,
                model_candidates=len(result["candidates"]),
                rejected=rejected,
                inference_seconds=round(float(result["inference_seconds"]), 4),
                object_region=region_details,
            )
            self._publish_debug(
                rgb_message,
                rgb,
                first_proposal_rgb,
                accepted=False,
                label=f"GA {reason.upper()}",
            )
            return

        depth_validated_candidates = []
        transformed_candidates = []
        for (
            candidate_rank,
            candidate_rgb,
            _candidate_depth,
            candidate_grasp,
            candidate_opening_details,
        ) in accepted_candidates:
            camera_pose = PoseStamped()
            camera_pose.header.stamp = depth_message.header.stamp
            camera_pose.header.frame_id = source_frame
            camera_pose.pose.position.x = float(
                candidate_grasp.position_camera_m[0]
            )
            camera_pose.pose.position.y = float(
                candidate_grasp.position_camera_m[1]
            )
            camera_pose.pose.position.z = float(
                candidate_grasp.position_camera_m[2]
            )
            camera_pose.pose.orientation.x = float(
                candidate_grasp.quaternion_camera_xyzw[0]
            )
            camera_pose.pose.orientation.y = float(
                candidate_grasp.quaternion_camera_xyzw[1]
            )
            camera_pose.pose.orientation.z = float(
                candidate_grasp.quaternion_camera_xyzw[2]
            )
            camera_pose.pose.orientation.w = float(
                candidate_grasp.quaternion_camera_xyzw[3]
            )
            transformed = do_transform_pose_stamped(camera_pose, transform)
            transformed.header.frame_id = self._target_frame
            transformed_candidates.append(transformed)
            depth_validated_candidates.append(
                {
                    "rank": candidate_rank,
                    "score": round(candidate_grasp.score, 6),
                    "opening_m": round(candidate_grasp.opening_m, 6),
                    "pixel": [
                        round(candidate_rgb.u, 3),
                        round(candidate_rgb.v, 3),
                    ],
                    "angle_rad": round(candidate_rgb.angle_rad, 6),
                    "depth_support_ratio": round(
                        candidate_grasp.depth_support_ratio, 6
                    ),
                    "surface_residual_ratio": round(
                        candidate_grasp.surface_residual_ratio, 6
                    ),
                    "valid_depth_points": candidate_grasp.valid_depth_points,
                    "opening_evidence": candidate_opening_details,
                    "pose": {
                        "position_xyz_m": [
                            float(transformed.pose.position.x),
                            float(transformed.pose.position.y),
                            float(transformed.pose.position.z),
                        ],
                        "orientation_xyzw": [
                            float(transformed.pose.orientation.x),
                            float(transformed.pose.orientation.y),
                            float(transformed.pose.orientation.z),
                            float(transformed.pose.orientation.w),
                        ],
                    },
                }
            )

        rank, proposal_rgb, _proposal_depth, grasp, opening_details = (
            accepted_candidates[0]
        )
        target_pose = transformed_candidates[0]
        self._pose_publisher.publish(target_pose)
        metadata = {
            "accepted": True,
            "observation_only": True,
            "rank": rank,
            "score": round(grasp.score, 6),
            "opening_m": round(grasp.opening_m, 6),
            "pixel": [round(proposal_rgb.u, 3), round(proposal_rgb.v, 3)],
            "angle_rad": round(proposal_rgb.angle_rad, 6),
            "depth_support_ratio": round(grasp.depth_support_ratio, 6),
            "surface_residual_ratio": round(grasp.surface_residual_ratio, 6),
            "valid_depth_points": grasp.valid_depth_points,
            "frame_id": self._target_frame,
            "model_sha256": result.get("model_sha256", ""),
            "inference_seconds": round(float(result["inference_seconds"]), 4),
            "rejected_higher_ranked": rejected,
            "rejected_candidates": rejected,
            "depth_validated_candidates": depth_validated_candidates,
            "pose_semantics": "visible_surface_candidate_not_gripper_tcp",
            "opening_evidence": opening_details,
            "object_region": region_details,
            "source_stamp": {
                "sec": int(depth_message.header.stamp.sec),
                "nanosec": int(depth_message.header.stamp.nanosec),
            },
        }
        self._publish_json(self._metadata_publisher, metadata)
        self._status(
            True,
            "depth_validated_candidate_published",
            score=metadata["score"],
            opening_m=metadata["opening_m"],
            frame_id=self._target_frame,
            motion_capable=False,
        )
        self._publish_debug(
            rgb_message,
            rgb,
            proposal_rgb,
            accepted=True,
            label=f"GA OK score={grasp.score:.2f} open={grasp.opening_m * 1000:.1f}mm",
        )


def main() -> int:
    rclpy.init()
    node = GraspAnythingPerception()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
