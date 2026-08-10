#!/usr/bin/env python3
"""Publish a lightweight PointCloud2 stream from the fixed wrist RGB-D depth image.

This node intentionally performs only depth back-projection and stride/range
filtering.  It does not create a persistent map or attempt robot self-filtering;
MoveIt's optional occupancy monitor consumes the resulting cloud.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


class DepthToPointCloudNode(Node):
    """Back-project valid depth pixels into an XYZ ``PointCloud2`` message."""

    def __init__(
        self,
        *,
        depth_topic: str,
        camera_info_topic: str,
        points_topic: str,
        frame_id: str,
        stride: int,
        min_depth_m: float,
        max_depth_m: float,
        max_publish_rate_hz: float,
        exclude_tool_box: bool = False,
        tool_frame: str = "tool0",
        tool_box_center_z_m: float = 0.15,
        tool_box_size_xy_m: float = 0.10,
        tool_box_size_z_m: float = 0.30,
        tool_box_padding_m: float = 0.02,
    ) -> None:
        super().__init__("wrist_depth_to_pointcloud")
        if min_depth_m >= max_depth_m:
            raise ValueError("min_depth must be less than max_depth")
        self._frame_id_override = frame_id.strip().lstrip("/")
        self._stride = int(stride)
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._min_publish_interval = 1.0 / float(max_publish_rate_hz)
        self._last_publish_time = float("-inf")
        self._last_report_time = float("-inf")
        self._camera_info: CameraInfo | None = None
        self._exclude_tool_box = bool(exclude_tool_box)
        self._tool_frame = tool_frame.strip().lstrip("/")
        self._tool_box_center_z_m = float(tool_box_center_z_m)
        self._tool_box_half_xy_m = 0.5 * float(tool_box_size_xy_m) + float(tool_box_padding_m)
        self._tool_box_half_z_m = 0.5 * float(tool_box_size_z_m) + float(tool_box_padding_m)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self._publisher = self.create_publisher(PointCloud2, points_topic, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._depth_callback, qos_profile_sensor_data)
        self.get_logger().info(
            f"Depth-to-cloud: {depth_topic} + {camera_info_topic} -> {points_topic}; "
            f"stride={self._stride}, range=[{self._min_depth_m:.2f}, {self._max_depth_m:.2f}] m, "
            f"tool-box-filter={self._exclude_tool_box}"
        )

    @staticmethod
    def _rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            raise ValueError("invalid TF quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    def _exclude_attached_tool_points(self, points: np.ndarray, source_frame: str) -> np.ndarray:
        """Remove only points inside the calibrated attached tool box."""
        if not self._exclude_tool_box:
            return points
        try:
            transform = self._tf_buffer.lookup_transform(
                self._tool_frame, source_frame, Time(), timeout=Duration(seconds=0.05)
            ).transform
            rotation = self._rotation_matrix(
                transform.rotation.x, transform.rotation.y,
                transform.rotation.z, transform.rotation.w,
            )
            translation = np.array(
                [transform.translation.x, transform.translation.y, transform.translation.z],
                dtype=np.float32,
            )
            tool_points = points @ rotation.T + translation
        except Exception as exc:
            self.get_logger().warning(
                f"Tool self-filter TF unavailable ({source_frame} -> {self._tool_frame}); "
                f"keeping cloud conservative: {exc}"
            )
            return points
        center = np.array([0.0, 0.0, self._tool_box_center_z_m], dtype=np.float32)
        delta = np.abs(tool_points - center)
        inside = (
            (delta[:, 0] <= self._tool_box_half_xy_m)
            & (delta[:, 1] <= self._tool_box_half_xy_m)
            & (delta[:, 2] <= self._tool_box_half_z_m)
        )
        return points[~inside]

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if len(message.k) < 6 or message.k[0] <= 0.0 or message.k[4] <= 0.0:
            self.get_logger().warning("Ignoring camera_info with invalid focal lengths")
            return
        self._camera_info = message

    @staticmethod
    def _depth_array(message: Image) -> np.ndarray | None:
        encoding = message.encoding.upper()
        if encoding == "32FC1":
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            bytes_per_pixel = 4
            scale = 1.0
        elif encoding == "16UC1":
            dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
            bytes_per_pixel = 2
            scale = 0.001
        else:
            return None
        if message.height <= 0 or message.width <= 0 or message.step < message.width * bytes_per_pixel:
            return None
        row_width = message.step // bytes_per_pixel
        expected_bytes = message.height * message.step
        if len(message.data) < expected_bytes:
            return None
        raw = np.frombuffer(message.data, dtype=dtype, count=message.height * row_width)
        return raw.reshape(message.height, row_width)[:, : message.width].astype(np.float32, copy=False) * scale

    def _depth_callback(self, message: Image) -> None:
        now = time.monotonic()
        if now - self._last_publish_time < self._min_publish_interval:
            return
        info = self._camera_info
        if info is None:
            if now - self._last_report_time > 5.0:
                self.get_logger().warning("Waiting for camera_info before publishing a point cloud")
                self._last_report_time = now
            return
        depth = self._depth_array(message)
        if depth is None:
            if now - self._last_report_time > 5.0:
                self.get_logger().warning(
                    f"Ignoring unsupported or malformed depth image encoding={message.encoding!r}"
                )
                self._last_report_time = now
            return

        sampled_depth = depth[:: self._stride, :: self._stride]
        valid = np.isfinite(sampled_depth)
        valid &= sampled_depth >= self._min_depth_m
        valid &= sampled_depth <= self._max_depth_m
        v_indices, u_indices = np.nonzero(valid)
        if not len(v_indices):
            return

        z = sampled_depth[valid]
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        u = u_indices.astype(np.float32) * self._stride
        v = v_indices.astype(np.float32) * self._stride
        points = np.empty((len(z), 3), dtype=np.float32)
        points[:, 0] = (u - cx) * z / fx
        points[:, 1] = (v - cy) * z / fy
        points[:, 2] = z
        cloud_frame = self._frame_id_override or info.header.frame_id or message.header.frame_id
        points = self._exclude_attached_tool_points(points, cloud_frame)
        if not len(points):
            return

        cloud = PointCloud2()
        cloud.header.stamp = message.header.stamp
        cloud.header.frame_id = cloud_frame
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.tobytes()
        self._publisher.publish(cloud)
        self._last_publish_time = now
        if now - self._last_report_time > 5.0:
            self.get_logger().info(
                f"Published {cloud.width} stride-sampled XYZ points in {cloud.header.frame_id}"
            )
            self._last_report_time = now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-topic", default="/wrist_rgbd/depth_image")
    parser.add_argument("--camera-info-topic", default="/wrist_rgbd/camera_info")
    parser.add_argument("--points-topic", default="/wrist_rgbd/points")
    parser.add_argument(
        "--frame-id",
        default="",
        help="Override cloud frame ID; default uses camera_info.header.frame_id.",
    )
    parser.add_argument("--stride", type=_positive_int, default=4)
    parser.add_argument("--min-depth", type=_positive_float, default=0.15, metavar="METERS")
    parser.add_argument("--max-depth", type=_positive_float, default=2.50, metavar="METERS")
    parser.add_argument(
        "--max-publish-rate",
        type=_positive_float,
        default=5.0,
        metavar="HZ",
        help="Bound CPU and OctoMap load; set to the desired cloud rate.",
    )
    parser.add_argument("--exclude-tool-box", action="store_true",
                        help="Filter the calibrated attached tool0 box from the cloud.")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--tool-box-center-z-m", type=float, default=0.15)
    parser.add_argument("--tool-box-size-xy-m", type=_positive_float, default=0.10)
    parser.add_argument("--tool-box-size-z-m", type=_positive_float, default=0.30)
    parser.add_argument("--tool-box-padding-m", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rclpy.init(args=None)
    node = DepthToPointCloudNode(
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        points_topic=args.points_topic,
        frame_id=args.frame_id,
        stride=args.stride,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        max_publish_rate_hz=args.max_publish_rate,
        exclude_tool_box=args.exclude_tool_box,
        tool_frame=args.tool_frame,
        tool_box_center_z_m=args.tool_box_center_z_m,
        tool_box_size_xy_m=args.tool_box_size_xy_m,
        tool_box_size_z_m=args.tool_box_size_z_m,
        tool_box_padding_m=args.tool_box_padding_m,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
