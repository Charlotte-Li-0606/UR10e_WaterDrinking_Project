#!/usr/bin/env python3
"""Estimate the mouth position from synchronized wrist RGB-D data.

The node uses MediaPipe Face Landmarker for 2D mouth landmarks and a robust
median depth patch to back-project the mouth centre with CameraInfo intrinsics.
Only valid estimates are published; consumers never receive a placeholder pose.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Final

import mediapipe as mp
import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


DEFAULT_MOUTH_LANDMARKS: Final = (13, 14, 78, 308)


class MouthPoseEstimator(Node):
    def __init__(self) -> None:
        super().__init__("mouth_pose_estimator")

        project_root = Path(__file__).resolve().parents[1]
        default_model = project_root / "assets" / "mediapipe" / "face_landmarker.task"
        self.declare_parameter("rgb_topic", "/wrist_rgbd/image")
        self.declare_parameter("depth_topic", "/wrist_rgbd/depth_image")
        self.declare_parameter("camera_info_topic", "/wrist_rgbd/camera_info")
        self.declare_parameter("pose_topic", "/feeding/perception/mouth_pose")
        self.declare_parameter("status_topic", "/feeding/perception/mouth_pose_status")
        self.declare_parameter("model_path", str(default_model))
        self.declare_parameter("mouth_landmark_indices", list(DEFAULT_MOUTH_LANDMARKS))
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter("depth_patch_radius_px", 4)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 4.0)
        self.declare_parameter("depth_scale_16uc1", 0.001)
        self.declare_parameter("smoothing_alpha", 0.45)

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self._depth_patch_radius = int(self.get_parameter("depth_patch_radius_px").value)
        self._min_depth_m = float(self.get_parameter("min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._depth_scale_16uc1 = float(self.get_parameter("depth_scale_16uc1").value)
        self._smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self._mouth_indices = tuple(int(i) for i in self.get_parameter("mouth_landmark_indices").value)
        if not self._mouth_indices:
            raise ValueError("mouth_landmark_indices cannot be empty")

        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}. Run scripts/setup_mouth_perception.sh."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.60,
            min_face_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pose_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("pose_topic").value), reliable_qos
        )
        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), reliable_qos
        )
        self._rgb_sub = message_filters.Subscriber(self, Image, self._rgb_topic, qos_profile=reliable_qos)
        self._depth_sub = message_filters.Subscriber(self, Image, self._depth_topic, qos_profile=reliable_qos)
        self._info_sub = message_filters.Subscriber(
            self, CameraInfo, self._camera_info_topic, qos_profile=reliable_qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._sync.registerCallback(self._on_synced_images)

        self._last_process_time = 0.0
        self._last_landmarker_timestamp_ms = -1
        self._smoothed_position: np.ndarray | None = None
        self.get_logger().info(
            "Mouth perception ready: RGB=%s depth=%s camera_info=%s -> %s"
            % (self._rgb_topic, self._depth_topic, self._camera_info_topic, self._pose_publisher.topic_name)
        )

    def destroy_node(self) -> bool:
        self._landmarker.close()
        return super().destroy_node()

    def _publish_status(self, detected: bool, reason: str, **details: float | int | str) -> None:
        payload: dict[str, bool | str | float | int] = {"detected": detected, "reason": reason}
        payload.update(details)
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self._status_publisher.publish(message)

    @staticmethod
    def _rgb_array(message: Image) -> np.ndarray | None:
        encoding = message.encoding.lower()
        if encoding not in {"rgb8", "bgr8"}:
            return None
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
        if encoding == "bgr8":
            return image[:, :, ::-1].copy()
        return image.copy()

    @staticmethod
    def _depth_array(message: Image, depth_scale_16uc1: float) -> np.ndarray | None:
        encoding = message.encoding.upper()
        if encoding == "32FC1":
            columns = message.step // np.dtype(np.float32).itemsize
            rows = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, columns)
            return rows[:, : message.width]
        if encoding == "16UC1":
            columns = message.step // np.dtype(np.uint16).itemsize
            rows = np.frombuffer(message.data, dtype=np.uint16).reshape(message.height, columns)
            return rows[:, : message.width].astype(np.float32) * depth_scale_16uc1
        return None

    def _median_depth(self, depth: np.ndarray, u: float, v: float) -> float | None:
        cx = int(round(u))
        cy = int(round(v))
        radius = self._depth_patch_radius
        x0, x1 = max(0, cx - radius), min(depth.shape[1], cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(depth.shape[0], cy + radius + 1)
        samples = depth[y0:y1, x0:x1]
        valid = samples[np.isfinite(samples)]
        valid = valid[(valid >= self._min_depth_m) & (valid <= self._max_depth_m)]
        return float(np.median(valid)) if valid.size else None

    def _on_synced_images(self, rgb: Image, depth_message: Image, camera_info: CameraInfo) -> None:
        now = time.monotonic()
        if self._max_rate_hz > 0.0 and now - self._last_process_time < 1.0 / self._max_rate_hz:
            return
        self._last_process_time = now

        rgb_array = self._rgb_array(rgb)
        if rgb_array is None:
            self._publish_status(False, "unsupported_rgb_encoding", encoding=rgb.encoding)
            return
        depth = self._depth_array(depth_message, self._depth_scale_16uc1)
        if depth is None:
            self._publish_status(False, "unsupported_depth_encoding", encoding=depth_message.encoding)
            return
        if camera_info.k[0] == 0.0 or camera_info.k[4] == 0.0:
            self._publish_status(False, "invalid_camera_intrinsics")
            return

        timestamp_ms = max(int(time.monotonic_ns() // 1_000_000), self._last_landmarker_timestamp_ms + 1)
        self._last_landmarker_timestamp_ms = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            self._smoothed_position = None
            self._publish_status(False, "no_face")
            return

        face = result.face_landmarks[0]
        if max(self._mouth_indices) >= len(face):
            self._publish_status(False, "invalid_mouth_landmark_indices", landmark_count=len(face))
            return
        u = float(np.mean([face[index].x for index in self._mouth_indices]) * rgb.width)
        v = float(np.mean([face[index].y for index in self._mouth_indices]) * rgb.height)
        depth_u = u * depth.shape[1] / rgb.width
        depth_v = v * depth.shape[0] / rgb.height
        z = self._median_depth(depth, depth_u, depth_v)
        if z is None:
            self._publish_status(False, "invalid_depth_at_mouth", pixel_u=round(u, 2), pixel_v=round(v, 2))
            return

        # RGB and depth are co-registered in this Gazebo RGB-D sensor. Scale
        # intrinsics only if the image resolutions differ.
        scale_x = depth.shape[1] / camera_info.width
        scale_y = depth.shape[0] / camera_info.height
        fx, fy = camera_info.k[0] * scale_x, camera_info.k[4] * scale_y
        cx, cy = camera_info.k[2] * scale_x, camera_info.k[5] * scale_y
        point = np.array([(depth_u - cx) * z / fx, (depth_v - cy) * z / fy, z], dtype=float)
        if self._smoothed_position is None:
            self._smoothed_position = point
        else:
            self._smoothed_position = self._smoothing_alpha * point + (1.0 - self._smoothing_alpha) * self._smoothed_position

        pose = PoseStamped()
        pose.header.stamp = rgb.header.stamp
        pose.header.frame_id = camera_info.header.frame_id or rgb.header.frame_id
        pose.pose.position.x = float(self._smoothed_position[0])
        pose.pose.position.y = float(self._smoothed_position[1])
        pose.pose.position.z = float(self._smoothed_position[2])
        # The RGB-D measurement estimates a point, not head orientation.
        pose.pose.orientation.w = 1.0
        self._pose_publisher.publish(pose)
        self._publish_status(
            True,
            "mouth_pose_published",
            pixel_u=round(u, 2),
            pixel_v=round(v, 2),
            depth_m=round(z, 4),
            frame_id=pose.header.frame_id,
        )


def main() -> int:
    rclpy.init()
    node = MouthPoseEstimator()
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
