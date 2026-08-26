#!/usr/bin/env python3
"""Basic registered RGB-D cup observation for Gazebo and aligned real D435i.

This node only observes and publishes poses. It never commands a controller,
executes a trajectory, or attaches a MoveIt object.
"""

from __future__ import annotations

import json
import time

import cv2
import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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


class CupPerception(Node):
    def __init__(self) -> None:
        super().__init__("pgi_cup_perception")
        defaults = {
            "rgb_topic": "/pgi_d435i/image",
            "depth_topic": "/pgi_d435i/depth_image",
            "camera_info_topic": "/pgi_d435i/camera_info",
            "observation_topic": "/pgi/perception/cup_observation",
            "grasp_pose_topic": "/pgi/perception/cup_grasp_pose",
            "status_topic": "/pgi/perception/cup_status",
            "debug_image_topic": "/pgi/perception/cup_debug_image",
            "target_frame": "base_link",
            "max_rate_hz": 10.0,
            "sync_queue_size": 20,
            "sync_slop_sec": 0.08,
            "depth_scale_16uc1": 0.001,
            "min_depth_m": 0.05,
            "max_depth_m": 3.0,
            "smoothing_alpha": 0.45,
            "hue_min": 90,
            "hue_max": 135,
            "saturation_min": 80,
            "value_min": 45,
            "min_mask_pixels": 120,
            "morphology_kernel_px": 5,
            "assume_upright_on_ground": True,
            "ground_z_m": 0.0,
            "grasp_center_z_from_base_m": 0.040,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._target_frame = str(self.get_parameter("target_frame").value)
        self._max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self._depth_scale = float(self.get_parameter("depth_scale_16uc1").value)
        self._min_depth = float(self.get_parameter("min_depth_m").value)
        self._max_depth = float(self.get_parameter("max_depth_m").value)
        self._alpha = float(self.get_parameter("smoothing_alpha").value)
        self._hue_min = int(self.get_parameter("hue_min").value)
        self._hue_max = int(self.get_parameter("hue_max").value)
        self._saturation_min = int(self.get_parameter("saturation_min").value)
        self._value_min = int(self.get_parameter("value_min").value)
        self._min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        self._kernel_px = int(self.get_parameter("morphology_kernel_px").value)
        self._assume_grounded = bool(
            self.get_parameter("assume_upright_on_ground").value
        )
        self._ground_z = float(self.get_parameter("ground_z_m").value)
        self._grasp_z = float(
            self.get_parameter("grasp_center_z_from_base_m").value
        )
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self._kernel_px < 1 or self._kernel_px % 2 == 0:
            raise ValueError("morphology_kernel_px must be a positive odd integer")
        if not self._min_depth < self._max_depth:
            raise ValueError("depth range is invalid")

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
        latest_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # Image processing may wait briefly for a timestamped transform. Keep
        # sensor callbacks in a group separate from the TF listener so a
        # second executor thread can continue filling the TF buffer.
        image_callback_group = MutuallyExclusiveCallbackGroup()
        debug_callback_group = MutuallyExclusiveCallbackGroup()
        self._observation_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("observation_topic").value),
            reliable_qos,
        )
        self._grasp_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("grasp_pose_topic").value),
            reliable_qos,
        )
        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), reliable_qos
        )
        self._debug_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            latest_image_qos,
        )
        # Keep the operator image independent from RGB-D synchronization.
        # Depth and CameraInfo can arrive at different rates under simulator
        # load, but every new RGB frame must still update RQT.
        self._debug_rgb_sub = self.create_subscription(
            Image,
            str(self.get_parameter("rgb_topic").value),
            self._on_rgb_debug,
            latest_image_qos,
            callback_group=debug_callback_group,
        )
        self._rgb_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("rgb_topic").value),
            qos_profile=sensor_qos,
            callback_group=image_callback_group,
        )
        self._depth_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("depth_topic").value),
            qos_profile=sensor_qos,
            callback_group=image_callback_group,
        )
        self._info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile=sensor_qos,
            callback_group=image_callback_group,
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
        self._smoothed_position: np.ndarray | None = None
        self.get_logger().info(
            "Basic PGI cup perception ready; output is observation-only and plan-only"
        )

    def _status(self, detected: bool, reason: str, **details) -> None:
        payload = {"detected": detected, "reason": reason, **details}
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self._status_publisher.publish(message)

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

    def _largest_blue_component(self, rgb: np.ndarray) -> tuple[np.ndarray, int] | None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self._hue_min, self._saturation_min, self._value_min]),
            np.array([self._hue_max, 255, 255]),
        )
        kernel = np.ones((self._kernel_px, self._kernel_px), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < self._min_mask_pixels:
            return None
        return (labels == component).astype(np.uint8), area

    def _publish_debug(
        self,
        source: Image,
        rgb: np.ndarray,
        *,
        status: str,
        mask: np.ndarray | None = None,
        u: float | None = None,
        v: float | None = None,
        detected: bool = False,
    ) -> None:
        """Publish the newest camera frame even when detection is unavailable.

        RQT intentionally displays this topic rather than the raw image so the
        operator can see detector state.  This method is called directly from
        every RGB frame and does not wait for depth, CameraInfo, or TF.
        """
        debug = rgb.copy()
        if mask is not None:
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(debug, contours, -1, (255, 255, 0), 2)
        if u is not None and v is not None:
            cv2.drawMarker(
                debug,
                (int(round(u)), int(round(v))),
                (255, 60, 60),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
        stamp = source.header.stamp
        label = f"LIVE {stamp.sec}.{stamp.nanosec:09d} | {status}"
        text_color = (80, 255, 80) if detected else (255, 220, 40)
        cv2.rectangle(
            debug,
            (0, 0),
            (debug.shape[1], 30),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            debug,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
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

    def _on_rgb_debug(self, rgb_message: Image) -> None:
        """Publish a live 2-D detector view for every usable RGB frame."""
        rgb = self._rgb_array(rgb_message)
        if rgb is None:
            return
        component = self._largest_blue_component(rgb)
        if component is None:
            self._publish_debug(rgb_message, rgb, status="NO CUP")
            return
        rgb_mask, _area = component
        rows, columns = np.nonzero(rgb_mask)
        if columns.size == 0:
            self._publish_debug(rgb_message, rgb, status="NO CUP")
            return
        self._publish_debug(
            rgb_message,
            rgb,
            status="CUP VISIBLE",
            mask=rgb_mask,
            u=float(np.median(columns)),
            v=float(np.median(rows)),
            detected=True,
        )

    def _on_images(self, rgb_message: Image, depth_message: Image, info: CameraInfo) -> None:
        now = time.monotonic()
        if self._max_rate_hz > 0.0 and now - self._last_process_time < 1.0 / self._max_rate_hz:
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
        if info.k[0] <= 0.0 or info.k[4] <= 0.0:
            self._status(False, "invalid_camera_intrinsics")
            return
        component = self._largest_blue_component(rgb)
        if component is None:
            self._smoothed_position = None
            self._status(False, "no_blue_cup")
            return
        rgb_mask, area = component
        depth_mask = cv2.resize(
            rgb_mask,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = depth_mask & np.isfinite(depth)
        valid &= (depth >= self._min_depth) & (depth <= self._max_depth)
        rows, columns = np.nonzero(valid)
        if columns.size < self._min_mask_pixels // 2:
            self._status(False, "insufficient_cup_depth", valid_pixels=int(columns.size))
            return
        u_depth = float(np.median(columns))
        v_depth = float(np.median(rows))
        z = float(np.median(depth[valid]))
        scale_x = depth.shape[1] / float(info.width)
        scale_y = depth.shape[0] / float(info.height)
        fx, fy = info.k[0] * scale_x, info.k[4] * scale_y
        cx, cy = info.k[2] * scale_x, info.k[5] * scale_y
        camera_point = np.array(
            [(u_depth - cx) * z / fx, (v_depth - cy) * z / fy, z], dtype=float
        )
        source_frame = info.header.frame_id or depth_message.header.frame_id
        if not source_frame:
            self._status(False, "missing_camera_frame")
            return
        camera_pose = PoseStamped()
        camera_pose.header.stamp = depth_message.header.stamp
        camera_pose.header.frame_id = source_frame
        camera_pose.pose.position.x = float(camera_point[0])
        camera_pose.pose.position.y = float(camera_point[1])
        camera_pose.pose.position.z = float(camera_point[2])
        camera_pose.pose.orientation.w = 1.0
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                Time.from_msg(depth_message.header.stamp),
                timeout=Duration(seconds=0.20),
            )
            observation = do_transform_pose_stamped(camera_pose, transform)
        except TransformException as error:
            self._status(False, "tf_unavailable", detail=str(error))
            return
        measured = np.array(
            [
                observation.pose.position.x,
                observation.pose.position.y,
                observation.pose.position.z,
            ],
            dtype=float,
        )
        if self._smoothed_position is None:
            self._smoothed_position = measured
        else:
            self._smoothed_position = (
                self._alpha * measured + (1.0 - self._alpha) * self._smoothed_position
            )
        observation.header.frame_id = self._target_frame
        observation.pose.position.x = float(self._smoothed_position[0])
        observation.pose.position.y = float(self._smoothed_position[1])
        observation.pose.position.z = float(self._smoothed_position[2])
        observation.pose.orientation.x = 0.0
        observation.pose.orientation.y = 0.0
        observation.pose.orientation.z = 0.0
        observation.pose.orientation.w = 1.0
        self._observation_publisher.publish(observation)

        grasp = PoseStamped()
        grasp.header = observation.header
        grasp.pose.position.x = observation.pose.position.x
        grasp.pose.position.y = observation.pose.position.y
        grasp.pose.position.z = (
            self._ground_z + self._grasp_z
            if self._assume_grounded
            else observation.pose.position.z
        )
        grasp.pose.orientation.w = 1.0
        self._grasp_publisher.publish(grasp)
        self._status(
            True,
            "basic_cup_pose_published",
            frame_id=self._target_frame,
            mask_pixels=area,
            depth_m=round(z, 4),
            grasp_z_m=round(grasp.pose.position.z, 4),
            plan_only=True,
        )


def main() -> int:
    rclpy.init()
    node = CupPerception()
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
