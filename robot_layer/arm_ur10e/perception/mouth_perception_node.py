#!/usr/bin/env python3
"""RGB-D MediaPipe mouth perception for the UR10e wrist camera.

MediaPipe is used only for 2D face / lip landmarks. The published target is
calculated from the RGB mouth pixel, the aligned depth image, CameraInfo
intrinsics, and a TF transform into ``base_link``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


# Inner upper lip, inner lower lip, and the two mouth corners. Using all four
# reduces sensitivity to one noisy lip landmark.
DEFAULT_MOUTH_LANDMARKS = (13, 14, 78, 308)


@dataclass(frozen=True)
class NodeOptions:
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    output_topic: str
    candidates_topic: str
    base_frame: str
    debug_image_topic: str
    status_topic: str
    model_path: Path
    sync_slop_sec: float
    depth_patch_radius_px: int
    min_depth_m: float
    max_depth_m: float
    max_jump_m: float
    smoothing_alpha: float
    max_rate_hz: float
    max_tf_age_sec: float
    max_faces: int
    mouth_landmarks: tuple[int, ...]


def _rotation_matrix(transform: TransformStamped) -> np.ndarray:
    """Return the 3x3 rotation matrix encoded by a TransformStamped."""
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("TF transform has a zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class MouthPerceptionNode(Node):
    def __init__(self, options: NodeOptions) -> None:
        super().__init__("mouth_perception_node")
        self._options = options
        if not options.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe task model is missing: {options.model_path}. "
                "Run scripts/setup_mouth_perception.sh first."
            )
        if not options.mouth_landmarks:
            raise ValueError("At least one mouth landmark index is required")

        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(options.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=options.max_faces,
            min_face_detection_confidence=0.60,
            min_face_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pose_publisher = self.create_publisher(PoseStamped, options.output_topic, qos)
        # A JSON string is used deliberately here instead of a new custom ROS
        # message: it keeps the single-pose contract intact while carrying the
        # image-x metadata required for left/center/right selection.
        self._candidates_publisher = self.create_publisher(String, options.candidates_topic, qos)
        self._status_publisher = self.create_publisher(String, options.status_topic, qos)
        self._debug_publisher = (
            self.create_publisher(Image, options.debug_image_topic, qos)
            if options.debug_image_topic
            else None
        )

        self._rgb_sub = message_filters.Subscriber(self, Image, options.rgb_topic, qos_profile=qos)
        self._depth_sub = message_filters.Subscriber(self, Image, options.depth_topic, qos_profile=qos)
        self._camera_info_sub = message_filters.Subscriber(
            self, CameraInfo, options.camera_info_topic, qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._camera_info_sub],
            queue_size=20,
            slop=options.sync_slop_sec,
        )
        self._sync.registerCallback(self._synchronized_callback)

        self._last_landmarker_timestamp_ms = -1
        self._last_processing_time = 0.0
        self._last_base_position: np.ndarray | None = None
        self._outlier_count = 0
        self._last_warning_times: dict[str, float] = {}
        self.get_logger().info(
            "Mouth perception: %s + %s + %s -> %s in %s"
            % (
                options.rgb_topic,
                options.depth_topic,
                options.camera_info_topic,
                options.output_topic,
                options.base_frame,
            )
        )

    def destroy_node(self) -> bool:
        self._face_landmarker.close()
        return super().destroy_node()

    def _warn_throttled(self, reason: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(reason, float("-inf")) >= 5.0:
            self.get_logger().warning(message)
            self._last_warning_times[reason] = now

    def _status(self, detected: bool, reason: str, **details: float | int | str) -> None:
        status = {"detected": detected, "reason": reason}
        status.update(details)
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_publisher.publish(message)

    def _publish_debug(self, rgb_bgr: np.ndarray, rgb: Image, text: str, mouth_pixel: tuple[float, float] | None = None) -> None:
        if self._debug_publisher is None:
            return
        image = rgb_bgr.copy()
        if mouth_pixel is not None:
            u, v = (int(round(value)) for value in mouth_pixel)
            cv2.circle(image, (u, v), 7, (0, 0, 255), 2)
            cv2.drawMarker(image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        output = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = rgb.header
        self._debug_publisher.publish(output)

    @staticmethod
    def _valid_depth_patch(depth_m: np.ndarray, u: float, v: float, radius: int, min_depth: float, max_depth: float) -> float | None:
        x = int(round(u))
        y = int(round(v))
        x0, x1 = max(0, x - radius), min(depth_m.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(depth_m.shape[0], y + radius + 1)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= min_depth) & (valid <= max_depth)]
        return float(np.median(valid)) if valid.size else None

    def _depth_image_meters(self, depth: Image) -> np.ndarray | None:
        try:
            raw = self._bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        except Exception as exc:  # cv_bridge provides encoding-specific details.
            self._warn_throttled("depth_conversion", f"Cannot convert depth image: {exc}")
            return None
        if depth.encoding.upper() == "32FC1":
            return np.asarray(raw, dtype=np.float32)
        if depth.encoding.upper() == "16UC1":
            return np.asarray(raw, dtype=np.float32) * 0.001
        self._warn_throttled("depth_encoding", f"Unsupported depth encoding: {depth.encoding}")
        return None

    def _transform_to_base(
        self, point_camera: np.ndarray, camera_frame: str, stamp
    ) -> tuple[np.ndarray, str] | None:
        """Transform a camera point into base, preferring the image timestamp.

        Gazebo can publish camera frames slightly ahead of the robot-state TF
        stream. In that case, use the latest transform only when it is within
        the bounded age configured for this static/slowly-moving setup.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                self._options.base_frame,
                camera_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.20),
            )
            mode = "image_stamp"
        except Exception as stamped_error:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._options.base_frame,
                    camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.20),
                )
                image_time = stamp.sec + stamp.nanosec * 1e-9
                transform_time = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                age = abs(image_time - transform_time)
                if transform_time > 0.0 and age > self._options.max_tf_age_sec:
                    self._warn_throttled(
                        "tf_age",
                        "Latest TF is %.3f s from image time (limit %.3f s)"
                        % (age, self._options.max_tf_age_sec),
                    )
                    return None
                mode = "latest_fallback"
            except Exception as latest_error:
                self._warn_throttled(
                    "tf",
                    "Mouth pose TF lookup failed at image stamp (%s); latest TF also failed (%s)"
                    % (stamped_error, latest_error),
                )
                return None
        translation = transform.transform.translation
        point_base = _rotation_matrix(transform) @ point_camera + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return point_base, mode

    def _filter_position(self, candidate: np.ndarray) -> np.ndarray | None:
        if self._last_base_position is None:
            self._last_base_position = candidate
            self._outlier_count = 0
            return candidate
        distance = float(np.linalg.norm(candidate - self._last_base_position))
        if distance > self._options.max_jump_m and self._outlier_count < 3:
            self._outlier_count += 1
            self._warn_throttled("outlier", f"Rejecting mouth-pose jump of {distance:.3f} m")
            return None
        self._outlier_count = 0
        alpha = self._options.smoothing_alpha
        self._last_base_position = alpha * candidate + (1.0 - alpha) * self._last_base_position
        return self._last_base_position

    def _synchronized_callback(self, rgb: Image, depth: Image, camera_info: CameraInfo) -> None:
        now = time.monotonic()
        if self._options.max_rate_hz > 0.0 and now - self._last_processing_time < 1.0 / self._options.max_rate_hz:
            return
        self._last_processing_time = now
        if camera_info.k[0] <= 0.0 or camera_info.k[4] <= 0.0:
            self._status(False, "invalid_camera_intrinsics")
            self._warn_throttled("intrinsics", "CameraInfo has invalid focal lengths")
            return

        try:
            rgb_bgr = self._bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        except Exception as exc:
            self._status(False, "rgb_conversion_failed")
            self._warn_throttled("rgb_conversion", f"Cannot convert RGB image: {exc}")
            return
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(int(time.monotonic_ns() // 1_000_000), self._last_landmarker_timestamp_ms + 1)
        self._last_landmarker_timestamp_ms = timestamp_ms
        result = self._face_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb), timestamp_ms
        )
        if not result.face_landmarks:
            self._status(False, "no_face")
            self._publish_debug(rgb_bgr, rgb, "No face")
            return

        depth_m = self._depth_image_meters(depth)
        if depth_m is None:
            self._status(False, "unsupported_depth_encoding", encoding=depth.encoding)
            return
        camera_frame = camera_info.header.frame_id or depth.header.frame_id or rgb.header.frame_id
        if not camera_frame:
            self._status(False, "missing_camera_frame")
            self._warn_throttled("camera_frame", "CameraInfo and image headers have no frame_id")
            return

        # RGB and depth are co-registered for this Gazebo RGB-D sensor. Scale
        # intrinsics if the depth and CameraInfo resolutions differ.
        scale_x = depth_m.shape[1] / camera_info.width
        scale_y = depth_m.shape[0] / camera_info.height
        fx, fy = camera_info.k[0] * scale_x, camera_info.k[4] * scale_y
        cx, cy = camera_info.k[2] * scale_x, camera_info.k[5] * scale_y
        candidates: list[dict[str, object]] = []
        rejected_depth = 0
        for landmarks in result.face_landmarks:
            if max(self._options.mouth_landmarks) >= len(landmarks):
                self._warn_throttled("landmarks", "Configured mouth landmark index exceeds detected landmark count")
                continue
            mouth_u = float(np.mean([landmarks[index].x for index in self._options.mouth_landmarks]) * rgb.width)
            mouth_v = float(np.mean([landmarks[index].y for index in self._options.mouth_landmarks]) * rgb.height)
            depth_u = mouth_u * depth_m.shape[1] / rgb.width
            depth_v = mouth_v * depth_m.shape[0] / rgb.height
            z = self._valid_depth_patch(
                depth_m,
                depth_u,
                depth_v,
                self._options.depth_patch_radius_px,
                self._options.min_depth_m,
                self._options.max_depth_m,
            )
            if z is None:
                rejected_depth += 1
                continue
            point_camera = np.array(
                [(depth_u - cx) * z / fx, (depth_v - cy) * z / fy, z], dtype=np.float64
            )
            transform_result = self._transform_to_base(point_camera, camera_frame, rgb.header.stamp)
            if transform_result is None:
                continue
            point_base, tf_mode = transform_result
            candidates.append(
                {
                    "position": point_base,
                    "image_x": mouth_u,
                    "image_y": mouth_v,
                    "depth_m": z,
                    "tf_mode": tf_mode,
                }
            )

        if not candidates:
            reason = "invalid_depth" if rejected_depth else "tf_unavailable"
            self._status(False, reason, detected_faces=len(result.face_landmarks), rejected_depth=rejected_depth)
            self._publish_debug(rgb_bgr, rgb, "No valid mouth candidate")
            return

        # Image order is the left/center/right candidate contract. The
        # perception node intentionally does not select a person; the feeding
        # target manager applies the user's left/center/right request.
        candidates.sort(key=lambda candidate: float(candidate["image_x"]))
        candidate_message = String()
        candidate_message.data = json.dumps(
            {
                "frame_id": self._options.base_frame,
                "stamp_sec": rgb.header.stamp.sec + rgb.header.stamp.nanosec * 1e-9,
                "image_center_x": rgb.width * 0.5,
                "candidates": [
                    {
                        "position": [round(float(value), 6) for value in candidate["position"]],
                        "image_x": round(float(candidate["image_x"]), 3),
                        "image_y": round(float(candidate["image_y"]), 3),
                        "depth_m": round(float(candidate["depth_m"]), 4),
                    }
                    for candidate in candidates
                ],
            },
            sort_keys=True,
        )
        self._candidates_publisher.publish(candidate_message)

        # Keep publishing the historic single-pose topic for old consumers.
        # This compatibility pose is not a target-selection decision; the
        # multi-target feeding manager consumes candidates_topic instead.
        primary = min(candidates, key=lambda candidate: abs(float(candidate["image_x"]) - rgb.width * 0.5))
        filtered_point = self._filter_position(np.asarray(primary["position"], dtype=np.float64))
        if filtered_point is None:
            self._status(False, "outlier_rejected")
            self._publish_debug(
                rgb_bgr,
                rgb,
                "Primary mouth outlier rejected",
                (float(primary["image_x"]), float(primary["image_y"])),
            )
            return

        pose = PoseStamped()
        pose.header.stamp = rgb.header.stamp
        pose.header.frame_id = self._options.base_frame
        pose.pose.position.x = float(filtered_point[0])
        pose.pose.position.y = float(filtered_point[1])
        pose.pose.position.z = float(filtered_point[2])
        # This node estimates position only; orientation is intentionally not inferred.
        pose.pose.orientation.w = 1.0
        self._pose_publisher.publish(pose)
        self._status(
            True,
            "mouth_pose_published",
            candidate_count=len(candidates),
            pixel_u=round(float(primary["image_x"]), 2),
            pixel_v=round(float(primary["image_y"]), 2),
            depth_m=round(float(primary["depth_m"]), 4),
            frame_id=self._options.base_frame,
            tf_mode=str(primary["tf_mode"]),
        )
        self._publish_debug(
            rgb_bgr,
            rgb,
            f"Mouth candidates: {len(candidates)}",
            (float(primary["image_x"]), float(primary["image_y"])),
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_options(argv: Sequence[str] | None = None) -> tuple[NodeOptions, list[str]]:
    default_model = _project_root() / "assets" / "mediapipe" / "face_landmarker.task"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/wrist_rgbd/image")
    parser.add_argument("--depth-topic", default="/wrist_rgbd/depth_image")
    parser.add_argument("--camera-info-topic", default="/wrist_rgbd/camera_info")
    parser.add_argument("--output-topic", default="/detected_mouth_pose")
    parser.add_argument("--candidates-topic", default="/detected_mouth_candidates")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--debug-image-topic", default="/mouth_detection/debug_image")
    parser.add_argument("--status-topic", default="/mouth_detection/status")
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--sync-slop-sec", type=float, default=0.08)
    parser.add_argument("--depth-patch-radius-px", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--max-jump-m", type=float, default=0.12)
    parser.add_argument("--smoothing-alpha", type=float, default=0.45)
    parser.add_argument("--max-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=5.0)
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--mouth-landmarks", type=int, nargs="+", default=list(DEFAULT_MOUTH_LANDMARKS))
    parsed, ros_args = parser.parse_known_args(argv)
    if parsed.max_faces < 1:
        parser.error("--max-faces must be at least one")
    options = NodeOptions(
        rgb_topic=parsed.rgb_topic,
        depth_topic=parsed.depth_topic,
        camera_info_topic=parsed.camera_info_topic,
        output_topic=parsed.output_topic,
        candidates_topic=parsed.candidates_topic,
        base_frame=parsed.base_frame,
        debug_image_topic=parsed.debug_image_topic,
        status_topic=parsed.status_topic,
        model_path=parsed.model_path,
        sync_slop_sec=parsed.sync_slop_sec,
        depth_patch_radius_px=parsed.depth_patch_radius_px,
        min_depth_m=parsed.min_depth_m,
        max_depth_m=parsed.max_depth_m,
        max_jump_m=parsed.max_jump_m,
        smoothing_alpha=parsed.smoothing_alpha,
        max_rate_hz=parsed.max_rate_hz,
        max_tf_age_sec=parsed.max_tf_age_sec,
        max_faces=parsed.max_faces,
        mouth_landmarks=tuple(parsed.mouth_landmarks),
    )
    return options, ros_args


def main(argv: Sequence[str] | None = None) -> int:
    options, ros_args = _parse_options(argv)
    rclpy.init(args=ros_args)
    node = MouthPerceptionNode(options)
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
