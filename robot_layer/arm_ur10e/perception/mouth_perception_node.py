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
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped
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
TOOL_FRAME = "tool0"


@dataclass(frozen=True)
class NodeOptions:
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    output_topic: str
    normal_topic: str
    candidates_topic: str
    base_frame: str
    debug_image_topic: str
    status_topic: str
    continuous_status_topic: str
    mount_calibration_status: str
    model_path: Path
    sync_slop_sec: float
    depth_patch_radius_px: int
    min_depth_m: float
    max_depth_m: float
    max_jump_m: float
    smoothing_alpha: float
    normal_smoothing_alpha: float
    max_rate_hz: float
    max_tf_age_sec: float
    max_faces: int
    mouth_landmarks: tuple[int, ...]


def _operator_warning_from_tracking_status(payload: str) -> tuple[str, ...]:
    """Return the safety-stop rows requested by the tracking runner."""
    try:
        status = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(status, dict) or status.get("collision_warning") is not True:
        return ()
    warning = status.get("operator_warning")
    if not isinstance(warning, str) or not warning.strip():
        return ()
    if status.get("state") == "SAFETY_STOPPED":
        return (warning.strip(), "HOLD not reached - guarded return only")
    return (warning.strip(),)


def _guarded_return_clears_operator_warning(payload: str) -> bool:
    """Accept only a verified guarded-return event as an overlay clear."""
    try:
        status = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(
        isinstance(status, dict)
        and status.get("state") == "GUARDED_RETURN_COMPLETE"
        and status.get("guarded_return_verified") is True
        and status.get("collision_warning") is False
    )


def _updated_operator_warning_lines(
    current: Sequence[str],
    payload: str,
) -> tuple[str, ...]:
    """Latch a collision warning until a verified guarded return clears it."""
    warning = _operator_warning_from_tracking_status(payload)
    if warning:
        return warning
    if _guarded_return_clears_operator_warning(payload):
        return ()
    return tuple(current)


def _candidate_image_labels(count: int) -> tuple[str, ...]:
    """Label valid mouths using the camera-image target-selection contract."""
    if count <= 0:
        return ()
    if count == 1:
        return ("C",)
    return tuple(
        "L" if index == 0 else "R" if index == count - 1 else "C"
        for index in range(count)
    )


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


def _displacement_scale_diagnostic(
    *,
    camera_translation_m: float,
    camera_forward_m: float,
    expected_camera_to_mouth_m: float,
    detected_mouth_jump_m: float,
) -> dict[str, float | None]:
    """Describe camera motion versus a raw base-frame mouth-position jump.

    These ratios are diagnostics, not calibration factors.  In particular,
    ``expected_range_over_jump`` is the requested rough ``0.094 / 1.46``
    comparison: it helps expose a close-range depth failure but must never be
    used to rescale a robot target.
    """
    values = (
        camera_translation_m,
        camera_forward_m,
        expected_camera_to_mouth_m,
        detected_mouth_jump_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("displacement diagnostic inputs must be finite")
    if (
        min(
            camera_translation_m,
            expected_camera_to_mouth_m,
            detected_mouth_jump_m,
        )
        < 0.0
    ):
        raise ValueError("displacement diagnostic magnitudes must be non-negative")
    if detected_mouth_jump_m <= 1.0e-9:
        return {
            "camera_translation_m": float(camera_translation_m),
            "camera_forward_m": float(camera_forward_m),
            "expected_camera_to_mouth_m": float(expected_camera_to_mouth_m),
            "detected_mouth_jump_m": float(detected_mouth_jump_m),
            "camera_translation_over_jump": None,
            "expected_range_over_jump": None,
            "jump_over_expected_range": None,
        }
    return {
        "camera_translation_m": float(camera_translation_m),
        "camera_forward_m": float(camera_forward_m),
        "expected_camera_to_mouth_m": float(expected_camera_to_mouth_m),
        "detected_mouth_jump_m": float(detected_mouth_jump_m),
        "camera_translation_over_jump": float(
            camera_translation_m / detected_mouth_jump_m
        ),
        "expected_range_over_jump": float(
            expected_camera_to_mouth_m / detected_mouth_jump_m
        ),
        "jump_over_expected_range": (
            None
            if expected_camera_to_mouth_m <= 1.0e-9
            else float(detected_mouth_jump_m / expected_camera_to_mouth_m)
        ),
    }


def _surface_normal_from_depth(
    depth_m: np.ndarray,
    center_u: float,
    center_v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius_px: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray | None:
    """Fit a camera-frame surface normal near the mouth depth sample.

    The normal is flipped toward the camera.  For a printed face or screen
    viewed by the wrist camera, that is the physical pre-mouth side.
    """
    if fx <= 0.0 or fy <= 0.0:
        return None
    center_x = int(round(center_u))
    center_y = int(round(center_v))
    radius = max(3, int(radius_px))
    points: list[list[float]] = []
    for pixel_y in range(max(0, center_y - radius), min(depth_m.shape[0], center_y + radius + 1)):
        for pixel_x in range(max(0, center_x - radius), min(depth_m.shape[1], center_x + radius + 1)):
            z = float(depth_m[pixel_y, pixel_x])
            if not math.isfinite(z) or not min_depth_m <= z <= max_depth_m:
                continue
            points.append([(pixel_x - cx) * z / fx, (pixel_y - cy) * z / fy, z])
    if len(points) < 12:
        return None
    cloud = np.asarray(points, dtype=np.float64)
    centroid = np.mean(cloud, axis=0)
    try:
        _, _, right_vectors = np.linalg.svd(cloud - centroid, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = np.asarray(right_vectors[-1], dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude < 1e-8:
        return None
    normal /= magnitude
    # Camera optical +Z points away from the camera.  The face/screen side
    # visible to the camera therefore has a normal toward the optical origin.
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal
    return normal


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
        self._normal_publisher = self.create_publisher(Vector3Stamped, options.normal_topic, qos)
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
        self._continuous_operator_warning_lines: tuple[str, ...] = ()
        self._continuous_status_subscription = self.create_subscription(
            String,
            options.continuous_status_topic,
            self._continuous_status_callback,
            qos,
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
        self._last_base_normal: np.ndarray | None = None
        self._outlier_count = 0
        self._last_face_seen_monotonic: float | None = None
        self._last_tf_mode: str | None = None
        self._latest_camera_frame: str | None = None
        self._latest_camera_position_base: np.ndarray | None = None
        self._latest_camera_optical_z_base: np.ndarray | None = None
        self._last_accepted_camera_position_base: np.ndarray | None = None
        self._last_accepted_camera_optical_z_base: np.ndarray | None = None
        self._last_displacement_diagnostic: dict[str, float | None] | None = None
        self._last_valid_mouth_depth_m: float | None = None
        self._depth_invalid_started_monotonic: float | None = None
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

    def _continuous_status_callback(self, message: String) -> None:
        """Latch collision warnings until the guarded return is verified."""
        self._continuous_operator_warning_lines = _updated_operator_warning_lines(
            self._continuous_operator_warning_lines,
            message.data,
        )

    @staticmethod
    def _format_xyz(label: str, value: Sequence[float] | None) -> str:
        """Format one live coordinate row for the rqt debug image."""
        if value is None or len(value) != 3:
            return f"{label}: unavailable"
        xyz = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in xyz):
            return f"{label}: unavailable"
        return f"{label}: ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m"

    def _debug_coordinate_lines(
        self,
        mouth_base_position: Sequence[float] | None,
    ) -> list[str]:
        """Read current TF without blocking perception and build overlay rows."""
        lines = ["base_link origin: (+0.000, +0.000, +0.000) m"]
        tool_transform = None
        try:
            tool_transform = self._tf_buffer.lookup_transform(
                self._options.base_frame,
                TOOL_FRAME,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except Exception:
            pass
        tool_position = None
        if tool_transform is not None:
            translation = tool_transform.transform.translation
            tool_position = np.array(
                [translation.x, translation.y, translation.z], dtype=np.float64
            )
        lines.append(self._format_xyz(f"{TOOL_FRAME} @ base_link", tool_position))

        camera_position = self._latest_camera_position_base
        if self._latest_camera_frame is not None:
            try:
                camera_transform = self._tf_buffer.lookup_transform(
                    self._options.base_frame,
                    self._latest_camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
                camera_translation = camera_transform.transform.translation
                camera_position = np.array(
                    [
                        camera_translation.x,
                        camera_translation.y,
                        camera_translation.z,
                    ],
                    dtype=np.float64,
                )
            except Exception:
                pass
        lines.append(self._format_xyz("camera @ base_link", camera_position))

        mouth_base = None
        if mouth_base_position is not None and len(mouth_base_position) == 3:
            candidate = np.asarray(mouth_base_position, dtype=np.float64)
            if np.all(np.isfinite(candidate)):
                mouth_base = candidate
        lines.append(self._format_xyz("mouth @ base_link", mouth_base))
        mouth_tool = None
        if (
            mouth_base is not None
            and tool_transform is not None
            and tool_position is not None
        ):
            # base_point = R_base_tool * tool_point + base_translation.
            mouth_tool = _rotation_matrix(tool_transform).T @ (
                mouth_base - tool_position
            )
        lines.append(self._format_xyz("mouth @ tool0", mouth_tool))

        if mouth_base is not None and camera_position is not None:
            lines.append(
                f"camera-mouth range: {np.linalg.norm(mouth_base - camera_position):.3f} m"
            )
        diagnostic = self._last_displacement_diagnostic
        if diagnostic is not None:
            lines.append(
                "camera move/fwd: "
                f"{1000.0 * float(diagnostic['camera_translation_m']):.1f} / "
                f"{1000.0 * float(diagnostic['camera_forward_m']):+.1f} mm; "
                f"mouth jump: {1000.0 * float(diagnostic['detected_mouth_jump_m']):.1f} mm"
            )
            ratio = diagnostic.get("expected_range_over_jump")
            exaggeration = diagnostic.get("jump_over_expected_range")
            if ratio is not None and exaggeration is not None:
                lines.append(
                    f"range/jump diagnostic: {float(ratio):.3f} "
                    f"({float(exaggeration):.1f}x jump/range)"
                )
        return lines

    def _publish_debug(
        self,
        rgb_bgr: np.ndarray,
        rgb: Image,
        text: str,
        mouth_pixel: tuple[float, float] | None = None,
        mouth_base_position: Sequence[float] | None = None,
        candidate_pixels: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        """Publish the annotated RGB frame consumed by rqt_image_view."""
        if self._debug_publisher is None:
            return
        image = rgb_bgr.copy()
        pixels = list(candidate_pixels or ())
        for label, pixel in zip(_candidate_image_labels(len(pixels)), pixels):
            u, v = (int(round(value)) for value in pixel)
            cv2.circle(image, (u, v), 8, (0, 165, 255), 1)
            cv2.putText(
                image,
                label,
                (u + 7, v - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
            )
        if mouth_pixel is not None:
            u, v = (int(round(value)) for value in mouth_pixel)
            cv2.circle(image, (u, v), 4, (0, 0, 255), 1)
            cv2.drawMarker(image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
        warning_lines = list(self._continuous_operator_warning_lines)
        lines = [
            *warning_lines,
            text,
            *self._debug_coordinate_lines(mouth_base_position),
        ]
        # Keep the diagnostic overlay readable without covering most of a
        # reduced-resolution camera preview.  At 640x480 this stays close to
        # the original size; smaller streams use a compact font and spacing.
        frame_scale = min(image.shape[1] / 640.0, image.shape[0] / 480.0)
        font_scale = max(0.30, min(0.48, 0.52 * frame_scale))
        line_step = max(14, min(20, int(round(22 * frame_scale))))
        left_margin = max(6, int(round(12 * frame_scale)))
        top_margin = max(15, int(round(28 * frame_scale)))
        for index, line in enumerate(lines):
            origin = (left_margin, top_margin + index * line_step)
            cv2.putText(
                image,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                image,
                line,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (
                    (0, 0, 255)
                    if index < len(warning_lines)
                    else (
                        (255, 255, 255)
                        if index == len(warning_lines)
                        else (0, 255, 255)
                    )
                ),
                1,
            )
        output = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = rgb.header
        self._debug_publisher.publish(output)

    @staticmethod
    def _valid_depth_patch(
        depth_m: np.ndarray,
        u: float,
        v: float,
        radius: int,
        min_depth: float,
        max_depth: float,
    ) -> tuple[float | None, int | None, int]:
        """Return robust mouth depth, expanding once around small depth holes."""
        x = int(round(u))
        y = int(round(v))
        base_radius = max(1, int(radius))
        # The 3 px primary patch preserves the validated coordinate policy.
        # The bounded 7 px fallback only fills local D435i holes; it does not
        # substitute another landmark or invent a depth.
        radii = (base_radius, max(base_radius, 7))
        for patch_radius in dict.fromkeys(radii):
            x0 = max(0, x - patch_radius)
            x1 = min(depth_m.shape[1], x + patch_radius + 1)
            y0 = max(0, y - patch_radius)
            y1 = min(depth_m.shape[0], y + patch_radius + 1)
            patch = depth_m[y0:y1, x0:x1]
            valid = patch[np.isfinite(patch)]
            valid = valid[(valid >= min_depth) & (valid <= max_depth)]
            if valid.size:
                return float(np.median(valid)), patch_radius, int(valid.size)
        return None, None, 0

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
    ) -> tuple[np.ndarray, str, np.ndarray] | None:
        """Transform a camera point into base, preferring the image timestamp.

        Gazebo can publish camera frames slightly ahead of the robot-state TF
        stream. In that case, use the latest transform only when it is within
        the bounded age configured for this static/slowly-moving setup.
        """
        try:
            # The mounted camera and robot are stationary during perception;
            # latest TF avoids interpolation jumps from mismatched timestamps.
            transform = self._tf_buffer.lookup_transform(
                self._options.base_frame, camera_frame, Time(), timeout=Duration(seconds=0.20)
            )
            image_time = stamp.sec + stamp.nanosec * 1e-9
            transform_time = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            age = abs(image_time - transform_time)
            if transform_time > 0.0 and age > self._options.max_tf_age_sec:
                raise RuntimeError(f"latest TF age {age:.3f}s exceeds limit")
            mode = "latest_stable"
        except Exception as latest_error:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._options.base_frame, camera_frame, Time.from_msg(stamp), timeout=Duration(seconds=0.20)
                )
                mode = "image_stamp_fallback"
            except Exception as stamped_error:
                self._warn_throttled(
                    "tf",
                    "Mouth pose latest TF failed (%s); stamped TF also failed (%s)"
                    % (latest_error, stamped_error),
                )
                return None
        translation = transform.transform.translation
        rotation = _rotation_matrix(transform)
        self._latest_camera_frame = camera_frame
        self._latest_camera_position_base = np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        self._latest_camera_optical_z_base = rotation[:, 2].copy()
        point_base = rotation @ point_camera + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return point_base, mode, rotation

    def _filter_position(self, candidate: np.ndarray) -> np.ndarray | None:
        if self._last_base_position is None:
            self._last_base_position = candidate
            self._last_accepted_camera_position_base = (
                None
                if self._latest_camera_position_base is None
                else self._latest_camera_position_base.copy()
            )
            self._last_accepted_camera_optical_z_base = (
                None
                if self._latest_camera_optical_z_base is None
                else self._latest_camera_optical_z_base.copy()
            )
            self._last_displacement_diagnostic = None
            self._outlier_count = 0
            return candidate
        distance = float(np.linalg.norm(candidate - self._last_base_position))
        if (
            self._latest_camera_position_base is not None
            and self._last_accepted_camera_position_base is not None
        ):
            camera_delta = (
                self._latest_camera_position_base
                - self._last_accepted_camera_position_base
            )
            optical_z = self._last_accepted_camera_optical_z_base
            camera_forward = (
                0.0
                if optical_z is None
                else float(np.dot(camera_delta, optical_z))
            )
            expected_range = float(
                np.linalg.norm(
                    self._last_base_position - self._latest_camera_position_base
                )
            )
            self._last_displacement_diagnostic = _displacement_scale_diagnostic(
                camera_translation_m=float(np.linalg.norm(camera_delta)),
                camera_forward_m=camera_forward,
                expected_camera_to_mouth_m=expected_range,
                detected_mouth_jump_m=distance,
            )
        if distance > self._options.max_jump_m and self._outlier_count < 3:
            self._outlier_count += 1
            self._warn_throttled("outlier", f"Rejecting mouth-pose jump of {distance:.3f} m")
            return None
        self._outlier_count = 0
        alpha = self._options.smoothing_alpha
        self._last_base_position = alpha * candidate + (1.0 - alpha) * self._last_base_position
        self._last_accepted_camera_position_base = (
            None
            if self._latest_camera_position_base is None
            else self._latest_camera_position_base.copy()
        )
        self._last_accepted_camera_optical_z_base = (
            None
            if self._latest_camera_optical_z_base is None
            else self._latest_camera_optical_z_base.copy()
        )
        return self._last_base_position

    def _reset_filter_state(self) -> None:
        """Drop stale pose smoothing after a genuine face-loss interval."""
        self._last_base_position = None
        self._last_base_normal = None
        self._outlier_count = 0
        self._last_accepted_camera_position_base = None
        self._last_accepted_camera_optical_z_base = None
        self._last_displacement_diagnostic = None

    def _filter_normal(self, candidate: np.ndarray) -> np.ndarray | None:
        magnitude = float(np.linalg.norm(candidate))
        if not math.isfinite(magnitude) or magnitude < 1e-8:
            return None
        candidate = candidate / magnitude
        if self._last_base_normal is None:
            self._last_base_normal = candidate
            return self._last_base_normal
        # Plane fitting has a sign ambiguity.  Keep it continuous before EMA.
        if float(np.dot(candidate, self._last_base_normal)) < 0.0:
            candidate = -candidate
        alpha = self._options.normal_smoothing_alpha
        filtered = alpha * candidate + (1.0 - alpha) * self._last_base_normal
        filtered_magnitude = float(np.linalg.norm(filtered))
        if filtered_magnitude < 1e-8:
            return None
        self._last_base_normal = filtered / filtered_magnitude
        return self._last_base_normal

    def _synchronized_callback(self, rgb: Image, depth: Image, camera_info: CameraInfo) -> None:
        now = time.monotonic()
        if self._options.max_rate_hz > 0.0 and now - self._last_processing_time < 1.0 / self._options.max_rate_hz:
            return
        self._last_processing_time = now
        if camera_info.k[0] <= 0.0 or camera_info.k[4] <= 0.0:
            self._status(False, "invalid_camera_intrinsics")
            self._warn_throttled("intrinsics", "CameraInfo has invalid focal lengths")
            return
        header_camera_frame = (
            camera_info.header.frame_id or depth.header.frame_id or rgb.header.frame_id
        )
        if header_camera_frame:
            self._latest_camera_frame = header_camera_frame

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
            self._depth_invalid_started_monotonic = None
            if (
                self._last_face_seen_monotonic is not None
                and now - self._last_face_seen_monotonic >= 0.75
            ):
                self._reset_filter_state()
            self._status(False, "no_face")
            self._publish_debug(rgb_bgr, rgb, "No face")
            return

        # Reacquisition after a loss starts a new camera/perception reference.
        if (
            self._last_face_seen_monotonic is not None
            and now - self._last_face_seen_monotonic >= 0.75
        ):
            self._reset_filter_state()
        self._last_face_seen_monotonic = now

        depth_m = self._depth_image_meters(depth)
        if depth_m is None:
            self._depth_invalid_started_monotonic = None
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
            z, depth_patch_radius, valid_depth_samples = self._valid_depth_patch(
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
            normal_camera = _surface_normal_from_depth(
                depth_m,
                depth_u,
                depth_v,
                fx,
                fy,
                cx,
                cy,
                self._options.depth_patch_radius_px,
                self._options.min_depth_m,
                self._options.max_depth_m,
            )
            transform_result = self._transform_to_base(point_camera, camera_frame, rgb.header.stamp)
            if transform_result is None:
                continue
            point_base, tf_mode, rotation = transform_result
            # A timestamped/latest TF switch can create an artificial jump.
            if self._last_tf_mode is not None and tf_mode != self._last_tf_mode:
                self._reset_filter_state()
            self._last_tf_mode = tf_mode
            normal_base = None if normal_camera is None else rotation @ normal_camera
            candidates.append(
                {
                    "position": point_base,
                    "surface_normal": normal_base,
                    "image_x": mouth_u,
                    "image_y": mouth_v,
                    "depth_m": z,
                    "depth_patch_radius_px": depth_patch_radius,
                    "valid_depth_samples": valid_depth_samples,
                    "tf_mode": tf_mode,
                }
            )

        if not candidates:
            reason = "invalid_depth" if rejected_depth else "tf_unavailable"
            if reason == "invalid_depth":
                if self._depth_invalid_started_monotonic is None:
                    self._depth_invalid_started_monotonic = time.monotonic()
                depth_invalid_duration_sec = max(
                    0.0,
                    time.monotonic() - self._depth_invalid_started_monotonic,
                )
            else:
                self._depth_invalid_started_monotonic = None
                depth_invalid_duration_sec = 0.0
            self._status(
                False,
                reason,
                detected_faces=len(result.face_landmarks),
                rejected_depth=rejected_depth,
                valid_depth_samples=0,
                minimum_depth_m=round(float(self._options.min_depth_m), 4),
                last_valid_depth_m=(
                    "unavailable"
                    if self._last_valid_mouth_depth_m is None
                    else round(float(self._last_valid_mouth_depth_m), 4)
                ),
                depth_invalid_duration_sec=round(depth_invalid_duration_sec, 4),
            )
            self._publish_debug(rgb_bgr, rgb, "No valid mouth candidate")
            return

        # Image order is the left/center/right candidate contract. The
        # perception node intentionally does not select a person; the feeding
        # target manager applies the user's left/center/right request.
        candidates.sort(key=lambda candidate: float(candidate["image_x"]))
        candidate_pixels = [
            (float(candidate["image_x"]), float(candidate["image_y"]))
            for candidate in candidates
        ]
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
                        "depth_patch_radius_px": int(
                            candidate["depth_patch_radius_px"]
                        ),
                        "valid_depth_samples": int(
                            candidate["valid_depth_samples"]
                        ),
                        "surface_normal": None
                        if candidate["surface_normal"] is None
                        else [round(float(value), 6) for value in candidate["surface_normal"]],
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
        self._last_valid_mouth_depth_m = float(primary["depth_m"])
        self._depth_invalid_started_monotonic = None
        filtered_point = self._filter_position(np.asarray(primary["position"], dtype=np.float64))
        if filtered_point is None:
            self._status(False, "outlier_rejected")
            self._publish_debug(
                rgb_bgr,
                rgb,
                "Primary mouth outlier rejected",
                (float(primary["image_x"]), float(primary["image_y"])),
                primary["position"],
                candidate_pixels,
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
        normal = primary["surface_normal"]
        if normal is not None:
            normal = self._filter_normal(np.asarray(normal, dtype=np.float64))
        if normal is not None:
            normal_message = Vector3Stamped()
            normal_message.header = pose.header
            normal_message.vector.x = float(normal[0])
            normal_message.vector.y = float(normal[1])
            normal_message.vector.z = float(normal[2])
            self._normal_publisher.publish(normal_message)
        self._status(
            True,
            "mouth_pose_published",
            candidate_count=len(candidates),
            pixel_u=round(float(primary["image_x"]), 2),
            pixel_v=round(float(primary["image_y"]), 2),
            depth_m=round(float(primary["depth_m"]), 4),
            minimum_depth_m=round(float(self._options.min_depth_m), 4),
            depth_patch_radius_px=int(primary["depth_patch_radius_px"]),
            valid_depth_samples=int(primary["valid_depth_samples"]),
            frame_id=self._options.base_frame,
            tf_mode=str(primary["tf_mode"]),
            surface_normal_available=normal is not None,
            mount_calibration_status=self._options.mount_calibration_status,
        )
        self._publish_debug(
            rgb_bgr,
            rgb,
            f"Mouth candidates: {len(candidates)}",
            (float(primary["image_x"]), float(primary["image_y"])),
            filtered_point,
            candidate_pixels,
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
    parser.add_argument("--normal-topic", default="/detected_mouth_normal")
    parser.add_argument("--candidates-topic", default="/detected_mouth_candidates")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--debug-image-topic", default="/mouth_detection/debug_image")
    parser.add_argument("--status-topic", default="/mouth_detection/status")
    parser.add_argument(
        "--continuous-status-topic",
        default="/continuous_mouth_tracking/status",
    )
    parser.add_argument("--mount-calibration-status", default="unknown")
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--sync-slop-sec", type=float, default=0.08)
    parser.add_argument("--depth-patch-radius-px", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--max-jump-m", type=float, default=0.12)
    parser.add_argument("--smoothing-alpha", type=float, default=0.45)
    parser.add_argument("--normal-smoothing-alpha", type=float, default=0.20)
    parser.add_argument("--max-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=5.0)
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--mouth-landmarks", type=int, nargs="+", default=list(DEFAULT_MOUTH_LANDMARKS))
    parsed, ros_args = parser.parse_known_args(argv)
    if parsed.max_faces < 1:
        parser.error("--max-faces must be at least one")
    if not 0.0 < parsed.normal_smoothing_alpha <= 1.0:
        parser.error("--normal-smoothing-alpha must be in (0, 1]")
    options = NodeOptions(
        rgb_topic=parsed.rgb_topic,
        depth_topic=parsed.depth_topic,
        camera_info_topic=parsed.camera_info_topic,
        output_topic=parsed.output_topic,
        normal_topic=parsed.normal_topic,
        candidates_topic=parsed.candidates_topic,
        base_frame=parsed.base_frame,
        debug_image_topic=parsed.debug_image_topic,
        status_topic=parsed.status_topic,
        continuous_status_topic=parsed.continuous_status_topic,
        mount_calibration_status=parsed.mount_calibration_status,
        model_path=parsed.model_path,
        sync_slop_sec=parsed.sync_slop_sec,
        depth_patch_radius_px=parsed.depth_patch_radius_px,
        min_depth_m=parsed.min_depth_m,
        max_depth_m=parsed.max_depth_m,
        max_jump_m=parsed.max_jump_m,
        smoothing_alpha=parsed.smoothing_alpha,
        normal_smoothing_alpha=parsed.normal_smoothing_alpha,
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
