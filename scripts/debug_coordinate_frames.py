#!/usr/bin/env python3
"""No-motion comparison of base_link, tool0/TCP, and camera optical axes.

This diagnostic subscribes to perception/camera topics, reads TF, publishes
RViz markers, and writes one JSON report.  It does not import MoveIt, create
action clients, call controller services, or publish robot commands.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_SOURCE = PROJECT_ROOT / "robot_layer/arm_ur10e/perception/mouth_perception_node.py"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

BASE_FRAME = "base_link"
PENDANT_BASE_FRAME = "base"
TOOL_FRAME = "tool0"
DEFAULT_CAMERA_FRAME = "d435i_color_optical_frame"
MOUTH_TOPIC = "/detected_mouth_pose"
CANDIDATES_TOPIC = "/detected_mouth_candidates"
STATUS_TOPIC = "/mouth_detection/status"
RGB_TOPIC = "/d435i/d435i/color/image_raw"
DEPTH_TOPIC = "/d435i/d435i/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/d435i/d435i/color/camera_info"
MARKER_TOPIC = "/coordinate_frame_debug"

STRAW_TIP_OFFSET_TOOL0_M = (0.110, 0.0, 0.0)
SUPPLIED_CAMERA_CENTER_TOOL0_M = (0.070, 0.0, 0.015)
MIN_STABLE_SAMPLES = 3
MAX_STABLE_RADIAL_SPREAD_M = 0.025
TF_DISCOVERY_TIMEOUT_SEC = 8.0
PROJECTION_RESIDUAL_TOLERANCE_M = 0.020
AXIS_ALIGNMENT_THRESHOLD = 0.70


@dataclass(frozen=True)
class MouthSample:
    position_m: tuple[float, float, float]
    received_monotonic: float
    source_time_sec: float | None


def _add(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(first, second)]


def _subtract(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(first, second)]


def _scale(vector: Sequence[float], scalar: float) -> list[float]:
    return [float(component) * float(scalar) for component in vector]


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> list[float] | None:
    magnitude = _norm(vector)
    if not math.isfinite(magnitude) or magnitude < 1e-9:
        return None
    return [float(component) / magnitude for component in vector]


def _rotate(quaternion_xyzw: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if magnitude < 1e-12:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = (value / magnitude for value in (x, y, z, w))
    vx, vy, vz = (float(value) for value in vector)
    return [
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    ]


def _transform_point(transform: dict[str, Any], point: Sequence[float]) -> list[float] | None:
    """Transform a position using a transform returned by ``Node.transform``."""
    if not transform.get("available"):
        return None
    return _add(
        _rotate(transform["orientation_quat_xyzw"], point),
        transform["translation_m"],
    )


def _transform_axes(
    transform: dict[str, Any], axes: dict[str, Sequence[float]]
) -> dict[str, list[float]] | None:
    """Rotate direction vectors; unlike positions, axes do not use translation."""
    if not transform.get("available"):
        return None
    return {
        name: _rotate(transform["orientation_quat_xyzw"], vector)
        for name, vector in axes.items()
    }


def _axes_in_parent(quaternion_xyzw: Sequence[float]) -> dict[str, list[float]]:
    return {
        "+X": _rotate(quaternion_xyzw, (1.0, 0.0, 0.0)),
        "+Y": _rotate(quaternion_xyzw, (0.0, 1.0, 0.0)),
        "+Z": _rotate(quaternion_xyzw, (0.0, 0.0, 1.0)),
    }


def _mean(points: Sequence[Sequence[float]]) -> list[float]:
    return [sum(float(point[index]) for point in points) / len(points) for index in range(3)]


def _sample_summary(samples: Sequence[MouthSample], duration_sec: float, now_monotonic: float) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "sample_count": 0,
            "duration_sec": duration_sec,
            "stable": False,
            "reason": "no valid /detected_mouth_pose samples were received",
        }
    points = [list(sample.position_m) for sample in samples]
    mean = _mean(points)
    minima = [min(point[index] for point in points) for index in range(3)]
    maxima = [max(point[index] for point in points) for index in range(3)]
    stddev = [
        math.sqrt(sum((point[index] - mean[index]) ** 2 for point in points) / len(points))
        for index in range(3)
    ]
    radial_spread = max(_norm(_subtract(point, mean)) for point in points)
    latest = samples[-1]
    return {
        "available": True,
        "sample_count": len(points),
        "duration_sec": duration_sec,
        "mean_position_m": mean,
        "stddev_m": stddev,
        "min_m": minima,
        "max_m": maxima,
        "axis_spread_m": _subtract(maxima, minima),
        "max_distance_from_mean_m": radial_spread,
        "latest_receive_age_sec": max(0.0, now_monotonic - latest.received_monotonic),
        "latest_source_stamp_sec": latest.source_time_sec,
        "stable": len(points) >= MIN_STABLE_SAMPLES and radial_spread <= MAX_STABLE_RADIAL_SPREAD_M,
        "stability_requirements": {
            "minimum_samples": MIN_STABLE_SAMPLES,
            "maximum_distance_from_mean_m": MAX_STABLE_RADIAL_SPREAD_M,
        },
    }


def _line_for(lines: Sequence[str], fragment: str) -> int | None:
    return next((index for index, line in enumerate(lines, start=1) if fragment in line), None)


def _line_for_after(lines: Sequence[str], fragment: str, after_fragment: str) -> int | None:
    after_line = _line_for(lines, after_fragment)
    if after_line is None:
        return None
    return next(
        (index for index, line in enumerate(lines[after_line:], start=after_line + 1) if fragment in line),
        None,
    )


def _perception_source_audit() -> dict[str, Any]:
    try:
        source = PERCEPTION_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        return {"available": False, "source_file": str(PERCEPTION_SOURCE), "reason": str(exc)}
    lines = source.splitlines()
    formula_x = "(depth_u - cx) * z / fx"
    formula_y = "(depth_v - cy) * z / fy"
    frame_selection = "camera_info.header.frame_id or depth.header.frame_id or rgb.header.frame_id"
    transform_target = "self._options.base_frame,"
    output_frame = "pose.header.frame_id = self._options.base_frame"
    return {
        "available": True,
        "source_file": str(PERCEPTION_SOURCE),
        "configured_defaults": {
            "rgb_topic": "/wrist_rgbd/image",
            "depth_topic": "/wrist_rgbd/depth_image",
            "camera_info_topic": "/wrist_rgbd/camera_info",
            "output_topic": "/detected_mouth_pose",
            "base_frame": "base_link",
        },
        "live_wrapper_overrides_defaults": True,
        "camera_frame_selection_expression": frame_selection,
        "camera_frame_selection_line": _line_for(lines, frame_selection),
        "camera_frame_precedence": [
            "camera_info.header.frame_id",
            "depth.header.frame_id",
            "rgb.header.frame_id",
        ],
        "optical_projection_formula": {
            "x": "(u - cx) * depth / fx",
            "y": "(v - cy) * depth / fy",
            "z": "depth",
            "x_line": _line_for(lines, formula_x),
            "y_line": _line_for(lines, formula_y),
            "z_line": _line_for(lines, formula_x),
            "verified_in_source": formula_x in source and formula_y in source,
        },
        "transform_target": "base_link through self._options.base_frame",
        "transform_lookup_line": _line_for_after(
            lines,
            transform_target,
            "transform = self._tf_buffer.lookup_transform(",
        ),
        "published_pose_frame": "self._options.base_frame",
        "published_pose_frame_line": _line_for(lines, output_frame),
        "publishes_base_frame_verified": output_frame in source,
    }


class CoordinateFrameDebugNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("debug_coordinate_frames")
        self.args = args
        self.samples: list[MouthSample] = []
        self.rejected_wrong_frame = 0
        self.latest_camera_info: dict[str, Any] | None = None
        self.latest_rgb: dict[str, Any] | None = None
        self.latest_depth: dict[str, Any] | None = None
        self.latest_candidates: dict[str, Any] | None = None
        self.latest_candidates_received: float | None = None
        self.latest_status: dict[str, Any] | None = None

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, args.mouth_topic, self._mouth_callback, reliable_qos)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_callback, sensor_qos)
        self.create_subscription(Image, args.rgb_topic, self._rgb_callback, sensor_qos)
        self.create_subscription(Image, args.depth_topic, self._depth_callback, sensor_qos)
        self.create_subscription(String, args.candidates_topic, self._candidates_callback, reliable_qos)
        self.create_subscription(String, args.status_topic, self._status_callback, reliable_qos)
        self.marker_publisher = self.create_publisher(MarkerArray, args.marker_topic, marker_qos)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

    @staticmethod
    def _header(message: Image | CameraInfo) -> dict[str, Any]:
        stamp = message.header.stamp
        return {
            "frame_id": message.header.frame_id,
            "stamp_sec": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            "width": int(message.width),
            "height": int(message.height),
        }

    def _mouth_callback(self, message: PoseStamped) -> None:
        frame = message.header.frame_id.strip().lstrip("/")
        if frame != self.args.base_frame.strip().lstrip("/"):
            self.rejected_wrong_frame += 1
            return
        point = message.pose.position
        position = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(value) for value in position):
            return
        stamp = message.header.stamp
        source_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.samples.append(MouthSample(position, time.monotonic(), source_time if source_time > 0.0 else None))

    def _camera_info_callback(self, message: CameraInfo) -> None:
        value = self._header(message)
        value.update(
            {
                "distortion_model": message.distortion_model,
                "k": [float(component) for component in message.k],
            }
        )
        self.latest_camera_info = value

    def _rgb_callback(self, message: Image) -> None:
        value = self._header(message)
        value["encoding"] = message.encoding
        self.latest_rgb = value

    def _depth_callback(self, message: Image) -> None:
        value = self._header(message)
        value["encoding"] = message.encoding
        self.latest_depth = value

    def _candidates_callback(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self.latest_candidates = value
            self.latest_candidates_received = time.monotonic()

    def _status_callback(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            value = {"raw": message.data}
        self.latest_status = value if isinstance(value, dict) else {"raw": value}

    def collect(self, duration_sec: float) -> None:
        self.samples.clear()
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))

    def transform(self, target_frame: str, source_frame: str) -> dict[str, Any]:
        deadline = time.monotonic() + TF_DISCOVERY_TIMEOUT_SEC
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.0),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                quaternion = [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
                return {
                    "available": True,
                    "target_frame": target_frame,
                    "source_frame": source_frame,
                    "interpretation": f"pose of {source_frame} expressed in {target_frame}",
                    "translation_m": [float(translation.x), float(translation.y), float(translation.z)],
                    "orientation_quat_xyzw": quaternion,
                    "axes_expressed_in_target": _axes_in_parent(quaternion),
                }
            except Exception as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.05)
        return {
            "available": False,
            "target_frame": target_frame,
            "source_frame": source_frame,
            "reason": str(last_error) if last_error is not None else "TF lookup timed out",
        }

    @staticmethod
    def point(position: Sequence[float]) -> Point:
        point = Point()
        point.x, point.y, point.z = (float(value) for value in position)
        return point

    def _sphere(
        self,
        markers: list[Marker],
        marker_id: int,
        position: Sequence[float],
        color: tuple[float, float, float],
        label: str,
    ) -> None:
        now = self.get_clock().now().to_msg()
        sphere = Marker()
        sphere.header.frame_id = self.args.base_frame
        sphere.header.stamp = now
        sphere.ns = "coordinate_frame_debug_points"
        sphere.id = marker_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = self.point(position)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.030
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (*color, 0.95)
        markers.append(sphere)

        text = Marker()
        text.header = sphere.header
        text.ns = "coordinate_frame_debug_labels"
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = self.point(_add(position, (0.0, 0.0, 0.045)))
        text.pose.orientation.w = 1.0
        text.scale.z = 0.025
        text.color.r, text.color.g, text.color.b, text.color.a = (*color, 1.0)
        text.text = f"{label} [{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}]"
        markers.append(text)

    def _arrow(
        self,
        markers: list[Marker],
        marker_id: int,
        start: Sequence[float],
        end: Sequence[float],
        color: tuple[float, float, float],
        label: str,
    ) -> None:
        now = self.get_clock().now().to_msg()
        arrow = Marker()
        arrow.header.frame_id = self.args.base_frame
        arrow.header.stamp = now
        arrow.ns = "coordinate_frame_debug_arrows"
        arrow.id = marker_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [self.point(start), self.point(end)]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = (0.008, 0.017, 0.024)
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (*color, 0.95)
        markers.append(arrow)

        midpoint = _scale(_add(start, end), 0.5)
        text = Marker()
        text.header = arrow.header
        text.ns = "coordinate_frame_debug_arrow_labels"
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = self.point(_add(midpoint, (0.0, 0.0, 0.025)))
        text.pose.orientation.w = 1.0
        text.scale.z = 0.020
        text.color.r, text.color.g, text.color.b, text.color.a = (*color, 1.0)
        text.text = label
        markers.append(text)

    def publish_markers(self, geometry: dict[str, Any]) -> int:
        markers: list[Marker] = []
        clear = Marker()
        clear.header.frame_id = self.args.base_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        markers.append(clear)

        tool = geometry.get("tool0_position_m")
        camera = geometry.get("camera_center_m")
        straw = geometry.get("straw_tip_m")
        mouth = geometry.get("mouth_mean_m")
        if mouth is not None:
            self._sphere(markers, 1, mouth, (1.0, 0.0, 0.0), "detected mouth")
        if straw is not None:
            self._sphere(markers, 2, straw, (0.0, 1.0, 0.0), "straw tip")
        if tool is not None:
            self._sphere(markers, 3, tool, (0.0, 0.25, 1.0), "tool0/TCP")
        if camera is not None:
            self._sphere(markers, 4, camera, (0.0, 1.0, 1.0), "camera optical center")

        axis_colors = {"+X": (1.0, 0.15, 0.15), "+Y": (0.15, 1.0, 0.15), "+Z": (0.2, 0.4, 1.0)}
        base_origin = (0.0, 0.0, 0.0)
        for index, axis in enumerate(("+X", "+Y", "+Z")):
            direction = geometry["base_axes"][axis]
            self._arrow(markers, 10 + index, base_origin, _add(base_origin, _scale(direction, 0.25)), axis_colors[axis], f"base_link {axis}")
        if tool is not None:
            for index, axis in enumerate(("+X", "+Y", "+Z")):
                direction = geometry["tool0_axes_in_base_link"][axis]
                self._arrow(markers, 20 + index, tool, _add(tool, _scale(direction, 0.18)), axis_colors[axis], f"tool0/TCP {axis}")
        if camera is not None:
            for index, axis in enumerate(("+X", "+Y", "+Z")):
                direction = geometry["camera_axes_in_base_link"][axis]
                self._arrow(markers, 30 + index, camera, _add(camera, _scale(direction, 0.18)), axis_colors[axis], f"camera optical {axis}")
        if camera is not None and mouth is not None:
            self._arrow(markers, 40, camera, mouth, (1.0, 0.65, 0.0), "camera -> mouth")
        if straw is not None and mouth is not None:
            self._arrow(markers, 41, straw, mouth, (0.8, 0.1, 1.0), "straw tip -> mouth")

        self.marker_publisher.publish(MarkerArray(markers=markers))
        deadline = time.monotonic() + self.args.marker_hold_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
        return len(markers) - 1


def _independent_projection_check(
    node: CoordinateFrameDebugNode,
    camera_tf: dict[str, Any],
    mouth_mean: Sequence[float] | None,
) -> dict[str, Any]:
    info = node.latest_camera_info
    rgb = node.latest_rgb
    depth = node.latest_depth
    payload = node.latest_candidates
    if not camera_tf.get("available"):
        return {"available": False, "reason": "camera optical TF is unavailable"}
    if info is None or rgb is None or depth is None:
        return {"available": False, "reason": "RGB/depth/CameraInfo headers are incomplete"}
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return {"available": False, "reason": "no live candidate pixel/depth payload was received"}
    image_center_x = float(payload.get("image_center_x", rgb["width"] * 0.5))
    candidate = min(candidates, key=lambda item: abs(float(item["image_x"]) - image_center_x))
    scale_x = float(depth["width"]) / float(info["width"])
    scale_y = float(depth["height"]) / float(info["height"])
    depth_u = float(candidate["image_x"]) * float(depth["width"]) / float(rgb["width"])
    depth_v = float(candidate["image_y"]) * float(depth["height"]) / float(rgb["height"])
    fx, fy = info["k"][0] * scale_x, info["k"][4] * scale_y
    cx, cy = info["k"][2] * scale_x, info["k"][5] * scale_y
    z = float(candidate["depth_m"])
    point_camera = [(depth_u - cx) * z / fx, (depth_v - cy) * z / fy, z]
    point_base = _add(
        camera_tf["translation_m"],
        _rotate(camera_tf["orientation_quat_xyzw"], point_camera),
    )
    published_candidate = [float(value) for value in candidate["position"]]
    candidate_residual = _norm(_subtract(point_base, published_candidate))
    return {
        "available": True,
        "selected_candidate": candidate,
        "camera_info_frame": info["frame_id"],
        "point_reconstructed_in_camera_optical_m": point_camera,
        "point_reconstructed_in_base_link_m": point_base,
        "published_candidate_in_base_link_m": published_candidate,
        "reconstruction_to_published_candidate_residual_m": candidate_residual,
        "reconstruction_to_smoothed_mouth_mean_residual_m": None
        if mouth_mean is None
        else _norm(_subtract(point_base, mouth_mean)),
        "within_projection_tolerance": candidate_residual <= PROJECTION_RESIDUAL_TOLERANCE_M,
        "projection_tolerance_m": PROJECTION_RESIDUAL_TOLERANCE_M,
        "image_center_offset_fraction": abs(float(candidate["image_x"]) - image_center_x) / float(rgb["width"]),
    }


def _verdict(
    alignments: dict[str, float] | None,
    projection: dict[str, Any],
    runtime_frames: dict[str, Any],
) -> dict[str, Any]:
    if alignments is None:
        return {
            "primary_category": None,
            "supported_categories": [],
            "verdict": "Inconclusive: no stable mouth mean was available.",
            "proposed_next_fix": "Restore mouth detection and repeat the no-motion capture.",
        }
    tool_dot = alignments["straw_to_mouth_vs_tool0_plus_x"]
    base_dot = alignments["straw_to_mouth_vs_base_link_plus_x"]
    camera_dot = alignments["straw_to_mouth_vs_camera_optical_plus_z"]
    frames_match = bool(runtime_frames.get("perception_source_frame_matches_configured_camera_optical"))
    projection_good = bool(projection.get("available") and projection.get("within_projection_tolerance"))
    centered = projection.get("image_center_offset_fraction")
    centered_face = isinstance(centered, (int, float)) and float(centered) <= 0.20

    supported: list[str] = []
    not_supported: list[str] = []
    if centered_face and (not frames_match or camera_dot < AXIS_ALIGNMENT_THRESHOLD):
        supported.append("B")
    else:
        not_supported.append("B")
    if projection.get("available") and not projection_good:
        supported.append("C")
    elif projection_good:
        not_supported.append("C")
    if (
        projection_good
        and camera_dot >= AXIS_ALIGNMENT_THRESHOLD
        and tool_dot >= AXIS_ALIGNMENT_THRESHOLD
        and tool_dot - base_dot >= 0.30
    ):
        supported.append("A")
        supported.append("D")
    if not supported:
        verdict = "Inconclusive: the live geometry does not cleanly select A/B/C/D/E."
        primary = None
        next_fix = "Inspect the RViz axes and repeat with a centered, stable face."
    elif "B" in supported:
        verdict = "B. Camera optical frame issue is suspected."
        primary = "B"
        next_fix = "Correct the CameraInfo/source optical frame or camera extrinsic before changing target policy."
    elif "C" in supported:
        verdict = "C. Perception projection issue is suspected."
        primary = "C"
        next_fix = "Correct RGB/depth registration or the pixel/depth/intrinsics projection before planning."
    else:
        verdict = (
            "A + D. Pendant TCP +X and base_link +X are different; TF/projection are internally consistent, "
            "and a fixed base-X pre-mouth offset is a policy error rather than a camera-axis error."
        )
        primary = "A"
        next_fix = (
            "Keep TF/projection unchanged. Express the desired approach in tool/TCP or camera geometry and "
            "transform it into base_link, or use the validated configurable feeding-vector policy."
        )
    return {
        "primary_category": primary,
        "supported_categories": supported,
        "not_supported_by_this_capture": not_supported,
        "category_E_status": (
            "Not established numerically: the report uses the supplied tool0->straw offset; "
            "physical straw-marker coincidence must be checked visually in RViz."
        ),
        "verdict": verdict,
        "proposed_next_fix": next_fix,
        "thresholds": {
            "axis_alignment": AXIS_ALIGNMENT_THRESHOLD,
            "projection_residual_m": PROJECTION_RESIDUAL_TOLERANCE_M,
        },
    }


def _format_vector(vector: Sequence[float] | None) -> str:
    if vector is None:
        return "unavailable"
    return f"[{vector[0]:+.6f}, {vector[1]:+.6f}, {vector[2]:+.6f}]"


def _print_report(report: dict[str, Any], report_path: Path) -> None:
    print("\nCoordinate-frame diagnostic (no motion)")
    print(f"Perception source frame: {report['runtime_frames']['perception_source_frame_used']}")
    for name, vector in report["axes"]["tool0_axes_expressed_in_base_link"].items():
        print(f"tool0/TCP {name} in base_link: {_format_vector(vector)}")
    for name, vector in report["axes"]["camera_optical_axes_expressed_in_base_link"].items():
        print(f"camera optical {name} in base_link: {_format_vector(vector)}")
    mouth = report["mouth_samples"].get("mean_position_m")
    print(f"mouth mean in base_link: {_format_vector(mouth)}")
    pendant_positions = report.get("positions_ur_base_m", {})
    print(f"tool0 in pendant base: {_format_vector(pendant_positions.get('tool0'))}")
    print(
        "mouth mean in pendant base: "
        f"{_format_vector(pendant_positions.get('detected_mouth_mean'))}"
    )
    alignments = report["alignment_dot_products"]
    if alignments is not None:
        print(f"dot(straw->mouth, tool0 +X): {alignments['straw_to_mouth_vs_tool0_plus_x']:+.6f}")
        print(f"dot(straw->mouth, base +X):  {alignments['straw_to_mouth_vs_base_link_plus_x']:+.6f}")
        print(f"dot(straw->mouth, camera +Z): {alignments['straw_to_mouth_vs_camera_optical_plus_z']:+.6f}")
    projection = report["independent_projection_check"]
    if projection.get("available"):
        print(
            "independent projection residual: "
            f"{projection['reconstruction_to_published_candidate_residual_m']:.6f} m"
        )
    else:
        print(f"independent projection residual: unavailable ({projection.get('reason')})")
    print(f"Verdict: {report['verdict']['verdict']}")
    print(f"Proposed next fix: {report['verdict']['proposed_next_fix']}")
    print(f"Marker topic: {report['markers']['topic']} (transient_local)")
    print(f"Report: {report_path}")
    print("No robot motion was sent.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--base-frame", default=BASE_FRAME)
    parser.add_argument(
        "--pendant-base-frame",
        default=PENDANT_BASE_FRAME,
        help="UR controller/teach-pendant base frame used for explicitly labelled comparison output.",
    )
    parser.add_argument("--tool-frame", default=TOOL_FRAME)
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--mouth-topic", default=MOUTH_TOPIC)
    parser.add_argument("--candidates-topic", default=CANDIDATES_TOPIC)
    parser.add_argument("--status-topic", default=STATUS_TOPIC)
    parser.add_argument("--rgb-topic", default=RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--camera-info-topic", default=CAMERA_INFO_TOPIC)
    parser.add_argument("--marker-topic", default=MARKER_TOPIC)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--marker-hold-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")
    if not math.isfinite(args.marker_hold_seconds) or args.marker_hold_seconds < 0.0:
        parser.error("--marker-hold-seconds must be non-negative and finite")
    return args


def main() -> int:
    args = _parse_args()
    rclpy.init()
    node = CoordinateFrameDebugNode(args)
    try:
        print(
            f"Collecting {args.duration:g} seconds of perception/TF data. "
            "Diagnostic only: no MoveIt, controller, or trajectory interface is created."
        )
        start = time.monotonic()
        node.collect(args.duration)
        end = time.monotonic()
        mouth_samples = _sample_summary(node.samples, end - start, end)

        base_to_tool = node.transform(args.base_frame, args.tool_frame)
        base_to_camera = node.transform(args.base_frame, args.camera_frame)
        tool_to_camera = node.transform(args.tool_frame, args.camera_frame)
        pendant_from_internal = node.transform(args.pendant_base_frame, args.base_frame)
        if not base_to_tool.get("available") or not base_to_camera.get("available"):
            raise RuntimeError(
                "required TF unavailable: "
                f"tool0={base_to_tool.get('reason')} camera={base_to_camera.get('reason')}"
            )

        tool_position = base_to_tool["translation_m"]
        camera_position = base_to_camera["translation_m"]
        tool_axes = base_to_tool["axes_expressed_in_target"]
        camera_axes = base_to_camera["axes_expressed_in_target"]
        straw_tip = _add(
            tool_position,
            _rotate(base_to_tool["orientation_quat_xyzw"], STRAW_TIP_OFFSET_TOOL0_M),
        )
        mouth_mean = mouth_samples.get("mean_position_m")
        vectors: dict[str, Any] = {
            "camera_to_mouth_base_link_m": None,
            "straw_tip_to_mouth_base_link_m": None,
            "tool0_to_mouth_base_link_m": None,
        }
        alignments: dict[str, float] | None = None
        if mouth_mean is not None:
            vectors = {
                "camera_to_mouth_base_link_m": _subtract(mouth_mean, camera_position),
                "straw_tip_to_mouth_base_link_m": _subtract(mouth_mean, straw_tip),
                "tool0_to_mouth_base_link_m": _subtract(mouth_mean, tool_position),
            }
            straw_direction = _normalize(vectors["straw_tip_to_mouth_base_link_m"])
            if straw_direction is not None:
                alignments = {
                    "straw_to_mouth_vs_tool0_plus_x": _dot(straw_direction, tool_axes["+X"]),
                    "straw_to_mouth_vs_base_link_plus_x": _dot(straw_direction, (1.0, 0.0, 0.0)),
                    "straw_to_mouth_vs_camera_optical_plus_z": _dot(straw_direction, camera_axes["+Z"]),
                }

        camera_info_frame = None if node.latest_camera_info is None else node.latest_camera_info["frame_id"]
        depth_frame = None if node.latest_depth is None else node.latest_depth["frame_id"]
        rgb_frame = None if node.latest_rgb is None else node.latest_rgb["frame_id"]
        selected_runtime_frame = camera_info_frame or depth_frame or rgb_frame
        runtime_frames = {
            "rgb_topic": args.rgb_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "mouth_output_topic": args.mouth_topic,
            "rgb_header": node.latest_rgb,
            "depth_header": node.latest_depth,
            "camera_info": node.latest_camera_info,
            "perception_source_frame_precedence": "CameraInfo, then depth Image, then RGB Image",
            "perception_source_frame_used": selected_runtime_frame,
            "configured_camera_optical_frame": args.camera_frame,
            "perception_source_frame_matches_configured_camera_optical": selected_runtime_frame == args.camera_frame,
            "published_mouth_frame": args.base_frame,
            "latest_perception_status": node.latest_status,
            "latest_candidates_frame": None
            if node.latest_candidates is None
            else node.latest_candidates.get("frame_id"),
        }
        projection = _independent_projection_check(node, base_to_camera, mouth_mean)
        camera_translation_check: dict[str, Any]
        if tool_to_camera.get("available"):
            measured = tool_to_camera["translation_m"]
            error = _subtract(measured, SUPPLIED_CAMERA_CENTER_TOOL0_M)
            camera_translation_check = {
                "supplied_tool0_to_camera_center_m": list(SUPPLIED_CAMERA_CENTER_TOOL0_M),
                "measured_tool0_to_camera_optical_m": measured,
                "error_m": error,
                "error_norm_m": _norm(error),
            }
        else:
            camera_translation_check = {
                "supplied_tool0_to_camera_center_m": list(SUPPLIED_CAMERA_CENTER_TOOL0_M),
                "available": False,
                "reason": tool_to_camera.get("reason"),
            }

        geometry = {
            "base_axes": {"+X": [1.0, 0.0, 0.0], "+Y": [0.0, 1.0, 0.0], "+Z": [0.0, 0.0, 1.0]},
            "tool0_position_m": tool_position,
            "camera_center_m": camera_position,
            "straw_tip_m": straw_tip,
            "mouth_mean_m": mouth_mean,
            "tool0_axes_in_base_link": tool_axes,
            "camera_axes_in_base_link": camera_axes,
        }
        marker_count = node.publish_markers(geometry)
        verdict = _verdict(alignments, projection, runtime_frames)

        positions_internal = {
            "tool0": tool_position,
            "camera_optical_center": camera_position,
            "straw_tip": straw_tip,
            "detected_mouth_mean": mouth_mean,
        }
        positions_pendant = {
            name: None if position is None else _transform_point(pendant_from_internal, position)
            for name, position in positions_internal.items()
        }
        tool_axes_pendant = _transform_axes(pendant_from_internal, tool_axes)
        camera_axes_pendant = _transform_axes(pendant_from_internal, camera_axes)

        timestamp = datetime.now().astimezone()
        report = {
            "schema_version": 1,
            "diagnostic": "coordinate_frame_debug",
            "captured_at": timestamp.isoformat(timespec="milliseconds"),
            "success": bool(mouth_samples.get("stable")),
            "safety": {
                "diagnostic_only": True,
                "moveit_imported": False,
                "action_clients_created": False,
                "controller_services_called": False,
                "trajectory_commands_sent": False,
                "robot_motion_sent": False,
            },
            "runtime_frames": runtime_frames,
            "perception_source_audit": _perception_source_audit(),
            "tf": {
                "base_link_to_tool0": base_to_tool,
                "base_link_to_camera_optical": base_to_camera,
                "tool0_to_camera_optical": tool_to_camera,
                "pendant_base_from_internal_base": pendant_from_internal,
            },
            "coordinate_standardization": {
                "canonical_internal_frame": args.base_frame,
                "teach_pendant_frame": args.pendant_base_frame,
                "rule": "All perception/planning stays in base_link. Pendant readings are transformed into base_link before comparison; dual-frame output is display-only.",
                "observed_default_relation": "For UR base <-> base_link: x_base=-x_base_link, y_base=-y_base_link, z_base=z_base_link (180 degree Z rotation).",
            },
            "axes": {
                "base_link_axes": geometry["base_axes"],
                "tool0_axes_expressed_in_base_link": tool_axes,
                "camera_optical_axes_expressed_in_base_link": camera_axes,
                "tool0_axes_expressed_in_ur_base": tool_axes_pendant,
                "camera_optical_axes_expressed_in_ur_base": camera_axes_pendant,
                "realsense_optical_convention": {
                    "+X": "image right",
                    "+Y": "image down",
                    "+Z": "camera forward/depth",
                },
            },
            "positions_base_link_m": positions_internal,
            "positions_ur_base_m": positions_pendant,
            "known_geometry": {
                "straw_tip_offset_tool0_m": list(STRAW_TIP_OFFSET_TOOL0_M),
                "camera_center_translation_check": camera_translation_check,
            },
            "mouth_samples": mouth_samples,
            "rejected_wrong_frame_mouth_samples": node.rejected_wrong_frame,
            "vectors": vectors,
            "alignment_dot_products": alignments,
            "independent_projection_check": projection,
            "verdict": verdict,
            "markers": {
                "topic": args.marker_topic,
                "qos_durability": "transient_local",
                "published_marker_count": marker_count,
                "contents": [
                    "red mouth sphere",
                    "green straw-tip sphere",
                    "blue tool0 sphere",
                    "cyan camera-center sphere",
                    "base_link XYZ arrows",
                    "tool0 XYZ arrows",
                    "camera optical XYZ arrows",
                    "camera-to-mouth arrow",
                    "straw-tip-to-mouth arrow",
                    "coordinate/frame text labels",
                ],
            },
        }
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            args.report_dir / f"coordinate_frame_debug_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        ).resolve()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_report(report, report_path)
        return 0 if report["success"] else 2
    except Exception as exc:
        print(f"Coordinate-frame diagnostic failed: {exc}")
        print("No robot motion was sent.")
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
