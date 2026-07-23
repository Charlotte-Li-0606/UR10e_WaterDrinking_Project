#!/usr/bin/env python3
"""Publish plan-only RViz diagnostics for perception-derived pre-mouth targets.

This script never imports MoveIt actions and never sends robot commands.  It
compares dynamic TCP-forward, legacy fixed base-X, camera-ray, and explicit
feeding-vector offsets, publishes the geometry as
``visualization_msgs/MarkerArray``, and prints the exact numeric values used
for review.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
CAMERA_FRAME = "d435i_color_optical_frame"
MOUTH_TOPIC = "/detected_mouth_pose"
MARKER_TOPIC = "/premouth_accuracy_diagnostics"
STRAW_TIP_OFFSET_TOOL0_M = (0.110, 0.0, 0.0)
MIN_SAMPLES = 3
MAX_MOUTH_SPREAD_M = 0.025
# The real axis sanity test showed that base +X is sideways in the current
# camera/user arrangement.  The safe side of the mouth is base -Y, so the
# default ``plus`` feeding-vector candidate is mouth + [0, -distance, 0].
DEFAULT_FEEDING_VECTOR = (0.0, -1.0, 0.0)
FEEDING_VECTOR_SIGNS = ("plus", "minus")
MIN_SAFE_DISTANCE_M = 0.030
MAX_SAFE_DISTANCE_M = 0.080
MAX_ABS_FEEDING_VECTOR_Z = 0.30
TF_DISCOVERY_TIMEOUT_SEC = 8.0


def _add(first: list[float], second: list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(first, second)]


def _subtract(first: list[float], second: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(first, second)]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _normalize_feeding_vector(vector: tuple[float, float, float]) -> list[float]:
    if not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("feeding vector components must be finite")
    magnitude = _norm([float(value) for value in vector])
    if magnitude < 1e-6:
        raise ValueError("feeding vector norm is too small to define an approach direction")
    return [float(value) / magnitude for value in vector]


def _rotate(quaternion: list[float], vector: tuple[float, float, float]) -> list[float]:
    x, y, z, w = quaternion
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if magnitude < 1e-12:
        raise RuntimeError("tool0 orientation has zero length")
    x, y, z, w = (value / magnitude for value in (x, y, z, w))
    vx, vy, vz = vector
    return [
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ]


@dataclass(frozen=True)
class MouthSample:
    position: tuple[float, float, float]
    monotonic_time: float


class PreMouthAccuracyDiagnostic(Node):
    def __init__(self) -> None:
        super().__init__("diagnose_premouth_accuracy")
        self.samples: list[MouthSample] = []
        self.create_subscription(PoseStamped, MOUTH_TOPIC, self._mouth_callback, 20)
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(MarkerArray, MARKER_TOPIC, marker_qos)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

    def _mouth_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id.strip().lstrip("/") != BASE_FRAME:
            return
        point = message.pose.position
        values = (float(point.x), float(point.y), float(point.z))
        if all(math.isfinite(value) for value in values):
            self.samples.append(MouthSample(values, time.monotonic()))
            if len(self.samples) > 512:
                del self.samples[:-256]

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, deadline - time.monotonic()))

    def stable_mouth(self, seconds: float) -> dict[str, Any]:
        start = time.monotonic()
        self.spin_for(seconds)
        points = [sample.position for sample in self.samples if sample.monotonic_time >= start]
        if len(points) < MIN_SAMPLES:
            raise RuntimeError(f"received only {len(points)} mouth samples; need at least {MIN_SAMPLES}")
        mean = [sum(values) / len(values) for values in zip(*points)]
        spread = max(_norm(_subtract(list(point), mean)) for point in points)
        if spread > MAX_MOUTH_SPREAD_M:
            raise RuntimeError(f"mouth pose is unstable: max spread {spread:.4f} m exceeds {MAX_MOUTH_SPREAD_M:.4f} m")
        return {"position_m": mean, "sample_count": len(points), "max_spread_m": spread, "sample_seconds": seconds}

    def transform(self, source: str, timeout_sec: float = TF_DISCOVERY_TIMEOUT_SEC) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(BASE_FRAME, source, rclpy.time.Time(), timeout=Duration(seconds=0.0))
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                return {
                    "position_m": [float(translation.x), float(translation.y), float(translation.z)],
                    "orientation_quat_xyzw": [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)],
                }
            except Exception as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f"TF {BASE_FRAME} -> {source} unavailable: {last_error}")

    @staticmethod
    def _point(position: list[float]) -> Point:
        point = Point()
        point.x, point.y, point.z = position
        return point

    def _sphere(self, marker_id: int, position: list[float], color: tuple[float, float, float], label: str) -> list[Marker]:
        sphere = Marker()
        sphere.header.frame_id = BASE_FRAME
        sphere.ns = "premouth_accuracy"
        sphere.id = marker_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = self._point(position)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.030
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (*color, 0.95)

        text = Marker()
        text.header.frame_id = BASE_FRAME
        text.ns = "premouth_accuracy_labels"
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = self._point(_add(position, [0.0, 0.0, 0.035]))
        text.pose.orientation.w = 1.0
        text.scale.z = 0.028
        text.color.r, text.color.g, text.color.b, text.color.a = (*color, 1.0)
        text.text = label
        return [sphere, text]

    @staticmethod
    def _coordinate_label(name: str, position: list[float]) -> str:
        return f"{name} [{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}]"

    def _arrow(self, marker_id: int, start: list[float], end: list[float], color: tuple[float, float, float], label: str) -> list[Marker]:
        arrow = Marker()
        arrow.header.frame_id = BASE_FRAME
        arrow.ns = "premouth_accuracy_vectors"
        arrow.id = marker_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.008, 0.018, 0.025
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (*color, 0.9)
        arrow.points = [self._point(start), self._point(end)]

        midpoint = [(a + b) * 0.5 for a, b in zip(start, end)]
        text = Marker()
        text.header.frame_id = BASE_FRAME
        text.ns = "premouth_accuracy_vector_labels"
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = self._point(_add(midpoint, [0.0, 0.0, 0.025]))
        text.pose.orientation.w = 1.0
        text.scale.z = 0.025
        text.color.r, text.color.g, text.color.b, text.color.a = (*color, 1.0)
        text.text = label
        return [arrow, text]

    def markers(self, values: dict[str, Any], selected: str) -> MarkerArray:
        mouth = values["detected_mouth_m"]
        camera = values["camera_position_m"]
        old = values["old_base_x_pre_mouth_m"]
        ray = values["camera_ray_pre_mouth_m"]
        tcp_forward = values["tcp_forward_pre_mouth_m"]
        feeding_plus = values["feeding_vector_plus_pre_mouth_m"]
        feeding_minus = values["feeding_vector_minus_pre_mouth_m"]
        straw = values["current_straw_tip_m"]
        tool = values["current_tool0_pose"]["position_m"]
        tool_x_axis = values["tool_x_axis_base"]
        selected_target = {
            "base-x": old,
            "camera-ray": ray,
            "tcp-forward": tcp_forward,
            "feeding-vector-plus": feeding_plus,
            "feeding-vector-minus": feeding_minus,
        }[selected]
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        array.markers.extend(self._sphere(1, mouth, (1.0, 0.0, 0.0), self._coordinate_label("detected mouth (red)", mouth)))
        array.markers.extend(self._sphere(2, old, (1.0, 1.0, 0.0), self._coordinate_label("old base-X (yellow)", old)))
        array.markers.extend(self._sphere(3, ray, (0.0, 1.0, 1.0), self._coordinate_label("camera-ray (cyan)", ray)))
        array.markers.extend(self._sphere(4, feeding_plus, (0.65, 0.10, 0.85), self._coordinate_label("feeding-vector plus (purple)", feeding_plus)))
        array.markers.extend(self._sphere(5, feeding_minus, (0.40, 0.05, 0.60), self._coordinate_label("feeding-vector minus (dark purple)", feeding_minus)))
        array.markers.extend(self._sphere(6, straw, (0.0, 1.0, 0.0), self._coordinate_label("current straw tip (green)", straw)))
        array.markers.extend(self._sphere(7, selected_target, (0.0, 0.3, 1.0), self._coordinate_label(f"selected final straw: {selected} (blue)", selected_target)))
        array.markers.extend(self._sphere(8, tcp_forward, (1.0, 0.45, 0.0), self._coordinate_label("tcp-forward pre-mouth (orange)", tcp_forward)))
        array.markers.extend(self._arrow(11, camera, mouth, (1.0, 1.0, 1.0), "camera -> mouth"))
        array.markers.extend(self._arrow(13, mouth, ray, (0.0, 1.0, 1.0), "mouth -> camera-ray"))
        array.markers.extend(self._arrow(14, mouth, feeding_plus, (0.65, 0.10, 0.85), "mouth -> feeding-vector plus"))
        array.markers.extend(self._arrow(15, mouth, feeding_minus, (0.40, 0.05, 0.60), "mouth -> feeding-vector minus"))
        array.markers.extend(self._arrow(16, tool, _add(tool, [0.18 * value for value in tool_x_axis]), (1.0, 0.45, 0.0), "tool0/TCP +X in base_link"))
        array.markers.extend(self._arrow(17, mouth, tcp_forward, (1.0, 0.45, 0.0), "mouth -> tcp-forward pre-mouth"))
        return array


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to keep publishing RViz markers after sampling.")
    parser.add_argument(
        "--mode",
        choices=("tcp-forward", "camera-ray", "base-x", "feeding-vector"),
        default="camera-ray",
    )
    parser.add_argument("--safe-distance", type=float, default=0.05)
    parser.add_argument(
        "--feeding-vector-x",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[0],
        help="Base-link X component of the feeding vector (real default: 0).",
    )
    parser.add_argument(
        "--feeding-vector-y",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[1],
        help="Base-link Y component of the feeding vector (real default: -1).",
    )
    parser.add_argument(
        "--feeding-vector-z",
        type=float,
        default=DEFAULT_FEEDING_VECTOR[2],
        help="Base-link Z component of the feeding vector (real default: 0).",
    )
    parser.add_argument("--feeding-vector-sign", choices=FEEDING_VECTOR_SIGNS, default="plus")
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Accepted explicitly for safety; this diagnostic never executes motion regardless.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Publish synthetic markers near the live tool0 pose to verify the RViz display; no perception or motion is used.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")
    if not math.isfinite(args.safe_distance) or not MIN_SAFE_DISTANCE_M <= args.safe_distance <= MAX_SAFE_DISTANCE_M:
        parser.error(f"--safe-distance must be within {MIN_SAFE_DISTANCE_M:.2f}–{MAX_SAFE_DISTANCE_M:.2f} m")
    try:
        normalized_feeding_vector = _normalize_feeding_vector(
            (args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if abs(normalized_feeding_vector[2]) > MAX_ABS_FEEDING_VECTOR_Z:
        parser.error(f"abs(normalized feeding-vector-z) must be <= {MAX_ABS_FEEDING_VECTOR_Z:.2f}")
    args.normalized_feeding_vector = normalized_feeding_vector
    return args


def main() -> int:
    args = _parse_args()
    if args.mode == "tcp-forward":
        print(f"Using tcp-forward policy: safe_distance={args.safe_distance:g}", flush=True)
    if args.mode == "feeding-vector":
        vector_text = ", ".join(f"{component:g}" for component in args.normalized_feeding_vector)
        print(
            f"Using feeding-vector policy: [{vector_text}], safe_distance={args.safe_distance:g}",
            flush=True,
        )
    rclpy.init()
    node = PreMouthAccuracyDiagnostic()
    try:
        tool = node.transform(TOOL_FRAME)
        camera = node.transform(CAMERA_FRAME)
        straw = _add(tool["position_m"], _rotate(tool["orientation_quat_xyzw"], STRAW_TIP_OFFSET_TOOL0_M))
        tool_x_axis_base = _normalize_feeding_vector(
            tuple(_rotate(tool["orientation_quat_xyzw"], (1.0, 0.0, 0.0)))
        )
        if args.demo:
            mouth = {
                "position_m": _add(straw, [0.0, 0.15, 0.05]),
                "sample_count": 0,
                "sample_seconds": 0.0,
                "max_spread_m": 0.0,
                "synthetic": True,
            }
        else:
            mouth = node.stable_mouth(seconds=3.0)
        mouth_position = mouth["position_m"]
        camera_to_mouth = _subtract(mouth_position, camera["position_m"])
        camera_distance = _norm(camera_to_mouth)
        if camera_distance < 1e-6:
            raise RuntimeError("camera and mouth positions are coincident")
        unit_ray = [value / camera_distance for value in camera_to_mouth]
        old = _add(mouth_position, [args.safe_distance, 0.0, 0.0])
        ray = _subtract(mouth_position, [args.safe_distance * value for value in unit_ray])
        ray_alternative = _add(mouth_position, [args.safe_distance * value for value in unit_ray])
        tcp_forward = _subtract(
            mouth_position,
            [args.safe_distance * value for value in tool_x_axis_base],
        )
        feeding_plus = _add(mouth_position, [args.safe_distance * value for value in args.normalized_feeding_vector])
        feeding_minus = _subtract(mouth_position, [args.safe_distance * value for value in args.normalized_feeding_vector])
        selected = {
            "base-x": old,
            "camera-ray": ray,
            "tcp-forward": tcp_forward,
            "feeding-vector": feeding_plus if args.feeding_vector_sign == "plus" else feeding_minus,
        }[args.mode]
        selected_name = (
            f"feeding-vector-{args.feeding_vector_sign}" if args.mode == "feeding-vector" else args.mode
        )
        values = {
            "success": True,
            "mode": args.mode,
            "safe_distance_m": args.safe_distance,
            "feeding_vector_input": [args.feeding_vector_x, args.feeding_vector_y, args.feeding_vector_z],
            "feeding_vector_normalized": args.normalized_feeding_vector,
            "feeding_vector_sign": args.feeding_vector_sign,
            "marker_topic": MARKER_TOPIC,
            "mouth_sampling": mouth,
            "current_tool0_pose": tool,
            "tool_x_axis_base": tool_x_axis_base,
            "camera_position_m": camera["position_m"],
            "detected_mouth_m": mouth_position,
            "current_straw_tip_m": straw,
            "old_base_x_pre_mouth_m": old,
            "camera_ray_pre_mouth_m": ray,
            "camera_ray_alternative_m": ray_alternative,
            "tcp_forward_pre_mouth_m": tcp_forward,
            "feeding_vector_plus_pre_mouth_m": feeding_plus,
            "feeding_vector_minus_pre_mouth_m": feeding_minus,
            "selected_pre_mouth_m": selected,
            "selected_policy": selected_name,
            "camera_to_mouth_vector_m": camera_to_mouth,
            "camera_to_mouth_unit_vector": unit_ray,
            "mouth_to_old_pre_mouth_vector_m": _subtract(old, mouth_position),
            "mouth_to_camera_ray_pre_mouth_vector_m": _subtract(ray, mouth_position),
            "mouth_to_tcp_forward_pre_mouth_vector_m": _subtract(tcp_forward, mouth_position),
            "mouth_to_feeding_vector_plus_pre_mouth_vector_m": _subtract(feeding_plus, mouth_position),
            "mouth_to_feeding_vector_minus_pre_mouth_vector_m": _subtract(feeding_minus, mouth_position),
            "current_straw_to_old_pre_mouth_distance_m": _norm(_subtract(straw, old)),
            "current_straw_to_camera_ray_pre_mouth_distance_m": _norm(_subtract(straw, ray)),
            "current_straw_to_tcp_forward_pre_mouth_distance_m": _norm(_subtract(straw, tcp_forward)),
            "current_straw_to_feeding_vector_plus_pre_mouth_distance_m": _norm(_subtract(straw, feeding_plus)),
            "current_straw_to_feeding_vector_minus_pre_mouth_distance_m": _norm(_subtract(straw, feeding_minus)),
            "execution_sent": False,
            "execution_disabled": True,
            "no_execute": bool(args.no_execute),
            "note": "Diagnostic only: no MoveIt or trajectory action is created by this script.",
        }
        if args.mode == "tcp-forward":
            print(f"tool_x_axis_base: {tool_x_axis_base}", flush=True)
            print(f"safe_distance: {args.safe_distance:g}", flush=True)
            print(
                f"pre_mouth - mouth: {_subtract(tcp_forward, mouth_position)}",
                flush=True,
            )
        marker_array = node.markers(values, selected_name)
        print(json.dumps(values, indent=2, sort_keys=True))
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            node.publisher.publish(marker_array)
            rclpy.spin_once(node, timeout_sec=min(0.2, deadline - time.monotonic()))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "reason": f"{exc.__class__.__name__}: {exc}", "execution_sent": False}, indent=2))
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
