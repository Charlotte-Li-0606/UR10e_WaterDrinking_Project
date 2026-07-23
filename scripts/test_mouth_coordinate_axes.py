#!/usr/bin/env python3
"""No-motion sanity test for mouth coordinates and wrist-camera TF axes.

Acquisition mode subscribes only to ``/detected_mouth_pose``, reads TF, writes
one JSON report, and publishes diagnostic RViz markers.  It does not import
MoveIt, create action clients, call controller services, or publish any robot
command.

Comparison mode is ROS-independent.  It compares labelled acquisition reports
against a ``center`` report and checks whether physical X/Y/Z movements appear
on the expected ``base_link`` axes and with the expected signs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

BASE_FRAME = "base_link"
PENDANT_BASE_FRAME = "base"
TOOL_FRAME = "tool0"
CAMERA_FRAME = "d435i_color_optical_frame"
MOUTH_TOPIC = "/detected_mouth_pose"
MARKER_TOPIC = "/mouth_coordinate_axes_debug"

MIN_SAMPLES = 3
MAX_STABLE_RADIAL_SPREAD_M = 0.025
MIN_MEANINGFUL_COMPARISON_DELTA_M = 0.020

LABELS = (
    "center",
    "user_moved_forward",
    "user_moved_backward",
    "user_moved_robot_right",
    "user_moved_robot_left",
    "user_moved_up",
    "user_moved_down",
)

COMPARISON_ORDER = (
    "user_moved_forward",
    "user_moved_backward",
    "user_moved_robot_right",
    "user_moved_robot_left",
    "user_moved_up",
    "user_moved_down",
)

EXPECTED_DELTAS: dict[str, tuple[str, int]] = {
    "user_moved_forward": ("x", 1),
    "user_moved_backward": ("x", -1),
    "user_moved_robot_right": ("y", 1),
    "user_moved_robot_left": ("y", -1),
    "user_moved_up": ("z", 1),
    "user_moved_down": ("z", -1),
}

COORDINATE_CONVENTION = {
    "+X": "robot forward",
    "-X": "robot backward",
    "+Y": "robot right",
    "-Y": "robot left",
    "+Z": "upward",
    "-Z": "downward",
}

KNOWN_TOOL_GEOMETRY = {
    "tool0_to_camera_optical_center_translation_m": [0.070, 0.0, 0.015],
    "tool0_to_straw_tip_translation_m": [0.110, 0.0, 0.0],
    "note": "Reference physical geometry supplied for this diagnostic; live TF is recorded separately.",
}


def _xyz(values: Sequence[float]) -> dict[str, float]:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def _xyz_list(value: Any, *, field: str) -> list[float]:
    if isinstance(value, dict):
        try:
            result = [float(value[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite x/y/z values") from exc
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            result = [float(component) for component in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite x/y/z values") from exc
    else:
        raise ValueError(f"{field} must be a three-element list or x/y/z object")
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field} must contain finite x/y/z values")
    return result


def _subtract(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(first, second)]


def _add(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(first, second)]


def _rotate(quaternion_xyzw: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(magnitude) or magnitude < 1e-12:
        raise ValueError("quaternion has zero or invalid norm")
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
    if not transform.get("available"):
        return None
    return _add(
        _rotate(transform["orientation_quat_xyzw"], point),
        transform["translation_m"],
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in vector))


def _timestamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y%m%d_%H%M%S"), now.isoformat(timespec="milliseconds")


def _mean(points: Sequence[Sequence[float]]) -> list[float]:
    return [sum(float(point[index]) for point in points) / len(points) for index in range(3)]


def _summarize_points(
    points: Sequence[Sequence[float]],
    *,
    duration_sec: float,
    latest_pose_age_sec: float | None,
    latest_receive_age_sec: float | None,
) -> dict[str, Any]:
    if not points:
        return {
            "count": 0,
            "duration_sec": float(duration_sec),
            "stable": False,
            "failure": "no valid base_link mouth poses were received",
            "latest_pose_age_sec": latest_pose_age_sec,
            "latest_receive_age_sec": latest_receive_age_sec,
        }

    mean = _mean(points)
    minima = [min(float(point[index]) for point in points) for index in range(3)]
    maxima = [max(float(point[index]) for point in points) for index in range(3)]
    spreads = [maximum - minimum for minimum, maximum in zip(minima, maxima)]
    stddev = [
        math.sqrt(sum((float(point[index]) - mean[index]) ** 2 for point in points) / len(points))
        for index in range(3)
    ]
    distances = [_norm(_subtract(point, mean)) for point in points]
    max_radial_spread = max(distances)
    stable = len(points) >= MIN_SAMPLES and max_radial_spread <= MAX_STABLE_RADIAL_SPREAD_M
    failure: str | None = None
    if len(points) < MIN_SAMPLES:
        failure = f"received {len(points)} samples; need at least {MIN_SAMPLES}"
    elif max_radial_spread > MAX_STABLE_RADIAL_SPREAD_M:
        failure = (
            f"maximum distance from mean is {max_radial_spread:.6f} m, over "
            f"{MAX_STABLE_RADIAL_SPREAD_M:.6f} m"
        )

    return {
        "count": len(points),
        "duration_sec": float(duration_sec),
        "mean_position_m": [float(value) for value in mean],
        "mean_m": _xyz(mean),
        "stddev_m": _xyz(stddev),
        "spread_m": _xyz(spreads),
        "min_m": _xyz(minima),
        "max_m": _xyz(maxima),
        "max_distance_from_mean_m": float(max_radial_spread),
        "latest_pose_age_sec": latest_pose_age_sec,
        "latest_receive_age_sec": latest_receive_age_sec,
        "stable": stable,
        "failure": failure,
        "stability_requirements": {
            "minimum_samples": MIN_SAMPLES,
            "maximum_distance_from_mean_m": MAX_STABLE_RADIAL_SPREAD_M,
        },
    }


def _extract_mean(report: dict[str, Any], *, path: Path) -> list[float]:
    samples = report.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"{path}: missing samples object")
    if "mean_position_m" in samples:
        return _xyz_list(samples["mean_position_m"], field=f"{path}: samples.mean_position_m")
    if "mean_m" in samples:
        return _xyz_list(samples["mean_m"], field=f"{path}: samples.mean_m")
    raise ValueError(f"{path}: missing mouth mean")


def _expand_report_paths(arguments: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        matches = sorted(glob.glob(argument))
        if not matches and Path(argument).is_file():
            matches = [argument]
        if not matches:
            raise ValueError(f"no report matched: {argument}")
        paths.extend(Path(match).resolve() for match in matches if Path(match).is_file())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_comparison_reports(arguments: Iterable[str]) -> dict[str, tuple[Path, dict[str, Any]]]:
    labelled: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in _expand_report_paths(arguments):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        if report.get("diagnostic") != "mouth_coordinate_axes":
            raise ValueError(f"{path}: not a mouth_coordinate_axes report")
        label = report.get("label")
        if label not in LABELS:
            raise ValueError(f"{path}: unsupported or missing label {label!r}")
        _extract_mean(report, path=path)
        # With shell globs, using the newest report for a repeated label is
        # more useful than failing because an earlier trial also matched.
        previous = labelled.get(label)
        if previous is None or path.stat().st_mtime >= previous[0].stat().st_mtime:
            labelled[label] = (path, report)
    if "center" not in labelled:
        raise ValueError("comparison requires one report labelled 'center'")
    return labelled


def _comparison_row(label: str, delta: Sequence[float]) -> dict[str, Any]:
    axes = ("x", "y", "z")
    magnitude = _norm(delta)
    dominant_index = max(range(3), key=lambda index: abs(float(delta[index])))
    dominant_axis = axes[dominant_index]
    dominant_value = float(delta[dominant_index])
    dominant_sign = 1 if dominant_value > 0.0 else -1 if dominant_value < 0.0 else 0
    expected_axis, expected_sign = EXPECTED_DELTAS[label]
    meaningful = magnitude >= MIN_MEANINGFUL_COMPARISON_DELTA_M
    axis_matches = dominant_axis == expected_axis
    sign_matches = dominant_sign == expected_sign
    if not meaningful:
        result = "INCONCLUSIVE: displacement too small"
    elif not axis_matches:
        result = "MISMATCH: dominant axis"
    elif not sign_matches:
        result = "MISMATCH: sign"
    else:
        result = "MATCH"
    expected_text = ("+" if expected_sign > 0 else "-") + expected_axis.upper()
    dominant_text = ("+" if dominant_sign > 0 else "-" if dominant_sign < 0 else "") + dominant_axis.upper()
    return {
        "label": label,
        "delta_m": [float(value) for value in delta],
        "delta_xyz_m": _xyz(delta),
        "norm_m": magnitude,
        "dominant_axis": dominant_axis,
        "dominant_sign": dominant_sign,
        "dominant": dominant_text,
        "expected_axis": expected_axis,
        "expected_sign": expected_sign,
        "expected": expected_text,
        "meaningful": meaningful,
        "axis_matches": axis_matches,
        "sign_matches": sign_matches,
        "result": result,
    }


def compare_reports(arguments: Iterable[str]) -> dict[str, Any]:
    labelled = _load_comparison_reports(arguments)
    center_path, center_report = labelled["center"]
    center = _extract_mean(center_report, path=center_path)
    center_stable = bool(center_report.get("samples", {}).get("stable"))
    rows: list[dict[str, Any]] = []
    for label in COMPARISON_ORDER:
        item = labelled.get(label)
        if item is None:
            continue
        path, report = item
        row = _comparison_row(label, _subtract(_extract_mean(report, path=path), center))
        row["report_path"] = str(path)
        samples = report.get("samples", {})
        row["capture_stable"] = center_stable and bool(samples.get("stable"))
        if not row["capture_stable"]:
            row["result"] = "INCONCLUSIVE: acquisition unstable"
        rows.append(row)

    mismatches = [
        row
        for row in rows
        if row["capture_stable"] and row["meaningful"] and row["result"] != "MATCH"
    ]
    unstable = [row for row in rows if not row["capture_stable"]]
    too_small = [row for row in rows if row["capture_stable"] and not row["meaningful"]]
    positive_axis_labels = {"user_moved_forward", "user_moved_robot_right", "user_moved_up"}
    positive_rows = {row["label"]: row for row in rows if row["label"] in positive_axis_labels}
    full_positive_coverage = positive_axis_labels.issubset(positive_rows)
    if not center_stable:
        verdict = "Inconclusive: the selected center acquisition is unstable; collect a new center report."
    elif mismatches:
        verdict = (
            "TF/camera extrinsic or declared base_link-to-physical-axis convention issue suspected: "
            "at least one physical movement maps to the wrong base_link axis or sign."
        )
    elif unstable:
        verdict = "Inconclusive: at least one center or movement acquisition is unstable; repeat that paired run."
    elif too_small:
        verdict = "Inconclusive: at least one movement was below the 2 cm meaningful-delta threshold; repeat that run."
    elif full_positive_coverage and all(row["result"] == "MATCH" for row in positive_rows.values()):
        verdict = (
            "The tested base_link X/Y/Z mappings and signs are consistent. If pre-mouth motion still shifts sideways, "
            "the pre-mouth direction policy is more likely wrong than TF."
        )
    elif rows:
        verdict = "Available axis tests match, but forward/right/up coverage is incomplete; collect the missing reports before choosing TF vs policy."
    else:
        verdict = "Inconclusive: only the center report was provided."

    result = {
        "center_report_path": str(center_path),
        "center_mean_position_m": center,
        "center_stable": center_stable,
        "reports_used": {label: str(path) for label, (path, _) in labelled.items()},
        "comparisons": rows,
        "verdict": verdict,
        "motion_commands_sent": False,
    }
    _print_comparison(result)
    return result


def _print_comparison(result: dict[str, Any]) -> None:
    center = result["center_mean_position_m"]
    print("Mouth-coordinate axis comparison (no ROS, no motion)")
    print(f"Center mean: x={center[0]:+.6f}  y={center[1]:+.6f}  z={center[2]:+.6f} m")
    print()
    print(f"{'movement - center':<28} {'delta X':>10} {'delta Y':>10} {'delta Z':>10} {'dominant':>10} {'expected':>10}  result")
    print("-" * 108)
    for row in result["comparisons"]:
        delta = row["delta_m"]
        print(
            f"{row['label']:<28} {delta[0]:+10.4f} {delta[1]:+10.4f} {delta[2]:+10.4f} "
            f"{row['dominant']:>10} {row['expected']:>10}  {row['result']}"
        )
    if not result["comparisons"]:
        print("(no movement reports supplied)")
    print()
    print(f"Verdict: {result['verdict']}")
    print("No robot motion was sent.")


@dataclass(frozen=True)
class MouthSample:
    position_m: tuple[float, float, float]
    received_monotonic: float
    source_time_sec: float | None


def acquire_report(args: argparse.Namespace) -> tuple[int, Path]:
    # ROS imports intentionally live only in acquisition mode.  Comparison
    # mode can run on an analysis PC without a sourced ROS environment.
    import rclpy
    import rclpy.time
    import tf2_ros
    from geometry_msgs.msg import Point, PoseStamped
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from visualization_msgs.msg import Marker, MarkerArray

    class MouthCoordinateAxesNode(Node):
        def __init__(self) -> None:
            super().__init__("test_mouth_coordinate_axes")
            self.samples: list[MouthSample] = []
            self.rejected_wrong_frame = 0
            self.rejected_non_finite = 0
            self.collecting = False
            sample_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            marker_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(PoseStamped, args.mouth_topic, self._mouth_callback, sample_qos)
            self.marker_publisher = self.create_publisher(MarkerArray, args.marker_topic, marker_qos)
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

        def _mouth_callback(self, message: PoseStamped) -> None:
            if not self.collecting:
                return
            frame = message.header.frame_id.strip().lstrip("/")
            if frame != args.base_frame.strip().lstrip("/"):
                self.rejected_wrong_frame += 1
                return
            point = message.pose.position
            position = (float(point.x), float(point.y), float(point.z))
            if not all(math.isfinite(value) for value in position):
                self.rejected_non_finite += 1
                return
            stamp = message.header.stamp
            source_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if source_time <= 0.0:
                source_time = None
            self.samples.append(MouthSample(position, time.monotonic(), source_time))

        def collect(self, seconds: float) -> None:
            self.samples.clear()
            self.collecting = True
            deadline = time.monotonic() + seconds
            try:
                while time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            finally:
                self.collecting = False

        def transform(self, target_frame: str, source_frame: str, timeout_sec: float = 2.0) -> dict[str, Any]:
            deadline = time.monotonic() + timeout_sec
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
                    stamp = transform.header.stamp
                    return {
                        "available": True,
                        "target_frame": target_frame,
                        "source_frame": source_frame,
                        "interpretation": f"pose of {source_frame} expressed in {target_frame}",
                        "translation_m": [float(translation.x), float(translation.y), float(translation.z)],
                        "orientation_quat_xyzw": [
                            float(rotation.x),
                            float(rotation.y),
                            float(rotation.z),
                            float(rotation.w),
                        ],
                        "stamp_sec": float(stamp.sec) + float(stamp.nanosec) * 1e-9,
                    }
                except Exception as exc:  # tf2 exception subclasses vary by ROS distribution.
                    last_error = exc
                    rclpy.spin_once(self, timeout_sec=0.05)
            return {
                "available": False,
                "target_frame": target_frame,
                "source_frame": source_frame,
                "interpretation": f"pose of {source_frame} expressed in {target_frame}",
                "reason": str(last_error) if last_error is not None else "TF lookup timed out",
            }

        @staticmethod
        def point(position: Sequence[float]) -> Point:
            point = Point()
            point.x, point.y, point.z = (float(value) for value in position)
            return point

        def publish_markers(self, mouth: Sequence[float], camera: Sequence[float] | None) -> int:
            now = self.get_clock().now().to_msg()
            markers = MarkerArray()
            clear = Marker()
            clear.header.frame_id = args.base_frame
            clear.header.stamp = now
            clear.action = Marker.DELETEALL
            markers.markers.append(clear)

            sphere = Marker()
            sphere.header.frame_id = args.base_frame
            sphere.header.stamp = now
            sphere.ns = "mouth_coordinate_axes"
            sphere.id = 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = self.point(mouth)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.035
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (1.0, 0.0, 0.0, 0.95)
            markers.markers.append(sphere)

            text = Marker()
            text.header.frame_id = args.base_frame
            text.header.stamp = now
            text.ns = "mouth_coordinate_axes_labels"
            text.id = 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = self.point([mouth[0], mouth[1], mouth[2] + 0.045])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.030
            text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 0.9, 0.9, 1.0)
            text.text = f"mouth {args.label}: x={mouth[0]:+.3f} y={mouth[1]:+.3f} z={mouth[2]:+.3f} m"
            markers.markers.append(text)

            if camera is not None:
                arrow = Marker()
                arrow.header.frame_id = args.base_frame
                arrow.header.stamp = now
                arrow.ns = "mouth_coordinate_axes_vectors"
                arrow.id = 3
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [self.point(camera), self.point(mouth)]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = (0.010, 0.022, 0.030)
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (1.0, 0.75, 0.0, 0.95)
                markers.markers.append(arrow)

            self.marker_publisher.publish(markers)
            # Let DDS announce the transient-local writer and transmit its
            # retained sample. RViz opened during this hold receives the last
            # marker set even though it subscribed after publish().
            deadline = time.monotonic() + args.marker_hold_seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            return len(markers.markers) - 1  # Exclude DELETEALL from the useful marker count.

    rclpy.init()
    node = MouthCoordinateAxesNode()
    report_path: Path | None = None
    try:
        print(
            f"Collecting {args.duration:.1f} s of {args.mouth_topic} for label {args.label!r}. "
            "Diagnostic-only: no robot command interfaces are created."
        )
        collection_start_monotonic = time.monotonic()
        node.collect(args.duration)
        collection_end_monotonic = time.monotonic()

        latest_receive_age: float | None = None
        latest_source_age: float | None = None
        if node.samples:
            latest = node.samples[-1]
            latest_receive_age = max(0.0, collection_end_monotonic - latest.received_monotonic)
            if latest.source_time_sec is not None:
                candidate = node.get_clock().now().nanoseconds * 1e-9 - latest.source_time_sec
                if math.isfinite(candidate) and -0.5 <= candidate <= 1_000_000.0:
                    latest_source_age = max(0.0, candidate)
        latest_pose_age = latest_source_age if latest_source_age is not None else latest_receive_age
        points = [list(sample.position_m) for sample in node.samples]
        summary = _summarize_points(
            points,
            duration_sec=collection_end_monotonic - collection_start_monotonic,
            latest_pose_age_sec=latest_pose_age,
            latest_receive_age_sec=latest_receive_age,
        )

        tf_values = {
            "base_link_to_tool0": node.transform(args.base_frame, args.tool_frame),
            "base_link_to_camera_optical": node.transform(args.base_frame, args.camera_frame),
            "tool0_to_camera_optical": node.transform(args.tool_frame, args.camera_frame),
            "ur_base_from_base_link": node.transform(args.pendant_base_frame, args.base_frame),
            "ur_base_to_tool0": node.transform(args.pendant_base_frame, args.tool_frame),
            "ur_base_to_camera_optical": node.transform(args.pendant_base_frame, args.camera_frame),
        }
        camera_to_mouth: list[float] | None = None
        camera_tf = tf_values["base_link_to_camera_optical"]
        if summary.get("mean_position_m") is not None and camera_tf.get("available"):
            camera_to_mouth = _subtract(summary["mean_position_m"], camera_tf["translation_m"])
        mouth_mean_ur_base = None
        if summary.get("mean_position_m") is not None:
            mouth_mean_ur_base = _transform_point(
                tf_values["ur_base_from_base_link"], summary["mean_position_m"]
            )

        marker_count = 0
        if summary.get("mean_position_m") is not None:
            camera_position = camera_tf.get("translation_m") if camera_tf.get("available") else None
            marker_count = node.publish_markers(summary["mean_position_m"], camera_position)

        filename_timestamp, captured_at = _timestamp()
        report = {
            "schema_version": 1,
            "diagnostic": "mouth_coordinate_axes",
            "success": bool(summary.get("stable")),
            "label": args.label,
            "captured_at": captured_at,
            "duration_requested_sec": float(args.duration),
            "safety": {
                "perception_tf_diagnostic_only": True,
                "moveit_imported": False,
                "action_clients_created": False,
                "controller_services_called": False,
                "trajectory_commands_sent": False,
                "robot_motion_sent": False,
            },
            "topics": {"mouth_pose": args.mouth_topic, "markers": args.marker_topic},
            "frames": {
                "canonical_internal_base": args.base_frame,
                "teach_pendant_base": args.pendant_base_frame,
                "tool": args.tool_frame,
                "camera_optical": args.camera_frame,
            },
            "coordinate_standardization": {
                "canonical_internal_frame": args.base_frame,
                "teach_pendant_frame": args.pendant_base_frame,
                "rule": "Perception and planning remain in base_link; pendant readings are transformed through TF before comparison.",
                "observed_default_relation": "x_base=-x_base_link, y_base=-y_base_link, z_base=z_base_link (180 degree Z rotation).",
            },
            "coordinate_convention": COORDINATE_CONVENTION,
            "known_tool_geometry": KNOWN_TOOL_GEOMETRY,
            "samples": summary,
            "rejected_samples": {
                "wrong_frame": node.rejected_wrong_frame,
                "non_finite": node.rejected_non_finite,
            },
            "tf": tf_values,
            "derived": {
                "camera_to_mouth_vector_base_link_m": camera_to_mouth,
                "camera_to_mouth_distance_m": _norm(camera_to_mouth) if camera_to_mouth is not None else None,
                "mouth_mean_ur_base_m": mouth_mean_ur_base,
            },
            "markers": {
                "topic": args.marker_topic,
                "qos_durability": "transient_local",
                "published_marker_count": marker_count,
                "contents": ["red mouth mean sphere", "XYZ text label", "camera-to-mouth arrow"],
            },
        }
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = (args.report_dir / f"mouth_axes_{args.label}_{filename_timestamp}.json").resolve()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_acquisition(report, report_path)
        return (0 if report["success"] else 2), report_path
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _format_transform(name: str, transform: dict[str, Any]) -> str:
    if not transform.get("available"):
        return f"{name}: unavailable ({transform.get('reason', 'unknown reason')})"
    position = transform["translation_m"]
    quaternion = transform["orientation_quat_xyzw"]
    return (
        f"{name}: xyz=[{position[0]:+.6f}, {position[1]:+.6f}, {position[2]:+.6f}] m  "
        f"quat_xyzw=[{quaternion[0]:+.6f}, {quaternion[1]:+.6f}, {quaternion[2]:+.6f}, {quaternion[3]:+.6f}]"
    )


def _print_acquisition(report: dict[str, Any], report_path: Path) -> None:
    samples = report["samples"]
    print()
    print("Mouth-coordinate acquisition result")
    print(_format_transform("current tool0 in base_link", report["tf"]["base_link_to_tool0"]))
    print(_format_transform("current tool0 in pendant base", report["tf"]["ur_base_to_tool0"]))
    print(_format_transform("current camera optical frame in base_link", report["tf"]["base_link_to_camera_optical"]))
    print(_format_transform("camera optical frame in tool0", report["tf"]["tool0_to_camera_optical"]))
    if samples.get("mean_position_m") is not None:
        mean = samples["mean_position_m"]
        print(f"current mouth mean in base_link: x={mean[0]:+.6f} y={mean[1]:+.6f} z={mean[2]:+.6f} m")
        pendant_mean = report["derived"].get("mouth_mean_ur_base_m")
        if pendant_mean is not None:
            print(
                "current mouth mean in pendant base: "
                f"x={pendant_mean[0]:+.6f} y={pendant_mean[1]:+.6f} z={pendant_mean[2]:+.6f} m"
            )
        stddev = samples["stddev_m"]
        spread = samples["spread_m"]
        print(
            f"samples={samples['count']} stable={samples['stable']} "
            f"std=[{stddev['x']:.6f}, {stddev['y']:.6f}, {stddev['z']:.6f}] m "
            f"spread=[{spread['x']:.6f}, {spread['y']:.6f}, {spread['z']:.6f}] m "
            f"latest_age={samples.get('latest_pose_age_sec')} s"
        )
    else:
        print(f"mouth mean unavailable: {samples.get('failure')}")
    vector = report["derived"]["camera_to_mouth_vector_base_link_m"]
    if vector is not None:
        print(
            f"camera -> mouth in base_link: "
            f"[{vector[0]:+.6f}, {vector[1]:+.6f}, {vector[2]:+.6f}] m "
            f"(norm {report['derived']['camera_to_mouth_distance_m']:.6f} m)"
        )
    else:
        print("camera -> mouth in base_link: unavailable")
    print(f"marker topic: {report['markers']['topic']} (transient_local)")
    print(f"report: {report_path}")
    print("No robot motion was sent.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--label", choices=LABELS, help="Label for one live acquisition run.")
    mode.add_argument(
        "--compare",
        nargs="+",
        metavar="REPORT",
        help="Compare labelled JSON reports. Shell-expanded or quoted glob patterns are accepted.",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="Live sample collection time in seconds (default: 15).")
    parser.add_argument("--mouth-topic", default=MOUTH_TOPIC)
    parser.add_argument("--marker-topic", default=MARKER_TOPIC)
    parser.add_argument("--base-frame", default=BASE_FRAME)
    parser.add_argument("--pendant-base-frame", default=PENDANT_BASE_FRAME)
    parser.add_argument("--tool-frame", default=TOOL_FRAME)
    parser.add_argument("--camera-frame", default=CAMERA_FRAME)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--marker-hold-seconds",
        type=float,
        default=3.0,
        help="Keep the transient-local marker writer alive after publishing (default: 3).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive and finite")
    if not math.isfinite(args.marker_hold_seconds) or args.marker_hold_seconds < 0.0:
        parser.error("--marker-hold-seconds must be non-negative and finite")
    return args


def main() -> int:
    args = _parse_args()
    if args.compare:
        try:
            compare_reports(args.compare)
        except ValueError as exc:
            print(f"comparison error: {exc}", file=sys.stderr)
            return 2
        return 0
    try:
        code, _ = acquire_report(args)
        return code
    except KeyboardInterrupt:
        print("Interrupted; no robot motion was sent.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"acquisition error: {exc}", file=sys.stderr)
        print("No robot motion was sent.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
