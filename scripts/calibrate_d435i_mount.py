#!/usr/bin/env python3
"""Robot-assisted, no-motion hand-eye calibration for the wrist D435i.

This tool never sends a robot command.  It establishes the physical
``tool0 -> d435i_link`` transform from three fixed ArUco marker centres:

1. Generate/display the board (IDs 10, 11, 12) with ``--mode make-board``.
2. Keep the board fixed.  In pendant/freedrive mode, manually align the straw
   tip to each marker centre and record it with ``--mode touch --marker-id``.
3. Put the robot in any stationary view where the D435i sees all markers, then
   run ``--mode solve --write-config``.  The solver uses RGB-D marker centres,
   the recorded physical straw points, and current TF; it does not move UR.

The output config is accepted only when the residual and marker geometry are
good enough for a physical pre-mouth direction.  Restart the project-local
D435i perception wrapper after a successful write so it loads the transform.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import rclpy
import rclpy.time
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
CAMERA_LINK_FRAME = "d435i_link"
CAMERA_OPTICAL_FRAME = "d435i_color_optical_frame"
STRAW_TIP_OFFSET_TOOL0_M = np.asarray([0.110, 0.0, 0.0], dtype=np.float64)
DEFAULT_TOUCHES = PROJECT_ROOT / "config/ur10e_real/d435i_handeye_touches.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config/ur10e_real/d435i_mount_calibration.json"
DEFAULT_BOARD = PROJECT_ROOT / "assets/calibration/d435i_handeye_board.png"
DEFAULT_MARKER_IDS = (10, 11, 12)
MAX_RMS_RESIDUAL_M = 0.010
MIN_MARKER_BASELINE_M = 0.030
MIN_TRIANGLE_AREA_M2 = 0.0005


def _matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_from_matrix(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return [float(x), float(y), float(z), float(w)]


def _rpy_from_matrix(rotation: np.ndarray) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return [roll, pitch, yaw]


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve target ~= rotation @ source + translation using Kabsch."""
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    return rotation, translation, residuals


class CalibrationNode(Node):
    def __init__(self, rgb_topic: str, depth_topic: str, camera_info_topic: str) -> None:
        super().__init__("d435i_mount_calibration")
        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.latest_info: CameraInfo | None = None
        self.create_subscription(Image, rgb_topic, self._rgb_callback, 5)
        self.create_subscription(Image, depth_topic, self._depth_callback, 5)
        self.create_subscription(CameraInfo, camera_info_topic, self._info_callback, 5)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)

    def _rgb_callback(self, message: Image) -> None:
        self.latest_rgb = message

    def _depth_callback(self, message: Image) -> None:
        self.latest_depth = message

    def _info_callback(self, message: CameraInfo) -> None:
        self.latest_info = message

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, deadline - time.monotonic()))

    def transform(self, target: str, source: str, timeout_sec: float = 3.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.0))
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                return _homogeneous(
                    _matrix_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w),
                    np.asarray([translation.x, translation.y, translation.z], dtype=np.float64),
                )
            except Exception as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f"TF {target} -> {source} unavailable: {last_error}")

    @staticmethod
    def _color_array(message: Image) -> np.ndarray:
        if message.encoding not in {"rgb8", "bgr8"}:
            raise RuntimeError(f"unsupported RGB encoding {message.encoding}")
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
        return image if message.encoding == "bgr8" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _depth_array(message: Image) -> np.ndarray:
        if message.encoding == "16UC1":
            rows = np.frombuffer(message.data, dtype=np.uint16).reshape(message.height, message.step // 2)
            return rows[:, : message.width].astype(np.float64) * 0.001
        if message.encoding == "32FC1":
            rows = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.step // 4)
            return rows[:, : message.width].astype(np.float64)
        raise RuntimeError(f"unsupported depth encoding {message.encoding}")

    @staticmethod
    def _depth_point(depth_m: np.ndarray, info: CameraInfo, pixel: np.ndarray) -> np.ndarray | None:
        x = int(round(float(pixel[0]) * depth_m.shape[1] / info.width))
        y = int(round(float(pixel[1]) * depth_m.shape[0] / info.height))
        values: list[float] = []
        for row in range(max(0, y - 3), min(depth_m.shape[0], y + 4)):
            for column in range(max(0, x - 3), min(depth_m.shape[1], x + 4)):
                value = float(depth_m[row, column])
                if math.isfinite(value) and 0.15 <= value <= 4.0:
                    values.append(value)
        if not values:
            return None
        z = float(np.median(values))
        scale_x = depth_m.shape[1] / info.width
        scale_y = depth_m.shape[0] / info.height
        fx, fy = info.k[0] * scale_x, info.k[4] * scale_y
        cx, cy = info.k[2] * scale_x, info.k[5] * scale_y
        if fx <= 0.0 or fy <= 0.0:
            return None
        return np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64)

    def marker_points(self, marker_ids: list[int], timeout_sec: float = 8.0) -> dict[int, np.ndarray]:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_rgb is None or self.latest_depth is None or self.latest_info is None:
                continue
            try:
                image = self._color_array(self.latest_rgb)
                depth = self._depth_array(self.latest_depth)
            except RuntimeError:
                continue
            corners, ids, _ = detector.detectMarkers(image)
            if ids is None:
                continue
            result: dict[int, np.ndarray] = {}
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                value = int(marker_id)
                if value not in marker_ids:
                    continue
                point = self._depth_point(depth, self.latest_info, np.mean(marker_corners.reshape(4, 2), axis=0))
                if point is not None:
                    result[value] = point
            if all(marker_id in result for marker_id in marker_ids):
                return result
        raise RuntimeError(f"did not obtain valid RGB-D centres for marker IDs {marker_ids}")


def _load_touches(path: Path, marker_ids: list[int]) -> dict[int, np.ndarray]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read touch file {path}: {exc}") from exc
    points = data.get("points", {})
    result: dict[int, np.ndarray] = {}
    for marker_id in marker_ids:
        entry = points.get(str(marker_id))
        values = None if not isinstance(entry, dict) else entry.get("base_position_m")
        if not isinstance(values, list) or len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"touch file lacks a valid base point for marker {marker_id}")
        result[marker_id] = np.asarray(values, dtype=np.float64)
    return result


def _write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_board(path: Path, marker_ids: list[int], overwrite: bool) -> dict[str, Any]:
    if len(marker_ids) != 3:
        raise RuntimeError("the calibration board requires exactly three marker IDs")
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {path}; pass --overwrite")
    canvas = np.full((1300, 1500), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    positions = ((150, 120), (980, 120), (565, 690))
    marker_size = 320
    for marker_id, (x, y) in zip(marker_ids, positions):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        canvas[y : y + marker_size, x : x + marker_size] = marker
        cv2.putText(canvas, f"ID {marker_id}", (x + 85, y + marker_size + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3)
    cv2.putText(canvas, "D435i / UR10e hand-eye calibration board", (245, 1235), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")
    return {"success": True, "mode": "make-board", "board_image": str(path), "marker_ids": marker_ids}


def _record_touch(args: argparse.Namespace) -> dict[str, Any]:
    if args.marker_id is None:
        raise RuntimeError("--marker-id is required with --mode touch")
    rclpy.init()
    node = CalibrationNode(args.rgb_topic, args.depth_topic, args.camera_info_topic)
    try:
        tool = node.transform(BASE_FRAME, TOOL_FRAME)
        straw = (tool @ np.asarray([*STRAW_TIP_OFFSET_TOOL0_M, 1.0], dtype=np.float64))[:3]
    finally:
        node.destroy_node()
        rclpy.shutdown()
    data: dict[str, Any]
    if args.touches.exists():
        try:
            data = json.loads(args.touches.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read touch file {args.touches}: {exc}") from exc
    else:
        data = {"frame_id": BASE_FRAME, "straw_tip_offset_tool0_m": STRAW_TIP_OFFSET_TOOL0_M.tolist(), "points": {}}
    data.setdefault("points", {})[str(args.marker_id)] = {
        "base_position_m": [float(value) for value in straw],
        "recorded_unix_sec": time.time(),
    }
    _write_json(args.touches, data, overwrite=True)
    return {"success": True, "mode": "touch", "marker_id": args.marker_id, "straw_tip_base_position_m": straw.tolist(), "touches": str(args.touches)}


def _solve(args: argparse.Namespace) -> dict[str, Any]:
    base_points = _load_touches(args.touches, args.marker_ids)
    ordered_base = np.asarray([base_points[marker_id] for marker_id in args.marker_ids], dtype=np.float64)
    baselines = [float(np.linalg.norm(ordered_base[i] - ordered_base[j])) for i in range(3) for j in range(i + 1, 3)]
    triangle_area = 0.5 * float(np.linalg.norm(np.cross(ordered_base[1] - ordered_base[0], ordered_base[2] - ordered_base[0])))
    if min(baselines) < MIN_MARKER_BASELINE_M or triangle_area < MIN_TRIANGLE_AREA_M2:
        raise RuntimeError("recorded marker centres are too close or collinear for calibration")

    rclpy.init()
    node = CalibrationNode(args.rgb_topic, args.depth_topic, args.camera_info_topic)
    try:
        camera_points = node.marker_points(args.marker_ids)
        base_tool = node.transform(BASE_FRAME, TOOL_FRAME)
        link_camera = node.transform(CAMERA_LINK_FRAME, CAMERA_OPTICAL_FRAME)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    camera = np.asarray([camera_points[marker_id] for marker_id in args.marker_ids], dtype=np.float64)
    tool_base = np.linalg.inv(base_tool)
    tool_points = np.asarray([(tool_base @ np.asarray([*point, 1.0]))[:3] for point in ordered_base], dtype=np.float64)
    tool_camera_rotation, tool_camera_translation, residuals = _rigid_transform(camera, tool_points)
    tool_camera = _homogeneous(tool_camera_rotation, tool_camera_translation)
    tool_link = tool_camera @ np.linalg.inv(link_camera)
    rms = float(math.sqrt(float(np.mean(residuals * residuals))))
    valid = rms <= MAX_RMS_RESIDUAL_M
    result = {
        "success": valid,
        "mode": "solve",
        "marker_ids": args.marker_ids,
        "residuals_m": [float(value) for value in residuals],
        "rms_residual_m": rms,
        "maximum_rms_residual_m": MAX_RMS_RESIDUAL_M,
        "minimum_marker_baseline_m": min(baselines),
        "triangle_area_m2": triangle_area,
        "tool0_to_d435i_color_optical_frame": {
            "translation_m": [float(value) for value in tool_camera[:3, 3]],
            "quaternion_xyzw": _quaternion_from_matrix(tool_camera[:3, :3]),
        },
        "tool0_to_d435i_link": {
            "translation_m": [float(value) for value in tool_link[:3, 3]],
            "rpy_rad": _rpy_from_matrix(tool_link[:3, :3]),
        },
        "touches": str(args.touches),
        "config": str(args.config),
    }
    if args.write_config:
        if not valid:
            raise RuntimeError(f"refusing to write calibration: RMS residual {rms:.4f} m exceeds {MAX_RMS_RESIDUAL_M:.4f} m")
        config = {
            "calibration_status": "verified",
            "description": "Physical three-marker RGB-D / straw-tip hand-eye calibration.",
            "frames": {"parent": TOOL_FRAME, "child": CAMERA_LINK_FRAME, "optical_frame": CAMERA_OPTICAL_FRAME},
            "tool0_to_d435i_link": result["tool0_to_d435i_link"],
            "calibration_metrics": {
                "marker_ids": args.marker_ids,
                "rms_residual_m": rms,
                "residuals_m": result["residuals_m"],
                "triangle_area_m2": triangle_area,
                "created_unix_sec": time.time(),
            },
        }
        _write_json(args.config, config, overwrite=args.overwrite)
        result["config_written"] = True
    else:
        result["config_written"] = False
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("make-board", "touch", "solve"), required=True)
    parser.add_argument("--marker-ids", type=int, nargs="+", default=list(DEFAULT_MARKER_IDS))
    parser.add_argument("--marker-id", type=int)
    parser.add_argument("--touches", type=Path, default=DEFAULT_TOUCHES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--board-image", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb-topic", default="/d435i/d435i/color/image_raw")
    parser.add_argument("--depth-topic", default="/d435i/d435i/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/d435i/d435i/color/camera_info")
    args = parser.parse_args()
    if len(args.marker_ids) != 3 or len(set(args.marker_ids)) != 3:
        parser.error("--marker-ids must contain exactly three distinct IDs")
    if args.mode != "touch" and args.marker_id is not None:
        parser.error("--marker-id is valid only with --mode touch")
    if args.mode == "solve" and not args.write_config and args.overwrite:
        parser.error("--overwrite requires --write-config or --mode make-board")
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.mode == "make-board":
            result = _make_board(args.board_image, args.marker_ids, args.overwrite)
        elif args.mode == "touch":
            result = _record_touch(args)
        else:
            result = _solve(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("success") else 2
    except Exception as exc:
        print(json.dumps({"success": False, "mode": args.mode, "reason": f"{exc.__class__.__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
