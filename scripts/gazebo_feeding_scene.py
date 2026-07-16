#!/usr/bin/env python3
"""Create and update the UR10e feeding scene markers in Gazebo.

Run this on the ThinkPad, because Gazebo is running there and exposes its
`/world/empty/create` and `/world/empty/set_pose` services through gz transport.
The 5090 MoveIt demo can then move the robot while this helper keeps the
CurrentStrawTip and CurrentCamera markers aligned with ROS TF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


MODEL_PREFIX = "feeding_"
WORLD_NAME = "empty"
BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"


DEFAULT_FEEDING = {
    "flange_to_camera_optical_center": [0.07, 0.0, -0.015],
    "flange_to_straw_tip": [0.11, 0.0, 0.0],
    "ready_straw_tip_position": [0.25, 0.25, 0.85],
    "pre_mouth_safe_position": [0.357, 0.860, 1.708],
    "mouth_target_position": [0.357, 0.940, 1.708],
    "flange_down_rpy": [math.pi, 0.0, 0.0],
}


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotation_matrix_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _parse_xyz(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(out) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return out


def _find_default_config() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.with_name("config.yaml"),
        here.parents[1] / "agent_server" / "robot_sdk" / "config.yaml",
    ]
    for parent in here.parents:
        candidates.append(parent / "robot_layer" / "arm_ur10e" / "agent_server" / "robot_sdk" / "config.yaml")
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_feeding_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_FEEDING)
    if path is not None and path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.update(data.get("feeding", {}))
    return cfg


def _run_gz(service: str, reqtype: str, req: str, timeout_ms: int = 3000) -> subprocess.CompletedProcess:
    if shutil.which("gz") is None:
        raise RuntimeError("`gz` command not found. Run: source /opt/ros/jazzy/setup.bash")
    return subprocess.run(
        [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(timeout_ms),
            "--req",
            req,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _gz_bool_success(result: subprocess.CompletedProcess) -> bool:
    if result.returncode != 0:
        return False
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return "data: true" in stdout or (stdout == "" and stderr == "")


def _remove_model(name: str) -> bool:
    result = _run_gz(
        f"/world/{WORLD_NAME}/remove/blocking",
        "gz.msgs.Entity",
        f"name: {json.dumps(name)} type: MODEL",
        timeout_ms=3000,
    )
    return _gz_bool_success(result)


def _sphere_sdf(name: str, position: Iterable[float], radius: float, color: Iterable[float]) -> str:
    x, y, z = [float(v) for v in position]
    r, g, b, a = [float(v) for v in color]
    return (
        f"<sdf version=\"1.7\"><model name=\"{name}\"><static>true</static>"
        f"<pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose><link name=\"link\">"
        "<visual name=\"visual\"><geometry>"
        f"<sphere><radius>{float(radius):.6f}</radius></sphere>"
        "</geometry><material>"
        f"<ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>"
        "</material></visual></link></model></sdf>"
    )


def _create_sphere(
    name: str,
    position: Iterable[float],
    radius: float,
    color: Iterable[float],
    *,
    replace: bool = True,
) -> bool:
    if replace:
        _remove_model(name)
    sdf = _sphere_sdf(name, position, radius, color)
    result = _run_gz(
        f"/world/{WORLD_NAME}/create/blocking",
        "gz.msgs.EntityFactory",
        "sdf: " + json.dumps(sdf),
        timeout_ms=30000,
    )
    if not _gz_bool_success(result):
        print(
            json.dumps(
                {
                    "event": "gazebo_create_failed",
                    "name": name,
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                },
                sort_keys=True,
            )
        )
        return False
    return True


def _existing_models() -> set[str]:
    result = subprocess.run(
        ["gz", "model", "--list"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {
        line.lstrip()[2:].strip()
        for line in result.stdout.splitlines()
        if line.lstrip().startswith("- ")
    }


def _set_pose(name: str, position: Iterable[float]) -> bool:
    x, y, z = [float(v) for v in position]
    req = (
        f"name: {json.dumps(name)} "
        f"position {{ x: {x:.6f} y: {y:.6f} z: {z:.6f} }} "
        "orientation { x: 0 y: 0 z: 0 w: 1 }"
    )
    result = _run_gz(f"/world/{WORLD_NAME}/set_pose", "gz.msgs.Pose", req, timeout_ms=1000)
    return _gz_bool_success(result)


def _path_dots(start: Iterable[float], end: Iterable[float], count: int) -> list[list[float]]:
    a = np.asarray(list(start), dtype=float)
    b = np.asarray(list(end), dtype=float)
    if count <= 1:
        return [a.tolist()]
    return [(a + (b - a) * i / (count - 1)).tolist() for i in range(count)]


def _scene_models(feeding: dict) -> dict[str, tuple[list[float], float, tuple[float, float, float, float]]]:
    mouth = np.asarray(feeding["mouth_target_position"], dtype=float)

    return {
        # The pre-mouth waypoint is logical-only so it cannot obstruct the
        # wrist RGB image. The mouth marker is intentionally small.
        MODEL_PREFIX + "mouth_target": (mouth.tolist(), 0.010, (1.0, 0.1, 0.1, 1.0)),
    }


def _all_model_names() -> list[str]:
    names = [
        MODEL_PREFIX + "ready_straw_tip",
        MODEL_PREFIX + "pre_mouth_safe_target",
        MODEL_PREFIX + "mouth_target",
        MODEL_PREFIX + "tool0_target_at_pre_mouth",
        MODEL_PREFIX + "camera_at_pre_mouth",
        MODEL_PREFIX + "current_straw_tip",
        MODEL_PREFIX + "current_camera",
        "codex_spawn_test_sphere",
    ]
    names.extend(f"{MODEL_PREFIX}path_ready_to_pre_{i:02d}" for i in range(16))
    names.extend(f"{MODEL_PREFIX}path_pre_to_mouth_{i:02d}" for i in range(16))
    return names


def _cleanup() -> None:
    for name in _all_model_names():
        _remove_model(name)
    print(json.dumps({"event": "gazebo_feeding_scene_cleanup", "world": WORLD_NAME}, sort_keys=True))


def _spawn_scene(feeding: dict) -> dict:
    models = _scene_models(feeding)
    created = []
    failed = []
    for name, (position, radius, color) in models.items():
        if _create_sphere(name, position, radius, color):
            created.append(name)
        else:
            failed.append(name)
    info = {
        "event": "gazebo_feeding_scene_spawned",
        "world": WORLD_NAME,
        "created": created,
        "failed": failed,
        "ready_straw_tip_position": feeding["ready_straw_tip_position"],
        "pre_mouth_safe_position": feeding["pre_mouth_safe_position"],
        "mouth_target_position": feeding["mouth_target_position"],
        "flange_to_straw_tip": feeding["flange_to_straw_tip"],
        "flange_to_camera_optical_center": feeding["flange_to_camera_optical_center"],
    }
    print(json.dumps(info, sort_keys=True))
    return info


def _spawn_missing_scene(feeding: dict) -> dict:
    models = _scene_models(feeding)
    existing = _existing_models()
    created = []
    reused = []
    failed = []
    for name, (position, radius, color) in models.items():
        if name in existing:
            reused.append(name)
        elif _create_sphere(name, position, radius, color, replace=False):
            created.append(name)
        else:
            failed.append(name)
    info = {
        "event": "gazebo_feeding_scene_reused",
        "world": WORLD_NAME,
        "created": created,
        "reused": reused,
        "failed": failed,
        "pre_mouth_safe_position": feeding["pre_mouth_safe_position"],
        "mouth_target_position": feeding["mouth_target_position"],
    }
    print(json.dumps(info, sort_keys=True))
    return info


def _track_current_markers(feeding: dict, rate_hz: float, duration: float) -> None:
    import rclpy
    import rclpy.time
    from rclpy.duration import Duration as RclpyDuration
    import tf2_ros

    rclpy.init(args=None)
    node = rclpy.create_node("gazebo_feeding_scene_tracker")
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)
    period = 1.0 / max(float(rate_hz), 0.1)
    deadline = None if duration <= 0.0 else time.time() + float(duration)
    straw_offset = np.asarray(feeding["flange_to_straw_tip"], dtype=float)
    camera_offset = np.asarray(feeding["flange_to_camera_optical_center"], dtype=float)
    try:
        while deadline is None or time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                tf = tf_buffer.lookup_transform(
                    BASE_FRAME,
                    TOOL_FRAME,
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=0.2),
                )
                p = np.asarray(
                    [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ],
                    dtype=float,
                )
                q = tf.transform.rotation
                rotation = rotation_matrix_from_quaternion(q.x, q.y, q.z, q.w)
                straw = (p + rotation @ straw_offset).tolist()
                camera = (p + rotation @ camera_offset).tolist()
                _set_pose(MODEL_PREFIX + "current_straw_tip", straw)
                _set_pose(MODEL_PREFIX + "current_camera", camera)
            except Exception as exc:
                print(json.dumps({"event": "tf_waiting", "error": str(exc)}, sort_keys=True))
            time.sleep(period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional UR10e SDK config.yaml path.")
    parser.add_argument("--ready", type=_parse_xyz, default=None, help="Override ready straw tip as x,y,z.")
    parser.add_argument("--pre-mouth", type=_parse_xyz, default=None, help="Override pre-mouth target as x,y,z.")
    parser.add_argument("--mouth", type=_parse_xyz, default=None, help="Override mouth target as x,y,z.")
    parser.add_argument("--rate", type=float, default=2.0, help="Current marker update rate in Hz.")
    parser.add_argument("--duration", type=float, default=0.0, help="Tracking duration. 0 means run until Ctrl+C.")
    parser.add_argument("--once", action="store_true", help="Spawn the scene once and exit without TF tracking.")
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Reuse marker models already defined in the world file; create only missing models.",
    )
    parser.add_argument("--cleanup", action="store_true", help="Remove feeding scene markers and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = args.config or _find_default_config()
    feeding = _load_feeding_config(cfg_path)
    if args.ready is not None:
        feeding["ready_straw_tip_position"] = args.ready
    if args.pre_mouth is not None:
        feeding["pre_mouth_safe_position"] = args.pre_mouth
    if args.mouth is not None:
        feeding["mouth_target_position"] = args.mouth

    if args.cleanup:
        _cleanup()
        return 0

    if args.use_existing:
        _spawn_missing_scene(feeding)
    else:
        _cleanup()
        _spawn_scene(feeding)
    if args.once:
        return 0
    print(json.dumps({"event": "gazebo_feeding_scene_attached_markers", "detail": "current camera and straw tip markers are fixed URDF links on tool0"}, sort_keys=True))
    if duration := float(args.duration):
        time.sleep(duration)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"event": "gazebo_feeding_scene_interrupted"}, sort_keys=True))
        raise SystemExit(130)
