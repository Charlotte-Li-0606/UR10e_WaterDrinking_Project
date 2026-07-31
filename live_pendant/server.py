#!/usr/bin/env python3
"""Serve a read-only, live browser pendant backed by ROS 2 topics.

The HTTP surface intentionally contains GET endpoints only.  It does not call
motion, controller-switching, or dashboard-write services.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import mimetypes
import signal
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


STATIC_DIR = Path(__file__).resolve().parent / "static"
ROBOT_MODE_NAMES = {
    -1: "NO CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM SAFETY",
    2: "BOOTING",
    3: "POWER OFF",
    4: "POWER ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING FIRMWARE",
}
SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE STOP",
    4: "RECOVERY",
    5: "SAFEGUARD STOP",
    6: "SYSTEM EMERGENCY STOP",
    7: "ROBOT EMERGENCY STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE JOINT ID",
    11: "UNDEFINED",
    12: "AUTO SAFEGUARD STOP",
    13: "THREE-POSITION STOP",
}
CORE_STREAMS = ("robot_mode", "safety_mode", "joints", "tcp")
STALE_AFTER_SECONDS = 2.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def quaternion_to_rotation_vector(x: float, y: float, z: float, w: float) -> list[float]:
    """Convert a quaternion into the axis-angle vector shown by PolyScope."""

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:  # Pick the shortest equivalent rotation.
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, w)))
    scale = angle / vector_norm
    return [x * scale, y * scale, z * scale]


class PendantState:
    """Thread-safe latest-value store shared by ROS and HTTP threads."""

    def __init__(self, source: str = "ros2") -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "source": source,
            "read_only": True,
            "robot": {
                "mode": None,
                "mode_code": None,
                "safety_mode": None,
                "safety_mode_code": None,
                "program_running": None,
                "program_state": None,
                "program_name": None,
                "remote_control": None,
                "speed_scaling": None,
            },
            "tcp": None,
            "joints": [],
            "io": None,
            "tool": None,
            "controllers": [],
            "camera": {"available": False, "width": None, "height": None, "encoding": None},
            "streams": {},
        }
        self._camera_jpeg: bytes | None = None

    def update(
        self, section: str, value: Any, stream: str | None = None, *, retained: bool = False
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._state[section] = value
            if stream:
                self._state["streams"][stream] = {
                    "received_at": utc_now(),
                    "monotonic": now,
                    "retained": retained,
                }

    def update_robot(self, values: dict[str, Any], stream: str, *, retained: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            self._state["robot"].update(values)
            self._state["streams"][stream] = {
                "received_at": utc_now(),
                "monotonic": now,
                "retained": retained,
            }

    def update_camera(self, jpeg: bytes, width: int, height: int, encoding: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._camera_jpeg = jpeg
            self._state["camera"] = {
                "available": True,
                "width": width,
                "height": height,
                "encoding": encoding,
            }
            self._state["streams"]["camera"] = {"received_at": utc_now(), "monotonic": now}

    def camera_jpeg(self) -> bytes | None:
        with self._lock:
            return self._camera_jpeg

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            result = copy.deepcopy(self._state)
        streams = result["streams"]
        for metadata in streams.values():
            stamp = metadata.pop("monotonic", None)
            metadata["age_seconds"] = None if stamp is None else round(max(0.0, now - stamp), 3)
            metadata["fresh"] = bool(metadata.get("retained")) or (
                stamp is not None and now - stamp <= STALE_AFTER_SECONDS
            )
        present = {name: streams.get(name, {}).get("fresh", False) for name in CORE_STREAMS}
        realtime_present = present["joints"] or present["tcp"]
        result["connected"] = all(present.values())
        result["connection_state"] = (
            "live" if result["connected"] else ("partial" if realtime_present else "offline")
        )
        result["generated_at"] = utc_now()
        result["uptime_seconds"] = round(now - self._started, 1)
        return result


class RosPendantBridge:
    """Own the ROS node and map live messages into :class:`PendantState`."""

    def __init__(self, state: PendantState, camera_topic: str) -> None:
        # Imports stay local so --demo and unit tests work without a sourced ROS environment.
        import cv2
        import rclpy
        from controller_manager_msgs.srv import ListControllers
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Bool, Float64
        from ur_dashboard_msgs.msg import RobotMode, SafetyMode
        from ur_dashboard_msgs.srv import GetProgramState, IsInRemoteControl
        from ur_msgs.msg import IOStates, ToolDataMsg

        self._rclpy = rclpy
        self._executor = SingleThreadedExecutor()
        self._node = rclpy.create_node("ur10e_live_pendant")
        self._state = state
        self._cv2 = cv2
        self._cv_bridge = CvBridge()
        self._last_camera_encode = 0.0
        self._stopping = False
        self._service_pending: dict[str, bool] = {}
        self._service_futures: set[Any] = set()
        retained_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._node.create_subscription(
            RobotMode,
            "/io_and_status_controller/robot_mode",
            self._on_robot_mode,
            retained_qos,
        )
        self._node.create_subscription(
            SafetyMode,
            "/io_and_status_controller/safety_mode",
            self._on_safety_mode,
            retained_qos,
        )
        self._node.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._on_program_running,
            retained_qos,
        )
        self._node.create_subscription(
            Float64,
            "/speed_scaling_state_broadcaster/speed_scaling",
            self._on_speed_scaling,
            retained_qos,
        )
        self._node.create_subscription(JointState, "/joint_states", self._on_joints, retained_qos)
        self._node.create_subscription(
            PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp, retained_qos
        )
        self._node.create_subscription(
            IOStates,
            "/io_and_status_controller/io_states",
            self._on_io,
            retained_qos,
        )
        self._node.create_subscription(
            ToolDataMsg,
            "/io_and_status_controller/tool_data",
            self._on_tool,
            retained_qos,
        )
        self._node.create_subscription(Image, camera_topic, self._on_camera, qos_profile_sensor_data)

        self._controller_client = self._node.create_client(ListControllers, "/controller_manager/list_controllers")
        self._remote_client = self._node.create_client(
            IsInRemoteControl, "/dashboard_client/is_in_remote_control"
        )
        self._program_client = self._node.create_client(GetProgramState, "/dashboard_client/program_state")
        self._service_timer = self._node.create_timer(2.0, self._poll_read_only_services)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name="live-pendant-ros", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._service_timer.cancel()
        for future in list(self._service_futures):
            future.cancel()
        self._executor.remove_node(self._node)
        self._executor.shutdown(timeout_sec=2.0)
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        self._thread.join(timeout=2.0)

    def _on_robot_mode(self, msg: Any) -> None:
        code = int(msg.mode)
        self._state.update_robot(
            {"mode": ROBOT_MODE_NAMES.get(code, f"UNKNOWN ({code})"), "mode_code": code},
            "robot_mode",
            retained=True,
        )

    def _on_safety_mode(self, msg: Any) -> None:
        code = int(msg.mode)
        self._state.update_robot(
            {"safety_mode": SAFETY_MODE_NAMES.get(code, f"UNKNOWN ({code})"), "safety_mode_code": code},
            "safety_mode",
            retained=True,
        )

    def _on_program_running(self, msg: Any) -> None:
        self._state.update_robot({"program_running": bool(msg.data)}, "program", retained=True)

    def _on_speed_scaling(self, msg: Any) -> None:
        self._state.update_robot({"speed_scaling": float(msg.data)}, "speed_scaling", retained=True)

    def _on_joints(self, msg: Any) -> None:
        velocities = list(msg.velocity)
        efforts = list(msg.effort)
        joints = []
        for index, name in enumerate(msg.name):
            position = float(msg.position[index]) if index < len(msg.position) else None
            velocity = float(velocities[index]) if index < len(velocities) else None
            effort = float(efforts[index]) if index < len(efforts) else None
            joints.append(
                {
                    "name": str(name),
                    "position_rad": position,
                    "position_deg": None if position is None else math.degrees(position),
                    "velocity_rad_s": velocity,
                    "velocity_deg_s": None if velocity is None else math.degrees(velocity),
                    "effort": effort,
                }
            )
        self._state.update("joints", joints, "joints")

    def _on_tcp(self, msg: Any) -> None:
        pose = msg.pose
        quaternion = pose.orientation
        self._state.update(
            "tcp",
            {
                "frame": str(msg.header.frame_id),
                "position_m": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                "position_mm": [
                    float(pose.position.x) * 1000.0,
                    float(pose.position.y) * 1000.0,
                    float(pose.position.z) * 1000.0,
                ],
                "rotation_vector_rad": quaternion_to_rotation_vector(
                    float(quaternion.x),
                    float(quaternion.y),
                    float(quaternion.z),
                    float(quaternion.w),
                ),
                "quaternion": [
                    float(quaternion.x),
                    float(quaternion.y),
                    float(quaternion.z),
                    float(quaternion.w),
                ],
            },
            "tcp",
        )

    def _on_io(self, msg: Any) -> None:
        def digital(items: Any) -> list[dict[str, Any]]:
            return [{"pin": int(item.pin), "state": bool(item.state)} for item in items]

        def analog(items: Any) -> list[dict[str, Any]]:
            return [
                {"pin": int(item.pin), "state": float(item.state), "domain": int(item.domain)} for item in items
            ]

        self._state.update(
            "io",
            {
                "digital_inputs": digital(msg.digital_in_states),
                "digital_outputs": digital(msg.digital_out_states),
                "flags": digital(msg.flag_states),
                "analog_inputs": analog(msg.analog_in_states),
                "analog_outputs": analog(msg.analog_out_states),
            },
            "io",
        )

    def _on_tool(self, msg: Any) -> None:
        self._state.update(
            "tool",
            {
                "voltage_v": float(msg.tool_voltage_48v),
                "output_voltage_v": int(msg.tool_output_voltage),
                "current_a": float(msg.tool_current),
                "temperature_c": float(msg.tool_temperature),
                "analog_input_2": float(msg.analog_input2),
                "analog_input_3": float(msg.analog_input3),
                "mode": int(msg.tool_mode),
            },
            "tool",
        )

    def _on_camera(self, msg: Any) -> None:
        now = time.monotonic()
        if now - self._last_camera_encode < 0.12:
            return
        self._last_camera_encode = now
        try:
            frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, encoded = self._cv2.imencode(".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self._state.update_camera(bytes(encoded), int(msg.width), int(msg.height), str(msg.encoding))
        except Exception as exc:  # Camera failure must not take down robot status.
            self._node.get_logger().warning(f"Could not encode camera frame: {exc}")

    def _call_service(
        self,
        name: str,
        client: Any,
        request: Any,
        callback: Callable[[Any], None],
    ) -> None:
        if self._service_pending.get(name) or not client.service_is_ready():
            return
        self._service_pending[name] = True
        future = client.call_async(request)
        self._service_futures.add(future)

        def done(completed: Any) -> None:
            self._service_pending[name] = False
            self._service_futures.discard(completed)
            if self._stopping or completed.cancelled():
                return
            try:
                callback(completed.result())
            except Exception as exc:
                self._node.get_logger().debug(f"Read-only {name} query failed: {exc}")

        future.add_done_callback(done)

    def _poll_read_only_services(self) -> None:
        if self._stopping:
            return
        from controller_manager_msgs.srv import ListControllers
        from ur_dashboard_msgs.srv import GetProgramState, IsInRemoteControl

        self._call_service(
            "controllers",
            self._controller_client,
            ListControllers.Request(),
            lambda response: self._state.update(
                "controllers",
                [
                    {"name": item.name, "state": item.state, "type": item.type}
                    for item in response.controller
                ],
                "controllers",
            ),
        )
        self._call_service(
            "remote_control",
            self._remote_client,
            IsInRemoteControl.Request(),
            lambda response: self._state.update_robot(
                {"remote_control": bool(response.remote_control) if response.success else None}, "remote_control"
            ),
        )
        self._call_service(
            "program_state",
            self._program_client,
            GetProgramState.Request(),
            lambda response: self._state.update_robot(
                {
                    "program_state": str(response.state.state) if response.success else None,
                    "program_name": str(response.program_name) if response.success else None,
                },
                "program_state",
            ),
        )


class DemoBridge:
    """Animate deterministic data for browser development and smoke tests."""

    def __init__(self, state: PendantState) -> None:
        self._state = state
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-pendant-demo", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        offsets = [-44.2, -94.8, -102.4, -73.1, 89.4, 13.8]
        while not self._stop.wait(0.1):
            phase = time.monotonic() * 0.35
            self._state.update_robot({"mode": "RUNNING", "mode_code": 7}, "robot_mode")
            self._state.update_robot({"safety_mode": "NORMAL", "safety_mode_code": 1}, "safety_mode")
            self._state.update_robot(
                {
                    "program_running": True,
                    "program_state": "PLAYING",
                    "program_name": "external_control.urp",
                    "remote_control": True,
                    "speed_scaling": 0.1,
                },
                "program",
            )
            self._state.update_robot({"speed_scaling": 0.1}, "speed_scaling")
            self._state.update(
                "joints",
                [
                    {
                        "name": name,
                        "position_rad": math.radians(offset + math.sin(phase + index) * 0.03),
                        "position_deg": offset + math.sin(phase + index) * 0.03,
                        "velocity_rad_s": 0.0,
                        "velocity_deg_s": 0.0,
                        "effort": 0.0,
                    }
                    for index, (name, offset) in enumerate(zip(names, offsets))
                ],
                "joints",
            )
            self._state.update(
                "tcp",
                {
                    "frame": "base",
                    "position_m": [0.139, -0.438, 0.900],
                    "position_mm": [138.8, -438.3, 900.2],
                    "rotation_vector_rad": [2.34, -2.08, -0.25],
                    "quaternion": [0.745, -0.661, -0.081, 0.041],
                },
                "tcp",
            )
            bits = [{"pin": pin, "state": pin in (0, 2, 7)} for pin in range(8)]
            self._state.update(
                "io",
                {
                    "digital_inputs": bits,
                    "digital_outputs": [{"pin": pin, "state": pin == 1} for pin in range(8)],
                    "flags": [],
                    "analog_inputs": [],
                    "analog_outputs": [],
                },
                "io",
            )
            self._state.update(
                "controllers",
                [
                    {"name": "scaled_joint_trajectory_controller", "state": "active", "type": "trajectory"},
                    {"name": "joint_state_broadcaster", "state": "active", "type": "broadcaster"},
                    {"name": "io_and_status_controller", "state": "active", "type": "gpio"},
                ],
                "controllers",
            )


class PendantRequestHandler(BaseHTTPRequestHandler):
    server_version = "UR10eLivePendant/1.0"

    @property
    def pendant_state(self) -> PendantState:
        return self.server.pendant_state  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._send_json(self.pendant_state.snapshot())
            return
        if path == "/api/health":
            snapshot = self.pendant_state.snapshot()
            self._send_json(
                {
                    "ok": True,
                    "connection_state": snapshot["connection_state"],
                    "source": snapshot["source"],
                    "read_only": True,
                }
            )
            return
        if path == "/api/camera.jpg":
            jpeg = self.pendant_state.camera_jpeg()
            if jpeg is None:
                self.send_error(HTTPStatus.NOT_FOUND, "No camera frame received")
                return
            self._send_bytes(jpeg, "image/jpeg", cache="no-store")
            return
        if path == "/":
            path = "/index.html"
        relative = path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        media_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self._send_bytes(candidate.read_bytes(), media_type, cache="no-cache")

    def _base_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'",
        )

    def _send_json(self, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", cache="no-store")

    def _send_bytes(self, data: bytes, media_type: str, cache: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self._base_headers()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format_string: str, *args: Any) -> None:
        # Keep periodic API polling out of the terminal; report actual HTTP errors.
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            super().log_message(format_string, *args)


class PendantHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: PendantState) -> None:
        super().__init__(address, PendantRequestHandler)
        self.pendant_state = state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8088, help="HTTP port (default: 8088)")
    parser.add_argument("--camera-topic", default="/wrist_rgbd/image", help="ROS image topic")
    parser.add_argument("--demo", action="store_true", help="Use animated sample data instead of ROS 2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = PendantState(source="demo" if args.demo else "ros2")
    if args.demo:
        bridge: DemoBridge | RosPendantBridge = DemoBridge(state)
    else:
        try:
            import rclpy
        except ImportError as exc:
            raise SystemExit(
                "ROS 2 Python packages are unavailable. Source /opt/ros/jazzy/setup.bash or use --demo."
            ) from exc
        rclpy.init(args=None)
        bridge = RosPendantBridge(state, args.camera_topic)

    server = PendantHTTPServer((args.host, args.port), state)
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    bridge.start()
    print(f"UR10e live pendant (read-only): http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
