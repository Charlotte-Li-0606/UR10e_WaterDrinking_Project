#!/usr/bin/env python3
"""Read-only UR External Control state monitor.

This diagnostic creates subscriptions and a ListControllers client only.  It
does not publish, switch controllers, call the dashboard, or create actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from ur_dashboard_msgs.msg import RobotMode, SafetyMode


RUNNING = 7
NORMAL = 1
EXPECTED_JOINTS = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
}


def _process_inventory() -> dict[str, list[dict[str, Any]]]:
    patterns = {
        "driver_launch": "ur_robot_driver ur_control.launch.py",
        "controller_manager": "/controller_manager/ros2_control_node",
        "move_group": "/moveit_ros_move_group/move_group",
        "realsense": "/realsense2_camera/realsense2_camera_node",
        "perception": "mouth_perception_node.py",
    }
    inventory: dict[str, list[dict[str, Any]]] = {name: [] for name in patterns}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
        except (OSError, UnicodeError):
            continue
        for name, pattern in patterns.items():
            if pattern in cmdline:
                inventory[name].append(
                    {
                        "pid": int(entry.name),
                        "state": stat_fields[2] if len(stat_fields) > 2 else None,
                        "command": cmdline,
                    }
                )
    return inventory


def _cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


class Monitor(Node):
    def __init__(self) -> None:
        super().__init__("external_control_read_only_monitor")
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.last_joint_received: float | None = None
        self.last_joint_names: set[str] = set()
        self.robot_mode: int | None = None
        self.safety_mode: int | None = None
        self.program_running: bool | None = None
        self.speed_scaling: float | None = None
        self.create_subscription(JointState, "/joint_states", self._joint, joint_qos)
        self.create_subscription(
            RobotMode, "/io_and_status_controller/robot_mode", lambda msg: setattr(self, "robot_mode", int(msg.mode)), state_qos
        )
        self.create_subscription(
            SafetyMode, "/io_and_status_controller/safety_mode", lambda msg: setattr(self, "safety_mode", int(msg.mode)), state_qos
        )
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            lambda msg: setattr(self, "program_running", bool(msg.data)),
            state_qos,
        )
        self.create_subscription(
            Float64,
            "/speed_scaling_state_broadcaster/speed_scaling",
            lambda msg: setattr(self, "speed_scaling", float(msg.data)),
            state_qos,
        )
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")

    def _joint(self, message: JointState) -> None:
        self.last_joint_received = time.monotonic()
        self.last_joint_names = set(message.name)

    def controller_state(self) -> dict[str, Any]:
        if not self.controllers.wait_for_service(timeout_sec=1.0):
            return {"available": False, "scaled_joint_trajectory_controller": None}
        future = self.controllers.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return {"available": False, "scaled_joint_trajectory_controller": None}
        states = {item.name: item.state for item in future.result().controller}
        return {
            "available": True,
            "scaled_joint_trajectory_controller": states.get("scaled_joint_trajectory_controller"),
        }

    def sample(self, previous_cpu: tuple[int, int]) -> tuple[dict[str, Any], tuple[int, int]]:
        rclpy.spin_once(self, timeout_sec=0.2)
        now = time.monotonic()
        controller = self.controller_state()
        current_cpu = _cpu_times()
        delta_total = current_cpu[0] - previous_cpu[0]
        delta_idle = current_cpu[1] - previous_cpu[1]
        cpu_percent = None if delta_total <= 0 else 100.0 * (delta_total - delta_idle) / delta_total
        joint_age = None if self.last_joint_received is None else now - self.last_joint_received
        processes = _process_inventory()
        return (
            {
                "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_monotonic_sec": now,
                "robot_mode": self.robot_mode,
                "robot_mode_running": self.robot_mode == RUNNING,
                "safety_mode": self.safety_mode,
                "safety_mode_normal": self.safety_mode == NORMAL,
                "robot_program_running": self.program_running,
                "speed_slider_percent": self.speed_scaling,
                "controller": controller,
                "joint_states_age_sec": joint_age,
                "joint_states_complete": EXPECTED_JOINTS.issubset(self.last_joint_names),
                "process_counts": {name: len(items) for name, items in processes.items()},
                "processes": processes,
                "system_cpu_percent": cpu_percent,
                "load_average_1m_5m_15m": list(os.getloadavg()),
            },
            current_cpu,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.interval <= 0.0:
        parser.error("duration and interval must be positive")

    rclpy.init()
    node = Monitor()
    samples: list[dict[str, Any]] = []
    trigger: dict[str, Any] | None = None
    started = time.monotonic()
    previous_cpu = _cpu_times()
    try:
        while True:
            sample, previous_cpu = node.sample(previous_cpu)
            sample["elapsed_sec"] = round(time.monotonic() - started, 3)
            sample.pop("elapsed_monotonic_sec", None)
            samples.append(sample)
            controller_state = sample["controller"].get("scaled_joint_trajectory_controller")
            duplicate = sample["process_counts"]["driver_launch"] != 1 or sample["process_counts"]["controller_manager"] != 1
            if sample["robot_mode"] is not None and not sample["robot_mode_running"]:
                trigger = {"reason": "robot_mode_not_running", "sample": sample}
            elif sample["robot_program_running"] is False:
                trigger = {"reason": "robot_program_not_running", "sample": sample}
            elif controller_state is not None and controller_state != "active":
                trigger = {"reason": "scaled_controller_not_active", "sample": sample}
            elif duplicate:
                trigger = {"reason": "driver_or_controller_manager_process_count_changed", "sample": sample}
            if trigger is not None or time.monotonic() - started >= args.duration:
                break
            deadline = min(started + args.duration, time.monotonic() + args.interval)
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=min(0.2, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    report = {
        "label": args.label,
        "requested_duration_sec": args.duration,
        "actual_duration_sec": round(time.monotonic() - started, 3),
        "no_motion_contract": "subscriptions and ListControllers only; no publishers, dashboard calls, controller switches, or actions",
        "trigger": trigger,
        "samples": samples,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(report_text + "\n", encoding="utf-8")
    print(report_text, flush=True)
    return 2 if trigger is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
