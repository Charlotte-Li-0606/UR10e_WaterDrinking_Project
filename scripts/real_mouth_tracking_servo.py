#!/usr/bin/env python3
"""Guarded real-UR10e relative mouth tracking through MoveIt Servo.

This is independent from the successful one-shot feed-water entrypoint.  It is
disarmed by default and requires the existing real-motion environment gate and
two explicit command-line confirmations before it can publish a twist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import ServoStatus
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, Float64, String
from tf2_ros import Buffer, TransformListener
from ur_dashboard_msgs.msg import RobotMode, SafetyMode

from robot_layer.arm_ur10e.control.motion_backend import MotionRequest
from robot_layer.arm_ur10e.control.relative_tracking import RelativeTrackingSession
from robot_layer.arm_ur10e.control.ros_servo_backend import RosServoCommandSink


class RealMouthTrackingServo(Node):
    """Lock a safe pre-mouth pose, then follow only small mouth displacement."""

    def __init__(self, *, execute: bool, max_duration_sec: float) -> None:
        super().__init__("real_mouth_tracking_servo")
        self.execute = bool(execute)
        self.max_duration_sec = float(max_duration_sec)
        self.started_monotonic = time.monotonic()
        self.last_target_monotonic = None
        self.last_controller_monotonic = None
        self.last_update_monotonic = None
        self.tracking_state = "SEARCHING"
        self.robot_mode = None
        self.safety_mode = None
        self.servo_status = None
        self.speed_slider_percent = None
        self.robot_program_running = None
        self.halted_reason = None
        self.last_wait_reason = None
        self.latest_mouth = None
        self.command_count = 0
        self.maximum_command_speed_mps = 0.0
        self.maximum_mouth_displacement_m = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.session = RelativeTrackingSession()
        self.sink = RosServoCommandSink(self, max_linear_mps=0.02,
                                        max_angular_rps=0.0, armed=self.execute)
        latched_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "/tracked_mouth_pose", self._mouth_cb, 10)
        self.create_subscription(String, "/mouth_tracking/status", self._tracking_status_cb, 10)
        self.create_subscription(ServoStatus, "/servo_node/status", self._servo_status_cb, 10)
        self.create_subscription(RobotMode, "/io_and_status_controller/robot_mode",
                                 self._robot_mode_cb, latched_state_qos)
        self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode",
                                 self._safety_mode_cb, latched_state_qos)
        self.create_subscription(Float64,
                                 "/speed_scaling_state_broadcaster/speed_scaling",
                                 self._speed_slider_cb, latched_state_qos)
        self.create_subscription(Bool,
                                 "/io_and_status_controller/robot_program_running",
                                 self._program_running_cb, latched_state_qos)
        self.create_subscription(JointTrajectoryControllerState,
                                 "/scaled_joint_trajectory_controller/controller_state",
                                 self._controller_cb, latched_state_qos)
        self.timer = self.create_timer(0.05, self._tick)

    def _tool_position(self):
        transform = self.tf_buffer.lookup_transform(
            "base_link", "tool0", Time(), timeout=Duration(seconds=0.10)
        )
        value = transform.transform.translation
        return (float(value.x), float(value.y), float(value.z))

    def _mouth_cb(self, msg: PoseStamped) -> None:
        if msg.header.frame_id.strip().lstrip("/") != "base_link":
            self._halt("mouth_frame_not_base_link")
            return
        self.latest_mouth = (float(msg.pose.position.x), float(msg.pose.position.y),
                             float(msg.pose.position.z))
        self.last_target_monotonic = time.monotonic()

    def _tracking_status_cb(self, msg: String) -> None:
        try:
            self.tracking_state = str(json.loads(msg.data).get("state", "SEARCHING"))
        except (TypeError, ValueError):
            self._halt("invalid_tracking_status")

    def _servo_status_cb(self, msg: ServoStatus) -> None:
        self.servo_status = int(msg.code)
        if msg.code != ServoStatus.NO_WARNING:
            self._halt(f"servo_status:{msg.code}:{msg.message}")

    def _robot_mode_cb(self, msg: RobotMode) -> None:
        self.robot_mode = int(msg.mode)
        if self.session.locked and self.robot_mode != RobotMode.RUNNING:
            self._halt(f"robot_mode:{self.robot_mode}")

    def _safety_mode_cb(self, msg: SafetyMode) -> None:
        self.safety_mode = int(msg.mode)
        if self.session.locked and self.safety_mode not in {SafetyMode.NORMAL, SafetyMode.REDUCED}:
            self._halt(f"safety_mode:{self.safety_mode}")

    def _controller_cb(self, _msg: JointTrajectoryControllerState) -> None:
        self.last_controller_monotonic = time.monotonic()

    def _speed_slider_cb(self, msg: Float64) -> None:
        self.speed_slider_percent = float(msg.data)
        if self.session.locked and not 0.0 < self.speed_slider_percent <= 10.0:
            self._halt(f"speed_slider_percent:{self.speed_slider_percent}")

    def _program_running_cb(self, msg: Bool) -> None:
        self.robot_program_running = bool(msg.data)
        if self.session.locked and not self.robot_program_running:
            self._halt("external_control_program_not_running")

    def _ready(self, now: float) -> str | None:
        if now - self.started_monotonic > self.max_duration_sec:
            return "tracking_duration_limit"
        if self.tracking_state in {"LOST", "ABORTED"}:
            return f"tracking_state:{self.tracking_state}"
        if self.latest_mouth is None or self.last_target_monotonic is None:
            return "no_tracked_mouth"
        if now - self.last_target_monotonic > 0.50:
            return "target_stale"
        if self.last_controller_monotonic is None or now - self.last_controller_monotonic > 0.50:
            return "controller_state_stale"
        if self.robot_mode != RobotMode.RUNNING:
            return "robot_not_running"
        if self.safety_mode not in {SafetyMode.NORMAL, SafetyMode.REDUCED}:
            return "safety_mode_not_ready"
        if self.speed_slider_percent is None or not 0.0 < self.speed_slider_percent <= 10.0:
            return "speed_slider_not_at_or_below_10_percent"
        if self.robot_program_running is not True:
            return "external_control_program_not_running"
        return None

    def _tick(self) -> None:
        if self.halted_reason is not None:
            return
        now = time.monotonic()
        reason = self._ready(now)
        if reason is not None:
            # Before the reference is locked, asynchronously arriving ROS
            # readiness inputs are allowed a startup window.  Once locked,
            # the same missing/stale input is an immediate stop.
            prelock_wait_reasons = {
                "no_tracked_mouth",
                "target_stale",
                "controller_state_stale",
                "robot_not_running",
                "safety_mode_not_ready",
                "speed_slider_not_at_or_below_10_percent",
                "external_control_program_not_running",
            }
            if self.session.locked or reason not in prelock_wait_reasons:
                self._halt(reason)
            elif reason != self.last_wait_reason:
                self.last_wait_reason = reason
                self.get_logger().info(f"Waiting for tracking readiness: {reason}")
            return
        try:
            tool = self._tool_position()
        except Exception as exc:
            self._halt(f"tool_tf_unavailable:{exc}")
            return
        if not self.session.locked:
            self.session.lock(self.latest_mouth, tool)
            self.last_update_monotonic = now
            self.last_wait_reason = None
            self.get_logger().info("Tracking reference locked; relative corrections enabled")
            return
        dt = now - (self.last_update_monotonic or now)
        self.last_update_monotonic = now
        command = self.session.update(self.latest_mouth, tool, dt_sec=dt)
        if not command.allowed:
            self._halt(command.reason)
            return
        request = MotionRequest(
            target_position=command.desired_tool_position_m,
            target_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            plan_only=not self.execute,
            reason="relative_mouth_tracking",
            linear_velocity_mps=command.linear_velocity_mps,
            angular_velocity_rps=command.angular_velocity_rps,
            preserve_orientation=True,
        )
        if self.execute:
            self.sink(request)
            self.command_count += 1
            self.maximum_command_speed_mps = max(
                self.maximum_command_speed_mps,
                math.sqrt(sum(value * value for value in command.linear_velocity_mps)),
            )
            self.maximum_mouth_displacement_m = max(
                self.maximum_mouth_displacement_m,
                command.mouth_displacement_m,
            )

    def _halt(self, reason: str) -> None:
        if self.halted_reason is not None:
            return
        self.halted_reason = str(reason)
        self.sink.halt()
        self.get_logger().error(f"Tracking halted: {self.halted_reason}")

    def report(self) -> dict:
        return {
            "reference_locked": self.session.locked,
            "execution_enabled": self.execute,
            "servo_command_count": self.command_count,
            "maximum_command_speed_mps": self.maximum_command_speed_mps,
            "maximum_mouth_displacement_m": self.maximum_mouth_displacement_m,
            "stop_reason": self.halted_reason,
        }

    def destroy_node(self):
        self.sink.halt()
        return super().destroy_node()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-real-motion", action="store_true")
    parser.add_argument("--max-duration", type=float, default=15.0)
    parsed, ros_args = parser.parse_known_args(argv)
    if parsed.execute:
        if os.environ.get("UR10E_ALLOW_REAL_EXECUTION") != "1":
            parser.error("real tracking requires UR10E_ALLOW_REAL_EXECUTION=1")
        if not parsed.confirm_real_motion:
            parser.error("real tracking requires --confirm-real-motion")
    if not math.isfinite(parsed.max_duration) or not 1.0 <= parsed.max_duration <= 30.0:
        parser.error("--max-duration must be between 1 and 30 seconds")
    return parsed, ros_args


def main(argv=None) -> None:
    parsed, ros_args = _parse_args(argv)
    rclpy.init(args=ros_args)
    node = RealMouthTrackingServo(execute=parsed.execute,
                                  max_duration_sec=parsed.max_duration)
    try:
        while rclpy.ok() and node.halted_reason is None:
            rclpy.spin_once(node, timeout_sec=0.10)
    except Exception:
        # ROS invalidates the context when an external timeout/SIGTERM stops a
        # non-motion diagnostic.  Preserve genuine runtime failures.
        if rclpy.ok():
            raise
    finally:
        if node.halted_reason is None:
            node._halt("process_shutdown")
        print(json.dumps(node.report(), sort_keys=True), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
