#!/usr/bin/env python3
"""Exercise only the simulated PGI jaw through GripperCommand.

The test deliberately refuses ROS domain 0 and requires Gazebo simulation
clock data before sending a goal. It never addresses an arm controller.
"""

import argparse
import json
import os
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


LEFT_JOINT = "pgi_left_finger_joint"
RIGHT_JOINT = "pgi_right_finger_joint"
ACTION_NAME = "/pgi_gripper_controller/gripper_cmd"


class PgiGazeboVerifier(Node):
    def __init__(self) -> None:
        super().__init__("pgi_140_80_gazebo_control_verifier")
        self.have_clock = False
        self.first_clock_ns: int | None = None
        self.clock_advanced = False
        self.positions: dict[str, float] = {}
        self.maximum_symmetry_error = 0.0
        self.create_subscription(Clock, "/clock", self._clock_callback, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.client = ActionClient(self, GripperCommand, ACTION_NAME)

    def _clock_callback(self, message: Clock) -> None:
        self.have_clock = True
        clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self.first_clock_ns is None:
            self.first_clock_ns = clock_ns
        elif clock_ns > self.first_clock_ns:
            self.clock_advanced = True

    def _joint_callback(self, message: JointState) -> None:
        self.positions.update(zip(message.name, message.position))
        if LEFT_JOINT in self.positions and RIGHT_JOINT in self.positions:
            error = abs(self.positions[LEFT_JOINT] - self.positions[RIGHT_JOINT])
            self.maximum_symmetry_error = max(self.maximum_symmetry_error, error)

    def jaw_positions(self) -> tuple[float, float]:
        return self.positions[LEFT_JOINT], self.positions[RIGHT_JOINT]

    def wait_for_simulation(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.have_clock
                and self.clock_advanced
                and LEFT_JOINT in self.positions
                and RIGHT_JOINT in self.positions
            ):
                return True
        return False

    def command(self, position: float, maximum_effort: float, timeout: float) -> dict:
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = maximum_effort

        started = time.monotonic()
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done() or send_future.result() is None:
            raise RuntimeError("Timed out while sending GripperCommand goal")

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            raise RuntimeError("GripperCommand goal was rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done() or result_future.result() is None:
            raise RuntimeError("Timed out while waiting for GripperCommand result")

        result_response = result_future.result()
        result = result_response.result
        rclpy.spin_once(self, timeout_sec=0.1)
        left, right = self.jaw_positions()
        return {
            "requested_position_m": position,
            "status": result_response.status,
            "succeeded": result_response.status == GoalStatus.STATUS_SUCCEEDED,
            "reached_goal": bool(result.reached_goal),
            "stalled": bool(result.stalled),
            "reported_position_m": float(result.position),
            "reported_effort_n": float(result.effort),
            "left_position_m": left,
            "right_position_m": right,
            "elapsed_wall_s": time.monotonic() - started,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify simulated PGI close/open motion on a nonzero ROS domain."
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-effort", type=float, default=140.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        numeric_domain = int(domain_id)
    except ValueError:
        print(f"Refusing invalid ROS_DOMAIN_ID={domain_id!r}", file=sys.stderr)
        return 2
    if numeric_domain == 0:
        print(
            "Refusing ROS domain 0. Set ROS_DOMAIN_ID to the isolated PGI "
            "simulation domain (the launch default is 92).",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    verifier = PgiGazeboVerifier()
    try:
        if not verifier.wait_for_simulation(args.timeout):
            raise RuntimeError(
                "No advancing Gazebo clock and fresh PGI joint states were received"
            )
        if not verifier.client.wait_for_server(timeout_sec=args.timeout):
            raise RuntimeError(f"Action server {ACTION_NAME} is unavailable")

        initial_left, initial_right = verifier.jaw_positions()
        closed = verifier.command(0.0, args.max_effort, args.timeout)
        opened = verifier.command(0.040, args.max_effort, args.timeout)
        total_stroke = (
            opened["left_position_m"]
            + opened["right_position_m"]
            - closed["left_position_m"]
            - closed["right_position_m"]
        )
        success = all(
            result["succeeded"] and result["reached_goal"] and not result["stalled"]
            for result in (closed, opened)
        )
        success = success and total_stroke >= 0.078
        success = success and verifier.maximum_symmetry_error <= 0.0001

        report = {
            "simulation_domain": numeric_domain,
            "action": ACTION_NAME,
            "initial_positions_m": {"left": initial_left, "right": initial_right},
            "close": closed,
            "open": opened,
            "measured_total_stroke_m": total_stroke,
            "maximum_symmetry_error_m": verifier.maximum_symmetry_error,
            "success": success,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if success else 1
    except RuntimeError as error:
        print(f"PGI Gazebo verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
