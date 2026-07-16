#!/usr/bin/env python3
import argparse
import math
import sys
import time

import rclpy
import rclpy.time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
import tf2_ros


DEFAULT_FRAME = "base_link"
DEFAULT_GROUP = "ur_manipulator"
DEFAULT_LINK = "tool0"
DEFAULT_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Move the UR10e tool0 end-effector pose through MoveIt Cartesian "
            "planning and the ROS2 trajectory controller."
        )
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--absolute",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="absolute tool0 target position in base_link, meters",
    )
    target.add_argument(
        "--relative",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=[0.0, 0.0, -0.10],
        help="relative tool0 offset in meters; default: 0 0 -0.10",
    )
    parser.add_argument(
        "--orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        help="target quaternion; default keeps the current tool0 orientation",
    )
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--link", default=DEFAULT_LINK)
    parser.add_argument("--action", default=DEFAULT_ACTION)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--min-fraction", type=float, default=0.99)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--allow-collisions", action="store_true")
    return parser.parse_args()


class MoveToEndposeDemo(Node):
    def __init__(self, args):
        super().__init__("move_to_endpose_demo")
        self.args = args
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, self.args.action
        )

    def wait_until_ready(self):
        checks = [
            ("/compute_ik", self.ik_client.wait_for_service),
            ("/compute_cartesian_path", self.cartesian_client.wait_for_service),
            (self.args.action, self.trajectory_client.wait_for_server),
        ]
        for name, wait_fn in checks:
            if not wait_fn(timeout_sec=10.0):
                raise RuntimeError(f"{name} is not available")

    def current_tool_pose(self):
        deadline = time.time() + 8.0
        last_error = None
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.args.frame,
                    self.args.link,
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=1.0),
                )
                pose = Pose()
                pose.position.x = tf.transform.translation.x
                pose.position.y = tf.transform.translation.y
                pose.position.z = tf.transform.translation.z
                pose.orientation = tf.transform.rotation
                return pose
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not read {self.args.frame} -> {self.args.link}: {last_error}"
        )

    def build_target_pose(self, current):
        target = Pose()
        if self.args.absolute is not None:
            target.position.x = self.args.absolute[0]
            target.position.y = self.args.absolute[1]
            target.position.z = self.args.absolute[2]
        else:
            target.position.x = current.position.x + self.args.relative[0]
            target.position.y = current.position.y + self.args.relative[1]
            target.position.z = current.position.z + self.args.relative[2]

        if self.args.orientation is None:
            target.orientation = current.orientation
        else:
            target.orientation.x = self.args.orientation[0]
            target.orientation.y = self.args.orientation[1]
            target.orientation.z = self.args.orientation[2]
            target.orientation.w = self.args.orientation[3]
        return target

    def check_ik(self, target):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.args.group
        request.ik_request.ik_link_name = self.args.link
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = self.args.frame
        request.ik_request.pose_stamped.pose = target
        request.ik_request.avoid_collisions = not self.args.allow_collisions
        request.ik_request.timeout.sec = 2

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        if future.result() is None:
            raise RuntimeError("IK service call failed")
        return future.result().error_code

    def plan_cartesian_path(self, target):
        request = GetCartesianPath.Request()
        request.header.frame_id = self.args.frame
        request.group_name = self.args.group
        request.link_name = self.args.link
        request.waypoints = [target]
        request.max_step = self.args.max_step
        request.jump_threshold = 0.0
        request.avoid_collisions = not self.args.allow_collisions
        request.start_state.is_diff = True

        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.result() is None:
            raise RuntimeError("Cartesian path service call failed")
        return future.result()

    def slow_down_trajectory(self, trajectory):
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            raise RuntimeError("Planned trajectory has fewer than 2 points")

        duration = max(self.args.duration, 1.0)
        for index, point in enumerate(points):
            seconds = 0.5 + (duration - 0.5) * index / max(1, len(points) - 1)
            whole = int(seconds)
            point.time_from_start = Duration(
                sec=whole,
                nanosec=int((seconds - whole) * 1_000_000_000),
            )

    def execute_trajectory(self, trajectory):
        self.slow_down_trajectory(trajectory)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory.joint_trajectory

        send_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory controller rejected the goal")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self.args.duration + 10.0
        )
        if result_future.result() is None:
            raise RuntimeError("Trajectory execution timed out")
        return result_future.result().result

    def run(self):
        self.wait_until_ready()

        start = self.current_tool_pose()
        target = self.build_target_pose(start)

        print_pose("START_TOOL0", start)
        print_pose("TARGET_TOOL0", target)

        ik_code = self.check_ik(target)
        print(f"IK_RESULT error_code={ik_code.val} message=\"{ik_code.message}\"")
        if ik_code.val != 1:
            print("ENDPOSE_DEMO: FAIL target has no IK solution")
            return 2

        cartesian = self.plan_cartesian_path(target)
        point_count = len(cartesian.solution.joint_trajectory.points)
        print(
            "CARTESIAN_PATH "
            f"fraction={cartesian.fraction:.3f} "
            f"points={point_count} "
            f"error_code={cartesian.error_code.val}"
        )
        if cartesian.fraction < self.args.min_fraction or point_count < 2:
            print("ENDPOSE_DEMO: FAIL Cartesian path is incomplete")
            return 3

        if self.args.no_execute:
            print("ENDPOSE_DEMO: PLAN_ONLY PASS")
            return 0

        result = self.execute_trajectory(cartesian.solution)
        end = self.current_tool_pose()
        distance = pose_distance(end, target)

        print(f"EXECUTION_RESULT code={result.error_code} msg=\"{result.error_string}\"")
        print_pose("END_TOOL0", end)
        print(f"DISTANCE_TO_TARGET {distance:.4f} m")

        if result.error_code == 0 and distance < 0.01:
            print("ENDPOSE_DEMO: PASS")
            return 0

        print("ENDPOSE_DEMO: FAIL execution did not reach the target")
        return 4


def pose_distance(a, b):
    return math.sqrt(
        (a.position.x - b.position.x) ** 2
        + (a.position.y - b.position.y) ** 2
        + (a.position.z - b.position.z) ** 2
    )


def print_pose(label, pose):
    print(
        f"{label} xyz=({pose.position.x:.4f},{pose.position.y:.4f},{pose.position.z:.4f}) "
        f"quat=({pose.orientation.x:.4f},{pose.orientation.y:.4f},"
        f"{pose.orientation.z:.4f},{pose.orientation.w:.4f})"
    )


def main():
    args = parse_args()
    rclpy.init()
    node = MoveToEndposeDemo(args)
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"ENDPOSE_DEMO: ERROR {exc}", file=sys.stderr)
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
