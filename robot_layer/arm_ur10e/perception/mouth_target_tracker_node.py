#!/usr/bin/env python3
"""Read-only ROS adapter for :class:`MouthTargetTracker`.

This node publishes tracking state only.  It has no MoveIt or controller
clients and cannot command robot motion.
"""

from __future__ import annotations

import json
import time
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .mouth_target_tracker import MouthTargetTracker


class MouthTargetTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("mouth_target_tracker")
        # The calibrated output is in base_link; allow a larger tracked-target
        # displacement while retaining stale-data and perception validity gates.
        self.tracker = MouthTargetTracker(replan_distance_m=0.35)
        self._last_pose_monotonic = None
        self.pose_pub = self.create_publisher(PoseStamped, "/tracked_mouth_pose", 10)
        self.status_pub = self.create_publisher(String, "/mouth_tracking/status", 10)
        self.create_subscription(PoseStamped, "/detected_mouth_pose", self._callback, 10)
        self.create_subscription(String, "/mouth_detection/status", self._perception_status_callback, 10)
        self.create_service(Trigger, "/mouth_tracking/reset", self._reset_callback)

    # Reset only the read-only tracker reference; this never affects the robot.
    def _reset_callback(self, request, response):
        del request
        self.tracker.begin_session()
        response.success = True
        response.message = "tracking session reset"
        return response

    # Clear the reference immediately when perception loses the face.
    def _perception_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if payload.get("detected") is False:
            self.tracker.begin_session()

    def _callback(self, message: PoseStamped) -> None:
        now = time.monotonic()
        # A new observation after a real perception gap starts a new session.
        # This prevents an old ABORTED reference from contaminating a later
        # face reacquisition while retaining the large-displacement guard
        # during continuous tracking.
        if (self._last_pose_monotonic is not None and
                now - self._last_pose_monotonic > self.tracker.lost_timeout_sec):
            self.tracker.begin_session()
        self._last_pose_monotonic = now
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        result = self.tracker.update(
            [message.pose.position.x, message.pose.position.y, message.pose.position.z],
            timestamp_sec=stamp,
        )
        status = String()
        status.data = json.dumps({"state": result.state.value, "target_id": result.target_id,
                                  "age_sec": result.age_sec, "displacement_m": result.displacement_m,
                                  "velocity_mps": result.velocity.tolist()})
        self.status_pub.publish(status)
        if result.state.value not in {"LOST", "ABORTED"}:
            output = PoseStamped()
            output.header = message.header
            output.pose.position.x, output.pose.position.y, output.pose.position.z = result.position.tolist()
            output.pose.orientation.w = 1.0
            self.pose_pub.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MouthTargetTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
