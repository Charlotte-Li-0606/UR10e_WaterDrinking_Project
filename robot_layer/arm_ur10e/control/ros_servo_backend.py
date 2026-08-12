"""Disarmed ROS 2 sink for MoveIt Servo Cartesian corrections.

The class only publishes when explicitly armed by its caller.  It is not
constructed by the existing feed-water path and never switches controllers.
"""

from __future__ import annotations

from geometry_msgs.msg import TwistStamped


class RosServoCommandSink:
    """Publish bounded base-frame twists and send explicit zero halts."""

    def __init__(self, node, *, topic="/servo_node/delta_twist_cmds",
                 max_linear_mps=0.02, max_angular_rps=0.10, armed=False):
        self.node = node
        self.publisher = node.create_publisher(TwistStamped, topic, 10)
        self.max_linear_mps = float(max_linear_mps)
        self.max_angular_rps = float(max_angular_rps)
        self.armed = bool(armed)

    def __call__(self, request) -> None:
        if not self.armed:
            raise RuntimeError("ROS Servo command sink is disarmed")
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        linear = request.linear_velocity_mps
        angular = request.angular_velocity_rps
        msg.twist.linear.x = max(-self.max_linear_mps, min(self.max_linear_mps, float(linear[0])))
        msg.twist.linear.y = max(-self.max_linear_mps, min(self.max_linear_mps, float(linear[1])))
        msg.twist.linear.z = max(-self.max_linear_mps, min(self.max_linear_mps, float(linear[2])))
        msg.twist.angular.x = max(-self.max_angular_rps, min(self.max_angular_rps, float(angular[0])))
        msg.twist.angular.y = max(-self.max_angular_rps, min(self.max_angular_rps, float(angular[1])))
        msg.twist.angular.z = max(-self.max_angular_rps, min(self.max_angular_rps, float(angular[2])))
        self.publisher.publish(msg)

    def halt(self) -> None:
        # Servo also times out, but explicit zero twists make the stop request
        # immediate and deterministic before the sink is disarmed.
        if self.armed:
            self.publish_zero()
        self.armed = False

    def publish_zero(self) -> None:
        """Hold position without surrendering command ownership."""
        if not self.armed:
            raise RuntimeError("ROS Servo command sink is disarmed")
        for _ in range(4):
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            self.publisher.publish(msg)
