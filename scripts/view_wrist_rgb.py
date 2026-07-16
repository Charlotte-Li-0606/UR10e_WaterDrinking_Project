#!/usr/bin/env python3
"""Display the live wrist RGB image published by the Gazebo ROS 2 bridge."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class WristRgbViewer(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("wrist_rgb_viewer")
        self._frame: np.ndarray | None = None
        self._stamp = "waiting for image"
        self.create_subscription(Image, topic, self._on_image, 10)

    def _on_image(self, message: Image) -> None:
        if message.encoding.lower() not in {"rgb8", "bgr8"}:
            self.get_logger().warning(f"Ignoring unsupported image encoding: {message.encoding}")
            return
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        frame = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
        self._frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if message.encoding.lower() == "rgb8" else frame.copy()
        self._stamp = f"{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}"

    def show(self, window_name: str) -> None:
        frame = self._frame
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for /wrist_rgbd/image ...", (85, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            frame = frame.copy()
        cv2.putText(frame, f"ROS 2 wrist RGB  |  {self._stamp}  |  q / Esc: close", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3)
        cv2.putText(frame, f"ROS 2 wrist RGB  |  {self._stamp}  |  q / Esc: close", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
        cv2.imshow(window_name, frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/wrist_rgbd/image", help="ROS 2 Image topic to display")
    args = parser.parse_args()

    rclpy.init()
    viewer = WristRgbViewer(args.topic)
    window_name = "UR10e Wrist RGB Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.05)
            viewer.show(window_name)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            time.sleep(0.01)
    finally:
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
