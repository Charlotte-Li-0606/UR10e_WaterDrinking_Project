#!/usr/bin/env python3
"""Add the Stage-3 cup to MoveIt's world without attaching or executing it."""

import argparse
import json
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from shape_msgs.msg import SolidPrimitive


CUP_ID = "pgi_staging_cup"
CUP_HEIGHT = 0.205
CUP_RADIUS = 0.050
CUP_CENTER_Z = 0.1025
STRAW_HEIGHT = 0.070
STRAW_RADIUS = 0.004
STRAW_CENTER_Z = 0.240


def cylinder(height: float, radius: float) -> SolidPrimitive:
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.CYLINDER
    primitive.dimensions = [height, radius]
    return primitive


def upright_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


class CupScenePublisher(Node):
    def __init__(self) -> None:
        super().__init__("pgi_140_80_cup_scene_publisher")
        self.first_clock_ns: int | None = None
        self.clock_advanced = False
        self.create_subscription(Clock, "/clock", self._clock_callback, 10)
        self.apply_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.get_client = self.create_client(GetPlanningScene, "/get_planning_scene")

    def _clock_callback(self, message: Clock) -> None:
        clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        if self.first_clock_ns is None:
            self.first_clock_ns = clock_ns
        elif clock_ns > self.first_clock_ns:
            self.clock_advanced = True

    def wait_for_simulation(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.clock_advanced:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.clock_advanced:
            raise RuntimeError("Gazebo /clock is unavailable or not advancing")

    def wait_for_services(self, timeout: float) -> None:
        if not self.apply_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("MoveIt /apply_planning_scene is unavailable")
        if not self.get_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("MoveIt /get_planning_scene is unavailable")

    def call(self, client, request, timeout: float = 10.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Service call timed out: {client.srv_name}")
        return future.result()

    def apply_cup(self, frame_id: str, x: float, y: float, z: float) -> dict:
        collision = CollisionObject()
        collision.header.frame_id = frame_id
        collision.id = CUP_ID
        collision.operation = CollisionObject.ADD
        collision.primitives = [
            cylinder(CUP_HEIGHT, CUP_RADIUS),
            cylinder(STRAW_HEIGHT, STRAW_RADIUS),
        ]
        collision.primitive_poses = [
            upright_pose(x, y, z + CUP_CENTER_Z),
            upright_pose(x, y, z + STRAW_CENTER_Z),
        ]

        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects.append(collision)

        color = ObjectColor()
        color.id = CUP_ID
        color.color.r = 0.20
        color.color.g = 0.55
        color.color.b = 0.90
        color.color.a = 0.90
        request.scene.object_colors.append(color)

        if not self.call(self.apply_client, request).success:
            raise RuntimeError("MoveIt rejected the cup planning-scene update")

        verify = GetPlanningScene.Request()
        verify.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )
        objects = self.call(self.get_client, verify).scene.world.collision_objects
        cup = next((item for item in objects if item.id == CUP_ID), None)
        if cup is None or len(cup.primitives) != 2 or len(cup.primitive_poses) != 2:
            raise RuntimeError("Cup was not retained in the MoveIt planning scene")

        return {
            "id": CUP_ID,
            "frame": frame_id,
            "base_pose_xyz_m": [x, y, z],
            "cup": {"height_m": CUP_HEIGHT, "radius_m": CUP_RADIUS},
            "straw": {
                "height_m": STRAW_HEIGHT,
                "radius_m": STRAW_RADIUS,
                "center_z_m": STRAW_CENTER_Z,
            },
            "attached": False,
            "apply_planning_scene_success": True,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--x", type=float, default=0.481542)
    parser.add_argument("--y", type=float, default=0.208414)
    parser.add_argument("--z", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        if int(domain_id) == 0:
            raise RuntimeError("Refusing ROS domain 0; use the isolated PGI simulation")
    except ValueError as error:
        print(f"Invalid ROS_DOMAIN_ID={domain_id!r}", file=sys.stderr)
        return 2

    rclpy.init()
    node = CupScenePublisher()
    try:
        node.wait_for_simulation(15.0)
        node.wait_for_services(30.0)
        print(json.dumps(node.apply_cup(args.frame, args.x, args.y, args.z), indent=2))
        return 0
    except RuntimeError as error:
        print(f"PGI cup scene failed: {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
