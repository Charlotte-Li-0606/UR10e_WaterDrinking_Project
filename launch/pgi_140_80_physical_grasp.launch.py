#!/usr/bin/env python3
"""Visible, isolated Stage-5 PGI native-contact grasp experiment.

The launch is inert by default.  It uses a separate dynamic cup, world, contact
variant, and controller YAML so Stage 4 remains reproducible.  It never starts
ur_robot_driver, RS485, or any real-hardware node.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _schedule_demo(context):
    mode = LaunchConfiguration("demo_mode").perform(context).strip().lower()
    if mode == "none":
        return []
    if mode not in {"plan", "execute"}:
        raise RuntimeError("demo_mode must be one of: none, plan, execute")
    command = [
        "/usr/bin/python3",
        str(PROJECT_ROOT / "scripts" / "pgi_physical_grasp_demo.py"),
    ]
    if mode == "execute":
        command.extend(["--execute-sim", "--confirm-simulation"])
    command.extend(
        [
            "--ros-args",
            "--params-file",
            str(PROJECT_ROOT / "config" / "pgi_physical_grasp.yaml"),
        ]
    )
    return [
        TimerAction(
            period=18.0,
            actions=[ExecuteProcess(cmd=command, output="screen")],
        )
    ]


def generate_launch_description():
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    combined = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PROJECT_ROOT / "launch" / "pgi_140_80_gazebo_moveit.launch.py")
        ),
        launch_arguments={
            "ros_domain_id": ros_domain_id,
            "gazebo_gui": LaunchConfiguration("gazebo_gui"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_cup_perception": LaunchConfiguration(
                "launch_cup_perception"
            ),
            "launch_camera_view": LaunchConfiguration("launch_camera_view"),
            "launch_grasp_anything": LaunchConfiguration("launch_grasp_anything"),
            "launch_grasp_anything_view": LaunchConfiguration(
                "launch_grasp_anything_view"
            ),
            "activate_arm_controller": "false",
            "pgi_logical_grasp_start": "false",
            "pgi_contact_physics": "true",
            "controllers_file": str(
                PROJECT_ROOT / "config" / "pgi_140_80_physical_controllers.yaml"
            ),
            "world_file": str(
                PROJECT_ROOT / "worlds" / "pgi_140_80_physical_grasp.sdf"
            ),
            "cup_model_file": str(
                PROJECT_ROOT / "models" / "pgi_physical_cup" / "model.sdf"
            ),
            "camera_mount_xyz": "0 0.085 0.030",
            "camera_mount_rpy": "0 -1.57079632679 0",
            "cup_x": LaunchConfiguration("cup_x"),
            "cup_y": LaunchConfiguration("cup_y"),
            "cup_z": LaunchConfiguration("cup_z"),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="106",
                description="Isolated nonzero ROS domain; domain 0 is refused.",
            ),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_cup_perception", default_value="true"),
            DeclareLaunchArgument("launch_camera_view", default_value="true"),
            DeclareLaunchArgument("launch_grasp_anything", default_value="false"),
            DeclareLaunchArgument(
                "launch_grasp_anything_view", default_value="false"
            ),
            DeclareLaunchArgument("cup_x", default_value="0.481542"),
            DeclareLaunchArgument("cup_y", default_value="0.208414"),
            DeclareLaunchArgument("cup_z", default_value="0.001"),
            DeclareLaunchArgument(
                "demo_mode",
                default_value="none",
                description="none, plan, or execute; execution affects Gazebo only.",
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable(
                "GZ_PARTITION", ["pgi_140_80_domain_", ros_domain_id]
            ),
            OpaqueFunction(function=_schedule_demo),
            combined,
        ]
    )
