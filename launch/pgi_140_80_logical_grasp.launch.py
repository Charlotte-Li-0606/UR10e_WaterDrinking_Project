#!/usr/bin/env python3
"""Visible Stage-4 PGI logical-grasp simulation.

The launch remains inert by default (``demo_mode:=none``). ``plan`` runs the
feasibility and ownership preflight without controller activation. ``execute``
requires the simulation-only runner's explicit confirmation flags and is still
refused on ROS domain 0.
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
        str(PROJECT_ROOT / "scripts" / "pgi_logical_grasp_demo.py"),
    ]
    if mode == "execute":
        command.extend(["--execute-sim", "--confirm-simulation"])
    command.extend(
        [
            "--ros-args",
            "--params-file",
            str(PROJECT_ROOT / "config" / "pgi_logical_grasp.yaml"),
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
            "launch_camera_view": LaunchConfiguration("launch_camera_view"),
            "activate_arm_controller": "false",
            # Detect from the established camera-ready start.  The logical
            # runner then transfers to the side-ready pose using Cartesian
            # paths only; no special initial joint state or OMPL is needed.
            "pgi_logical_grasp_start": "false",
            # Preserve the provisional camera pose already reviewed in RViz.
            # It remains simulation-only and never alters the real D435i
            # calibration.
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
                default_value="92",
                description="Isolated nonzero ROS domain; domain 0 is refused.",
            ),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_camera_view", default_value="true"),
            DeclareLaunchArgument("cup_x", default_value="0.481542"),
            DeclareLaunchArgument("cup_y", default_value="0.208414"),
            DeclareLaunchArgument("cup_z", default_value="0.0"),
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
