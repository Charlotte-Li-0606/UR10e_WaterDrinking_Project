#!/usr/bin/env python3
"""Combined visible Gazebo + plan-only MoveIt launch for PGI Stage 3."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    world_file = LaunchConfiguration("world_file")
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz")
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy")
    spawn_cup = LaunchConfiguration("spawn_cup")
    cup_x = LaunchConfiguration("cup_x")
    cup_y = LaunchConfiguration("cup_y")
    cup_z = LaunchConfiguration("cup_z")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PROJECT_ROOT / "launch" / "pgi_140_80_gazebo.launch.py")
        ),
        launch_arguments={
            "ros_domain_id": ros_domain_id,
            "gazebo_gui": gazebo_gui,
            "launch_rviz": "false",
            "world_file": world_file,
            "camera_mount_xyz": camera_mount_xyz,
            "camera_mount_rpy": camera_mount_rpy,
        }.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PROJECT_ROOT / "launch" / "pgi_140_80_moveit.launch.py")
        ),
        launch_arguments={
            "ros_domain_id": ros_domain_id,
            "launch_rviz": launch_rviz,
            "use_sim_time": "true",
            "allow_trajectory_execution": "false",
            "camera_mount_xyz": camera_mount_xyz,
            "camera_mount_rpy": camera_mount_rpy,
        }.items(),
    )

    gazebo_cup = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        condition=IfCondition(spawn_cup),
        arguments=[
            "-file",
            str(PROJECT_ROOT / "models" / "pgi_staging_cup" / "model.sdf"),
            "-name",
            "pgi_staging_cup",
            "-allow_renaming",
            "false",
            "-x",
            cup_x,
            "-y",
            cup_y,
            "-z",
            cup_z,
        ],
    )

    moveit_cup = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            str(PROJECT_ROOT / "scripts" / "publish_pgi_140_80_cup_scene.py"),
            "--x",
            cup_x,
            "--y",
            cup_y,
            "--z",
            cup_z,
        ],
        output="screen",
        condition=IfCondition(spawn_cup),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="92",
                description="Isolated ROS domain for Gazebo and MoveIt.",
            ),
            DeclareLaunchArgument(
                "gazebo_gui", default_value="true", description="Open Gazebo GUI."
            ),
            DeclareLaunchArgument(
                "launch_rviz", default_value="true", description="Open MoveIt RViz."
            ),
            DeclareLaunchArgument(
                "world_file", default_value="empty.sdf", description="Gazebo world."
            ),
            DeclareLaunchArgument(
                "camera_mount_xyz", default_value="0 -0.065 0.020"
            ),
            DeclareLaunchArgument("camera_mount_rpy", default_value="0 0 0"),
            DeclareLaunchArgument(
                "spawn_cup",
                default_value="true",
                description="Spawn the independent Stage-3 cup in Gazebo and MoveIt.",
            ),
            # Reuse the saved X/Y location but place the independent cup base
            # on the Gazebo ground plane. Both simulators receive one pose.
            DeclareLaunchArgument("cup_x", default_value="0.481542"),
            DeclareLaunchArgument("cup_y", default_value="0.208414"),
            DeclareLaunchArgument("cup_z", default_value="0.0"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable(
                "GZ_PARTITION", ["pgi_140_80_domain_", ros_domain_id]
            ),
            gazebo,
            TimerAction(period=7.0, actions=[gazebo_cup]),
            # Let Gazebo publish its first clock and joint states before
            # MoveIt's current-state monitor begins checking the model.
            TimerAction(period=8.0, actions=[moveit]),
            # The publisher also waits for the service, so a slow MoveGroup
            # startup cannot silently omit the collision object.
            TimerAction(period=9.0, actions=[moveit_cup]),
        ]
    )
