#!/usr/bin/env python3
"""Gazebo-only Steps 1-2 bringup for the opt-in PGI-140-80 simulation.

This wrapper reuses the established UR Gazebo launch and robot description. It
does not start MoveIt, perception, ur_robot_driver, or any real hardware node.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    world_file = LaunchConfiguration("world_file")
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz")
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PROJECT_ROOT / "launch" / "ur_sim_control_feeding.launch.py")
        ),
        launch_arguments={
            "ur_type": "ur10e",
            "description_file": str(
                PROJECT_ROOT / "urdf" / "ur_gz_feeding_markers.urdf.xacro"
            ),
            "controllers_file": str(
                PROJECT_ROOT / "config" / "pgi_140_80_sim_controllers.yaml"
            ),
            "use_pgi_gripper": "true",
            "activate_pgi_controller": "true",
            # MoveIt is a later step. Load the arm controller inactive so no
            # arm trajectory interface owns or moves the simulated arm.
            "activate_joint_controller": "false",
            "initial_joint_controller": "joint_trajectory_controller",
            "launch_rviz": launch_rviz,
            "rviz_config_file": str(PROJECT_ROOT / "config" / "pgi_140_80.rviz"),
            "gazebo_gui": gazebo_gui,
            "world_file": world_file,
            "robot_base_x": "0.0",
            "robot_base_y": "0.0",
            "robot_base_z": "0.0",
            "camera_mount_xyz": camera_mount_xyz,
            "camera_mount_rpy": camera_mount_rpy,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="92",
                description="Isolated ROS domain for PGI simulation; never use the live robot domain.",
            ),
            DeclareLaunchArgument(
                "gazebo_gui", default_value="true", description="Open the Gazebo GUI."
            ),
            DeclareLaunchArgument(
                "launch_rviz", default_value="true", description="Open the PGI RViz view."
            ),
            DeclareLaunchArgument(
                "world_file",
                default_value="empty.sdf",
                description="Gazebo world path or a world from the Gazebo collection.",
            ),
            DeclareLaunchArgument(
                "camera_mount_xyz",
                default_value="0 -0.065 0.020",
                description="Provisional interposer-to-camera translation in metres.",
            ),
            DeclareLaunchArgument(
                "camera_mount_rpy",
                default_value="0 0 0",
                description="Provisional interposer-to-camera rotation in radians.",
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("GZ_PARTITION", ["pgi_140_80_domain_", ros_domain_id]),
            base_launch,
        ]
    )
