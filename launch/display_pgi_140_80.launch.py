#!/usr/bin/env python3
"""RViz-only display of the existing UR description with opt-in PGI model.

This launch file starts no Gazebo process, ros2_control controller, MoveIt
process, robot driver, action client, or hardware communication process.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_FILE = PROJECT_ROOT / "urdf" / "ur_gz_feeding_markers.urdf.xacro"
RVIZ_CONFIG_FILE = PROJECT_ROOT / "config" / "pgi_140_80.rviz"


def generate_launch_description():
    use_pgi_gripper = LaunchConfiguration("use_pgi_gripper")
    use_joint_state_gui = LaunchConfiguration("use_joint_state_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz")
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy")

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            str(DESCRIPTION_FILE),
            " name:=ur ur_type:=ur10e tf_prefix:=''",
            " use_pgi_gripper:=",
            use_pgi_gripper,
            " camera_mount_xyz:='",
            camera_mount_xyz,
            "' camera_mount_rpy:='",
            camera_mount_rpy,
            "' robot_base_x:=0 robot_base_y:=0 robot_base_z:=0",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_pgi_gripper",
                default_value="true",
                description=(
                    "Opt into the provisional PGI-140-80 assembly. The canonical "
                    "Xacro itself defaults this option to false."
                ),
            ),
            DeclareLaunchArgument(
                "camera_mount_xyz",
                default_value="0 0.085 0.030",
                description="Provisional interposer-to-D435i mount translation in metres.",
            ),
            DeclareLaunchArgument(
                "camera_mount_rpy",
                default_value="0 -1.57079632679 0",
                description="Provisional interposer-to-D435i mount roll/pitch/yaw in radians.",
            ),
            DeclareLaunchArgument(
                "use_joint_state_gui",
                default_value="true",
                description="Use sliders to move the independent left jaw; right jaw mimics it.",
            ),
            DeclareLaunchArgument(
                "launch_rviz", default_value="true", description="Start RViz2."
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[robot_description],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="pgi_joint_state_publisher_gui",
                output="screen",
                parameters=[robot_description],
                condition=IfCondition(use_joint_state_gui),
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="pgi_joint_state_publisher",
                output="screen",
                parameters=[robot_description],
                condition=UnlessCondition(use_joint_state_gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="pgi_description_rviz",
                output="screen",
                arguments=["-d", str(RVIZ_CONFIG_FILE)],
                condition=IfCondition(launch_rviz),
            ),
        ]
    )
