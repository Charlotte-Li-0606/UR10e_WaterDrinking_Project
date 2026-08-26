#!/usr/bin/env python3
"""Combined visible Gazebo + plan-only MoveIt launch for PGI Stage 3."""

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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _schedule_moveit(context):
    """Capture top-level values before nested Gazebo arguments can change them."""
    captured = {
        name: LaunchConfiguration(name).perform(context)
        for name in (
            "ros_domain_id",
            "launch_rviz",
            "camera_mount_xyz",
            "camera_mount_rpy",
            "pgi_contact_physics",
        )
    }
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(PROJECT_ROOT / "launch" / "pgi_140_80_moveit.launch.py")
        ),
        launch_arguments={
            "ros_domain_id": captured["ros_domain_id"],
            "launch_rviz": captured["launch_rviz"],
            "use_sim_time": "true",
            "allow_trajectory_execution": "false",
            "camera_mount_xyz": captured["camera_mount_xyz"],
            "camera_mount_rpy": captured["camera_mount_rpy"],
            "pgi_contact_physics": captured["pgi_contact_physics"],
        }.items(),
    )
    return [TimerAction(period=8.0, actions=[moveit])]


def generate_launch_description():
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz")
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy")
    controllers_file = LaunchConfiguration("controllers_file")
    pgi_contact_physics = LaunchConfiguration("pgi_contact_physics")
    activate_arm_controller = LaunchConfiguration("activate_arm_controller")
    pgi_logical_grasp_start = LaunchConfiguration("pgi_logical_grasp_start")
    spawn_cup = LaunchConfiguration("spawn_cup")
    launch_cup_perception = LaunchConfiguration("launch_cup_perception")
    launch_camera_view = LaunchConfiguration("launch_camera_view")
    cup_x = LaunchConfiguration("cup_x")
    cup_y = LaunchConfiguration("cup_y")
    cup_z = LaunchConfiguration("cup_z")
    cup_model_file = LaunchConfiguration("cup_model_file")

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
            "controllers_file": controllers_file,
            "pgi_contact_physics": pgi_contact_physics,
            "activate_arm_controller": activate_arm_controller,
            "pgi_logical_grasp_start": pgi_logical_grasp_start,
        }.items(),
    )

    gazebo_cup = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        condition=IfCondition(spawn_cup),
        arguments=[
            "-file",
            cup_model_file,
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

    cup_perception = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            str(PROJECT_ROOT / "scripts" / "pgi_cup_perception.py"),
            "--ros-args",
            "--params-file",
            str(PROJECT_ROOT / "config" / "pgi_cup_perception.yaml"),
        ],
        output="screen",
        condition=IfCondition(launch_cup_perception),
    )

    camera_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="pgi_cup_camera_view",
        output="log",
        additional_env={"QT_QPA_PLATFORM": "xcb"},
        condition=IfCondition(launch_camera_view),
        arguments=["/pgi/perception/cup_debug_image"],
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
                "world_file",
                default_value=str(PROJECT_ROOT / "worlds" / "pgi_140_80_camera.sdf"),
                description="Gazebo world with the rendering Sensors system."
            ),
            DeclareLaunchArgument(
                "camera_mount_xyz", default_value="0 0.085 0.030"
            ),
            DeclareLaunchArgument(
                "camera_mount_rpy", default_value="0 -1.57079632679 0"
            ),
            DeclareLaunchArgument(
                "activate_arm_controller",
                default_value="false",
                description=(
                    "Load the isolated Gazebo arm controller active. The logical "
                    "grasp runner normally switches it only after preflight."
                ),
            ),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=str(
                    PROJECT_ROOT / "config" / "pgi_140_80_sim_controllers.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "pgi_contact_physics",
                default_value="false",
                description="Enable provisional Stage-5 finger contact parameters.",
            ),
            DeclareLaunchArgument(
                "pgi_logical_grasp_start",
                default_value="false",
                description="Use the opt-in Cartesian side-ready joint state.",
            ),
            DeclareLaunchArgument(
                "spawn_cup",
                default_value="true",
                description="Spawn the independent Stage-3 cup in Gazebo and MoveIt.",
            ),
            DeclareLaunchArgument(
                "cup_model_file",
                default_value=str(
                    PROJECT_ROOT / "models" / "pgi_staging_cup" / "model.sdf"
                ),
                description="Gazebo cup SDF; Stage 5 supplies the dynamic variant.",
            ),
            DeclareLaunchArgument(
                "launch_cup_perception",
                default_value="true",
                description="Run the basic plan-only registered RGB-D cup detector.",
            ),
            DeclareLaunchArgument(
                "launch_camera_view",
                default_value="true",
                description="Open rqt_image_view on the cup detector debug image.",
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
            # Capture MoveIt/RViz values before Gazebo's nested launch writes
            # its own launch_rviz:=false into the shared launch context. The
            # returned timer starts MoveIt after the first clock/joint states.
            OpaqueFunction(function=_schedule_moveit),
            gazebo,
            TimerAction(period=7.0, actions=[gazebo_cup]),
            # The publisher also waits for the service, so a slow MoveGroup
            # startup cannot silently omit the collision object.
            TimerAction(period=9.0, actions=[moveit_cup]),
            TimerAction(period=10.0, actions=[cup_perception]),
            TimerAction(period=12.0, actions=[camera_view]),
        ]
    )
