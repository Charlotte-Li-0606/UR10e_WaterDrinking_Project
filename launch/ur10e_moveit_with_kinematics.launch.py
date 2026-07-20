#!/usr/bin/env python3
"""UR10e MoveIt launch wrapper that explicitly gives kinematics to move_group.

The stock `ur_moveit_config ur_moveit.launch.py` in this setup passes
`robot_description_kinematics` to RViz, but the running `/move_group` node did
not expose that parameter. Without it, `/compute_ik` returns no solution for
pose targets. This wrapper keeps the same launch shape and adds
`moveit_config.robot_description_kinematics` to the move_group parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSORS_3D_CONFIG = PROJECT_ROOT / "config" / "sensors_3d.yaml"


def load_yaml(package_name: str, file_path: str):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def declare_arguments():
    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="false", description="Launch RViz?"),
            DeclareLaunchArgument(
                "ur_type",
                default_value="ur10e",
                description="Type/series of used UR robot.",
                choices=[
                    "ur3",
                    "ur5",
                    "ur10",
                    "ur3e",
                    "ur5e",
                    "ur7e",
                    "ur10e",
                    "ur12e",
                    "ur16e",
                    "ur8long",
                    "ur15",
                    "ur18",
                    "ur20",
                    "ur30",
                ],
            ),
            DeclareLaunchArgument(
                "warehouse_sqlite_path",
                default_value=os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
                description="Path where the warehouse database should be stored",
            ),
            DeclareLaunchArgument("launch_servo", default_value="false", description="Launch Servo?"),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Using or not time from simulation",
            ),
            DeclareLaunchArgument(
                "publish_robot_description_semantic",
                default_value="true",
                description="MoveGroup publishes robot description semantic",
            ),
            DeclareLaunchArgument(
                "use_octomap",
                default_value="false",
                description="Enable the experimental wrist PointCloud2 OctoMap obstacle layer.",
            ),
            DeclareLaunchArgument(
                "trajectory_controller",
                default_value="joint_trajectory_controller",
                description=(
                    "MoveIt FollowJointTrajectory controller. Keep the Gazebo default "
                    "joint_trajectory_controller; use scaled_joint_trajectory_controller "
                    "only with the physical UR driver."
                ),
            ),
        ]
    )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _launch_setup(context):
    launch_rviz = LaunchConfiguration("launch_rviz")
    ur_type = LaunchConfiguration("ur_type")
    warehouse_sqlite_path = LaunchConfiguration("warehouse_sqlite_path")
    launch_servo = LaunchConfiguration("launch_servo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    publish_robot_description_semantic = LaunchConfiguration("publish_robot_description_semantic")
    use_octomap = _as_bool(LaunchConfiguration("use_octomap").perform(context))
    trajectory_controller = LaunchConfiguration("trajectory_controller").perform(context)
    valid_controllers = {"joint_trajectory_controller", "scaled_joint_trajectory_controller"}
    if trajectory_controller not in valid_controllers:
        raise RuntimeError(
            "trajectory_controller must be one of "
            f"{sorted(valid_controllers)}, got {trajectory_controller!r}"
        )

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": ur_type})
        .to_moveit_configs()
    )

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": warehouse_sqlite_path,
    }
    moveit_parameters = moveit_config.to_dict()
    # Preserve the UR configuration's standard time-optimal parameterization,
    # then apply MoveIt's native Ruckig response adapter for jerk-limited
    # smoothing before the plan reaches the trajectory controller.
    moveit_parameters["ompl"]["response_adapters"] = [
        "default_planning_response_adapters/AddTimeOptimalParameterization",
        "default_planning_response_adapters/AddRuckigTrajectorySmoothing",
        "default_planning_response_adapters/ValidateSolution",
        "default_planning_response_adapters/DisplayMotionPath",
    ]
    # Gazebo starts joint_trajectory_controller while the physical UR driver
    # starts scaled_joint_trajectory_controller.  Advertising only the active
    # endpoint prevents MoveIt from selecting an inactive action server.
    controller_config = moveit_parameters["moveit_simple_controller_manager"]
    controller_config["controller_names"] = [trajectory_controller]
    for controller_name in valid_controllers:
        controller_config[controller_name]["default"] = controller_name == trajectory_controller
    if use_octomap:
        try:
            with SENSORS_3D_CONFIG.open(encoding="utf-8") as stream:
                sensors_3d_parameters = yaml.safe_load(stream) or {}
        except OSError as exc:
            raise RuntimeError(f"Could not load project sensors_3d config {SENSORS_3D_CONFIG}: {exc}") from exc
        if not isinstance(sensors_3d_parameters, dict):
            raise RuntimeError(f"Project sensors_3d config {SENSORS_3D_CONFIG} must contain a mapping")
        moveit_parameters.update(sensors_3d_parameters)
        octomap_log = LogInfo(
            msg=(
                "MoveIt OctoMap is ENABLED: expecting PointCloud2 on /wrist_rgbd/points "
                "from scripts/run_depth_to_pointcloud.sh"
            )
        )
    else:
        # Keep the dynamic layer opt-in. Existing deterministic PlanningScene
        # collision objects continue to work when the occupancy monitor is off.
        # Do not pass sensors=[] here: ROS parameters cannot represent an
        # untyped empty array and launch converts it to an invalid empty tuple.
        # Omitting the optional configuration disables the occupancy monitor.
        moveit_parameters.pop("sensors", None)
        octomap_log = LogInfo(msg="MoveIt OctoMap is disabled (use_octomap:=false)")

    wait_robot_description = Node(
        package="ur_robot_driver",
        executable="wait_for_robot_description",
        output="screen",
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_parameters,
            moveit_config.robot_description_kinematics,
            warehouse_ros_config,
            {
                "use_sim_time": use_sim_time,
                "publish_robot_description_semantic": publish_robot_description_semantic,
            },
        ],
    )

    servo_yaml = load_yaml("ur_moveit_config", "config/ur_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        condition=IfCondition(launch_servo),
        executable="servo_node",
        parameters=[
            moveit_config.to_dict(),
            moveit_config.robot_description_kinematics,
            servo_params,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            warehouse_ros_config,
            {"use_sim_time": use_sim_time},
        ],
    )

    return [
        octomap_log,
        wait_robot_description,
        RegisterEventHandler(
            OnProcessExit(
                target_action=wait_robot_description,
                on_exit=[move_group_node, rviz_node, servo_node],
            )
        ),
    ]


def generate_launch_description():
    ld = LaunchDescription()
    ld.add_entity(declare_arguments())
    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
