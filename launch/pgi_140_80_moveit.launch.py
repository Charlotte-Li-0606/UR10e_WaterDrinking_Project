#!/usr/bin/env python3
"""MoveIt plan-only bringup for the isolated PGI-140-80 simulation.

This launch builds MoveIt's robot model from the same opt-in PGI Xacro used by
Gazebo, then adds project-local SRDF and controller semantics. It never starts
ur_robot_driver, MoveIt Servo, perception, or a real hardware interface.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_FILE = PROJECT_ROOT / "urdf" / "ur_gz_feeding_markers.urdf.xacro"
SRDF_FILE = PROJECT_ROOT / "config" / "pgi_140_80.srdf"
MOVEIT_CONTROLLERS_FILE = (
    PROJECT_ROOT / "config" / "pgi_140_80_moveit_controllers.yaml"
)
SIM_CONTROLLERS_FILE = PROJECT_ROOT / "config" / "pgi_140_80_sim_controllers.yaml"
RRTCONNECT = "RRTConnect"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_ompl(moveit_parameters: dict) -> None:
    ompl = moveit_parameters.get("ompl")
    if not isinstance(ompl, dict):
        raise RuntimeError("MoveIt configuration does not contain OMPL parameters")
    planner_configs = ompl.get("planner_configs")
    if not isinstance(planner_configs, dict) or RRTCONNECT not in planner_configs:
        raise RuntimeError("MoveIt configuration is missing the RRTConnect profile")

    common = {
        "default_planner_config": RRTCONNECT,
        "planner_configs": [RRTCONNECT],
        "longest_valid_segment_fraction": 0.005,
    }
    ompl["ur_manipulator"] = {
        **common,
        "projection_evaluator": "joints(shoulder_pan_joint,shoulder_lift_joint)",
    }
    ompl["pgi_gripper"] = {
        **common,
        "projection_evaluator": "joints(pgi_left_finger_joint)",
    }
    ompl["ur10e_pgi"] = {
        **common,
        "projection_evaluator": "joints(shoulder_pan_joint,shoulder_lift_joint)",
    }


def _launch_setup(context):
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    allow_trajectory_execution = _as_bool(
        LaunchConfiguration("allow_trajectory_execution").perform(context)
    )
    if allow_trajectory_execution:
        raise RuntimeError(
            "PGI Stage 3 is plan-only and refuses allow_trajectory_execution:=true"
        )
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz").perform(context)
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy").perform(context)

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(
            str(DESCRIPTION_FILE),
            {
                "name": "ur",
                "ur_type": "ur10e",
                "tf_prefix": "",
                "safety_limits": "true",
                "safety_pos_margin": "0.15",
                "safety_k_position": "20",
                "simulation_controllers": str(SIM_CONTROLLERS_FILE),
                "robot_base_x": "0.0",
                "robot_base_y": "0.0",
                "robot_base_z": "0.0",
                "use_pgi_gripper": "true",
                "use_pgi_sim_control": "true",
                "camera_mount_xyz": camera_mount_xyz,
                "camera_mount_rpy": camera_mount_rpy,
            },
        )
        .robot_description_semantic(str(SRDF_FILE))
        .trajectory_execution(
            str(MOVEIT_CONTROLLERS_FILE), moveit_manage_controllers=False
        )
        .planning_pipelines(
            default_planning_pipeline="ompl", pipelines=["ompl"], load_all=False
        )
        .to_moveit_configs()
    )
    moveit_parameters = moveit_config.to_dict()
    _configure_ompl(moveit_parameters)

    # Keep the public MoveIt range at the documented 40 mm/jaw even though
    # Gazebo's DART-only physical hard stop has a 0.5 mm numerical guard band.
    joint_limits = moveit_parameters["robot_description_planning"]["joint_limits"]
    for joint_name in ("pgi_left_finger_joint", "pgi_right_finger_joint"):
        joint_limits[joint_name] = {
            "has_position_limits": True,
            "min_position": 0.0,
            "max_position": 0.040,
            "has_velocity_limits": True,
            "max_velocity": 0.0364,
            # Provisional, conservative planning value. The official data
            # specifies 1.1 s full travel but no acceleration limit.
            "has_acceleration_limits": True,
            "max_acceleration": 0.10,
        }

    kinematics = moveit_parameters["robot_description_kinematics"]["ur_manipulator"]
    kinematics["kinematics_solver_timeout"] = 0.05

    # Stage 3 is deliberately plan-only. Controller mappings are loaded and
    # validated, but trajectory execution remains disabled until a later stage.
    runtime_parameters = {
        "use_sim_time": use_sim_time,
        "allow_trajectory_execution": allow_trajectory_execution,
        "publish_robot_description": False,
        "publish_robot_description_semantic": True,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[moveit_parameters, runtime_parameters],
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_pgi_moveit",
        output="log",
        condition=IfCondition(launch_rviz),
        arguments=["-d", rviz_config_file],
        parameters=[moveit_parameters, runtime_parameters],
    )

    return [
        LogInfo(msg="PGI MoveIt Stage 3 is PLAN-ONLY; no trajectory will be sent"),
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="92",
                description="Isolated ROS domain shared with the PGI Gazebo simulation.",
            ),
            DeclareLaunchArgument(
                "launch_rviz", default_value="true", description="Open MoveIt RViz."
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="true", description="Use Gazebo /clock."
            ),
            DeclareLaunchArgument(
                "allow_trajectory_execution",
                default_value="false",
                description="Must remain false: Stage 3 refuses trajectory execution.",
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
            OpaqueFunction(function=_launch_setup),
        ]
    )
