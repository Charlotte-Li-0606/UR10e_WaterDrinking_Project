# Copyright (c) 2021 Stogl Robotics Consulting UG (haftungsbeschränkt)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the {copyright_holder} nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Denis Stogl

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    AndSubstitution,
    Command,
    FindExecutable,
    EnvironmentVariable,
    LaunchConfiguration,
    NotSubstitution,
    PathJoinSubstitution,
    IfElseSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    # General arguments
    controllers_file = LaunchConfiguration("controllers_file")
    tf_prefix = LaunchConfiguration("tf_prefix")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    description_file = LaunchConfiguration("description_file")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")
    robot_base_x = LaunchConfiguration("robot_base_x")
    robot_base_y = LaunchConfiguration("robot_base_y")
    robot_base_z = LaunchConfiguration("robot_base_z")
    use_pgi_gripper = LaunchConfiguration("use_pgi_gripper")
    pgi_contact_physics = LaunchConfiguration("pgi_contact_physics")
    pgi_logical_grasp_start = LaunchConfiguration("pgi_logical_grasp_start")
    activate_pgi_controller = LaunchConfiguration("activate_pgi_controller")
    pgi_gripper_controller = LaunchConfiguration("pgi_gripper_controller")
    camera_mount_xyz = LaunchConfiguration("camera_mount_xyz")
    camera_mount_rpy = LaunchConfiguration("camera_mount_rpy")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            description_file,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            "ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "tf_prefix:=",
            tf_prefix,
            " ",
            "simulation_controllers:=",
            controllers_file,
            " ",
            "robot_base_x:=",
            robot_base_x,
            " ",
            "robot_base_y:=",
            robot_base_y,
            " ",
            "robot_base_z:=",
            robot_base_z,
            " ",
            "use_pgi_gripper:=",
            use_pgi_gripper,
            " ",
            "use_pgi_sim_control:=",
            use_pgi_gripper,
            " ",
            "pgi_contact_physics:=",
            pgi_contact_physics,
            " ",
            "pgi_logical_grasp_start:=",
            pgi_logical_grasp_start,
            " ",
            "camera_mount_xyz:='",
            camera_mount_xyz,
            "' ",
            "camera_mount_rpy:='",
            camera_mount_rpy,
            "'",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(launch_rviz),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # Delay rviz start after `joint_state_broadcaster`
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    # There may be other controllers of the joints, but this is the initially-started one
    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager"],
        condition=IfCondition(activate_joint_controller),
    )
    initial_joint_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager", "--inactive"],
        condition=UnlessCondition(activate_joint_controller),
    )

    # The gripper has independent controller ownership. It exists only when
    # the opt-in PGI description is selected, so the established simulation
    # and all real-robot launch paths remain unchanged by default.
    pgi_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[pgi_gripper_controller, "-c", "/controller_manager"],
        condition=IfCondition(AndSubstitution(use_pgi_gripper, activate_pgi_controller)),
    )
    pgi_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[pgi_gripper_controller, "-c", "/controller_manager", "--inactive"],
        condition=IfCondition(
            AndSubstitution(use_pgi_gripper, NotSubstitution(activate_pgi_controller))
        ),
    )

    # GZ nodes
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string",
            robot_description_content,
            "-name",
            "ur",
            "-allow_renaming",
            "true",
        ],
    )

    gz_launch_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={
            "gz_args": IfElseSubstitution(
                gazebo_gui,
                if_value=[" -r -v 4 ", world_file],
                else_value=[" -s -r -v 4 ", world_file],
            )
        }.items(),
    )

    # Loading a world and creating an entity are asynchronous in Gazebo.  If
    # creation races the world load, the world reset removes the new UR model.
    # Wait for the world to be ready, then configure controllers only after the
    # robot creation request has completed.
    delayed_gz_spawn_entity = TimerAction(period=5.0, actions=[gz_spawn_entity])
    start_controllers_after_robot_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gz_spawn_entity,
            on_exit=[
                joint_state_broadcaster_spawner,
                initial_joint_controller_spawner_stopped,
                initial_joint_controller_spawner_started,
                pgi_controller_spawner_stopped,
                pgi_controller_spawner_started,
            ],
        )
    )

    # Make the /clock topic available in ROS
    gz_sim_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
    )

    nodes_to_start = [
        robot_state_publisher_node,
        delay_rviz_after_joint_state_broadcaster_spawner,
        start_controllers_after_robot_spawn,
        gz_launch_description,
        delayed_gz_spawn_entity,
        gz_sim_bridge,
    ]

    return nodes_to_start


def generate_launch_description():
    project_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[project_models_dir, ":", EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value="")],
    )

    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
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
            default_value="ur5e",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "config", "ur10e_sim_controllers.yaml")
            ),
            description="Absolute path to YAML file with the controllers configuration.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value='""',
            description="Prefix of the joint names, useful for "
            "multi-robot setup. If changed than also joint names in the controllers' configuration "
            "have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "activate_joint_controller",
            default_value="true",
            description="Enable headless mode for robot control",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Robot controller to start.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "urdf", "ur_gz.urdf.xacro"]
            ),
            description="URDF/XACRO description file (absolute path) with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_description"), "rviz", "view_robot.rviz"]
            ),
            description="Rviz config file (absolute path) to use when launching rviz.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gazebo_gui", default_value="true", description="Start gazebo with GUI?"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "world_file",
            default_value="empty.sdf",
            description="Gazebo world file (absolute path or filename from the gazebosim worlds collection) containing a custom world.",
        )
    )
    declared_arguments.extend(
        [
            DeclareLaunchArgument(
                "robot_base_x", default_value="0.0", description="UR10e base_link X in Gazebo world (m)."
            ),
            DeclareLaunchArgument(
                "robot_base_y", default_value="0.55", description="UR10e base_link Y in Gazebo world (m)."
            ),
            DeclareLaunchArgument(
                "robot_base_z", default_value="0.60", description="UR10e base_link Z in Gazebo world (m)."
            ),
        ]
    )
    declared_arguments.extend(
        [
            DeclareLaunchArgument(
                "use_pgi_gripper",
                default_value="false",
                description=(
                    "Opt into the provisional PGI model and its Gazebo ros2_control "
                    "interfaces. False preserves the established simulation."
                ),
            ),
            DeclareLaunchArgument(
                "activate_pgi_controller",
                default_value="true",
                description="Start the PGI GripperCommand controller when the model is enabled.",
            ),
            DeclareLaunchArgument(
                "pgi_contact_physics",
                default_value="false",
                description=(
                    "Enable provisional Stage-5 finger contact/friction and the "
                    "40 N per-jaw effort limit. False preserves earlier stages."
                ),
            ),
            DeclareLaunchArgument(
                "pgi_logical_grasp_start",
                default_value="false",
                description=(
                    "Use the simulation-only Cartesian side-ready initial pose. "
                    "False preserves the established camera-ready pose."
                ),
            ),
            DeclareLaunchArgument(
                "pgi_gripper_controller",
                default_value="pgi_gripper_controller",
                description="Controller-manager name of the PGI GripperCommand controller.",
            ),
            DeclareLaunchArgument(
                "camera_mount_xyz",
                default_value="0 0.085 0.030",
                description="Provisional simulation-only D435i mount translation in metres.",
            ),
            DeclareLaunchArgument(
                "camera_mount_rpy",
                default_value="0 -1.57079632679 0",
                description="Provisional simulation-only D435i mount rotation in radians.",
            ),
        ]
    )

    return LaunchDescription(
        [model_resource_path] + declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
