#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
if [ -z "${XAUTHORITY:-}" ]; then
  XAUTHORITY_CANDIDATE="$(ls "${XDG_RUNTIME_DIR}"/.mutter-Xwaylandauth.* 2>/dev/null | head -n 1 || true)"
  if [ -n "${XAUTHORITY_CANDIDATE}" ]; then
    export XAUTHORITY="${XAUTHORITY_CANDIDATE}"
  fi
fi

PROJECT_DIR="${HOME}/ur_drinking_project"
LOG_DIR="${PROJECT_DIR}/logs"
WORLD_FILE="${PROJECT_DIR}/worlds/ur10e_feeding_empty.sdf"
DESCRIPTION_FILE="${PROJECT_DIR}/urdf/ur_gz_feeding_markers.urdf.xacro"
CONFIG_FILE="${PROJECT_DIR}/config/ur10e_sdk_config.yaml"
export GZ_SIM_RESOURCE_PATH="${PROJECT_DIR}/models:${GZ_SIM_RESOURCE_PATH:-}"

mkdir -p "${LOG_DIR}"

if [ ! -f "${PROJECT_DIR}/models/my_human_face/meshes/standing.dae" ] || \
   [ ! -f "${PROJECT_DIR}/models/my_human_face/materials/textures/young_lightskinned_male_diffuse.png" ]; then
  echo "Human visual asset is incomplete: restore models/my_human_face/meshes/standing.dae and its materials/textures directory before launching Gazebo." >&2
fi

pkill -f gazebo_feeding_scene.py || true
pkill -f ros2_mouth_perception.py || true
pkill -f mouth_perception_node.py || true
pkill -f ur10e_moveit_with_kinematics.launch.py || true
pkill -f move_group || true
pkill -f robot_state_publisher || true
pkill -f 'controller_manager/spawner' || true
pkill -f spawner_joint_state_broadcaster || true
pkill -f spawner_scaled_joint_trajectory_controller || true
pkill -f static_transform_publisher || true
pkill -f 'parameter_bridge /clock@rosgraph_msgs/msg/Clock' || true
pkill -f "parameter_bridge.*wrist_rgbd" || true
pkill -f depth_to_pointcloud_node.py || true
pkill -f ur_sim_control_feeding.launch.py || true
pkill -f ur_sim_control.launch.py || true
pkill -f "gz sim" || true
sleep 2

setsid ros2 launch "${PROJECT_DIR}/launch/ur_sim_control_feeding.launch.py" \
  ur_type:=ur10e \
  launch_rviz:=false \
  gazebo_gui:=true \
  robot_base_x:=0.0 \
  robot_base_y:=0.55 \
  robot_base_z:=0.60 \
  controllers_file:="${PROJECT_DIR}/config/ur10e_sim_controllers.yaml" \
  initial_joint_controller:=joint_trajectory_controller \
  world_file:="${WORLD_FILE}" \
  description_file:="${DESCRIPTION_FILE}" \
  > "${LOG_DIR}/ur10e_feeding_gazebo.log" 2>&1 < /dev/null &

sleep 8

setsid ros2 run ros_gz_bridge parameter_bridge \
  "/wrist_rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/wrist_rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/wrist_rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  > "${LOG_DIR}/wrist_rgbd_bridge.log" 2>&1 < /dev/null &

USE_OCTOMAP="${USE_OCTOMAP:-false}"
if [ "${USE_OCTOMAP}" = "true" ] || [ "${USE_OCTOMAP}" = "1" ]; then
  setsid "${PROJECT_DIR}/scripts/run_depth_to_pointcloud.sh" \
    > "${LOG_DIR}/depth_to_pointcloud.log" 2>&1 < /dev/null &
fi

setsid "${PROJECT_DIR}/scripts/start_mouth_perception.sh" \
  > "${LOG_DIR}/mouth_perception.log" 2>&1 < /dev/null &

setsid ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 world odom \
  --ros-args -p use_sim_time:=true \
  > "${LOG_DIR}/world_to_odom_static_tf.log" 2>&1 < /dev/null &

setsid ros2 launch "${PROJECT_DIR}/launch/ur10e_moveit_with_kinematics.launch.py" \
  ur_type:=ur10e \
  launch_rviz:=false \
  use_sim_time:=true \
  use_octomap:="${USE_OCTOMAP}" \
  > "${LOG_DIR}/moveit_with_kinematics.log" 2>&1 < /dev/null &

sleep 4

python3 "${PROJECT_DIR}/scripts/gazebo_feeding_scene.py" \
  --config "${CONFIG_FILE}" \
  --once \
  --use-existing \
  > "${LOG_DIR}/gazebo_feeding_scene.log" 2>&1

echo "UR10e feeding Gazebo, MoveIt, sim-time TF, and flange-attached markers started."
echo "World file: ${WORLD_FILE}"
echo "Robot description: ${DESCRIPTION_FILE}"
