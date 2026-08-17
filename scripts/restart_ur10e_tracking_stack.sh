#!/usr/bin/env bash
# Restart the project-owned real UR10e tracking stack without planning or motion.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${UR10E_ROBOT_IP:-192.168.2.102}"
LOG_DIR="${PROJECT_DIR}/logs/runtime_$(date +%Y%m%d)_initialization"
mkdir -p "${LOG_DIR}"

source /opt/ros/jazzy/setup.bash
set -u
unset ROS_STATIC_PEERS
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=0

if pgrep -af '[c]odex_feed_water\.sh|[r]eal_feed_water_integrated\.py|[r]eal_mouth_tracking_servo\.py' >/dev/null; then
  echo "Refusing initialization while a feed-water or real tracking workflow is active." >&2
  exit 2
fi

current_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
declare -A owned_pgids=()
patterns=(
  '[r]qt_image_view'
  '[r]un_real_d435i_mouth_perception\.sh'
  '[m]outh_target_tracker_node'
  '[u]r10e_moveit_with_kinematics\.launch\.py'
  '[r]ealsense2_camera.*rs_launch\.py'
  '[u]r_robot_driver.*ur_control\.launch\.py'
)

for pattern in "${patterns[@]}"; do
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ "${pgid}" =~ ^[0-9]+$ && "${pgid}" -gt 1 && "${pgid}" != "${current_pgid}" ]]; then
      owned_pgids["${pgid}"]=1
    fi
  done < <(pgrep -f "${pattern}" || true)
done

if ((${#owned_pgids[@]})); then
  echo "Stopping project-owned process groups: ${!owned_pgids[*]}"
  for pgid in "${!owned_pgids[@]}"; do
    kill -INT -- "-${pgid}" 2>/dev/null || true
  done
  deadline=$((SECONDS + 12))
  while ((SECONDS < deadline)); do
    any_alive=false
    for pgid in "${!owned_pgids[@]}"; do
      if kill -0 -- "-${pgid}" 2>/dev/null; then
        any_alive=true
        break
      fi
    done
    [[ "${any_alive}" == false ]] && break
    sleep 0.25
  done
  for pgid in "${!owned_pgids[@]}"; do
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi
  done
  sleep 1
fi

start_group() {
  local name="$1"
  shift
  setsid "$@" >"${LOG_DIR}/${name}.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${LOG_DIR}/${name}.pid"
  echo "Started ${name} (PID ${pid})"
}

wait_for() {
  local label="$1"
  local timeout_sec="$2"
  shift 2
  local deadline=$((SECONDS + timeout_sec))
  while ((SECONDS < deadline)); do
    if "$@" >/dev/null 2>&1; then
      echo "Ready: ${label}"
      return 0
    fi
    sleep 0.5
  done
  echo "Initialization check timed out: ${label}" >&2
  return 1
}

has_ros_name() {
  local kind="$1"
  local name="$2"
  case "${kind}" in
    node) ros2 node list 2>/dev/null | grep -Fxq "${name}" ;;
    service) ros2 service list 2>/dev/null | grep -Fxq "${name}" ;;
    topic) ros2 topic list 2>/dev/null | grep -Fxq "${name}" ;;
    *) return 2 ;;
  esac
}

has_topic_subscriber() {
  ros2 topic info "$1" 2>/dev/null \
    | grep -Eq '^Subscription count: [1-9][0-9]*$'
}

start_group ur_driver env UR10E_ROBOT_IP="${ROBOT_IP}" \
  "${PROJECT_DIR}/scripts/start_ur10e_real_driver.sh"
start_group realsense "${PROJECT_DIR}/scripts/start_real_d435i_ros2.sh"

wait_for "controller manager" 30 has_ros_name service /controller_manager/list_controllers
wait_for "D435i color topic" 30 has_ros_name topic /d435i/d435i/color/image_raw
wait_for "D435i aligned-depth topic" 30 has_ros_name topic /d435i/d435i/aligned_depth_to_color/image_raw

start_group moveit_servo env UR10E_LAUNCH_SERVO=true UR10E_USE_OCTOMAP=false \
  "${PROJECT_DIR}/scripts/start_ur10e_real_moveit.sh"
start_group mouth_perception "${PROJECT_DIR}/scripts/run_real_d435i_mouth_perception.sh"
start_group mouth_tracker env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
  /usr/bin/python3 -m robot_layer.arm_ur10e.perception.mouth_target_tracker_node

wait_for "MoveGroup" 35 has_ros_name node /move_group
wait_for "MoveIt Servo" 35 has_ros_name node /servo_node
wait_for "mouth target tracker" 30 has_ros_name node /mouth_target_tracker
wait_for "mouth status topic" 30 has_ros_name topic /mouth_detection/status
wait_for "mouth debug-image topic" 30 has_ros_name topic /mouth_detection/debug_image
wait_for "MediaPipe color subscription" 30 has_topic_subscriber /d435i/d435i/color/image_raw
wait_for "MediaPipe aligned-depth subscription" 30 has_topic_subscriber /d435i/d435i/aligned_depth_to_color/image_raw

# Force the viewer through XWayland and discard stale/off-screen geometry from
# a previous desktop session. Keep it above the main desktop so initialization
# produces a visible image window, not merely a live background subscriber.
start_group rqt_image env QT_QPA_PLATFORM=xcb \
  /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view \
  --clear-config --on-top /mouth_detection/debug_image
wait_for "rqt image viewer" 20 pgrep -f '[r]qt_image_view'
wait_for "rqt debug-image subscription" 20 has_topic_subscriber /mouth_detection/debug_image

controller_state="inactive"
if ros2 control list_controllers 2>/dev/null \
  | grep -Eq '^scaled_joint_trajectory_controller[[:space:]].*[[:space:]]active[[:space:]]*$'; then
  controller_state="active"
fi

program_running="unknown"
program_output="$(
  timeout 5 ros2 service call \
    /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}' \
    2>/dev/null || true
)"
if grep -q 'program_running=True' <<<"${program_output}"; then
  program_running="true"
elif grep -q 'program_running=False' <<<"${program_output}"; then
  program_running="false"
fi

speed_scaling="unknown"
speed_output="$(
  timeout 5 ros2 topic echo --once --field data \
    /speed_scaling_state_broadcaster/speed_scaling 2>/dev/null || true
)"
parsed_speed="$(
  awk '/^[+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ {print; exit}' \
    <<<"${speed_output}"
)"
[[ -n "${parsed_speed}" ]] && speed_scaling="${parsed_speed}"

real_execution_ready="false"
if [[ "${controller_state}" == "active" && "${program_running}" == "true" ]] \
  && awk -v value="${speed_scaling}" \
    'BEGIN {exit !(value != "unknown" && value + 0.0 > 0.0)}'; then
  real_execution_ready="true"
fi

cat <<EOF
Initialization complete (no planning or motion command was sent).
Logs: ${LOG_DIR}
UR driver: started
D435i RGB-D: started
MoveIt + Servo: started (OctoMap disabled)
MediaPipe mouth perception: started
Mouth target tracker: started
rqt_image_view: started
scaled_joint_trajectory_controller: ${controller_state}
External Control program running: ${program_running}
speed scaling: ${speed_scaling}
real_execution_ready: ${real_execution_ready}
EOF

if [[ "${real_execution_ready}" != "true" ]]; then
  echo "Initialization completed, but real execution remains blocked until the pendant/External Control state is ready." >&2
fi
