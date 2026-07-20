#!/usr/bin/env bash
# Simulator-only entrypoint for the reusable UR10e OpenClaw tool plan.
#
# This intentionally does not call llm_feeding_agent.py or the legacy
# high-level feed_water wrapper. An OpenClaw chat has already selected this
# skill, so the fixed validated reusable plan is the only execution surface.
# A normal matching request starts/verifies Gazebo and executes this plan.
# --plan-only remains available only for an explicit no-simulation validation.
set -eo pipefail

source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
MODE="--execute"

case "$#" in
  0)
    ;;
  1)
    case "$1" in
      --plan-only)
        MODE="--plan-only"
        ;;
      --execute)
        MODE="--execute"
        ;;
      *)
        echo "Usage: $0 [--plan-only|--execute]" >&2
        exit 64
        ;;
    esac
    ;;
  *)
    echo "Usage: $0 [--plan-only|--execute]" >&2
    exit 64
    ;;
esac

simulator_ready() {
  local actions
  actions="$(ros2 action list 2>/dev/null || true)"
  grep -Fxq '/move_action' <<<"${actions}" && \
    grep -Fxq '/joint_trajectory_controller/follow_joint_trajectory' <<<"${actions}"
}

if [[ "$MODE" == "--execute" ]]; then
  if ! simulator_ready; then
    "$PROJECT_DIR/scripts/start_ur10e_feeding_sim.sh"
  fi
  for _attempt in $(seq 1 45); do
    if simulator_ready; then
      break
    fi
    sleep 1
  done
  if ! simulator_ready; then
    printf '%s\n' '{"success":false,"stage":"simulator_startup","reason":"Gazebo/MoveIt did not expose /move_action and the joint trajectory controller within 45 seconds; no reusable tool was executed."}'
    exit 2
  fi
fi

PLAN='{
  "steps": [
    {"tool":"get_observation","args":{}},
    {"tool":"detect_target","args":{"target_type":"mouth","detector":"mediapipe"}},
    {"tool":"active_search","args":{"target_type":"mouth","detector":"mediapipe","max_time_sec":30.0,"strategy":"safe_scan","execute":false}},
    {"tool":"select_target","args":{"target_type":"mouth","strategy":"center"}},
    {"tool":"move_tool_to_target","args":{"tool":"straw_tip","target":"pre_mouth","execute":false}},
    {"tool":"check_progress","args":{"task":"feed_water","critic":"rule_based"}}
  ]
}'

if [[ "$MODE" == "--execute" ]]; then
  PLAN='{
  "steps": [
    {"tool":"get_observation","args":{}},
    {"tool":"detect_target","args":{"target_type":"mouth","detector":"mediapipe"}},
    {"tool":"active_search","args":{"target_type":"mouth","detector":"mediapipe","max_time_sec":30.0,"strategy":"safe_scan","execute":true}},
    {"tool":"select_target","args":{"target_type":"mouth","strategy":"center"}},
    {"tool":"move_tool_to_target","args":{"tool":"straw_tip","target":"pre_mouth","execute":true}},
    {"tool":"check_progress","args":{"task":"feed_water","critic":"rule_based"}},
    {"tool":"hold","args":{"duration_sec":3.0}}
  ]
}'
fi

if [[ "$MODE" == "--execute" ]]; then
  exec "$PROJECT_DIR/scripts/openclaw_feeding_tool.sh" --execute --plan-json "$PLAN"
fi
exec "$PROJECT_DIR/scripts/openclaw_feeding_tool.sh" --validate-only --plan-json "$PLAN"
