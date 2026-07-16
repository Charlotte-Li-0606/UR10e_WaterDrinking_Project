#!/usr/bin/env bash
# Compatibility entrypoint for the reusable UR10e OpenClaw tool plan.
#
# This intentionally does not call llm_feeding_agent.py or the legacy
# high-level feed_water wrapper. An OpenClaw chat has already selected this
# skill, so the fixed validated reusable plan is the only execution surface.
set -euo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
MODE="--plan-only"

case "$#" in
  0)
    ;;
  1)
    case "$1" in
      --plan-only)
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
exec "$PROJECT_DIR/scripts/openclaw_feeding_tool.sh" --plan-json "$PLAN"
