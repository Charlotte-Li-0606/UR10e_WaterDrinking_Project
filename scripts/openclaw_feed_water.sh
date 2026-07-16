#!/usr/bin/env bash
# Thin OpenClaw entrypoint for the existing validated UR10e feed_water runner.
# It intentionally contains no perception, planning, or robot-control logic.
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

# The project runner uses OpenAI-compatible Chat Completions. Prefer an
# explicitly supplied environment credential; otherwise use the local
# OpenClaw secret reference configured for this machine. Nothing is printed.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  OPENCLAW_SECRET_FILE="${OPENCLAW_SECRET_FILE:-$HOME/.openclaw/secrets/my-proxy.json}"
  if [[ -r "$OPENCLAW_SECRET_FILE" ]]; then
    export OPENAI_API_KEY="$(jq -er '.apiKey | strings | select(length > 0)' "$OPENCLAW_SECRET_FILE")"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not configured; no feed_water plan was requested." >&2
  exit 78
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://43.165.176.234:8080/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
# Preserve the existing runner's structured JSON if ROS/MoveIt rejects a plan.
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
# run_llm_feeding_agent.sh supplies the project's ROS 2 and overlay setup.
exec "$PROJECT_DIR/scripts/run_llm_feeding_agent.sh" \
  --task "Feed water to me" \
  "$MODE" \
  --print-plan
