#!/usr/bin/env bash
# Append the daily project snapshot and open the simulator and real-robot views.

set -uo pipefail

PROJECT_DIR="/home/dase-hw101/ur_drinking_project"
DAILY_LOG_FILE="${PROJECT_DIR}/logs/daily_log.txt"
URSIM_URL="http://192.168.2.101:6080/vnc.html"
REAL_ROBOT_IP="192.168.2.102"
LIVE_PENDANT_URL="http://127.0.0.1:8088"
LIVE_PENDANT_LOG="${PROJECT_DIR}/logs/live_pendant.log"
LIVE_PENDANT_PID="${PROJECT_DIR}/logs/live_pendant.pid"

mkdir -p "${PROJECT_DIR}/logs"

append_snapshot() {
  local timestamp branch tracked_changes recent_commits robot_route robot_ping
  timestamp="$(date --iso-8601=seconds)"
  branch="$(git -C "${PROJECT_DIR}" branch --show-current 2>/dev/null || true)"
  tracked_changes="$(git -C "${PROJECT_DIR}" status --short --untracked-files=no 2>/dev/null || true)"
  recent_commits="$(git -C "${PROJECT_DIR}" log --since='today 00:00' --pretty='format:%h %ad %s' --date=short 2>/dev/null || true)"
  robot_route="$(ip route get "${REAL_ROBOT_IP}" 2>/dev/null | head -n 1 || true)"
  if ping -c 1 -W 1 "${REAL_ROBOT_IP}" >/dev/null 2>&1; then
    robot_ping="reachable"
  else
    robot_ping="unreachable"
  fi

  {
    printf '===== UR drinking project daily log: %s =====\n' "${timestamp}"
    printf 'Git branch: %s\n' "${branch:-unknown}"
    printf 'Commits made today:\n%s\n' "${recent_commits:-  none}"
    printf 'Tracked working-tree changes:\n%s\n' "${tracked_changes:-  none}"
    printf 'Real UR10e network: %s (%s)\n' "${robot_ping}" "${REAL_ROBOT_IP}"
    printf 'Real UR10e route: %s\n' "${robot_route:-unavailable}"
  } >&9
}

ensure_ursim() {
  local running
  running="$(docker inspect --format '{{.State.Running}}' ursim 2>/dev/null || true)"
  if [ "${running}" = "true" ]; then
    printf 'URSim: already running\n' >&9
  elif [ -n "${running}" ]; then
    if docker start ursim >/dev/null 2>&1; then
      printf 'URSim: existing container started\n' >&9
    else
      printf 'URSim: failed to start existing container\n' >&9
      return
    fi
  else
    if (set +u; source /opt/ros/jazzy/setup.bash && ros2 run ur_client_library start_ursim.sh -m ur10e -d) \
      >/dev/null 2>&1; then
      printf 'URSim: new UR10e container started\n' >&9
    else
      printf 'URSim: failed to create/start UR10e container\n' >&9
      return
    fi
  fi

  for _attempt in $(seq 1 20); do
    if curl --fail --silent --max-time 2 "${URSIM_URL}" >/dev/null 2>&1; then
      printf 'URSim web UI: %s\n' "${URSIM_URL}" >&9
      if /usr/bin/xdg-open "${URSIM_URL}" >/dev/null 2>&1; then
        printf 'URSim browser page: open request sent\n' >&9
      else
        printf 'URSim browser page: open request failed; use the URL above\n' >&9
      fi
      return
    fi
    sleep 0.5
  done
  printf 'URSim web UI: container is running but the page did not become reachable\n' >&9
}

ensure_live_pendant() {
  local health_url attempt
  health_url="${LIVE_PENDANT_URL}/api/health"

  if curl --fail --silent --max-time 1 "${health_url}" >/dev/null 2>&1; then
    printf 'Real UR10e live pendant: already running\n' >&9
  else
    nohup "${PROJECT_DIR}/scripts/start_live_pendant.sh" \
      >"${LIVE_PENDANT_LOG}" 2>&1 &
    printf '%s\n' "$!" >"${LIVE_PENDANT_PID}"
    for attempt in $(seq 1 20); do
      if curl --fail --silent --max-time 1 "${health_url}" >/dev/null 2>&1; then
        printf 'Real UR10e live pendant: started (PID %s)\n' "$(cat "${LIVE_PENDANT_PID}")" >&9
        break
      fi
      sleep 0.25
    done
  fi

  if curl --fail --silent --max-time 1 "${health_url}" >/dev/null 2>&1; then
    printf 'Real UR10e live pendant page: %s\n' "${LIVE_PENDANT_URL}" >&9
    if /usr/bin/xdg-open "${LIVE_PENDANT_URL}" >/dev/null 2>&1; then
      printf 'Live pendant browser page: open request sent\n' >&9
    else
      printf 'Live pendant browser page: open request failed; use the URL above\n' >&9
    fi
  else
    printf 'Real UR10e live pendant failed to start; inspect %s\n' "${LIVE_PENDANT_LOG}" >&9
  fi
}

exec 9>>"${DAILY_LOG_FILE}"
flock 9
append_snapshot
ensure_ursim
ensure_live_pendant
printf '\n' >&9
