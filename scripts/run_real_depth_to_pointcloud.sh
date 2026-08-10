#!/usr/bin/env bash
# Publish the physical D435i aligned depth as the same bounded PointCloud2
# topic consumed by the project's opt-in MoveIt OctoMap configuration.
# This script never starts MoveIt and never sends a robot/controller command.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_DIR}/scripts/run_depth_to_pointcloud.sh" \
  --depth-topic "${D435I_DEPTH_TOPIC:-/d435i/d435i/aligned_depth_to_color/image_raw}" \
  --camera-info-topic "${D435I_CAMERA_INFO_TOPIC:-/d435i/d435i/color/camera_info}" \
  --points-topic "${D435I_POINTS_TOPIC:-/wrist_rgbd/points}" \
  --stride "${D435I_POINTCLOUD_STRIDE:-4}" \
  --min-depth "${D435I_POINTCLOUD_MIN_DEPTH_M:-0.20}" \
  --max-depth "${D435I_POINTCLOUD_MAX_DEPTH_M:-2.00}" \
  --max-publish-rate "${D435I_POINTCLOUD_MAX_RATE_HZ:-5.0}" \
  --exclude-tool-box \
  --tool-frame "${D435I_TOOL_FRAME:-tool0}" \
  --tool-box-center-z-m "${D435I_TOOL_BOX_CENTER_Z_M:-0.15}" \
  --tool-box-size-xy-m "${D435I_TOOL_BOX_SIZE_XY_M:-0.10}" \
  --tool-box-size-z-m "${D435I_TOOL_BOX_SIZE_Z_M:-0.30}" \
  --tool-box-padding-m "${D435I_TOOL_BOX_PADDING_M:-0.02}" \
  "$@"
