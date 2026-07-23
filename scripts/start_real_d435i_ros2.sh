#!/usr/bin/env bash
# Start one physical Intel RealSense D435i ROS 2 driver instance for the
# wrist-mounted perception test.  This intentionally does not remap or
# replace the simulation /wrist_rgbd topics.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u

exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:=d435i \
  camera_namespace:=d435i \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  enable_rgbd:=false \
  enable_gyro:=false \
  enable_accel:=false \
  unite_imu_method:=0 \
  rgb_camera.color_profile:=640x480x15 \
  depth_module.depth_profile:=640x480x15 \
  align_depth.enable:=true \
  pointcloud.enable:=false
