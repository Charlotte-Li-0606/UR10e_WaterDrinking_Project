# Feeding RGB-D Observation Contract

The UR10e feeding setup has no gripper. The RGB-D camera is fixed tool
geometry, while the cup and straw are a fixed tool attachment. The robot does
not grasp or detach the cup during this simulation.

## ROS2 observations

- RGB image: `/wrist_rgbd/image` (`sensor_msgs/msg/Image`)
- Depth image: `/wrist_rgbd/depth_image` (`sensor_msgs/msg/Image`)
- Camera calibration: `/wrist_rgbd/camera_info` (`sensor_msgs/msg/CameraInfo`)
- Joint state: `/joint_states`
- Tool pose: TF `base_link -> tool0`
- Camera optical frame: TF `tool0 -> wrist_rgbd_camera_optical_frame`
- Straw tip: TF `tool0 -> feeding_straw_tip_marker`
- Mouth target: `feeding.mouth_target_position` in `config/ur10e_sdk_config.yaml`

## RGB-D mouth perception

`robot_layer/arm_ur10e/perception/mouth_perception_node.py` synchronizes `/wrist_rgbd/image`,
`/wrist_rgbd/depth_image`, and `/wrist_rgbd/camera_info` with ROS 2's
`ApproximateTimeSynchronizer`. It uses the bundled MediaPipe Face Landmarker
model, takes the median valid depth around the inner-lip landmarks, and
back-projects that pixel with `CameraInfo.K`. It then uses TF to transform the
point from the camera optical frame into `base_link`, rejects implausible jumps,
and applies an exponential smoothing filter in `base_link`.

It publishes valid estimates only:

- `/detected_mouth_pose` (`geometry_msgs/PoseStamped`)
- `/mouth_detection/debug_image` (`sensor_msgs/Image`)
- `/mouth_detection/status` (`std_msgs/String` JSON)

`perception/active_target_manager.py` is a consumer-side wrapper around this
single-pose topic. It currently records the pose as the logical `center`
target, maintains a bounded stable-pose queue, and lets the feeding agent lock
`left`, `center`, or `right` without changing the MediaPipe publisher. When a
future perception publisher provides multiple candidates with image-x values,
the manager will assign left/right/center and continue tracking the candidate
nearest to the locked target's prior pose.

`detected_mouth_pose.header.frame_id` is `base_link`. The RGB-D point is first
computed in `wrist_rgbd_camera_optical_frame` from `CameraInfo.header.frame_id`
and then transformed through TF. Its position is the 3D mouth point in metres.
The identity orientation explicitly means orientation is not estimated.

The main simulation-start script launches the node automatically after the
RGB-D bridge. To install its pinned Python dependencies and official model on
another checkout, run `./scripts/setup_mouth_perception.sh`. Start the node
manually with `./scripts/run_mouth_perception.sh`.

The image topics are one-way Gazebo-to-ROS bridges. RGB observation uses the
bundled textured mesh at `models/my_human_face/meshes/standing.dae`; its PNG
textures remain in `models/my_human_face/materials/textures`. The visual and
the enlarged primitive collision proxy share the root pose `(0.35, 1.10, 0)`.
The visible mouth target is `(0.357, 0.940, 1.708)`, 1 cm before the mesh's
mouth surface; the simple collision proxy remains independent of its visual.

To inspect the same live image that ROS 2 receives, run:

```bash
./scripts/view_wrist_rgb.sh
```

This opens the **UR10e Wrist RGB Viewer**; press `q` or `Esc` to close it.
Gazebo's existing GUI remains the full 3D scene viewer.

`UR10eRobotEnv` subscribes to the three RGB-D ROS topics and exposes raw RGB
and depth frames, camera calibration, `tool0` pose, straw-tip TF pose, and the
configured mouth-target pose through `get_observation()` and `get_state()`.

## Intended actions

Downstream control should use feeding primitives only:
`move_straw_tip_to_pre_mouth()`, `move_straw_tip_to_mouth()`, and `retreat()`.
Gripper, grasp, attach-cup, and detach-cup actions are intentionally outside
this simulation contract.
