# UR10e water-drinking project

The working physical water-feeding baseline is preserved on `main` and tagged
`pre-pgi-gripper-baseline-2026-08-25`. PGI gripper work is isolated on
`feature/pgi-140-80-simulation` and is simulation-description-only.

## PGI-140-80 RViz model

The reusable Xacro models the configured
`PGI-140-80-W-S-M1-L5-J1-F1-01`, its two symmetric jaws, F1 mount,
camera-interposer placeholder, D435i placeholder, and logical grasp center.
The canonical description remains opt-in and defaults to:

```text
use_pgi_gripper:=false
```

Run the static checks:

```bash
cd /home/dase-hw101/ur_drinking_project
scripts/verify_pgi_140_80_description.sh
```

Display the PGI model without Gazebo, MoveIt, controllers, or hardware:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/display_pgi_140_80.launch.py \
  use_pgi_gripper:=true
```

Set `use_pgi_gripper:=false` in the same display launch to show the established
non-PGI tool model. The dedicated display launch defaults to `true` because its
only purpose is PGI review; the shared robot Xacro defaults to `false`.

The left-jaw slider controls `pgi_left_finger_joint` over `0.000-0.040 m`.
`pgi_right_finger_joint` mirrors it on the opposite axis, producing the
documented 80 mm total stroke.

The dedicated RViz configuration opens a close tool view. The provisional
D435i is the black rectangular body on the gripper's `+Y` front side; the
colored axes named `D435i depth optical frame` mark
`d435i_depth_optical_frame`. The separate `PGI grasp center` axes
mark the future cup attachment frame. Use the mouse wheel to zoom out from the
tool-focused view when the complete UR10e needs to be inspected.

The default provisional transform is `xyz="0 0.085 0.030"` and
`rpy="0 -1.57079632679 0"`, which puts the camera on the front side with its
optical +Z axis looking down. The `camera_mount_xyz` and `camera_mount_rpy`
arguments exist only in the opt-in simulation. The interposer, J1 envelope, F1 solid
envelope, optical-frame baselines, and box-derived body inertia also remain
provisional. They do not change the validated real D435i calibration.

Official CAD source metadata, exactness checks, licensing status, and download
instructions are in
[`docs/vendor_assets/pgi_140_80.md`](docs/vendor_assets/pgi_140_80.md). The
complete ten-step plan is in
[`docs/gripper_simulation_workflow.md`](docs/gripper_simulation_workflow.md).

## PGI Gazebo, simulated control, and MoveIt

The first four Gazebo stages are implemented on the feature branch:

1. spawn the UR10e, PGI gripper, and provisional D435i assembly in Gazebo;
2. control the symmetric jaws through `gz_ros2_control` and the standard
   `control_msgs/action/GripperCommand` interface;
3. load simulation-only MoveIt semantics for the arm, gripper, camera adapter,
   touch links, grasp center, and an independent staging cup, plus registered
   RGB-D topics and an observation-only cup detector;
4. plan and execute a guarded logical side grasp using MoveIt, switch the cup
   between world and attached-object ownership, lift it, place it back, detach,
   and return to the camera-ready pose.

Launch the isolated simulation with both Gazebo and RViz visible:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/pgi_140_80_gazebo.launch.py
```

The launch defaults to ROS domain `92` and Gazebo partition
`pgi_140_80_domain_92`. It does not start MoveIt, perception,
`ur_robot_driver`, or RS485. The arm trajectory controller is loaded inactive;
only the simulated jaw controller is active.

The standard action is:

```text
/pgi_gripper_controller/gripper_cmd
```

Its `position` is one jaw's displacement: `0.000 m` is closed and `0.040 m` is
fully open. The right jaw follows symmetrically, so the combined stroke is
80 mm. Run the guarded simulation-only close/open check in another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_gazebo_control.py
```

The verifier refuses ROS domain 0 and requires Gazebo `/clock` plus both PGI
joint states before it sends any jaw goal. Stop the launch with `Ctrl-C`.

Launch the complete Stage-3 view with Gazebo and MoveIt RViz:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/pgi_140_80_gazebo_moveit.launch.py
```

The combined launch opens Gazebo, MoveIt RViz, and an `rqt_image_view` window
for the detection overlay. Use `launch_camera_view:=false` to omit only the
image window, or `launch_cup_perception:=false` to omit the basic detector. The
registered simulation camera contract is:

```text
/pgi_d435i/image
/pgi_d435i/depth_image
/pgi_d435i/camera_info
```

The observation-only outputs are:

```text
/pgi/perception/cup_observation
/pgi/perception/cup_grasp_pose
/pgi/perception/cup_status
/pgi/perception/cup_debug_image
```

The detector synchronizes RGB, depth, and camera intrinsics, segments the blue
cup, back-projects its depth pixels, and transforms the result into
`base_link`. Topic names and thresholds are configurable in
`config/pgi_cup_perception.yaml`. This common ROS contract narrows the
simulation-to-real gap: a future aligned D435i source can be remapped to the
same three inputs. The current HSV segmentation and provisional camera pose
are not production perception or a real hand-eye calibration.

MoveIt is deliberately plan-only: trajectory execution and controller
switching are disabled. The controller mapping is loaded for validation, but
no arm or jaw trajectory is sent by this launch. Verify the current model,
collision matrix, IK, and plan-only paths with:

```bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_moveit.py --require-cup
```

The default `pgi_staging_cup` is 205 mm high and tapers from a 50 mm bottom
diameter to a 120 mm top diameter. Eight stacked bands provide matching visual
and conservative collision geometry. The saved provisional mass is 0.15 kg,
and the centered exposed straw is 4 mm radius / 70 mm long. Its default base
pose is `(0.481542, 0.208414, 0.0)` metres: the saved
X/Y location is retained, while the old floating calibration height is removed
so the cup bottom sits on the ground. The pose can be changed consistently in
Gazebo and MoveIt with `cup_x`, `cup_y`, and `cup_z`. Use
`spawn_cup:=false` to omit it.

The recommended external grasp center is 40 mm above the cup base. With the
provisional 50 mm-long J1 fingers, the covered height is 15-65 mm and the cup
diameter over that band is approximately 55-72.2 mm, below the verified 80 mm
maximum opening. This is a geometric candidate only; force, friction, and
grasp stability remain unvalidated.

The cup remains static in Gazebo during Stage 4. The logical runner changes its
MoveIt ownership to an attached object and updates its Gazebo pose
kinematically while it is held; it does not claim force closure or friction.

Launch the visible, inert-by-default Stage-4 scene on an isolated nonzero ROS
domain:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ros2 launch launch/pgi_140_80_logical_grasp.launch.py \
  ros_domain_id:=106 demo_mode:=none
```

Run a plan-only validation from another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_logical_grasp_demo.py \
  --ros-args --params-file config/pgi_logical_grasp.yaml
```

Simulation execution requires both explicit flags and refuses ROS domain 0:

```bash
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_logical_grasp_demo.py \
  --execute-sim --confirm-simulation \
  --ros-args --params-file config/pgi_logical_grasp.yaml
```

The initial high branch change and final return use collision-checked MoveIt
Pilz PTP trajectories. Transfer, staging, oblique 15-degree side approach,
120 mm lift/place, retreat, and unstage use MoveIt Cartesian paths. The
top-down candidate is rejected because the 120 mm rim exceeds the 80 mm jaw
opening and its checked pose collides with the tool/camera. Stage 5 physical
contact, friction, release, and drop testing remains pending.
