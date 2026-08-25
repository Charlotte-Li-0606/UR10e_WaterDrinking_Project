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
D435i is the black rectangular body beside the gripper; the colored axes named
`D435i camera frame` mark `d435i_link`. The separate `PGI grasp center` axes
mark the future cup attachment frame. Use the mouse wheel to zoom out from the
tool-focused view when the complete UR10e needs to be inspected.

The `camera_mount_xyz` and `camera_mount_rpy` arguments are provisional and
exist only in the opt-in simulation. The interposer, J1 envelope, F1 solid
envelope, optical-frame baselines, and box-derived body inertia also remain
provisional. They do not change the validated real D435i calibration.

Official CAD source metadata, exactness checks, licensing status, and download
instructions are in
[`docs/vendor_assets/pgi_140_80.md`](docs/vendor_assets/pgi_140_80.md). The
complete ten-step plan is in
[`docs/gripper_simulation_workflow.md`](docs/gripper_simulation_workflow.md).

## PGI Gazebo, simulated control, and MoveIt

The first three Gazebo stages are implemented on the feature branch:

1. spawn the UR10e, PGI gripper, and provisional D435i assembly in Gazebo;
2. control the symmetric jaws through `gz_ros2_control` and the standard
   `control_msgs/action/GripperCommand` interface;
3. load simulation-only MoveIt semantics for the arm, gripper, camera adapter,
   touch links, grasp center, and an independent staging cup.

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

MoveIt is deliberately plan-only: trajectory execution and controller
switching are disabled. The controller mapping is loaded for validation, but
no arm or jaw trajectory is sent by this launch. Verify the current model,
collision matrix, IK, and plan-only paths with:

```bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_moveit.py --require-cup
```

The default `pgi_staging_cup` reuses the saved cup data: 50 mm radius,
205 mm body height, 0.15 kg mass, and a centered 4 mm radius / 70 mm exposed
straw. Its default base pose is `(0.481542, 0.208414, 0.0)` metres: the saved
X/Y location is retained, while the old floating calibration height is removed
so the cup bottom sits on the ground. The pose can be changed consistently in
Gazebo and MoveIt with `cup_x`, `cup_y`, and `cup_z`. Use
`spawn_cup:=false` to omit it.

The cup is static and independent in Stage 3. It is a collision obstacle, not
an attached object, and it does not yet follow `pgi_grasp_center`. Logical
attach/detach ownership is the next pending stage; physical contact, friction,
release/drop testing, and simulated feeding remain later stages.
