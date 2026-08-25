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

The next pending implementation step is `gz_ros2_control` with a standard
`GripperCommand` simulation interface. It is intentionally not part of this
branch's current implementation.
