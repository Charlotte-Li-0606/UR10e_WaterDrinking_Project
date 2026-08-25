# PGI-140-80 gripper simulation workflow

This is the retained project plan for the exact configured gripper
`PGI-140-80-W-S-M1-L5-J1-F1-01`. The original description scope was Steps
1-3. The current simulation increment also completes Gazebo model validation
and simulated jaw control. Nothing here authorizes physical robot motion or
RS485 communication.

## Status

| Step | Work item | Status |
| --- | --- | --- |
| 1 | Preserve and version the successful feeding baseline | Complete — baseline branch and annotated tag published |
| 2 | Obtain and verify exact official PGI-140-80, J1 fingertip, and F1 flange CAD | Current scope — official sources identified; exact download/checksum and redistribution permission unresolved |
| 3 | Build a parameterized URDF/Xacro model and verify it in RViz | Current scope — provisional kinematic model implemented; exact mesh replacement remains gated by Step 2 |
| 4 | Add `gz_ros2_control` and a standard `GripperCommand` simulation interface | Complete in isolated Gazebo simulation; native contact coupling remains a later physics task |
| 5 | Add the gripper, camera adapter, fingers, and grasp center to MoveIt | Complete — isolated, plan-only MoveIt model and staging cup verified |
| 6 | Validate logical cup grasping using attach/detach before contact physics | Pending |
| 7 | Add physical contact, friction, cup mass, grasp/release, and drop testing | Pending |
| 8 | Integrate the simulated gripper with reusable high-level tools | Pending |
| 9 | Replace the simulation backend with the real RS485 backend | Pending |
| 10 | Recalculate payload, center of gravity, TCP, camera transform, and collision geometry before real execution | Pending |

## Intended transform tree

```text
tool0
  -> pgi_camera_interposer
      -> pgi_mount
          -> pgi_body
              -> pgi_left_finger
              -> pgi_right_finger
              -> pgi_grasp_center
      -> d435i_mount
          -> d435i_link
              -> d435i_color_optical_frame
              -> d435i_depth_optical_frame
```

The camera remains rigidly attached to the interposer, never to a finger.
`pgi_grasp_center` is a logical frame. The Stage-3 cup is an independent world
object; its future attached-object and straw-tip frames remain deliberately
absent from the gripper body.

## Step 1 — preserve the baseline

1. Audit Git status, current branch, remotes, recent commits, untracked and
   ignored files, file sizes, and tracked/staged content for secrets.
2. Preserve user files and generated evidence on disk; do not reset, clean,
   delete, or overwrite them.
3. Ignore credentials, build/install/log products, environments, ROS bags,
   generated execution reports, recordings, vendor downloads, and CAD
   conversion intermediates.
4. Commit the behavior-preserving baseline and create the annotated tag
   `pre-pgi-gripper-baseline-2026-08-25`.
5. Publish the baseline branch and tag without force-pushing.
6. Create `feature/pgi-140-80-simulation`; keep all gripper work there and do
   not merge it into `main` yet.

## Step 2 — exact official CAD

Use only DH Robotics assets for the exact PGI body and matching J1/F1 parts.
The PGIA family must not silently replace PGI: first verify its dimensions,
mounting faces, cable exit, finger carriage, and F1 interface against the PGI
drawing. Do not use third-party CAD.

For every retrieved asset, record:

- official source URL and retrieval date;
- exact model variant and archive filename;
- SHA-256 checksum;
- CAD units and model scale;
- assembly origin and X/Y/Z axis convention;
- opening zero and travel convention;
- J1 and F1 part identity;
- applicable license or written redistribution permission.

Keep raw vendor files in the gitignored directory
`assets/vendor/dh_robotics/pgi_140_80/`. Commit only metadata, checksums,
download/conversion instructions, and legally redistributable project-created
meshes. If redistribution permission remains unclear, never commit the vendor
archive or derivatives that reproduce protected detailed geometry.

The current evidence and unresolved download details are recorded in
[`vendor_assets/pgi_140_80.md`](vendor_assets/pgi_140_80.md).

## Step 3 — parameterized description and RViz

### Architecture

Extend the existing `urdf/ur_gz_feeding_markers.urdf.xacro`; do not establish a
second UR description. The reusable gripper macro is
`urdf/pgi_140_80_macro.xacro`. The canonical option defaults to:

```text
use_pgi_gripper:=false
```

With the option false, the established simulation cup/camera description is
retained. With it true, that incompatible fixed tool is omitted and the PGI
assembly is added. The real D435i calibration file is never read or modified by
this option.

### Kinematics

- `pgi_left_finger_joint`: prismatic, axis `+X`, range `0.000-0.040 m`.
- `pgi_right_finger_joint`: prismatic, axis `-X`, range `0.000-0.040 m`.
- Right mimics left with multiplier `+1.0` and offset `0.0`.
- Zero means closed; positive motion opens both jaws.
- Maximum gap change is `2 x 0.040 = 0.080 m`, matching the documented total
  stroke. This interpretation must be rechecked against the exact J1 CAD.

The body uses the documented 1 kg mass. Its inertia is explicitly a
provisional uniform-box estimate from the documented body envelope. Finger,
interposer, mount, and camera masses are not claimed. Primitive collision
geometry is intentionally conservative and simple.

Detailed vendor visual meshes are not present because the exact asset and its
redistribution license have not been verified. The primitive visuals support
topology, transform, and joint-motion review only; they are not manufacturing
or collision-validation geometry.

### Provisional camera mount

The opt-in model exposes:

```text
camera_mount_xyz
camera_mount_rpy
```

They describe the provisional `pgi_camera_interposer -> d435i_mount`
transform. Final values require adapter design, CAD inspection, hand-eye
calibration, and collision validation. They must never overwrite
`config/ur10e_real/d435i_mount_calibration.json`.

### Verification gates

1. Source ROS 2 Jazzy and the project workspace.
2. Expand both the default and PGI Xacro variants.
3. Run `check_urdf` on both outputs.
4. Reject duplicate link/joint names, invalid topology, or invalid inertia.
5. Confirm PGI links are absent from the default variant.
6. Confirm both jaw limits, opposite axes, mimic relation, and 80 mm total
   stroke.
7. Confirm visual/collision origins agree for every provisional primitive.
8. Display the UR10e + PGI model in RViz using only robot/joint state
   publishers.
9. Confirm the camera branch is fixed to the interposer.
10. Keep the RViz-only launch independent from Gazebo control, MoveIt
    execution, `ur_robot_driver`, and RS485.

Run the non-motion checks with:

```bash
scripts/verify_pgi_140_80_description.sh
```

Display it with:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/display_pgi_140_80.launch.py \
  use_pgi_gripper:=true
```

Move the independent left-jaw slider from `0.000` to `0.040`; the right jaw
must move symmetrically through its mimic relation.

## Current six-stage Gazebo workflow

This shorter sequence is the working implementation order used after the
description/RViz milestone. It is stored here so later conversations use the
same stage names.

| Gazebo stage | Work item | Status |
| --- | --- | --- |
| 1 | Spawn and validate the UR10e, PGI gripper, camera placeholder, inertias, and collision primitives in Gazebo | Complete |
| 2 | Add `gz_ros2_control` and a standard `GripperCommand` jaw interface | Complete |
| 3 | Add the gripper, camera adapter, fingers, touch links, grasp center, and independent cup obstacle to MoveIt | Complete — plan-only |
| 4 | Validate logical cup attach/detach and ownership before contact physics | Pending |
| 5 | Add contact, friction, cup mass, grasp/release, and drop tests | Pending |
| 6 | Integrate the complete simulated feeding workflow through reusable high-level tools | Pending |

### Gazebo stages 1-2 implementation

`launch/pgi_140_80_gazebo.launch.py` reuses the existing UR Gazebo description
and keeps the PGI model opt-in. It starts an isolated ROS domain and Gazebo
partition, but it does not start MoveIt, perception, a real robot driver, or
RS485. The UR joint trajectory controller is loaded inactive, while
`pgi_gripper_controller` is active.

The controller exposes:

```text
/pgi_gripper_controller/gripper_cmd
  control_msgs/action/GripperCommand
```

The commanded position is the left-jaw displacement. `0.000 m` is closed and
`0.040 m` is fully open. The right jaw has state interfaces but no independent
command interface; it mimics the left jaw on its opposite physical axis. The
joint speed is limited to `0.0364 m/s`, approximating the documented 1.1 s
full-stroke time.

The DART-only physical hard stop is `0.0405 m`, providing a 0.5 mm numerical
guard band so a position-controlled jaw does not latch against the exact upper
stop. Controller-manager command-limit enforcement and the ros2_control
interface keep the public range at `0.000-0.040 m`. The guard band is not
commandable travel and does not change the nominal 80 mm total stroke.

Launch the visible simulation:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/pgi_140_80_gazebo.launch.py
```

Then verify close/open motion from a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_gazebo_control.py
```

The test refuses domain 0, requires Gazebo clock data, sends goals only to the
PGI simulated action, measures total stroke and jaw symmetry, and finishes in
the open position.

Gazebo's current DART backend reports that it does not create a native mimic
physics constraint. `gz_ros2_control` nevertheless mirrors the simulated jaw
state accurately enough for the kinematic control test. This does not validate
finger contact forces or object grasping; native coupled contact behavior must
be resolved and tested in Stage 5. The existing detailed UR mesh collision
warnings under DART also remain outside this jaw-control milestone. PGI uses
simple collision primitives for this phase.

### Gazebo stage 3 implementation

`launch/pgi_140_80_moveit.launch.py` uses the same opt-in Xacro as Gazebo and
adds project-local SRDF semantics for `ur_manipulator`, `pgi_gripper`, and the
combined `ur10e_pgi` group. `pgi_140_80` is the MoveIt end effector; the left
jaw is the command joint and the right jaw is passive because it mimics the
left. Internal gripper, fixed camera-adapter, and adjacent-link collision pairs
are disabled explicitly rather than by a broad wildcard.

Stage 3 sets `allow_trajectory_execution=false` and
`moveit_manage_controllers=false`. It loads the arm and standard
`GripperCommand` mappings for consistency checks but never activates the arm
controller, switches controllers, or executes a trajectory. The provisional
jaw acceleration limit of `0.10 m/s^2` exists only so MoveIt can time-parameterize
plan-only results; it is not an official PGI hardware rating.

`models/pgi_staging_cup/model.sdf` reuses the saved 0.15 kg cup model, 50 mm
radius, 205 mm body height, and centered 4 mm radius / 70 mm exposed straw. The
combined launch supplies the same parameterized base pose to Gazebo and to the
MoveIt planning scene. It retains the previously saved X/Y location but
replaces the old floating calibration height with a ground-level cup base:

```text
cup_x=0.481542  cup_y=0.208414  cup_z=0.0
```

Both cup and straw use simple cylinder collision geometry. The cup remains a
static, independent world object and is explicitly verified as not attached;
this does not implement Stage 4 ownership or Stage 5 contact physics.

Launch the visible combined stage:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/pgi_140_80_gazebo_moveit.launch.py
```

Then run the simulation-only plan checks:

```bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_moveit.py --require-cup
```

The verifier refuses domain 0, requires advancing Gazebo time and complete
joint states, confirms the cup dimensions and unattached status, validates the
collision matrix and current state, solves IK for `pgi_grasp_center`, and plans
small arm and jaw changes without calling any execution action.

## Later workflow, intentionally not implemented

Step 6 validates cup ownership with logical attach/detach before any contact
model. Step 7 tunes contact, friction, mass, release, and drop behavior. Step 8
exposes reusable high-level simulated tools. Step 9 implements a separate real
Modbus RTU/RS485 backend. Step 10 is a mandatory physical-integration review of
payload, center of gravity, TCP, hand-eye transform, and all collision geometry
before any real execution can be considered.
