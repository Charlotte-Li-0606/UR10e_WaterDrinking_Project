# PGI-140-80 gripper simulation workflow

This is the retained project plan for the exact configured gripper
`PGI-140-80-W-S-M1-L5-J1-F1-01`. The original description scope was Steps
1-3. Later simulation increments now cover Gazebo model/control, MoveIt,
logical ownership, and provisional DART contact through Step 7. Nothing here
authorizes physical robot motion or RS485 communication.

## Status

| Step | Work item | Status |
| --- | --- | --- |
| 1 | Preserve and version the successful feeding baseline | Complete — baseline branch and annotated tag published |
| 2 | Obtain and verify exact official PGI-140-80, J1 fingertip, and F1 flange CAD | Current scope — official sources identified; exact download/checksum and redistribution permission unresolved |
| 3 | Build a parameterized URDF/Xacro model and verify it in RViz | Current scope — provisional kinematic model implemented; exact mesh replacement remains gated by Step 2 |
| 4 | Add `gz_ros2_control` and a standard `GripperCommand` simulation interface | Complete in isolated Gazebo simulation |
| 5 | Add the gripper, camera adapter, fingers, and grasp center to MoveIt | Complete — isolated, plan-only MoveIt model, RGB-D observer, and staging cup verified |
| 6 | Validate logical cup grasping using attach/detach before contact physics | Complete — camera-derived side grasp, MoveIt route, ownership, lift/place, detach, and return verified |
| 7 | Add physical contact, friction, cup mass, grasp/release, and drop testing | Complete in provisional DART model — 120 mm native lift and ground-supported release pass; 20/5 mm free-drop trials tip the cup and remain recorded negative results |
| 8 | Integrate the simulated gripper with reusable high-level tools | Complete for the current camera-to-cup grasp/lift/place cycle; simulated mouth feeding is not modeled |
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

The simulation default is `xyz="0 0.085 0.030"` and
`rpy="0 -1.57079632679 0"`. In the provisional PGI convention this places the
D435i on the gripper's `+Y` front side and points optical +Z down. The rendered
RGB-D sensor origin is aligned with the provisional depth optical baseline so
the image back-projection and published TF use the same origin. These values
are simulation-only assumptions, not a replacement for real adapter CAD or
hand-eye calibration.

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
| 4 | Validate logical cup attach/detach and ownership before contact physics | Complete |
| 5 | Add contact, friction, cup mass, grasp/release, and drop tests | Complete in provisional DART model — stable ground-supported release selected after 20/5 mm drop trials tipped the cup |
| 6 | Integrate the complete simulated feeding workflow through reusable high-level tools | Complete for the current camera-to-cup grasp/lift/place cycle; the boundary intentionally uses the narrower `cup_grasp_cycle` name |

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
state accurately enough for the kinematic control test, and Stage 5 later
demonstrates empirical contact lift with this model. That result is not proof
that the provisional coupling or force distribution predicts the real PGI.
The existing detailed UR mesh collision warnings under DART also remain
outside this jaw-control milestone. PGI uses simple collision primitives for
this phase.

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

`models/pgi_staging_cup/model.sdf` uses the saved provisional 0.15 kg mass and
205 mm body height, but incorporates the measured taper: 50 mm bottom diameter
to 120 mm top diameter. Eight stacked cylinder bands are shared by the visual
and conservative collision approximation. The centered exposed straw remains
4 mm radius / 70 mm long. A project-created
`models/pgi_staging_cup/meshes/tapered_cup.dae` preserves the smooth reference
envelope. The runtime model uses primitive bands because Ogre2 did not
reliably import the intended blue material from that COLLADA file; explicit
SDF primitive material gives the detector deterministic input. The combined
launch supplies the same parameterized base pose to Gazebo and to the
MoveIt planning scene. It retains the previously saved X/Y location but
replaces the old floating calibration height with a ground-level cup base:

```text
cup_x=0.481542  cup_y=0.208414  cup_z=0.0
```

The recommended external grasp center is 40 mm above the base. The provisional
50 mm J1 finger span covers heights 15-65 mm, where the linear taper is about
55-72.2 mm in diameter and therefore fits inside the 80 mm opening. The cup
remains a static, independent world object and is explicitly verified as not
attached; this does not implement Stage 4 ownership or Stage 5 contact physics.

The same combined launch now starts an ideal registered Gazebo RGB-D source,
bridges RGB, depth, and `CameraInfo` to ROS, and runs
`scripts/pgi_cup_perception.py`. The basic observer synchronizes those inputs,
segments the blue cup, back-projects median cup depth, transforms the result at
the image timestamp, and publishes observation and grounded grasp poses in
`base_link`. It uses a separate TF executor callback group so timestamped TF
updates continue while image processing waits briefly for a transform.

This narrows the later simulation-to-real integration gap because aligned real
D435i topics can be remapped to the same input contract without changing the
published pose contract. It does not make simulation calibration valid on the
real adapter. HSV segmentation, the upright-ground assumption, the front
camera transform, and the grasp height are provisional; no planning or
execution command is issued by the observer.

Launch the visible combined stage:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/pgi_140_80_gazebo_moveit.launch.py
```

The launch opens Gazebo, the proven UR MoveIt RViz layout, and
`rqt_image_view` on `/pgi/perception/cup_debug_image`. Set
`launch_camera_view:=false` to omit that image window or
`launch_cup_perception:=false` to omit the observer.

Then run the simulation-only plan checks:

```bash
cd /home/dase-hw101/ur_drinking_project
ROS_DOMAIN_ID=92 scripts/verify_pgi_140_80_moveit.py --require-cup
```

The verifier refuses domain 0, requires advancing Gazebo time and complete
joint states, confirms the cup dimensions and unattached status, validates the
collision matrix and current state, solves IK for `pgi_grasp_center`, and plans
small arm and jaw changes without calling any execution action.

### Gazebo stage 4 implementation

`launch/pgi_140_80_logical_grasp.launch.py` composes the existing Gazebo,
MoveIt, RViz, RGB-D observer, and RQT components without changing their normal
defaults. It starts from the established camera-ready pose and remains inert
unless a separate plan or explicit simulation execution is requested.

`scripts/pgi_logical_grasp_demo.py` refuses ROS domain 0, confirms advancing
simulation time, validates the controller states and world/base alignment,
freezes a fresh camera target, and checks that it agrees with the Gazebo model
pose. It evaluates and rejects the top-down grasp: the 120 mm rim cannot pass
through the 80 mm opening and the checked state collides with the cup/tool
assembly.

The selected route is a 15-degree downward oblique radial side grasp. MoveIt
Pilz PTP changes from camera-ready to the high transfer state on the validated
low-side IK branch. MoveIt Cartesian paths then cover transfer, side-ready,
staging, pre-grasp, grasp, 120 mm lift/place, retreat, and unstage. The reverse
Pilz PTP returns to camera-ready. OMPL is not used by this logical workflow.
Every trajectory is planned before the simulation arm controller is activated.

At the logical hold, the MoveIt cup collision object changes from world
ownership to an `AttachedCollisionObject` on `pgi_grasp_center`; the configured
touch links are limited to the gripper. Gazebo's static cup pose follows the
same relative transform through its isolated set-pose service. After place,
the object is restored to world ownership, the gripper opens, and the complete
retreat is executed. A successful run ends with the cup detached, on the
ground, the camera detecting it again, and the arm controller inactive.

This deliberately does not use contact force or friction to hold the cup. The
logical jaw position stops before the measured static-cup contact point. The
wall-clock timeout scales from trajectory simulation time only to tolerate a
Gazebo real-time factor below 1.0; it does not alter path speed or collision
checking.

### Gazebo stage 5 implementation

Stage 5 is opt-in and does not change the Stage-4 logical baseline. The
separate files `worlds/pgi_140_80_physical_grasp.sdf`,
`models/pgi_physical_cup/model.sdf`, and
`config/pgi_140_80_physical_controllers.yaml` enable a dynamic cup and the
contact-specific DART/controller parameters only for
`launch/pgi_140_80_physical_grasp.launch.py`. The launch is inert unless an
operator separately runs the guarded simulation runner.

The dynamic cup retains the saved 0.15 kg mass and eight-band 50-to-120 mm
taper. Its 0.129 m center of mass is the calculated solid-frustum volume
centroid; its cylinder-envelope inertia remains provisional. Finger/cup
friction values of 1.5/1.2 and the contact stiffness/damping are simulation
tuning values, not measured material properties. The gripper uses the
documented minimum selectable 40 N limit per jaw.

`scripts/pgi_physical_grasp_demo.py` reuses the camera-derived Stage-4 route,
but uses a 32 mm radial backoff so the cup sits 8 mm deeper between the fingers.
A 20 mm trial was rejected because it caused `pgi_body` contact. For the final
approach, the runner temporarily permits only
`pgi_left_finger`/`pgi_staging_cup` and
`pgi_right_finger`/`pgi_staging_cup` in MoveIt's allowed-collision matrix, then
restores the original matrix. All arm trajectories remain MoveIt planned and
collision checked. The attached collision object is planning-scene ownership
only: the runner never calls `set_pose`, never follows the cup kinematically,
and verifies motion from `/model/pgi_staging_cup/pose`.

The Stage-5 configuration now opts into the simulation-only
`relax_transit_flange_orientation` experiment. The cup-clear
`transfer <-> side-ready` legs use collision-checked Pilz PTP instead of a
fixed-orientation Cartesian segment. A bounded 30-degree local-Z spin at the
high transfer pose demonstrates that MoveIt may select `wrist_3_joint`; no
individual wrist joint is commanded directly. The spin is removed before
side-ready. The approach, grasp, attached lift/place, and release continue to
use the exact validated gripper orientation. The Stage-4 logical configuration
keeps this option disabled, and none of these parameters apply to the real
feeding workflow or relax its tool-axis requirement.

Progressive tests produced these results:

- 30 mm proof lift: 28.99 mm measured rise, 6.00 degree maximum carried tilt,
  and 0.0026 mm drift during the two-second hold; ground release passed.
- 120 mm full lift: 119.02 mm measured rise, 5.99 degree maximum carried tilt,
  and 0.0027 mm drift during the two-second hold; ground-supported release
  finished at 1.36 degrees and the complete return passed.
- 20 mm and 5 mm free-release trials: the tall tapered empty cup tipped by
  80.33 and 80.32 degrees. These are retained negative drop-test results. The
  stable default first places the cup on the ground and then opens the jaws.
- 30-degree free-space flange-spin trial: MoveIt planned and executed
  29.9995 degrees of `wrist_3_joint` excursion and 30.0000 degrees of measured
  flange rotation. Native contact still lifted the cup 118.94 mm, held it with
  0.0025 mm drift, placed and released it, and returned with the arm controller
  inactive.

Launch and plan without execution:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ros2 launch launch/pgi_140_80_physical_grasp.launch.py \
  ros_domain_id:=106 demo_mode:=none
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_physical_grasp_demo.py \
  --ros-args --params-file config/pgi_physical_grasp.yaml
```

Simulation execution additionally requires `--execute-sim` and
`--confirm-simulation`. The runner refuses ROS domain 0, requires advancing
Gazebo time, leaves MoveGroup plan-only, and deactivates the isolated arm
controller at completion or failure.

### Gazebo stage 6 implementation

Stage 6 does not duplicate any planning or contact code. The pure validation
module `robot_layer/arm_ur10e/agent_tools/pgi_simulation_tools.py` exposes only:

```text
plan_cup_grasp_cycle
execute_cup_grasp_cycle
```

Both tools accept an empty argument object only. Joint values, poses, cup
positions, force, speed, controller, gripper, attach/detach, and wrist-3
commands cannot cross this boundary. The runner delegates to the fixed
Stage-5 physical workflow and its versioned parameter file.

`scripts/codex_pgi_simulation.sh` is the canonical entrypoint. It defaults to
isolated ROS domain 106, refuses domain 0, removes inherited real-execution
environment gates, and never launches `ur_robot_driver`, RS485, Gazebo, or
MoveIt. A single-workflow lock prevents two planning/execution clients from
changing the same simulation scene concurrently. Execution additionally
requires both `--execute-sim` and `--confirm-simulation`.

Examples:

```bash
# Pure request validation; no ROS initialization.
scripts/codex_pgi_simulation.sh \
  --tool execute_cup_grasp_cycle --validate-only

# Complete perception and MoveIt preflight; no trajectory is sent.
ROS_DOMAIN_ID=106 scripts/codex_pgi_simulation.sh \
  --tool plan_cup_grasp_cycle

# One complete native-contact Gazebo cycle.
ROS_DOMAIN_ID=106 scripts/codex_pgi_simulation.sh \
  --tool execute_cup_grasp_cycle \
  --execute-sim --confirm-simulation
```

The returned JSON distinguishes Stage 6 from the delegated Stage-5 report and
always states `simulation_only` and `real_robot_command_sent`. The label is
deliberately `cup_grasp_cycle`: no human, mouth target, straw-on-grasped-cup
transform, or drinking interaction exists in the current Gazebo world.

The recorded Stage-6 execution from a fresh launch passed with 3.27 mm camera
target/model XY error, 119.12 mm measured lift, 0.0024 mm hold drift, 5.45
degrees maximum observed carried tilt, and 1.63 degrees tilt after release.
The cup was detached and the arm controller was inactive at completion.

One earlier run against a long-lived, previously exercised physics session
slipped during lift and reached 80.31 degrees. The 15-degree guard stopped the
workflow, restored MoveIt ownership, and deactivated the arm controller. A
fresh simulation launch then passed. After any physical-contact failure,
restart the isolated Gazebo launch before rerunning; the Stage-6 tool does not
silently teleport or reset the dynamic cup. More repeated trials are still
needed before claiming statistical grasp reliability.

### Experimental Grasp-Anything perception branch

`feature/grasp-anything-simulation` adds an opt-in, observation-only perception
experiment after the reproducible six-stage baseline. It does not replace the
fixed Stage-5/6 cup target and is not connected to its planner or controller.

The official RGB-only Grasp-Anything checkpoint proposes 2-D parallel-jaw
centres and closing directions. A local ROS adapter uses transformed registered
depth to crop the largest object above the ground without colour or fixed cup
dimensions, measures physical opening from its silhouette, fits a visible
surface normal, checks a calibrated camera self-mask, and publishes a candidate
pose in `base_link`. Missing depth, excessive occlusion, poor surface fit, and
openings outside the PGI 5-80 mm range are refusals.

The upright overhead scene produced a correct 120.1 mm over-stroke refusal.
The horizontal scene produced a 66.635 mm observation-only candidate at the
narrow end with 0.697 model score and 98.8% local depth support. The verifier
requires a pinned model checksum, advancing Gazebo clock, fresh debug images,
structured observation-only status, and refuses ROS domain 0.

Current scope is perception evidence only. Before it can replace the known-cup
target, the following work remains:

1. add an active side/oblique camera view so a wide tapered cup can expose a
   graspable lower band;
2. extend the completed single upright/horizontal checks to supported tilted
   and partially occluded scenes, including correct refusal under severe
   occlusion, then repeat every case statistically;
3. select the intended object without relying on colour or a fixed cup model;
4. convert the visible-surface candidate to PGI TCP, pre-grasp, and retreat
   poses while accounting for finger length and cup thickness;
5. pass every candidate through MoveIt IK, reachability, collision, approach,
   closure, lift, and stability checks before allowing simulated execution.

Source provenance and the limitations of the 2-D checkpoint are in
`docs/vendor_assets/grasp_anything.md`.

## Later workflow, intentionally not implemented

Step 9 implements a separate real Modbus RTU/RS485 backend. Step 10 is a mandatory
physical-integration review of payload, center of gravity, TCP, hand-eye
transform, and all collision geometry before any real execution can be
considered.
