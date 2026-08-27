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

All six Gazebo stages are implemented on the feature branch:

1. spawn the UR10e, PGI gripper, and provisional D435i assembly in Gazebo;
2. control the symmetric jaws through `gz_ros2_control` and the standard
   `control_msgs/action/GripperCommand` interface;
3. load simulation-only MoveIt semantics for the arm, gripper, camera adapter,
   touch links, grasp center, and an independent staging cup, plus registered
   RGB-D topics and an observation-only cup detector;
4. plan and execute a guarded logical side grasp using MoveIt, switch the cup
   between world and attached-object ownership, lift it, place it back, detach,
   and return to the camera-ready pose;
5. repeat the side grasp with a dynamic 0.15 kg cup and native DART contact,
   verify a 120 mm lift/hold, place the cup on the ground, release it, and
   return without setting or kinematically following the cup pose;
6. expose the complete current cup-handling cycle through a simulation-only,
   parameter-free high-level tool boundary with plan and execute operations.

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

### Experimental unknown-object grasp proposals

For cups whose shape, image position, or orientation is not known in advance,
the feature branch contains an opt-in connection to the official
Grasp-Anything RGB model. Install the pinned CPU-only dependency once:

```bash
cd /home/dase-hw101/ur_drinking_project
scripts/setup_grasp_anything_cpu.sh
```

Then launch it with the isolated Gazebo scene:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ros2 launch launch/pgi_140_80_physical_grasp.launch.py \
  ros_domain_id:=106 demo_mode:=none \
  launch_cup_perception:=false launch_camera_view:=false \
  launch_grasp_anything:=true launch_grasp_anything_view:=true \
  controllers_file:=/home/dase-hw101/ur_drinking_project/config/pgi_140_80_grasp_anything_controllers.yaml
```

The opt-in outputs are deliberately separate from the established detector:

```text
/pgi/grasp_anything/candidate_pose
/pgi/grasp_anything/candidate
/pgi/grasp_anything/status
/pgi/grasp_anything/debug_image
```

The neural model proposes a 2-D parallel-jaw centre and closing direction. An
object-agnostic registered-depth mask first crops the largest visible object
above the ground; it assumes neither cup colour nor cup dimensions. Registered
depth then measures the physical jaw opening, fits a local surface, and turns
the result into an observation-only pose in `base_link`. The adapter checks
model score, object membership, depth support, surface fit, the calibrated
camera self-mask, and the PGI 5-80 mm opening range. It never calls MoveIt or a
controller. The opt-in `pgi_grasp_anything_moveit_demo.py` bridge subscribes to
the pose plus matching metadata, verifies the pinned model checksum and source
timestamp, builds segmented depth collision geometry, and lets MoveIt reject
unreachable or colliding candidates. A seriously occluded or mechanically
infeasible candidate is refused.

In the recorded upright overhead test, the candidate was on the cup but the
depth silhouette required 120.1 mm, so the 80 mm PGI correctly refused it. In
the current horizontal test, the candidate is at the narrow end with a 69.675
mm depth opening and score 0.681. The bridge uses the same registered depth
frame to estimate the object's principal axis and slides the candidate 21.1 mm
toward the closest observed body band below the 80 mm stroke. The resulting 76
mm grasp uses a 70-degree downward oblique approach and horizontal jaw closing.

Verify a running observation pipeline without any control command:

```bash
ROS_DOMAIN_ID=106 scripts/verify_pgi_grasp_anything_perception.py --timeout 20
```

The verifier refuses domain 0, validates the pinned model checksum, requires
advancing Gazebo time and fresh debug images, and accepts either a structured
candidate or a structured geometric refusal as a healthy perception result.

Run the connected MoveIt preflight without sending any trajectory:

```bash
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_grasp_anything_moveit_demo.py \
  --ros-args --params-file config/pgi_grasp_anything_moveit.yaml
```

The connected route uses collision-checked Pilz PTP for the high transfer and
coarse-staging transition. The precise `staging -> pre-grasp -> grasp -> lift`
segments remain Cartesian with the image-derived final orientation. A recorded
plan-only run validated the complete approach, attached lift/place, retreat,
coarse-staging return, and camera-ready return.

Gazebo-only execution is explicitly gated:

```bash
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_grasp_anything_moveit_demo.py \
  --execute-sim --confirm-simulation \
  --ros-args --params-file config/pgi_grasp_anything_moveit.yaml
```

The horizontal-cup contact experiment is not yet a successful grasp/place
qualification. Coarse staging and contact execute correctly, but the best 40 N
run rotated the cup 31.28 degrees during lift; the unchanged 15-degree guard
stopped the workflow. An 80 N trial rotated 35.90 degrees, so the remaining
issue is contact geometry/force closure rather than insufficient command
force. No threshold was relaxed and no result is approved for real execution.

Exact source, checksum, preprocessing deviation, self-mask assumption, and
limitations are recorded in
[`docs/vendor_assets/grasp_anything.md`](docs/vendor_assets/grasp_anything.md).

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
opening and its checked pose collides with the tool/camera.

### Stage 5 native-contact grasp

Stage 5 is isolated from the saved logical baseline. It uses a separate dynamic
cup model, world, controller parameters, launch file, and opt-in
`pgi_contact_physics` Xacro flag. Launch the visible scene; it remains inert by
default:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
cd /home/dase-hw101/ur_drinking_project
ros2 launch launch/pgi_140_80_physical_grasp.launch.py \
  ros_domain_id:=106 demo_mode:=none
```

Plan-only validation is the default behavior of the runner:

```bash
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_physical_grasp_demo.py \
  --ros-args --params-file config/pgi_physical_grasp.yaml
```

Explicit simulation execution requires both flags:

```bash
ROS_DOMAIN_ID=106 /usr/bin/python3 scripts/pgi_physical_grasp_demo.py \
  --execute-sim --confirm-simulation \
  --ros-args --params-file config/pgi_physical_grasp.yaml
```

MoveIt plans every arm segment. During the final approach only the two exact
finger-tip/cup pairs are temporarily allowed in the collision matrix; the cup
may not contact `pgi_body`, the camera assembly, the wrist, or other links.
MoveIt attached-object ownership is used only for carried-object collision
planning. The runner never calls the Gazebo set-pose service and measures the
dynamic cup pose before contact, after lift/hold, and after release.

The validated default uses a 32 mm radial backoff, 40 N maximum effort per jaw,
a 120 mm requested lift, and ground-supported release. One recorded run lifted
the cup 119.02 mm, held it for 2 seconds with 0.0027 mm pose drift, limited cup
tilt to 5.99 degrees while carried, released it at 1.36 degrees, and returned
with the arm controller inactive. Provisional friction coefficients are 1.5
for the fingers and 1.2 for the cup; the cup inertia is a documented cylinder
approximation, not measured physical data.

Free-release trials are retained as negative results: opening at 20 mm and 5
mm above the ground tipped the tall tapered cup by 80.33 and 80.32 degrees.
The stable default therefore places the cup base on the ground before opening.
This is a simulation-model conclusion, not a real gripper payload or drop
qualification.

Stage 5 additionally enables `relax_transit_flange_orientation` with a bounded
30-degree local-Z spin at the high transfer pose. Only the cup-clear
transfer-to-side-ready and reverse legs use collision-checked Pilz PTP without
a Cartesian flange-orientation lock. MoveIt chooses the joint motion; the code
never commands `wrist_3_joint` directly. The exact side-ready, approach, grasp,
lift, place, and release orientations remain locked. A measured simulation run
produced 29.9995 degrees of wrist-3 excursion and 30.0000 degrees of flange
rotation while preserving the native-contact grasp. Stage 4 keeps the option
disabled, and the real feeding workflow's tool-axis constraint is unchanged.

### Stage 6: reusable high-level simulation tools

Keep the Stage-5 launch above running with `demo_mode:=none`. Then validate the
new high-level tool surface without initializing ROS:

```bash
scripts/codex_pgi_simulation.sh \
  --tool plan_cup_grasp_cycle --validate-only
scripts/codex_pgi_simulation.sh \
  --tool execute_cup_grasp_cycle --validate-only
```

Run camera-target checks and the complete MoveIt preflight without sending a
trajectory:

```bash
ROS_DOMAIN_ID=106 scripts/codex_pgi_simulation.sh \
  --tool plan_cup_grasp_cycle
```

Run one complete Gazebo-only cup cycle:

```bash
ROS_DOMAIN_ID=106 scripts/codex_pgi_simulation.sh \
  --tool execute_cup_grasp_cycle \
  --execute-sim --confirm-simulation
```

The adapter has a project lock, refuses ROS domain 0 and real-execution
environment gates, and accepts no runtime joint, pose, cup, force, controller,
or gripper arguments. It starts neither Gazebo nor MoveIt: the separately
launched isolated simulation must already be healthy. It delegates planning
and execution to the validated Stage-5 runner and returns one structured
Stage-6 result.

The tool name deliberately says `cup_grasp_cycle`, not `feed_water`. This
stage closes the current camera-to-cup, native grasp, lift/hold, place/release,
and return loop. A human, mouth target, grasped-cup straw transform, and
simulated drinking interaction are not modeled and are not claimed complete.
The next project roadmap item is a separate real RS485 backend; it must not be
connected until the physical integration gates in Steps 9-10 are satisfied.

A fresh-launch Stage-6 run measured a 119.12 mm lift, 0.0024 mm hold drift,
5.45-degree maximum carried tilt, and ended with the cup detached and arm
controller inactive. An earlier long-lived physics session produced a guarded
80.31-degree slip during lift; the tool stopped safely. Restart the isolated
Gazebo launch after any contact failure because the tool intentionally does
not teleport the cup back to its initial pose. Repeated trials remain pending
before the provisional contact model can be called statistically reliable.
