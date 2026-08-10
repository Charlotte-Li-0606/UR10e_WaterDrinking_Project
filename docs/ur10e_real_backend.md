# UR10e real-backend setup

The project has one ROS 2 / MoveIt SDK: `UR10eRobotEnv`. Its safe feeding
adapter is `FeedingSkillLibrary`. `UR10E_BACKEND` selects ROS endpoints for
that same implementation; it does not select a second SDK or use Piper code.

## Backend policy

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `UR10E_BACKEND` | `sim` | `sim` uses Gazebo's `joint_trajectory_controller`; `real` uses the UR driver's `scaled_joint_trajectory_controller`. |
| `UR10E_ROBOT_IP` | unset | Required by the manual real-driver start command and any supervised real execution process. It is never stored in project files. |
| `UR10E_ALLOW_REAL_EXECUTION` | `0` | The real `feed_water` adapter rejects execution unless this is exactly `1`. |

Selecting `UR10E_BACKEND=real` does not enable motion. The shared SDK blocks
every executing MoveIt or trajectory action unless the execution variable is
enabled, `/joint_states` is complete, `base_link -> tool0` is available, the
expected scaled controller is active, and its `FollowJointTrajectory` action
server is available. Real planning-only requests do not require the execution
variable. The real backend defaults and caps MoveIt velocity and acceleration
scaling at `0.30`.

Codex is the preferred operator-facing agent and uses the canonical
`feed_water` interface through `scripts/codex_feed_water.sh`. The older
OpenClaw integration remains preserved as a legacy fallback but is not part of
the Codex route. Simulation remains backed by `FeedingSkillLibrary`; with
`UR10E_BACKEND=real`, the safe-tool runner
accepts only one high-level `feed_water` call and delegates it to the same
integrated real state machine. That state machine locks the center-selected
person among the visible mouth candidates, performs the bounded
translation-only active search when that target is absent, freezes the
corrected camera-ray 50 mm pre-mouth target, and uses the wrist OctoMap for
same-target alternate-path replanning. The real branch exposes no arbitrary
joint, pose, trajectory, controller, gripper, direct-mouth, tilt, pour, or
retreat action. It plans by default. Execution requires `--execute`,
`--confirm-real-motion`, and
`UR10E_ALLOW_REAL_EXECUTION=1`, and terminates at a motionless 2–5 second
pre-mouth hold.

The active search freezes the initial flange orientation and applies the
calibrated flange-to-D435i extrinsic to every direction. It first attempts 40
mm opposite the camera optical viewing axis to obtain a wider frame, then
checks 50 mm image-left, image-right, image-up, and image-down offsets around
that wider-view center. If a nominal endpoint is unreachable from the current
fixed-orientation robot pose, it tries bounded smaller distances (30/20 mm for
backward and 40/30/20 mm for a directional scan). Every failed attempt is
classified with an IK/collision/path diagnostic; if all bounded alternatives
for one direction fail, that direction is reported and skipped so the other
directions can still be searched. It never rotates the flange, plans and
collision-checks each translation independently, and requests cancellation as
soon as any face candidate appears. The complete search remains bounded to 15
seconds.

For the final pre-mouth route, the real workflow keeps both the static
collision objects and live wrist-camera OctoMap enabled in one combined
PlanningScene. It first checks an exact-orientation Pilz LIN path. If that path
is clear, it uses it directly; only a rejected direct path triggers the bounded
orientation-constrained OMPL detour to the same frozen target. No obstacle
layer is disabled, and a failure of both planners is reported separately from
IK failure.

For an explicitly requested Codex plan without motion:

```bash
scripts/codex_feed_water.sh --plan-only --hold-duration 3
```

A standalone user-directed drinking request is explicit motion authorization
for the Codex skill. Its guarded execution form is:

```bash
UR10E_ALLOW_REAL_EXECUTION=1 scripts/codex_feed_water.sh \
  --execute --confirm-real-motion --hold-duration 3
```

The guarded pre-mouth planner accepts a pendant speed-slider setting from 5%
through a 60% ceiling. Its per-invocation tool0 displacement ceiling is the
UR10e's nominal 1.30 m reach; target radius, inverse-kinematics, joint-limit,
and collision checks still apply independently. The mature adapter uses 0.30
MoveIt velocity and acceleration scaling for its generated trajectory.
The real camera-ray nominal pre-mouth stand-off is 50 mm. Adaptive goal
selection evaluates 50, 70, 90, 120, and 150 mm candidates, each 30 mm closer
than the preceding 80, 100, 120, 150, and 180 mm policy, while retaining all
IK, collision, clearance, and human-safety checks.
Short-lived execution requests subscribe to the UR driver's latched robot,
safety, and program state with matching transient-local QoS, and use a
one-second stable-mouth sampling window to avoid a separate preflight delay.
During motion, the mouth-drift watchdog ignores isolated wrist-camera depth
transients and cancels only when a fresh stable window of at least three
base-frame samples confirms more than 50 mm of drift. The stricter 30 mm
pre-execution drift guard remains unchanged.

The real camera startup command is
`scripts/run_real_d435i_mouth_perception.sh`. Its default mount file is
`config/ur10e_real/d435i_mount_calibration.json`, containing the physically
validated 2026-07-23 axis correction. Before any real execution, the pre-mouth
pipeline requires `d435i_color_optical_frame`, resolves
`base_link -> d435i_color_optical_frame`, compares live
`tool0 -> d435i_link` against that configuration, and refuses a missing,
uncorrected provisional, or mismatched transform.

## Locally verified Jazzy launch interfaces

The following commands and arguments were read from this ThinkPad's installed
ROS 2 Jazzy packages using `ros2 launch ... --show-args`. Do not run them until
the operator has prepared the real robot, network, safety hardware, and UR
External Control program.

Terminal 1, driver only:

```bash
export UR10E_ROBOT_IP='<robot-ip>'
/home/dase-hw101/ur_drinking_project/scripts/start_ur10e_real_driver.sh
```

This expands to the installed upstream launch interface:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:=<robot-ip> \
  kinematics_params_file:=/path/to/ur10e_calibration.yaml \
  launch_rviz:=false
```

The driver script requires the extracted calibration file at
`config/ur10e_real/ur10e_calibration.yaml` by default (or an explicit
`UR10E_KINEMATICS_FILE`). It unsets `ROS_STATIC_PEERS` and uses a localhost
ROS discovery graph. This does not alter network addresses or the TCP
connection to the UR controller; it prevents a Wi-Fi simulator from
publishing a competing robot description or TF tree.

Terminal 2, MoveIt only after the driver publishes robot description and joint
state:

```bash
/home/dase-hw101/ur_drinking_project/scripts/start_ur10e_real_moveit.sh
```

This starts the project MoveIt wrapper, which supplies kinematics to
`move_group`, loads the wrist point-cloud occupancy updater, and advertises
only the active physical controller:

```bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/ur10e_moveit_with_kinematics.launch.py \
  ur_type:=ur10e launch_rviz:=false launch_servo:=false use_sim_time:=false \
  use_octomap:=true trajectory_controller:=scaled_joint_trajectory_controller
```

The real D435i perception process and `scripts/run_real_depth_to_pointcloud.sh`
must already be publishing fresh raw and MoveIt-filtered point clouds. The
integrated workflow refuses before motion if either stream is missing or more
than 0.75 seconds old.

No project script combines driver startup with a feeding trajectory.

## Read-only verification

With the driver and MoveIt already running, run these read-only checks:

```bash
source /opt/ros/jazzy/setup.bash
ros2 node list
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 run tf2_ros tf2_echo base_link tool0
ros2 action list | rg 'move_action|scaled_joint_trajectory_controller/follow_joint_trajectory'
ros2 param get /move_group robot_description_semantic | rg ur_manipulator
```

The expected active controller is `scaled_joint_trajectory_controller`. The
expected planning group is `ur_manipulator`. `base_link -> tool0` must resolve
before any execution can be considered.

The project status probe sends no action goal and no controller command:

```bash
/home/dase-hw101/ur_drinking_project/scripts/check_ur10e_real_status.sh
```

To ask MoveIt to preflight the fixed straw-tip-to-pre-mouth target without
motion, with wrist camera and MediaPipe mouth perception already publishing:

```bash
/home/dase-hw101/ur_drinking_project/scripts/ur10e_real_plan_only.sh
```

That script forcibly sets `UR10E_ALLOW_REAL_EXECUTION=0` and calls only the
existing `move_tool_to_target(..., execute=False)` path. It can fail safely if
perception, planning scene, TF, or MoveIt is unavailable.

## First physical-motion smoke test

`scripts/real_ur10e_smoke_test.py` is a separate commissioning utility. It
does not use the feeding, LLM, OpenClaw, active-search, or perception paths.
It always selects `UR10E_BACKEND=real` and has a fixed 2 cm `base_link +Z`
tool0 target; the target orientation is copied from the current tool0 pose and
held by Pilz `LIN` Cartesian planning. Pilz LIN makes the tool path linear;
because the start and target orientations are identical, it preserves tool
orientation without relying on an OMPL path-constraint manifold. The utility
accepts no joint target, absolute pose, or user-provided offset.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/dase-hw101/ur_drinking_project
export UR10E_ALLOW_REAL_EXECUTION=0
python3 scripts/real_ur10e_smoke_test.py --mode check
```

`check` is read-only and reports `/joint_states`, `base_link -> tool0`, the
active controller list, and availability of `/move_action` without calling an
action. It does not start MoveIt. After a separately reviewed MoveIt instance
is already running, this remains plan-only:

```bash
python3 scripts/real_ur10e_smoke_test.py --mode plan
```

Execution is intentionally not the default and requires all three gates: a
successful check and plan, `--mode execute --confirm-real-motion`, and the
exact process environment `UR10E_ALLOW_REAL_EXECUTION=1`. The first physical
motion requires a clear test area, the pendant speed slider at 5–30%, a
manually verified payload, and an operator with immediate E-stop access.

## Feeding geometry and frames

`config/ur10e_sdk_config.yaml` is the canonical software configuration for
the calibrated feeding offsets:

- flange/tool0 to camera optical center: `[+0.070, 0.000, -0.015]` in the
  ROS `tool0` frame;
- flange/tool0 to straw tip: `[+0.110, 0.000, 0.000]` in the ROS `tool0`
  frame.

The user's convention is X forward, Y right, Z up. ROS `base_link` convention
is X forward, Y left, Z up, so aligned user coordinates map as
`[x, -y, z]`. With the established flange-down orientation, `tool0 +Z` points
down, which is why the measured camera `+0.015 m` upward offset is `-0.015 m`
in `tool0`. The simulation URDF's fixed joints have the matching values. This
documentation and backend work do not alter the existing TF signs, axes, tool
frame names, pre-mouth standoff, keepouts, PlanningScene, or OctoMap behavior.

## Before the first physical test

- Verify the mounted camera and straw transforms against the physical tool;
  do not assume Gazebo's fixed-link calibration is mechanically identical.
- Verify UR External Control, network routing, robot mode, protective stops,
  E-stop, clearance, and operator procedure.
- Run the read-only status probe and plan-only test first.
- Independently review the existing human keepout geometry and workspace
  limits for the physical room and person.
- Enable `UR10E_ALLOW_REAL_EXECUTION=1` only in a deliberate, supervised
  terminal after all of the above. No OpenClaw invocation should be used as
  the first physical motion test.
