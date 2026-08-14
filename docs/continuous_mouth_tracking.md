# Continuous mouth tracking mode

The continuous mode is an explicit alternative to the existing one-shot and
segmented feed-water paths. Both older paths remain the default and retain
their existing behavior. No LLM or remote-network call is made in the tracking
loop.

## Runtime flow

1. Verify the live UR state and configured `initial_position`.
2. Collect identity-locked MediaPipe RGB-D mouth observations for up to 3 s.
3. Use a robust median for three or more samples, or a provisional estimate
   from the available valid samples. No valid sample means no invented target
   and no outbound command; the existing active-search path may run.
4. Preserve the fixed human and attached tool collision objects. The dynamic
   OctoMap layer is disabled by default.
5. Plan one conservative 250 mm coarse staging pose on the calibrated camera
   ray before Servo takes ownership.
   Recovery priority is Cartesian, Pilz, then OMPL.
6. Use MoveIt Servo Twist commands continuously from staging through the 80 mm
   base_link -Y pre-mouth hold. Commands are bounded and filtered; raw MediaPipe
   points are never sent directly to Servo.
7. Halt Servo before any recovery planner acquires motion ownership. Resume
   only after the prior planned execution completes and the target is fresh.
8. Perform the existing guarded return to the immutable initial position after
   success, and attempt it after every stop that follows a sent motion. Failure
   recovery preserves and collision-checks the current human scene.

Tool0 +Z remains aligned with `base_link` -Z. The controller applies only the
angular correction needed for this axis alignment, so spin/yaw about the
vertical tool axis remains free. It never commands `wrist_3_joint` directly.

## Configuration

Parameters are in `config/continuous_mouth_tracking.yaml`. Important defaults:

- no-target acquisition timeout: 3.0 s; a fresh valid provisional target starts
  tracking immediately;
- stale-target zero-velocity hold begins at 0.30 s. Confirmed visual face loss
  aborts after 1.00 s; a visible face with temporarily invalid depth remains
  stationary for up to 2.00 s while rebuilding a stable multi-frame 3D target;
- finite close-range mouth depth is accepted from 50 mm instead of applying
  the former 150 mm policy cutoff. Zero, negative, NaN, empty, or otherwise
  unavailable depth remains invalid and never produces RGB-only motion;
- final/provisional base_link -Y standoff: 50/250 mm;
- Servo recovery hysteresis: enter at 100 mm, exit at 60 mm;
- during the initial approach, the coarse-staging convergence distance is not
  treated as mouth drift; bounded Servo commands close that gap while the
  workspace, collision, tilt, speed, acceleration, and duration guards remain
  active;
- recovery starts a new physical mouth-displacement reference after successful
  execution and waits for a new live target before Servo resumes;
- maximum linear speed: 20 mm/s (10 mm/s while provisional);
- maximum linear acceleration: 0.10 m/s²;
- MoveIt Servo's collision-proximity scaling is not multiplied by a second
  application-level 0.25 factor; collision deceleration and hard halts remain
  active, and the 20 mm/s command cap is unchanged;
- maximum angular correction speed: 0.15 rad/s;
- target stale/face-lost/depth-reacquisition timeout: 0.30/1.00/2.00 s;
- maximum tracking duration: 45 s;
- dynamic OctoMap: disabled by default.

These are engineering starting values, not safety-certified limits.

## ROS interfaces

The implementation uses the installed Jazzy MoveIt Servo configuration:

- input: `/servo_node/delta_twist_cmds` (`geometry_msgs/TwistStamped`);
- status: `/servo_node/status` (`moveit_msgs/ServoStatus`);
- command selection: `/servo_node/switch_command_type`;
- pause/unpause: `/servo_node/pause_servo`;
- diagnostic target: `/continuous_mouth_tracking/target_pose`;
- diagnostic state: `/continuous_mouth_tracking/status`.

Servo outputs `trajectory_msgs/JointTrajectory` to the existing
`/scaled_joint_trajectory_controller/joint_trajectory` interface. The project
does not publish direct UR controller commands and does not switch controllers.

## No-motion tests

Run the deterministic tracking simulation and project regression suite:

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=.:$PYTHONPATH pytest -q \
  robot_layer/arm_ur10e/control/test_continuous_servo_tracking.py \
  robot_layer/arm_ur10e/perception/test_continuous_mouth_tracker.py
PYTHONPATH=.:$PYTHONPATH pytest -q robot_layer/arm_ur10e
```

The first command covers static/moving targets, provisional acquisition,
over-range fallback, target loss, Servo halts, vertical-axis correction,
exclusive command ownership, and OctoMap degraded states. Neither command sends
ROS motion.

Continuous recovery remains local even when OMPL is the final fallback. OMPL
receives path constraints around the current joint branch, and every resulting
trajectory is rejected before execution unless waypoint FK proves that tool0
stays inside an envelope derived from the requested correction. Per-joint
excursion, cumulative joint travel, duration, collision, and vertical-axis
checks all remain mandatory. A collision-free route that reaches a remote IK
branch is therefore not considered executable.

## Explicit mode selection

A plan-only invocation is:

```bash
UR10E_BACKEND=real scripts/run_feed_water_real_direct.py \
  --plan-only --continuous-mouth-tracking --hold-duration 5
```

`--use-octomap` is optional. When omitted, deterministic fixed collision
objects remain active and no stationary OctoMap rebuild is required. When it is
present but rebuilding fails, diagnostics report
`dynamic_obstacle_layer_unavailable` and outbound motion is withheld; the
system never claims dynamic obstacle coverage. A running MoveGroup that still
contains occupancy while the option is disabled is also refused as a launch
configuration mismatch rather than being mislabeled as deterministic-only.

Real execution remains protected by `UR10E_ALLOW_REAL_EXECUTION=1`,
`--execute`, `--confirm-real-motion`, the center-target policy, the initial
position check, controller/External Control/robot/safety checks, collision and
workspace validation, and the existing return gates. Implementation and tests
do not start a real execution automatically.
