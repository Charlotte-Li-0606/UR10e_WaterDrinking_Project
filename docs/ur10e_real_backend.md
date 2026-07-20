# UR10e real-backend setup

The project has one ROS 2 / MoveIt SDK: `UR10eRobotEnv`. Its safe feeding
adapter is `FeedingSkillLibrary`. `UR10E_BACKEND` selects ROS endpoints for
that same implementation; it does not select a second SDK or use Piper code.

## Backend policy

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `UR10E_BACKEND` | `sim` | `sim` uses Gazebo's `joint_trajectory_controller`; `real` uses the UR driver's `scaled_joint_trajectory_controller`. |
| `UR10E_ROBOT_IP` | unset | Required by the manual real-driver start command and any supervised real execution process. It is never stored in project files. |
| `UR10E_ALLOW_REAL_EXECUTION` | `0` | A real execution request is rejected unless it is exactly enabled (`1`, `true`, `yes`, or `on`). |

Selecting `UR10E_BACKEND=real` does not enable motion. The shared SDK blocks
every executing MoveIt or trajectory action unless the execution variable is
enabled, `/joint_states` is complete, `base_link -> tool0` is available, the
expected scaled controller is active, and its `FollowJointTrajectory` action
server is available. Real planning-only requests do not require the execution
variable. The real backend caps MoveIt velocity and acceleration scaling at
`0.05`.

OpenClaw and the LLM runner keep using `FeedingSkillLibrary` and the existing
validated tools. `feed_water(execute=False)` remains backward compatible. The
OpenClaw compatibility command remains simulator-oriented; it must not be used
to start a real robot.

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
ros2 launch ur_robot_driver ur10e.launch.py \
  robot_ip:=<robot-ip> \
  use_mock_hardware:=false \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  activate_joint_controller:=true
```

Terminal 2, MoveIt only after the driver publishes robot description and joint
state:

```bash
/home/dase-hw101/ur_drinking_project/scripts/start_ur10e_real_moveit.sh
```

This uses the locally installed interface:

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur10e launch_rviz:=false launch_servo:=false use_sim_time:=false
```

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
