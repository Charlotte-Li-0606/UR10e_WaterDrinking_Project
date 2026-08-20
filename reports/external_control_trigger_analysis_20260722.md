# UR External Control no-motion trigger analysis — 2026-07-22

No robot-motion goal, controller switch, dashboard command, process kill, or
process restart was issued during this investigation.

## Result

The first observed trigger was creation of a new read-only ROS 2 Python
participant, using this command:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS ROS_LOCALHOST_ONLY
python3 scripts/monitor_external_control_state.py \
  --label baseline_existing_stack_no_command \
  --duration 60 \
  --interval 5 \
  --report-file reports/external_control_matrix_01_baseline_20260722.json
```

The monitor creates subscriptions and a `ListControllers` client only.  It has
no publishers, action clients, dashboard clients, or controller-switch client.

Timeline (HKT):

| Time | Event |
| --- | --- |
| 16:09:41.326948 | UR robot connected to the reverse interface. |
| 16:09:41.329483 | `scaled_joint_trajectory_controller` activated. |
| 16:13:56.562000 | The new Python ROS 2 participant log was created. |
| 16:13:56.693841 | The 500 Hz controller loop took 13.184789 ms and missed 7 cycles. |
| 16:13:56.739905 | `Connection to reverse interface dropped.` |
| 16:13:57.171963 | Controller manager began deactivating `scaled_joint_trajectory_controller`. |

The reverse interface had stayed connected for about 255 seconds with the
existing MoveIt, RealSense, and perception processes present.  The requested
60-second window was stopped at the first trigger instead of continuing to
check-only or plan-only.

## Matrix

| Stage | Result | Evidence |
| --- | --- | --- |
| Driver process | Present, no duplicate | One `ur_control.launch.py` process, PID 853777. |
| Controller manager | Present, no duplicate | One `ros2_control_node`, PID 853788. |
| MoveIt presence | Stable before trigger | One `move_group`, PID 826523; already running throughout the 255-second pre-trigger interval. |
| RealSense presence | Stable before trigger | One `realsense2_camera_node`, PID 851710; already running throughout the pre-trigger interval. |
| Perception presence | Stable before trigger | One `mouth_perception_node.py`, PID 858414; already running throughout the pre-trigger interval. |
| Read-only monitor participant | **First trigger candidate** | Reverse interface dropped 0.178 seconds after its ROS log was created. |
| Check-only | Not run | Matrix stopped at first trigger. |
| Plan-only | Not run | Matrix stopped at first trigger; no MoveGroup goal was sent. |

A truly clean driver-only → MoveIt → RealSense → perception incremental launch
was not possible because all four layers were already running.  They were not
killed or restarted because that requires operator approval.

The monitor had not yet received the transient robot-mode/program state or a
joint-state sample when the drop occurred, so those fields are `null` in its
JSON.  The driver log is the authoritative evidence for reverse-interface and
controller state in this sub-second interval.  System CPU at the first sample
was approximately 10.13%, load average was 3.72/3.05/2.88 on 16 CPUs, and
process counts remained exactly one driver and one controller manager.

## Historical correlation

This was not an isolated coincidence.  Existing logs contain four other
reverse-interface drops 0.16–0.19 seconds after a new Python ROS 2 log was
created:

| Python ROS log created | Reverse interface dropped | Delta |
| --- | --- | ---: |
| 15:22:18.152 | 15:22:18.341817 | 0.190 s |
| 15:31:29.624 | 15:31:29.786605 | 0.163 s |
| 15:39:10.072 | 15:39:10.250377 | 0.178 s |
| 15:59:31.668 | 15:59:31.854415 | 0.186 s |
| 16:13:56.562 | 16:13:56.739905 | 0.178 s |

## Stop/pause/kill audit

No real-robot project code calls dashboard `stop`, `pause`, `power_off`,
`shutdown`, or `quit`.  The installed dashboard client process died at
15:40:03.646 with exit code `-13` (`SIGPIPE`); the later reverse-interface
reconnect/drop cycles therefore were not dashboard stop service calls.

`scripts/start_ur10e_feeding_sim.sh` contains broad `pkill` commands for
simulator cleanup, including `move_group`, `robot_state_publisher`, and
perception names.  That simulator script was not running during this trial and
does not target the live `ros2_control_node` process, but it should not be run
beside the real stack.  The real perception launcher's `kill` is confined to
its own static-TF child during cleanup.

## Likely cause

The immediate cause recorded by the driver is loss of the UR reverse
interface, not a dashboard stop and not controller deactivation.  Controller
deactivation is a consequence of that loss.

The strongest current explanation is a scheduling/discovery latency spike
when a new ROS 2 participant joins.  The controller manager is configured for
500 Hz (2 ms period), but its startup log says it could not enable FIFO
real-time scheduling.  In the reproduced event, participant creation was
followed by a 13.18 ms controller-loop overrun and then the reverse-interface
drop.  Overall CPU was not saturated and duplicate driver/controller processes
did not appear.  This is a well-supported inference from timing, not yet proof
of which DDS callback or scheduler interaction caused the stall.
