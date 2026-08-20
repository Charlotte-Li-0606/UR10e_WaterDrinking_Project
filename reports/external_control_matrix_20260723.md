# UR External Control no-motion matrix — 2026-07-23

## Outcome

No tested command stopped UR External Control after the clean ROS restart. The
reverse interface stayed connected through driver-only, MoveIt-only,
RealSense-only, perception-only, rqt image viewing, check-only, and plan-only.
No robot trajectory was executed.

The active test program was the reconstructed morning-success v1, SHA-256:

`10720a7784120681039172f5d3bdc9f99c07e0f9a9ea512fc08a9902e1f93136`

Version 2 remains preserved in:

`stash@{0}: version2-current-vector-and-guarded-probe-20260722`

## Incremental results

| Layer | Start (+08:00) | Minimum observation | Result |
| --- | --- | ---: | --- |
| Fresh UR driver | 09:31:12 | >90 s | Pass; connected at 09:31:17, robot RUNNING/NORMAL |
| MoveIt | 09:32:57 | 80 s | Pass; no planning request during this layer |
| RealSense D435i | 09:34:25 | 73 s | Pass |
| Mouth perception | 09:35:44 | >60 s | Pass; initially locked onto scene before target placement |
| rqt image view | 09:37:11 | >8 min | Pass; `/mouth_detection/debug_image` |
| Perception reset on steady target | 09:38:30 | 97 s before check | Pass |
| Check-only | 09:41:59 | >79 s | Pass; stable mouth pose and controller checks |
| Plan-only | 09:43:34 | >100 s | Pass; plan succeeded and no execution was sent |

At the final observation (09:45:14), exactly one `ros2_control_node`, one
`move_group`, one RealSense node, one perception node, and one rqt image viewer
were running. System load was 2.31 / 2.18 / 1.94. Approximate process CPU was
32.4% controller manager, 19.6% RealSense, 13.0% MoveIt, 74.6% perception, and
5.1% rqt image view.

## Exact check-only command

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS ROS_LOCALHOST_ONLY
export UR10E_ALLOW_REAL_EXECUTION=0
python3 -u scripts/real_premouth_from_perception_plan.py \
  --mode check \
  --premouth-policy camera-ray \
  --safe-distance 0.05 \
  --sample-seconds 5 \
  --report-file reports/premouth_camera_ray_v1_target_ready_check_20260723.json
```

Result: success, 8 mouth samples, maximum distance from mean 0.003997 m,
surface-normal maximum angular spread 2.505 degrees, fresh complete joint
state, active joint-state broadcaster and scaled trajectory controller,
`execution_disabled: true`, and `execution_sent: false`.

Report SHA-256:
`abb5416fe1fb9f1833b0fa0047b97353f9229276dc11272344d07293cfea48e0`

## Exact plan-only command

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS ROS_LOCALHOST_ONLY
export UR10E_ALLOW_REAL_EXECUTION=0
python3 -u scripts/real_premouth_from_perception_plan.py \
  --mode plan \
  --premouth-policy camera-ray \
  --safe-distance 0.05 \
  --sample-seconds 5 \
  --report-file reports/premouth_camera_ray_v1_target_ready_plan_only_20260723.json
```

Result: MoveIt/Pilz LIN planning success, 54 trajectory points, planned duration
5.2695 s, planned tool translation 0.2413 m, planning time 0.00847 s,
`execution_disabled: true`, and `execution_sent: false`. MoveIt logged only
that the motion plan was computed successfully; it did not log trajectory
execution.

Report SHA-256:
`85f089f07c96724cd6ef014508e4ed3d348b64cb5985117d0ec16e0e4090a4d3`

## Controller timing observations

The 500 Hz controller manager still cannot obtain FIFO real-time scheduling.
Brief overruns occurred when some ROS participants started:

- Perception reset: 2.876 ms and a 3.246 ms overrun warning
- First check attempt with no detected face: 3.197 ms
- Successful check: 2.471 ms and 2.864 ms
- Successful plan-only: no new driver warning observed

None caused a reverse-interface disconnect today. The larger 26.264 ms event
was during initial controller activation and also did not disconnect.

## Interpretation

Today's matrix does not identify a deterministic command-layer trigger. It also
rules out the feeding-vector code as the cause because v1 (which predates that
code) and the same ROS layers stayed connected and planned successfully.

Yesterday's overnight failure remains a separate event: the reverse interface
dropped at 17:04:21, followed by robot POWER_OFF and safety FAULT. Because that
event occurred more than three minutes after rqt was added and did not reproduce
today after more than eight minutes with rqt, it cannot be attributed to the
rqt command from the available evidence. The non-real-time 500 Hz scheduling
remains a reliability risk, but it was not sufficient by itself to reproduce
the stop in this run.

