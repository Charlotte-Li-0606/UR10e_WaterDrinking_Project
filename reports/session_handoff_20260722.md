# Session handoff — 2026-07-22

## Preserved code versions

- Active probe is the reconstructed morning-success version (v1):
  `scripts/real_premouth_from_perception_plan.py`
- V1 SHA-256:
  `10720a7784120681039172f5d3bdc9f99c07e0f9a9ea512fc08a9902e1f93136`
- V1 archive:
  `scripts/versions/real_premouth_from_perception_plan_v1_morning_success.py`
- Version 2 is preserved in Git stash:
  `stash@{0}: version2-current-vector-and-guarded-probe-20260722`
- Do not pop or drop the stash until the user chooses which version to retain.

## Live stack left running

The user asked to leave the current work in place for tomorrow. The following
fresh stack was left running and no motion command was sent:

- UR real driver (`ros2_control_node` PID 877656)
- MoveIt (`move_group` PID 878132)
- RealSense D435i (`realsense2_camera_node` PID 878300)
- Mouth perception (`mouth_perception_node.py` PID 878501)
- `rqt_image_view` on `/mouth_detection/debug_image` (PID 879140)

External Control was connected and `scaled_joint_trajectory_controller` was
active at the last check. The robot speed slider reported 15%.

## Incremental no-motion result after clean restart

Each layer stayed connected for at least 60 seconds:

1. Driver only: pass
2. MoveIt only: pass
3. RealSense only: pass
4. Mouth perception only: pass
5. Check only: pass for controller connectivity, but validation failed because
   no `/detected_mouth_pose` message was received in five seconds

The check-only command caused one brief controller-manager overrun at
`2026-07-22 16:58:08.749 +08:00` (2.230803 ms at a 2 ms/500 Hz target), but it
did not drop the reverse interface or deactivate the trajectory controller.

Check report:
`reports/premouth_camera_ray_v1_after_clean_restart_check_20260722.json`

Earlier diagnosis:
`reports/external_control_trigger_analysis_20260722.md`

## Tomorrow's next safe step

1. Confirm the live processes and pendant state; do not assume an overnight
   connection remained healthy.
2. Put the mouth target clearly in the camera/debug image.
3. Repeat v1 `--mode check` with `UR10E_ALLOW_REAL_EXECUTION=0`.
4. If the mouth pose validates and External Control remains connected for 60
   seconds, run `--mode plan` only. Do not execute robot motion without fresh,
   explicit user approval.

## Morning status — 2026-07-23

The processes remained alive overnight, but the robot connection did not. The
driver recorded this sequence:

- `2026-07-22 17:04:21.013 +08:00`: reverse-interface connection dropped and
  the scaled trajectory controller was deactivated
- `2026-07-22 17:04:27.598 +08:00`: robot mode changed to `POWER_OFF`
- `2026-07-22 17:04:29.336 +08:00`: safety mode changed to `FAULT`
- Afterwards, the existing driver continuously reported that RTDE background
  reading was not running

The old process tree is therefore alive but unusable. No new check or plan was
sent on the morning of July 23. A clean stack restart plus pendant fault recovery
and a fresh External Control Play are required before testing can continue.
