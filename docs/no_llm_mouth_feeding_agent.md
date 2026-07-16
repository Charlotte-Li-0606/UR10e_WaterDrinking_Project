# No-LLM Mouth-to-Pre-Mouth Agent

`scripts/run_no_llm_feeding_agent.sh` consumes `/detected_mouth_pose` and
performs one conservative perception-to-motion decision. It has no language
model, gripper logic, or mouth-contact command.

The default invocation is dry-run only:

```bash
scripts/run_no_llm_feeding_agent.sh
```

It requires five recent, stable `PoseStamped` messages, validates that the
mouth and the offset pre-mouth straw-tip target are within the configured
`base_link` workspace, applies a conservative 1.30 m tool0 reach envelope,
and publishes:

- `/feeding_agent/selected_mouth_pose`
- `/feeding_agent/pre_mouth_target`
- `/feeding_agent/status`

The default offset is `(0, -0.08, 0)` m. It keeps the straw 8 cm in front of
the face in this scene. Tune it with `--pre-mouth-offset x,y,z` and tune the
workspace with `--workspace-min x,y,z --workspace-max x,y,z`.
The reach envelope is configurable with `--max-tool-radius-m`.

Only this explicit command permits one MoveIt motion:

```bash
scripts/run_no_llm_feeding_agent.sh --execute
```

With `--execute`, the node first asks MoveIt for a plan-only preflight. Only
after that succeeds does it call the existing
`UR10eRobotEnv.move_straw_tip_to_pre_mouth()` SDK method. It then exits. It
never calls `move_straw_tip_to_mouth()`.

The default planner uses MoveIt's collision-checked MoveGroup pipeline. It
time-parameterizes the plan and applies the native Ruckig response adapter for
jerk-limited smoothing before the trajectory controller executes it. The
Cartesian SDK path preserves MoveIt's supplied timing and derivatives when it
is used by an integrator.
