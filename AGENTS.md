# Codex project instructions

## UR10e drinking workflow

- For requests such as “I want water”, “help me drink”, “feed me water”, or
  “bring the straw to my mouth”, use the `feed-water-ur10e` Codex skill.
- Use `scripts/codex_feed_water.sh`; do not route Codex requests through
  OpenClaw, its gateway, its workspace skill, or `openclaw_*.sh`.
- Preserve the OpenClaw implementation as a legacy fallback. Do not delete or
  modify it unless the user explicitly asks to maintain or remove that version.
- The physical workflow supports only the high-level, center-target
  `feed_water` operation. Its outbound segment ends at a validated pre-mouth
  hold. Keep tool0 +Z
  aligned with base_link -Z within 5 degrees throughout active-search and
  obstacle-detour trajectories, while allowing spin around tool0 Z. Validate
  every trajectory waypoint with MoveIt FK before execution and refuse if the
  constraint cannot be met; never relax it silently. Active-search
  backward/up/down steps remain bounded Cartesian goals; left/right remain
  bounded MoveIt pose goals about tool0 local Z. MoveIt may select
  wrist_3_joint motion, but never command wrist_3_joint directly. Require the
  final pre-mouth goal to satisfy its validated full target orientation. Plan
  and collision-check every segment through MoveIt, enforce joint and workspace
  limits, and preserve all unrelated runtime gates. Never substitute arbitrary
  user-supplied joints, poses, trajectories, controller commands, gripper
  actions, direct mouth contact, cup tilt, or pouring.

## Guarded return to initial position

- The high-level real `feed_water` workflow may include one automatic return
  after a successful pre-mouth hold. The return is part of that same guarded
  operation; it is not exposed as a standalone real-motion tool and does not
  broaden authorization to arbitrary joint or pose commands.
- The only approved return target is the versioned `initial_position` defined
  by project configuration. Reject runtime overrides of its joints, pose,
  frame, orientation, or name. Treat the configured joint target as
  authoritative and use its recorded tool pose only as an FK verification
  reference in the verified MoveIt planning frame.
- Preserve tool0 +Z alignment with base_link -Z within 5 degrees throughout
  the return, while allowing free spin around tool0 Z. Validate every sampled
  return waypoint with MoveIt FK before execution and never command
  `wrist_3_joint` directly.
- Collision checking is mandatory for the complete return. A Cartesian return
  may be used only when MoveIt reports a complete, collision-free path. If it
  is incomplete or invalid, use a collision-checked MoveIt detour to the same
  fixed target. Never send a raw joint/controller command, ignore a collision,
  or silently relax the flange-axis constraint.
- Recheck the controller, External Control program, robot/safety mode, speed
  slider, start state, target state, attached tool geometry, fixed human
  objects, and current OctoMap immediately before sending the return
  trajectory. If any gate, target verification, plan, or execution check
  fails, remain at pre-mouth and report the exact refusal; do not attempt an
  alternate unguarded retreat.
- Plan-only mode may validate the return but must never send it. Real return
  execution retains `UR10E_ALLOW_REAL_EXECUTION=1` and
  `--confirm-real-motion`, and the structured report must distinguish outbound
  and return planning/execution results.

## Real-motion authorization

- A standalone user-directed request to drink or be fed water authorizes one
  guarded execution. Invoke exactly one fresh request:

  ```bash
  UR10E_ALLOW_REAL_EXECUTION=1 scripts/codex_feed_water.sh \
    --execute --confirm-real-motion --hold-duration 3
  ```

- Use `scripts/codex_feed_water.sh --plan-only` only when the user explicitly
  asks to plan, inspect, or avoid motion.
- Quoted examples, configuration changes, hypothetical discussion, diagnosis,
  status inspection, implementation, and testing do not authorize motion.
- Do not weaken or pre-answer the pipeline’s runtime gates. A refusal is a
  completed safe outcome. Never fall back to another motion path.
- Summarize the structured result accurately, including whether planning or
  execution was attempted and whether a trajectory was sent.
