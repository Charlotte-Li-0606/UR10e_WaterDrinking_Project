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
  hold. The nominal camera-ray stand-off is 50 mm, adaptive candidates are
  50/70/90/120/150 mm, and real MoveIt velocity/acceleration scaling is capped
  at 0.30. Keep tool0 +Z
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
  authoritative and verify it against the versioned calibrated MoveIt FK
  reference in `base_link`. Preserve the operator-displayed PolyScope pose as
  audit data only: a live no-motion check found a 400.4 mm position offset, so
  its active feature must not be assumed to be the physical UR base frame.
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
    --execute --confirm-real-motion --hold-duration 5
  ```

- Use `scripts/codex_feed_water.sh --plan-only` only when the user explicitly
  asks to plan, inspect, or avoid motion.
- Quoted examples, configuration changes, hypothetical discussion, diagnosis,
  status inspection, implementation, and testing do not authorize motion.
- Do not weaken or pre-answer the pipeline’s runtime gates. A refusal is a
  completed safe outcome. Never fall back to another motion path.
- Summarize the structured result accurately, including whether planning or
  execution was attempted and whether a trajectory was sent.

## Guarded real mouth-tracking test

- A standalone user request to start or continue the real mouth-tracking test
  authorizes one bounded tracking session through
  `scripts/real_mouth_tracking_servo.py`. This is separate from the one-shot
  `feed_water` workflow and must not modify or replace that workflow.
- Invoke exactly one fresh tracking session with the existing real-motion
  environment and confirmation gates:

  ```bash
  UR10E_ALLOW_REAL_EXECUTION=1 PYTHONPATH=.:${PYTHONPATH} \
    /usr/bin/python3 scripts/real_mouth_tracking_servo.py \
    --execute --confirm-real-motion --max-duration 15
  ```

- The runner may only lock the current collision-free tool pose and follow the
  same selected mouth by relative translation. It must command zero angular
  velocity, preserve the locked tool orientation, keep target displacement at
  or below 60 mm, Cartesian speed at or below 20 mm/s, acceleration at or
  below 0.10 m/s², and the tool within the 1.30 m workspace radius.
- For the first physical test, require the pendant speed slider to be at or
  below 10%, MoveIt Servo initialized, the scaled trajectory controller
  active, normal or reduced safety mode, the robot running under External
  Control, fresh `base_link` mouth and `tool0` TF data, and the operator's stop
  control immediately accessible.
- Immediately publish bounded zero twists and disarm on target loss/staleness,
  target `ABORTED`, TF failure, controller-state loss, collision, singularity,
  joint limit, workspace/displacement violation, robot or safety-mode change,
  duration timeout, operator stop, or process shutdown. Never switch
  controllers, command direct mouth contact, alter the locked orientation, or
  fall back to an arbitrary trajectory.
- Quoted examples, planning, implementation, status inspection, and disarmed
  tests do not authorize tracking motion. Report whether the reference locked,
  whether any Servo command was published, the maximum displacement/speed,
  and the final stop reason.

## Tracked feed-water execution

- A standalone request for “feed me water with tracking” authorizes one
  integrated guarded execution through the canonical Codex entrypoint:

  ```bash
  UR10E_ALLOW_REAL_EXECUTION=1 scripts/codex_feed_water.sh \
    --execute --confirm-real-motion --hold-duration 5 \
    --track-mouth-during-execution
  ```

- The tracking flag is explicit opt-in; ordinary `feed_water` keeps the
  existing frozen-target one-shot behavior. During tracked execution, MoveIt
  exclusively owns the controller for the collision-checked approach. Stable
  significant mouth drift cancels that trajectory and triggers at most two
  fresh plans from the stopped state. Servo may publish only after MoveGroup
  finishes and releases the controller, and only for bounded relative
  translation during the pre-mouth hold.
- Never publish MoveIt execution and Servo commands concurrently. Preserve the
  validated pre-mouth offset, locked tool orientation, human collision scene,
  OctoMap, 60 mm relative target limit, 20 mm/s Servo limit, 0.10 m/s²
  acceleration limit, target-loss halt, and guarded return. Report all replan
  attempts, Servo command count, measured tool displacement/orientation error,
  and the final stop reason.
