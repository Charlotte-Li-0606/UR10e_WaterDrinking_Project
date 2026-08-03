# Codex project instructions

## UR10e drinking workflow

- For requests such as “I want water”, “help me drink”, “feed me water”, or
  “bring the straw to my mouth”, use the `feed-water-ur10e` Codex skill.
- Use `scripts/codex_feed_water.sh`; do not route Codex requests through
  OpenClaw, its gateway, its workspace skill, or `openclaw_*.sh`.
- Preserve the OpenClaw implementation as a legacy fallback. Do not delete or
  modify it unless the user explicitly asks to maintain or remove that version.
- The physical workflow supports only the high-level, center-target
  `feed_water` operation ending at an 80 mm pre-mouth hold. Its guarded active
  search may include a predefined `wrist_3` left/right sweep of at most 60
  degrees from the recorded initial flange orientation. Stop that sweep on a
  stable mouth detection; if no mouth is found, return to the recorded initial
  flange orientation. Plan and collision-check every sweep and return segment
  through MoveIt, enforce joint and workspace limits, and preserve all runtime
  gates. Never substitute arbitrary user-supplied joints, poses, trajectories,
  controller commands, gripper actions, direct mouth contact, cup tilt,
  pouring, or an automatic retreat outside this bounded search profile.

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
