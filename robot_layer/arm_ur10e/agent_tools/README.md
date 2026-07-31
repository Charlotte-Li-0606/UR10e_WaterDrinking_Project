# UR10e feeding tools

`feeding_tools.py` is a conservative, structured wrapper for a future agent.
It does not connect an LLM and does not expose joint, controller, arbitrary
pose, or gripper commands.

`FeedingSkillLibrary` provides:

- The agent-approved reusable surface: `get_observation()`,
  `detect_target(target_type="mouth", detector="mediapipe")`,
  `active_search(target_type="mouth", detector="mediapipe",
  max_time_sec=15.0, strategy="safe_scan")`,
  `select_target(target_type="mouth", strategy="center")`,
  `move_tool_to_target(tool="straw_tip", target="pre_mouth", execute=False)`,
  `check_progress(task="feed_water", critic="rule_based")`,
  `hold(duration_sec=3.0)`, `retreat(target="ready", execute=False)`, and
  the backwards-compatible `feed_water()` wrapper. The dispatcher and LLM
  validator expose only this list.
- `feed_water()` composes the reusable tools for the working water-assistance
  sequence. It preserves the existing MediaPipe/active-search, fixed
  flange-down pre-mouth, PlanningScene, optional OctoMap, MoveIt, progress,
  and no-closer hold behavior.

- `select_active_target(left|center|right)` and `get_active_target_state()` —
  lock one logical target and report its queue freshness/stability. The current
  single-pose perception publisher is treated as `center`; the manager is
  ready to assign future multiple candidates by image x coordinate.
- `get_latest_mouth_pose()` and `wait_for_stable_mouth_pose()` — consume the
  live `/detected_mouth_pose` topic through the active target's bounded pose
  queue, reject missing/stale/jumpy/unstable input, and return a pose in
  `base_link`.
- `search_for_mouth(max_time_sec=15, selection="center", execute=False)` —
  returns a stable target immediately when available, otherwise preflights a
  translation-only search. The fixed policy first widens camera standoff in
  three 3 cm segments, then traces a ±2 cm lateral/vertical ring. All targets
  are absolute from the captured origin, every segment is at most 3 cm, flange
  rotation is disabled, and the scan stops at the first fresh mouth candidate
  while three stability samples accumulate. The effective budget is capped at
  15 seconds; dry-run mode never sends a trajectory.
- `compute_pre_mouth_target()` — applies the configured 8 cm safety offset,
  workspace checks, and conservative tool reach check.
- `move_straw_tip_to_pre_mouth(execute=False)` — applies MoveIt human
  keepouts, preflights the existing SDK primitive, and only executes when
  `execute=True`.
- `retreat_to_ready(execute=False)` — the equivalent predefined ready retreat.
- `adjust_cup_vertical(delta_z, execute=False)` — keeps the rigid control
  point's base-frame X/Y and current flange-down orientation fixed, then plans
  only a Cartesian Z translation. The current config has no cup-center offset,
  so this explicitly uses `straw_tip` as the control point. Each call is
  limited to ±3 cm and a control-point Z range of 0.40–1.80 m.
- `get_robot_observation()` — compact robot, straw-tip, camera calibration,
  and mouth-pose metadata; it does not return or expose raw control commands.
- `stop_motion_or_hold_position()` — reports whether the synchronous tool call
  is stationary. It intentionally does not publish direct controller stops.

`move_straw_tip_to_mouth_optional()` exists for a later, explicitly reviewed
integration only. It is disabled by default and the smoke-test CLI never turns
it on. It is not part of the agent-approved tool surface. The first-version
safe tool surface is pre-mouth only.

There is no gripper in this project.

## Smoke tests

From the project root, with the simulator running:

```bash
source /opt/ros/jazzy/setup.bash
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --observe
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --active-target-state
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --select-target center
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --search-mouth --search-timeout 15
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --plan-pre-mouth
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --move-pre-mouth
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --adjust-cup-vertical 0.01
```

The final command is dry-run. To send exactly the existing pre-mouth MoveIt
motion after its preflight succeeds:

```bash
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --move-pre-mouth --execute
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --adjust-cup-vertical 0.01 --execute
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --search-mouth --search-timeout 15 --execute
```

## Deterministic PlanningScene obstacles

Before a pre-mouth move, vertical adjustment, or ready retreat, the library
now refreshes deterministic MoveIt world objects from the selected
`/detected_mouth_pose`.  This is primitive collision geometry only; it does
not use an OctoMap or point-cloud occupancy map.

```bash
python3 robot_layer/arm_ur10e/agent_tools/planning_scene_manager.py --apply
python3 robot_layer/arm_ur10e/agent_tools/planning_scene_manager.py --remove
scripts/update_planning_scene_obstacles.sh
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py --plan-pre-mouth --use-planning-scene
```

`--use-planning-scene` and `--no-planning-scene` override the smoke-test
instance setting.  The default library setting is enabled.  Execution fails
safely if the current mouth-derived objects cannot be applied or verified;
plan-only mode reports a warning and may continue without the dynamic update.
