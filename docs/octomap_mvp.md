# Wrist RGB-D OctoMap MVP

This project has an optional dynamic obstacle layer for MoveIt.  It keeps the
existing deterministic PlanningScene head, torso, and face-safety collision
objects in place; it does not replace them or add a 3D occupancy grid to
Gazebo.

It requires the standard Jazzy perception plugin package:

```bash
sudo apt install ros-jazzy-moveit-ros-perception
```

The package is installed on this machine and provides
`occupancy_map_monitor/PointCloudOctomapUpdater`.

## Start it

Start the normal feeding simulator with the experimental layer enabled:

```bash
cd /home/dase-hw101/ur_drinking_project
USE_OCTOMAP=true scripts/start_ur10e_feeding_sim.sh
```

That command starts `depth_to_pointcloud_node.py` and passes
`use_octomap:=true` to the project MoveIt launch.  The feature remains opt-in:
the plain `scripts/start_ur10e_feeding_sim.sh` command starts the existing
feeding setup without OctoMap.

The cloud converter can also be started independently:

```bash
scripts/run_depth_to_pointcloud.sh
```

It converts `/wrist_rgbd/depth_image` (`32FC1` metres or `16UC1` millimetres)
and `/wrist_rgbd/camera_info` to `/wrist_rgbd/points`.  Its defaults are
stride 4 and a valid-depth range of 0.15--2.50 m.  Run it with `--help` for
topic, frame, stride, and range overrides.

For the physical D435i, use the separate no-motion wrapper:

```bash
scripts/run_real_depth_to_pointcloud.sh
```

It reads the aligned real depth and color camera-info topics but publishes the
same `/wrist_rgbd/points` interface. Its conservative defaults are stride 4,
5 Hz, and 0.20--2.00 m. The 20 cm lower bound intentionally rejects the
camera/tool near field; a 10 mm trigger is not feasible with the D435i, the
3 cm occupancy voxels, or this point-cloud update rate. This wrapper starts no
MoveIt node and sends no robot or controller command.

## MoveIt configuration and verification

`config/sensors_3d.yaml` selects
`occupancy_map_monitor/PointCloudOctomapUpdater`, an OctoMap resolution of
0.03 m, a 2.5 m maximum range, and publishes the filtered result on
`/wrist_rgbd/filtered_cloud`.  The source cloud uses the wrist camera optical
frame; MoveIt transforms it through the existing wrist-camera TF chain into
its planning frame.  The deployed sensor configuration uses `base_link` as
the OctoMap frame setting.

The previous 4 cm voxels were reduced to 3 cm.  This increases the potential
voxel count by about 2.4x, while retaining the current stride-4, 5 Hz cloud
limits.  A 2 cm map would need roughly 8x the old voxel count and is not the
next safe default until depth-noise and self-filtering tests justify it.

After starting with `USE_OCTOMAP=true`, verify the connections without dumping
the cloud payload:

```bash
ros2 topic info /wrist_rgbd/points -v
ros2 topic hz /wrist_rgbd/filtered_cloud
rg -i 'pointcloud|octomap|occupancy' logs/moveit_with_kinematics.log
```

`move_group` must appear as a subscriber to `/wrist_rgbd/points`; it publishes
`/wrist_rgbd/filtered_cloud` after processing valid cloud updates.  RViz can
also display the OctoMap or filtered cloud if desired.

First use the normal safe dry-run, then execute only after it succeeds:

```bash
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py \
  --plan-pre-mouth --use-planning-scene

python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py \
  --move-pre-mouth --use-planning-scene --execute
```

The second command still applies and verifies the deterministic PlanningScene
objects before planning.  The LLM runner uses the same validated feeding
tools, so its motion steps use this scene as well.

With `OPENAI_API_KEY` and the normal model/base-URL environment variables set,
exercise the real-LLM path through its virtual-environment launcher:

```bash
scripts/run_llm_feeding_agent.sh --task "Feed water" --execute --print-plan
```

The output must include `"planner_source": "llm"` and
`"event": "llm_plan_received"` before any validated tool runs.

## Validate dynamic-obstacle effect

The purple `octomap_depth_test_obstacle` is a Gazebo model only.  It has a
visual and Gazebo collision geometry, but its physical contacts are disabled
so it cannot perturb the simulated robot and confound this perception test.
The helper never publishes a MoveIt CollisionObject.  Its only route to
MoveIt is therefore the wrist depth image, `/wrist_rgbd/points`, the
point-cloud updater, and the OctoMap world layer.

For an explicit manual comparison, start with the obstacle absent, plan, then
spawn it and plan the exact same motion without execution:

```bash
USE_OCTOMAP=true scripts/start_ur10e_feeding_sim.sh
scripts/spawn_octomap_test_obstacle.sh --remove

python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py \
  --plan-pre-mouth --use-planning-scene

scripts/spawn_octomap_test_obstacle.sh --spawn \
  --pose "0.35 0.82 1.65 0 0 0" \
  --size "0.12 0.12 0.25"
sleep 5

python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py \
  --plan-pre-mouth --use-planning-scene
```

The pose is in Gazebo world coordinates and is calibrated for the supplied
elevated-base scene.  It is near, but does not overlap, the pre-mouth target.
The older generic example pose `0.35 0.65 1.00` is not necessarily in this
camera view; use the calibrated pose above or tune it while watching Gazebo.

For a reproducible structured check, run:

```bash
python3 robot_layer/arm_ur10e/demos/octomap_planning_validation.py
```

It removes any old test model, gathers a no-obstacle baseline, uses one fixed
mouth pose for both plan-only requests, spawns the Gazebo model, waits for a
fresh cloud update, and reports JSON with raw/filtered cloud point counts in
the obstacle volume plus baseline and obstacle planning results.  The test
leaves the purple model in Gazebo for inspection; remove it with:

```bash
scripts/spawn_octomap_test_obstacle.sh --remove
```

On the current scene, the validation observed the obstacle in both clouds and
the previously successful pre-mouth request was safely rejected after the
OctoMap update.  This demonstrates an occupancy-planning effect, not a
safety-certified collision layer.

## Same-target dynamic replanning

`FeedingSkillLibrary.move_straw_tip_to_pre_mouth_with_dynamic_avoidance()`
remains the independent simulation function. The canonical physical path does
not call that simulation helper. It uses
`scripts/real_feed_water_integrated.py`, while
`scripts/real_dynamic_obstacle_avoidance_plan.py` remains the independent
real plan-only diagnostic. Neither path uses MoveIt Hybrid Planning or
replaces the current MoveIt launch structure. The guarded real workflow sends
a normal MoveGroup plan-and-execute request with:

- one frozen pre-mouth coordinate and final flange-down orientation;
- `ompl/RRTConnectkConfigDefault`, so a non-linear detour is possible;
- the existing deterministic human collision objects plus the OctoMap;
- MoveGroup path monitoring and exactly three replan attempts;
- zero replan delay, so it never enters a wait-until-clear mode.

When an occupancy update invalidates the remaining trajectory, MoveGroup
stops execution and immediately replans from the stopped state to the same
goal constraints. If no route exists, the action fails and remains stopped.
The target is not recomputed from newer mouth perception during that action.

Plan it without motion:

```bash
python3 robot_layer/arm_ur10e/demos/feeding_tools_smoke_test.py \
  --dynamic-avoidance --use-planning-scene
```

The command fails closed unless `/move_group` reports the OctoMap updater and
both a non-empty `/wrist_rgbd/points` input and MoveIt's non-empty
`/wrist_rgbd/filtered_cloud` output have arrived within 0.75 seconds. Requiring
the filtered stream prevents a live camera with a broken TF/updater path from
being mistaken for a working occupancy map.

Direct real execution of the generic helper remains hard-blocked in both the
high-level library and SDK. Real dynamic execution is reachable only inside
the single guarded `feed_water` operation. That state machine requires fresh
raw and filtered clouds, freezes the selected mouth and target coordinate,
uses exactly three zero-delay replan attempts, cancels if the cloud becomes
stale or selected identity becomes unsafe, and retains all normal real-motion
runtime gates.

For that no-motion promotion gate, use the dedicated MoveIt launcher after
stopping the ordinary real MoveIt instance. It sets MoveGroup's
`allow_trajectory_execution` parameter to false:

```bash
# Terminal 1: D435i driver and the calibrated real perception/TF process run
# as usual. Terminal 2 publishes only the bounded cloud:
scripts/run_real_depth_to_pointcloud.sh

# Terminal 3: plan-only MoveIt with the OctoMap updater:
scripts/start_ur10e_real_moveit_octomap_plan_only.sh

# Terminal 4: the independent dynamic profile; no --execute flag:
UR10E_BACKEND=real python3 \
  scripts/real_dynamic_obstacle_avoidance_plan.py
```

The result must report `execution_sent: false`, a fresh cloud, the verified
OctoMap parameters, the frozen target, and a successful OMPL plan. Do not use
this setup for real motion. Repeated plan-only trials with the cup/straw fully
installed must first show no self-occupied voxels blocking the start state.
The real plan keeps the final orientation at the proven 0.001 rad tolerance
and permits at most 0.05 rad (2.86 degrees) per orientation-error axis along a
detour; this is a bounded planner tolerance, not a camera-search rotation.

## Constrained pre-mouth standoff range

The original fixed 8 cm policy is recorded in
[the policy snapshot](snapshots/pre_mouth_fixed_8cm_before_range.md).  The
LLM feeding tools now use a constrained 5--10 cm standoff zone rather than
relaxing the approach into arbitrary poses:

- Candidates remain exactly on the fixed base-link `-Y` approach ray from the
  detected mouth.
- Candidates are 10, 9, 8, 7, 6, then 5 cm from the mouth.  The furthest
  candidate is tried first to maximize face clearance.
- Every candidate must pass workspace and tool-radius guards, deterministic
  head/torso/face PlanningScene checks, configured human keepouts, and a
  MoveIt plan-only collision preflight.
- Only the first successful candidate can execute.  The robot still stops at
  that pre-mouth position; direct mouth contact remains disabled.

This allows a depth-derived obstacle to block one standoff without making the
robot try an unsafe arbitrary pose.  If none of the six bounded candidates is
collision-free, execution fails safely.

## Limitations and disabling

This is deliberately an MVP.  The wrist camera can observe the robot,
flange-mounted cup holder, and straw.  MoveIt performs its normal robot
self-filtering, but no task-specific tool/cup masking or persistent-map
filtering has been added.  Depth noise or self-points can therefore make a
plan conservative or fail.  The converter only applies depth-range and stride
filtering.

The wrist camera also has occlusion and field-of-view blind spots. OctoMap is
therefore a planning input, not a protective safety sensor. The UR safety
system, conservative speed, and operator stop remain necessary. Reliable
real deployment will require a calibrated cup/straw/camera exclusion model or
mask and should ideally add a fixed external depth view for coverage behind
the wrist camera.

To return to the proven deterministic collision-object setup, restart without
the flag:

```bash
USE_OCTOMAP=false scripts/start_ur10e_feeding_sim.sh
```

No controller, gripper, Gazebo human model, or point-cloud occupancy
implementation outside MoveIt's experimental OctoMap monitor is changed. The
SDK addition is limited to constructing the guarded OMPL MoveGroup request and
reporting feedback; it does not publish raw controller trajectories.
