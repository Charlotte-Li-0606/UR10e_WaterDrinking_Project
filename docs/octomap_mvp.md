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

To return to the proven deterministic collision-object setup, restart without
the flag:

```bash
USE_OCTOMAP=false scripts/start_ur10e_feeding_sim.sh
```

No controller, SDK, gripper, Gazebo human model, or point-cloud occupancy
implementation outside MoveIt's experimental OctoMap monitor is changed.
