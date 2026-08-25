# PGI-140-80 gripper simulation workflow

This is the retained project plan for the exact configured gripper
`PGI-140-80-W-S-M1-L5-J1-F1-01`. Only Steps 1-3 are in the current scope.
Nothing in this phase authorizes physical robot motion, Gazebo control, or
RS485 communication.

## Status

| Step | Work item | Status |
| --- | --- | --- |
| 1 | Preserve and version the successful feeding baseline | Current scope — complete locally; GitHub publication awaits SSH authentication |
| 2 | Obtain and verify exact official PGI-140-80, J1 fingertip, and F1 flange CAD | Current scope — official sources identified; exact download/checksum and redistribution permission unresolved |
| 3 | Build a parameterized URDF/Xacro model and verify it in RViz | Current scope — provisional kinematic model implemented; exact mesh replacement remains gated by Step 2 |
| 4 | Add `gz_ros2_control` and a standard `GripperCommand` simulation interface | Pending |
| 5 | Add the gripper, camera adapter, fingers, and grasp center to MoveIt | Pending |
| 6 | Validate logical cup grasping using attach/detach before contact physics | Pending |
| 7 | Add physical contact, friction, cup mass, grasp/release, and drop testing | Pending |
| 8 | Integrate the simulated gripper with reusable high-level tools | Pending |
| 9 | Replace the simulation backend with the real RS485 backend | Pending |
| 10 | Recalculate payload, center of gravity, TCP, camera transform, and collision geometry before real execution | Pending |

## Intended transform tree

```text
tool0
  -> pgi_camera_interposer
      -> pgi_mount
          -> pgi_body
              -> pgi_left_finger
              -> pgi_right_finger
              -> pgi_grasp_center
      -> d435i_mount
          -> d435i_link
              -> d435i_color_optical_frame
              -> d435i_depth_optical_frame
```

The camera remains rigidly attached to the interposer, never to a finger.
`pgi_grasp_center` is a logical frame. A cup and its straw tip are future
attached-object frames and are deliberately absent from the gripper body.

## Step 1 — preserve the baseline

1. Audit Git status, current branch, remotes, recent commits, untracked and
   ignored files, file sizes, and tracked/staged content for secrets.
2. Preserve user files and generated evidence on disk; do not reset, clean,
   delete, or overwrite them.
3. Ignore credentials, build/install/log products, environments, ROS bags,
   generated execution reports, recordings, vendor downloads, and CAD
   conversion intermediates.
4. Commit the behavior-preserving baseline and create the annotated tag
   `pre-pgi-gripper-baseline-2026-08-25`.
5. Publish the baseline branch and tag without force-pushing.
6. Create `feature/pgi-140-80-simulation`; keep all gripper work there and do
   not merge it into `main` yet.

## Step 2 — exact official CAD

Use only DH Robotics assets for the exact PGI body and matching J1/F1 parts.
The PGIA family must not silently replace PGI: first verify its dimensions,
mounting faces, cable exit, finger carriage, and F1 interface against the PGI
drawing. Do not use third-party CAD.

For every retrieved asset, record:

- official source URL and retrieval date;
- exact model variant and archive filename;
- SHA-256 checksum;
- CAD units and model scale;
- assembly origin and X/Y/Z axis convention;
- opening zero and travel convention;
- J1 and F1 part identity;
- applicable license or written redistribution permission.

Keep raw vendor files in the gitignored directory
`assets/vendor/dh_robotics/pgi_140_80/`. Commit only metadata, checksums,
download/conversion instructions, and legally redistributable project-created
meshes. If redistribution permission remains unclear, never commit the vendor
archive or derivatives that reproduce protected detailed geometry.

The current evidence and unresolved download details are recorded in
[`vendor_assets/pgi_140_80.md`](vendor_assets/pgi_140_80.md).

## Step 3 — parameterized description and RViz

### Architecture

Extend the existing `urdf/ur_gz_feeding_markers.urdf.xacro`; do not establish a
second UR description. The reusable gripper macro is
`urdf/pgi_140_80_macro.xacro`. The canonical option defaults to:

```text
use_pgi_gripper:=false
```

With the option false, the established simulation cup/camera description is
retained. With it true, that incompatible fixed tool is omitted and the PGI
assembly is added. The real D435i calibration file is never read or modified by
this option.

### Kinematics

- `pgi_left_finger_joint`: prismatic, axis `+X`, range `0.000-0.040 m`.
- `pgi_right_finger_joint`: prismatic, axis `-X`, range `0.000-0.040 m`.
- Right mimics left with multiplier `+1.0` and offset `0.0`.
- Zero means closed; positive motion opens both jaws.
- Maximum gap change is `2 x 0.040 = 0.080 m`, matching the documented total
  stroke. This interpretation must be rechecked against the exact J1 CAD.

The body uses the documented 1 kg mass. Its inertia is explicitly a
provisional uniform-box estimate from the documented body envelope. Finger,
interposer, mount, and camera masses are not claimed. Primitive collision
geometry is intentionally conservative and simple.

Detailed vendor visual meshes are not present because the exact asset and its
redistribution license have not been verified. The primitive visuals support
topology, transform, and joint-motion review only; they are not manufacturing
or collision-validation geometry.

### Provisional camera mount

The opt-in model exposes:

```text
camera_mount_xyz
camera_mount_rpy
```

They describe the provisional `pgi_camera_interposer -> d435i_mount`
transform. Final values require adapter design, CAD inspection, hand-eye
calibration, and collision validation. They must never overwrite
`config/ur10e_real/d435i_mount_calibration.json`.

### Verification gates

1. Source ROS 2 Jazzy and the project workspace.
2. Expand both the default and PGI Xacro variants.
3. Run `check_urdf` on both outputs.
4. Reject duplicate link/joint names, invalid topology, or invalid inertia.
5. Confirm PGI links are absent from the default variant.
6. Confirm both jaw limits, opposite axes, mimic relation, and 80 mm total
   stroke.
7. Confirm visual/collision origins agree for every provisional primitive.
8. Display the UR10e + PGI model in RViz using only robot/joint state
   publishers.
9. Confirm the camera branch is fixed to the interposer.
10. Do not start Gazebo control, MoveIt execution, `ur_robot_driver`, or RS485.

Run the non-motion checks with:

```bash
scripts/verify_pgi_140_80_description.sh
```

Display it with:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dase-hw101/ros2_ws/install/setup.bash
ros2 launch /home/dase-hw101/ur_drinking_project/launch/display_pgi_140_80.launch.py \
  use_pgi_gripper:=true
```

Move the independent left-jaw slider from `0.000` to `0.040`; the right jaw
must move symmetrically through its mimic relation.

## Later workflow, intentionally not implemented

Step 4 adds simulated control only after exact kinematics and geometry pass
review. Step 5 adds MoveIt groups, end-effector semantics, touch links, camera
adapter collision geometry, and the grasp-center frame. Step 6 validates cup
ownership with logical attach/detach before any contact model. Step 7 tunes
contact, friction, mass, release, and drop behavior. Step 8 exposes reusable
high-level simulated tools. Step 9 implements a separate real Modbus RTU/RS485
backend. Step 10 is a mandatory physical-integration review of payload, center
of gravity, TCP, hand-eye transform, and all collision geometry before any real
execution can be considered.
