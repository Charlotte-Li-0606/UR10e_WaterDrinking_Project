# Grasp-Anything source and local model record

## Pinned official source

| Field | Recorded value |
|---|---|
| Project | Grasp-Anything: Large-scale Grasp Dataset from Foundation Models |
| Official repository | <https://github.com/Fsoft-AIC/Grasp-Anything> |
| Local install date | 2026-08-27 |
| Pinned commit | `d7755f43c5518bd6590b25021054f862e65bddd5` |
| Upstream model filename | `weights/model_grasp_anything` |
| Model SHA-256 | `65984ef3364790c1ece107f22bcbeb67dc8fba21784087bb3d8ff183a3582e0a` |
| Upstream license | MIT |
| License SHA-256 | `5aa84220e54962436a294d30330c9293afb1d0a899ff77533bdd7e6dd984fc88` |
| Model input | RGB, 3 channels |
| Model output | 2-D centre pixel, in-plane angle, pixel opening, and quality score |

The complete upstream checkout, its model files, and the Python virtual
environment are deliberately stored under gitignored paths:

```text
assets/vendor/grasp_anything/
.venv/grasp_anything/
```

Run `scripts/setup_grasp_anything_cpu.sh` to reproduce the checkout and CPU
environment. The script checks out the exact commit and refuses a model whose
checksum differs from the value above. Raw upstream weights are not copied
into project source or committed a second time.

## Project adapter

The upstream checkpoint is a 2-D parallel-jaw RGB grasp detector, not a ROS
service and not a complete 6-D robot grasp planner. This project therefore
keeps PyTorch in a localhost-only process and uses a separate ROS adapter:

```text
Gazebo/registered D435i RGB
  -> localhost Grasp-Anything CPU inference
  -> 2-D proposals
registered depth + CameraInfo
  -> visible surface point, normal, metric jaw opening
TF
  -> observation-only candidate in base_link
```

The upstream preprocessing uses a 224 x 224 crop. The ROS adapter uses
registered depth transformed to `base_link` to select the largest connected
component 8-350 mm above the ground, without colour or model dimensions. It
adds context, takes an adaptive square RGB crop (112 px minimum in the current
camera), and sends that crop to the service. The service resizes its input to
224 x 224 and maps position, angle, and learned pixel opening back to the crop;
the adapter then maps the result back to the full registered image. Adaptive
object scaling is a recorded project deviation and needs dataset-level
evaluation before physical use. A CPU inference takes approximately 0.9-1.3
seconds on this machine, so the ROS adapter is capped at 0.5 Hz.

The current wrist-mounted camera sees a fixed part of the PGI body at the right
edge. `self_mask_right_fraction: 0.73` rejects that known self-observation.
This provisional image mask must be revalidated whenever the camera mount,
resolution, crop, or field of view changes.

## Limits and safety meaning

- The model can propose previously unseen object shapes and image locations;
  it does not guarantee that the selected object is a cup.
- Registered depth recovers a visible surface pose only. The published point
  is not yet the PGI TCP or a collision-checked pre-grasp pose.
- A tilted visible surface can produce a tilted 6-D candidate. This does not
  replace multi-view reconstruction or prove force closure.
- Partial occlusion is accepted only while at least 60 local depth samples,
  30% patch support, and a sufficiently planar visible surface remain. Severe
  occlusion is refused instead of guessed.
- The learned rectangle provides centre and closing direction. Physical
  opening comes from the selected registered-depth silhouette plus 4 mm
  clearance; the learned opening remains diagnostic evidence. Depth openings
  outside 5-80 mm are refused because the PGI total stroke is 80 mm.
- Outputs use `/pgi/grasp_anything/*` and are not consumed by the current
  MoveIt/contact demo. No trajectory or controller command can be produced by
  either perception process.

Recorded Gazebo evidence on 2026-08-27:

- upright overhead cup: candidate on cup, 120.1 mm depth opening, correctly
  refused as wider than the PGI stroke;
- horizontal cup: candidate at the narrow end, 66.635 mm depth opening, score
  0.697, 98.8% local depth support, accepted as an observation-only pose;
- severe sparse-depth occlusion: rejected by the geometry unit test;
- physically supported tilted-cup and realistic partial-occlusion scene tests
  remain pending.

The official Grasp-Anything-6D research repository is
<https://github.com/Fsoft-AIC/Language-Driven-6-DoF-Grasp-Detection-Using-Negative-Prompt-Guidance>.
At the recorded date it documents dataset/training and offline generation, but
does not supply a ready project checkpoint that can simply replace the 2-D
model. It is therefore not represented as an available runtime backend.
