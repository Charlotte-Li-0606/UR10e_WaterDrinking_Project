# Feeding-vector real-execution preflight — 2026-07-23

No robot motion was sent during this preflight.

## Version state

- Active probe: saved version 2, SHA-256
  `55089720fa1b7f53d3afb78fc0afe78ae402ded02f42c0f9555cf4ebf7341257`
- Version 1 remains archived at
  `scripts/versions/real_premouth_from_perception_plan_v1_morning_success.py`
- Version 2 also remains preserved in Git stash `stash@{0}`.

## Read-only result

The v2 check for `feeding-vector`, input `[1, 0, 0]`, sign `plus`, and 0.05 m
stand-off succeeded. The full face was detected at approximately 0.467 m camera
depth. Eight samples had a maximum position spread of 0.000426 m. The surface
normal was stable with mean `[0.205353, -0.970545, 0.125986]` and maximum
angular spread 1.877 degrees. The speed slider was 15% and both required
controllers were active.

The controller manager logged one 4.941 ms overrun when the check participant
joined, but the reverse interface remained connected.

## Reasons execution remains blocked

1. The v2 execution gate intentionally accepts only a visually validated
   `camera-ray` policy; it rejects `feeding-vector` before creating an execution
   action client.
2. The fixed `[1, 0, 0]` feeding vector has dot product 0.205 with the measured
   camera-facing surface normal. A 0.05 m Euclidean offset along that vector
   therefore provides only about 0.0103 m signed clearance from the detected
   face plane.
3. With the target at its current position, the fixed-vector straw-tip
   displacement would be approximately 0.446 m, exceeding v2's 0.30 m planning
   guard. Using the measured face normal would still require approximately
   0.396 m at the current target location.
4. The wrist-camera mount calibration is still marked `provisional`.

Before any real execution, the operator must choose whether "vector method"
means the saved fixed base-frame vector or the detected surface-normal vector.
The target must also be returned near the earlier approximately 0.33 m camera
depth so a fresh no-motion plan can be validated within the translation guard.

Check report:
`reports/premouth_feeding_vector_posx_v2_check_20260723.json`

