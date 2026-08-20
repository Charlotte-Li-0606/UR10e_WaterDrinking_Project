# Real pre-mouth feeding-vector policy

The no-motion mouth-axis measurements did not show a gross camera TF axis
swap or a major optical-frame rotation error. In the current real setup,
camera horizontal maps mainly to `base_link` X, camera depth maps mainly to
`base_link` Y, and image-up maps to `base_link` Z. A fixed base-X pre-mouth
offset therefore moves mainly sideways from the user/camera perspective and
is retained only as a legacy diagnostic policy.

Camera-ray remains available for geometry comparison, but it is a diagnostic
direction rather than the selected feeding direction. The real default for
the configurable feeding-vector is now `[0.0, -1.0, 0.0]`. With the default
`plus` sign and 0.05 m safe distance, the selected point is:

`pre_mouth = mouth + [0.0, -0.05, 0.0]`

The plan-only probe preserves the current `tool0` orientation and computes:

`target_tool0_position = pre_mouth - R_tool0_current * [0.110, 0.0, 0.0]`

No robot motion is part of this policy update or its marker validation.
