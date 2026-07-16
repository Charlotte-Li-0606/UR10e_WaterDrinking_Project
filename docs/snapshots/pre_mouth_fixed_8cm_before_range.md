# Fixed 8 cm pre-mouth policy snapshot

Created before enabling the constrained pre-mouth standoff range.

The previous LLM/feeding-tools policy used exactly one straw-tip target:

```text
pre_mouth_target = detected_mouth_in_base_link + [0.0, -0.08, 0.0]
```

It retained the current flange-down yaw, required the target to pass the
workspace and tool-radius guards, applied the deterministic human PlanningScene
objects and keepouts, then used MoveIt plan-only preflight before execution.

Baseline file hashes:

```text
6bfb1089c1f62424ffb6c9f70148eb8339b0fc398ad4538d78c61bf0b3127ad6  robot_layer/arm_ur10e/agent_tools/feeding_tools.py
593018e27bfa9bac5c6f04033a8c88691c81200a3fb92c04013bbc7cf27223b5  robot_layer/arm_ur10e/agent_server/no_llm_feeding_agent_node.py
2968b76031cecd2d5865df8e50547542a179379ec6d3d12a307307ddcdc35acb  config/ur10e_sdk_config.yaml
```

The range implementation changes only `feeding_tools.py`.  The no-LLM node,
SDK, controller, and fixed PlanningScene geometry remain at this snapshot's
behavior.
