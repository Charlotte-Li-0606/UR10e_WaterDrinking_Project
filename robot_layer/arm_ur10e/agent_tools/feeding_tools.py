#!/usr/bin/env python3
"""Safe, structured feeding tools built on the working no-LLM pipeline.

This module deliberately does not expose joints, controller commands, generic
Cartesian poses, or a gripper.  It accepts only the live perception result and
uses the existing UR10e SDK's fixed-orientation straw-tip primitives.

The default motion policy is dry-run.  Passing ``execute=True`` is required
for a tool to send the already-preflighted *pre-mouth* MoveIt motion.  Direct
mouth motion is disabled unless an integrator explicitly enables it when
creating the library; the smoke-test CLI never enables it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from robot_layer.arm_ur10e.perception.active_target_manager import (
    ActiveTargetManager,
)
from robot_layer.arm_ur10e.agent_tools.planning_scene_manager import (
    PlanningSceneObstacleConfig,
    PlanningSceneObstacleManager,
)


def _project_root() -> Path:
    """Return the repository root without requiring an installed package."""
    return Path(__file__).resolve().parents[3]


def _load_sdk_type():
    """Load the project SDK exactly as the established no-LLM node does."""
    module_path = _project_root() / "robot_layer" / "arm_ur10e" / "agent_server" / "robot_sdk" / "ur10e_sdk.py"
    module_name = "ur10e_sdk_for_feeding_tools"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.UR10eRobotEnv
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load UR10e SDK from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.UR10eRobotEnv


def _rotation_matrix(transform: TransformStamped) -> np.ndarray:
    """Return a normalized 3x3 rotation matrix from a TF transform."""
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("TF transform has a zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Return a normalized rotation matrix from an SDK quaternion [x,y,z,w]."""
    if len(quaternion) != 4:
        raise ValueError("Quaternion must have four values")
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _jsonable(value: Any) -> Any:
    """Convert numpy-backed SDK values into tool-result JSON primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _xyz(value: Sequence[float]) -> list[float]:
    return [round(float(item), 6) for item in value]


SAFE_FEEDING_TOOL_NAMES = frozenset(
    {
        "get_feeding_observation",
        "detect_mouth",
        "active_search_mouth",
        "select_target",
        "move_straw_tip_to_pre_mouth",
        "check_feeding_progress",
        "hold_pre_mouth",
        "retreat_to_ready",
        "feed_water",
    }
)


class FeedingToolValidationError(ValueError):
    """Raised when a caller requests anything outside the safe tool surface."""


def _validate_safe_target_selection(value: Any) -> str:
    if not isinstance(value, str):
        raise FeedingToolValidationError("target_selection must be one of: left, center, right")
    selection = value.strip().lower()
    if selection not in {"left", "center", "right"}:
        raise FeedingToolValidationError("target_selection must be one of: left, center, right")
    return selection


def _validate_safe_search_time(value: Any) -> float:
    if isinstance(value, bool):
        raise FeedingToolValidationError("max_search_time_sec must be finite and between 0.1 and 30.0")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise FeedingToolValidationError("max_search_time_sec must be finite and between 0.1 and 30.0") from exc
    if not math.isfinite(timeout) or not 0.1 <= timeout <= 30.0:
        raise FeedingToolValidationError("max_search_time_sec must be finite and between 0.1 and 30.0")
    return timeout


def _validated_execute_argument(value: Any, *, cli_execute: bool) -> bool:
    if not isinstance(value, bool):
        raise FeedingToolValidationError("execute must be a boolean")
    # CLI/operator permission is the only authority that may enable a motion.
    return bool(cli_execute and value)


def validate_safe_feeding_tool_call(
    tool: Any,
    args: Mapping[str, Any] | None = None,
    *,
    cli_execute: bool = False,
) -> dict[str, Any]:
    """Normalize one ABot/OpenClaw-style feeding tool call.

    The accepted surface deliberately contains no joints, poses, trajectories,
    controller commands, gripper actions, attachment actions, or direct mouth
    contact. ``cli_execute=False`` always forces every motion-capable tool into
    its dry-run form.
    """
    if not isinstance(tool, str) or tool not in SAFE_FEEDING_TOOL_NAMES:
        raise FeedingToolValidationError("tool is not in the approved feeding tool set")
    if args is None:
        raw_args: Mapping[str, Any] = {}
    elif isinstance(args, Mapping):
        raw_args = args
    else:
        raise FeedingToolValidationError(f"{tool}.args must be an object")

    allowed: dict[str, set[str]] = {
        "get_feeding_observation": set(),
        "detect_mouth": set(),
        "active_search_mouth": {"max_search_time_sec", "target_selection", "execute"},
        "select_target": {"target_selection"},
        "move_straw_tip_to_pre_mouth": {"execute"},
        "check_feeding_progress": set(),
        "hold_pre_mouth": {"duration_sec"},
        "retreat_to_ready": {"execute"},
        "feed_water": {
            "target_selection",
            "execute",
            "max_search_time_sec",
            "allow_direct_mouth_contact",
            "allow_vertical_adjust",
        },
    }
    extra = set(raw_args) - allowed[tool]
    if extra:
        raise FeedingToolValidationError(f"{tool} received unsupported arguments: {', '.join(sorted(extra))}")

    if tool in {"get_feeding_observation", "detect_mouth", "check_feeding_progress"}:
        return {"tool": tool, "args": {}}
    if tool == "select_target":
        return {"tool": tool, "args": {"target_selection": _validate_safe_target_selection(raw_args.get("target_selection", "center"))}}
    if tool == "active_search_mouth":
        return {
            "tool": tool,
            "args": {
                "max_search_time_sec": _validate_safe_search_time(raw_args.get("max_search_time_sec", 30.0)),
                "target_selection": _validate_safe_target_selection(raw_args.get("target_selection", "center")),
                "execute": _validated_execute_argument(raw_args.get("execute", False), cli_execute=cli_execute),
            },
        }
    if tool in {"move_straw_tip_to_pre_mouth", "retreat_to_ready"}:
        return {
            "tool": tool,
            "args": {"execute": _validated_execute_argument(raw_args.get("execute", False), cli_execute=cli_execute)},
        }
    if tool == "hold_pre_mouth":
        value = raw_args.get("duration_sec", 3.0)
        if isinstance(value, bool):
            raise FeedingToolValidationError("duration_sec must be finite and between 0.1 and 30.0")
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise FeedingToolValidationError("duration_sec must be finite and between 0.1 and 30.0") from exc
        if not math.isfinite(duration) or not 0.1 <= duration <= 30.0:
            raise FeedingToolValidationError("duration_sec must be finite and between 0.1 and 30.0")
        return {"tool": tool, "args": {"duration_sec": duration}}

    # The remaining approved tool is the backwards-compatible high-level call.
    direct_contact = raw_args.get("allow_direct_mouth_contact", False)
    if not isinstance(direct_contact, bool):
        raise FeedingToolValidationError("allow_direct_mouth_contact must be a boolean")
    if direct_contact:
        raise FeedingToolValidationError("direct mouth contact is not supported by the current feeding MVP")
    vertical_adjust = raw_args.get("allow_vertical_adjust", True)
    if not isinstance(vertical_adjust, bool):
        raise FeedingToolValidationError("allow_vertical_adjust must be a boolean")
    return {
        "tool": tool,
        "args": {
            "target_selection": _validate_safe_target_selection(raw_args.get("target_selection", "center")),
            "execute": _validated_execute_argument(raw_args.get("execute", False), cli_execute=cli_execute),
            "max_search_time_sec": _validate_safe_search_time(raw_args.get("max_search_time_sec", 30.0)),
            "allow_direct_mouth_contact": False,
            "allow_vertical_adjust": vertical_adjust,
        },
    }


def validate_safe_feeding_tool_plan(plan: Mapping[str, Any] | str, *, cli_execute: bool = False) -> dict[str, Any]:
    """Validate a sequence of approved safe calls for a tool-capable agent."""
    if isinstance(plan, str):
        try:
            raw_plan: Any = json.loads(plan)
        except json.JSONDecodeError as exc:
            raise FeedingToolValidationError(f"tool plan is not valid JSON: {exc.msg}") from exc
    else:
        raw_plan = plan
    if not isinstance(raw_plan, Mapping):
        raise FeedingToolValidationError("tool plan must be an object")
    steps = raw_plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise FeedingToolValidationError("tool plan must contain at least one safe step")
    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise FeedingToolValidationError(f"step {index} must be an object")
        normalized_steps.append(
            validate_safe_feeding_tool_call(step.get("tool"), step.get("args", {}), cli_execute=cli_execute)
        )
    return {"steps": normalized_steps, "cli_execute": bool(cli_execute)}


@dataclass(frozen=True)
class FeedingSafetyConfig:
    """Conservative defaults copied from the proven no-LLM feeding node."""

    mouth_topic: str = "/detected_mouth_pose"
    # Metadata-preserving multi-face stream.  The legacy mouth_topic remains
    # available for one-person deployments and older consumers.
    mouth_candidates_topic: str = "/detected_mouth_candidates"
    base_frame: str = "base_link"
    # Legacy/preferred target retained for compatibility with existing logs.
    # Actual pre-mouth execution considers only the constrained standoffs
    # below, along the fixed approach axis, and never commands mouth contact.
    pre_mouth_offset: tuple[float, float, float] = (0.0, -0.08, 0.0)
    pre_mouth_approach_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
    pre_mouth_standoff_min_m: float = 0.05
    pre_mouth_standoff_max_m: float = 0.10
    pre_mouth_standoff_step_m: float = 0.01
    pre_mouth_preferred_standoff_m: float = 0.08
    workspace_min: tuple[float, float, float] = (0.05, 0.15, 0.35)
    workspace_max: tuple[float, float, float] = (0.85, 1.25, 1.95)
    max_tool_radius_m: float = 1.30
    # Two-face MediaPipe runs at about 2 Hz on this ThinkPad. Three samples
    # still span the one-second stability window, whereas five can never fit
    # before the oldest sample ages out.
    stable_samples: int = 3
    stability_window_sec: float = 1.0
    max_pose_spread_m: float = 0.025
    max_pose_age_sec: float = 1.0
    target_queue_size: int = 20
    target_stale_timeout_sec: float = 1.5
    target_max_position_std_m: float = 0.03
    target_max_jump_m: float = 0.08
    wait_timeout_sec: float = 20.0
    duration_sec: float = 7.0
    # MoveGroup uses MoveIt's Time-Optimal and Ruckig response adapters for a
    # smooth, jerk-limited trajectory while keeping MoveIt's collision checks.
    planning_mode: str = "move_group"
    max_abs_delta_z_per_call_m: float = 0.03
    allowed_control_point_z_min_m: float = 0.40
    allowed_control_point_z_max_m: float = 1.80
    search_max_time_sec: float = 30.0
    search_max_steps: int = 15
    search_vertical_step_m: float = 0.02
    search_lateral_step_m: float = 0.02
    # In this UR10e camera mounting, +base_link X increases the wrist-camera
    # standoff from the seated/standing human.  This is the calibrated safe
    # "back" direction, despite the older generic example using -X.
    search_back_x_step_m: float = 0.03
    search_step_duration_sec: float = 2.0
    search_observation_wait_sec: float = 0.25
    # Deterministic MoveIt world objects are independent of Gazebo visuals.
    # They are refreshed from the selected mouth pose before motion preflight.
    use_planning_scene: bool = True
    planning_scene_required_for_execute: bool = True
    planning_scene_verify: bool = True
    planning_scene_head_radius_m: float = 0.12
    planning_scene_head_offset_m: tuple[float, float, float] = (0.0, 0.10, 0.03)
    planning_scene_face_safety_radius_m: float = 0.16
    planning_scene_face_safety_offset_m: tuple[float, float, float] = (0.0, 0.15, 0.03)
    planning_scene_torso_offset_m: tuple[float, float, float] = (0.0, 0.10, -0.35)
    planning_scene_torso_size_m: tuple[float, float, float] = (0.40, 0.25, 0.60)
    planning_scene_service_timeout_sec: float = 5.0


class FeedingSkillLibrary:
    """Synchronous safe skills intended to be called by a future agent layer.

    Each public method returns a JSON-serializable dictionary.  The caller can
    inspect ``success`` and ``reason`` instead of parsing console output.
    This library is intentionally single-threaded; use one instance per agent
    process and call :meth:`close` when finished.
    """

    def __init__(
        self,
        config: FeedingSafetyConfig | None = None,
        *,
        allow_direct_mouth_motion: bool = False,
        use_planning_scene: bool | None = None,
    ) -> None:
        self.config = config or FeedingSafetyConfig()
        self._validate_config(self.config)
        self._allow_direct_mouth_motion = bool(allow_direct_mouth_motion)
        self._use_planning_scene = (
            bool(self.config.use_planning_scene)
            if use_planning_scene is None
            else bool(use_planning_scene)
        )
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._node = Node("feeding_skill_library")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._active_targets = ActiveTargetManager(
            max_queue_size=self.config.target_queue_size,
            stale_timeout_sec=self.config.target_stale_timeout_sec,
            stability_window_sec=self.config.stability_window_sec,
            max_position_std_m=self.config.target_max_position_std_m,
            max_jump_m=self.config.target_max_jump_m,
            min_samples=self.config.stable_samples,
        )
        self._last_input_reason = "mouth pose is missing"
        self._last_candidates_received_monotonic = float("-inf")
        self._search_fallback: dict[str, Any] | None = None
        self._mouth_sub = self._node.create_subscription(
            PoseStamped, self.config.mouth_topic, self._mouth_callback, qos
        )
        self._mouth_candidates_sub = self._node.create_subscription(
            String,
            self.config.mouth_candidates_topic,
            self._mouth_candidates_callback,
            qos,
        )
        self._tf_buffer = Buffer()
        # The public wait methods spin this node, avoiding a hidden executor.
        self._tf_listener = TransformListener(self._tf_buffer, self._node, spin_thread=False)

        # Match scripts/run_no_llm_feeding_agent.sh: when callers only source
        # ROS, use this project's active controller/MoveIt configuration
        # rather than the SDK's older standalone default configuration.
        project_sdk_config = _project_root() / "config" / "ur10e_sdk_config.yaml"
        if project_sdk_config.is_file():
            os.environ.setdefault("UR10E_SDK_CONFIG", str(project_sdk_config))

        sdk_type = _load_sdk_type()
        # The library owns a separate SDK node.  It uses only the SDK's fixed
        # straw-tip primitives, never generic LLM-provided robot commands.
        self._env = sdk_type(init_ros_node=False)
        self._keepout_result: dict[str, Any] | None = None
        self._planning_scene_result: dict[str, Any] | None = None
        # Small, process-local progress record for the explicit safe-tool
        # sequence. It stores observations/results only; never a caller pose,
        # controller handle, or trajectory.
        self._last_tool_results: dict[str, dict[str, Any]] = {}
        self._last_feeding_stage = "idle"
        self._last_failure: dict[str, Any] | None = None
        self._closed = False

    @staticmethod
    def _validate_config(config: FeedingSafetyConfig) -> None:
        if any(low >= high for low, high in zip(config.workspace_min, config.workspace_max)):
            raise ValueError("Every workspace minimum must be lower than its maximum")
        if config.stable_samples < 1:
            raise ValueError("stable_samples must be at least one")
        if config.stability_window_sec <= 0.0 or config.max_pose_age_sec <= 0.0:
            raise ValueError("stability_window_sec and max_pose_age_sec must be positive")
        if config.max_pose_spread_m <= 0.0 or config.max_tool_radius_m <= 0.0:
            raise ValueError("safety distance limits must be positive")
        if config.target_queue_size < config.stable_samples:
            raise ValueError("target_queue_size must be at least stable_samples")
        if min(
            config.target_stale_timeout_sec,
            config.target_max_position_std_m,
            config.target_max_jump_m,
            config.search_max_time_sec,
            config.search_vertical_step_m,
            config.search_lateral_step_m,
            config.search_step_duration_sec,
            config.search_observation_wait_sec,
        ) <= 0.0:
            raise ValueError("target queue and search limits must be positive")
        if abs(config.search_back_x_step_m) > 0.03 or max(
            config.search_vertical_step_m, config.search_lateral_step_m
        ) > 0.03:
            raise ValueError("each predefined search step must be no larger than 0.03 m")
        if config.search_max_time_sec > 30.0 or config.search_max_steps < 1:
            raise ValueError("search_max_time_sec must be at most 30 seconds and search_max_steps positive")
        if config.max_abs_delta_z_per_call_m <= 0.0:
            raise ValueError("max_abs_delta_z_per_call_m must be positive")
        axis = np.asarray(config.pre_mouth_approach_axis, dtype=np.float64)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.allclose(axis, np.asarray((0.0, -1.0, 0.0)), atol=1e-9)
        ):
            raise ValueError("pre-mouth candidates must remain on the fixed base_link -Y approach axis")
        if not (
            0.05 <= config.pre_mouth_standoff_min_m
            <= config.pre_mouth_preferred_standoff_m
            <= config.pre_mouth_standoff_max_m
            <= 0.10
        ):
            raise ValueError("pre-mouth standoff must stay within the conservative 0.05–0.10 m interval")
        if not (0.0 < config.pre_mouth_standoff_step_m <= 0.01):
            raise ValueError("pre-mouth standoff step must be in (0, 0.01] m")
        legacy_offset = np.asarray(config.pre_mouth_offset, dtype=np.float64)
        if (
            legacy_offset.shape != (3,)
            or not np.all(np.isfinite(legacy_offset))
            or not np.allclose(
                legacy_offset,
                axis * float(config.pre_mouth_preferred_standoff_m),
                atol=1e-9,
            )
        ):
            raise ValueError("pre_mouth_offset must match the preferred fixed-axis standoff")
        if config.allowed_control_point_z_min_m >= config.allowed_control_point_z_max_m:
            raise ValueError("allowed control-point Z minimum must be below maximum")
        if config.planning_mode not in {"cartesian", "move_group", "ik_joint_target"}:
            raise ValueError("planning_mode must be cartesian, move_group, or ik_joint_target")
        if (
            config.planning_scene_head_radius_m <= 0.0
            or config.planning_scene_face_safety_radius_m <= 0.0
            or config.planning_scene_service_timeout_sec <= 0.0
        ):
            raise ValueError("PlanningScene radii and service timeout must be positive")
        for name, value in (
            ("planning_scene_head_offset_m", config.planning_scene_head_offset_m),
            ("planning_scene_face_safety_offset_m", config.planning_scene_face_safety_offset_m),
            ("planning_scene_torso_offset_m", config.planning_scene_torso_offset_m),
            ("planning_scene_torso_size_m", config.planning_scene_torso_size_m),
        ):
            if len(value) != 3 or not all(math.isfinite(float(component)) for component in value):
                raise ValueError(f"{name} must contain three finite values")
        if any(float(component) <= 0.0 for component in config.planning_scene_torso_size_m):
            raise ValueError("planning_scene_torso_size_m must contain positive values")

    def __enter__(self) -> "FeedingSkillLibrary":
        return self

    def __exit__(self, *_unused: object) -> None:
        self.close()

    def close(self) -> None:
        """Release ROS nodes without stopping or commanding robot motion."""
        if self._closed:
            return
        self._closed = True
        self._env.close()
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _failure(self, tool: str, reason: str, **fields: Any) -> dict[str, Any]:
        return _jsonable({"success": False, "tool": tool, "reason": reason, **fields})

    def _success(self, tool: str, **fields: Any) -> dict[str, Any]:
        return _jsonable({"success": True, "tool": tool, "reason": None, **fields})

    def _remember_safe_tool_result(self, stage: str, result: Mapping[str, Any]) -> dict[str, Any]:
        """Retain compact local progress without creating a second control path."""
        serialized = _jsonable(result)
        tool = str(serialized.get("tool") or stage)
        self._last_tool_results[tool] = serialized
        if serialized.get("success"):
            self._last_feeding_stage = stage
            self._last_failure = None
        else:
            self._last_feeding_stage = "failed"
            self._last_failure = {
                "failed_step": tool,
                "reason": str(serialized.get("reason") or "safe feeding tool failed"),
            }
        return serialized

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def _pose_in_base(self, message: PoseStamped) -> tuple[np.ndarray | None, str | None]:
        source_frame = message.header.frame_id.strip().lstrip("/")
        if not source_frame:
            return None, "mouth pose frame_id is missing"
        raw = np.array(
            [message.pose.position.x, message.pose.position.y, message.pose.position.z], dtype=np.float64
        )
        if not np.all(np.isfinite(raw)):
            return None, "mouth pose contains non-finite coordinates"
        if source_frame == self.config.base_frame.lstrip("/"):
            return raw, None
        try:
            transform = self._tf_buffer.lookup_transform(
                self.config.base_frame,
                source_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.20),
            )
        except Exception as exc:
            return None, f"mouth pose TF is unavailable: {exc}"
        translation = transform.transform.translation
        return _rotation_matrix(transform) @ raw + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        ), None

    def _mouth_callback(self, message: PoseStamped) -> None:
        # Once a metadata-preserving candidate stream is live, it is the
        # authoritative source for the active target.  The legacy single pose
        # remains for compatibility only and must not overwrite a user-chosen
        # left or right target.
        if time.monotonic() - self._last_candidates_received_monotonic < 0.75:
            return
        position, reason = self._pose_in_base(message)
        if position is None:
            self._last_input_reason = reason or "mouth pose was rejected"
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        update = self._active_targets.update_from_detection(
            {
                "position": position,
                "timestamp_sec": stamp,
                "received_monotonic": time.monotonic(),
                "source_frame": message.header.frame_id,
            }
        )
        self._last_input_reason = "mouth pose is available" if update.get("success") else str(update.get("reason"))

    def _mouth_candidates_callback(self, message: String) -> None:
        """Consume all perception candidates without choosing a person.

        The publisher supplies image-x sorted candidates and their base-link
        positions.  ``ActiveTargetManager`` alone maps the already selected
        user target (left/center/right) to one of those candidates and keeps
        its nearest-neighbour identity lock.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self._last_input_reason = "mouth candidate message is not valid JSON"
            return
        if not isinstance(payload, Mapping):
            self._last_input_reason = "mouth candidate message must be an object"
            return
        frame = str(payload.get("frame_id", "")).strip().lstrip("/")
        if frame != self.config.base_frame.lstrip("/"):
            self._last_input_reason = "mouth candidates must be expressed in base_link"
            return
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            self._last_input_reason = "mouth candidate message has no candidate list"
            return
        try:
            stamp = float(payload.get("stamp_sec"))
        except (TypeError, ValueError):
            stamp = None
        try:
            image_center_x = float(payload.get("image_center_x"))
        except (TypeError, ValueError):
            image_center_x = None

        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                continue
            candidate = {
                "position": raw.get("position"),
                "timestamp_sec": stamp,
                "source_frame": frame,
                "image_x": raw.get("image_x"),
                "image_center_x": image_center_x,
                "confidence": raw.get("confidence"),
            }
            candidates.append(candidate)
        update = self._active_targets.update_from_detection(candidates=candidates)
        if update.get("success"):
            self._last_candidates_received_monotonic = time.monotonic()
            self._last_input_reason = "mouth candidates are available"
        else:
            self._last_input_reason = str(update.get("reason") or "no valid mouth candidates were supplied")

    def select_active_target(self, selection: str = "center") -> dict[str, Any]:
        """Lock the logical left, center, or right target for later pose reads."""
        try:
            result = self._active_targets.select_target(selection)
        except ValueError as exc:
            return self._failure("select_active_target", str(exc))
        # A fresh user selection replaces any explicit timeout fallback from a
        # previous search session.
        self._search_fallback = None
        return self._success("select_active_target", **{key: value for key, value in result.items() if key not in {"success", "reason"}})

    def select_target(self, target_selection: str = "center") -> dict[str, Any]:
        """Safe intent-level alias for the active left/center/right selector."""
        selected = self.select_active_target(target_selection)
        result = {**selected, "tool": "select_target", "target_selection": target_selection}
        return self._remember_safe_tool_result("target_selected", result)

    def reset_active_target(self) -> dict[str, Any]:
        """Clear all target queues and return the active selection to center."""
        result = self._active_targets.reset_active_target()
        return self._success("reset_active_target", **{key: value for key, value in result.items() if key not in {"success", "reason"}})

    def get_active_target_state(self, *, spin_timeout_sec: float = 0.10) -> dict[str, Any]:
        """Return queue freshness/stability metadata for the locked target."""
        self._spin_for(spin_timeout_sec)
        state = self._active_targets.get_state()
        return self._success(
            "get_active_target_state",
            topic=self.config.mouth_topic,
            fallback=self._search_fallback,
            **state,
        )

    def _latest_sample_result(self, tool: str) -> dict[str, Any]:
        latest = self._active_targets.get_active_latest_pose()
        if not latest.get("success"):
            return self._failure(
                tool,
                str(latest.get("reason") or self._last_input_reason),
                topic=self.config.mouth_topic,
                active_target_label=self._active_targets.active_target_label,
            )
        return self._success(
            tool,
            mouth_pose=latest["pose"],
            topic=self.config.mouth_topic,
            active_target_label=latest["active_target_label"],
            active_target_id=latest["active_target_id"],
            received_samples=latest.get("num_samples", 0),
        )

    def get_latest_mouth_pose(
        self,
        *,
        spin_timeout_sec: float = 0.10,
        selection: str | None = None,
    ) -> dict[str, Any]:
        """Return a fresh live pose in ``base_link`` or a structured failure."""
        if selection is not None:
            selected = self.select_active_target(selection)
            if not selected.get("success"):
                selected["tool"] = "get_latest_mouth_pose"
                return selected
        self._spin_for(spin_timeout_sec)
        return self._latest_sample_result("get_latest_mouth_pose")

    def detect_mouth(self) -> dict[str, Any]:
        """Read the latest existing MediaPipe mouth result; no detector is recreated."""
        latest = self.get_latest_mouth_pose(spin_timeout_sec=0.10)
        if latest.get("success"):
            result = self._success(
                "detect_mouth",
                mouth_detected=True,
                mouth_pose=latest.get("mouth_pose"),
                active_target_label=latest.get("active_target_label"),
                active_target_id=latest.get("active_target_id"),
                source_topic=self.config.mouth_topic,
            )
        else:
            result = self._success(
                "detect_mouth",
                mouth_detected=False,
                mouth_pose=None,
                source_topic=self.config.mouth_topic,
                note=str(latest.get("reason") or "no current mouth pose"),
            )
        return self._remember_safe_tool_result("mouth_detected" if result["mouth_detected"] else "waiting_for_mouth", result)

    def _stable_mouth_result(self) -> dict[str, Any]:
        stable = self._active_targets.get_active_stable_pose()
        if not stable.get("success"):
            details = {
                key: value
                for key, value in stable.items()
                if key not in {"success", "stable", "reason"}
            }
            return self._failure(
                "wait_for_stable_mouth_pose",
                str(stable.get("reason") or self._last_input_reason),
                topic=self.config.mouth_topic,
                **details,
            )
        return self._success(
            "wait_for_stable_mouth_pose",
            mouth_pose=stable["pose"],
            active_target_label=stable["active_target_label"],
            active_target_id=stable["active_target_id"],
            sample_count=stable["num_samples"],
            position_std_m=stable["position_std_m"],
            max_jump_m=stable["max_jump_m"],
            observed_window_sec=stable["observed_window_sec"],
        )

    def wait_for_stable_mouth_pose(
        self,
        *,
        timeout_sec: float | None = None,
        selection: str | None = None,
    ) -> dict[str, Any]:
        """Wait for a stable selected target without overriding active intent."""
        requested = (
            self._active_targets.active_target_label
            if selection is None
            else str(selection).strip().lower()
        )
        fallback = self._search_fallback
        fallback_applies = fallback is not None and requested == fallback["requested_selection"]
        if selection is None:
            resolved_selection = requested
        elif fallback_applies:
            # The user requested (for example) right, the 30 s search found
            # only one stable person, and search explicitly resolved that
            # documented fallback to center. Do not silently switch targets in
            # any other situation.
            resolved_selection = str(fallback["resolved_selection"])
        else:
            selected = self.select_active_target(selection)
            if not selected.get("success"):
                selected["tool"] = "wait_for_stable_mouth_pose"
                return selected
            resolved_selection = requested
        timeout = self.config.wait_timeout_sec if timeout_sec is None else max(0.0, float(timeout_sec))
        deadline = time.monotonic() + timeout
        last_result = self._stable_mouth_result()
        while rclpy.ok() and time.monotonic() < deadline:
            if last_result["success"]:
                if fallback_applies:
                    last_result.update(
                        {
                            "fallback_used": True,
                            "requested_selection": requested,
                            "resolved_selection": resolved_selection,
                        }
                    )
                return last_result
            self._spin_for(min(0.10, max(0.0, deadline - time.monotonic())))
            last_result = self._stable_mouth_result()
        if last_result["success"]:
            if fallback_applies:
                last_result.update(
                    {
                        "fallback_used": True,
                        "requested_selection": requested,
                        "resolved_selection": resolved_selection,
                    }
                )
            return last_result
        last_result["timeout_sec"] = timeout
        if fallback_applies:
            last_result.update(
                {
                    "fallback_used": True,
                    "requested_selection": requested,
                    "resolved_selection": resolved_selection,
                }
            )
        return last_result

    def _validate_mouth_pose_input(self, mouth_pose: Mapping[str, Any] | None) -> dict[str, Any]:
        if mouth_pose is None:
            return self.wait_for_stable_mouth_pose()
        frame = str(mouth_pose.get("frame_id", "")).strip().lstrip("/")
        if frame != self.config.base_frame.lstrip("/"):
            return self._failure("compute_pre_mouth_target", "mouth pose must be expressed in base_link")
        position = mouth_pose.get("position")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 3:
            return self._failure("compute_pre_mouth_target", "mouth pose must contain three position values")
        try:
            values = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError):
            return self._failure("compute_pre_mouth_target", "mouth pose contains invalid position values")
        if not np.all(np.isfinite(values)):
            return self._failure("compute_pre_mouth_target", "mouth pose contains non-finite coordinates")
        age = mouth_pose.get("received_age_sec")
        if age is not None and float(age) > self.config.target_stale_timeout_sec:
            return self._failure("compute_pre_mouth_target", "mouth pose is stale", received_age_sec=float(age))
        return self._success(
            "compute_pre_mouth_target",
            mouth_pose={"position": _xyz(values), "frame_id": self.config.base_frame},
        )

    def _within_workspace(self, position: np.ndarray) -> bool:
        return bool(
            np.all(position >= np.asarray(self.config.workspace_min))
            and np.all(position <= np.asarray(self.config.workspace_max))
        )

    def _pre_mouth_standoff_candidates(self) -> list[float]:
        """Return safe standoffs, ordered from greatest face clearance first."""
        minimum = float(self.config.pre_mouth_standoff_min_m)
        maximum = float(self.config.pre_mouth_standoff_max_m)
        step = float(self.config.pre_mouth_standoff_step_m)
        candidates: list[float] = []
        value = maximum
        while value >= minimum - 1e-9:
            candidates.append(round(max(minimum, value), 6))
            value -= step
        if not math.isclose(candidates[-1], minimum, abs_tol=1e-9):
            candidates.append(round(minimum, 6))
        return candidates

    def compute_pre_mouth_target(
        self,
        mouth_pose: Mapping[str, Any] | None = None,
        *,
        standoff_m: float | None = None,
    ) -> dict[str, Any]:
        """Build one fixed-axis pre-mouth target without relaxing face clearance.

        The public default remains the historical 8 cm target for a stable
        preview. ``move_straw_tip_to_pre_mouth`` then tries the full 5–10 cm
        constrained set with MoveIt collision checking before it can execute.
        """
        validated = self._validate_mouth_pose_input(mouth_pose)
        if not validated["success"]:
            validated["tool"] = "compute_pre_mouth_target"
            return validated
        mouth = np.asarray(validated["mouth_pose"]["position"], dtype=np.float64)
        standoff = (
            float(self.config.pre_mouth_preferred_standoff_m)
            if standoff_m is None
            else float(standoff_m)
        )
        if not math.isfinite(standoff) or not (
            self.config.pre_mouth_standoff_min_m - 1e-9
            <= standoff
            <= self.config.pre_mouth_standoff_max_m + 1e-9
        ):
            return self._failure(
                "compute_pre_mouth_target",
                "requested pre-mouth standoff is outside the safe 0.05–0.10 m interval",
                requested_standoff_m=standoff_m,
            )
        approach_axis = np.asarray(self.config.pre_mouth_approach_axis, dtype=np.float64)
        pre_mouth = mouth + approach_axis * standoff
        if not self._within_workspace(mouth):
            return self._failure(
                "compute_pre_mouth_target",
                "mouth pose is outside the allowed workspace",
                mouth_pose=validated["mouth_pose"],
                workspace_min=list(self.config.workspace_min),
                workspace_max=list(self.config.workspace_max),
            )
        if not self._within_workspace(pre_mouth):
            return self._failure(
                "compute_pre_mouth_target",
                "pre-mouth target is outside the allowed workspace",
                mouth_pose=validated["mouth_pose"],
                pre_mouth_target=_xyz(pre_mouth),
            )
        # Keep the current downward wrist yaw.  Flange-down constrains the
        # tool Z axis, not its yaw; resetting yaw here can turn the camera
        # away from the person and make an otherwise reachable pre-mouth
        # target collide with the established human keepout.
        control, control_error = self._vertical_control_point()
        if control is None:
            return self._failure(
                "compute_pre_mouth_target",
                control_error or "could not preserve the current flange-down orientation",
                mouth_pose=validated["mouth_pose"],
            )
        flange_down_rpy = [float(value) for value in control["current_rpy"]]
        try:
            plan = self._env.plan_straw_tip_to_pre_mouth_pose(
                pre_mouth.tolist(),
                flange_down_rpy=flange_down_rpy,
            )
        except Exception as exc:
            return self._failure("compute_pre_mouth_target", f"could not compute tool0 target: {exc}")
        tool0_target = np.asarray(plan["tool0_target_position"], dtype=np.float64)
        tool0_radius = float(np.linalg.norm(tool0_target))
        if not self._within_workspace(tool0_target) or tool0_radius > self.config.max_tool_radius_m:
            return self._failure(
                "compute_pre_mouth_target",
                "tool0 target is outside the conservative reach envelope",
                mouth_pose=validated["mouth_pose"],
                pre_mouth_target=_xyz(pre_mouth),
                tool0_target=_xyz(tool0_target),
                tool0_radius_m=round(tool0_radius, 6),
                max_tool_radius_m=self.config.max_tool_radius_m,
            )
        return self._success(
            "compute_pre_mouth_target",
            mouth_pose=validated["mouth_pose"],
            pre_mouth_target=_xyz(pre_mouth),
            pre_mouth_offset=_xyz(approach_axis * standoff),
            pre_mouth_approach_axis=list(self.config.pre_mouth_approach_axis),
            pre_mouth_standoff_m=round(standoff, 6),
            candidate_standoffs_m=self._pre_mouth_standoff_candidates(),
            tool0_target=_xyz(tool0_target),
            tool0_radius_m=round(tool0_radius, 6),
            planner_target=plan,
            flange_down_alignment=round(float(control["flange_down_alignment"]), 6),
        )

    def _planning_scene_config(self) -> PlanningSceneObstacleConfig:
        """Translate the feeding safety configuration into obstacle geometry."""
        return PlanningSceneObstacleConfig(
            base_frame=self.config.base_frame,
            mouth_topic=self.config.mouth_topic,
            head_radius_m=self.config.planning_scene_head_radius_m,
            head_offset_m=self.config.planning_scene_head_offset_m,
            face_safety_radius_m=self.config.planning_scene_face_safety_radius_m,
            face_safety_offset_m=self.config.planning_scene_face_safety_offset_m,
            torso_offset_m=self.config.planning_scene_torso_offset_m,
            torso_size_m=self.config.planning_scene_torso_size_m,
            service_timeout_sec=self.config.planning_scene_service_timeout_sec,
            mouth_wait_timeout_sec=self.config.wait_timeout_sec,
        )

    def ensure_planning_scene_obstacles(
        self,
        mouth_pose: Mapping[str, Any] | None = None,
        *,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Apply dynamic human primitives before a MoveIt motion preflight.

        A current pose is used when supplied by the selected target.  For
        retreat/vertical adjustments this method waits briefly for the active
        target queue.  Missing perception is a warning for dry-run planning,
        but is a safe failure for execution when this protection is required.
        """
        if not self._use_planning_scene:
            return self._success(
                "ensure_planning_scene_obstacles",
                enabled=False,
                applied=False,
                note="PlanningScene obstacle manager was explicitly disabled.",
            )

        validated = self._validate_mouth_pose_input(mouth_pose)
        if not validated.get("success"):
            reason = str(validated.get("reason") or "detected mouth pose is unavailable")
            if execute and self.config.planning_scene_required_for_execute:
                return self._failure(
                    "ensure_planning_scene_obstacles",
                    reason,
                    enabled=True,
                    applied=False,
                    execute=True,
                    note="Execution requires a current mouth-derived PlanningScene obstacle update.",
                )
            return self._success(
                "ensure_planning_scene_obstacles",
                enabled=True,
                applied=False,
                warning=reason,
                note="Continuing without a dynamic PlanningScene update because this is plan-only mode.",
            )

        manager = PlanningSceneObstacleManager(self._planning_scene_config())
        try:
            applied = manager.apply(
                validated["mouth_pose"]["position"],
                verify=self.config.planning_scene_verify,
            )
        except Exception as exc:
            applied = {"success": False, "reason": f"PlanningScene manager raised: {exc}"}
        finally:
            manager.destroy_node()

        if not applied.get("success"):
            reason = str(applied.get("reason") or "PlanningScene obstacle update failed")
            if execute and self.config.planning_scene_required_for_execute:
                return self._failure(
                    "ensure_planning_scene_obstacles",
                    reason,
                    enabled=True,
                    applied=False,
                    execute=True,
                    planning_scene=applied,
                )
            return self._success(
                "ensure_planning_scene_obstacles",
                enabled=True,
                applied=False,
                warning=reason,
                planning_scene=applied,
                note="Continuing without a verified dynamic PlanningScene update because this is plan-only mode.",
            )
        return self._success(
            "ensure_planning_scene_obstacles",
            enabled=True,
            applied=True,
            planning_scene=applied,
        )

    def _ensure_planning_scene_for_motion(
        self,
        tool: str,
        mouth_pose: Mapping[str, Any] | None,
        *,
        execute: bool,
    ) -> dict[str, Any] | None:
        self._planning_scene_result = self.ensure_planning_scene_obstacles(mouth_pose, execute=execute)
        if self._planning_scene_result.get("success"):
            return None
        return self._failure(
            tool,
            str(self._planning_scene_result.get("reason") or "PlanningScene obstacle update failed"),
            planning_scene=self._planning_scene_result,
        )

    def _ensure_human_keepout(self, tool: str) -> dict[str, Any] | None:
        try:
            self._keepout_result = self._env.apply_human_keepout()
        except Exception as exc:
            return self._failure(tool, f"human keepout could not be applied: {exc}")
        if not self._keepout_result.get("success"):
            return self._failure(tool, "MoveIt rejected the human keepout", keepout=self._keepout_result)
        return None

    def _preflight_pre_mouth_target(self, target: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        """Collision-check one already constrained candidate without motion."""
        pre_mouth = target["pre_mouth_target"]
        flange_down_rpy = target["planner_target"]["flange_down_rpy"]
        planning_mode = self.config.planning_mode
        try:
            preflight = self._env.move_straw_tip_to_pre_mouth(
                pre_mouth_safe_position=pre_mouth,
                flange_down_rpy=flange_down_rpy,
                duration=self.config.duration_sec,
                plan_only=True,
                planning_mode=planning_mode,
            )
        except Exception as exc:
            return ({"success": False, "reason": f"MoveIt preflight raised an exception: {exc}"}, planning_mode)

        # The existing Cartesian fallback is retained, but only for this same
        # fixed-axis candidate and only after MoveGroup collision checking.
        if not preflight.get("success") and planning_mode == "move_group":
            try:
                cartesian_preflight = self._env.move_straw_tip_to_pre_mouth(
                    pre_mouth_safe_position=pre_mouth,
                    flange_down_rpy=flange_down_rpy,
                    duration=self.config.duration_sec,
                    plan_only=True,
                    planning_mode="cartesian",
                )
            except Exception as exc:
                cartesian_preflight = {
                    "success": False,
                    "reason": f"MoveIt Cartesian fallback preflight raised an exception: {exc}",
                }
            if cartesian_preflight.get("success"):
                preflight = cartesian_preflight
                planning_mode = "cartesian"
        return preflight, planning_mode

    def _move_straw_tip_to_pre_mouth_impl(
        self,
        mouth_pose: Mapping[str, Any] | None = None,
        *,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Use the furthest collision-free target in the fixed 5–10 cm zone."""
        preview = self.compute_pre_mouth_target(mouth_pose)
        if not preview["success"]:
            return self._failure("move_straw_tip_to_pre_mouth", preview["reason"], target=preview, execute=execute)
        planning_scene_failure = self._ensure_planning_scene_for_motion(
            "move_straw_tip_to_pre_mouth",
            preview["mouth_pose"],
            execute=execute,
        )
        if planning_scene_failure is not None:
            planning_scene_failure.update({"execute": execute, "target": preview})
            return planning_scene_failure
        keepout_failure = self._ensure_human_keepout("move_straw_tip_to_pre_mouth")
        if keepout_failure is not None:
            keepout_failure.update({"execute": execute, "target": preview})
            return keepout_failure

        candidate_attempts: list[dict[str, Any]] = []
        selected_target: dict[str, Any] | None = None
        selected_preflight: dict[str, Any] | None = None
        selected_planning_mode: str | None = None
        for standoff_m in self._pre_mouth_standoff_candidates():
            candidate = self.compute_pre_mouth_target(preview["mouth_pose"], standoff_m=standoff_m)
            if not candidate.get("success"):
                candidate_attempts.append(
                    {
                        "standoff_m": standoff_m,
                        "success": False,
                        "reason": candidate.get("reason"),
                    }
                )
                continue
            preflight, planning_mode = self._preflight_pre_mouth_target(candidate)
            candidate_attempts.append(
                {
                    "standoff_m": standoff_m,
                    "pre_mouth_target": candidate["pre_mouth_target"],
                    "planner": planning_mode,
                    "planner_result": preflight,
                    "success": bool(preflight.get("success")),
                }
            )
            if preflight.get("success"):
                selected_target = candidate
                selected_preflight = preflight
                selected_planning_mode = planning_mode
                break

        if selected_target is None or selected_preflight is None or selected_planning_mode is None:
            return self._failure(
                "move_straw_tip_to_pre_mouth",
                "MoveIt could not plan any safe pre-mouth standoff in the 0.05–0.10 m zone",
                target=preview,
                candidate_attempts=candidate_attempts,
                execute=execute,
            )
        pre_mouth = selected_target["pre_mouth_target"]
        flange_down_rpy = selected_target["planner_target"]["flange_down_rpy"]
        if not execute:
            return self._success(
                "move_straw_tip_to_pre_mouth",
                execute=False,
                mouth_pose=selected_target["mouth_pose"],
                pre_mouth_target=pre_mouth,
                selected_standoff_m=selected_target["pre_mouth_standoff_m"],
                candidate_attempts=candidate_attempts,
                planning_mode=selected_planning_mode,
                planner_result=selected_preflight,
                keepout=self._keepout_result,
                planning_scene=self._planning_scene_result,
                note="Dry-run only; selected the furthest collision-free fixed-axis pre-mouth candidate.",
            )
        try:
            result = self._env.move_straw_tip_to_pre_mouth(
                pre_mouth_safe_position=pre_mouth,
                flange_down_rpy=flange_down_rpy,
                duration=self.config.duration_sec,
                plan_only=False,
                planning_mode=selected_planning_mode,
            )
        except Exception as exc:
            return self._failure(
                "move_straw_tip_to_pre_mouth",
                f"MoveIt execution raised an exception: {exc}",
                target=selected_target,
                candidate_attempts=candidate_attempts,
                execute=True,
                planner_result=selected_preflight,
            )
        if not result.get("success"):
            return self._failure(
                "move_straw_tip_to_pre_mouth",
                "MoveIt did not complete the selected pre-mouth motion",
                target=selected_target,
                candidate_attempts=candidate_attempts,
                execute=True,
                planner_result=selected_preflight,
                move_result=result,
            )
        return self._success(
            "move_straw_tip_to_pre_mouth",
            execute=True,
            mouth_pose=selected_target["mouth_pose"],
            pre_mouth_target=pre_mouth,
            selected_standoff_m=selected_target["pre_mouth_standoff_m"],
            candidate_attempts=candidate_attempts,
            planning_mode=selected_planning_mode,
            planner_result=selected_preflight,
            move_result=result,
            keepout=self._keepout_result,
            planning_scene=self._planning_scene_result,
            note="Stopped at the selected safe pre-mouth candidate; no direct mouth contact command was sent.",
        )

    def move_straw_tip_to_pre_mouth(
        self,
        mouth_pose: Mapping[str, Any] | None = None,
        *,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Plan or execute only the fixed flange-down safe pre-mouth target."""
        result = self._move_straw_tip_to_pre_mouth_impl(mouth_pose, execute=execute)
        if result.get("success"):
            stage = "at_pre_mouth" if execute else "pre_mouth_plan_validated"
        else:
            stage = "pre_mouth_blocked"
        return self._remember_safe_tool_result(stage, result)

    def _search_primitive_delta(self, primitive: str) -> np.ndarray:
        """Return one fixed base-link scan increment; never model-provided."""
        steps = {
            "search_up_small": (0.0, 0.0, self.config.search_vertical_step_m),
            "search_down_small": (0.0, 0.0, -self.config.search_vertical_step_m),
            "search_left_small": (0.0, self.config.search_lateral_step_m, 0.0),
            "search_right_small": (0.0, -self.config.search_lateral_step_m, 0.0),
            "search_back_small": (self.config.search_back_x_step_m, 0.0, 0.0),
        }
        if primitive not in steps:
            raise ValueError(f"unknown safe mouth-search primitive: {primitive}")
        return np.asarray(steps[primitive], dtype=np.float64)

    def _preflight_search_primitive(self, primitive: str) -> dict[str, Any]:
        """Prepare one fixed, flange-down search target without moving.

        Search is a camera-view recovery operation, so it must retain the
        current downward wrist yaw.  Resetting to the configured default
        flange-down RPY would satisfy the vertical constraint while pointing
        the wrist camera away from the person.
        """
        delta = self._search_primitive_delta(primitive)
        if float(np.linalg.norm(delta)) > 0.03 + 1e-9:
            return self._failure("search_for_mouth", "configured search increment exceeds 0.03 m", primitive=primitive)
        control, control_error = self._vertical_control_point()
        if control is None:
            return self._failure(
                "search_for_mouth",
                control_error or "could not preserve the current flange-down orientation",
                primitive=primitive,
            )
        current_rpy = [float(value) for value in control["current_rpy"]]
        try:
            current = np.asarray(self._env.get_straw_tip_pose()["position"], dtype=np.float64)
        except Exception as exc:
            return self._failure("search_for_mouth", f"could not read the current straw-tip pose: {exc}", primitive=primitive)
        target = current + delta
        if not self._within_workspace(target):
            return self._failure(
                "search_for_mouth",
                "safe search target is outside the allowed workspace",
                primitive=primitive,
                current_straw_tip=_xyz(current),
                target_straw_tip=_xyz(target),
            )
        try:
            planner_target = self._env.plan_straw_tip_to_pose(
                target.tolist(),
                flange_down_rpy=current_rpy,
                label=f"mouth_search_{primitive}",
            )
        except Exception as exc:
            return self._failure("search_for_mouth", f"could not construct safe search target: {exc}", primitive=primitive)
        tool0_target = np.asarray(planner_target["tool0_target_position"], dtype=np.float64)
        # ``workspace_min/max`` constrain the straw-tip control point (the
        # rigid cup/straw assembly), which was checked above.  The flange/tool0
        # can legitimately lie outside that control-point box because of its
        # fixed rotated offset: the known initial pose has tool0 Y≈0.10 while
        # the straw tip remains safely at Y≈0.21.  Keep the separate tool0
        # conservative reach and vertical limits instead of falsely rejecting
        # a yaw-preserving view-recovery translation.
        tool0_radius = float(np.linalg.norm(tool0_target))
        if (
            not np.all(np.isfinite(tool0_target))
            or tool0_radius > self.config.max_tool_radius_m
            or not (
                self.config.allowed_control_point_z_min_m
                <= float(tool0_target[2])
                <= self.config.allowed_control_point_z_max_m
            )
        ):
            return self._failure(
                "search_for_mouth",
                "safe search tool0 target is outside the conservative reach envelope",
                primitive=primitive,
                current_straw_tip=_xyz(current),
                target_straw_tip=_xyz(target),
                tool0_target=_xyz(tool0_target),
                max_tool_radius_m=self.config.max_tool_radius_m,
                allowed_tool0_z_min_m=self.config.allowed_control_point_z_min_m,
                allowed_tool0_z_max_m=self.config.allowed_control_point_z_max_m,
            )
        try:
            preflight = self._env.move_straw_tip_to_position(
                target.tolist(),
                label=f"mouth_search_{primitive}",
                flange_down_rpy=current_rpy,
                duration=self.config.search_step_duration_sec,
                plan_only=True,
                # This local, short Cartesian move preserves both the current
                # downward camera yaw and the fixed flange-down convention.
                planning_mode="cartesian",
            )
        except Exception as exc:
            return self._failure(
                "search_for_mouth",
                f"MoveIt search preflight raised an exception: {exc}",
                primitive=primitive,
                target_straw_tip=_xyz(target),
            )
        if not preflight.get("success"):
            return self._failure(
                "search_for_mouth",
                "MoveIt could not plan the safe search primitive",
                primitive=primitive,
                target_straw_tip=_xyz(target),
                planner_result=preflight,
            )
        return self._success(
            "search_for_mouth",
            primitive=primitive,
            current_straw_tip=_xyz(current),
            target_straw_tip=_xyz(target),
            planner_target=planner_target,
            planner_result=preflight,
            keep_flange_down=True,
            flange_down_alignment=round(float(control["flange_down_alignment"]), 6),
            flange_down_rpy=current_rpy,
        )

    def _execute_search_primitive(self, preflight: Mapping[str, Any]) -> dict[str, Any]:
        """Execute exactly one already-preflighted fixed search primitive."""
        primitive = str(preflight["primitive"])
        target = preflight["target_straw_tip"]
        try:
            result = self._env.move_straw_tip_to_position(
                target,
                label=f"mouth_search_{primitive}",
                flange_down_rpy=preflight["flange_down_rpy"],
                duration=self.config.search_step_duration_sec,
                plan_only=False,
                planning_mode="cartesian",
            )
        except Exception as exc:
            return self._failure(
                "search_for_mouth",
                f"MoveIt search execution raised an exception: {exc}",
                primitive=primitive,
                target_straw_tip=target,
                planner_result=preflight.get("planner_result"),
            )
        if not result.get("success"):
            return self._failure(
                "search_for_mouth",
                "MoveIt did not complete the safe search primitive",
                primitive=primitive,
                target_straw_tip=target,
                planner_result=preflight.get("planner_result"),
                move_result=result,
            )
        return self._success(
            "search_for_mouth",
            primitive=primitive,
            target_straw_tip=target,
            planner_result=preflight.get("planner_result"),
            move_result=result,
            keep_flange_down=True,
            flange_down_alignment=preflight.get("flange_down_alignment"),
        )

    def search_for_mouth(
        self,
        *,
        max_time_sec: float = 30.0,
        selection: str = "center",
        execute: bool = False,
    ) -> dict[str, Any]:
        """Find a stable active mouth pose with a bounded fixed scan pattern.

        This method deliberately accepts no direction, pose, joint, or
        controller command from a caller.  In dry-run mode it never moves: it
        either returns a currently stable active target or reports the first
        preflighted search step that would be used with ``execute=True``.
        """
        try:
            timeout = float(max_time_sec)
        except (TypeError, ValueError):
            return self._failure("search_for_mouth", "max_time_sec must be a finite number", execute=execute)
        if not math.isfinite(timeout) or not 0.1 <= timeout <= self.config.search_max_time_sec:
            return self._failure(
                "search_for_mouth",
                f"max_time_sec must be between 0.1 and {self.config.search_max_time_sec:.1f}",
                execute=execute,
            )
        selected = self.select_active_target(selection)
        if not selected.get("success"):
            selected["tool"] = "search_for_mouth"
            selected["execute"] = execute
            return selected
        started = time.monotonic()
        deadline = started + timeout
        # A newly constructed tool library has no callback history yet. Wait
        # through one bounded stability window before declaring that a visible
        # target is missing.  This avoids an unnecessary physical search when
        # multi-face MediaPipe is simply publishing at a slower rate.
        observation_deadline = min(
            deadline,
            started + max(0.5, self.config.stability_window_sec + 0.25),
        )
        initial = self._stable_mouth_result()
        while not initial.get("success") and time.monotonic() < observation_deadline:
            self._spin_for(min(0.10, max(0.0, observation_deadline - time.monotonic())))
            initial = self._stable_mouth_result()
        if initial.get("success"):
            return self._success(
                "search_for_mouth",
                execute=execute,
                selection=selection,
                found_without_motion=True,
                mouth_pose=initial["mouth_pose"],
                active_target_label=initial.get("active_target_label"),
                active_target_id=initial.get("active_target_id"),
                search_steps=[],
            )

        # The camera's calibrated recovery direction is tried first.  This is
        # important when the face was lost because the camera was too close or
        # the face reached the image edge: a small retreat widens the view
        # before the vertical/lateral scan.  Every item remains one of the
        # fixed, preflighted primitives; no model-provided direction or pose
        # can enter this sequence.
        scan_pattern = (
            # A no-face view caused by being too close needs several small,
            # calibrated standoff corrections before lateral exploration. The
            # 3 cm cap, 15-step cap, and 30-second deadline still apply.
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_back_small",
            "search_up_small",
            "search_down_small",
            "search_left_small",
            "search_right_small",
        )
        first_preflight = self._preflight_search_primitive(scan_pattern[0])
        # The SDK's TF listener is populated asynchronously.  A newly created
        # library may need a short time to receive the base_link ->
        # feeding_straw_tip_marker connection after the perception wait.  This
        # retry is preflight-only, recognizes only that transient read error,
        # and remains inside the caller's search deadline.
        while (
            not first_preflight.get("success")
            and str(first_preflight.get("reason", "")).startswith(
                "could not read the current straw-tip pose:"
            )
            and time.monotonic() < deadline
        ):
            self._spin_for(min(0.10, max(0.0, deadline - time.monotonic())))
            first_preflight = self._preflight_search_primitive(scan_pattern[0])
        if not first_preflight.get("success"):
            first_preflight["execute"] = execute
            first_preflight["selection"] = selection
            return first_preflight
        if not execute:
            return self._failure(
                "search_for_mouth",
                "mouth pose is not stable; dry-run search did not send motion",
                execute=False,
                selection=selection,
                initial_target_state=initial,
                next_safe_search_step=first_preflight,
            )

        steps: list[dict[str, Any]] = []
        for index in range(self.config.search_max_steps):
            # Do not begin a trajectory that cannot complete and be observed
            # within the advertised search deadline.  The small allowance
            # covers normal MoveIt/action overhead around the requested 2 s
            # fixed primitive duration.
            required_time = (
                self.config.search_step_duration_sec
                + self.config.search_observation_wait_sec
                + 0.50
            )
            if time.monotonic() + required_time > deadline:
                break
            primitive = scan_pattern[index % len(scan_pattern)]
            preflight = first_preflight if index == 0 else self._preflight_search_primitive(primitive)
            if not preflight.get("success"):
                return self._failure(
                    "search_for_mouth",
                    "safe mouth search stopped because the next primitive could not be planned",
                    execute=True,
                    selection=selection,
                    elapsed_sec=round(time.monotonic() - started, 4),
                    search_steps=steps,
                    failed_step=preflight,
                )
            moved = self._execute_search_primitive(preflight)
            steps.append(moved)
            if not moved.get("success"):
                return self._failure(
                    "search_for_mouth",
                    "safe mouth search stopped because a primitive did not complete",
                    execute=True,
                    selection=selection,
                    elapsed_sec=round(time.monotonic() - started, 4),
                    search_steps=steps,
                )
            remaining = timeout - (time.monotonic() - started)
            if remaining > 0.0:
                self._spin_for(min(self.config.search_observation_wait_sec, remaining))
            stable = self._stable_mouth_result()
            if stable.get("success"):
                return self._success(
                    "search_for_mouth",
                    execute=True,
                    selection=selection,
                    found_without_motion=False,
                    mouth_pose=stable["mouth_pose"],
                    active_target_label=stable.get("active_target_label"),
                    active_target_id=stable.get("active_target_id"),
                    elapsed_sec=round(time.monotonic() - started, 4),
                    search_steps=steps,
                )
        # The requested person was not recovered within the full bounded
        # search.  Honor the user's explicit fallback rule only when exactly
        # one currently visible candidate has itself passed the stable queue.
        single_fallback = self._active_targets.get_single_visible_stable_pose()
        if single_fallback.get("success"):
            resolved_selection = "center"
            self._active_targets.select_target(resolved_selection)
            self._search_fallback = {
                "requested_selection": selection,
                "resolved_selection": resolved_selection,
                "reason": "requested target was not found within the search deadline; exactly one stable mouth remained",
            }
            return self._success(
                "search_for_mouth",
                execute=True,
                selection=selection,
                requested_selection=selection,
                resolved_selection=resolved_selection,
                fallback_used=True,
                found_without_motion=False,
                mouth_pose=single_fallback["pose"],
                active_target_label=resolved_selection,
                active_target_id=resolved_selection,
                elapsed_sec=round(time.monotonic() - started, 4),
                search_steps=steps,
            )
        return self._failure(
            "search_for_mouth",
            "mouth search timed out before a stable active target was found",
            execute=True,
            selection=selection,
            elapsed_sec=round(time.monotonic() - started, 4),
            max_time_sec=timeout,
            max_steps=self.config.search_max_steps,
            search_steps=steps,
        )

    def active_search_mouth(
        self,
        max_search_time_sec: float = 30.0,
        target_selection: str = "center",
        *,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Run only the established bounded active-search policy.

        ``execute`` remains false unless the validated caller has explicit
        operator permission. The method accepts no scan direction, pose, or
        controller parameter.
        """
        result = self.search_for_mouth(
            max_time_sec=max_search_time_sec,
            selection=target_selection,
            execute=execute,
        )
        result = {
            **result,
            "tool": "active_search_mouth",
            "max_search_time_sec": max_search_time_sec,
            "target_selection": target_selection,
        }
        stage = "mouth_found" if result.get("success") else "mouth_search_failed"
        return self._remember_safe_tool_result(stage, result)

    def move_straw_tip_to_mouth_optional(
        self,
        mouth_pose: Mapping[str, Any] | None = None,
        *,
        execute: bool = False,
    ) -> dict[str, Any]:
        """Optional direct-mouth primitive, disabled by default and not used by the CLI."""
        if not self._allow_direct_mouth_motion:
            return self._failure(
                "move_straw_tip_to_mouth_optional",
                "direct mouth motion is disabled by this library instance",
                execute=execute,
                note="Use the pre-mouth tool in the first version. A future integrator must explicitly opt in.",
            )
        validated = self._validate_mouth_pose_input(mouth_pose)
        if not validated["success"]:
            return self._failure("move_straw_tip_to_mouth_optional", validated["reason"], execute=execute)
        mouth = validated["mouth_pose"]["position"]
        if not self._within_workspace(np.asarray(mouth, dtype=np.float64)):
            return self._failure("move_straw_tip_to_mouth_optional", "mouth pose is outside the allowed workspace", execute=execute)
        keepout_failure = self._ensure_human_keepout("move_straw_tip_to_mouth_optional")
        if keepout_failure is not None:
            return keepout_failure
        try:
            preflight = self._env.move_straw_tip_to_position(
                mouth, label="detected_mouth_pose", duration=self.config.duration_sec, plan_only=True,
                planning_mode=self.config.planning_mode,
            )
        except Exception as exc:
            return self._failure("move_straw_tip_to_mouth_optional", f"MoveIt preflight raised an exception: {exc}")
        if not preflight.get("success"):
            return self._failure(
                "move_straw_tip_to_mouth_optional", "MoveIt could not plan the direct mouth target", planner_result=preflight
            )
        if not execute:
            return self._success(
                "move_straw_tip_to_mouth_optional", execute=False, mouth_pose=validated["mouth_pose"],
                planner_result=preflight, note="Dry-run only; direct mouth contact remains an explicit opt-in."
            )
        result = self._env.move_straw_tip_to_position(
            mouth, label="detected_mouth_pose", duration=self.config.duration_sec, plan_only=False,
            planning_mode=self.config.planning_mode,
        )
        if not result.get("success"):
            return self._failure(
                "move_straw_tip_to_mouth_optional", "MoveIt did not complete the direct mouth motion", move_result=result
            )
        return self._success(
            "move_straw_tip_to_mouth_optional", execute=True, mouth_pose=validated["mouth_pose"], move_result=result
        )

    def _vertical_control_point(self) -> tuple[dict[str, Any] | None, str | None]:
        """Read the rigid tool pose and resolve the configured control point.

        The current scene has no cup-center reference offset, so the straw tip
        is the explicit first control point.  If a future config adds one of
        the recognized cup offsets, the exact same task-space operation uses
        it without exposing arbitrary tool poses to the caller.
        """
        try:
            tool_pose = self._env.get_robot_end_pose()
            tool_position = np.asarray(tool_pose["position"], dtype=np.float64)
            rotation = _rotation_matrix_from_quaternion(tool_pose["orientation_quat"])
            rpy = [float(value) for value in tool_pose["orientation_euler"]]
        except Exception as exc:
            return None, f"could not read the current tool0 pose: {exc}"
        if tool_position.shape != (3,) or not np.all(np.isfinite(tool_position)) or len(rpy) != 3:
            return None, "current tool0 pose is invalid"

        # Flange local +Z must point down in base_link.  Yaw is deliberately
        # unconstrained, allowing the current rigid cup/straw orientation to
        # be preserved rather than resetting the final wrist yaw.
        flange_down_alignment = float(np.dot(rotation[:, 2], np.array([0.0, 0.0, -1.0])))
        if flange_down_alignment < 0.995:
            return None, "current flange orientation is not sufficiently downward"

        feeding_cfg = self._env.cfg.get("feeding", {})
        control_point = "straw_tip"
        offset_value: Sequence[float] = self._env.flange_to_straw_tip
        for key in ("cup_center_offset", "flange_to_cup_center", "cup_reference_offset"):
            configured = feeding_cfg.get(key)
            if configured is not None:
                control_point = "cup_center"
                offset_value = configured
                break
        try:
            offset = np.asarray(offset_value, dtype=np.float64)
        except (TypeError, ValueError):
            return None, f"{control_point} offset is invalid"
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            return None, f"{control_point} offset must contain three finite values"

        return {
            "control_point": control_point,
            "offset_tool0": offset,
            "current_position": tool_position + rotation @ offset,
            "current_tool0_pose": tool_pose,
            "current_rpy": rpy,
            "flange_down_alignment": flange_down_alignment,
        }, None

    def adjust_cup_vertical(self, delta_z: float, *, execute: bool = False) -> dict[str, Any]:
        """Move the rigid cup/straw control point only along base-link Z.

        This is deliberately a task-space operation, not a final-wrist-joint
        adjustment.  It preserves the complete current tool orientation and
        asks the existing Cartesian MoveIt path planner to choose any required
        multi-joint UR10e motion.
        """
        try:
            delta = float(delta_z)
        except (TypeError, ValueError):
            return self._failure("adjust_cup_vertical", "delta_z must be a finite number", execute=execute)
        if not math.isfinite(delta):
            return self._failure("adjust_cup_vertical", "delta_z must be a finite number", execute=execute)
        if abs(delta) > self.config.max_abs_delta_z_per_call_m:
            return self._failure(
                "adjust_cup_vertical",
                "delta_z exceeds the maximum allowed per call",
                execute=execute,
                delta_z=delta,
                max_abs_delta_z_per_call_m=self.config.max_abs_delta_z_per_call_m,
            )

        control, control_error = self._vertical_control_point()
        if control is None:
            return self._failure("adjust_cup_vertical", control_error or "control point is unavailable", execute=execute)
        current = np.asarray(control["current_position"], dtype=np.float64)
        target = current.copy()
        target[2] += delta
        if not (
            self.config.allowed_control_point_z_min_m
            <= float(target[2])
            <= self.config.allowed_control_point_z_max_m
        ):
            return self._failure(
                "adjust_cup_vertical",
                "target z outside allowed range",
                execute=execute,
                control_point=control["control_point"],
                current_control_point=_xyz(current),
                target_control_point=_xyz(target),
                delta_z=delta,
                allowed_z_min=self.config.allowed_control_point_z_min_m,
                allowed_z_max=self.config.allowed_control_point_z_max_m,
            )

        # Convert the constrained control-point target into the matching tool0
        # target with the *current* flange-down orientation.  This preserves
        # cup/straw rigidity and locks control-point X/Y exactly by construction.
        offset = np.asarray(control["offset_tool0"], dtype=np.float64)
        label = f"{control['control_point']}_vertical_only"
        try:
            planner_target = self._env.plan_straw_tip_to_pose(
                target.tolist(),
                flange_to_straw_tip=offset.tolist(),
                flange_down_rpy=control["current_rpy"],
                label=label,
            )
        except Exception as exc:
            return self._failure("adjust_cup_vertical", f"could not compute tool0 target: {exc}", execute=execute)
        tool0_target = np.asarray(planner_target["tool0_target_position"], dtype=np.float64)
        tool0_radius = float(np.linalg.norm(tool0_target))
        if not self._within_workspace(tool0_target) or tool0_radius > self.config.max_tool_radius_m:
            return self._failure(
                "adjust_cup_vertical",
                "tool0 target is outside the conservative reach envelope",
                execute=execute,
                control_point=control["control_point"],
                current_control_point=_xyz(current),
                target_control_point=_xyz(target),
                tool0_target=_xyz(tool0_target),
                tool0_radius_m=round(tool0_radius, 6),
            )

        planning_scene_failure = self._ensure_planning_scene_for_motion(
            "adjust_cup_vertical",
            None,
            execute=execute,
        )
        if planning_scene_failure is not None:
            return planning_scene_failure
        keepout_failure = self._ensure_human_keepout("adjust_cup_vertical")
        if keepout_failure is not None:
            return keepout_failure
        common = {
            "control_point": control["control_point"],
            "execute": execute,
            "current_control_point": _xyz(current),
            "target_control_point": _xyz(target),
            "delta_z": delta,
            "xy_locked": True,
            "vertical_only_in_base_link": True,
            "keep_flange_down": True,
            "flange_down_alignment": round(float(control["flange_down_alignment"]), 6),
            "planner_target": planner_target,
            "keepout": self._keepout_result,
            "planning_scene": self._planning_scene_result,
        }
        try:
            preflight = self._env.move_straw_tip_to_position(
                target.tolist(),
                label=label,
                flange_to_straw_tip=offset.tolist(),
                flange_down_rpy=control["current_rpy"],
                duration=self.config.duration_sec,
                plan_only=True,
                # Cartesian planning is required so the entire control-point
                # path is a straight base-frame vertical translation.
                planning_mode="cartesian",
            )
        except Exception as exc:
            return self._failure(
                "adjust_cup_vertical", f"MoveIt vertical-motion preflight raised an exception: {exc}", **common
            )
        if not preflight.get("success"):
            return self._failure(
                "adjust_cup_vertical", "MoveIt could not plan the vertical-only control-point motion",
                planner_result=preflight, **common
            )
        if not execute:
            return self._success(
                "adjust_cup_vertical",
                planner_result=preflight,
                note="Dry-run only; pass execute=True to send this vertical-only Cartesian motion.",
                **common,
            )
        try:
            result = self._env.move_straw_tip_to_position(
                target.tolist(),
                label=label,
                flange_to_straw_tip=offset.tolist(),
                flange_down_rpy=control["current_rpy"],
                duration=self.config.duration_sec,
                plan_only=False,
                planning_mode="cartesian",
            )
        except Exception as exc:
            return self._failure(
                "adjust_cup_vertical", f"MoveIt vertical-motion execution raised an exception: {exc}",
                planner_result=preflight, **common
            )
        if not result.get("success"):
            return self._failure(
                "adjust_cup_vertical", "MoveIt did not complete the vertical-only control-point motion",
                planner_result=preflight, move_result=result, **common
            )
        return self._success(
            "adjust_cup_vertical",
            planner_result=preflight,
            move_result=result,
            note="The rigid cup/straw control point was commanded only along base_link Z.",
            **common,
        )

    def retreat_to_ready(self, *, execute: bool = False) -> dict[str, Any]:
        """Preflight, then optionally return the straw tip to the configured ready pose."""
        ready = np.asarray(self._env.ready_straw_tip_position, dtype=np.float64)
        if not self._within_workspace(ready):
            return self._failure("retreat_to_ready", "configured ready target is outside the allowed workspace")
        plan = self._env.plan_straw_tip_to_pose(ready.tolist(), label="ready_straw_tip_position")
        tool0_target = np.asarray(plan["tool0_target_position"], dtype=np.float64)
        if not self._within_workspace(tool0_target) or float(np.linalg.norm(tool0_target)) > self.config.max_tool_radius_m:
            return self._failure("retreat_to_ready", "ready tool0 target is outside the conservative reach envelope", planner_target=plan)
        planning_scene_failure = self._ensure_planning_scene_for_motion(
            "retreat_to_ready",
            None,
            execute=execute,
        )
        if planning_scene_failure is not None:
            return planning_scene_failure
        keepout_failure = self._ensure_human_keepout("retreat_to_ready")
        if keepout_failure is not None:
            return keepout_failure
        try:
            preflight = self._env.move_straw_tip_to_position(
                ready.tolist(), label="ready_straw_tip_position", duration=self.config.duration_sec,
                plan_only=True, planning_mode=self.config.planning_mode,
            )
        except Exception as exc:
            return self._failure("retreat_to_ready", f"MoveIt preflight raised an exception: {exc}")
        if not preflight.get("success"):
            return self._failure("retreat_to_ready", "MoveIt could not plan the ready retreat", planner_result=preflight)
        if not execute:
            return self._success(
                "retreat_to_ready", execute=False, ready_straw_tip_target=_xyz(ready), planner_result=preflight,
                planning_scene=self._planning_scene_result,
                note="Dry-run only; pass execute=True to send the ready retreat."
            )
        result = self._env.move_straw_tip_to_position(
            ready.tolist(), label="ready_straw_tip_position", duration=self.config.duration_sec,
            plan_only=False, planning_mode=self.config.planning_mode,
        )
        if not result.get("success"):
            return self._failure("retreat_to_ready", "MoveIt did not complete the ready retreat", move_result=result)
        return self._success(
            "retreat_to_ready", execute=True, ready_straw_tip_target=_xyz(ready), move_result=result,
            planning_scene=self._planning_scene_result,
        )

    def get_robot_observation(self, *, spin_timeout_sec: float = 0.10) -> dict[str, Any]:
        """Return compact state/camera metadata safe for an agent to inspect."""
        self._spin_for(spin_timeout_sec)
        latest_mouth = self._latest_sample_result("get_latest_mouth_pose")
        try:
            robot_state = self._env.get_robot_state()
            tool0_pose = self._env.get_robot_end_pose()
            straw_tip_pose = self._env.get_straw_tip_pose()
            camera_info = self._env.get_camera_info()
        except Exception as exc:
            return self._failure("get_robot_observation", f"could not read robot observation: {exc}")
        return self._success(
            "get_robot_observation",
            robot_state=robot_state,
            tool0_pose=tool0_pose,
            straw_tip_pose=straw_tip_pose,
            camera_info=camera_info,
            detected_mouth=latest_mouth,
            fixed_tool_geometry={
                "tool0_to_camera_optical_center_m": list(self._env.flange_to_camera_optical_center),
                "tool0_to_straw_tip_m": list(self._env.flange_to_straw_tip),
            },
            safety_config=asdict(self.config),
            direct_mouth_motion_enabled=self._allow_direct_mouth_motion,
        )

    def _octomap_status(self) -> dict[str, Any]:
        """Report whether the running MoveIt instance has the cloud updater.

        OctoMap remains an optional MoveIt layer.  The feeding methods do not
        need a separate motion path for it: when MoveIt has loaded
        ``wrist_rgbd_pointcloud``, the normal MoveIt preflight/execution calls
        already use that occupancy map.  This small read-only probe makes that
        fact explicit in the high-level tool result without changing MoveIt or
        relying only on the runner's environment variables.
        """
        requested_by_environment = os.environ.get("USE_OCTOMAP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        client = self._node.create_client(GetParameters, "/move_group/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=1.0):
                return {
                    "enabled": False,
                    "verified": False,
                    "requested_by_environment": requested_by_environment,
                    "reason": "MoveIt parameter service is unavailable; OctoMap status could not be verified",
                }
            request = GetParameters.Request()
            request.names = ["sensors", "octomap_resolution"]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=1.0)
            response = future.result()
            if response is None or len(response.values) < 2:
                return {
                    "enabled": False,
                    "verified": False,
                    "requested_by_environment": requested_by_environment,
                    "reason": "MoveIt returned no OctoMap parameter values",
                }
            sensors = list(response.values[0].string_array_value)
            resolution = float(response.values[1].double_value)
            return {
                "enabled": "wrist_rgbd_pointcloud" in sensors,
                "verified": True,
                "requested_by_environment": requested_by_environment,
                "sensors": sensors,
                "octomap_resolution_m": resolution,
            }
        except Exception as exc:
            return {
                "enabled": False,
                "verified": False,
                "requested_by_environment": requested_by_environment,
                "reason": f"could not query MoveIt OctoMap status: {exc}",
            }
        finally:
            self._node.destroy_client(client)

    def get_feeding_observation(self) -> dict[str, Any]:
        """Return the current read-only state needed by a feeding agent."""
        observation = self.get_robot_observation()
        target_state = self.get_active_target_state(spin_timeout_sec=0.05)
        octomap = self._octomap_status()
        stable = self._stable_mouth_result()
        detected = observation.get("detected_mouth", {}) if observation.get("success") else {}
        result = {
            "success": bool(observation.get("success")),
            "tool": "get_feeding_observation",
            "reason": observation.get("reason"),
            "robot_state": observation.get("robot_state"),
            "straw_tip_pose": observation.get("straw_tip_pose"),
            "tool0_pose": observation.get("tool0_pose"),
            "mouth_detected": bool(detected.get("success")),
            "detected_mouth_pose": detected.get("mouth_pose"),
            "mouth_stable": bool(stable.get("success")),
            "stable_mouth_pose": stable.get("mouth_pose"),
            "target_selection": target_state.get("active_target_label"),
            "target_state": target_state,
            "planning_scene": self._planning_scene_result,
            "planning_scene_enabled": self._use_planning_scene,
            "octomap": octomap,
            "feeding_stage": self._last_feeding_stage,
        }
        serialized = _jsonable(result)
        self._last_tool_results["get_feeding_observation"] = serialized
        return serialized

    def check_feeding_progress(self) -> dict[str, Any]:
        """Summarize the current rule-based state of the safe feeding flow."""
        latest = self.get_latest_mouth_pose(spin_timeout_sec=0.05)
        stable = self._stable_mouth_result()
        mouth_pose = stable.get("mouth_pose") if stable.get("success") else latest.get("mouth_pose")
        target_preview = self.compute_pre_mouth_target(mouth_pose) if isinstance(mouth_pose, Mapping) else None
        last_move = self._last_tool_results.get("move_straw_tip_to_pre_mouth")
        last_hold = self._last_tool_results.get("hold_pre_mouth")
        current_tip: Mapping[str, Any] | None
        try:
            candidate_tip = self._env.get_straw_tip_pose()
            current_tip = candidate_tip if isinstance(candidate_tip, Mapping) else None
        except Exception:
            current_tip = None

        pre_mouth_target = None
        if isinstance(last_move, Mapping):
            pre_mouth_target = last_move.get("pre_mouth_target")
        if pre_mouth_target is None and isinstance(target_preview, Mapping):
            pre_mouth_target = target_preview.get("pre_mouth_target")

        distance_to_pre_mouth: float | None = None
        if isinstance(current_tip, Mapping) and isinstance(pre_mouth_target, Sequence) and not isinstance(pre_mouth_target, (str, bytes)):
            try:
                current = np.asarray(current_tip.get("position"), dtype=np.float64)
                target = np.asarray(pre_mouth_target, dtype=np.float64)
                if current.shape == (3,) and target.shape == (3,) and np.all(np.isfinite(current)) and np.all(np.isfinite(target)):
                    distance_to_pre_mouth = round(float(np.linalg.norm(current - target)), 6)
            except (TypeError, ValueError):
                distance_to_pre_mouth = None

        planning_success = None if last_move is None else bool(last_move.get("success"))
        scene = self._planning_scene_result
        obstacle_blocked = None
        if isinstance(last_move, Mapping) and not last_move.get("success"):
            reason = str(last_move.get("reason") or "").lower()
            obstacle_blocked = any(token in reason for token in ("keepout", "collision", "planning", "obstacle"))
        elif isinstance(scene, Mapping) and scene.get("enabled") and not scene.get("applied", False):
            obstacle_blocked = True

        reached_pre_mouth = (
            bool(last_move and last_move.get("success") and last_move.get("execute"))
            and distance_to_pre_mouth is not None
            and distance_to_pre_mouth <= 0.02
        )
        holding = bool(last_hold and last_hold.get("success") and last_hold.get("holding"))
        failure = self._last_failure
        reason = None if failure is None else failure.get("reason")
        result = _jsonable(
            {
                "success": True,
                "tool": "check_feeding_progress",
                "reason": reason,
                "mouth_detected": bool(latest.get("success")),
                "mouth_stable": bool(stable.get("success")),
                "target_selected": self._active_targets.active_target_label,
                "pre_mouth_target_available": bool(target_preview and target_preview.get("success")),
                "pre_mouth_target": pre_mouth_target,
                "distance_to_pre_mouth": distance_to_pre_mouth,
                "planning_success": planning_success,
                "obstacle_blocked": obstacle_blocked,
                "reached_pre_mouth": reached_pre_mouth,
                "holding": holding,
                "failed_step": None if failure is None else failure.get("failed_step"),
                "feeding_stage": self._last_feeding_stage,
                "planning_scene": scene,
                "octomap": self._octomap_status(),
            }
        )
        self._last_tool_results["check_feeding_progress"] = result
        return result

    def hold_pre_mouth(self, duration_sec: float = 3.0) -> dict[str, Any]:
        """Hold only after a successful executed pre-mouth motion; never approach closer."""
        if isinstance(duration_sec, bool):
            return self._remember_safe_tool_result(
                "hold_failed", self._failure("hold_pre_mouth", "duration_sec must be finite and between 0.1 and 30.0")
            )
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            return self._remember_safe_tool_result(
                "hold_failed", self._failure("hold_pre_mouth", "duration_sec must be finite and between 0.1 and 30.0")
            )
        if not math.isfinite(duration) or not 0.1 <= duration <= 30.0:
            return self._remember_safe_tool_result(
                "hold_failed", self._failure("hold_pre_mouth", "duration_sec must be finite and between 0.1 and 30.0")
            )
        last_move = self._last_tool_results.get("move_straw_tip_to_pre_mouth")
        if not last_move or not last_move.get("success") or not last_move.get("execute"):
            return self._remember_safe_tool_result(
                "hold_failed",
                self._failure(
                    "hold_pre_mouth",
                    "a successful executed pre-mouth motion is required before holding",
                    last_pre_mouth_result=last_move,
                ),
            )
        settled = self.stop_motion_or_hold_position(settle_timeout_sec=min(3.0, duration))
        if not settled.get("success"):
            return self._remember_safe_tool_result(
                "hold_failed", self._failure("hold_pre_mouth", str(settled.get("reason")), settle=settled)
            )
        # This is a deliberate no-motion dwell. The SDK trajectory calls are
        # synchronous, and this method never publishes a controller command.
        time.sleep(duration)
        result = self._success(
            "hold_pre_mouth",
            holding=True,
            duration_sec=round(duration, 4),
            settle=settled,
            note="Held at the existing pre-mouth pose; no closer mouth motion was commanded.",
        )
        return self._remember_safe_tool_result("holding_pre_mouth", result)

    def feed_water(
        self,
        target_selection: str = "center",
        execute: bool = False,
        max_search_time_sec: float = 30.0,
        allow_direct_mouth_contact: bool = False,
        allow_vertical_adjust: bool = True,
    ) -> dict[str, Any]:
        """Run the fixed, safe MVP feeding sequence through pre-mouth hold.

        This is intentionally the only high-level operation a language-model
        plan needs.  It accepts intent-level target selection and bounded
        policy flags, never joints, poses, trajectories, controllers, gripper
        actions, attachment actions, or a mouth-contact command.

        ``max_search_time_sec`` is capped locally at 30 seconds as a defense
        in depth.  The LLM-plan validator rejects values above that limit, so
        a model cannot use the cap to request a longer search.
        """
        steps: list[dict[str, Any]] = []

        def record(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
            serialized = _jsonable(result)
            steps.append({"tool": name, "result": serialized})
            return serialized

        def failed(step: str, reason: str) -> dict[str, Any]:
            return {
                "success": False,
                "tool": "feed_water",
                "target_selection": target,
                "execute": execute,
                "steps": steps,
                "final_state": "failed",
                "failed_step": step,
                "reason": reason,
            }

        if not isinstance(target_selection, str):
            return failed("validate_arguments", "target_selection must be one of: left, center, right")
        target = target_selection.strip().lower()
        if target not in {"left", "center", "right"}:
            return failed("validate_arguments", "target_selection must be one of: left, center, right")
        if not isinstance(execute, bool):
            return failed("validate_arguments", "execute must be a boolean")
        if not isinstance(allow_direct_mouth_contact, bool):
            return failed("validate_arguments", "allow_direct_mouth_contact must be a boolean")
        if allow_direct_mouth_contact:
            return failed(
                "validate_arguments",
                "direct mouth contact is not supported by the current safety-reviewed MVP",
            )
        if not isinstance(allow_vertical_adjust, bool):
            return failed("validate_arguments", "allow_vertical_adjust must be a boolean")
        try:
            requested_search_time = float(max_search_time_sec)
        except (TypeError, ValueError):
            return failed("validate_arguments", "max_search_time_sec must be a finite number")
        if not math.isfinite(requested_search_time) or requested_search_time < 0.1:
            return failed("validate_arguments", "max_search_time_sec must be finite and at least 0.1")
        search_time = min(requested_search_time, self.config.search_max_time_sec)

        observation = record("get_feeding_observation", self.get_feeding_observation())
        if not observation.get("success"):
            return failed("get_feeding_observation", str(observation.get("reason") or "robot observation failed"))

        # This read-only sample provides the explicit safe-tool sequence with
        # the latest existing MediaPipe result before bounded active search.
        record("detect_mouth", self.detect_mouth())

        selected = record("select_target", self.select_target(target))
        if not selected.get("success"):
            return failed("select_target", str(selected.get("reason") or "target selection failed"))

        search = record(
            "active_search_mouth",
            self.active_search_mouth(
                max_search_time_sec=search_time,
                target_selection=target,
                execute=execute,
            ),
        )
        if not search.get("success"):
            return failed("active_search_mouth", str(search.get("reason") or "active mouth search failed"))

        stable = record(
            "wait_for_stable_mouth_pose",
            self.wait_for_stable_mouth_pose(timeout_sec=5.0, selection=target),
        )
        if not stable.get("success"):
            return failed(
                "wait_for_stable_mouth_pose",
                str(stable.get("reason") or "stable selected mouth pose was unavailable"),
            )
        mouth_pose = stable.get("mouth_pose")
        if not isinstance(mouth_pose, Mapping):
            return failed("wait_for_stable_mouth_pose", "stable mouth result did not include a pose")

        scene = record(
            "ensure_planning_scene_obstacles",
            self.ensure_planning_scene_obstacles(mouth_pose, execute=execute),
        )
        if not scene.get("success") or not scene.get("applied"):
            return failed(
                "ensure_planning_scene_obstacles",
                str(scene.get("reason") or scene.get("warning") or "PlanningScene safety objects were not verified"),
            )

        # MoveIt automatically consumes this optional layer during the
        # subsequent preflight/motion.  OctoMap absence is not a failure: the
        # deterministic human PlanningScene objects remain mandatory above.
        octomap = self._octomap_status()
        record("check_octomap_status", {"success": True, **octomap})

        preview = record("compute_pre_mouth_target", self.compute_pre_mouth_target(mouth_pose))
        if not preview.get("success"):
            return failed("compute_pre_mouth_target", str(preview.get("reason") or "pre-mouth target failed"))

        move = record(
            "move_straw_tip_to_pre_mouth",
            self.move_straw_tip_to_pre_mouth(mouth_pose, execute=execute),
        )
        if not move.get("success"):
            return failed(
                "move_straw_tip_to_pre_mouth",
                str(move.get("reason") or "MoveIt could not reach a safe pre-mouth target"),
            )

        # The current intent-level API intentionally provides no height
        # offset.  A zero/guessed adjustment would add a second motion with no
        # feeding benefit, so retain the existing bounded vertical primitive
        # for a future reviewed high-level policy rather than invoke it here.
        steps.append(
            {
                "tool": "adjust_cup_vertical",
                "result": {
                    "success": True,
                    "skipped": True,
                    "enabled": allow_vertical_adjust,
                    "reason": "No reviewed vertical adjustment request was provided; the pre-mouth pose is held.",
                },
            }
        )

        hold = record(
            "hold_pre_mouth" if execute else "stop_motion_or_hold_position",
            self.hold_pre_mouth(duration_sec=3.0) if execute else self.stop_motion_or_hold_position(),
        )
        if not hold.get("success"):
            return failed(str(hold.get("tool") or "hold_pre_mouth"), str(hold.get("reason") or "robot did not settle"))

        # Do not retreat after a feeding request: the requested terminal state
        # is a safe pre-mouth hold.  A separately reviewed caller may invoke
        # the existing retreat_to_ready tool after this high-level operation.
        steps.append(
            {
                "tool": "retreat_to_ready",
                "result": {
                    "success": True,
                    "skipped": True,
                    "reason": "feed_water terminates at the safe pre-mouth hold; no retreat was requested.",
                },
            }
        )
        return {
            "success": True,
            "tool": "feed_water",
            "target_selection": target,
            "execute": execute,
            "max_search_time_sec": search_time,
            "steps": steps,
            "final_state": "holding_pre_mouth" if execute else "pre_mouth_plan_validated",
            "failed_step": None,
            "reason": None,
        }

    def stop_motion_or_hold_position(
        self,
        *,
        stationary_velocity_rad_s: float = 0.01,
        settle_timeout_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Wait briefly for a completed trajectory to settle, then report hold.

        This deliberately does not publish controller commands or arbitrary
        poses.  SDK motions are synchronous and the library exposes no
        background trajectory handle.  A real emergency stop remains outside
        the future LLM tool surface.
        """
        threshold = max(0.0, float(stationary_velocity_rad_s))
        start = time.monotonic()
        deadline = start + max(0.0, float(settle_timeout_sec))
        max_velocity = 0.0
        while True:
            state = self._env.get_robot_state()
            velocities = np.asarray(state.get("joint_velocities", []), dtype=np.float64)
            max_velocity = float(np.max(np.abs(velocities))) if velocities.size else 0.0
            if max_velocity <= threshold:
                return self._success(
                    "stop_motion_or_hold_position",
                    holding=True,
                    max_joint_velocity_rad_s=round(max_velocity, 6),
                    settle_wait_sec=round(time.monotonic() - start, 4),
                    note="No asynchronous motion is active from this synchronous tool library.",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.05, remaining))
        return self._failure(
            "stop_motion_or_hold_position",
            "robot is still moving after the post-trajectory settle timeout; this safe library does not issue direct controller stop commands",
            max_joint_velocity_rad_s=round(max_velocity, 6),
            stationary_velocity_limit_rad_s=round(threshold, 6),
            settle_timeout_sec=round(max(0.0, float(settle_timeout_sec)), 4),
            emergency_stop_note="Use the external simulator/controller safety stop if an immediate stop is required.",
        )
