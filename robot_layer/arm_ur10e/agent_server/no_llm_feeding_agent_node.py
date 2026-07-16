#!/usr/bin/env python3
"""Conservative no-LLM bridge from a detected mouth pose to pre-mouth motion.

This node deliberately has a narrow scope.  It consumes a perception result,
selects a stable point in ``base_link``, offsets it to a pre-mouth straw-tip
target, and (only with ``--execute``) calls the existing UR10e SDK MoveIt
primitive.  It never calls the SDK's ``move_straw_tip_to_mouth`` method.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_sdk():
    """Load the project SDK without requiring the project to be installed."""
    module_path = _project_root() / "robot_layer" / "arm_ur10e" / "agent_server" / "robot_sdk" / "ur10e_sdk.py"
    spec = importlib.util.spec_from_file_location("ur10e_sdk_for_no_llm_agent", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load UR10e SDK from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.UR10eRobotEnv


def _rotation_matrix(transform: TransformStamped) -> np.ndarray:
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


def _parse_xyz(value: str) -> tuple[float, float, float]:
    try:
        output = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers: x,y,z") from exc
    if len(output) != 3 or not all(math.isfinite(item) for item in output):
        raise argparse.ArgumentTypeError("expected three finite numbers: x,y,z")
    return output


@dataclass(frozen=True)
class AgentOptions:
    mouth_topic: str
    selected_mouth_topic: str
    pre_mouth_topic: str
    status_topic: str
    base_frame: str
    execute: bool
    pre_mouth_offset: tuple[float, float, float]
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_tool_radius_m: float
    stable_samples: int
    stability_window_sec: float
    max_pose_spread_m: float
    max_pose_age_sec: float
    wait_timeout_sec: float
    duration_sec: float
    planning_mode: str


@dataclass(frozen=True)
class PoseSample:
    position: np.ndarray
    stamp_sec: float
    received_monotonic: float


class NoLLMFeedingAgent(Node):
    """One safe no-LLM perception-to-pre-mouth decision loop."""

    def __init__(self, options: AgentOptions) -> None:
        super().__init__("no_llm_feeding_agent")
        self._options = options
        if any(low >= high for low, high in zip(options.workspace_min, options.workspace_max)):
            raise ValueError("Every workspace minimum must be lower than its maximum")
        if options.stable_samples < 1:
            raise ValueError("--stable-samples must be at least one")
        if options.stability_window_sec <= 0.0 or options.max_pose_age_sec <= 0.0:
            raise ValueError("Stability window and pose age must be positive")
        if options.max_tool_radius_m <= 0.0:
            raise ValueError("--max-tool-radius-m must be positive")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._selected_mouth_pub = self.create_publisher(PoseStamped, options.selected_mouth_topic, qos)
        self._pre_mouth_pub = self.create_publisher(PoseStamped, options.pre_mouth_topic, qos)
        self._status_pub = self.create_publisher(String, options.status_topic, qos)
        self._mouth_sub = self.create_subscription(PoseStamped, options.mouth_topic, self._mouth_callback, qos)
        self._tf_buffer = Buffer()
        # The main loop below spins this node, so keep TF on that same executor.
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self._samples: deque[PoseSample] = deque(maxlen=max(20, options.stable_samples * 4))
        self._started_at = time.monotonic()
        self._last_reason = "waiting_for_mouth_pose"
        self._done = False
        self.exit_code = 0
        self._timer = self.create_timer(0.10, self._decision_loop)

        # The SDK owns a separate ROS node for MoveIt/action clients.  It is
        # initialized after this agent's subscription so no motion can happen
        # before the input validation gate below succeeds.
        robot_env_type = _load_sdk()
        self._env = robot_env_type(init_ros_node=False)
        try:
            keepout = self._env.apply_human_keepout()
        except Exception as exc:
            raise RuntimeError(
                "Could not install the human collision keepout in MoveIt; refusing to enable feeding motion"
            ) from exc
        if not keepout.get("success"):
            raise RuntimeError("MoveIt did not accept the human collision keepout")
        self._status("human_keepout_applied", **keepout)
        self._status("started", execute=options.execute, motion_permitted=options.execute)
        self.get_logger().info(
            "No-LLM feeding agent: %s -> pre-mouth offset %s; execute=%s"
            % (options.mouth_topic, list(options.pre_mouth_offset), options.execute)
        )

    def destroy_node(self) -> bool:
        if hasattr(self, "_env"):
            self._env.close()
        return super().destroy_node()

    def _status(self, state: str, **fields: object) -> None:
        payload = {"state": state, "execute": self._options.execute, **fields}
        message = String()
        message.data = json.dumps(payload, sort_keys=True, default=float)
        self._status_pub.publish(message)
        self.get_logger().info(message.data)

    def _finish(self, state: str, *, failure: bool = False, **fields: object) -> None:
        if self._done:
            return
        self._done = True
        self.exit_code = 2 if failure else 0
        self._timer.cancel()
        self._status(state, **fields)

    def _warn_state(self, state: str, **fields: object) -> None:
        if self._last_reason != state:
            self._last_reason = state
            self._status(state, **fields)

    def _pose_in_base(self, message: PoseStamped) -> np.ndarray | None:
        source_frame = message.header.frame_id.strip().lstrip("/")
        if not source_frame:
            self._warn_state("rejected_missing_frame")
            return None
        raw = np.array(
            [message.pose.position.x, message.pose.position.y, message.pose.position.z], dtype=np.float64
        )
        if not np.all(np.isfinite(raw)):
            self._warn_state("rejected_nonfinite_mouth_pose")
            return None
        if source_frame == self._options.base_frame.lstrip("/"):
            return raw
        try:
            transform = self._tf_buffer.lookup_transform(
                self._options.base_frame,
                source_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.20),
            )
        except Exception as exc:
            self._warn_state("rejected_tf_unavailable", source_frame=source_frame, detail=str(exc))
            return None
        translation = transform.transform.translation
        return _rotation_matrix(transform) @ raw + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )

    def _mouth_callback(self, message: PoseStamped) -> None:
        if self._done:
            return
        position = self._pose_in_base(message)
        if position is None:
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        self._samples.append(PoseSample(position, stamp, time.monotonic()))

    def _stable_mouth_position(self) -> np.ndarray | None:
        now = time.monotonic()
        fresh = [sample for sample in self._samples if now - sample.received_monotonic <= self._options.max_pose_age_sec]
        self._samples = deque(fresh, maxlen=self._samples.maxlen)
        if len(fresh) < self._options.stable_samples:
            self._warn_state("waiting_for_stable_mouth", received_samples=len(fresh))
            return None
        recent = fresh[-self._options.stable_samples :]
        observed_window = recent[-1].received_monotonic - recent[0].received_monotonic
        if observed_window > self._options.stability_window_sec:
            self._warn_state("rejected_mouth_pose_stale", observed_window_sec=round(observed_window, 3))
            return None
        positions = np.asarray([sample.position for sample in recent], dtype=np.float64)
        selected = np.median(positions, axis=0)
        max_spread = float(np.max(np.linalg.norm(positions - selected, axis=1)))
        if max_spread > self._options.max_pose_spread_m:
            self._warn_state(
                "rejected_mouth_pose_unstable",
                max_spread_m=round(max_spread, 4),
                limit_m=self._options.max_pose_spread_m,
            )
            return None
        return selected

    def _within_workspace(self, position: np.ndarray) -> bool:
        lower = np.asarray(self._options.workspace_min, dtype=np.float64)
        upper = np.asarray(self._options.workspace_max, dtype=np.float64)
        return bool(np.all(position >= lower) and np.all(position <= upper))

    def _pose_message(self, position: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._options.base_frame
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.w = 1.0
        return pose

    @staticmethod
    def _xyz(position: np.ndarray | Sequence[float]) -> list[float]:
        return [round(float(value), 6) for value in position]

    def _decision_loop(self) -> None:
        if self._done:
            return
        if time.monotonic() - self._started_at > self._options.wait_timeout_sec:
            self._finish(
                "stopped_no_stable_mouth_pose",
                failure=True,
                timeout_sec=self._options.wait_timeout_sec,
                last_reason=self._last_reason,
            )
            return
        mouth = self._stable_mouth_position()
        if mouth is None:
            return
        pre_mouth = mouth + np.asarray(self._options.pre_mouth_offset, dtype=np.float64)
        if not self._within_workspace(mouth):
            self._finish(
                "stopped_mouth_outside_workspace",
                failure=True,
                mouth_position=self._xyz(mouth),
                workspace_min=list(self._options.workspace_min),
                workspace_max=list(self._options.workspace_max),
            )
            return
        if not self._within_workspace(pre_mouth):
            self._finish(
                "stopped_pre_mouth_outside_workspace",
                failure=True,
                pre_mouth_position=self._xyz(pre_mouth),
                workspace_min=list(self._options.workspace_min),
                workspace_max=list(self._options.workspace_max),
            )
            return

        # Use the SDK planner to obtain the actual tool0 target before any
        # command.  The straw tip, mouth, and target are all base-link points.
        plan = self._env.plan_straw_tip_to_pre_mouth_pose(pre_mouth.tolist())
        tool0_target = np.asarray(plan["tool0_target_position"], dtype=np.float64)
        if not self._within_workspace(tool0_target):
            self._finish(
                "stopped_tool0_target_outside_workspace",
                failure=True,
                tool0_target_position=self._xyz(tool0_target),
            )
            return
        tool0_radius = float(np.linalg.norm(tool0_target))
        if tool0_radius > self._options.max_tool_radius_m:
            self._finish(
                "stopped_tool0_target_outside_workspace",
                failure=True,
                tool0_target_position=self._xyz(tool0_target),
                tool0_radius_m=round(tool0_radius, 4),
                max_tool_radius_m=self._options.max_tool_radius_m,
                note="The target exceeds the conservative UR10e reach envelope; no MoveIt request was sent.",
            )
            return

        self._selected_mouth_pub.publish(self._pose_message(mouth))
        self._pre_mouth_pub.publish(self._pose_message(pre_mouth))
        self._status(
            "selected_pre_mouth_target",
            mouth_position=self._xyz(mouth),
            pre_mouth_position=self._xyz(pre_mouth),
            pre_mouth_offset=list(self._options.pre_mouth_offset),
            tool0_target_position=self._xyz(tool0_target),
            keep_flange_down=True,
        )
        if not self._options.execute:
            self._finish(
                "dry_run_complete",
                motion_commanded=False,
                planned_straw_tip_position=self._xyz(pre_mouth),
                note="Pass --execute to command exactly one pre-mouth MoveIt motion.",
            )
            return

        # Ask the SDK / MoveIt to plan first.  A plan-only request cannot send
        # a controller trajectory, so a kinematically unreachable target stops
        # here instead of reaching the execution branch below.
        try:
            preflight = self._env.move_straw_tip_to_pre_mouth(
                pre_mouth_safe_position=pre_mouth.tolist(),
                duration=self._options.duration_sec,
                plan_only=True,
                planning_mode=self._options.planning_mode,
            )
        except Exception as exc:
            self._finish("stopped_moveit_preflight_exception", failure=True, detail=str(exc))
            return
        if not preflight.get("success"):
            self._finish(
                "stopped_pre_mouth_unreachable",
                failure=True,
                preflight_result=preflight,
                note="No trajectory was sent because MoveIt could not plan this pre-mouth target.",
            )
            return

        # This is intentionally the only motion command in this node.  There
        # is no mouth-approach primitive here.
        try:
            result = self._env.move_straw_tip_to_pre_mouth(
                pre_mouth_safe_position=pre_mouth.tolist(),
                duration=self._options.duration_sec,
                plan_only=False,
                planning_mode=self._options.planning_mode,
            )
        except Exception as exc:
            self._finish("stopped_moveit_exception", failure=True, detail=str(exc))
            return
        if not result.get("success"):
            self._finish("stopped_moveit_failed", failure=True, move_result=result)
            return
        self._finish(
            "pre_mouth_motion_complete",
            motion_commanded=True,
            planned_straw_tip_position=self._xyz(pre_mouth),
            move_result=result,
            note="Stopped at pre-mouth; this node never moves directly to the mouth.",
        )


def _parse_options(argv: Sequence[str] | None = None) -> tuple[AgentOptions, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouth-topic", default="/detected_mouth_pose")
    parser.add_argument("--selected-mouth-topic", default="/feeding_agent/selected_mouth_pose")
    parser.add_argument("--pre-mouth-topic", default="/feeding_agent/pre_mouth_target")
    parser.add_argument("--status-topic", default="/feeding_agent/status")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--execute", action="store_true", help="Permit exactly one MoveIt pre-mouth motion.")
    parser.add_argument(
        "--pre-mouth-offset",
        type=_parse_xyz,
        default=(0.0, -0.08, 0.0),
        help="Safety offset from mouth to straw-tip pre-mouth target, in base_link metres.",
    )
    # The elevated base moves the valid pre-mouth approach into the positive
    # near-Y region of base_link. The radial tool envelope remains the hard
    # reach guard; these bounds are the conservative Cartesian envelope.
    parser.add_argument("--workspace-min", type=_parse_xyz, default=(0.05, 0.15, 0.35))
    parser.add_argument("--workspace-max", type=_parse_xyz, default=(0.85, 1.25, 1.95))
    parser.add_argument(
        "--max-tool-radius-m",
        type=float,
        default=1.30,
        help="Conservative maximum base_link distance for the planned tool0 target.",
    )
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--stability-window-sec", type=float, default=1.0)
    parser.add_argument("--max-pose-spread-m", type=float, default=0.025)
    parser.add_argument("--max-pose-age-sec", type=float, default=1.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=20.0)
    parser.add_argument("--duration-sec", type=float, default=7.0)
    parser.add_argument(
        "--planning-mode",
        choices=("cartesian", "move_group", "ik_joint_target"),
        default="cartesian",
        help="Existing SDK planning mode; Cartesian is preferred for the safe approach segment.",
    )
    parsed, ros_args = parser.parse_known_args(argv)
    options = AgentOptions(
        mouth_topic=parsed.mouth_topic,
        selected_mouth_topic=parsed.selected_mouth_topic,
        pre_mouth_topic=parsed.pre_mouth_topic,
        status_topic=parsed.status_topic,
        base_frame=parsed.base_frame,
        execute=parsed.execute,
        pre_mouth_offset=parsed.pre_mouth_offset,
        workspace_min=parsed.workspace_min,
        workspace_max=parsed.workspace_max,
        max_tool_radius_m=parsed.max_tool_radius_m,
        stable_samples=parsed.stable_samples,
        stability_window_sec=parsed.stability_window_sec,
        max_pose_spread_m=parsed.max_pose_spread_m,
        max_pose_age_sec=parsed.max_pose_age_sec,
        wait_timeout_sec=parsed.wait_timeout_sec,
        duration_sec=parsed.duration_sec,
        planning_mode=parsed.planning_mode,
    )
    return options, ros_args


def main(argv: Sequence[str] | None = None) -> int:
    options, ros_args = _parse_options(argv)
    rclpy.init(args=ros_args)
    node: NoLLMFeedingAgent | None = None
    executor = SingleThreadedExecutor()
    try:
        node = NoLLMFeedingAgent(options)
        # Keep this node off rclpy's global executor.  The existing SDK uses
        # spin_until_future_complete() for its independent MoveIt client node;
        # separating executors prevents the SDK from attempting to re-enter a
        # spinning default executor during the one allowed motion command.
        executor.add_node(node)
        while rclpy.ok() and not node._done:
            executor.spin_once(timeout_sec=0.25)
        return node.exit_code
    except KeyboardInterrupt:
        return 130
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
