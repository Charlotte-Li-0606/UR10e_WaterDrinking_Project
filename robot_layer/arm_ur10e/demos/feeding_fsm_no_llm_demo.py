#!/usr/bin/env python3
"""No-LLM UR10e feeding demo using Gazebo visualization and MoveIt motion.

This demo has one fixed task:

    put the flange-mounted straw tip at the pre-mouth safe target

The Gazebo scene is created by `gazebo_feeding_scene.py` on the ThinkPad, where
Gazebo is running. This script runs the robot motion through the UR10e SDK and
MoveIt. The visible feeding motion starts only after the robot is placed in a
ready pose with the flange pointing down.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_ros_sdk():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robot_layer").is_dir():
            sdk_path = parent / "robot_layer" / "arm_ur10e" / "agent_server" / "robot_sdk" / "ur10e_sdk.py"
            spec = importlib.util.spec_from_file_location("ur10e_sdk", sdk_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load UR10e SDK from {sdk_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return (
                module.UR10eRobotEnv,
                module.rotation_matrix_from_quaternion,
                module.quaternion_from_euler,
            )
    raise RuntimeError("Could not locate ABot-Claw repository root")


UR10eRobotEnv, rotation_matrix_from_quaternion, quaternion_from_euler = _load_ros_sdk()


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _find_project_root()
GAZEBO_SCENE_HELPER = PROJECT_ROOT / "scripts" / "gazebo_feeding_scene.py"
DEFAULT_SDK_CONFIG = PROJECT_ROOT / "config" / "ur10e_sdk_config.yaml"

VERIFIED_READY_DOWN_JOINTS = [
    4.85517249210659,
    -1.2604356804929226,
    -1.5191366042796963,
    -5.074409531604118,
    4.712389271978613,
    -2.998809111570852,
]


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _parse_xyz(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(out) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return out


def _estimate_tool_point_position(env, tool_offset: list[float]) -> list[float]:
    pose = env.get_robot_end_pose()
    position = np.asarray(pose["position"], dtype=float)
    qx, qy, qz, qw = [float(x) for x in pose["orientation_quat"]]
    rotation = rotation_matrix_from_quaternion(qx, qy, qz, qw)
    offset = np.asarray(tool_offset, dtype=float)
    return (position + rotation @ offset).tolist()


def _estimate_straw_tip_position(env) -> list[float]:
    return _estimate_tool_point_position(env, env.flange_to_straw_tip)


def _desired_flange_down_quat(env) -> list[float]:
    roll, pitch, yaw = [float(x) for x in env.flange_down_rpy]
    return list(quaternion_from_euler(roll, pitch, yaw))


def _orientation_dot(a: list[float], b: list[float]) -> float:
    return abs(sum(float(x) * float(y) for x, y in zip(a, b)))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _format_xyz(values: list[float]) -> str:
    return ",".join(f"{float(v):.6f}" for v in values)


class FeedingNoLLMFSM:
    def __init__(
        self,
        env,
        *,
        execute: bool,
        duration: float,
        hold_ready_seconds: float,
        wait_seconds: float,
        pre_mouth_safe_position: list[float] | None,
        ready_straw_tip_position: list[float] | None,
        retreat: bool,
        prepare_ready: bool,
        gazebo_tracker: bool,
        tracker_rate: float,
        tracker_use_existing: bool,
        stop_tracker_on_exit: bool,
    ):
        self.env = env
        self.execute = bool(execute)
        self.duration = float(duration)
        self.hold_ready_seconds = float(hold_ready_seconds)
        self.wait_seconds = float(wait_seconds)
        self.pre_mouth_safe_position = pre_mouth_safe_position or list(env.pre_mouth_safe_position)
        self.ready_straw_tip_position = ready_straw_tip_position or list(env.ready_straw_tip_position)
        self.retreat_enabled = bool(retreat)
        self.prepare_ready_enabled = bool(prepare_ready)
        self.gazebo_tracker_enabled = bool(gazebo_tracker)
        self.tracker_rate = float(tracker_rate)
        self.tracker_use_existing = bool(tracker_use_existing)
        self.stop_tracker_on_exit = bool(stop_tracker_on_exit)
        self.gazebo_tracker_process: subprocess.Popen | None = None
        self.gazebo_tracker_log = PROJECT_ROOT / "logs" / "feeding_fsm_gazebo_tracker.log"
        self.events: list[dict[str, Any]] = []
        self.start_joints: list[float] = []

    def _record(self, state: str, **data: Any) -> None:
        item = {"state": state, "time": round(time.time(), 3)}
        item.update(data)
        self.events.append(item)
        print(json.dumps(item, sort_keys=True))

    def describe_gazebo_scene_targets(self) -> dict:
        pre = list(self.pre_mouth_safe_position)
        mouth = list(self.env.mouth_target_position)
        ready = list(self.ready_straw_tip_position)
        pre_plan = self.env.plan_straw_tip_to_pose(pre, label="pre_mouth_safe_target")
        ready_plan = self.env.plan_straw_tip_to_pose(ready, label="ready_straw_tip")
        qx, qy, qz, qw = pre_plan["tool0_target_orientation_quat"]
        rotation = rotation_matrix_from_quaternion(qx, qy, qz, qw)
        pre_tool = np.asarray(pre_plan["tool0_target_position"], dtype=float)
        camera_at_pre = (
            pre_tool + rotation @ np.asarray(self.env.flange_to_camera_optical_center, dtype=float)
        ).tolist()
        info = {
            "world": "empty",
            "gazebo_helper": str(GAZEBO_SCENE_HELPER.relative_to(PROJECT_ROOT)),
            "ready_straw_tip_position": ready,
            "pre_mouth_safe_position": pre,
            "mouth_target_position": mouth,
            "tool0_target_at_pre_mouth": pre_plan["tool0_target_position"],
            "camera_optical_center_at_pre_mouth": camera_at_pre,
            "flange_to_straw_tip": list(self.env.flange_to_straw_tip),
            "flange_to_camera_optical_center": list(self.env.flange_to_camera_optical_center),
            "flange_down_rpy": list(self.env.flange_down_rpy),
            "ready_plan": ready_plan,
            "pre_plan": pre_plan,
        }
        self._record("GAZEBO_SCENE_TARGETS", **info)
        return info

    def check_robot_state(self, state_name: str = "CHECK_ROBOT_STATE") -> dict:
        state = self.env.get_robot_state()
        pose = self.env.get_robot_end_pose()
        desired_quat = _desired_flange_down_quat(self.env)
        self.start_joints = [float(x) for x in state["joint_positions"]]
        current_straw = _estimate_straw_tip_position(self.env)
        current_camera = _estimate_tool_point_position(self.env, self.env.flange_to_camera_optical_center)
        flange_down_dot = _orientation_dot(pose["orientation_quat"], desired_quat)
        self._record(
            state_name,
            joint_positions=self.start_joints,
            tool0_position=pose["position"],
            tool0_orientation_quat=pose["orientation_quat"],
            desired_flange_down_quat=desired_quat,
            flange_down_orientation_dot=flange_down_dot,
            estimated_straw_tip_position=current_straw,
            estimated_camera_optical_center_position=current_camera,
        )
        return {
            "state": state,
            "pose": pose,
            "estimated_straw_tip_position": current_straw,
            "estimated_camera_optical_center_position": current_camera,
            "flange_down_orientation_dot": flange_down_dot,
        }

    def prepare_ready_down(self) -> dict:
        self._record(
            "PREPARE_READY_DOWN",
            execute=self.execute and self.prepare_ready_enabled,
            ready_straw_tip_position=self.ready_straw_tip_position,
            keep_flange_down=True,
            locked_flange_down_rpy=list(self.env.flange_down_rpy),
            planner="MoveIt IK joint target",
        )
        if not self.prepare_ready_enabled:
            return {"skipped": True, "reason": "prepare-ready disabled"}
        result = self.env.move_straw_tip_to_position(
            self.ready_straw_tip_position,
            label="ready_straw_tip_position",
            flange_down_rpy=self.env.flange_down_rpy,
            duration=self.duration,
            plan_only=not self.execute,
            planning_mode="ik_joint_target",
        )
        self._record(
            "READY_DOWN_RESULT",
            success=bool(result.get("success")),
            stage=result.get("stage"),
            planner=result.get("planner"),
            move_result=result.get("move_result"),
        )
        if not result.get("success"):
            self._record(
                "READY_DOWN_FALLBACK_JOINT_TARGET",
                reason="MoveIt IK ready-down failed; using verified flange-down start joints.",
                joint_positions=VERIFIED_READY_DOWN_JOINTS,
            )
            fallback = self.env.move_joints(VERIFIED_READY_DOWN_JOINTS, duration=self.duration)
            self._record(
                "READY_DOWN_FALLBACK_RESULT",
                success=bool(fallback.get("success")),
                move_result=fallback,
            )
            if not fallback.get("success"):
                raise RuntimeError(
                    f"Ready-down preparation failed. IK result: {result}; fallback result: {fallback}"
                )
            result = {
                "success": True,
                "stage": "verified_ready_down_joint_fallback",
                "planner": "joint_trajectory_ready_setup",
                "joint_positions": VERIFIED_READY_DOWN_JOINTS,
                "move_result": fallback,
                "ik_attempt": result,
            }
        return result

    def hold_ready_down(self) -> None:
        ready_info = self.check_robot_state("START_FEEDING_DEMO_READY_DOWN")
        if ready_info["flange_down_orientation_dot"] < 0.995:
            raise RuntimeError(
                "Ready pose is not flange-down enough. Refusing to start feeding motion."
            )
        if self.hold_ready_seconds > 0.0:
            time.sleep(self.hold_ready_seconds)

    def check_pre_mouth_target(self, robot_info: dict) -> dict:
        plan = self.env.plan_straw_tip_to_pre_mouth_pose(
            pre_mouth_safe_position=self.pre_mouth_safe_position,
        )
        current_straw = robot_info["estimated_straw_tip_position"]
        target = plan["pre_mouth_safe_position"]
        distance = _distance(current_straw, target)
        self._record(
            "CHECK_PRE_MOUTH_TARGET",
            pre_mouth_safe_position=target,
            tool0_target_position=plan["tool0_target_position"],
            tool0_target_orientation_quat=plan["tool0_target_orientation_quat"],
            straw_to_pre_mouth_distance_m=distance,
            keep_flange_down=True,
            locked_flange_down_rpy=list(self.env.flange_down_rpy),
        )
        return {"plan": plan, "start_distance_m": distance}

    def move_straw_to_pre_mouth(self) -> dict:
        self._record(
            "MOVE_STRAW_TO_PRE_MOUTH",
            execute=self.execute,
            planner="MoveIt Cartesian planner",
            keep_flange_down=True,
            locked_flange_down_rpy=list(self.env.flange_down_rpy),
        )
        result = self.env.move_straw_tip_to_position(
            self.pre_mouth_safe_position,
            label="pre_mouth_safe_position",
            flange_down_rpy=self.env.flange_down_rpy,
            duration=self.duration,
            plan_only=not self.execute,
            planning_mode="cartesian",
        )
        result["plan"]["pre_mouth_safe_position"] = result["plan"]["target_straw_tip_position"]
        self._record(
            "MOVE_RESULT",
            success=bool(result.get("success")),
            stage=result.get("stage"),
            planner=result.get("planner"),
            move_result=result.get("move_result"),
        )
        if not result.get("success"):
            raise RuntimeError(f"MoveIt pre-mouth motion failed: {result}")
        return result

    def wait_at_pre_mouth(self) -> None:
        self._record("WAIT_AT_PRE_MOUTH", seconds=self.wait_seconds)
        time.sleep(self.wait_seconds)

    def retreat(self) -> dict:
        self._record(
            "RETREAT_READY",
            execute=self.execute and self.retreat_enabled,
            ready_straw_tip_position=self.ready_straw_tip_position,
            keep_flange_down=True,
            locked_flange_down_rpy=list(self.env.flange_down_rpy),
        )
        if not self.retreat_enabled:
            return {"skipped": True, "reason": "retreat disabled"}
        result = self.env.move_straw_tip_to_position(
            self.ready_straw_tip_position,
            label="ready_straw_tip_position",
            flange_down_rpy=self.env.flange_down_rpy,
            duration=self.duration,
            plan_only=not self.execute,
            planning_mode="cartesian",
        )
        if not result.get("success"):
            raise RuntimeError(f"MoveIt ready retreat failed: {result}")
        return result


    def start_gazebo_tracker(self) -> dict:
        if not self.gazebo_tracker_enabled:
            info = {"enabled": False, "reason": "gazebo tracker disabled"}
            self._record("GAZEBO_TRACKER_SKIPPED", **info)
            return info
        if not GAZEBO_SCENE_HELPER.exists():
            raise RuntimeError(f"Gazebo scene helper not found: {GAZEBO_SCENE_HELPER}")

        self.gazebo_tracker_log.parent.mkdir(parents=True, exist_ok=True)
        config_path = Path(os.environ.get("UR10E_SDK_CONFIG", DEFAULT_SDK_CONFIG))
        cmd = [
            sys.executable,
            str(GAZEBO_SCENE_HELPER),
            "--rate",
            str(self.tracker_rate),
            "--ready",
            _format_xyz(self.ready_straw_tip_position),
            "--pre-mouth",
            _format_xyz(self.pre_mouth_safe_position),
        ]
        if config_path.exists():
            cmd.extend(["--config", str(config_path)])
        if self.tracker_use_existing:
            cmd.append("--use-existing")

        log_handle = self.gazebo_tracker_log.open("a", encoding="utf-8")
        log_handle.write(f"\n--- feeding_fsm tracker start {time.time():.3f} ---\n")
        log_handle.flush()
        self.gazebo_tracker_process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.gazebo_tracker_process._codex_log_handle = log_handle  # type: ignore[attr-defined]
        time.sleep(1.0)
        returncode = self.gazebo_tracker_process.poll()
        info = {
            "enabled": True,
            "pid": self.gazebo_tracker_process.pid,
            "returncode": returncode,
            "log": str(self.gazebo_tracker_log),
            "helper": str(GAZEBO_SCENE_HELPER),
            "rate_hz": self.tracker_rate,
            "use_existing": self.tracker_use_existing,
        }
        self._record("GAZEBO_TRACKER_STARTED", **info)
        if returncode is not None:
            raise RuntimeError(f"Gazebo tracker exited early with code {returncode}; see {self.gazebo_tracker_log}")
        return info

    def stop_gazebo_tracker(self) -> None:
        process = self.gazebo_tracker_process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_handle = getattr(process, "_codex_log_handle", None)
        if log_handle is not None:
            log_handle.close()
        self._record("GAZEBO_TRACKER_STOPPED", returncode=process.returncode)

    def run(self) -> dict:
        tracker_info = self.start_gazebo_tracker() if self.gazebo_tracker_enabled else {"enabled": False, "reason": "attached URDF markers"}
        scene_info = self.describe_gazebo_scene_targets()
        initial_info = self.check_robot_state("INITIAL_ROBOT_STATE")
        ready_response = self.prepare_ready_down()
        self.hold_ready_down()
        ready_info = self.check_robot_state("READY_DOWN_CONFIRMED")
        target_info = self.check_pre_mouth_target(ready_info)
        move_response = self.move_straw_to_pre_mouth()
        after_info = self.check_robot_state("AFTER_PRE_MOUTH_MOTION")
        after_distance = _distance(
            after_info["estimated_straw_tip_position"],
            move_response["plan"]["pre_mouth_safe_position"],
        )
        self.wait_at_pre_mouth()
        retreat_response = self.retreat()
        final_info = self.check_robot_state("FINAL_ROBOT_STATE")

        return {
            "success": True,
            "mode": "fixed_fsm_without_llm_gazebo_moveit",
            "states": [event["state"] for event in self.events],
            "execute": self.execute,
            "gazebo_tracker": tracker_info,
            "scene_info": scene_info,
            "initial_info": initial_info,
            "ready_response": ready_response,
            "ready_info": ready_info,
            "target_info": target_info,
            "move_response": move_response,
            "after_info": after_info,
            "after_move_distance_to_pre_mouth_m": after_distance,
            "retreat_response": retreat_response,
            "final_info": final_info,
            "events": self.events,
            "note": (
                "No LLM is used here. Current camera and straw-tip markers are "
                "fixed visual links on tool0, so Gazebo moves them as part of the UR10e model; "
                "this process uses MoveIt to move the real UR10e simulation with "
                "flange-down orientation during the ready-to-pre-mouth feeding motion."
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--hold-ready-seconds", type=float, default=1.5)
    parser.add_argument("--wait-seconds", type=float, default=3.0)
    parser.add_argument("--pre-mouth", type=_parse_xyz, default=None, help="Override pre-mouth target as x,y,z.")
    parser.add_argument("--ready", type=_parse_xyz, default=None, help="Override ready straw-tip target as x,y,z.")
    parser.add_argument("--execute", action="store_true", help="Execute the MoveIt trajectory. Without this, plan only.")
    parser.add_argument("--retreat", action="store_true", help="Return to ready after waiting at pre-mouth.")
    parser.add_argument("--tracker-rate", type=float, default=10.0, help="Gazebo current marker update rate in Hz.")
    parser.add_argument(
        "--gazebo-tracker",
        action="store_true",
        help="Launch the legacy world-model Gazebo marker tracker for debugging.",
    )
    parser.add_argument(
        "--replace-gazebo-markers",
        action="store_true",
        help="Recreate Gazebo feeding markers instead of reusing existing world markers.",
    )
    parser.add_argument(
        "--stop-gazebo-tracker-on-exit",
        action="store_true",
        help="Terminate the Gazebo marker tracker when this demo exits.",
    )
    parser.add_argument(
        "--assume-ready-down",
        action="store_true",
        help="Skip the ready-down setup if the robot is already in the flange-down start pose.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = UR10eRobotEnv()
    fsm = FeedingNoLLMFSM(
        env,
        execute=args.execute,
        duration=args.duration,
        hold_ready_seconds=args.hold_ready_seconds,
        wait_seconds=args.wait_seconds,
        pre_mouth_safe_position=args.pre_mouth,
        ready_straw_tip_position=args.ready,
        retreat=args.retreat,
        prepare_ready=not args.assume_ready_down,
        gazebo_tracker=args.gazebo_tracker,
        tracker_rate=args.tracker_rate,
        tracker_use_existing=not args.replace_gazebo_markers,
        stop_tracker_on_exit=args.stop_gazebo_tracker_on_exit,
    )
    try:
        result = fsm.run()
    except Exception as exc:
        print(
            json.dumps(
                {"success": False, "error": str(exc), "events": fsm.events},
                default=_json_default,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    finally:
        if fsm.stop_tracker_on_exit:
            fsm.stop_gazebo_tracker()
        env.close()
    print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
