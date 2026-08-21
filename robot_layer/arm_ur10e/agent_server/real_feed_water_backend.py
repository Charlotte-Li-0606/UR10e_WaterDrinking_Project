#!/usr/bin/env python3
"""Real-only ``feed_water`` adapter for the integrated guarded pipeline.

This module deliberately contains no perception projection, target geometry,
MoveIt planning, trajectory, joint, controller, cup-tilt, or pour logic. It
invokes the active ``scripts/real_feed_water_integrated.py`` implementation for
every original execution mode, including explicit continuous camera-ray
tracking. The separately preserved base-Y backup is never selected here.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_FEED_WATER_SCRIPT = PROJECT_ROOT / "scripts/real_feed_water_integrated.py"
REPORT_DIR = PROJECT_ROOT / "reports"

REAL_BACKEND = "real"
SAFE_DISTANCE_M = 0.050
MAXIMUM_PLAN_TRANSLATION_M = 1.30
MOUTH_SAMPLE_SECONDS = 1.0
TRAJECTORY_VELOCITY_SCALING = 0.30
TRAJECTORY_ACCELERATION_SCALING = 0.30
MIN_HOLD_SECONDS = 2.0
MAX_HOLD_SECONDS = 5.0
DEFAULT_HOLD_SECONDS = 5.0
PIPELINE_TIMEOUT_SECONDS = 240.0
MAXIMUM_PIPELINE_ERROR_DETAIL_CHARACTERS = 2000
SUPPORTED_TARGET_SELECTIONS = frozenset({"left", "center", "right"})


def _timestamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y%m%d_%H%M%S"), now.isoformat(timespec="milliseconds")


def _finite_hold_duration(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("hold_duration_sec must be between 2 and 5 seconds")
    duration = float(value)
    if not math.isfinite(duration) or not MIN_HOLD_SECONDS <= duration <= MAX_HOLD_SECONDS:
        raise ValueError("hold_duration_sec must be between 2 and 5 seconds")
    return duration


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pipeline_error_detail(completed: subprocess.CompletedProcess[str]) -> str | None:
    """Return bounded child diagnostics without exposing an unbounded log."""
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout
    if not detail:
        return None
    if len(detail) > MAXIMUM_PIPELINE_ERROR_DETAIL_CHARACTERS:
        detail = detail[-MAXIMUM_PIPELINE_ERROR_DETAIL_CHARACTERS:]
        detail = "[truncated to final output] " + detail
    return detail


def _pipeline_command(
    *,
    execute: bool,
    report_path: Path,
    target_selection: str,
    hold_duration_sec: float,
    track_mouth_during_execution: bool = False,
    continuous_mouth_tracking: bool = False,
    use_octomap: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REAL_FEED_WATER_SCRIPT),
        "--execute" if execute else "--plan-only",
        "--target-selection",
        target_selection,
        "--mouth-sample-seconds",
        str(MOUTH_SAMPLE_SECONDS),
        "--trajectory-velocity-scaling",
        str(TRAJECTORY_VELOCITY_SCALING),
        "--trajectory-acceleration-scaling",
        str(TRAJECTORY_ACCELERATION_SCALING),
        "--hold-duration",
        str(hold_duration_sec),
        "--report-file",
        str(report_path),
    ]
    if execute:
        command.extend(["--confirm-real-motion", "--allow-validated-camera-ray-execute"])
    else:
        command.append("--no-execute")
    if track_mouth_during_execution:
        command.append("--track-mouth-during-execution")
    if continuous_mouth_tracking:
        command.append("--continuous-mouth-tracking")
    if use_octomap:
        command.append("--use-octomap")
    return command


def _failure_report(
    *,
    captured_at: str,
    report_path: Path,
    execute: bool,
    hold_duration_sec: float,
    stage: str,
    reason: str,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "tool": "feed_water",
        "selected_backend": REAL_BACKEND,
        "captured_at": captured_at,
        "success": False,
        "mode": "execute" if execute else "plan_only",
        "stage": stage,
        "reason": reason,
        "safety_gates": dict(gates),
        "execution_attempted": False,
        "execution_sent": False,
        "direct_mouth_contact": False,
        "cup_tilt_commanded": False,
        "pour_commanded": False,
        "automatic_retreat_sent": False,
        "hold_duration_sec": hold_duration_sec,
        "final_state": "refused",
        "report_path": str(report_path),
    }
    _write_report(report_path, report)
    return report


def _tool_report(
    *,
    captured_at: str,
    report_path: Path,
    pipeline_report_path: Path,
    pipeline: Mapping[str, Any],
    execute: bool,
    hold_duration_sec: float,
    elapsed_sec: float,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    active_search = (
        pipeline.get("active_search")
        if isinstance(pipeline.get("active_search"), Mapping)
        else {}
    )
    checks = pipeline.get("checks") if isinstance(pipeline.get("checks"), Mapping) else {}
    if not checks and isinstance(active_search.get("checks"), Mapping):
        checks = active_search["checks"]
    actual = pipeline.get("actual") if isinstance(pipeline.get("actual"), Mapping) else {}
    execution_result = (
        pipeline.get("execution_result")
        if isinstance(pipeline.get("execution_result"), Mapping)
        else None
    )
    integrated = (
        pipeline.get("integrated_real_feed_water")
        if isinstance(pipeline.get("integrated_real_feed_water"), Mapping)
        else {}
    )
    adaptive = (
        pipeline.get("adaptive_goal_selection")
        if isinstance(pipeline.get("adaptive_goal_selection"), Mapping)
        else {}
    )
    selected_goal = (
        adaptive.get("selected_candidate")
        if isinstance(adaptive.get("selected_candidate"), Mapping)
        else {}
    )
    plan_result = (
        pipeline.get("plan_result")
        if isinstance(pipeline.get("plan_result"), Mapping)
        else {}
    )
    return_result = (
        pipeline.get("return_to_initial_position")
        if isinstance(pipeline.get("return_to_initial_position"), Mapping)
        else {}
    )
    hold_result = (
        pipeline.get("pre_mouth_hold")
        if isinstance(pipeline.get("pre_mouth_hold"), Mapping)
        else {}
    )
    success = bool(pipeline.get("success"))
    reason = pipeline.get("reason")
    if not reason and isinstance(pipeline.get("failures"), list):
        reason = "; ".join(str(item) for item in pipeline["failures"])
    if not reason and not success:
        reason = f"validated pre-mouth pipeline refused at {pipeline.get('stage', 'unknown stage')}"
    successful_stage = (
        "returned_initial_position"
        if execute and return_result.get("success")
        else "pre_mouth_and_return_target_validated"
        if not execute and return_result.get("success")
        else "holding_pre_mouth"
    )
    reported_final_state = pipeline.get("final_state")
    if reported_final_state in {
        "initial_position",
        "pre_mouth_and_return_target_validated",
        "holding_pre_mouth",
    }:
        final_state = str(reported_final_state)
    else:
        final_state = (
            "initial_position"
            if success and execute and return_result.get("success")
            else "pre_mouth_and_return_target_validated"
            if success and not execute and return_result.get("success")
            else "refused"
            if not success
            else "holding_pre_mouth"
        )
    report_stage = (
        successful_stage
        if success
        else str(pipeline.get("stage") or "refused")
    )
    execution_state_unknown = bool(pipeline.get("execution_state_unknown"))
    if execution_state_unknown:
        execution_attempted: bool | None = None
        execution_sent: bool | None = None
        outbound_execution_attempted: bool | None = None
        outbound_execution_sent: bool | None = None
    else:
        execution_attempted = bool(
            pipeline.get("execution_attempted")
            or return_result.get("execution_attempted")
        )
        execution_sent = bool(
            pipeline.get("execution_sent")
            or return_result.get("execution_sent")
        )
        outbound_execution_attempted = bool(pipeline.get("execution_attempted"))
        outbound_execution_sent = bool(pipeline.get("execution_sent"))
    report = {
        "schema_version": 1,
        "tool": "feed_water",
        "selected_backend": REAL_BACKEND,
        "captured_at": captured_at,
        "success": success,
        "mode": "execute" if execute else "plan_only",
        "stage": report_stage,
        "reason": reason,
        "camera_tf_used": checks.get("camera_tf"),
        "camera_mount_match": checks.get("camera_mount_match"),
        "mount_calibration": checks.get("mount_calibration") or pipeline.get("mount_calibration"),
        "detected_mouth_pose": pipeline.get("detected_mouth_pose"),
        "pre_mouth_target": pipeline.get("pre_mouth_pose"),
        "active_search": active_search,
        "dynamic_octomap_readiness": pipeline.get("dynamic_octomap_readiness"),
        "dynamic_scene_preparation": pipeline.get("dynamic_scene_preparation"),
        "integrated_real_feed_water": integrated,
        "adaptive_goal_selection": adaptive,
        "selected_candidate_standoff_m": adaptive.get("selected_standoff_m"),
        "selected_candidate_yaw_deg": adaptive.get("selected_yaw_deg"),
        "target_tool0_pose": pipeline.get("target_tool0_pose"),
        "selected_straw_tip_pose": adaptive.get("selected_straw_tip_pose"),
        "flange_vertical_axis_error_deg": selected_goal.get(
            "flange_vertical_axis_error_deg"
        ),
        "clearance_values": selected_goal.get("clearance"),
        "cartesian_planning_succeeded": bool(
            plan_result.get("route_strategy")
            == "complete_collision_checked_cartesian_path"
            and plan_result.get("success")
        ),
        "ompl_needed": bool(plan_result.get("detour_attempted")),
        "planned_displacement_m": pipeline.get("planned_tool0_translation_m"),
        "planned_displacement_norm_m": pipeline.get("planned_tool0_translation_norm_m"),
        "maximum_planned_displacement_m": MAXIMUM_PLAN_TRANSLATION_M,
        "execution_attempted": execution_attempted,
        "execution_sent": execution_sent,
        "execution_state_unknown": execution_state_unknown,
        "outbound_execution_attempted": outbound_execution_attempted,
        "outbound_execution_sent": outbound_execution_sent,
        "return_to_initial_position": return_result,
        "return_execution_attempted": bool(return_result.get("execution_attempted")),
        "return_execution_sent": bool(return_result.get("execution_sent")),
        "final_straw_tip_pose": actual.get("final_straw_tip_pose"),
        "final_error_m": actual.get("final_straw_tip_to_pre_mouth_error_m"),
        "controller_result": execution_result,
        "hold_duration_sec": hold_duration_sec,
        "hold_completed": bool(hold_result.get("completed")),
        "completion_status": pipeline.get("completion_status"),
        "recovered_warnings": pipeline.get("recovered_warnings", []),
        "pre_mouth_hold": hold_result,
        "mouth_tracking_during_execution": bool(
            integrated.get("mouth_tracking_during_execution")
        ),
        "tracking_replan_attempts": pipeline.get("tracking_replan_attempts"),
        "premouth_tracking": hold_result.get("tracking")
        if isinstance(hold_result, Mapping)
        else None,
        "safety_gates": {
            **dict(gates),
            "stable_mouth_pose": bool(checks.get("mouth_pose", {}).get("stable")),
            "multi_target_identity_lock": bool(integrated.get("multi_target_identity_lock")),
            "active_search_vertical_axis_constraint": bool(
                integrated.get("active_search_vertical_axis_constraint")
            ),
            "tool_axis_spin_free": bool(
                integrated.get("tool_axis_spin_free")
            ),
            "vertical_axis_obstacle_detour": bool(
                integrated.get("vertical_axis_detour_after_direct_rejection")
            ),
            "dynamic_same_target_replanning": bool(integrated.get("same_target_replanning")),
            "corrected_camera_tf_loaded": bool(checks.get("camera_mount_match", {}).get("matches")),
            "execution_runtime_gates_required": execute,
            "controller_active": None
            if not execute
            else bool(
                pipeline.get("pre_execution_controller_state", {}).get(
                    "scaled_joint_trajectory_controller_active"
                )
            ),
            "external_control_running": None
            if not execute
            else pipeline.get("pre_execution_robot_program_running", checks.get("robot_program_running")),
            "safety_mode_normal": None
            if not execute
            else pipeline.get("pre_execution_safety_mode_normal", checks.get("safety_mode_normal")),
            "robot_mode_running": None
            if not execute
            else pipeline.get("pre_execution_robot_mode_running", checks.get("robot_mode_running")),
            "speed_slider_percent": None
            if not execute
            else pipeline.get("pre_execution_speed_slider_percent", checks.get("speed_slider_percent")),
            "pipeline_passed": success,
        },
        "direct_mouth_contact": False,
        "cup_tilt_commanded": False,
        "pour_commanded": False,
        "automatic_retreat_sent": bool(
            return_result.get("automatic_retreat_sent", False)
        ),
        "final_state": final_state,
        "elapsed_sec": elapsed_sec,
        "pipeline_report_path": str(pipeline_report_path),
        "report_path": str(report_path),
        "pipeline_result": dict(pipeline),
    }
    _write_report(report_path, report)
    return report


def run_real_feed_water(
    *,
    execute: bool,
    confirm_real_motion: bool = False,
    target_selection: str = "center",
    hold_duration_sec: float = DEFAULT_HOLD_SECONDS,
    track_mouth_during_execution: bool = False,
    continuous_mouth_tracking: bool = False,
    use_octomap: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or plan the real safe pre-mouth-only ``feed_water`` operation."""
    environment = dict(os.environ if environ is None else environ)
    timestamp, captured_at = _timestamp()
    report_path = REPORT_DIR / f"feed_water_real_{timestamp}.json"
    pipeline_report_path = REPORT_DIR / f"feed_water_real_premouth_{timestamp}.json"
    try:
        duration = _finite_hold_duration(hold_duration_sec)
    except (TypeError, ValueError) as exc:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=0.0,
            stage="argument_validation",
            reason=str(exc),
            gates={},
        )

    gates = {
        "backend_real": environment.get("UR10E_BACKEND") == REAL_BACKEND,
        "real_execution_environment_enabled": (
            environment.get("UR10E_ALLOW_REAL_EXECUTION") == "1" if execute else None
        ),
        "explicit_runtime_confirmation": bool(confirm_real_motion) if execute else None,
        "target_selection_supported": target_selection in SUPPORTED_TARGET_SELECTIONS,
        "target_is_center_mouth": target_selection == "center",
        "pre_mouth_only": True,
        "no_direct_mouth_contact": True,
        "no_tilt_or_pour": True,
        "hold_duration_in_range": True,
        "tracked_feed_requested": bool(track_mouth_during_execution),
        "continuous_mouth_tracking_requested": bool(continuous_mouth_tracking),
        "use_octomap": bool(use_octomap),
    }
    if not gates["backend_real"]:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="backend_selection",
            reason="real feed_water requires UR10E_BACKEND=real",
            gates=gates,
        )
    if not gates["target_selection_supported"]:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="target_selection",
            reason="target_selection must be one of: left, center, right",
            gates=gates,
        )
    if execute and not gates["real_execution_environment_enabled"]:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=True,
            hold_duration_sec=duration,
            stage="execution_gate",
            reason="real feed_water execution requires UR10E_ALLOW_REAL_EXECUTION=1",
            gates=gates,
        )
    if execute and not gates["explicit_runtime_confirmation"]:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=True,
            hold_duration_sec=duration,
            stage="execution_gate",
            reason="real feed_water execution requires explicit runtime confirmation",
            gates=gates,
        )

    child_environment = dict(environment)
    child_environment["UR10E_BACKEND"] = REAL_BACKEND
    if not execute:
        child_environment.pop("UR10E_ALLOW_REAL_EXECUTION", None)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            _pipeline_command(
                execute=execute,
                report_path=pipeline_report_path,
                target_selection=target_selection,
                hold_duration_sec=duration,
                track_mouth_during_execution=track_mouth_during_execution,
                continuous_mouth_tracking=continuous_mouth_tracking,
                use_octomap=use_octomap,
            ),
            cwd=PROJECT_ROOT,
            env=child_environment,
            text=True,
            # Keep the final JSON stdout bounded for report parsing while
            # allowing structured stage transitions on stderr to reach the
            # operator immediately.
            stdout=subprocess.PIPE,
            stderr=None,
            timeout=PIPELINE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="pipeline_timeout",
            reason=f"validated real pre-mouth pipeline exceeded {PIPELINE_TIMEOUT_SECONDS:.0f} seconds",
            gates=gates,
        )
    try:
        pipeline = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        detail = _pipeline_error_detail(completed)
        reason = f"validated real pre-mouth pipeline did not produce a readable report: {exc}"
        if detail:
            reason += f"; pipeline output: {detail}"
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="pipeline_report",
            reason=reason,
            gates={**gates, "pipeline_exit_code": completed.returncode},
        )

    elapsed = time.monotonic() - started
    return _tool_report(
        captured_at=captured_at,
        report_path=report_path,
        pipeline_report_path=pipeline_report_path,
        pipeline=pipeline,
        execute=execute,
        hold_duration_sec=duration,
        elapsed_sec=elapsed,
        gates={**gates, "pipeline_exit_code": completed.returncode},
    )
