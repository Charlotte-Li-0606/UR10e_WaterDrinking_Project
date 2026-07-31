#!/usr/bin/env python3
"""Real-only ``feed_water`` adapter for the integrated guarded pipeline.

This module deliberately contains no perception projection, target geometry,
MoveIt planning, trajectory, joint, controller, cup-tilt, or pour logic.  It
invokes ``scripts/real_feed_water_integrated.py``.  That real-only state
machine retains selected-person identity, performs bounded translation-only
active search when needed, freezes the camera-ray 80 mm pre-mouth target, and
uses the wrist OctoMap for same-target alternate-path replanning.  This adapter
then optionally performs a motionless dwell at the final pre-mouth pose.
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
SAFE_DISTANCE_M = 0.080
MAXIMUM_PLAN_TRANSLATION_M = 1.30
MOUTH_SAMPLE_SECONDS = 1.0
MIN_HOLD_SECONDS = 2.0
MAX_HOLD_SECONDS = 5.0
DEFAULT_HOLD_SECONDS = 3.0
PIPELINE_TIMEOUT_SECONDS = 240.0


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


def _pipeline_command(
    *,
    execute: bool,
    report_path: Path,
    target_selection: str,
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
        "0.10",
        "--trajectory-acceleration-scaling",
        "0.10",
        "--report-file",
        str(report_path),
    ]
    if execute:
        command.extend(["--confirm-real-motion", "--allow-validated-camera-ray-execute"])
    else:
        command.append("--no-execute")
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
    success = bool(pipeline.get("success"))
    reason = pipeline.get("reason")
    if not reason and isinstance(pipeline.get("failures"), list):
        reason = "; ".join(str(item) for item in pipeline["failures"])
    if not reason and not success:
        reason = f"validated pre-mouth pipeline refused at {pipeline.get('stage', 'unknown stage')}"
    report = {
        "schema_version": 1,
        "tool": "feed_water",
        "selected_backend": REAL_BACKEND,
        "captured_at": captured_at,
        "success": success,
        "mode": "execute" if execute else "plan_only",
        "stage": "holding_pre_mouth" if success and execute else "pre_mouth_plan_validated" if success else "refused",
        "reason": reason,
        "camera_tf_used": checks.get("camera_tf"),
        "camera_mount_match": checks.get("camera_mount_match"),
        "mount_calibration": checks.get("mount_calibration") or pipeline.get("mount_calibration"),
        "detected_mouth_pose": pipeline.get("detected_mouth_pose"),
        "pre_mouth_target": pipeline.get("pre_mouth_pose"),
        "active_search": active_search,
        "dynamic_octomap_readiness": pipeline.get("dynamic_octomap_readiness"),
        "integrated_real_feed_water": integrated,
        "target_tool0_pose": pipeline.get("target_tool0_pose"),
        "planned_displacement_m": pipeline.get("planned_tool0_translation_m"),
        "planned_displacement_norm_m": pipeline.get("planned_tool0_translation_norm_m"),
        "maximum_planned_displacement_m": MAXIMUM_PLAN_TRANSLATION_M,
        "execution_attempted": bool(pipeline.get("execution_attempted")),
        "execution_sent": bool(pipeline.get("execution_sent")),
        "final_straw_tip_pose": actual.get("final_straw_tip_pose"),
        "final_error_m": actual.get("final_straw_tip_to_pre_mouth_error_m"),
        "controller_result": execution_result,
        "hold_duration_sec": hold_duration_sec,
        "hold_completed": bool(success and execute),
        "safety_gates": {
            **dict(gates),
            "stable_mouth_pose": bool(checks.get("mouth_pose", {}).get("stable")),
            "multi_target_identity_lock": bool(integrated.get("multi_target_identity_lock")),
            "translation_only_active_search": bool(integrated.get("translation_only_search")),
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
        "automatic_retreat_sent": bool(pipeline.get("automatic_retreat_sent", False)),
        "final_state": "holding_pre_mouth" if success and execute else "pre_mouth_plan_validated" if success else "refused",
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
        "target_is_center_mouth": target_selection == "center",
        "pre_mouth_only": True,
        "no_direct_mouth_contact": True,
        "no_tilt_or_pour": True,
        "hold_duration_in_range": True,
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
    if not gates["target_is_center_mouth"]:
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="target_selection",
            reason="the validated real backend currently supports only the center mouth target",
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
            ),
            cwd=PROJECT_ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
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
        return _failure_report(
            captured_at=captured_at,
            report_path=report_path,
            execute=execute,
            hold_duration_sec=duration,
            stage="pipeline_report",
            reason=f"validated real pre-mouth pipeline did not produce a readable report: {exc}",
            gates={**gates, "pipeline_exit_code": completed.returncode},
        )

    if execute and pipeline.get("success"):
        # Deliberate no-command dwell. The validated MoveIt action is
        # synchronous and has already reached the 80 mm pre-mouth target.
        time.sleep(duration)
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
