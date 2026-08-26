"""Validated high-level tool surface for the isolated PGI Gazebo workflow.

This module contains no ROS client and no robot command implementation.  It
only validates the two approved Stage-6 operations and builds the fixed
command that delegates to the already verified Stage-5 workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHYSICAL_WORKFLOW = PROJECT_ROOT / "scripts" / "pgi_physical_grasp_demo.py"
PHYSICAL_PARAMETERS = PROJECT_ROOT / "config" / "pgi_physical_grasp.yaml"
SYSTEM_PYTHON = Path("/usr/bin/python3")

SAFE_PGI_SIMULATION_TOOL_NAMES = frozenset(
    {
        "plan_cup_grasp_cycle",
        "execute_cup_grasp_cycle",
    }
)


class PgiSimulationToolValidationError(ValueError):
    """Raised when a request leaves the approved simulation-only surface."""


def validate_safe_pgi_simulation_tool_call(
    tool: Any,
    args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one high-level PGI simulation request.

    Neither operation accepts runtime motion parameters.  Cup geometry,
    targets, joint goals, speeds, contact values, and controller names remain
    fixed in versioned project files.
    """
    if not isinstance(tool, str) or tool not in SAFE_PGI_SIMULATION_TOOL_NAMES:
        raise PgiSimulationToolValidationError(
            "tool is not in the approved PGI simulation tool set"
        )
    if args is None:
        raw_args: Mapping[str, Any] = {}
    elif isinstance(args, Mapping):
        raw_args = args
    else:
        raise PgiSimulationToolValidationError(f"{tool}.args must be an object")
    if raw_args:
        raise PgiSimulationToolValidationError(
            f"{tool} accepts no runtime arguments; received: "
            f"{', '.join(sorted(str(name) for name in raw_args))}"
        )
    return {
        "tool": tool,
        "args": {},
        "motion_capable": tool == "execute_cup_grasp_cycle",
    }


def build_pgi_simulation_command(
    call: Mapping[str, Any],
    report_path: Path,
) -> list[str]:
    """Build the fixed Stage-5 subprocess command for an approved call."""
    normalized = validate_safe_pgi_simulation_tool_call(
        call.get("tool"), call.get("args", {})
    )
    command = [
        str(SYSTEM_PYTHON),
        str(PHYSICAL_WORKFLOW),
    ]
    if normalized["motion_capable"]:
        command.extend(["--execute-sim", "--confirm-simulation"])
    command.extend(
        [
            "--report-json",
            str(report_path),
            "--suppress-console-report",
            "--ros-args",
            "--params-file",
            str(PHYSICAL_PARAMETERS),
        ]
    )
    return command


def read_pgi_simulation_report(report_path: Path) -> dict[str, Any]:
    """Load and minimally validate the delegated workflow result."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Stage-5 workflow did not produce a valid report: {error}") from error
    if not isinstance(report, dict):
        raise RuntimeError("Stage-5 workflow report must be a JSON object")
    return report


def summarize_pgi_simulation_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the detailed Stage-5 evidence to a stable high-level result."""
    summary: dict[str, Any] = {
        "success": bool(report.get("success", False)),
        "delegated_stage": report.get("stage", 5),
        "mode": report.get("mode"),
        "camera_model_xy_error_m": report.get("camera_model_xy_error_m"),
        "real_robot_command_sent": bool(report.get("real_robot_command_sent", False)),
    }
    if report.get("error"):
        summary["error"] = str(report["error"])
        summary["moveit_cup_may_remain_attached"] = bool(
            report.get("moveit_cup_may_remain_attached", False)
        )

    guards = report.get("guards")
    if isinstance(guards, Mapping):
        controllers = guards.get("controllers")
        controller_states = {}
        if isinstance(controllers, Mapping):
            controller_states = {
                str(name): details.get("state")
                for name, details in controllers.items()
                if isinstance(details, Mapping)
            }
        summary["guards"] = {
            "ros_domain_id": guards.get("ros_domain_id"),
            "move_group_plan_only": guards.get("move_group_plan_only"),
            "world_base_translation_error_m": guards.get(
                "world_base_translation_error_m"
            ),
            "world_base_rotation_error_deg": guards.get(
                "world_base_rotation_error_deg"
            ),
            "controller_states": controller_states,
        }

    strategy = report.get("selected_strategy")
    if isinstance(strategy, Mapping):
        summary["strategy"] = {
            "name": strategy.get("name"),
            "transit_flange_orientation_relaxed": strategy.get(
                "transit_flange_orientation_relaxed"
            ),
            "transit_flange_spin_deg": strategy.get("transit_flange_spin_deg"),
            "grasp_orientation_locked": strategy.get("grasp_orientation_locked"),
        }

    plans = report.get("plans")
    if isinstance(plans, Mapping):
        summary["planning"] = {
            "segment_count": len(plans),
            "backends": sorted(
                {
                    str(plan.get("backend"))
                    for plan in plans.values()
                    if isinstance(plan, Mapping) and plan.get("backend")
                }
            ),
        }

    execution = report.get("execution")
    if isinstance(execution, Mapping):
        if report.get("mode") == "plan_only":
            summary["execution"] = {
                "attempted": bool(execution.get("attempted", False)),
                "trajectory_sent": bool(execution.get("trajectory_sent", False)),
                "controller_switched": bool(
                    execution.get("controller_switched", False)
                ),
            }
            return summary
        observed_tilts = [
            float(item["tilt_deg"])
            for name, item in execution.items()
            if str(name).startswith("cup_")
            and isinstance(item, Mapping)
            and isinstance(item.get("tilt_deg"), (int, float))
        ]
        lift_delta = execution.get("measured_lift_delta_m")
        summary["execution"] = {
            "attempted": True,
            "success": bool(execution.get("success", False)),
            "native_contact_used": execution.get("native_contact_used"),
            "gazebo_pose_following_used": execution.get(
                "gazebo_pose_following_used"
            ),
            "measured_lift_z_m": (
                lift_delta[2]
                if isinstance(lift_delta, list) and len(lift_delta) == 3
                else None
            ),
            "hold_drift_m": execution.get("hold_drift_m"),
            "maximum_observed_cup_tilt_deg": max(observed_tilts)
            if observed_tilts
            else None,
            "release_drop_m": execution.get("measured_release_drop_m"),
            "cup_attached_at_end": execution.get("cup_attached_at_end"),
            "arm_controller_active_at_end": execution.get(
                "arm_controller_active_at_end"
            ),
        }
    elif isinstance(report.get("execution_partial"), Mapping):
        summary["execution_partial_stages"] = list(
            report["execution_partial"].keys()
        )
    return summary
