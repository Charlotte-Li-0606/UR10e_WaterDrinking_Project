#!/usr/bin/env python3
"""Run one approved PGI Stage-6 tool against an already running Gazebo stack.

The runner never launches Gazebo, MoveIt, a robot driver, or RS485.  It only
delegates to the fixed Stage-5 physical-grasp workflow after validating the
high-level call and the isolated simulation environment.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_tools.pgi_simulation_tools import (  # noqa: E402
    SAFE_PGI_SIMULATION_TOOL_NAMES,
    PgiSimulationToolValidationError,
    build_pgi_simulation_command,
    read_pgi_simulation_report,
    summarize_pgi_simulation_report,
    validate_safe_pgi_simulation_tool_call,
)


LOCK_PATH = Path("/tmp/ur_drinking_project_pgi_simulation.lock")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, choices=sorted(SAFE_PGI_SIMULATION_TOOL_NAMES))
    parser.add_argument(
        "--args-json",
        default="{}",
        help="Must be an empty JSON object; runtime motion parameters are not accepted.",
    )
    parser.add_argument(
        "--execute-sim",
        action="store_true",
        help="Required for execute_cup_grasp_cycle; never authorizes real motion.",
    )
    parser.add_argument(
        "--confirm-simulation",
        action="store_true",
        help="Second explicit confirmation required for execute_cup_grasp_cycle.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the high-level request without initializing ROS or running Stage 5.",
    )
    return parser.parse_args()


def _result(success: bool, **fields: Any) -> dict[str, Any]:
    return {
        "event": "safe_pgi_simulation_tool_result",
        "success": success,
        "stage": 6,
        "simulation_only": True,
        "real_robot_command_sent": False,
        **fields,
    }


def _print_result(result: dict[str, Any]) -> int:
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("success") else 2


def _validate_environment() -> int:
    raw_domain = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        domain_id = int(raw_domain)
    except ValueError as error:
        raise PgiSimulationToolValidationError(
            f"ROS_DOMAIN_ID must be a nonzero integer, got {raw_domain!r}"
        ) from error
    if domain_id == 0:
        raise PgiSimulationToolValidationError(
            "ROS domain 0 is refused; Stage 6 is simulation-only"
        )
    if os.environ.get("UR10E_BACKEND", "").strip().lower() == "real":
        raise PgiSimulationToolValidationError(
            "UR10E_BACKEND=real is incompatible with the PGI simulation tool"
        )
    if os.environ.get("UR10E_ALLOW_REAL_EXECUTION", "").strip() == "1":
        raise PgiSimulationToolValidationError(
            "UR10E_ALLOW_REAL_EXECUTION=1 must be unset for the PGI simulation tool"
        )
    return domain_id


def _run_delegated(command: list[str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        # Keep stdout reserved for the final machine-readable Stage-6 result.
        print(line, end="", file=sys.stderr, flush=True)
    return process.wait()


def main() -> int:
    cli = _parse_args()
    try:
        raw_args = json.loads(cli.args_json)
        call = validate_safe_pgi_simulation_tool_call(cli.tool, raw_args)
    except json.JSONDecodeError as error:
        return _print_result(
            _result(False, final_state="validation_refused", reason=f"tool JSON is invalid: {error.msg}")
        )
    except PgiSimulationToolValidationError as error:
        return _print_result(
            _result(False, final_state="validation_refused", reason=str(error))
        )

    if cli.validate_only:
        if cli.execute_sim or cli.confirm_simulation:
            return _print_result(
                _result(
                    False,
                    call=call,
                    final_state="validation_refused",
                    reason="--validate-only cannot be combined with execution flags",
                )
            )
        return _print_result(
            _result(
                True,
                call=call,
                mode="validate_only",
                final_state="tool_validated",
                note="No ROS, Gazebo, MoveIt, controller, or robot driver was initialized.",
            )
        )

    motion_capable = bool(call["motion_capable"])
    if motion_capable and not (cli.execute_sim and cli.confirm_simulation):
        return _print_result(
            _result(
                False,
                call=call,
                final_state="authorization_refused",
                reason=(
                    "execute_cup_grasp_cycle requires both --execute-sim and "
                    "--confirm-simulation"
                ),
            )
        )
    if not motion_capable and (cli.execute_sim or cli.confirm_simulation):
        return _print_result(
            _result(
                False,
                call=call,
                final_state="authorization_refused",
                reason="plan_cup_grasp_cycle does not accept execution flags",
            )
        )

    try:
        domain_id = _validate_environment()
    except PgiSimulationToolValidationError as error:
        return _print_result(
            _result(False, call=call, final_state="environment_refused", reason=str(error))
        )

    lock_file = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _print_result(
                _result(
                    False,
                    call=call,
                    final_state="busy",
                    reason="another PGI simulation workflow is already active",
                )
            )

        descriptor, report_name = tempfile.mkstemp(
            prefix="pgi_stage6_", suffix=".json", dir="/tmp"
        )
        os.close(descriptor)
        report_path = Path(report_name)
        try:
            command = build_pgi_simulation_command(call, report_path)
            return_code = _run_delegated(command)
            try:
                workflow = read_pgi_simulation_report(report_path)
            except RuntimeError as error:
                return _print_result(
                    _result(
                        False,
                        call=call,
                        mode="execute_sim" if motion_capable else "plan_only",
                        ros_domain_id=domain_id,
                        final_state="missing_workflow_report",
                        delegated_return_code=return_code,
                        reason=str(error),
                    )
                )
        finally:
            report_path.unlink(missing_ok=True)
    finally:
        lock_file.close()

    success = return_code == 0
    workflow_summary = summarize_pgi_simulation_report(workflow)
    return _print_result(
        _result(
            success,
            call=call,
            mode="execute_sim" if motion_capable else "plan_only",
            ros_domain_id=domain_id,
            final_state="complete" if success else "workflow_failed",
            delegated_return_code=return_code,
            workflow_summary=workflow_summary,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
