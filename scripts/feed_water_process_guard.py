#!/usr/bin/env python3
"""Find and gracefully stop orphaned real feed-water workflow processes.

The canonical shell entrypoint holds a single-execution lock before invoking
this helper.  Only exact project-owned workflow scripts are in scope; the UR
driver, MoveIt, Servo, perception, camera, and RQT processes are never matched.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = {
    PROJECT_ROOT / "scripts" / "real_mouth_tracking_servo.py": "tracking_runner",
    PROJECT_ROOT / "scripts" / "real_feed_water_integrated.py": "integrated_runner",
    PROJECT_ROOT
    / "robot_layer"
    / "arm_ur10e"
    / "agent_server"
    / "feeding_safe_tool_runner.py": "safe_tool_runner",
    PROJECT_ROOT / "scripts" / "run_feed_water_real_direct.py": "direct_runner",
}
INTERRUPT_TIMEOUT_SEC = 3.0
TERMINATE_TIMEOUT_SEC = 2.0


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    role: str
    command: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report matching workflow processes without sending a signal.",
    )
    return parser.parse_args()


def _read_command(pid: int) -> tuple[str, ...]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ()
    return tuple(
        token.decode("utf-8", errors="replace")
        for token in raw.split(b"\0")
        if token
    )


def _read_stat(pid: int) -> tuple[int, int] | None:
    try:
        value = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    # The process name is parenthesized and may contain spaces. Fields after
    # the final ')' start at procfs stat field 3 (state).
    fields = value[value.rfind(")") + 2 :].split()
    try:
        return int(fields[1]), int(fields[19])  # PPID (4), starttime (22)
    except (IndexError, ValueError):
        return None


def _owned_by_current_user(pid: int) -> bool:
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8"
        ).splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    uid_line = next((line for line in lines if line.startswith("Uid:")), "")
    try:
        real_uid = int(uid_line.split()[1])
    except (IndexError, ValueError):
        return False
    return real_uid == os.getuid()


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    current = int(pid)
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        stat = _read_stat(current)
        if stat is None:
            break
        current = stat[0]
    ancestors.add(1)
    return ancestors


def _normalized_script_tokens(
    command: Iterable[str], *, process_cwd: Path
) -> set[Path]:
    scripts: set[Path] = set()
    for token in command:
        if not token.endswith((".py", ".sh")):
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = process_cwd / candidate
        try:
            scripts.add(candidate.resolve(strict=False))
        except OSError:
            continue
    return scripts


def _workflow_role(
    command: tuple[str, ...], *, process_cwd: Path = PROJECT_ROOT
) -> str | None:
    script_tokens = _normalized_script_tokens(
        command,
        process_cwd=process_cwd,
    )
    matches = [
        role
        for script, role in WORKFLOW_SCRIPTS.items()
        if script.resolve(strict=False) in script_tokens
    ]
    if len(matches) != 1:
        return None
    role = matches[0]
    if role == "safe_tool_runner":
        try:
            tool_index = command.index("--tool")
        except ValueError:
            return None
        if tool_index + 1 >= len(command) or command[tool_index + 1] != "feed_water":
            return None
    return role


def _matching_processes() -> list[ProcessIdentity]:
    excluded = _ancestor_pids(os.getpid())
    matches: list[ProcessIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded or not _owned_by_current_user(pid):
            continue
        command = _read_command(pid)
        try:
            process_cwd = (Path("/proc") / str(pid) / "cwd").resolve(
                strict=True
            )
        except (FileNotFoundError, OSError, PermissionError):
            continue
        role = _workflow_role(command, process_cwd=process_cwd)
        stat = _read_stat(pid)
        if role is None or stat is None:
            continue
        matches.append(
            ProcessIdentity(
                pid=pid,
                start_ticks=stat[1],
                role=role,
                command=command,
            )
        )
    role_order = {
        "tracking_runner": 0,
        "integrated_runner": 0,
        "safe_tool_runner": 1,
        "direct_runner": 2,
    }
    return sorted(matches, key=lambda item: (role_order[item.role], item.pid))


def _identity_is_alive(process: ProcessIdentity) -> bool:
    stat = _read_stat(process.pid)
    return bool(stat is not None and stat[1] == process.start_ticks)


def _wait_for_exit(
    processes: Iterable[ProcessIdentity], *, timeout_sec: float
) -> list[ProcessIdentity]:
    pending = [process for process in processes if _identity_is_alive(process)]
    deadline = time.monotonic() + float(timeout_sec)
    while pending and time.monotonic() < deadline:
        time.sleep(0.05)
        pending = [
            process for process in pending if _identity_is_alive(process)
        ]
    return pending


def _signal(processes: Iterable[ProcessIdentity], signum: signal.Signals) -> None:
    for process in processes:
        if not _identity_is_alive(process):
            continue
        try:
            os.kill(process.pid, signum)
        except (PermissionError, ProcessLookupError):
            continue


def _process_report(process: ProcessIdentity) -> dict[str, object]:
    return {
        "pid": process.pid,
        "role": process.role,
        "command": list(process.command),
    }


def main() -> int:
    args = _parse_args()
    matches = _matching_processes()
    report: dict[str, object] = {
        "success": True,
        "stage": "stale_workflow_process_cleanup",
        "check_only": bool(args.check_only),
        "matched_processes": [_process_report(item) for item in matches],
        "interrupt_sent_to": [],
        "terminate_sent_to": [],
        "surviving_processes": [],
        "protected_process_scope": [
            "ur_robot_driver",
            "move_group",
            "servo_node",
            "mouth_perception_node",
            "realsense_camera",
            "rqt_image_view",
        ],
    }
    if args.check_only or not matches:
        print(json.dumps(report, sort_keys=True))
        return 0

    _signal(matches, signal.SIGINT)
    report["interrupt_sent_to"] = [item.pid for item in matches]
    survivors = _wait_for_exit(matches, timeout_sec=INTERRUPT_TIMEOUT_SEC)
    if survivors:
        _signal(survivors, signal.SIGTERM)
        report["terminate_sent_to"] = [item.pid for item in survivors]
        survivors = _wait_for_exit(
            survivors,
            timeout_sec=TERMINATE_TIMEOUT_SEC,
        )
    report["surviving_processes"] = [
        _process_report(item) for item in survivors
    ]
    if survivors:
        report.update(
            success=False,
            reason=(
                "stale workflow processes survived SIGINT and SIGTERM; "
                "new execution is refused"
            ),
        )
        print(json.dumps(report, sort_keys=True))
        return 2
    report["reason"] = None
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
