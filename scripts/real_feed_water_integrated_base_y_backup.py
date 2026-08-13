#!/usr/bin/env python3
"""Launch the preserved continuous base-Y feed-water backup in isolation.

This adapter deliberately leaves the active camera-ray implementation untouched.
It installs the backup tracker and controller under the canonical import names for
this child process only, then loads the backup integrated runner and points its two
project-local configuration constants at the real project and backup config.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = PROJECT_ROOT / "backups" / "continuous_base_y_tracking_20260813"
BACKUP_TRACKER = (
    BACKUP_ROOT
    / "robot_layer"
    / "arm_ur10e"
    / "perception"
    / "continuous_mouth_tracker.py"
)
BACKUP_CONTROLLER = (
    BACKUP_ROOT
    / "robot_layer"
    / "arm_ur10e"
    / "control"
    / "continuous_servo_tracking.py"
)
BACKUP_RUNNER = BACKUP_ROOT / "scripts" / "real_feed_water_integrated.py"
BACKUP_CONFIG = BACKUP_ROOT / "config" / "continuous_mouth_tracking.yaml"

TRACKER_MODULE = "robot_layer.arm_ur10e.perception.continuous_mouth_tracker"
CONTROLLER_MODULE = "robot_layer.arm_ur10e.control.continuous_servo_tracking"
RUNNER_MODULE = "continuous_base_y_backup_real_feed_water_integrated"


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load backup module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _report_path_from_argv() -> Path | None:
    try:
        index = sys.argv.index("--report-file")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return None


def _write_exception_report(exc: Exception) -> None:
    report_path = _report_path_from_argv()
    if report_path is None:
        return
    detail = str(exc).strip()
    report = {
        "success": False,
        "stage": "pipeline_exception",
        "reason": (
            f"{exc.__class__.__name__}: {detail}"
            if detail
            else exc.__class__.__name__
        ),
        "execution_attempted": None,
        "execution_sent": None,
        "execution_state_unknown": True,
        "final_state": "refused",
        "continuous_tracking_implementation": "base_y_backup",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_backup_runner() -> ModuleType:
    """Load and configure the preserved runner for this isolated process."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _load_module(TRACKER_MODULE, BACKUP_TRACKER)
    _load_module(CONTROLLER_MODULE, BACKUP_CONTROLLER)
    runner = _load_module(RUNNER_MODULE, BACKUP_RUNNER)
    runner.PROJECT_ROOT = PROJECT_ROOT
    initial_position_config = (
        PROJECT_ROOT / "config" / "ur10e_real" / "initial_position.json"
    )
    runner.INITIAL_POSITION_CONFIG = initial_position_config
    # The backup function captured its original path as a default argument at
    # module load time; replace that default for this isolated process.
    runner._load_initial_position_config.__defaults__ = (
        initial_position_config,
    )
    runner.CONTINUOUS_TRACKING_CONFIG = BACKUP_CONFIG
    return runner


def main() -> int:
    try:
        runner = load_backup_runner()
        return int(runner.main())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _write_exception_report(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
