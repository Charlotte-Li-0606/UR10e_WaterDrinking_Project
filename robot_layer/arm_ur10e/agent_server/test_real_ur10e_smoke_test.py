"""No-motion unit tests for the guarded physical UR10e smoke-test CLI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "real_ur10e_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("real_ur10e_smoke_test_for_tests", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class _Arguments:
    def __init__(self, mode: str, confirm_real_motion: bool) -> None:
        self.mode = mode
        self.confirm_real_motion = confirm_real_motion


class RealUR10eSmokeTestGuards(unittest.TestCase):
    def test_default_smoke_target_is_fixed_two_centimeters_base_z(self) -> None:
        target = SMOKE._build_target(
            {
                "position": [0.1, -0.2, 0.3],
                "orientation_quat": [0.0, 0.0, 0.0, 1.0],
            }
        )

        self.assertEqual([0.1, -0.2, 0.32], target["position_m"])
        self.assertEqual([0.0, 0.0, 0.0, 1.0], target["orientation_quat_xyzw"])
        self.assertTrue(target["orientation_preserved"])

    def test_smoke_plan_uses_reviewed_linear_cartesian_planner(self) -> None:
        self.assertEqual("pilz_industrial_motion_planner", SMOKE.SMOKE_PLANNING_PIPELINE)
        self.assertEqual("LIN", SMOKE.SMOKE_PLANNER_ID)

    def test_execute_requires_both_cli_confirmation_and_environment_gate(self) -> None:
        previous = os.environ.pop("UR10E_ALLOW_REAL_EXECUTION", None)
        try:
            self.assertEqual(
                "execution requires --confirm-real-motion",
                SMOKE._guard_execute(_Arguments("execute", False)),
            )
            self.assertEqual(
                "execution requires UR10E_ALLOW_REAL_EXECUTION=1",
                SMOKE._guard_execute(_Arguments("execute", True)),
            )
            os.environ["UR10E_ALLOW_REAL_EXECUTION"] = "1"
            self.assertIsNone(SMOKE._guard_execute(_Arguments("execute", True)))
        finally:
            if previous is None:
                os.environ.pop("UR10E_ALLOW_REAL_EXECUTION", None)
            else:
                os.environ["UR10E_ALLOW_REAL_EXECUTION"] = previous

    def test_plan_and_check_modes_never_require_an_execution_gate(self) -> None:
        self.assertIsNone(SMOKE._guard_execute(_Arguments("check", False)))
        self.assertIsNone(SMOKE._guard_execute(_Arguments("plan", False)))


if __name__ == "__main__":
    unittest.main()
