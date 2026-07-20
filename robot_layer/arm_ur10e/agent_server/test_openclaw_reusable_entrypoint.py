"""Regression checks for the OpenClaw compatibility command route."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "openclaw_feed_water.sh"
BRIDGE = PROJECT_ROOT / "scripts" / "openclaw_feeding_tool.sh"
SIMULATOR_GATE = PROJECT_ROOT / "scripts" / "ensure_ur10e_feeding_sim.sh"
MOVEIT_LAUNCH = PROJECT_ROOT / "launch" / "ur10e_moveit_with_kinematics.launch.py"


class OpenClawReusableEntrypointTest(unittest.TestCase):
    def test_compatibility_command_dispatches_reusable_plan_not_llm_wrapper(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")
        bridge_content = BRIDGE.read_text(encoding="utf-8")
        simulator_gate = SIMULATOR_GATE.read_text(encoding="utf-8")

        self.assertIn('openclaw_feeding_tool.sh', content)
        self.assertIn('MODE="--execute"', content)
        self.assertIn('--validate-only --plan-json', content)
        self.assertNotIn('run_llm_feeding_agent.sh', content)
        self.assertNotIn('openclaw_feed_water.sh --plan-only', content)
        self.assertNotIn('"tool":"feed_water"', content)
        self.assertIn('ensure_ur10e_feeding_sim.sh', bridge_content)
        self.assertIn('UR10E_BACKEND', bridge_content)
        self.assertIn('real)', bridge_content)
        self.assertIn('start_ur10e_feeding_sim.sh', simulator_gate)
        self.assertIn('simulator_ready()', simulator_gate)
        for tool in (
            "get_observation",
            "detect_target",
            "active_search",
            "select_target",
            "move_tool_to_target",
            "check_progress",
            "hold",
        ):
            with self.subTest(tool=tool):
                self.assertIn(f'"tool":"{tool}"', content)

    def test_moveit_launch_does_not_pass_an_untyped_empty_sensor_array(self) -> None:
        content = MOVEIT_LAUNCH.read_text(encoding="utf-8")

        self.assertNotIn('moveit_parameters["sensors"] = []', content)
        self.assertIn('moveit_parameters.pop("sensors", None)', content)


if __name__ == "__main__":
    unittest.main()
