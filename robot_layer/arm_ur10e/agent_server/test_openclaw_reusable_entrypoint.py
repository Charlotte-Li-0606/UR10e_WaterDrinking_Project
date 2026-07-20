"""Regression checks for the OpenClaw compatibility command route."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "openclaw_feed_water.sh"


class OpenClawReusableEntrypointTest(unittest.TestCase):
    def test_compatibility_command_dispatches_reusable_plan_not_llm_wrapper(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn('openclaw_feeding_tool.sh', content)
        self.assertIn('--validate-only --plan-json', content)
        self.assertNotIn('run_llm_feeding_agent.sh', content)
        self.assertNotIn('openclaw_feed_water.sh --plan-only', content)
        self.assertNotIn('"tool":"feed_water"', content)
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


if __name__ == "__main__":
    unittest.main()
