"""No-motion checks for the Codex-native feed_water route."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "codex_feed_water.sh"
DIRECT_RUNNER = PROJECT_ROOT / "scripts" / "run_feed_water_real_direct.py"
PROJECT_INSTRUCTIONS = PROJECT_ROOT / "AGENTS.md"
CODEX_SKILL = Path("/home/dase-hw101/.codex/skills/feed-water-ur10e/SKILL.md")
OPENCLAW_ENTRYPOINT = PROJECT_ROOT / "scripts" / "openclaw_feeding_tool.sh"


class CodexFeedWaterEntrypointTest(unittest.TestCase):
    def test_codex_route_is_real_only_and_bypasses_openclaw(self) -> None:
        content = ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn("export UR10E_BACKEND=real", content)
        self.assertIn("run_feed_water_real_direct.py", content)
        self.assertNotIn("openclaw_feeding_tool.sh", content)
        self.assertNotIn("ensure_ur10e_feeding_sim.sh", content)
        self.assertTrue(DIRECT_RUNNER.is_file())

    def test_codex_instructions_gate_execution_and_keep_plan_only(self) -> None:
        instructions = PROJECT_INSTRUCTIONS.read_text(encoding="utf-8")
        skill = CODEX_SKILL.read_text(encoding="utf-8")

        self.assertIn("codex_feed_water.sh --plan-only", instructions)
        self.assertIn("UR10E_ALLOW_REAL_EXECUTION=1", instructions)
        self.assertIn("--confirm-real-motion", instructions)
        self.assertIn("80 mm pre-mouth", skill)
        self.assertIn("codex_feed_water.sh", skill)
        self.assertIn("UR10E_ALLOW_REAL_EXECUTION=1", skill)
        self.assertIn("--execute --confirm-real-motion --hold-duration 5", skill)
        self.assertIn("final_state: initial_position", skill)
        self.assertIn("guarded return", skill)
        self.assertIn("plan-only", skill)

    def test_openclaw_fallback_is_preserved(self) -> None:
        self.assertTrue(OPENCLAW_ENTRYPOINT.is_file())


if __name__ == "__main__":
    unittest.main()
