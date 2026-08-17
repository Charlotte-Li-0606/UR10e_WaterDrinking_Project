"""No-motion checks for the Codex-native feed_water route."""

from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = PROJECT_ROOT / "scripts" / "codex_feed_water.sh"
DIRECT_RUNNER = PROJECT_ROOT / "scripts" / "run_feed_water_real_direct.py"
PROCESS_GUARD = PROJECT_ROOT / "scripts" / "feed_water_process_guard.py"
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

    def test_codex_route_serializes_and_clears_only_stale_workflow_runners(
        self,
    ) -> None:
        entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
        guard = PROCESS_GUARD.read_text(encoding="utf-8")

        self.assertIn("flock -n 9", entrypoint)
        self.assertIn("feed_water_process_guard.py", entrypoint)
        self.assertIn('"--execute"', entrypoint)
        self.assertTrue(PROCESS_GUARD.is_file())
        self.assertIn("real_feed_water_integrated.py", guard)
        self.assertIn("run_feed_water_real_direct.py", guard)
        self.assertIn("feeding_safe_tool_runner.py", guard)
        self.assertIn("real_mouth_tracking_servo.py", guard)
        self.assertNotIn("pkill", guard)
        self.assertNotIn("SIGKILL", guard)

        standalone_tracking = (
            PROJECT_ROOT / "scripts" / "real_mouth_tracking_servo.py"
        ).read_text(encoding="utf-8")
        self.assertIn("WORKFLOW_LOCK_PATH", standalone_tracking)
        self.assertIn("LOCK_EX | fcntl.LOCK_NB", standalone_tracking)

    def test_codex_instructions_gate_execution_and_keep_plan_only(self) -> None:
        instructions = PROJECT_INSTRUCTIONS.read_text(encoding="utf-8")
        skill = CODEX_SKILL.read_text(encoding="utf-8")

        self.assertIn("codex_feed_water.sh --plan-only", instructions)
        self.assertIn("UR10E_ALLOW_REAL_EXECUTION=1", instructions)
        self.assertIn("--confirm-real-motion", instructions)
        self.assertIn("50 mm pre-mouth", skill)
        self.assertIn("0.30", skill)
        self.assertIn("codex_feed_water.sh", skill)
        self.assertIn("UR10E_ALLOW_REAL_EXECUTION=1", skill)
        self.assertIn("--execute --confirm-real-motion --hold-duration 5", skill)
        self.assertIn("--continuous-mouth-tracking", skill)
        self.assertIn("with tracking", skill)
        self.assertIn("guardedly return", skill)
        self.assertIn("final_state: initial_position", skill)
        self.assertIn("guarded return", skill)
        self.assertIn("plan-only", skill)

    def test_openclaw_fallback_is_preserved(self) -> None:
        self.assertTrue(OPENCLAW_ENTRYPOINT.is_file())


if __name__ == "__main__":
    unittest.main()
