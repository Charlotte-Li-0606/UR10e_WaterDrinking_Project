"""No-motion checks for exact stale workflow process matching."""

from __future__ import annotations

import unittest

from scripts import feed_water_process_guard as guard


class FeedWaterProcessGuardTest(unittest.TestCase):
    def test_matches_absolute_integrated_runner(self) -> None:
        script = guard.PROJECT_ROOT / "scripts" / "real_feed_water_integrated.py"

        role = guard._workflow_role(("/usr/bin/python3", str(script), "--execute"))

        self.assertEqual("integrated_runner", role)

    def test_matches_project_relative_tracking_runner(self) -> None:
        role = guard._workflow_role(
            (
                "/usr/bin/python3",
                "scripts/real_mouth_tracking_servo.py",
                "--execute",
            ),
            process_cwd=guard.PROJECT_ROOT,
        )

        self.assertEqual("tracking_runner", role)

    def test_safe_tool_runner_requires_feed_water_tool(self) -> None:
        script = (
            guard.PROJECT_ROOT
            / "robot_layer"
            / "arm_ur10e"
            / "agent_server"
            / "feeding_safe_tool_runner.py"
        )

        self.assertEqual(
            "safe_tool_runner",
            guard._workflow_role(
                ("python3", str(script), "--tool", "feed_water")
            ),
        )
        self.assertIsNone(
            guard._workflow_role(
                ("python3", str(script), "--tool", "detect_target")
            )
        )

    def test_does_not_match_long_running_stack_processes(self) -> None:
        commands = (
            ("/opt/ros/jazzy/lib/moveit_ros_move_group/move_group",),
            ("/opt/ros/jazzy/lib/moveit_servo/servo_node",),
            (
                "/home/dase-hw101/ur_drinking_project/venv/bin/python",
                str(
                    guard.PROJECT_ROOT
                    / "robot_layer"
                    / "arm_ur10e"
                    / "perception"
                    / "mouth_perception_node.py"
                ),
            ),
            ("/opt/ros/jazzy/lib/rqt_image_view/rqt_image_view",),
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(guard._workflow_role(command))


if __name__ == "__main__":
    unittest.main()
