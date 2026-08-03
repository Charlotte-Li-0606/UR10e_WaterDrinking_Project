"""No-ROS-motion checks for UR10e simulation/real backend selection."""

from __future__ import annotations

import unittest

from robot_layer.arm_ur10e.agent_server.robot_sdk.backend import (
    BackendConfigurationError,
    RealExecutionBlockedError,
    require_real_execution_authorized,
    resolve_ur10e_backend_settings,
)


CONFIG = {
    "backend": {
        "default": "sim",
        "sim": {
            "trajectory_action": "/joint_trajectory_controller/follow_joint_trajectory",
            "expected_controller": "joint_trajectory_controller",
        },
        "real": {
            "trajectory_action": "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            "expected_controller": "scaled_joint_trajectory_controller",
        },
    }
}


class UR10eBackendSafetyTest(unittest.TestCase):
    def test_sim_is_the_default_backend(self) -> None:
        settings = resolve_ur10e_backend_settings(CONFIG, environ={})

        self.assertEqual("sim", settings.name)
        self.assertEqual("joint_trajectory_controller", settings.expected_controller)
        self.assertFalse(settings.real_execution_allowed)

    def test_real_execution_is_rejected_by_default(self) -> None:
        settings = resolve_ur10e_backend_settings(CONFIG, environ={"UR10E_BACKEND": "real"})

        self.assertEqual("real", settings.name)
        self.assertFalse(settings.real_execution_allowed)
        self.assertEqual(0.60, settings.max_velocity_limit)
        self.assertEqual(0.60, settings.max_acceleration_limit)
        with self.assertRaises(RealExecutionBlockedError):
            require_real_execution_authorized(settings)

    def test_real_execution_needs_the_explicit_process_gate(self) -> None:
        settings = resolve_ur10e_backend_settings(
            CONFIG,
            environ={
                "UR10E_BACKEND": "real",
                "UR10E_ALLOW_REAL_EXECUTION": "1",
                "UR10E_ROBOT_IP": "192.0.2.10",
            },
        )

        self.assertTrue(settings.real_execution_allowed)
        self.assertTrue(settings.status()["robot_ip_configured"])
        require_real_execution_authorized(settings)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(BackendConfigurationError):
            resolve_ur10e_backend_settings(CONFIG, environ={"UR10E_BACKEND": "piper"})


if __name__ == "__main__":
    unittest.main()
