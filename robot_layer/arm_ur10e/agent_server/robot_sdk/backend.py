"""Small backend-selection policy for the canonical UR10e ROS SDK.

This module deliberately does not implement a second robot SDK.  It only
selects the already-existing ROS 2 / MoveIt transport configuration and keeps
physical-robot execution opt-in.  Both simulation and physical robots use the
same ``UR10eRobotEnv`` and the same MoveIt planning calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_BACKENDS = frozenset({"sim", "real"})


class BackendConfigurationError(ValueError):
    """Raised when a requested UR10e backend is not supported."""


class RealExecutionBlockedError(RuntimeError):
    """Raised before any real-robot command is sent without explicit consent."""


@dataclass(frozen=True)
class UR10eBackendSettings:
    """Resolved connection and execution policy for one UR10e SDK instance."""

    name: str
    trajectory_action: str
    expected_controller: str
    robot_ip: str | None
    real_execution_allowed: bool
    max_velocity_limit: float | None
    max_acceleration_limit: float | None

    @property
    def is_real(self) -> bool:
        return self.name == "real"

    def status(self) -> dict[str, object]:
        """Return non-sensitive diagnostics; the robot address is not echoed."""
        return {
            "backend": self.name,
            "trajectory_action": self.trajectory_action,
            "expected_controller": self.expected_controller,
            "robot_ip_configured": bool(self.robot_ip),
            "real_execution_allowed": self.real_execution_allowed,
            "max_velocity_limit": self.max_velocity_limit,
            "max_acceleration_limit": self.max_acceleration_limit,
        }


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in _TRUE_VALUES


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def resolve_ur10e_backend_settings(
    config: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> UR10eBackendSettings:
    """Resolve the configured backend without contacting ROS or a robot.

    ``UR10E_BACKEND`` intentionally defaults to ``sim``.  A real backend can
    be selected for read-only status and MoveIt plan-only work, but its motion
    path stays blocked until ``UR10E_ALLOW_REAL_EXECUTION=1`` is supplied in
    that process environment.
    """
    cfg = _mapping(config)
    environment = os.environ if environ is None else environ
    backend_cfg = _mapping(cfg.get("backend"))
    requested = str(environment.get("UR10E_BACKEND", backend_cfg.get("default", "sim"))).strip().lower()
    if requested not in _BACKENDS:
        raise BackendConfigurationError("UR10E_BACKEND must be either 'sim' or 'real'")

    selected_cfg = _mapping(backend_cfg.get(requested))
    default_action = (
        "/joint_trajectory_controller/follow_joint_trajectory"
        if requested == "sim"
        else "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    )
    default_controller = "joint_trajectory_controller" if requested == "sim" else "scaled_joint_trajectory_controller"
    robot_ip = str(environment.get("UR10E_ROBOT_IP", "")).strip() or None
    real_allowed = _enabled(environment.get("UR10E_ALLOW_REAL_EXECUTION", "0"))

    return UR10eBackendSettings(
        name=requested,
        trajectory_action=str(selected_cfg.get("trajectory_action", default_action)),
        expected_controller=str(selected_cfg.get("expected_controller", default_controller)),
        robot_ip=robot_ip,
        real_execution_allowed=real_allowed if requested == "real" else False,
        max_velocity_limit=0.60 if requested == "real" else None,
        max_acceleration_limit=0.60 if requested == "real" else None,
    )


def require_real_execution_authorized(settings: UR10eBackendSettings) -> None:
    """Reject real motion before a controller, action, or trajectory is touched."""
    if settings.is_real and not settings.real_execution_allowed:
        raise RealExecutionBlockedError(
            "Real UR10e execution is blocked. Set UR10E_ALLOW_REAL_EXECUTION=1 only after "
            "the physical robot, workspace, controller, and operator safety checks are ready."
        )
    if settings.is_real and not settings.robot_ip:
        raise RealExecutionBlockedError(
            "Real UR10e execution requires UR10E_ROBOT_IP to be set in the supervised operator process."
        )
