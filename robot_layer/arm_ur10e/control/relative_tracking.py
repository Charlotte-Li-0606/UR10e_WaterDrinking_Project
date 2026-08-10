"""Pure relative-mouth tracking policy used by the real Servo runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RelativeTrackingCommand:
    allowed: bool
    reason: str
    desired_tool_position_m: tuple[float, float, float]
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_rps: tuple[float, float, float]
    mouth_displacement_m: float


class RelativeTrackingSession:
    """Map small mouth displacement to bounded tool translation.

    The tool orientation is not recomputed.  Angular velocity is always zero,
    preserving the already validated pre-mouth approach orientation.
    """

    def __init__(self, *, max_target_displacement_m: float = 0.06,
                 max_tool_radius_m: float = 1.30, max_linear_speed_mps: float = 0.02,
                 max_linear_acceleration_mps2: float = 0.10, gain: float = 0.8):
        self.max_target_displacement_m = float(max_target_displacement_m)
        self.max_tool_radius_m = float(max_tool_radius_m)
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.max_linear_acceleration_mps2 = float(max_linear_acceleration_mps2)
        self.gain = float(gain)
        self._mouth_reference = None
        self._tool_reference = None
        self._last_velocity = np.zeros(3, dtype=np.float64)

    def lock(self, mouth_position_m: Sequence[float], tool_position_m: Sequence[float]) -> None:
        self._mouth_reference = self._vector(mouth_position_m, "mouth_position_m")
        self._tool_reference = self._vector(tool_position_m, "tool_position_m")
        self._last_velocity[:] = 0.0

    @property
    def locked(self) -> bool:
        return self._mouth_reference is not None and self._tool_reference is not None

    @staticmethod
    def _vector(values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain three finite values")
        return result

    def update(self, mouth_position_m: Sequence[float], tool_position_m: Sequence[float],
               *, dt_sec: float) -> RelativeTrackingCommand:
        if not self.locked:
            return RelativeTrackingCommand(False, "tracking_session_not_locked", (0.0, 0.0, 0.0),
                                           (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        mouth = self._vector(mouth_position_m, "mouth_position_m")
        tool = self._vector(tool_position_m, "tool_position_m")
        delta = mouth - self._mouth_reference
        displacement = float(np.linalg.norm(delta))
        desired = self._tool_reference + delta
        if displacement > self.max_target_displacement_m:
            return self._halt("target_displacement_limit", desired, displacement)
        if float(np.linalg.norm(desired)) > self.max_tool_radius_m:
            return self._halt("tool_workspace_radius_limit", desired, displacement)
        velocity = self.gain * (desired - tool)
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_linear_speed_mps and speed > 0.0:
            velocity *= self.max_linear_speed_mps / speed
        dt = max(float(dt_sec), 1.0e-3)
        change = velocity - self._last_velocity
        max_change = self.max_linear_acceleration_mps2 * dt
        change_norm = float(np.linalg.norm(change))
        if change_norm > max_change and change_norm > 0.0:
            velocity = self._last_velocity + change * (max_change / change_norm)
        self._last_velocity = velocity
        return RelativeTrackingCommand(True, "ok", tuple(desired), tuple(velocity),
                                       (0.0, 0.0, 0.0), displacement)

    @staticmethod
    def _halt(reason: str, desired: np.ndarray, displacement: float) -> RelativeTrackingCommand:
        return RelativeTrackingCommand(False, reason, tuple(desired), (0.0, 0.0, 0.0),
                                       (0.0, 0.0, 0.0), displacement)
