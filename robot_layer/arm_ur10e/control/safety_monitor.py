"""Fail-closed checks shared by future tracking controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from robot_layer.arm_ur10e.perception.mouth_target_tracker import TrackedMouthTarget, TrackingState


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str


class SafetyMonitor:
    """Validate target freshness, workspace, speed, and flange tilt."""

    def __init__(self, *, workspace_min=(-1.3, -1.3, -0.2), workspace_max=(1.3, 1.3, 1.8),
                 max_speed_mps=0.02, max_acceleration_mps2=0.10,
                 max_tilt_deg=5.0, max_tracking_sec=15.0):
        self.workspace_min = tuple(float(v) for v in workspace_min)
        self.workspace_max = tuple(float(v) for v in workspace_max)
        self.max_speed_mps = float(max_speed_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_tracking_sec = float(max_tracking_sec)
        self._last_velocity = None
        self._last_timestamp = None

    # Reject targets that are stale, lost, out of bounds, or too fast.
    def check_target(self, target: TrackedMouthTarget, *, elapsed_sec: float) -> SafetyDecision:
        if target.state == TrackingState.LOST:
            return SafetyDecision(False, "target_state:LOST")
        # ABORTED denotes a displacement-triggered recovery, not permission to execute.
        if target.state == TrackingState.ABORTED:
            return SafetyDecision(True, "replan_required")
        if not math.isfinite(target.age_sec) or target.age_sec > 0.50:
            return SafetyDecision(False, "target_stale")
        if any(p < lo or p > hi for p, lo, hi in zip(target.position, self.workspace_min, self.workspace_max)):
            return SafetyDecision(False, "target_outside_workspace")
        if math.sqrt(float(target.velocity @ target.velocity)) > self.max_speed_mps:
            return SafetyDecision(False, "target_speed_limit")
        if self._last_velocity is not None and self._last_timestamp is not None:
            dt = float(target.timestamp_sec) - self._last_timestamp
            if dt > 1.0e-3:
                acceleration = math.sqrt(float(((target.velocity - self._last_velocity) @
                                                (target.velocity - self._last_velocity)))) / dt
                if acceleration > self.max_acceleration_mps2:
                    return SafetyDecision(False, "target_acceleration_limit")
        self._last_velocity = target.velocity.copy()
        self._last_timestamp = float(target.timestamp_sec)
        if elapsed_sec > self.max_tracking_sec:
            return SafetyDecision(False, "tracking_duration_limit")
        return SafetyDecision(True, "ok")

    # Reject a sampled tool axis unless it remains close to base -Z.
    def check_flange_tilt(self, downward_axis, *, base_down=(0.0, 0.0, -1.0)) -> SafetyDecision:
        axis = tuple(float(v) for v in downward_axis)
        norm = math.sqrt(sum(v * v for v in axis))
        if norm == 0.0:
            return SafetyDecision(False, "zero_flange_axis")
        dot = sum(a * b for a, b in zip(axis, base_down)) / norm
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        return SafetyDecision(angle <= self.max_tilt_deg, "ok" if angle <= self.max_tilt_deg else "flange_tilt_limit")
