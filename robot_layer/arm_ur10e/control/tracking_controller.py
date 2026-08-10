"""Tracking controller with an explicit plan-only/real-servo mode switch."""

from __future__ import annotations

from dataclasses import dataclass

from .motion_backend import MotionBackend, MotionRequest, MotionResult
from .safety_monitor import SafetyMonitor
from robot_layer.arm_ur10e.perception.mouth_target_tracker import MouthTargetTracker, TrackedMouthTarget


@dataclass(frozen=True)
class TrackingDecision:
    result: MotionResult
    safety_reason: str
    requires_replan: bool


class TrackingController:
    """Turn a safe tracked target into a bounded Cartesian correction."""

    def __init__(self, tracker: MouthTargetTracker, safety: SafetyMonitor,
                 backend: MotionBackend, *, mode: str = "plan_only",
                 current_position_provider=None, correction_gain: float = 1.0):
        if mode not in {"plan_only", "real_servo"}:
            raise ValueError("mode must be plan_only or real_servo")
        self.tracker = tracker
        self.safety = safety
        self.backend = backend
        self.mode = mode
        self.current_position_provider = current_position_provider
        self.correction_gain = float(correction_gain)

    # Evaluate a target and ask the selected backend for a plan-only result.
    def correct(self, target: TrackedMouthTarget, *, elapsed_sec: float = 0.0,
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0)) -> TrackingDecision:
        safety = self.safety.check_target(target, elapsed_sec=elapsed_sec)
        replan = self.tracker.requires_replan(target)
        if not safety.allowed:
            return TrackingDecision(MotionResult(False, self.backend.name, False, False, safety.reason),
                                    safety.reason, replan)
        if self.current_position_provider is not None:
            current = self.current_position_provider()
            error = tuple(float(g - c) * self.correction_gain
                          for g, c in zip(target.position, current))
        else:
            error = (0.0, 0.0, 0.0)
        request = MotionRequest(tuple(float(v) for v in target.position), tuple(orientation_xyzw),
                                self.mode != "real_servo",
                                "large_displacement_replan" if replan else "local_tracking",
                                linear_velocity_mps=error,
                                preserve_orientation=True)
        result = self.backend.plan(request)
        return TrackingDecision(result, "ok", replan)

    # Immediately disarm the backend and publish its halt command when available.
    def halt(self, reason: str = "operator_stop") -> TrackingDecision:
        if hasattr(self.backend, "halt"):
            self.backend.halt()
        return TrackingDecision(MotionResult(False, self.backend.name, False, False, reason, True), reason, False)
