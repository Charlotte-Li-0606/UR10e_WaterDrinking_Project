"""Backend selection policy for safe local correction and detours."""

from __future__ import annotations

from dataclasses import dataclass

from .motion_backend import MotionBackend, MotionRequest, MotionResult
from .safety_monitor import SafetyMonitor
from robot_layer.arm_ur10e.perception.mouth_target_tracker import MouthTargetTracker, TrackedMouthTarget


@dataclass(frozen=True)
class ReplanDecision:
    result: MotionResult
    backend_order: tuple[str, ...]
    reason: str


class ReplanningManager:
    """Try Cartesian/Pilz first and reserve OMPL for genuine detours."""

    def __init__(self, tracker: MouthTargetTracker, safety: SafetyMonitor,
                 cartesian: MotionBackend, pilz: MotionBackend, ompl: MotionBackend):
        self.tracker, self.safety = tracker, safety
        self.backends = (cartesian, pilz, ompl)

    # Select the first safe plan-only backend; never silently execute.
    def plan_recovery(self, target: TrackedMouthTarget, *, elapsed_sec: float = 0.0) -> ReplanDecision:
        check = self.safety.check_target(target, elapsed_sec=elapsed_sec)
        if not check.allowed:
            return ReplanDecision(MotionResult(False, "none", False, False, check.reason), (), check.reason)
        request = MotionRequest(tuple(float(v) for v in target.position), (0.0, 0.0, 0.0, 1.0), True,
                                "detour_recovery" if self.tracker.requires_replan(target) else "local_recovery")
        order = tuple(backend.name for backend in self.backends)
        for backend in self.backends:
            result = backend.plan(request)
            if result.success:
                return ReplanDecision(result, order, "first_valid_backend")
        return ReplanDecision(MotionResult(False, "none", False, False, "all_backends_failed"), order, "all_backends_failed")
