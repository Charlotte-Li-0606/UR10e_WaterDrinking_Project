"""Motion backend contracts for the guarded tracking path.

All backends remain plan-only unless an execution sink is explicitly injected
and the caller arms it.  This keeps the one-shot feed-water path independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class MotionRequest:
    target_position: tuple[float, float, float]
    target_orientation_xyzw: tuple[float, float, float, float]
    plan_only: bool = True
    reason: str = "tracking"
    linear_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    preserve_orientation: bool = True


@dataclass(frozen=True)
class MotionResult:
    success: bool
    backend: str
    planned: bool
    executed: bool = False
    reason: str = ""
    halted: bool = False


class MotionBackend(Protocol):
    name: str

    def plan(self, request: MotionRequest) -> MotionResult: ...


class DryRunBackend:
    """Safe placeholder used until a backend is explicitly integrated."""

    def __init__(self, name: str) -> None:
        self.name = name

    def plan(self, request: MotionRequest) -> MotionResult:
        if not request.plan_only:
            return MotionResult(False, self.name, False, False, "execution_disabled_in_dry_run_backend")
        return MotionResult(True, self.name, True, False, "plan_only_placeholder")


class ServoBackend(DryRunBackend):
    """Cartesian Servo adapter with an explicit, normally-disarmed sink."""

    def __init__(self, command_sink=None, *, armed: bool = False) -> None:
        super().__init__("servo")
        self._command_sink = command_sink
        self.armed = bool(armed)

    def plan(self, request: MotionRequest) -> MotionResult:
        if request.plan_only:
            return MotionResult(True, self.name, True, False, "servo_plan_only")
        if not self.armed or self._command_sink is None:
            return MotionResult(False, self.name, False, False, "servo_backend_disarmed")
        self._command_sink(request)
        return MotionResult(True, self.name, False, True, "servo_command_published")

    def halt(self) -> None:
        if self._command_sink is not None and hasattr(self._command_sink, "halt"):
            self._command_sink.halt()
        self.armed = False


class CartesianBackend(DryRunBackend):
    def __init__(self) -> None:
        super().__init__("cartesian")


class PilzBackend(DryRunBackend):
    def __init__(self) -> None:
        super().__init__("pilz")


class OmplBackend(DryRunBackend):
    def __init__(self) -> None:
        super().__init__("ompl")
