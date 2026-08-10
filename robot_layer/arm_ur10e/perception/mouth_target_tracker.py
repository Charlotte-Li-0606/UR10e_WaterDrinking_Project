"""Read-only moving-mouth tracker built on the existing mouth observations.

The tracker contains no ROS publishers, MoveIt clients, or robot commands.  It
is intentionally usable in plan-only tests before it is connected to motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence

import numpy as np


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    LOCKED = "LOCKED"
    TRACKING = "TRACKING"
    LOST = "LOST"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class TrackedMouthTarget:
    position: np.ndarray
    velocity: np.ndarray
    timestamp_sec: float
    confidence: float
    target_id: str
    state: TrackingState
    age_sec: float
    displacement_m: float
    jitter_m: float


class MouthTargetTracker:
    """Filter one selected mouth target and fail closed on stale data."""

    def __init__(
        self,
        *,
        target_timeout_sec: float = 0.25,
        lost_timeout_sec: float = 0.50,
        min_confidence: float = 0.60,
        correction_distance_m: float = 0.010,
        replan_distance_m: float = 0.090,
        smoothing_alpha: float = 0.35,
        max_history: int = 20,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if target_timeout_sec <= 0.0 or lost_timeout_sec < target_timeout_sec:
            raise ValueError("timeouts must be positive and ordered")
        self.target_timeout_sec = float(target_timeout_sec)
        self.lost_timeout_sec = float(lost_timeout_sec)
        self.min_confidence = float(min_confidence)
        self.correction_distance_m = float(correction_distance_m)
        self.replan_distance_m = float(replan_distance_m)
        self.alpha = float(smoothing_alpha)
        self._history: list[np.ndarray] = []
        self._position: Optional[np.ndarray] = None
        self._reference: Optional[np.ndarray] = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._timestamp: Optional[float] = None
        self._confidence = 0.0
        self._target_id = "center"
        self._state = TrackingState.SEARCHING
        self._max_history = max(3, int(max_history))

    @property
    def state(self) -> TrackingState:
        return self._state

    def reset(self) -> None:
        """Clear the reference and begin a new tracking session."""
        self._history.clear()
        self._position = self._reference = self._timestamp = None
        self._velocity[:] = 0.0
        self._confidence = 0.0
        self._state = TrackingState.SEARCHING

    def begin_session(self) -> None:
        """Explicitly reset before a new operator/test tracking session."""
        self.reset()

    def update(
        self,
        position: Sequence[float],
        *,
        timestamp_sec: float,
        confidence: float = 1.0,
        target_id: str = "center",
    ) -> TrackedMouthTarget:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position must contain three finite values")
        stamp = float(timestamp_sec)
        confidence = float(confidence)
        if not math.isfinite(stamp) or not math.isfinite(confidence):
            raise ValueError("timestamp and confidence must be finite")
        if confidence < self.min_confidence:
            self._state = TrackingState.LOST if self._position is not None else TrackingState.SEARCHING
            return self.snapshot(now_sec=stamp)
        # A recovered target after loss starts from a fresh reference.
        if self._state == TrackingState.LOST:
            self.reset()
        if self._timestamp is not None and stamp < self._timestamp:
            return self.snapshot(now_sec=self._timestamp)
        if self._position is None:
            filtered = point.copy()
            self._reference = point.copy()
            self._state = TrackingState.LOCKED
        else:
            dt = max(stamp - self._timestamp, 1e-6)
            raw_displacement = float(np.linalg.norm(point - self._reference))
            filtered = self.alpha * point + (1.0 - self.alpha) * self._position
            self._velocity = self.alpha * ((filtered - self._position) / dt) + (1.0 - self.alpha) * self._velocity
            self._state = TrackingState.TRACKING if raw_displacement < self.replan_distance_m else TrackingState.ABORTED
        self._position = filtered
        self._timestamp = stamp
        self._confidence = confidence
        self._target_id = str(target_id)
        self._history.append(filtered.copy())
        del self._history[:-self._max_history]
        return self.snapshot(now_sec=stamp)

    def snapshot(self, *, now_sec: float) -> TrackedMouthTarget:
        if self._position is None or self._timestamp is None:
            return TrackedMouthTarget(np.zeros(3), np.zeros(3), float(now_sec), 0.0,
                                     self._target_id, self._state, math.inf, 0.0, 0.0)
        age = max(0.0, float(now_sec) - self._timestamp)
        if age > self.lost_timeout_sec:
            self._state = TrackingState.LOST
        elif age > self.target_timeout_sec and self._state != TrackingState.ABORTED:
            self._state = TrackingState.LOST
        displacement = float(np.linalg.norm(self._position - self._reference)) if self._reference is not None else 0.0
        jitter = float(np.mean(np.linalg.norm(np.asarray(self._history) - self._position, axis=1))) if self._history else 0.0
        return TrackedMouthTarget(self._position.copy(), self._velocity.copy(), self._timestamp,
                                  self._confidence, self._target_id, self._state, age, displacement, jitter)

    def requires_replan(self, target: Optional[TrackedMouthTarget] = None) -> bool:
        target = target or self.snapshot(now_sec=self._timestamp or 0.0)
        return target.displacement_m >= self.replan_distance_m or target.state == TrackingState.ABORTED
