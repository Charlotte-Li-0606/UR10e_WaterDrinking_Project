"""Robust local mouth tracking for the opt-in continuous Servo mode."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Sequence

import numpy as np


class ContinuousTrackingState(str, Enum):
    NO_TARGET = "NO_TARGET"
    SEARCHING = "SEARCHING"
    PROVISIONAL_TARGET = "PROVISIONAL_TARGET"
    TRACKING = "TRACKING"
    STABLE_TARGET = "STABLE_TARGET"
    HOLDING = "HOLDING"
    TARGET_LOST = "TARGET_LOST"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class ContinuousMouthObservation:
    position_m: np.ndarray
    source_timestamp_sec: float
    received_monotonic_sec: float
    depth_m: float
    confidence: float
    target_id: str
    sequence_id: int = 0
    frame_id: str = "base_link"
    raw_position_m: np.ndarray | None = None


@dataclass(frozen=True)
class ContinuousMouthTarget:
    available: bool
    position_m: np.ndarray
    predicted_position_m: np.ndarray
    velocity_mps: np.ndarray
    source_timestamp_sec: float
    age_sec: float
    confidence: float
    target_id: str
    state: ContinuousTrackingState
    provisional: bool
    stable: bool
    sample_count: int
    spread_m: float
    prediction_m: float
    reason: str | None
    sequence_id: int = 0
    frame_id: str = "base_link"
    received_monotonic_sec: float = 0.0
    raw_position_m: np.ndarray | None = None
    filter_alpha: float = 1.0


@dataclass(frozen=True)
class InitialAcquisitionDecision:
    complete: bool
    target: ContinuousMouthTarget
    state: ContinuousTrackingState
    active_search_required: bool
    reason: str


class InitialTargetAcquirer:
    """Accept the first fresh valid target, including a provisional target."""

    def __init__(self, *, started_monotonic_sec: float, timeout_sec: float = 3.0) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive and finite")
        self.started_monotonic_sec = float(started_monotonic_sec)
        self.timeout_sec = float(timeout_sec)

    def evaluate(
        self,
        target: ContinuousMouthTarget,
        *,
        now_monotonic_sec: float,
    ) -> InitialAcquisitionDecision:
        elapsed = float(now_monotonic_sec) - self.started_monotonic_sec
        if target.available and target.stable:
            return InitialAcquisitionDecision(
                True,
                target,
                ContinuousTrackingState.STABLE_TARGET,
                False,
                "stable_target_acquired",
            )
        if target.available:
            return InitialAcquisitionDecision(
                True,
                target,
                ContinuousTrackingState.PROVISIONAL_TARGET,
                False,
                "provisional_target_acquired",
            )
        if elapsed < self.timeout_sec:
            return InitialAcquisitionDecision(
                False,
                target,
                ContinuousTrackingState.SEARCHING,
                False,
                "waiting_for_valid_target",
            )
        return InitialAcquisitionDecision(
            True,
            target,
            ContinuousTrackingState.NO_TARGET,
            True,
            "no_valid_observation_after_acquisition_timeout",
        )


class ContinuousMouthTracker:
    """Filter valid RGB-D mouth observations without inventing a target."""

    def __init__(
        self,
        *,
        target_timeout_sec: float = 0.30,
        lost_target_timeout_sec: float = 0.50,
        minimum_confidence: float = 0.60,
        stable_sample_count: int = 3,
        stable_max_spread_m: float = 0.025,
        prediction_horizon_sec: float = 0.10,
        maximum_prediction_m: float = 0.020,
        minimum_depth_m: float = 0.05,
        maximum_depth_m: float = 1.30,
        history_size: int = 5,
        filter_time_constant_sec: float = 0.08,
        filter_min_alpha: float = 0.20,
        filter_max_alpha: float = 0.85,
        filter_full_response_displacement_m: float = 0.10,
    ) -> None:
        if not 0.0 < target_timeout_sec <= lost_target_timeout_sec:
            raise ValueError("target timeouts must be positive and ordered")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if stable_sample_count < 2:
            raise ValueError("stable_sample_count must be at least two")
        if stable_max_spread_m <= 0.0:
            raise ValueError("stable_max_spread_m must be positive")
        if not 0.0 <= prediction_horizon_sec <= 0.50:
            raise ValueError("prediction_horizon_sec must be in [0, 0.5]")
        if maximum_prediction_m < 0.0:
            raise ValueError("maximum_prediction_m must be non-negative")
        if not 0.0 < minimum_depth_m < maximum_depth_m:
            raise ValueError("depth limits must be positive and ordered")
        if not math.isfinite(filter_time_constant_sec) or filter_time_constant_sec <= 0.0:
            raise ValueError("filter_time_constant_sec must be positive and finite")
        if not 0.0 < filter_min_alpha <= filter_max_alpha <= 1.0:
            raise ValueError("filter alpha limits must satisfy 0 < min <= max <= 1")
        if (
            not math.isfinite(filter_full_response_displacement_m)
            or filter_full_response_displacement_m <= 0.0
        ):
            raise ValueError(
                "filter_full_response_displacement_m must be positive and finite"
            )
        self.target_timeout_sec = float(target_timeout_sec)
        self.lost_target_timeout_sec = float(lost_target_timeout_sec)
        self.minimum_confidence = float(minimum_confidence)
        self.stable_sample_count = int(stable_sample_count)
        self.stable_max_spread_m = float(stable_max_spread_m)
        self.prediction_horizon_sec = float(prediction_horizon_sec)
        self.maximum_prediction_m = float(maximum_prediction_m)
        self.minimum_depth_m = float(minimum_depth_m)
        self.maximum_depth_m = float(maximum_depth_m)
        self.filter_time_constant_sec = float(filter_time_constant_sec)
        self.filter_min_alpha = float(filter_min_alpha)
        self.filter_max_alpha = float(filter_max_alpha)
        self.filter_full_response_displacement_m = float(
            filter_full_response_displacement_m
        )
        # The deque is diagnostics/filter history only. Motion always consumes
        # _latest_observation, so old camera targets can never form a FIFO.
        self._observations: deque[ContinuousMouthObservation] = deque(
            maxlen=max(self.stable_sample_count, int(history_size))
        )
        self._latest_observation: ContinuousMouthObservation | None = None
        self._last_filter_alpha = 1.0
        self._next_sequence_id = 1
        self._lock = threading.RLock()
        self._state = ContinuousTrackingState.NO_TARGET
        self._last_rejection: str | None = None

    @property
    def state(self) -> ContinuousTrackingState:
        return self._state

    def reset(self, *, searching: bool = False) -> None:
        with self._lock:
            self._observations.clear()
            self._latest_observation = None
            self._last_filter_alpha = 1.0
            self._next_sequence_id = 1
            self._state = (
                ContinuousTrackingState.SEARCHING
                if searching
                else ContinuousTrackingState.NO_TARGET
            )
            self._last_rejection = None

    @staticmethod
    def _vector(values: Sequence[float]) -> np.ndarray:
        point = np.asarray(values, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position_m must contain three finite values")
        return point

    def add_observation(
        self,
        position_m: Sequence[float],
        *,
        source_timestamp_sec: float,
        received_monotonic_sec: float,
        depth_m: float,
        confidence: float = 1.0,
        target_id: str = "center",
        sequence_id: int | None = None,
        frame_id: str = "base_link",
    ) -> tuple[bool, str]:
        """Accept one timestamped RGB-D observation or reject it explicitly."""
        try:
            point = self._vector(position_m)
            source_stamp = float(source_timestamp_sec)
            received = float(received_monotonic_sec)
            depth = float(depth_m)
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            self._last_rejection = f"invalid_observation:{exc}"
            return False, self._last_rejection
        if not all(
            math.isfinite(value)
            for value in (source_stamp, received, depth, confidence_value)
        ) or source_stamp <= 0.0:
            self._last_rejection = "invalid_timestamp_or_numeric_field"
            return False, self._last_rejection
        if not self.minimum_depth_m <= depth <= self.maximum_depth_m:
            self._last_rejection = "invalid_depth"
            return False, self._last_rejection
        if not 0.0 <= confidence_value <= 1.0:
            self._last_rejection = "invalid_confidence"
            return False, self._last_rejection
        if confidence_value < self.minimum_confidence:
            self._last_rejection = "confidence_below_minimum"
            return False, self._last_rejection
        normalized_frame = str(frame_id).strip().lstrip("/")
        if not normalized_frame:
            self._last_rejection = "invalid_frame_id"
            return False, self._last_rejection
        with self._lock:
            previous = self._latest_observation
            if previous is not None and (
                source_stamp <= previous.source_timestamp_sec
                or received <= previous.received_monotonic_sec
            ):
                self._last_rejection = "non_monotonic_timestamp"
                return False, self._last_rejection
            selected_sequence = (
                self._next_sequence_id if sequence_id is None else int(sequence_id)
            )
            if selected_sequence <= 0 or (
                previous is not None and selected_sequence <= previous.sequence_id
            ):
                self._last_rejection = "non_monotonic_sequence_id"
                return False, self._last_rejection
            if previous is None:
                filtered = point.copy()
                alpha = 1.0
            else:
                dt = max(1.0e-6, source_stamp - previous.source_timestamp_sec)
                timestamp_alpha = 1.0 - math.exp(
                    -dt / self.filter_time_constant_sec
                )
                displacement = float(
                    np.linalg.norm(point - np.asarray(previous.raw_position_m))
                )
                motion_fraction = min(
                    1.0,
                    displacement / self.filter_full_response_displacement_m,
                )
                motion_alpha = self.filter_min_alpha + motion_fraction * (
                    self.filter_max_alpha - self.filter_min_alpha
                )
                alpha = min(
                    self.filter_max_alpha,
                    max(self.filter_min_alpha, timestamp_alpha, motion_alpha),
                )
                filtered = previous.position_m + alpha * (
                    point - previous.position_m
                )
            observation = ContinuousMouthObservation(
                position_m=filtered.copy(),
                source_timestamp_sec=source_stamp,
                received_monotonic_sec=received,
                depth_m=depth,
                confidence=confidence_value,
                target_id=str(target_id),
                sequence_id=selected_sequence,
                frame_id=normalized_frame,
                raw_position_m=point.copy(),
            )
            self._latest_observation = observation
            self._observations.append(observation)
            self._last_filter_alpha = float(alpha)
            self._next_sequence_id = selected_sequence + 1
            self._last_rejection = None
        return True, "accepted"

    @staticmethod
    def _estimate_position(points: np.ndarray) -> np.ndarray:
        # Two samples use their average; three or more use a robust median.
        if len(points) == 2:
            return np.mean(points, axis=0)
        return np.median(points, axis=0)

    @staticmethod
    def _estimate_velocity(
        observations: Sequence[ContinuousMouthObservation],
    ) -> np.ndarray:
        velocities: list[np.ndarray] = []
        for before, after in zip(observations, observations[1:]):
            dt = after.source_timestamp_sec - before.source_timestamp_sec
            if dt > 1.0e-3:
                velocities.append((after.position_m - before.position_m) / dt)
        return (
            np.median(np.asarray(velocities), axis=0)
            if velocities
            else np.zeros(3, dtype=np.float64)
        )

    def target(self, *, now_monotonic_sec: float) -> ContinuousMouthTarget:
        now = float(now_monotonic_sec)
        if not math.isfinite(now):
            raise ValueError("now_monotonic_sec must be finite")
        with self._lock:
            latest = self._latest_observation
            observations = tuple(self._observations)
            state = self._state
            last_rejection = self._last_rejection
            filter_alpha = self._last_filter_alpha
        if latest is None:
            return self._empty_target(state, last_rejection)
        age = max(0.0, now - latest.received_monotonic_sec)
        if age > self.lost_target_timeout_sec:
            self._state = ContinuousTrackingState.TARGET_LOST
            return self._empty_target(
                self._state,
                "target_lost_timeout",
                age_sec=age,
                target_id=latest.target_id,
            )
        if age > self.target_timeout_sec:
            # A short RGB-D dropout is common at close range.  Mark this as a
            # bounded hold/reacquisition interval instead of declaring target
            # loss immediately.  The motion layer must command zero velocity
            # while this unavailable target is returned.
            self._state = ContinuousTrackingState.TRACKING
            return self._empty_target(
                self._state,
                "target_stale_grace",
                age_sec=age,
                target_id=latest.target_id,
            )

        recent = [
            observation
            for observation in observations
            if now - observation.received_monotonic_sec
            <= self.lost_target_timeout_sec
            and observation.target_id == latest.target_id
        ]
        points = np.asarray([observation.position_m for observation in recent])
        # The mailbox target is always the newest filtered sample. History is
        # used only for spread, stability, velocity, and diagnostics.
        position = latest.position_m.copy()
        distances = np.linalg.norm(points - position, axis=1)
        spread = float(np.max(distances)) if len(distances) else math.inf
        confidence = float(np.median([item.confidence for item in recent]))
        stable = bool(
            len(recent) >= self.stable_sample_count
            and spread <= self.stable_max_spread_m
            and confidence >= self.minimum_confidence
        )
        provisional = not stable
        velocity = self._estimate_velocity(recent)
        prediction = np.zeros(3, dtype=np.float64)
        if stable and confidence >= max(0.75, self.minimum_confidence):
            prediction = velocity * self.prediction_horizon_sec
            magnitude = float(np.linalg.norm(prediction))
            if magnitude > self.maximum_prediction_m and magnitude > 0.0:
                prediction *= self.maximum_prediction_m / magnitude
        predicted = position + prediction
        self._state = (
            ContinuousTrackingState.STABLE_TARGET
            if stable
            else ContinuousTrackingState.PROVISIONAL_TARGET
        )
        return ContinuousMouthTarget(
            available=True,
            position_m=position,
            predicted_position_m=predicted,
            velocity_mps=velocity,
            source_timestamp_sec=latest.source_timestamp_sec,
            age_sec=age,
            confidence=confidence,
            target_id=latest.target_id,
            state=self._state,
            provisional=provisional,
            stable=stable,
            sample_count=len(recent),
            spread_m=spread,
            prediction_m=float(np.linalg.norm(prediction)),
            reason=None,
            sequence_id=int(latest.sequence_id),
            frame_id=latest.frame_id,
            received_monotonic_sec=float(latest.received_monotonic_sec),
            raw_position_m=np.asarray(latest.raw_position_m).copy(),
            filter_alpha=float(filter_alpha),
        )

    def mark_holding(self) -> None:
        if self._state == ContinuousTrackingState.STABLE_TARGET:
            self._state = ContinuousTrackingState.HOLDING

    def abort(self, reason: str = "aborted") -> ContinuousMouthTarget:
        self._state = ContinuousTrackingState.ABORTED
        return self._empty_target(self._state, str(reason))

    @staticmethod
    def _empty_target(
        state: ContinuousTrackingState,
        reason: str | None,
        *,
        age_sec: float = math.inf,
        target_id: str = "center",
    ) -> ContinuousMouthTarget:
        zero = np.zeros(3, dtype=np.float64)
        return ContinuousMouthTarget(
            available=False,
            position_m=zero.copy(),
            predicted_position_m=zero.copy(),
            velocity_mps=zero.copy(),
            source_timestamp_sec=0.0,
            age_sec=age_sec,
            confidence=0.0,
            target_id=target_id,
            state=state,
            provisional=False,
            stable=False,
            sample_count=0,
            spread_m=math.inf,
            prediction_m=0.0,
            reason=reason or "no_valid_target",
        )
