"""Target selection and stable pose queues for mouth perception.

The current MediaPipe node publishes one ``/detected_mouth_pose``.  This
module deliberately accepts that single pose as the ``center`` target today,
while keeping the data model ready for a future detector that reports several
face or mouth candidates with image-x coordinates.

No ROS node is created here.  The manager is reusable by the feeding tools and
by a future multi-target perception publisher, and is straightforward to test
without the simulator.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TARGET_SELECTIONS = frozenset({"left", "center", "right"})


def _jsonable_position(position: np.ndarray | Sequence[float]) -> list[float]:
    return [round(float(value), 6) for value in position]


@dataclass(frozen=True)
class TargetPoseSample:
    """One canonical ``base_link`` pose sample retained by a target queue."""

    position: np.ndarray
    timestamp_sec: float
    received_monotonic: float
    source_frame: str
    confidence: float | None = None
    image_x: float | None = None


class TargetPoseQueue:
    """Bounded, freshness-aware queue that returns a robust stable pose."""

    def __init__(
        self,
        *,
        max_queue_size: int = 20,
        stale_timeout_sec: float = 1.5,
        stability_window_sec: float = 1.0,
        max_position_std_m: float = 0.03,
        max_jump_m: float = 0.08,
        min_samples: int = 5,
    ) -> None:
        if max_queue_size < 1 or min_samples < 1:
            raise ValueError("max_queue_size and min_samples must be at least one")
        if min_samples > max_queue_size:
            raise ValueError("min_samples cannot exceed max_queue_size")
        if min(stale_timeout_sec, stability_window_sec, max_position_std_m, max_jump_m) <= 0.0:
            raise ValueError("pose queue time and distance limits must be positive")
        self.max_queue_size = int(max_queue_size)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.stability_window_sec = float(stability_window_sec)
        self.max_position_std_m = float(max_position_std_m)
        self.max_jump_m = float(max_jump_m)
        self.min_samples = int(min_samples)
        self._samples: deque[TargetPoseSample] = deque(maxlen=self.max_queue_size)
        self._last_rejection_reason = "mouth pose is missing"

    def clear(self) -> None:
        self._samples.clear()
        self._last_rejection_reason = "mouth pose is missing"

    def add_pose(
        self,
        pose: Sequence[float],
        *,
        timestamp_sec: float | None = None,
        source_frame: str = "base_link",
        confidence: float | None = None,
        image_x: float | None = None,
        received_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Validate and retain a canonical pose sample.

        A large jump is retained for diagnostics and rejected by
        :meth:`stable_pose`; this makes the result explain why the target was
        not considered stable instead of silently hiding an outlier.
        """
        try:
            position = np.asarray(pose, dtype=np.float64)
        except (TypeError, ValueError):
            self._last_rejection_reason = "mouth pose contains invalid coordinates"
            return {"success": False, "reason": self._last_rejection_reason}
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            self._last_rejection_reason = "mouth pose must contain three finite coordinates"
            return {"success": False, "reason": self._last_rejection_reason}
        frame = str(source_frame).strip().lstrip("/")
        if not frame:
            self._last_rejection_reason = "mouth pose source frame is missing"
            return {"success": False, "reason": self._last_rejection_reason}
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                self._last_rejection_reason = "mouth pose confidence is invalid"
                return {"success": False, "reason": self._last_rejection_reason}
            if not math.isfinite(confidence):
                self._last_rejection_reason = "mouth pose confidence is non-finite"
                return {"success": False, "reason": self._last_rejection_reason}
        if image_x is not None:
            try:
                image_x = float(image_x)
            except (TypeError, ValueError):
                self._last_rejection_reason = "mouth candidate image_x is invalid"
                return {"success": False, "reason": self._last_rejection_reason}
            if not math.isfinite(image_x):
                self._last_rejection_reason = "mouth candidate image_x is non-finite"
                return {"success": False, "reason": self._last_rejection_reason}
        now = time.monotonic() if received_monotonic is None else float(received_monotonic)
        stamp = time.time() if timestamp_sec is None else float(timestamp_sec)
        if not math.isfinite(now) or not math.isfinite(stamp):
            self._last_rejection_reason = "mouth pose timestamp is invalid"
            return {"success": False, "reason": self._last_rejection_reason}
        self._samples.append(
            TargetPoseSample(
                position=position.copy(),
                timestamp_sec=stamp,
                received_monotonic=now,
                source_frame=frame,
                confidence=confidence,
                image_x=image_x,
            )
        )
        self._last_rejection_reason = "mouth pose is available"
        return {"success": True, "reason": None, "queue_size": len(self._samples)}

    def _fresh_samples(self, *, now: float | None = None) -> list[TargetPoseSample]:
        current = time.monotonic() if now is None else float(now)
        return [
            sample
            for sample in self._samples
            if current - sample.received_monotonic <= self.stale_timeout_sec
        ]

    @staticmethod
    def _pose_dict(sample: TargetPoseSample, *, position: np.ndarray | None = None, age_sec: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "position": _jsonable_position(sample.position if position is None else position),
            "frame_id": "base_link",
            "source_frame": sample.source_frame,
            "source_stamp_sec": round(sample.timestamp_sec, 6),
            "received_age_sec": round(max(0.0, age_sec), 4),
        }
        if sample.confidence is not None:
            result["confidence"] = round(sample.confidence, 6)
        if sample.image_x is not None:
            result["image_x"] = round(sample.image_x, 3)
        return result

    def latest(self) -> dict[str, Any]:
        """Return the newest fresh sample or a structured stale/missing error."""
        now = time.monotonic()
        fresh = self._fresh_samples(now=now)
        if not fresh:
            reason = "mouth pose stale" if self._samples else self._last_rejection_reason
            return {"success": False, "stable": False, "reason": reason}
        sample = fresh[-1]
        return {
            "success": True,
            "stable": False,
            "reason": None,
            "pose": self._pose_dict(sample, age_sec=now - sample.received_monotonic),
            "num_samples": len(fresh),
        }

    def stable_pose(self) -> dict[str, Any]:
        """Return the median stable pose or a structured rejection reason."""
        now = time.monotonic()
        fresh = self._fresh_samples(now=now)
        if not fresh:
            reason = "mouth pose stale" if self._samples else self._last_rejection_reason
            return {"success": False, "stable": False, "reason": reason}
        newest = fresh[-1]
        recent = [
            sample
            for sample in fresh
            if newest.received_monotonic - sample.received_monotonic <= self.stability_window_sec
        ]
        if len(recent) < self.min_samples:
            return {
                "success": False,
                "stable": False,
                "reason": "not enough recent mouth pose samples",
                "num_samples": len(recent),
                "required_samples": self.min_samples,
            }
        points = np.asarray([sample.position for sample in recent], dtype=np.float64)
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        max_jump = float(np.max(jumps)) if jumps.size else 0.0
        if max_jump > self.max_jump_m:
            return {
                "success": False,
                "stable": False,
                "reason": "mouth pose jump exceeds limit",
                "max_jump_m": round(max_jump, 6),
                "limit_m": self.max_jump_m,
                "num_samples": len(recent),
            }
        selected = np.median(points, axis=0)
        position_std = float(np.sqrt(np.mean(np.sum((points - selected) ** 2, axis=1))))
        if position_std > self.max_position_std_m:
            return {
                "success": False,
                "stable": False,
                "reason": "mouth pose is unstable",
                "position_std_m": round(position_std, 6),
                "limit_m": self.max_position_std_m,
                "num_samples": len(recent),
            }
        confidence_values = [sample.confidence for sample in recent if sample.confidence is not None]
        pose = self._pose_dict(newest, position=selected, age_sec=now - newest.received_monotonic)
        if confidence_values:
            pose["confidence"] = round(float(np.median(confidence_values)), 6)
        return {
            "success": True,
            "stable": True,
            "reason": None,
            "pose": pose,
            "num_samples": len(recent),
            "age_sec": round(max(0.0, now - newest.received_monotonic), 4),
            "position_std_m": round(position_std, 6),
            "max_jump_m": round(max_jump, 6),
            "observed_window_sec": round(newest.received_monotonic - recent[0].received_monotonic, 4),
        }

    def is_stale(self) -> bool:
        return not bool(self.latest().get("success"))

    def is_stable(self) -> bool:
        return bool(self.stable_pose().get("success"))

    def state(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "queue_size": len(self._samples),
            "stale": not bool(latest.get("success")),
            "latest": latest,
        }


class ActiveTargetManager:
    """Lock a logical person target and expose only its stable pose queue."""

    def __init__(self, **queue_kwargs: Any) -> None:
        self._queues = {label: TargetPoseQueue(**queue_kwargs) for label in sorted(TARGET_SELECTIONS)}
        self._active_target_label = "center"
        self._active_reference_position: np.ndarray | None = None
        self._active_reference_image_x: float | None = None
        self._last_candidate_count = 0
        self._last_candidate_received_monotonic: float | None = None

    @property
    def active_target_label(self) -> str:
        return self._active_target_label

    @staticmethod
    def _validate_selection(selection: str) -> str:
        normalized = str(selection).strip().lower()
        if normalized not in TARGET_SELECTIONS:
            raise ValueError("selection must be one of: left, center, right")
        return normalized

    def select_target(self, selection: str) -> dict[str, Any]:
        selected = self._validate_selection(selection)
        # A different requested person must not inherit the previous person's
        # nearest-neighbour lock.
        if selected != self._active_target_label:
            self._active_reference_position = None
            self._active_reference_image_x = None
        self._active_target_label = selected
        return {
            "success": True,
            "reason": None,
            "active_target_label": self._active_target_label,
            "active_target_id": self._active_target_label,
        }

    @staticmethod
    def _candidate_mapping(candidate: Mapping[str, Any] | Sequence[float]) -> dict[str, Any] | None:
        raw: Mapping[str, Any]
        if isinstance(candidate, Mapping):
            raw = candidate
            position = raw.get("position")
        else:
            raw = {}
            position = candidate
        # FeedingSkillLibrary transforms ROS poses with NumPy before calling
        # this manager.  Accept both ordinary Python coordinate sequences and
        # NumPy arrays; reject textual values before numerical coercion.
        if isinstance(position, (str, bytes)):
            return None
        try:
            values = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return None
        image_x = raw.get("image_x")
        image_center_x = raw.get("image_center_x")
        try:
            image_x = None if image_x is None else float(image_x)
            image_center_x = None if image_center_x is None else float(image_center_x)
        except (TypeError, ValueError):
            return None
        if (image_x is not None and not math.isfinite(image_x)) or (
            image_center_x is not None and not math.isfinite(image_center_x)
        ):
            return None
        return {
            "position": values,
            "timestamp_sec": raw.get("timestamp_sec"),
            "source_frame": raw.get("source_frame", "base_link"),
            "confidence": raw.get("confidence"),
            "image_x": image_x,
            "image_center_x": image_center_x,
            "received_monotonic": raw.get("received_monotonic"),
        }

    def _nearest_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if self._active_reference_position is not None:
            return min(
                candidates,
                key=lambda candidate: float(np.linalg.norm(candidate["position"] - self._active_reference_position)),
            )
        if self._active_reference_image_x is not None:
            candidates_with_x = [candidate for candidate in candidates if candidate["image_x"] is not None]
            if candidates_with_x:
                return min(
                    candidates_with_x,
                    key=lambda candidate: abs(float(candidate["image_x"]) - self._active_reference_image_x),
                )
        return candidates[0]

    @staticmethod
    def _assigned_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if len(candidates) == 1:
            return {"center": candidates[0]}
        # Image x is the future multi-face contract. If an upstream detector
        # lacks it temporarily, use base-frame X as a deterministic fallback.
        ordered = sorted(
            candidates,
            key=lambda candidate: float(candidate["image_x"])
            if candidate["image_x"] is not None
            else float(candidate["position"][0]),
        )
        center_x_values = [candidate["image_x"] for candidate in ordered if candidate["image_x"] is not None]
        if center_x_values:
            declared_centers = [
                candidate["image_center_x"]
                for candidate in ordered
                if candidate.get("image_center_x") is not None
            ]
            image_center = (
                float(np.median(np.asarray(declared_centers, dtype=np.float64)))
                if declared_centers
                else 0.5 * (float(min(center_x_values)) + float(max(center_x_values)))
            )
            center = min(
                ordered,
                key=lambda candidate: abs(
                    float(candidate["image_x"] if candidate["image_x"] is not None else image_center) - image_center
                ),
            )
        else:
            center = ordered[len(ordered) // 2]
        return {"left": ordered[0], "center": center, "right": ordered[-1]}

    def _append(self, label: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
        result = self._queues[label].add_pose(
            candidate["position"],
            timestamp_sec=candidate.get("timestamp_sec"),
            source_frame=str(candidate.get("source_frame", "base_link")),
            confidence=candidate.get("confidence"),
            image_x=candidate.get("image_x"),
            received_monotonic=candidate.get("received_monotonic"),
        )
        return result

    def update_from_detection(
        self,
        detection: Mapping[str, Any] | Sequence[float] | None = None,
        *,
        candidates: Sequence[Mapping[str, Any] | Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        """Update queues from one current pose or future multi-face candidates."""
        raw_candidates = list(candidates) if candidates is not None else ([] if detection is None else [detection])
        normalized = [candidate for item in raw_candidates if (candidate := self._candidate_mapping(item)) is not None]
        if not normalized:
            self._last_candidate_count = 0
            self._last_candidate_received_monotonic = time.monotonic()
            return {
                "success": False,
                "reason": "no valid mouth candidates were supplied",
                "active_target_label": self._active_target_label,
            }
        self._last_candidate_count = len(normalized)
        self._last_candidate_received_monotonic = time.monotonic()
        assigned = self._assigned_candidates(normalized)
        active_candidate: dict[str, Any] | None
        if len(normalized) > 1 and self._active_reference_position is not None:
            active_candidate = self._nearest_candidate(normalized)
        else:
            active_candidate = assigned.get(self._active_target_label)

        updated_labels: list[str] = []
        # Keep non-active logical queues warm for later explicit selection.
        for label, candidate in assigned.items():
            if label == self._active_target_label:
                continue
            if self._append(label, candidate).get("success"):
                updated_labels.append(label)
        if active_candidate is not None and self._append(self._active_target_label, active_candidate).get("success"):
            updated_labels.append(self._active_target_label)
            self._active_reference_position = np.asarray(active_candidate["position"], dtype=np.float64).copy()
            image_x = active_candidate.get("image_x")
            self._active_reference_image_x = None if image_x is None else float(image_x)

        return {
            "success": active_candidate is not None,
            "reason": None if active_candidate is not None else "selected target is not currently visible",
            "active_target_label": self._active_target_label,
            "candidate_count": len(normalized),
            "updated_labels": sorted(set(updated_labels)),
        }

    def get_active_latest_pose(self) -> dict[str, Any]:
        result = self._queues[self._active_target_label].latest()
        result.update(
            {
                "active_target_label": self._active_target_label,
                "active_target_id": self._active_target_label,
            }
        )
        return result

    def get_active_stable_pose(self) -> dict[str, Any]:
        result = self._queues[self._active_target_label].stable_pose()
        result.update(
            {
                "active_target_label": self._active_target_label,
                "active_target_id": self._active_target_label,
            }
        )
        return result

    def get_single_visible_stable_pose(self) -> dict[str, Any]:
        """Return the one stable visible candidate for explicit search fallback.

        This does not alter ``active_target_label``.  Callers must make any
        requested-target fallback explicit in their structured result.
        """
        if self._last_candidate_count != 1 or self._last_candidate_received_monotonic is None:
            return {
                "success": False,
                "stable": False,
                "reason": "a single visible mouth candidate is not available",
            }
        age = time.monotonic() - self._last_candidate_received_monotonic
        center_queue = self._queues["center"]
        if age > center_queue.stale_timeout_sec:
            return {
                "success": False,
                "stable": False,
                "reason": "single visible mouth candidate is stale",
            }
        result = center_queue.stable_pose()
        result.update(
            {
                "active_target_label": "center",
                "active_target_id": "center",
                "candidate_count": 1,
            }
        )
        return result

    def get_state(self) -> dict[str, Any]:
        return {
            "success": True,
            "reason": None,
            "active_target_label": self._active_target_label,
            "active_target_id": self._active_target_label,
            "last_candidate_count": self._last_candidate_count,
            "last_candidate_age_sec": (
                None
                if self._last_candidate_received_monotonic is None
                else round(max(0.0, time.monotonic() - self._last_candidate_received_monotonic), 4)
            ),
            "targets": {label: self._queues[label].state() for label in sorted(TARGET_SELECTIONS)},
            "active_stable_pose": self.get_active_stable_pose(),
        }

    def reset_active_target(self) -> dict[str, Any]:
        for queue in self._queues.values():
            queue.clear()
        self._active_target_label = "center"
        self._active_reference_position = None
        self._active_reference_image_x = None
        self._last_candidate_count = 0
        self._last_candidate_received_monotonic = None
        return {
            "success": True,
            "reason": None,
            "active_target_label": self._active_target_label,
            "active_target_id": self._active_target_label,
        }
