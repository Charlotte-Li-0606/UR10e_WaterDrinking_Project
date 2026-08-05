"""Fail-closed mouth-candidate tracking for the guarded real UR10e route.

The MediaPipe node publishes every visible mouth in image-x order.  This
module turns an explicit ``left``/``center``/``right`` request into one stable
3D target and then locks that person by nearest-neighbour position.  A large
jump or an ambiguous match invalidates the observation instead of silently
switching people.

This module is deliberately ROS-free so the selection and identity policy can
be unit-tested without MoveIt, a camera, or robot hardware.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TARGET_SELECTIONS = frozenset({"left", "center", "right"})


def validate_target_selection(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("target_selection must be one of: left, center, right")
    selection = value.strip().lower()
    if selection not in TARGET_SELECTIONS:
        raise ValueError("target_selection must be one of: left, center, right")
    return selection


def _finite_vector(value: Any, length: int) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        return None
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def _mean(points: Sequence[Sequence[float]]) -> list[float]:
    return [sum(float(point[axis]) for point in points) / len(points) for axis in range(3)]


@dataclass(frozen=True)
class MouthCandidate:
    position_m: tuple[float, float, float]
    image_x: float
    image_y: float | None
    depth_m: float | None
    surface_normal: tuple[float, float, float] | None

    def report(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "image_x": self.image_x,
            "image_y": self.image_y,
            "depth_m": self.depth_m,
            "surface_normal": None if self.surface_normal is None else list(self.surface_normal),
        }


@dataclass(frozen=True)
class SelectedMouthSample:
    candidate: MouthCandidate
    candidates: tuple[MouthCandidate, ...]
    selected_candidate_index: int
    received_monotonic: float
    source_stamp_sec: float | None
    raw_candidate_count: int
    duplicate_candidates_suppressed: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RejectedMouthSample:
    received_monotonic: float
    reason: str
    identity_unsafe: bool
    diagnostics: Mapping[str, Any] | None


class RealMouthTargetTracker:
    """Select and retain one real-camera mouth target without identity fallback."""

    def __init__(
        self,
        target_selection: str = "center",
        *,
        base_frame: str = "base_link",
        max_identity_jump_m: float = 0.12,
        ambiguity_margin_m: float = 0.03,
        duplicate_candidate_max_distance_m: float = 0.04,
        duplicate_candidate_max_image_distance_px: float = 48.0,
        ambiguity_confirmation_samples: int = 3,
        ambiguity_confirmation_max_gap_sec: float = 0.5,
        history_size: int = 512,
    ) -> None:
        self.target_selection = validate_target_selection(target_selection)
        self.base_frame = str(base_frame).strip().lstrip("/")
        if not self.base_frame:
            raise ValueError("base_frame must not be empty")
        if not math.isfinite(max_identity_jump_m) or max_identity_jump_m <= 0.0:
            raise ValueError("max_identity_jump_m must be positive and finite")
        if not math.isfinite(ambiguity_margin_m) or ambiguity_margin_m < 0.0:
            raise ValueError("ambiguity_margin_m must be non-negative and finite")
        if (
            not math.isfinite(duplicate_candidate_max_distance_m)
            or duplicate_candidate_max_distance_m <= 0.0
        ):
            raise ValueError("duplicate_candidate_max_distance_m must be positive and finite")
        if (
            not math.isfinite(duplicate_candidate_max_image_distance_px)
            or duplicate_candidate_max_image_distance_px <= 0.0
        ):
            raise ValueError(
                "duplicate_candidate_max_image_distance_px must be positive and finite"
            )
        if ambiguity_confirmation_samples < 1:
            raise ValueError("ambiguity_confirmation_samples must be at least one")
        if (
            not math.isfinite(ambiguity_confirmation_max_gap_sec)
            or ambiguity_confirmation_max_gap_sec <= 0.0
        ):
            raise ValueError(
                "ambiguity_confirmation_max_gap_sec must be positive and finite"
            )
        if history_size < 3:
            raise ValueError("history_size must be at least three")
        self.max_identity_jump_m = float(max_identity_jump_m)
        self.ambiguity_margin_m = float(ambiguity_margin_m)
        self.duplicate_candidate_max_distance_m = float(
            duplicate_candidate_max_distance_m
        )
        self.duplicate_candidate_max_image_distance_px = float(
            duplicate_candidate_max_image_distance_px
        )
        self.ambiguity_confirmation_samples = int(ambiguity_confirmation_samples)
        self.ambiguity_confirmation_max_gap_sec = float(
            ambiguity_confirmation_max_gap_sec
        )
        self._samples: deque[SelectedMouthSample] = deque(maxlen=int(history_size))
        self._rejections: deque[RejectedMouthSample] = deque(maxlen=int(history_size))
        self._reference_position: tuple[float, float, float] | None = None
        self._consecutive_ambiguity_updates = 0
        self._last_ambiguity_monotonic: float | None = None
        self._last_update: dict[str, Any] = {
            "success": False,
            "reason": "no mouth-candidate message has been received",
        }

    @staticmethod
    def _optional_finite(value: Any) -> float | None:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    @classmethod
    def _candidate(cls, raw: Any) -> MouthCandidate | None:
        if not isinstance(raw, Mapping):
            return None
        position = _finite_vector(raw.get("position"), 3)
        image_x = cls._optional_finite(raw.get("image_x"))
        if position is None or image_x is None:
            return None
        normal = _finite_vector(raw.get("surface_normal"), 3)
        return MouthCandidate(
            position_m=(position[0], position[1], position[2]),
            image_x=image_x,
            image_y=cls._optional_finite(raw.get("image_y")),
            depth_m=cls._optional_finite(raw.get("depth_m")),
            surface_normal=None if normal is None else (normal[0], normal[1], normal[2]),
        )

    @staticmethod
    def _initial_index(
        selection: str,
        candidates: Sequence[MouthCandidate],
        image_center_x: float,
    ) -> int | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return 0 if selection == "center" else None
        if selection == "left":
            return 0
        if selection == "right":
            return len(candidates) - 1
        return min(range(len(candidates)), key=lambda index: abs(candidates[index].image_x - image_center_x))

    @staticmethod
    def _image_distance(first: MouthCandidate, second: MouthCandidate) -> float | None:
        if first.image_y is None or second.image_y is None:
            return None
        return math.hypot(first.image_x - second.image_x, first.image_y - second.image_y)

    def _deduplicate_candidates(
        self,
        candidates: Sequence[MouthCandidate],
        *,
        image_center_x: float,
    ) -> tuple[list[MouthCandidate], list[dict[str, Any]]]:
        """Collapse only detections that represent the same physical mouth.

        MediaPipe can emit two overlapping face tracks for one partially
        visible face near the image boundary.  Treating their two mouth
        landmarks as two people makes the identity guard fail even though the
        scene contains one person.  A pair is considered a duplicate only when
        both its base-frame mouth positions and its 2D mouth pixels are close.
        If either measurement is unavailable or outside its tight threshold,
        both candidates remain and the normal ambiguity guard applies.
        """

        retained: list[MouthCandidate] = []
        suppressed: list[dict[str, Any]] = []
        for candidate in candidates:
            duplicate_index: int | None = None
            duplicate_position_distance = float("inf")
            duplicate_image_distance: float | None = None
            for index, existing in enumerate(retained):
                position_distance = _distance(candidate.position_m, existing.position_m)
                image_distance = self._image_distance(candidate, existing)
                if (
                    position_distance <= self.duplicate_candidate_max_distance_m
                    and image_distance is not None
                    and image_distance
                    <= self.duplicate_candidate_max_image_distance_px
                ):
                    duplicate_index = index
                    duplicate_position_distance = position_distance
                    duplicate_image_distance = image_distance
                    break
            if duplicate_index is None:
                retained.append(candidate)
                continue

            existing = retained[duplicate_index]
            if self._reference_position is not None:
                candidate_score = _distance(candidate.position_m, self._reference_position)
                existing_score = _distance(existing.position_m, self._reference_position)
                representative_reason = "closest_to_locked_3d_target"
            else:
                candidate_score = abs(candidate.image_x - image_center_x)
                existing_score = abs(existing.image_x - image_center_x)
                representative_reason = "closest_to_requested_image_center"
            if candidate_score < existing_score:
                retained[duplicate_index] = candidate
                kept, removed = candidate, existing
            else:
                kept, removed = existing, candidate
            suppressed.append(
                {
                    "reason": "same_mouth_duplicate",
                    "position_distance_m": duplicate_position_distance,
                    "image_distance_px": duplicate_image_distance,
                    "maximum_position_distance_m": (
                        self.duplicate_candidate_max_distance_m
                    ),
                    "maximum_image_distance_px": (
                        self.duplicate_candidate_max_image_distance_px
                    ),
                    "representative_selection": representative_reason,
                    "kept": kept.report(),
                    "suppressed": removed.report(),
                }
            )
        retained.sort(key=lambda item: item.image_x)
        return retained, suppressed

    def _reject(
        self,
        reason: str,
        *,
        received_monotonic: float,
        identity_unsafe: bool,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        copied_diagnostics = None if diagnostics is None else dict(diagnostics)
        self._rejections.append(
            RejectedMouthSample(
                received_monotonic=received_monotonic,
                reason=reason,
                identity_unsafe=identity_unsafe,
                diagnostics=copied_diagnostics,
            )
        )
        self._last_update = {
            "success": False,
            "reason": reason,
            "target_selection": self.target_selection,
            "identity_unsafe": identity_unsafe,
        }
        if copied_diagnostics is not None:
            self._last_update["diagnostics"] = copied_diagnostics
        return dict(self._last_update)

    def update_json(self, message_data: str, *, received_monotonic: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if received_monotonic is None else float(received_monotonic)
        if not math.isfinite(now):
            raise ValueError("received_monotonic must be finite")
        try:
            payload = json.loads(message_data)
        except (TypeError, json.JSONDecodeError):
            return self._reject("mouth-candidate payload is not valid JSON", received_monotonic=now, identity_unsafe=False)
        if not isinstance(payload, Mapping):
            return self._reject("mouth-candidate payload must be an object", received_monotonic=now, identity_unsafe=False)
        frame = str(payload.get("frame_id", "")).strip().lstrip("/")
        if frame != self.base_frame:
            return self._reject(
                f"mouth candidates must use {self.base_frame}; received {frame or 'an empty frame'}",
                received_monotonic=now,
                identity_unsafe=False,
            )
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            return self._reject("mouth-candidate payload has no candidate list", received_monotonic=now, identity_unsafe=False)
        parsed_candidates = [self._candidate(raw) for raw in raw_candidates]
        if any(candidate is None for candidate in parsed_candidates):
            return self._reject(
                "mouth-candidate payload contains an invalid candidate",
                received_monotonic=now,
                identity_unsafe=False,
            )
        raw_candidates_parsed = sorted(
            (candidate for candidate in parsed_candidates if candidate is not None),
            key=lambda candidate: candidate.image_x,
        )
        if not raw_candidates_parsed:
            return self._reject("no valid mouth candidates are visible", received_monotonic=now, identity_unsafe=False)
        center = self._optional_finite(payload.get("image_center_x"))
        if center is None:
            center = 0.5 * (
                raw_candidates_parsed[0].image_x
                + raw_candidates_parsed[-1].image_x
            )
        candidates, duplicate_candidates_suppressed = self._deduplicate_candidates(
            raw_candidates_parsed,
            image_center_x=center,
        )

        identity_unsafe = False
        if self._reference_position is None:
            selected_index = self._initial_index(self.target_selection, candidates, center)
            if selected_index is None:
                return self._reject(
                    f"requested {self.target_selection} target is not visible; one candidate is center only",
                    received_monotonic=now,
                    identity_unsafe=False,
                )
            selection_method = "image_position"
        else:
            distances = sorted(
                ((_distance(candidate.position_m, self._reference_position), index) for index, candidate in enumerate(candidates)),
                key=lambda item: item[0],
            )
            nearest_distance, selected_index = distances[0]
            if nearest_distance > self.max_identity_jump_m:
                return self._reject(
                    f"selected target moved {nearest_distance:.4f} m, above the {self.max_identity_jump_m:.4f} m identity limit",
                    received_monotonic=now,
                    identity_unsafe=True,
                    diagnostics={
                        "classification": "IDENTITY_JUMP",
                        "reference_position_m": list(self._reference_position),
                        "nearest_candidate_distance_m": nearest_distance,
                        "maximum_identity_jump_m": self.max_identity_jump_m,
                        "raw_candidate_count": len(raw_candidates_parsed),
                        "candidate_count_after_deduplication": len(candidates),
                        "duplicate_candidates_suppressed": (
                            duplicate_candidates_suppressed
                        ),
                        "visible_candidates": [
                            candidate.report() for candidate in candidates
                        ],
                    },
                )
            if (
                len(distances) > 1
                and distances[1][0] <= self.max_identity_jump_m
                and distances[1][0] - nearest_distance < self.ambiguity_margin_m
            ):
                if (
                    self._last_ambiguity_monotonic is None
                    or now - self._last_ambiguity_monotonic
                    > self.ambiguity_confirmation_max_gap_sec
                ):
                    self._consecutive_ambiguity_updates = 1
                else:
                    self._consecutive_ambiguity_updates += 1
                self._last_ambiguity_monotonic = now
                identity_unsafe = (
                    self._consecutive_ambiguity_updates
                    >= self.ambiguity_confirmation_samples
                )
                if identity_unsafe:
                    reason = (
                        "selected target match is persistently ambiguous between "
                        "multiple visible people"
                    )
                    classification = "AMBIGUOUS_IDENTITY_MATCH"
                else:
                    reason = (
                        "selected target match is temporarily ambiguous; waiting "
                        "for repeated observations"
                    )
                    classification = "AMBIGUOUS_IDENTITY_MATCH_PENDING"
                return self._reject(
                    reason,
                    received_monotonic=now,
                    identity_unsafe=identity_unsafe,
                    diagnostics={
                        "classification": classification,
                        "confirmation_count": self._consecutive_ambiguity_updates,
                        "required_confirmation_samples": (
                            self.ambiguity_confirmation_samples
                        ),
                        "maximum_confirmation_gap_sec": (
                            self.ambiguity_confirmation_max_gap_sec
                        ),
                        "reference_position_m": list(self._reference_position),
                        "nearest_candidate_index": selected_index,
                        "nearest_candidate_distance_m": nearest_distance,
                        "second_candidate_index": distances[1][1],
                        "second_candidate_distance_m": distances[1][0],
                        "distance_gap_m": distances[1][0] - nearest_distance,
                        "maximum_identity_jump_m": self.max_identity_jump_m,
                        "ambiguity_margin_m": self.ambiguity_margin_m,
                        "raw_candidate_count": len(raw_candidates_parsed),
                        "candidate_count_after_deduplication": len(candidates),
                        "duplicate_candidates_suppressed": (
                            duplicate_candidates_suppressed
                        ),
                        "visible_candidates": [
                            candidate.report() for candidate in candidates
                        ],
                    },
                )
            selection_method = "locked_3d_nearest"

        self._consecutive_ambiguity_updates = 0
        self._last_ambiguity_monotonic = None
        selected = candidates[selected_index]
        self._reference_position = selected.position_m
        source_stamp = self._optional_finite(payload.get("stamp_sec"))
        self._samples.append(
            SelectedMouthSample(
                candidate=selected,
                candidates=tuple(candidates),
                selected_candidate_index=selected_index,
                received_monotonic=now,
                source_stamp_sec=source_stamp,
                raw_candidate_count=len(raw_candidates_parsed),
                duplicate_candidates_suppressed=tuple(
                    duplicate_candidates_suppressed
                ),
            )
        )
        self._last_update = {
            "success": True,
            "reason": None,
            "target_selection": self.target_selection,
            "selection_method": selection_method,
            "candidate_count": len(candidates),
            "raw_candidate_count": len(raw_candidates_parsed),
            "duplicate_candidate_count": len(duplicate_candidates_suppressed),
            "duplicate_candidates_suppressed": duplicate_candidates_suppressed,
            "selected_candidate_index": selected_index,
            "selected_position_m": list(selected.position_m),
            "identity_unsafe": identity_unsafe,
        }
        return dict(self._last_update)

    @staticmethod
    def _surface_normal(
        samples: Sequence[SelectedMouthSample],
        *,
        frame_id: str,
    ) -> dict[str, Any]:
        normals = [sample.candidate.surface_normal for sample in samples if sample.candidate.surface_normal is not None]
        if not normals:
            return {"available": False, "reason": "selected candidate has no surface normal"}
        average = [sum(normal[axis] for normal in normals) / len(normals) for axis in range(3)]
        magnitude = math.sqrt(sum(component * component for component in average))
        if magnitude < 1e-8:
            return {"available": False, "reason": "selected candidate surface normals cancel"}
        normalized = [component / magnitude for component in average]
        return {
            "available": True,
            "frame_id": frame_id,
            "sample_count": len(normals),
            "mean_vector": normalized,
        }

    def observation(
        self,
        *,
        started_monotonic: float,
        now_monotonic: float | None = None,
        max_age_sec: float = 1.0,
        minimum_samples: int = 3,
        max_spread_m: float = 0.025,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        unsafe = [
            rejection
            for rejection in self._rejections
            if rejection.received_monotonic >= started_monotonic
            and rejection.identity_unsafe
        ]
        if unsafe:
            latest_unsafe = unsafe[-1]
            return {
                "available": False,
                "stable": False,
                "reason": latest_unsafe.reason,
                "target_selection": self.target_selection,
                "identity_locked": self._reference_position is not None,
                "identity_unsafe": True,
                "rejection_diagnostics": latest_unsafe.diagnostics,
            }
        recent = [
            sample
            for sample in self._samples
            if sample.received_monotonic >= started_monotonic
            and now - sample.received_monotonic <= max_age_sec
        ]
        latest_sample_time = recent[-1].received_monotonic if recent else float("-inf")
        unresolved_rejections = [
            rejection
            for rejection in self._rejections
            if rejection.received_monotonic >= started_monotonic
            and rejection.received_monotonic >= latest_sample_time
        ]
        if unresolved_rejections:
            latest_rejection = unresolved_rejections[-1]
            return {
                "available": False,
                "stable": False,
                "reason": latest_rejection.reason,
                "target_selection": self.target_selection,
                "identity_locked": self._reference_position is not None,
                "identity_unsafe": False,
                "rejection_diagnostics": latest_rejection.diagnostics,
            }
        if not recent:
            return {
                "available": False,
                "stable": False,
                "reason": str(self._last_update.get("reason") or "no selected mouth samples were received"),
                "target_selection": self.target_selection,
                "identity_locked": self._reference_position is not None,
                "identity_unsafe": bool(self._last_update.get("identity_unsafe")),
            }
        positions = [sample.candidate.position_m for sample in recent]
        mean_position = _mean(positions)
        max_distance = max(_distance(position, mean_position) for position in positions)
        stddev = [
            math.sqrt(sum((position[axis] - mean_position[axis]) ** 2 for position in positions) / len(positions))
            for axis in range(3)
        ]
        latest = recent[-1]
        return {
            "available": True,
            "stable": len(recent) >= minimum_samples and max_distance <= max_spread_m,
            "frame_id": self.base_frame,
            "target_selection": self.target_selection,
            "identity_locked": True,
            "identity_unsafe": False,
            "sample_count": len(recent),
            "mean_position_m": mean_position,
            "latest_position_m": list(latest.candidate.position_m),
            "jitter_stddev_m": stddev,
            "max_distance_from_mean_m": max_distance,
            "latest_received_age_sec": max(0.0, now - latest.received_monotonic),
            "latest_source_stamp_sec": latest.source_stamp_sec,
            "candidate_count": len(latest.candidates),
            "raw_candidate_count": latest.raw_candidate_count,
            "duplicate_candidate_count": len(
                latest.duplicate_candidates_suppressed
            ),
            "duplicate_candidates_suppressed": list(
                latest.duplicate_candidates_suppressed
            ),
            "selected_candidate_index": latest.selected_candidate_index,
            "visible_candidates": [candidate.report() for candidate in latest.candidates],
            "surface_normal": self._surface_normal(recent, frame_id=self.base_frame),
            "stability_requirements": {
                "minimum_samples": minimum_samples,
                "maximum_spread_m": max_spread_m,
                "maximum_age_sec": max_age_sec,
                "maximum_identity_jump_m": self.max_identity_jump_m,
                "ambiguity_margin_m": self.ambiguity_margin_m,
                "ambiguity_confirmation_samples": (
                    self.ambiguity_confirmation_samples
                ),
                "ambiguity_confirmation_max_gap_sec": (
                    self.ambiguity_confirmation_max_gap_sec
                ),
                "duplicate_candidate_max_distance_m": (
                    self.duplicate_candidate_max_distance_m
                ),
                "duplicate_candidate_max_image_distance_px": (
                    self.duplicate_candidate_max_image_distance_px
                ),
            },
        }

    def current_state(self, *, max_age_sec: float = 1.0, now_monotonic: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self._samples:
            return {
                "available": False,
                "reason": str(self._last_update.get("reason") or "selected target is unavailable"),
                "identity_unsafe": bool(self._last_update.get("identity_unsafe")),
            }
        if not self._last_update.get("success"):
            return {
                "available": False,
                "reason": str(self._last_update.get("reason") or "selected target is unavailable"),
                "identity_unsafe": bool(self._last_update.get("identity_unsafe")),
            }
        latest = self._samples[-1]
        age = now - latest.received_monotonic
        if age > max_age_sec:
            return {"available": False, "reason": "selected target and obstacle candidates are stale", "age_sec": age}
        if self._last_update.get("identity_unsafe"):
            return {"available": False, "reason": self._last_update.get("reason"), "identity_unsafe": True}
        return {
            "available": True,
            "target_selection": self.target_selection,
            "selected_position_m": list(latest.candidate.position_m),
            "selected_candidate_index": latest.selected_candidate_index,
            "candidate_count": len(latest.candidates),
            "raw_candidate_count": latest.raw_candidate_count,
            "duplicate_candidate_count": len(
                latest.duplicate_candidates_suppressed
            ),
            "duplicate_candidates_suppressed": list(
                latest.duplicate_candidates_suppressed
            ),
            "visible_candidates": [candidate.report() for candidate in latest.candidates],
            "age_sec": max(0.0, age),
            "identity_unsafe": False,
        }
