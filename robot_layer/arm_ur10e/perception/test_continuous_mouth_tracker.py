"""Simulation-only tests for robust continuous mouth acquisition."""

import numpy as np

from .continuous_mouth_tracker import (
    ContinuousMouthTracker,
    ContinuousTrackingState,
    InitialTargetAcquirer,
)


def _add(tracker, position, index, *, confidence=0.9, depth=0.8):
    return tracker.add_observation(
        position,
        source_timestamp_sec=100.0 + index * 0.1,
        received_monotonic_sec=10.0 + index * 0.1,
        depth_m=depth,
        confidence=confidence,
    )


def test_two_valid_observations_form_provisional_average_after_timeout():
    tracker = ContinuousMouthTracker()
    _add(tracker, (0.80, 0.20, 0.60), 0)
    _add(tracker, (0.82, 0.20, 0.60), 1)
    target = tracker.target(now_monotonic_sec=10.11)
    acquisition = InitialTargetAcquirer(
        started_monotonic_sec=7.0,
        timeout_sec=3.0,
    ).evaluate(target, now_monotonic_sec=10.11)

    assert acquisition.complete
    assert acquisition.state == ContinuousTrackingState.PROVISIONAL_TARGET
    assert target.provisional
    assert np.allclose(target.position_m, (0.81, 0.20, 0.60))


def test_stable_target_completes_before_timeout_with_robust_median():
    tracker = ContinuousMouthTracker()
    _add(tracker, (0.80, 0.20, 0.60), 0)
    _add(tracker, (0.801, 0.199, 0.601), 1)
    _add(tracker, (0.799, 0.201, 0.599), 2)
    target = tracker.target(now_monotonic_sec=10.21)
    acquisition = InitialTargetAcquirer(
        started_monotonic_sec=10.0,
    ).evaluate(target, now_monotonic_sec=10.21)

    assert acquisition.complete
    assert target.stable
    assert target.state == ContinuousTrackingState.STABLE_TARGET
    assert np.allclose(target.position_m, (0.80, 0.20, 0.60))


def test_no_initial_target_requests_search_without_inventing_pose():
    tracker = ContinuousMouthTracker()
    target = tracker.target(now_monotonic_sec=13.1)
    acquisition = InitialTargetAcquirer(
        started_monotonic_sec=10.0,
    ).evaluate(target, now_monotonic_sec=13.1)

    assert acquisition.complete
    assert acquisition.active_search_required
    assert not target.available
    assert np.allclose(target.position_m, np.zeros(3))


def test_invalid_depth_and_timestamp_are_rejected():
    tracker = ContinuousMouthTracker()
    accepted_depth, depth_reason = _add(
        tracker, (0.8, 0.2, 0.6), 0, depth=0.0
    )
    accepted_stamp, stamp_reason = tracker.add_observation(
        (0.8, 0.2, 0.6),
        source_timestamp_sec=0.0,
        received_monotonic_sec=10.0,
        depth_m=0.8,
        confidence=0.9,
    )

    assert not accepted_depth and depth_reason == "invalid_depth"
    assert not accepted_stamp and stamp_reason == "invalid_timestamp_or_numeric_field"


def test_prediction_is_bounded_and_disabled_for_provisional_target():
    tracker = ContinuousMouthTracker(maximum_prediction_m=0.02)
    _add(tracker, (0.80, 0.20, 0.60), 0)
    _add(tracker, (0.81, 0.20, 0.60), 1)
    provisional = tracker.target(now_monotonic_sec=10.11)
    _add(tracker, (0.82, 0.20, 0.60), 2)
    stable = tracker.target(now_monotonic_sec=10.21)

    assert provisional.prediction_m == 0.0
    assert stable.prediction_m <= 0.02


def test_stale_target_holds_before_target_lost_timeout():
    tracker = ContinuousMouthTracker()
    _add(tracker, (0.80, 0.20, 0.60), 0)

    target = tracker.target(now_monotonic_sec=10.31)

    assert not target.available
    assert target.state == ContinuousTrackingState.TRACKING
    assert target.reason == "target_stale_grace"
    assert abs(target.age_sec - 0.31) < 1.0e-9


def test_lost_timeout_transitions_to_target_lost():
    tracker = ContinuousMouthTracker()
    _add(tracker, (0.80, 0.20, 0.60), 0)

    target = tracker.target(now_monotonic_sec=11.0)

    assert not target.available
    assert target.state == ContinuousTrackingState.TARGET_LOST
    assert target.reason == "target_lost_timeout"
