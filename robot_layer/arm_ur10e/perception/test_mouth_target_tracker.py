import numpy as np
import pytest

from .mouth_target_tracker import MouthTargetTracker, TrackingState


def test_tracker_locks_then_tracks_and_estimates_velocity():
    tracker = MouthTargetTracker()
    first = tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    second = tracker.update([0.31, 0.4, 1.0], timestamp_sec=1.1)
    assert first.state == TrackingState.LOCKED
    assert second.state == TrackingState.TRACKING
    assert second.velocity[0] > 0.0


def test_tracker_marks_large_displacement_for_replan():
    tracker = MouthTargetTracker(replan_distance_m=0.09)
    tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    target = tracker.update([0.42, 0.4, 1.0], timestamp_sec=1.1)
    assert target.state == TrackingState.ABORTED
    assert tracker.requires_replan(target)


def test_tracker_fails_closed_on_stale_target_and_bad_confidence():
    tracker = MouthTargetTracker()
    tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    stale = tracker.snapshot(now_sec=1.6)
    assert stale.state == TrackingState.LOST
    rejected = tracker.update([0.5, 0.4, 1.0], timestamp_sec=1.7, confidence=0.1)
    assert rejected.state == TrackingState.LOST
    assert np.allclose(rejected.position, [0.3, 0.4, 1.0], atol=0.01)


def test_tracker_explicit_reset_starts_new_reference():
    tracker = MouthTargetTracker(replan_distance_m=0.09)
    tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    tracker.update([0.42, 0.4, 1.0], timestamp_sec=1.1)
    tracker.begin_session()
    target = tracker.update([0.42, 0.4, 1.0], timestamp_sec=2.0)
    assert target.state == TrackingState.LOCKED
    assert target.displacement_m == 0.0


def test_tracker_recovery_after_lost_target_uses_new_reference():
    tracker = MouthTargetTracker()
    tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    assert tracker.snapshot(now_sec=2.0).state == TrackingState.LOST
    target = tracker.update([0.5, 0.4, 1.0], timestamp_sec=2.1)
    assert target.state == TrackingState.LOCKED
    assert target.displacement_m == 0.0
