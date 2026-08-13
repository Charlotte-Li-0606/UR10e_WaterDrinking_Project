"""No-motion regression tests for the active base-Y backup runner."""

from __future__ import annotations

from unittest.mock import Mock, patch

from scripts.real_feed_water_integrated_base_y_backup import load_backup_runner


def test_timeout_boundary_retains_fresh_provisional_target() -> None:
    runner = load_backup_runner()
    node = runner.RealIntegratedFeedWater.__new__(runner.RealIntegratedFeedWater)
    node._continuous_parameters = {
        "control_rate_hz": 50.0,
        "initial_target_acquisition_timeout_sec": 3.0,
    }

    missing_tracker = runner.ContinuousMouthTracker()
    missing_tracker.reset(searching=True)
    missing = missing_tracker.target(now_monotonic_sec=13.01)

    fresh_tracker = runner.ContinuousMouthTracker()
    fresh_tracker.reset(searching=True)
    accepted, reason = fresh_tracker.add_observation(
        [-1.0, 0.2, 0.6],
        source_timestamp_sec=100.0,
        received_monotonic_sec=13.02,
        depth_m=0.25,
        confidence=1.0,
    )
    assert accepted, reason
    fresh = fresh_tracker.target(now_monotonic_sec=13.02)

    node._continuous_tracker = Mock()
    node._continuous_tracker.target.side_effect = [missing, fresh]
    node._publish_continuous_diagnostics = Mock()
    node._continuous_last_observation_update = {
        "accepted": True,
        "reason": "accepted",
    }

    with patch.object(runner.rclpy, "ok", return_value=True), patch.object(
        runner.rclpy, "spin_once"
    ) as spin_once, patch.object(
        runner.time,
        "monotonic",
        side_effect=[10.0, 13.01, 13.02, 13.03],
    ):
        result = node._acquire_continuous_target()

    assert result["success"]
    assert result["state"] == "PROVISIONAL_TARGET"
    assert result["target"].available
    assert spin_once.call_count == 2
