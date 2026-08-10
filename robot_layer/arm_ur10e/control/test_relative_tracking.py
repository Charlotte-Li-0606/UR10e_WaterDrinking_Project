import pytest

from .relative_tracking import RelativeTrackingSession


def test_small_mouth_displacement_produces_bounded_translation_only():
    session = RelativeTrackingSession(max_linear_acceleration_mps2=10.0)
    session.lock((0.4, 0.0, 0.8), (0.3, -0.1, 0.4))
    result = session.update((0.42, 0.0, 0.8), (0.3, -0.1, 0.4), dt_sec=0.1)
    assert result.allowed
    assert result.desired_tool_position_m == pytest.approx((0.32, -0.1, 0.4))
    assert result.angular_velocity_rps == (0.0, 0.0, 0.0)
    assert sum(value * value for value in result.linear_velocity_mps) ** 0.5 <= 0.02


def test_large_target_displacement_halts():
    session = RelativeTrackingSession(max_target_displacement_m=0.06)
    session.lock((0.4, 0.0, 0.8), (0.3, -0.1, 0.4))
    result = session.update((0.47, 0.0, 0.8), (0.3, -0.1, 0.4), dt_sec=0.1)
    assert not result.allowed
    assert result.reason == "target_displacement_limit"
    assert result.linear_velocity_mps == (0.0, 0.0, 0.0)


def test_workspace_radius_halts():
    session = RelativeTrackingSession(max_tool_radius_m=1.30)
    session.lock((0.4, 0.0, 0.8), (1.29, 0.0, 0.0))
    result = session.update((0.42, 0.0, 0.8), (1.29, 0.0, 0.0), dt_sec=0.1)
    assert not result.allowed
    assert result.reason == "tool_workspace_radius_limit"
