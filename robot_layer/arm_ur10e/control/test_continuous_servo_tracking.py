"""Simulation-only tests for the continuous MoveIt Servo control policy."""

import numpy as np

from robot_layer.arm_ur10e.perception.continuous_mouth_tracker import (
    ContinuousMouthTarget,
    ContinuousTrackingState,
)
from .continuous_servo_tracking import (
    ContinuousServoConfig,
    ContinuousServoController,
    MotionCommandArbiter,
    MotionCommandOwner,
    calibrated_axis_premouth_target,
    camera_ray_premouth_target,
    octomap_layer_status,
    rotate_vector_xyzw,
    run_recovery_backends,
    vertical_axis_angular_correction,
)


FLANGE_DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)
CAMERA_POSITION = (0.0, 0.0, 0.0)
MOUTH = (1.0, 0.0, 0.0)


def _target(position=MOUTH, *, stable=True, available=True):
    point = np.asarray(position, dtype=np.float64)
    return ContinuousMouthTarget(
        available=available,
        position_m=point,
        predicted_position_m=point,
        velocity_mps=np.zeros(3),
        source_timestamp_sec=1.0,
        age_sec=0.01,
        confidence=0.9 if available else 0.0,
        target_id="center",
        state=(
            ContinuousTrackingState.STABLE_TARGET
            if stable and available
            else ContinuousTrackingState.PROVISIONAL_TARGET
            if available
            else ContinuousTrackingState.NO_TARGET
        ),
        provisional=available and not stable,
        stable=available and stable,
        sample_count=3 if stable else 1,
        spread_m=0.002,
        prediction_m=0.0,
        reason=None if available else "no_valid_target",
    )


def _desired_tool(mouth=MOUTH, standoff=0.05):
    _, tool = calibrated_axis_premouth_target(
        mouth_position_m=mouth,
        tool_orientation_xyzw=FLANGE_DOWN_XYZW,
        straw_tip_offset_tool0_m=(0.110, 0.0, 0.0),
        standoff_m=standoff,
    )
    return tool


def _update(controller, target, tool, *, elapsed=0.0, dt=0.02, status=0):
    return controller.update(
        target,
        current_tool0_position_m=tool,
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=status,
        elapsed_sec=elapsed,
        dt_sec=dt,
    )


def test_camera_ray_target_retains_validated_standoff_and_tool_offset():
    straw, tool = camera_ray_premouth_target(
        mouth_position_m=MOUTH,
        camera_position_m=CAMERA_POSITION,
        tool_orientation_xyzw=FLANGE_DOWN_XYZW,
        straw_tip_offset_tool0_m=(0.110, 0.0, 0.0),
        standoff_m=0.08,
    )
    assert np.allclose(straw, (0.92, 0.0, 0.0))
    assert np.allclose(tool + np.array((0.110, 0.0, 0.0)), straw)


def test_continuous_target_is_exactly_50_mm_on_tool0_positive_y():
    mouth = (-0.95, 0.25, 0.65)
    live_flange_down = (0.72116807, 0.69268808, -0.00916012, 0.00399217)
    straw, tool = calibrated_axis_premouth_target(
        mouth_position_m=mouth,
        tool_orientation_xyzw=live_flange_down,
        straw_tip_offset_tool0_m=(0.110, 0.0, 0.0),
        standoff_m=0.05,
    )

    expected_offset_base = rotate_vector_xyzw(
        live_flange_down, (0.0, 0.05, 0.0)
    )
    straw_offset_base = straw - np.asarray(mouth)
    assert np.allclose(straw_offset_base, expected_offset_base)
    assert np.isclose(np.linalg.norm(straw_offset_base), 0.05)
    assert np.allclose(
        tool + rotate_vector_xyzw(live_flange_down, (0.110, 0.0, 0.0)),
        straw,
    )


def test_acceleration_and_speed_limits_apply_to_cartesian_command():
    controller = ContinuousServoController()
    desired = _desired_tool()
    decision = _update(
        controller,
        _target(),
        desired - np.array((0.20, 0.0, 0.0)),
        dt=0.02,
    )

    assert np.linalg.norm(decision.linear_velocity_mps) <= 0.020000001
    assert np.linalg.norm(decision.linear_velocity_mps) <= 0.020000001


def test_static_mouth_reaches_hold_without_segmented_pauses():
    controller = ContinuousServoController()
    tool = _desired_tool() - np.array((0.08, 0.0, 0.0))
    command_count = 0
    for index in range(3000):
        decision = _update(controller, _target(), tool, elapsed=index * 0.02)
        assert decision.command_allowed
        assert not decision.recovery_required
        command_count += 1
        tool += np.asarray(decision.linear_velocity_mps) * 0.02
        if decision.hold_ready:
            break

    assert command_count > 1
    assert decision.hold_ready
    assert decision.target_error_m <= 0.01


def test_provisional_target_moves_at_existing_reduced_speed_without_hold():
    controller = ContinuousServoController()
    desired = _desired_tool(standoff=0.25)
    decision = _update(
        controller,
        _target(stable=False),
        desired - np.array((0.05, 0.0, 0.0)),
        dt=0.20,
    )

    assert decision.command_allowed
    assert decision.speed_limit_mps == 0.010
    assert np.linalg.norm(decision.linear_velocity_mps) <= 0.010000001
    assert not decision.hold_ready


def test_fresh_provisional_target_can_latch_hold_at_exact_premouth_geometry():
    controller = ContinuousServoController()
    controller.reset(commanded_standoff_m=0.05)

    decision = _update(
        controller,
        _target(stable=False),
        _desired_tool(standoff=0.05),
        dt=0.02,
    )

    assert decision.command_allowed
    assert decision.target_error_m == 0.0
    assert decision.hold_ready
    assert decision.state == "HOLDING"


def test_large_live_mouth_displacement_requests_bounded_recovery():
    controller = ContinuousServoController()
    _update(controller, _target(), _desired_tool())
    decision = _update(
        controller,
        _target((1.0, 0.11, 0.0)),
        _desired_tool(),
        elapsed=0.02,
    )

    assert decision.recovery_required
    assert not decision.command_allowed
    assert decision.fallback_reason == "servo_target_displacement_limit"


def test_recovery_backend_order_is_cartesian_then_pilz_then_ompl():
    calls = []
    selected, attempted = run_recovery_backends(
        {
            "cartesian": lambda: calls.append("cartesian") or False,
            "pilz": lambda: calls.append("pilz") or True,
            "ompl": lambda: calls.append("ompl") or True,
        }
    )

    assert selected == "pilz"
    assert attempted == ("cartesian", "pilz")
    assert calls == ["cartesian", "pilz"]


def test_no_target_target_loss_and_servo_halts_command_no_motion():
    controller = ContinuousServoController()
    missing = _update(controller, _target(available=False), _desired_tool())
    assert not missing.command_allowed
    assert missing.linear_velocity_mps == (0.0, 0.0, 0.0)

    for status in (2, 5, 6):
        halted = _update(controller, _target(), _desired_tool(), status=status)
        assert not halted.command_allowed
        assert halted.safety_stop_reason == f"servo_halt_status:{status}"


def test_flange_down_correction_leaves_spin_free():
    down_with_yaw = (0.9238795, 0.3826834, 0.0, 0.0)
    correction = vertical_axis_angular_correction(
        down_with_yaw, gain=1.0, maximum_speed_rps=0.15
    )
    assert np.allclose(correction, (0.0, 0.0, 0.0), atol=1.0e-6)

    tilted = (0.9990482, 0.0, 0.0, 0.0436194)
    correction = vertical_axis_angular_correction(
        tilted, gain=1.0, maximum_speed_rps=0.15
    )
    assert 0.0 < np.linalg.norm(correction) <= 0.15


def test_motion_arbiter_allows_only_one_owner():
    arbiter = MotionCommandArbiter()
    assert arbiter.acquire(MotionCommandOwner.SERVO)
    assert not arbiter.acquire(MotionCommandOwner.PLANNER)
    arbiter.release(MotionCommandOwner.SERVO)
    assert arbiter.acquire(MotionCommandOwner.PLANNER)


def test_octomap_configuration_reporting_is_preserved():
    disabled = octomap_layer_status(use_octomap=False, rebuild_succeeded=None)
    active = octomap_layer_status(use_octomap=True, rebuild_succeeded=True)
    failed = octomap_layer_status(use_octomap=True, rebuild_succeeded=False)

    assert disabled["status"] == "dynamic_obstacle_layer_disabled"
    assert active["dynamic_obstacle_layer_active"]
    assert failed["degraded"]
