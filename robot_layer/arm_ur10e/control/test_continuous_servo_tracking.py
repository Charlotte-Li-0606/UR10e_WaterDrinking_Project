"""Simulation-only tests for continuous MoveIt Servo control policy."""

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
    camera_ray_premouth_target,
    octomap_layer_status,
    run_recovery_backends,
    vertical_axis_angular_correction,
)


FLANGE_DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)
CAMERA_POSITION = (0.0, 0.0, 0.0)


def _target(position=(1.0, 0.0, 0.0), *, stable=True, available=True):
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
        sample_count=3 if stable else 2,
        spread_m=0.002,
        prediction_m=0.0,
        reason=None if available else "no_valid_target",
    )


def _desired_tool(mouth=(1.0, 0.0, 0.0), standoff=0.08):
    _, tool = camera_ray_premouth_target(
        mouth_position_m=mouth,
        camera_position_m=CAMERA_POSITION,
        tool_orientation_xyzw=FLANGE_DOWN_XYZW,
        straw_tip_offset_tool0_m=(0.110, 0.0, 0.0),
        standoff_m=standoff,
    )
    return tool


def test_static_mouth_reaches_target_without_segmented_pauses():
    controller = ContinuousServoController()
    tool = _desired_tool() - np.array((0.08, 0.0, 0.0))
    allowed_commands = 0
    for index in range(500):
        decision = controller.update(
            _target(),
            current_tool0_position_m=tool,
            current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
            camera_position_m=CAMERA_POSITION,
            servo_status_code=0,
            elapsed_sec=index * 0.02,
            dt_sec=0.02,
        )
        assert not decision.recovery_required
        assert decision.command_allowed
        allowed_commands += 1
        tool += np.asarray(decision.linear_velocity_mps) * 0.02
        if decision.hold_ready:
            break

    assert allowed_commands > 1
    assert decision.hold_ready
    assert decision.target_error_m <= 0.01


def test_mouth_motion_within_ten_centimetres_stays_in_servo():
    controller = ContinuousServoController()
    tool = _desired_tool()
    maximum_error = 0.0
    for index in range(101):
        mouth = (1.0, 0.0008 * index, 0.0)
        decision = controller.update(
            _target(mouth),
            current_tool0_position_m=tool,
            current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
            camera_position_m=CAMERA_POSITION,
            servo_status_code=0,
            elapsed_sec=index * 0.02,
            dt_sec=0.02,
        )
        assert decision.command_allowed
        assert not decision.recovery_required
        maximum_error = max(maximum_error, decision.target_error_m)
        tool += np.asarray(decision.linear_velocity_mps) * 0.02

    assert maximum_error <= 0.10


def test_target_beyond_servo_range_uses_cartesian_pilz_ompl_priority():
    controller = ContinuousServoController()
    decision = controller.update(
        _target(),
        current_tool0_position_m=_desired_tool() - np.array((0.13, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=1.0,
        dt_sec=0.02,
    )
    calls = []
    selected, attempted = run_recovery_backends(
        {
            "cartesian": lambda: calls.append("cartesian") or False,
            "pilz": lambda: calls.append("pilz") or True,
            "ompl": lambda: calls.append("ompl") or True,
        }
    )

    assert decision.recovery_required
    assert not decision.command_allowed
    assert selected == "pilz"
    assert attempted == ("cartesian", "pilz")
    assert calls == ["cartesian", "pilz"]


def test_bounded_startup_envelope_enters_normal_tracking_without_pause():
    controller = ContinuousServoController()
    desired = _desired_tool()

    startup = controller.update(
        _target(),
        current_tool0_position_m=desired - np.array((0.105, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.0,
        dt_sec=0.02,
    )
    entered = controller.update(
        _target(),
        current_tool0_position_m=desired - np.array((0.099, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.02,
        dt_sec=0.02,
    )
    outside_again = controller.update(
        _target(),
        current_tool0_position_m=desired - np.array((0.105, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.04,
        dt_sec=0.02,
    )

    assert startup.command_allowed
    assert not startup.recovery_required
    assert entered.command_allowed
    assert outside_again.recovery_required
    assert outside_again.fallback_reason == "servo_tracking_error_limit"


def test_startup_error_above_bounded_envelope_uses_recovery():
    decision = ContinuousServoController().update(
        _target(),
        current_tool0_position_m=_desired_tool() - np.array((0.121, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.0,
        dt_sec=0.02,
    )

    assert decision.recovery_required
    assert not decision.command_allowed
    assert decision.fallback_reason == "servo_startup_error_limit"


def test_provisional_target_uses_large_standoff_and_reduced_speed_without_hold():
    controller = ContinuousServoController()
    target = _target(stable=False)
    desired = _desired_tool(standoff=0.17)
    decision = controller.update(
        target,
        current_tool0_position_m=desired - np.array((0.05, 0.0, 0.0)),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=1.0,
        dt_sec=0.02,
    )

    assert decision.command_allowed
    assert decision.speed_limit_mps == 0.01
    assert not decision.hold_ready
    assert np.allclose(decision.desired_tool0_position_m, desired)


def test_no_target_and_target_loss_publish_no_motion():
    controller = ContinuousServoController()
    missing = controller.update(
        _target(available=False),
        current_tool0_position_m=_desired_tool(),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.0,
        dt_sec=0.02,
    )
    lost_target = _target()
    lost_target = ContinuousMouthTarget(
        **{**lost_target.__dict__, "available": False,
           "state": ContinuousTrackingState.TARGET_LOST,
           "reason": "target_stale"}
    )
    lost = controller.update(
        lost_target,
        current_tool0_position_m=_desired_tool(),
        current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
        camera_position_m=CAMERA_POSITION,
        servo_status_code=0,
        elapsed_sec=0.1,
        dt_sec=0.02,
    )

    assert not missing.command_allowed
    assert not lost.command_allowed
    assert missing.linear_velocity_mps == (0.0, 0.0, 0.0)
    assert lost.safety_stop_reason == "target_stale"


def test_servo_collision_singularity_and_joint_limit_codes_halt():
    for status in (2, 5, 6):
        decision = ContinuousServoController().update(
            _target(),
            current_tool0_position_m=_desired_tool(),
            current_tool0_orientation_xyzw=FLANGE_DOWN_XYZW,
            camera_position_m=CAMERA_POSITION,
            servo_status_code=status,
            elapsed_sec=0.0,
            dt_sec=0.02,
        )
        assert not decision.command_allowed
        assert decision.safety_stop_reason == f"servo_halt_status:{status}"


def test_octomap_disabled_and_failed_enabled_modes_are_explicit():
    disabled = octomap_layer_status(use_octomap=False, rebuild_succeeded=None)
    failed = octomap_layer_status(use_octomap=True, rebuild_succeeded=False)
    active = octomap_layer_status(use_octomap=True, rebuild_succeeded=True)
    mismatch = octomap_layer_status(
        use_octomap=False,
        rebuild_succeeded=None,
        occupancy_present=True,
    )

    assert disabled["status"] == "dynamic_obstacle_layer_disabled"
    assert not disabled["dynamic_obstacle_layer_active"]
    assert failed["status"] == "dynamic_obstacle_layer_unavailable"
    assert failed["degraded"]
    assert active["dynamic_obstacle_layer_active"]
    assert mismatch["status"] == "dynamic_obstacle_layer_configuration_mismatch"
    assert not mismatch["configuration_valid"]


def test_vertical_axis_correction_keeps_yaw_free():
    down_with_yaw = (0.9238795, 0.3826834, 0.0, 0.0)
    correction = vertical_axis_angular_correction(
        down_with_yaw,
        gain=1.0,
        maximum_speed_rps=0.15,
    )
    assert np.allclose(correction, (0.0, 0.0, 0.0), atol=1.0e-6)

    tilted = (0.9990482, 0.0, 0.0, 0.0436194)
    correction = vertical_axis_angular_correction(
        tilted,
        gain=1.0,
        maximum_speed_rps=0.15,
    )
    assert 0.0 < np.linalg.norm(correction) <= 0.15


def test_motion_arbiter_excludes_servo_and_planner_commands():
    arbiter = MotionCommandArbiter()
    assert arbiter.acquire(MotionCommandOwner.SERVO)
    assert not arbiter.acquire(MotionCommandOwner.PLANNER)
    arbiter.release(MotionCommandOwner.SERVO)
    assert arbiter.acquire(MotionCommandOwner.PLANNER)
    assert not arbiter.acquire(MotionCommandOwner.SERVO)
