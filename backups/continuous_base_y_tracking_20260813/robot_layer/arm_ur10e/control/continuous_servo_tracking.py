"""Continuous base-frame hybrid pre-mouth control policy for MoveIt Servo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Sequence

import numpy as np

from robot_layer.arm_ur10e.perception.continuous_mouth_tracker import (
    ContinuousMouthTarget,
    ContinuousTrackingState,
)


SERVO_NO_WARNING = 0
SERVO_HALT_CODES = frozenset({2, 5, 6})
RECOVERY_BACKEND_ORDER = ("cartesian", "pilz", "ompl")
MINIMUM_CAMERA_RAY_FORWARD_MARGIN_M = 0.005


class MotionCommandOwner(str, Enum):
    IDLE = "IDLE"
    SERVO = "SERVO"
    PLANNER = "PLANNER"


class MotionCommandArbiter:
    """Prevent Servo and trajectory execution from owning motion together."""

    def __init__(self) -> None:
        self._owner = MotionCommandOwner.IDLE

    @property
    def owner(self) -> MotionCommandOwner:
        return self._owner

    def acquire(self, owner: MotionCommandOwner) -> bool:
        if owner == MotionCommandOwner.IDLE:
            raise ValueError("IDLE cannot acquire motion ownership")
        if self._owner not in {MotionCommandOwner.IDLE, owner}:
            return False
        self._owner = owner
        return True

    def release(self, owner: MotionCommandOwner) -> None:
        if self._owner != owner:
            raise RuntimeError(
                f"cannot release {owner.value} while {self._owner.value} owns motion"
            )
        self._owner = MotionCommandOwner.IDLE


@dataclass(frozen=True)
class ContinuousServoConfig:
    final_pre_mouth_standoff_m: float = 0.050
    provisional_standoff_m: float = 0.250
    servo_tracking_max_error_m: float = 0.10
    servo_replan_enter_m: float = 0.10
    servo_replan_exit_m: float = 0.06
    maximum_standoff_rate_mps: float = 0.010
    standoff_progress_error_m: float = 0.030
    standoff_final_tolerance_m: float = 0.001
    hold_entry_tolerance_m: float = 0.010
    maximum_linear_speed_mps: float = 0.020
    provisional_linear_speed_mps: float = 0.010
    maximum_linear_acceleration_mps2: float = 0.10
    maximum_angular_speed_rps: float = 0.15
    orientation_correction_gain: float = 1.0
    control_gain: float = 0.8
    x_control_gain: float = 0.8
    y_control_gain: float = 0.8
    z_control_gain: float = 0.8
    y_approach_speed_mps: float = 0.020
    approach_direction_base: tuple[float, float, float] = (0.0, -1.0, 0.0)
    maximum_tool_radius_m: float = 1.30
    maximum_flange_tilt_deg: float = 5.0
    maximum_tracking_duration_sec: float = 45.0

    def __post_init__(self) -> None:
        if not 0.05 <= self.final_pre_mouth_standoff_m <= 0.18:
            raise ValueError("final pre-mouth standoff must remain in [0.05, 0.18] m")
        if self.provisional_standoff_m < self.final_pre_mouth_standoff_m:
            raise ValueError("provisional standoff must not be closer than final standoff")
        if not (
            0.0 < self.servo_replan_exit_m < self.servo_replan_enter_m
            and self.servo_tracking_max_error_m >= self.servo_replan_enter_m
        ):
            raise ValueError("Servo hysteresis thresholds are invalid")
        if not 0.0 < self.maximum_standoff_rate_mps <= self.maximum_linear_speed_mps:
            raise ValueError("standoff rate must be positive and no greater than linear speed")
        if not 0.0 < self.standoff_progress_error_m <= self.servo_replan_enter_m:
            raise ValueError("standoff progress error must be within the local tracking range")
        if not 0.0 < self.standoff_final_tolerance_m <= 0.01:
            raise ValueError("standoff final tolerance must be within (0, 0.01] m")
        if not 0.0 < self.provisional_linear_speed_mps <= self.maximum_linear_speed_mps:
            raise ValueError("provisional speed must be positive and no greater than maximum")
        if self.maximum_linear_acceleration_mps2 <= 0.0:
            raise ValueError("maximum acceleration must be positive")
        if self.maximum_angular_speed_rps <= 0.0:
            raise ValueError("maximum angular speed must be positive")
        if self.orientation_correction_gain <= 0.0:
            raise ValueError("orientation correction gain must be positive")
        if min(self.x_control_gain, self.y_control_gain, self.z_control_gain) <= 0.0:
            raise ValueError("Cartesian control gains must be positive")
        if not 0.0 < self.y_approach_speed_mps <= self.maximum_linear_speed_mps:
            raise ValueError("Y approach speed must be within the real linear speed limit")
        direction = np.asarray(self.approach_direction_base, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("approach direction must contain three finite values")
        if not np.allclose(direction, (0.0, -1.0, 0.0), atol=1.0e-9):
            raise ValueError(
                "continuous real tracking must retain the validated base_link -Y approach"
            )


@dataclass(frozen=True)
class ContinuousServoDecision:
    command_allowed: bool
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_rps: tuple[float, float, float]
    desired_tool0_position_m: tuple[float, float, float]
    desired_straw_tip_position_m: tuple[float, float, float]
    target_error_m: float
    target_displacement_m: float
    commanded_standoff_m: float
    requested_standoff_m: float
    standoff_transition_active: bool
    state: str
    hold_ready: bool
    recovery_required: bool
    fallback_reason: str | None
    safety_stop_reason: str | None
    speed_limit_mps: float
    current_straw_tip_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cartesian_error_m: tuple[float, float, float] = (math.inf, math.inf, math.inf)


def rotate_vector_xyzw(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    value = np.asarray(vector, dtype=np.float64)
    if quaternion.shape != (4,) or value.shape != (3,):
        raise ValueError("quaternion/vector dimensions are invalid")
    magnitude = float(np.linalg.norm(quaternion))
    if not math.isfinite(magnitude) or magnitude < 1.0e-9:
        raise ValueError("orientation quaternion is invalid")
    x, y, z, w = quaternion / magnitude
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ value


def tool_vertical_tilt_deg(orientation_xyzw: Sequence[float]) -> float:
    axis = rotate_vector_xyzw(orientation_xyzw, (0.0, 0.0, 1.0))
    cosine = float(np.dot(axis, np.array((0.0, 0.0, -1.0))))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def vertical_axis_angular_correction(
    orientation_xyzw: Sequence[float],
    *,
    gain: float,
    maximum_speed_rps: float,
) -> np.ndarray:
    """Correct tool tilt in base_link while leaving vertical-axis spin free."""
    current_axis = rotate_vector_xyzw(orientation_xyzw, (0.0, 0.0, 1.0))
    desired_axis = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    cross = np.cross(current_axis, desired_axis)
    cross_norm = float(np.linalg.norm(cross))
    dot = max(-1.0, min(1.0, float(np.dot(current_axis, desired_axis))))
    angle = math.atan2(cross_norm, dot)
    if cross_norm < 1.0e-9 or angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    correction = cross / cross_norm * angle * float(gain)
    speed = float(np.linalg.norm(correction))
    if speed > maximum_speed_rps:
        correction *= float(maximum_speed_rps) / speed
    return correction


def camera_ray_premouth_target(
    *,
    mouth_position_m: Sequence[float],
    camera_position_m: Sequence[float],
    tool_orientation_xyzw: Sequence[float],
    straw_tip_offset_tool0_m: Sequence[float],
    standoff_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return straw-tip and tool0 targets on the validated camera-ray line."""
    mouth = np.asarray(mouth_position_m, dtype=np.float64)
    camera = np.asarray(camera_position_m, dtype=np.float64)
    if mouth.shape != (3,) or camera.shape != (3,) or not (
        np.all(np.isfinite(mouth)) and np.all(np.isfinite(camera))
    ):
        raise ValueError("mouth and camera positions must contain finite XYZ values")
    if not math.isfinite(standoff_m) or standoff_m < 0.05:
        raise ValueError("standoff must preserve the 50 mm minimum pre-mouth offset")
    camera_side = camera - mouth
    distance = float(np.linalg.norm(camera_side))
    if distance < 1.0e-6:
        raise ValueError("camera and mouth positions are coincident")
    if standoff_m > distance - MINIMUM_CAMERA_RAY_FORWARD_MARGIN_M:
        raise ValueError(
            "standoff would place the target at or behind the wrist camera "
            f"({standoff_m:.3f} m standoff, {distance:.3f} m camera-mouth distance)"
        )
    straw_target = mouth + standoff_m * camera_side / distance
    offset_base = rotate_vector_xyzw(
        tool_orientation_xyzw, straw_tip_offset_tool0_m
    )
    return straw_target, straw_target - offset_base


def base_y_premouth_target(
    *,
    mouth_position_m: Sequence[float],
    tool_orientation_xyzw: Sequence[float],
    straw_tip_offset_tool0_m: Sequence[float],
    standoff_m: float,
    approach_direction_base: Sequence[float] = (0.0, -1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return straw-tip/tool0 goals along the validated base_link -Y axis."""
    mouth = np.asarray(mouth_position_m, dtype=np.float64)
    direction = np.asarray(approach_direction_base, dtype=np.float64)
    if mouth.shape != (3,) or not np.all(np.isfinite(mouth)):
        raise ValueError("mouth position must contain finite base_link XYZ values")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("approach direction must contain finite base_link XYZ values")
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1.0e-9:
        raise ValueError("approach direction is zero")
    direction = direction / magnitude
    if not np.allclose(direction, (0.0, -1.0, 0.0), atol=1.0e-9):
        raise ValueError("approach direction does not match validated base_link -Y")
    if not math.isfinite(standoff_m) or standoff_m < 0.05:
        raise ValueError("standoff must preserve the validated 50 mm offset")
    straw_target = mouth + float(standoff_m) * direction
    offset_base = rotate_vector_xyzw(
        tool_orientation_xyzw, straw_tip_offset_tool0_m
    )
    return straw_target, straw_target - offset_base


class ContinuousServoController:
    """Generate uninterrupted bounded twist commands or an explicit stop."""

    def __init__(
        self,
        config: ContinuousServoConfig | None = None,
        *,
        straw_tip_offset_tool0_m: Sequence[float] = (0.110, 0.0, 0.0),
    ) -> None:
        self.config = config or ContinuousServoConfig()
        self.straw_tip_offset_tool0_m = tuple(
            float(value) for value in straw_tip_offset_tool0_m
        )
        self._last_velocity = np.zeros(3, dtype=np.float64)
        self._recovery_latched = False
        self._reference_target: np.ndarray | None = None
        self._commanded_standoff_m = self.config.provisional_standoff_m

    @property
    def recovery_latched(self) -> bool:
        return self._recovery_latched

    @property
    def commanded_standoff_m(self) -> float:
        """Current continuous approach standoff used to generate the target."""
        return float(self._commanded_standoff_m)

    def reset(self, *, commanded_standoff_m: float | None = None) -> None:
        """Start a new target-reference epoch for the fixed 50 mm target."""
        self._last_velocity[:] = 0.0
        self._recovery_latched = False
        self._reference_target = None
        self._commanded_standoff_m = self.config.final_pre_mouth_standoff_m

    def pause_for_reacquisition(self) -> None:
        """Restart acceleration limiting from zero without losing target history."""
        self._last_velocity[:] = 0.0

    def update(
        self,
        target: ContinuousMouthTarget,
        *,
        current_tool0_position_m: Sequence[float],
        current_tool0_orientation_xyzw: Sequence[float],
        camera_position_m: Sequence[float],
        servo_status_code: int,
        elapsed_sec: float,
        dt_sec: float,
    ) -> ContinuousServoDecision:
        current = np.asarray(current_tool0_position_m, dtype=np.float64)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            return self._stop("tool0_pose_invalid")
        if elapsed_sec > self.config.maximum_tracking_duration_sec:
            return self._stop("tracking_duration_limit")
        if not target.available:
            return self._stop(target.reason or "target_unavailable")
        if target.state in {
            ContinuousTrackingState.TARGET_LOST,
            ContinuousTrackingState.ABORTED,
            ContinuousTrackingState.NO_TARGET,
        }:
            return self._stop(f"target_state:{target.state.value}")
        tilt = tool_vertical_tilt_deg(current_tool0_orientation_xyzw)
        if tilt > self.config.maximum_flange_tilt_deg:
            return self._stop("flange_tilt_limit")
        if int(servo_status_code) in SERVO_HALT_CODES:
            return self._stop(f"servo_halt_status:{int(servo_status_code)}")

        dt = max(float(dt_sec), 1.0e-3)
        mouth = np.asarray(target.predicted_position_m, dtype=np.float64)
        if mouth.shape != (3,) or not np.all(np.isfinite(mouth)):
            return self._stop("mouth_position_invalid")
        try:
            straw_target, tool_target = base_y_premouth_target(
                mouth_position_m=target.predicted_position_m,
                tool_orientation_xyzw=current_tool0_orientation_xyzw,
                straw_tip_offset_tool0_m=self.straw_tip_offset_tool0_m,
                standoff_m=self.config.final_pre_mouth_standoff_m,
                approach_direction_base=self.config.approach_direction_base,
            )
        except ValueError as exc:
            return self._stop(f"premouth_target_invalid:{exc}")
        self._commanded_standoff_m = self.config.final_pre_mouth_standoff_m
        requested_standoff = self.config.final_pre_mouth_standoff_m
        straw_offset_base = rotate_vector_xyzw(
            current_tool0_orientation_xyzw, self.straw_tip_offset_tool0_m
        )
        current_straw = current + straw_offset_base
        error_vector = straw_target - current_straw
        error = float(np.linalg.norm(error_vector))
        if self._reference_target is None:
            self._reference_target = target.position_m.copy()
        displacement = float(np.linalg.norm(target.position_m - self._reference_target))

        # Recovery is driven by actual mouth displacement, not by Servo's
        # temporary convergence lag or by the changing approach standoff.
        if displacement > self.config.servo_replan_enter_m:
            self._recovery_latched = True
            return self._recovery(
                "servo_target_displacement_limit",
                tool_target,
                straw_target,
                error,
                displacement,
            )
        if self._recovery_latched:
            if displacement > self.config.servo_replan_exit_m:
                return self._recovery(
                    "servo_target_displacement_hysteresis",
                    tool_target,
                    straw_target,
                    error,
                    displacement,
                )
            self._recovery_latched = False
        if float(np.linalg.norm(tool_target)) > self.config.maximum_tool_radius_m:
            return self._stop("tool_workspace_radius_limit")

        speed_limit = (
            self.config.provisional_linear_speed_mps
            if target.provisional
            else self.config.maximum_linear_speed_mps
        )
        # MoveIt Servo already applies its collision-proximity scaling before
        # executing this bounded command. Do not stack a second 75% slowdown
        # in the application layer; hard halt codes are still rejected above.
        # X/Z use proportional target following. Y keeps the existing real
        # speed cap while far away and decelerates proportionally at terminal
        # approach, avoiding both artificial segments and target overshoot.
        velocity = np.asarray(
            (
                self.config.x_control_gain * error_vector[0],
                math.copysign(
                    min(
                        self.config.y_approach_speed_mps,
                        self.config.y_control_gain * abs(error_vector[1]),
                    ),
                    error_vector[1],
                )
                if abs(error_vector[1]) > 0.0
                else 0.0,
                self.config.z_control_gain * error_vector[2],
            ),
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(velocity))
        if speed > speed_limit and speed > 0.0:
            velocity *= speed_limit / speed
        change = velocity - self._last_velocity
        max_change = self.config.maximum_linear_acceleration_mps2 * dt
        # Apply the configured real acceleration limit independently to each
        # commanded base axis so vx/vy/vz remain continuous.
        velocity = self._last_velocity + np.clip(change, -max_change, max_change)
        limited_speed = float(np.linalg.norm(velocity))
        if limited_speed > speed_limit and limited_speed > 0.0:
            velocity *= speed_limit / limited_speed
        self._last_velocity = velocity
        angular_velocity = vertical_axis_angular_correction(
            current_tool0_orientation_xyzw,
            gain=self.config.orientation_correction_gain,
            maximum_speed_rps=self.config.maximum_angular_speed_rps,
        )
        hold_ready = bool(
            target.stable
            and abs(
                self._commanded_standoff_m
                - self.config.final_pre_mouth_standoff_m
            ) <= self.config.standoff_final_tolerance_m
            and error <= self.config.hold_entry_tolerance_m
        )
        return ContinuousServoDecision(
            command_allowed=True,
            linear_velocity_mps=tuple(float(value) for value in velocity),
            angular_velocity_rps=tuple(float(value) for value in angular_velocity),
            desired_tool0_position_m=tuple(float(value) for value in tool_target),
            desired_straw_tip_position_m=tuple(float(value) for value in straw_target),
            target_error_m=error,
            target_displacement_m=displacement,
            commanded_standoff_m=self._commanded_standoff_m,
            requested_standoff_m=requested_standoff,
            standoff_transition_active=False,
            state=("HOLDING" if hold_ready else "TRACKING"),
            hold_ready=hold_ready,
            recovery_required=False,
            fallback_reason=None,
            safety_stop_reason=None,
            speed_limit_mps=speed_limit,
            current_straw_tip_position_m=tuple(
                float(value) for value in current_straw
            ),
            cartesian_error_m=tuple(float(value) for value in error_vector),
        )

    def _stop(self, reason: str) -> ContinuousServoDecision:
        self._last_velocity[:] = 0.0
        return ContinuousServoDecision(
            command_allowed=False,
            linear_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_rps=(0.0, 0.0, 0.0),
            desired_tool0_position_m=(0.0, 0.0, 0.0),
            desired_straw_tip_position_m=(0.0, 0.0, 0.0),
            target_error_m=math.inf,
            target_displacement_m=0.0,
            commanded_standoff_m=self._commanded_standoff_m,
            requested_standoff_m=self._commanded_standoff_m,
            standoff_transition_active=False,
            state="ABORTED",
            hold_ready=False,
            recovery_required=False,
            fallback_reason=None,
            safety_stop_reason=str(reason),
            speed_limit_mps=0.0,
        )

    def _recovery(
        self,
        reason: str,
        tool_target: np.ndarray,
        straw_target: np.ndarray,
        error: float,
        displacement: float,
    ) -> ContinuousServoDecision:
        self._last_velocity[:] = 0.0
        return ContinuousServoDecision(
            command_allowed=False,
            linear_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_rps=(0.0, 0.0, 0.0),
            desired_tool0_position_m=tuple(float(value) for value in tool_target),
            desired_straw_tip_position_m=tuple(float(value) for value in straw_target),
            target_error_m=error,
            target_displacement_m=displacement,
            commanded_standoff_m=self._commanded_standoff_m,
            requested_standoff_m=self._commanded_standoff_m,
            standoff_transition_active=False,
            state="RECOVERY_REQUIRED",
            hold_ready=False,
            recovery_required=True,
            fallback_reason=str(reason),
            safety_stop_reason=None,
            speed_limit_mps=0.0,
        )


def run_recovery_backends(
    callbacks: dict[str, Callable[[], bool]],
) -> tuple[str | None, tuple[str, ...]]:
    """Try Cartesian, Pilz, then OMPL, stopping after the first success."""
    attempted: list[str] = []
    for name in RECOVERY_BACKEND_ORDER:
        callback = callbacks.get(name)
        if callback is None:
            continue
        attempted.append(name)
        if bool(callback()):
            return name, tuple(attempted)
    return None, tuple(attempted)


def octomap_layer_status(
    *,
    use_octomap: bool,
    rebuild_succeeded: bool | None,
    occupancy_present: bool = False,
) -> dict[str, object]:
    """Describe the optional dynamic layer without overstating protection."""
    if not use_octomap:
        if occupancy_present:
            return {
                "use_octomap": False,
                "dynamic_obstacle_layer_active": False,
                "degraded": True,
                "configuration_valid": False,
                "status": "dynamic_obstacle_layer_configuration_mismatch",
            }
        return {
            "use_octomap": False,
            "dynamic_obstacle_layer_active": False,
            "degraded": False,
            "configuration_valid": True,
            "status": "dynamic_obstacle_layer_disabled",
        }
    if rebuild_succeeded:
        return {
            "use_octomap": True,
            "dynamic_obstacle_layer_active": True,
            "degraded": False,
            "configuration_valid": True,
            "status": "dynamic_obstacle_layer_active",
        }
    return {
        "use_octomap": True,
        "dynamic_obstacle_layer_active": False,
        "degraded": True,
        "configuration_valid": True,
        "status": "dynamic_obstacle_layer_unavailable",
    }
