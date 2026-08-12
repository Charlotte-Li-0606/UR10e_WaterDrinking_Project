"""Continuous camera-ray pre-mouth control policy for MoveIt Servo."""

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
SERVO_DECELERATE_CODES = frozenset({1, 3, 4})
SERVO_HALT_CODES = frozenset({2, 5, 6})
RECOVERY_BACKEND_ORDER = ("cartesian", "pilz", "ompl")


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
    final_pre_mouth_standoff_m: float = 0.080
    provisional_standoff_m: float = 0.250
    servo_tracking_max_error_m: float = 0.10
    servo_startup_max_error_m: float = 0.25
    servo_replan_enter_m: float = 0.10
    servo_replan_exit_m: float = 0.06
    hold_entry_tolerance_m: float = 0.010
    maximum_linear_speed_mps: float = 0.020
    provisional_linear_speed_mps: float = 0.010
    maximum_linear_acceleration_mps2: float = 0.10
    maximum_angular_speed_rps: float = 0.15
    orientation_correction_gain: float = 1.0
    control_gain: float = 0.8
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
        if not (
            self.servo_tracking_max_error_m
            <= self.servo_startup_max_error_m
            <= 0.30
        ):
            raise ValueError("Servo startup envelope must be within [tracking, 0.30] m")
        if not 0.0 < self.provisional_linear_speed_mps <= self.maximum_linear_speed_mps:
            raise ValueError("provisional speed must be positive and no greater than maximum")
        if self.maximum_linear_acceleration_mps2 <= 0.0:
            raise ValueError("maximum acceleration must be positive")
        if self.maximum_angular_speed_rps <= 0.0:
            raise ValueError("maximum angular speed must be positive")
        if self.orientation_correction_gain <= 0.0:
            raise ValueError("orientation correction gain must be positive")


@dataclass(frozen=True)
class ContinuousServoDecision:
    command_allowed: bool
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_rps: tuple[float, float, float]
    desired_tool0_position_m: tuple[float, float, float]
    desired_straw_tip_position_m: tuple[float, float, float]
    target_error_m: float
    target_displacement_m: float
    state: str
    hold_ready: bool
    recovery_required: bool
    fallback_reason: str | None
    safety_stop_reason: str | None
    speed_limit_mps: float


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
    straw_target = mouth + standoff_m * camera_side / distance
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
        self._startup_approach_active = True
        self._reference_target: np.ndarray | None = None

    @property
    def recovery_latched(self) -> bool:
        return self._recovery_latched

    def reset(self) -> None:
        self._last_velocity[:] = 0.0
        self._recovery_latched = False
        self._startup_approach_active = True
        self._reference_target = None

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

        standoff = (
            self.config.provisional_standoff_m
            if target.provisional
            else self.config.final_pre_mouth_standoff_m
        )
        try:
            straw_target, tool_target = camera_ray_premouth_target(
                mouth_position_m=target.predicted_position_m,
                camera_position_m=camera_position_m,
                tool_orientation_xyzw=current_tool0_orientation_xyzw,
                straw_tip_offset_tool0_m=self.straw_tip_offset_tool0_m,
                standoff_m=standoff,
            )
        except ValueError as exc:
            return self._stop(f"premouth_target_invalid:{exc}")
        error_vector = tool_target - current
        error = float(np.linalg.norm(error_vector))
        if self._reference_target is None:
            self._reference_target = target.position_m.copy()
        displacement = float(np.linalg.norm(target.position_m - self._reference_target))

        # The conservative coarse staging pose remains far enough from the
        # face for reliable RGB-D.  Its bounded startup envelope lets Servo
        # close that initial gap at limited speed without redefining the
        # 100 mm ongoing tracking limit.  Startup cannot be re-entered until
        # the controller is explicitly reset for a new run.
        if displacement > self.config.servo_tracking_max_error_m:
            self._recovery_latched = True
            return self._recovery(
                "servo_target_displacement_limit",
                tool_target,
                straw_target,
                error,
                displacement,
            )
        if self._startup_approach_active:
            if error > self.config.servo_startup_max_error_m:
                self._recovery_latched = True
                return self._recovery(
                    "servo_startup_error_limit",
                    tool_target,
                    straw_target,
                    error,
                    displacement,
                )
            if error <= self.config.servo_replan_enter_m:
                self._startup_approach_active = False

        if self._recovery_latched:
            if error > self.config.servo_replan_exit_m:
                return self._recovery("servo_error_hysteresis", tool_target, straw_target, error, displacement)
            self._recovery_latched = False
        if not self._startup_approach_active and error > self.config.servo_replan_enter_m:
            self._recovery_latched = True
            return self._recovery("servo_tracking_error_limit", tool_target, straw_target, error, displacement)
        if not self._startup_approach_active and error > self.config.servo_tracking_max_error_m:
            self._recovery_latched = True
            return self._recovery("servo_tracking_range", tool_target, straw_target, error, displacement)
        if float(np.linalg.norm(tool_target)) > self.config.maximum_tool_radius_m:
            return self._stop("tool_workspace_radius_limit")

        speed_limit = (
            self.config.provisional_linear_speed_mps
            if target.provisional
            else self.config.maximum_linear_speed_mps
        )
        if int(servo_status_code) in SERVO_DECELERATE_CODES:
            speed_limit *= 0.25
        velocity = self.config.control_gain * error_vector
        speed = float(np.linalg.norm(velocity))
        if speed > speed_limit and speed > 0.0:
            velocity *= speed_limit / speed
        dt = max(float(dt_sec), 1.0e-3)
        change = velocity - self._last_velocity
        max_change = self.config.maximum_linear_acceleration_mps2 * dt
        change_norm = float(np.linalg.norm(change))
        if change_norm > max_change and change_norm > 0.0:
            velocity = self._last_velocity + change * (max_change / change_norm)
        self._last_velocity = velocity
        angular_velocity = vertical_axis_angular_correction(
            current_tool0_orientation_xyzw,
            gain=self.config.orientation_correction_gain,
            maximum_speed_rps=self.config.maximum_angular_speed_rps,
        )
        hold_ready = bool(
            target.stable and error <= self.config.hold_entry_tolerance_m
        )
        return ContinuousServoDecision(
            command_allowed=True,
            linear_velocity_mps=tuple(float(value) for value in velocity),
            angular_velocity_rps=tuple(float(value) for value in angular_velocity),
            desired_tool0_position_m=tuple(float(value) for value in tool_target),
            desired_straw_tip_position_m=tuple(float(value) for value in straw_target),
            target_error_m=error,
            target_displacement_m=displacement,
            state=("HOLDING" if hold_ready else "TRACKING"),
            hold_ready=hold_ready,
            recovery_required=False,
            fallback_reason=None,
            safety_stop_reason=None,
            speed_limit_mps=speed_limit,
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
