from robot_layer.arm_ur10e.perception.mouth_target_tracker import MouthTargetTracker, TrackingState
from .motion_backend import CartesianBackend, OmplBackend, PilzBackend
from .replanning_manager import ReplanningManager
from .safety_monitor import SafetyMonitor
from .tracking_controller import TrackingController


def _target(displacement=0.0):
    tracker = MouthTargetTracker(replan_distance_m=0.09)
    tracker.update([0.3, 0.4, 1.0], timestamp_sec=1.0)
    return tracker, tracker.update([0.3 + displacement, 0.4, 1.0], timestamp_sec=1.1)


def test_tracking_controller_is_plan_only():
    tracker, target = _target(0.01)
    decision = TrackingController(tracker, SafetyMonitor(), CartesianBackend()).correct(target)
    assert decision.result.success and decision.result.planned
    assert not decision.result.executed


def test_large_displacement_uses_recovery_order():
    tracker, target = _target(0.12)
    manager = ReplanningManager(tracker, SafetyMonitor(max_speed_mps=1.0), CartesianBackend(), PilzBackend(), OmplBackend())
    decision = manager.plan_recovery(target)
    assert target.state == TrackingState.ABORTED
    assert decision.backend_order == ("cartesian", "pilz", "ompl")
    assert decision.result.backend == "cartesian"


def test_safety_rejects_lost_target():
    tracker, target = _target(0.01)
    lost = tracker.snapshot(now_sec=2.0)
    decision = TrackingController(tracker, SafetyMonitor(), CartesianBackend()).correct(lost)
    assert not decision.result.success
    assert "target_state" in decision.safety_reason
