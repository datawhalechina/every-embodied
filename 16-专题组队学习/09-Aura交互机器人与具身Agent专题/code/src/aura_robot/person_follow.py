"""Guarded person-following policy for the fixed Aura face camera.

The FrontierX/RDK documentation describes an on-device ``body_tracking_node``
that consumes a body bounding box and publishes ``/follow_cmd_vel``.  This
module is the SDK-neutral policy layer behind the same idea: a target to the
right of the image produces a right turn (ROS REP-103: negative ``angular.z``),
a target to the left produces a left turn, and forward motion is allowed only
after the target is centered.  It deliberately does not publish ROS messages
or call Aura motion endpoints.

The defaults are interaction limits, not the vendor's faster 0.1 m/s and
0.3 rad/s tracking defaults.  A real actuator adapter must add independent
obstacle, battery, stale-frame, and emergency-stop gates before sending them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class FollowState(StrEnum):
    STOP = "stop"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    APPROACH = "approach"
    HOLD = "hold"


@dataclass(frozen=True)
class PersonTarget:
    """A target extracted from a body detector or a tracking model."""

    center_x_px: float
    height_px: float | None = None
    confidence: float = 1.0
    visible: bool = True


@dataclass(frozen=True)
class FollowDecision:
    state: FollowState
    linear_x: float
    angular_z: float
    offset_px: float | None
    reason: str


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(float(value))


def stop_decision(reason: str, *, offset_px: float | None = None) -> FollowDecision:
    return FollowDecision(FollowState.STOP, 0.0, 0.0, offset_px, reason)


def decide_follow(
    target: PersonTarget | None,
    *,
    image_width: float = 960.0,
    center_tolerance_px: float = 20.0,
    min_confidence: float = 0.5,
    lost_frames: int = 0,
    lost_frame_limit: int = 30,
    approach_height_px: float | None = None,
    interaction_linear_m_s: float = 0.02,
    interaction_angular_rad_s: float = 0.12,
) -> FollowDecision:
    """Return one bounded, explainable interaction command.

    ``approach_height_px`` is intentionally optional.  The public node docs do
    not define a metric distance calibration, so callers must provide a camera
    and detector-specific height threshold before forward motion is enabled.
    """
    numeric = (image_width, center_tolerance_px, min_confidence,
               interaction_linear_m_s, interaction_angular_rad_s)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("follow limits must be finite")
    if image_width <= 0 or center_tolerance_px < 0 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid image or confidence limits")
    if interaction_linear_m_s < 0 or interaction_angular_rad_s < 0:
        raise ValueError("interaction speeds must be non-negative")
    if lost_frames < 0 or lost_frame_limit < 0:
        raise ValueError("lost frame count out of bounds")
    if not _finite(approach_height_px) or (approach_height_px is not None and approach_height_px < 0):
        raise ValueError("approach_height_px must be non-negative or None")
    if target is None or not target.visible:
        return stop_decision("target_not_visible")
    if not _finite(target.center_x_px) or not _finite(target.height_px) or not _finite(target.confidence):
        raise ValueError("target fields must be finite")
    if target.confidence < min_confidence:
        return stop_decision("target_confidence_low")
    if lost_frames >= lost_frame_limit and lost_frame_limit > 0:
        return stop_decision("target_lost_limit")
    center = image_width / 2.0
    offset = float(target.center_x_px) - center
    if abs(offset) > center_tolerance_px:
        # Image x grows to the right.  REP-103 angular.z > 0 is left/CCW,
        # therefore a person on the right needs a negative angular command.
        if offset > 0:
            return FollowDecision(FollowState.TURN_RIGHT, 0.0,
                                  -interaction_angular_rad_s, offset,
                                  "target_right_of_center")
        return FollowDecision(FollowState.TURN_LEFT, 0.0,
                              interaction_angular_rad_s, offset,
                              "target_left_of_center")
    if approach_height_px is not None and target.height_px is not None and target.height_px < approach_height_px:
        return FollowDecision(FollowState.APPROACH, interaction_linear_m_s, 0.0,
                              offset, "target_centered_and_far")
    return FollowDecision(FollowState.HOLD, 0.0, 0.0, offset, "target_centered")


@dataclass
class GestureFollowGate:
    """Start/stop gate matching the documented hand-gesture IDs."""

    start_gesture_id: int = 5  # open palm
    stop_gesture_id: int = 11  # OK
    armed: bool = False

    def update(self, gesture_id: int | None) -> bool:
        if gesture_id == self.start_gesture_id:
            self.armed = True
        elif gesture_id == self.stop_gesture_id:
            self.armed = False
        return self.armed

    def decide(self, target: PersonTarget | None, **kwargs: object) -> FollowDecision:
        if not self.armed:
            return stop_decision("follow_not_armed")
        return decide_follow(target, **kwargs)


def target_from_lightnav_response(response: object) -> PersonTarget | None:
    """Adapt a LightNav tracking response's ``pointing.opos_px`` to this policy.

    The adapter is intentionally strict: a tracking server that only emits
    waypoints, or marks the target as not visible, cannot silently authorize
    motion.  ``height_px`` remains unknown because LightNav's pointing output
    is a pixel location rather than a calibrated body box.
    """
    if not isinstance(response, dict):
        return None
    data = response.get("data")
    if not isinstance(data, dict) or data.get("rc") != 0:
        return None
    pointing = data.get("pointing")
    if not isinstance(pointing, dict) or data.get("visible") is not True:
        return None
    point = pointing.get("opos_px")
    frame_size = pointing.get("frame_size")
    if (not isinstance(point, list) or len(point) != 2 or
            not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in point)):
        return None
    if not isinstance(frame_size, list) or len(frame_size) != 2:
        return None
    width, height = frame_size
    if (not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in frame_size)
            or width <= 0 or height <= 0):
        return None
    return PersonTarget(center_x_px=float(point[0]), confidence=1.0, visible=True)
