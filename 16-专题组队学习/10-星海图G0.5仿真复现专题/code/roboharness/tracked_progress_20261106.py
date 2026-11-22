"""Monitor an observation-bound tracked point during a bounded action chunk.

Displacements use the fixed robot-base frame. The caller must ground the
point before executing and ensure the base remains fixed throughout the chunk.
The tracker reports visual measurements, not object identity or grasp success.
"""
import math


def _track(binding, tracking, name, minimum_confidence):
    if not isinstance(tracking, dict) or tracking.get('available') is not True:
        raise ValueError('Tracking is unavailable')
    if tracking.get('xyz_in_robot_base_coord_frame') != 'current_robot_base':
        raise ValueError('Unexpected tracked coordinate frame')
    point = (tracking.get('tracks') or {}).get(name)
    if not isinstance(point, dict) or point.get('status') != 'observed':
        raise ValueError('Tracked point is not observed')
    if point.get('episode_id') != binding['episode_id'] or point.get('observation_sequence') != binding['sequence']:
        raise ValueError('Tracked point belongs to a different observation')
    confidence = point.get('confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < minimum_confidence:
        raise ValueError('Tracked point confidence is below threshold')
    xyz = point.get('xyz_in_robot_base_coord_m')
    if (not isinstance(xyz, list) or len(xyz) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz)):
        raise ValueError('Tracked point has no valid position')
    if not isinstance(point.get('track_id'), str) or not point['track_id']:
        raise ValueError('Tracked point has no identity')
    return point, [float(x) for x in xyz]


class TrackedProgress:
    """Stop on lost or inconsistent tracking, or a measured displacement goal."""

    def __init__(self, binding, tracking, name, direction, target_m,
                 *, minimum_confidence=0.5, max_step_m=0.10, stall_window=0, min_progress_m=0.0):
        if not isinstance(name, str) or not name:
            raise ValueError('A grounded point name is required')
        if (not isinstance(direction, (list, tuple)) or len(direction) != 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in direction)):
            raise ValueError('A finite displacement direction is required')
        norm = math.sqrt(sum(x*x for x in direction))
        if norm < 1e-9:
            raise ValueError('Displacement direction cannot be zero')
        for value in (target_m, max_step_m):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Distance thresholds must be finite and positive')
        if type(minimum_confidence) not in (int, float) or not 0 <= minimum_confidence <= 1:
            raise ValueError('Invalid confidence threshold')
        if (type(stall_window) is not int or stall_window not in {0, 2, 3, 4, 5, 6, 7, 8} or
                type(min_progress_m) not in (int, float) or not math.isfinite(min_progress_m) or
                min_progress_m < 0 or (stall_window == 0 and min_progress_m != 0)):
            raise ValueError('Invalid bounded stall guard')
        point, xyz = _track(binding, tracking, name, minimum_confidence)
        self.name, self.track_id = name, point['track_id']
        self.start, self.previous = xyz, xyz
        self.sequence = binding['sequence']
        self.direction = [float(x / norm) for x in direction]
        self.target_m, self.max_step_m = float(target_m), float(max_step_m)
        self.minimum_confidence = minimum_confidence
        self.stall_window, self.min_progress_m = stall_window, float(min_progress_m)
        self.progress_history = [0.0]

    def __call__(self, before, after, receipt):
        evidence = {'point': self.name, 'sequence_before': before['sequence'],
                    'sequence_after': after['sequence']}
        try:
            if before['sequence'] != self.sequence or after['sequence'] != self.sequence + 1:
                raise ValueError('Progress observations are not consecutive')
            point, xyz = _track(after, receipt.get('persistent_tracking'), self.name, self.minimum_confidence)
            if point['track_id'] != self.track_id:
                raise ValueError('Tracked point identity changed')
            step_m = math.dist(xyz, self.previous)
            if step_m > self.max_step_m:
                raise ValueError('Tracked point moved beyond the per-step limit')
            progress = sum((x - start) * axis for x, start, axis in zip(xyz, self.start, self.direction))
            evidence.update(position_m=xyz, initial_position_m=self.start,
                            track_id=self.track_id, direction=self.direction,
                            target_m=self.target_m, step_displacement_m=step_m,
                            directed_displacement_m=progress, confidence=point['confidence'])
            self.previous, self.sequence = xyz, after['sequence']
            self.progress_history.append(progress)
            if progress >= self.target_m:
                decision = 'stop_success'
            elif (self.stall_window and len(self.progress_history) > self.stall_window and
                  progress - self.progress_history[-1-self.stall_window] < self.min_progress_m):
                decision = 'stop_replan'
                evidence.update(reason='insufficient_progress', stall_window=self.stall_window,
                                min_progress_m=self.min_progress_m)
            else:
                decision = 'continue'
            return {'decision': decision, 'evidence': evidence}
        except (KeyError, TypeError, ValueError) as error:
            evidence['reason'] = str(error)
            return {'decision': 'stop_replan', 'evidence': evidence}


class TrackedGuard:
    """Guard target visibility and tracking continuity during a pickup chunk."""

    def __init__(self, binding, tracking, name, *, minimum_confidence=0.5, max_step_m=0.10):
        point, xyz = _track(binding, tracking, name, minimum_confidence)
        self.name, self.track_id, self.previous = name, point['track_id'], xyz
        self.sequence, self.minimum_confidence, self.max_step_m = binding['sequence'], minimum_confidence, max_step_m

    def __call__(self, before, after, receipt):
        evidence = {'point': self.name, 'sequence_before': before['sequence'], 'sequence_after': after['sequence']}
        try:
            if before['sequence'] != self.sequence or after['sequence'] != self.sequence + 1:
                raise ValueError('Progress observations are not consecutive')
            point, xyz = _track(after, receipt.get('persistent_tracking'), self.name, self.minimum_confidence)
            if point['track_id'] != self.track_id:
                raise ValueError('Tracked point identity changed')
            step_m = math.dist(xyz, self.previous)
            if step_m > self.max_step_m:
                raise ValueError('Tracked point moved beyond the per-step limit')
            self.previous, self.sequence = xyz, after['sequence']
            evidence.update(position_m=xyz, track_id=self.track_id, step_displacement_m=step_m,
                            confidence=point['confidence'])
            return {'decision': 'continue', 'evidence': evidence}
        except (KeyError, TypeError, ValueError) as error:
            evidence['reason'] = str(error)
            return {'decision': 'stop_replan', 'evidence': evidence}
