"""Select a candidate or geometry, then observe each bounded action step.

For G0.5, tracking_call_id identifies a completed tool call whose tracked
point is observed in the current frame. Direction and distance describe a
measurable objective in robot-base coordinates; success is a displacement
threshold and must be followed by task and handoff verification.
Candidate summaries report end-effector geometry and, for each step, the
model's normalized gripper output and the effective open, close or retain
intent after the selected arm and gripper policy are applied. These are
commands, not measurements of object contact or grasp success.
When already near the target, a small predicted movement is not by itself a
reason to reject a candidate. Compare its bounded motion, target geometry and
effective gripper intent with the current subtask; when they are compatible,
consider a short prefix of up to four steps within the declared maximum, then
inspect the measured result.
Choose geometry when the candidate is incompatible or current evidence does
not support trying it. Never describe generated candidates as executed steps.
"""
NAME = 'g05_select_chunk'
SCHEMA = {'type': 'object', 'properties': {
    'batch_id': {'type': 'string'}, 'route': {'enum': ['g05', 'geometry']},
    'candidate_index': {'type': ['integer', 'null']}, 'reason': {'type': 'string'},
    'tracking_call_id': {'type': ['string', 'null']}, 'point': {'type': ['string', 'null']},
    'direction': {'type': ['array', 'null'], 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
    'target_m': {'type': ['number', 'null']}, 'maximum_steps': {'type': 'integer', 'minimum': 1, 'maximum': 8}},
    'required': ['batch_id', 'route', 'candidate_index', 'reason', 'tracking_call_id', 'point',
                 'direction', 'target_m', 'maximum_steps'], 'additionalProperties': False}

if __name__ == '__main__':
    raise RuntimeError('g05_select_chunk requires the orchestrated host')
