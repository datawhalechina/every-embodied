"""Recorded, contact-free kinematic translation of a piano and its supports."""

import numpy as np

from g05_recovery import piano_shift


class PianoTranslation:
    def __init__(self, model):
        import mujoco
        self.model = model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "piano/")
        if self.body < 0 or model.body_parentid[self.body] != 0:
            raise ValueError("Expected piano root attached to the world")
        self.legs = [i for i in range(model.ngeom) if
                     (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("piano_leg_")]
        if len(self.legs) != 2 or any(model.geom_bodyid[i] != 0 for i in self.legs):
            raise ValueError("Expected two world-attached piano supports")
        self.origin = model.body_pos[self.body].copy()
        self.leg_origins = model.geom_pos[self.legs].copy()

    def apply(self, shift):
        shift = piano_shift(shift)
        self.model.body_pos[self.body] = self.origin + shift
        self.model.geom_pos[self.legs] = self.leg_origins + shift


def score_with_pause(score, boundary, pause):
    import copy
    from polyphonic_score import validate_score
    if not all(np.isfinite(x) and x > 0 for x in (boundary, pause)):
        raise ValueError("Invalid pause")
    result = copy.deepcopy(score)
    beat = 60 / score["tempo_bpm"]
    for note in result["notes"]:
        start = 1 + note["start_beat"] * beat
        end = start + note["duration_beats"] * beat
        if start < boundary < end:
            raise ValueError("Pause must not cut a sustained score note")
        if start >= boundary:
            note["start_beat"] += pause / beat
    validate_score(result, song_mode=True)
    return result
