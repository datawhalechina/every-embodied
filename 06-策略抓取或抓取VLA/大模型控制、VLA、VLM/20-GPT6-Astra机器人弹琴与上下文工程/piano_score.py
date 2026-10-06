"""Limits of the current one-index-finger-per-hand contact controller."""

from piano_context import validate_plan


CONTACT_REGISTERS = {"left": (48, 60), "right": (57, 69)}
WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}


class ContactRelease:
    """Release a scheduled note after measured activation, not after imagined success."""

    def __init__(self, parts, tempo_bpm):
        self.parts = parts
        self.beat_seconds = 60.0 / tempo_bpm
        self.fired = {side: {} for side in parts}

    def current(self, side, time):
        for index, note in enumerate(self.parts[side]):
            start = 1.0 + note["start_beat"] * self.beat_seconds
            end = start + note["duration_beats"] * self.beat_seconds
            if start <= time < end:
                return index, note, start
        return None

    def observe(self, time, active_midi):
        for side in self.parts:
            current = self.current(side, time)
            if current is not None:
                index, note, start = current
                if time >= start + 0.5 and note["midi"] in active_midi:
                    self.fired[side].setdefault(index, time)

    def height(self, side, time, planned_height):
        from physical_piano import smooth

        current = self.current(side, time)
        if current is None or current[0] not in self.fired[side]:
            return planned_height
        elapsed = time - self.fired[side][current[0]]
        return max(planned_height, 0.008 + smooth(elapsed / 0.12) * (0.080 - 0.008))

    def tail_lift(self, side, time):
        from physical_piano import smooth

        last = len(self.parts[side]) - 1
        if last not in self.fired[side]:
            return 0.0
        return 0.22 * smooth((time - self.fired[side][last] - 0.2) / 0.3)


def validate_contact_score(plan):
    report = validate_plan(plan)
    beat_seconds = 60.0 / plan["tempo_bpm"]
    pitches = {side: {note["midi"] for note in plan["notes"] if note["hand"] == side} for side in ("left", "right")}
    if pitches["left"] & pitches["right"]:
        raise ValueError("This evaluator requires disjoint left/right pitch sets")
    last_end = {"left": -1.0, "right": -1.0}
    for note in plan["notes"]:
        side = note["hand"]
        low, high = CONTACT_REGISTERS[side]
        if not low <= note["midi"] <= high:
            raise ValueError(f"{side} part exceeds the teaching register [{low}, {high}]")
        if note["midi"] % 12 not in WHITE_PITCH_CLASSES:
            raise ValueError("Black-key contact geometry is not supported by this controller")
        if note["start_beat"] < last_end[side] - 1e-9:
            raise ValueError(f"{side} part is polyphonic; supply an explicit monophonic arrangement")
        if note["duration_beats"] * beat_seconds < 0.85 - 1e-9:
            raise ValueError("Each note window needs at least 0.85 s for approach, press and release")
        last_end[side] = note["start_beat"] + note["duration_beats"]
    if report["duration_seconds"] > 60:
        raise ValueError("Start with an excerpt no longer than 60 seconds after time stretching")
    return {**report, "controller": "one_index_finger_per_active_hand", "timing": "staccato_note_windows_not_full_MIDI_sustain"}


def compare_timing(onsets, notes, tempo_bpm, tolerance=0.05):
    """Require every measured event, including duplicates, to occupy its score window."""
    beat_seconds = 60.0 / tempo_bpm
    if len(onsets) != len(notes):
        return False
    for event, note in zip(onsets, notes):
        start = 1.0 + note["start_beat"] * beat_seconds + 0.5
        end = 1.0 + (note["start_beat"] + note["duration_beats"]) * beat_seconds - 0.18
        if event["midi"] != note["midi"] or not start - tolerance <= event["time_seconds"] <= end + tolerance:
            return False
    return True


def rest_lift(time, notes, tempo_bpm):
    """Park an idle hand above the playing hand, with a smooth pre-positioning ramp."""
    from physical_piano import smooth

    beat_seconds = 60.0 / tempo_bpm
    previous_end = None
    for note in notes:
        start = 1.0 + note["start_beat"] * beat_seconds
        end = start + note["duration_beats"] * beat_seconds
        if start <= time <= end:
            return 0.0
        if time < start:
            if previous_end is None and start <= 1.0:
                return 0.0
            up = 1.0 if previous_end is None else smooth((time - previous_end) / 0.3)
            return 0.22 * min(up, smooth((start - time) / 0.5))
        previous_end = end
    return 0.22 * smooth((time - previous_end) / 0.3) if previous_end is not None else 0.22
