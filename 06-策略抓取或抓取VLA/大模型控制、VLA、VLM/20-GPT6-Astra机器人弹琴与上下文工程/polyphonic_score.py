"""Finger-assigned polyphony with measured onset, release, and chord validation."""

import math

from physical_piano import smooth
from piano_context import validate_plan


FINGERS = ("thumb", "index", "middle", "ring", "little")


class KeyCycles:
    """A score-independent 90% press / 50% release key-travel sensor."""

    def __init__(self, count=88):
        self.active = [False] * count

    def update(self, travel):
        if len(travel) != len(self.active) or any(not math.isfinite(float(x)) for x in travel):
            raise ValueError("Expected finite normalized key travel")
        changes = []
        for index, depth in enumerate(travel):
            was = self.active[index]
            now = depth > 0.50 if was else depth >= 0.90
            self.active[index] = now
            if now != was:
                changes.append((index, now))
        return changes


def validate_score(score, song_mode=False):
    if not isinstance(score, dict) or set(score) != {"title", "tempo_bpm", "notes"}:
        raise ValueError("Expected title, tempo_bpm, notes")
    if not isinstance(score["notes"], list):
        raise ValueError("Expected a list of notes")
    plain = dict(score, notes=[])
    occupied = {}
    for note in score["notes"]:
        required = {"midi", "start_beat", "duration_beats", "hand", "finger"}
        optional = {"release_lead_seconds"} if song_mode else set()
        if not isinstance(note, dict) or not required <= set(note) or set(note) - required - optional:
            raise ValueError("Each note needs pitch, onset, duration, hand and finger")
        if note["finger"] not in FINGERS:
            raise ValueError("Unknown finger")
        plain["notes"].append({k: v for k, v in note.items() if k != "finger"})
    if song_mode:
        from song_score import validate_song
        result = validate_song(plain)
    else:
        result = validate_plan(plain)
    seconds = 60.0 / score["tempo_bpm"]
    for note in score["notes"]:
        identity = (note["hand"], note["finger"])
        start, end = note["start_beat"] * seconds, (note["start_beat"] + note["duration_beats"]) * seconds
        if start < occupied.get(identity, -1) - 1e-9:
            raise ValueError("One finger cannot hold two keys at once")
        occupied[identity] = end
        if not song_mode and note["midi"] % 12 not in {0, 2, 4, 5, 7, 9, 11}:
            raise ValueError("This scene currently calibrates white keys only")
        if not song_mode and end - start < 0.16:
            raise ValueError("Use notes at least 160 ms long")
    if not song_mode and result["duration_seconds"] > 60:
        raise ValueError("Use an excerpt of at most 60 seconds")
    return result


def performance_passed(report):
    return (all(report.get(key) is True for key in
                ("note_count_match", "timing_match", "chords_match", "finger_matches"))
            and report.get("interhand_contact_steps") == 0)


def finger_target(time, notes, key_y, home, beat_seconds, warmup=1.0,
                  hover=0.950, pressed=0.908):
    """Move while released, anticipate onset, and hold until the written release."""
    t = time - warmup
    y = key_y[home]
    previous_end = -math.inf
    for note in notes:
        start = note["start_beat"] * beat_seconds
        end = start + note["duration_beats"] * beat_seconds
        if t < end:
            destination = key_y[note["midi"]]
            move_start = max(previous_end + 0.10, start - 0.45)
            progress = smooth((t - move_start) / max(0.05, start - 0.10 - move_start))
            y += progress * (destination - y)
            # Travel to the key before the onset; press and release on their own clocks.
            press = smooth((t - (start - 0.09)) / 0.09)
            if t < start - 0.09:
                press = 0.0
            return y, hover + press * (pressed - hover)
        release = smooth((t - end) / 0.08)
        y = key_y[note["midi"]]
        if t < end + 0.08:
            return y, pressed + release * (hover - pressed)
        previous_end = end
    return y, hover


def evaluate(score, events, warmup=1.0, onset_tolerance=0.12, release_tolerance=0.16, song_mode=False):
    validate_score(score, song_mode=song_mode)
    beat = 60.0 / score["tempo_bpm"]
    expected = [{**n, "on": warmup + n["start_beat"] * beat,
                 "off": warmup + (n["start_beat"] + n["duration_beats"]) * beat} for n in score["notes"]]
    active, measured, malformed = {}, [], []
    for event in events:
        pitch, time = event["midi"], event["time_seconds"]
        if event["type"] == "NoteOn":
            if pitch in active:
                malformed.append(event)
            active[pitch] = event
        elif event["type"] == "NoteOff":
            if pitch not in active:
                malformed.append(event)
                continue
            measured.append({"midi": pitch, "on": active.pop(pitch)["time_seconds"], "off": time})
    malformed.extend(active.values())
    # Match within the onset window before diagnostic pitch-only matching below.
    # Otherwise a missed C can steal a much later C and inflate the matched count.
    timed_pairs = []
    for pitch in sorted({n["midi"] for n in expected}):
        wanted = sorted((n for n in expected if n["midi"] == pitch), key=lambda n: n["on"])
        heard = sorted((n for n in measured if n["midi"] == pitch), key=lambda n: n["on"])
        i = j = 0
        while i < len(wanted) and j < len(heard):
            delta = heard[j]["on"] - wanted[i]["on"]
            if delta < -onset_tolerance:
                j += 1
            elif delta > onset_tolerance:
                i += 1
            else:
                timed_pairs.append((wanted[i], heard[j]))
                i += 1
                j += 1
    true_positive = len(timed_pairs)
    onset_metrics = {"true_positive": true_positive, "expected_count": len(expected),
                     "measured_count": len(measured),
                     "precision": true_positive / max(1, len(measured)),
                     "recall": true_positive / max(1, len(expected)),
                     "f1": 2 * true_positive / max(1, len(expected) + len(measured)),
                     "tolerance_seconds": onset_tolerance}
    unmatched = list(range(len(measured)))
    matched, missing = [], []
    for note in expected:
        candidates = [i for i in unmatched if measured[i]["midi"] == note["midi"]]
        if not candidates:
            missing.append(note)
            continue
        best = min(candidates, key=lambda i: abs(measured[i]["on"] - note["on"]))
        unmatched.remove(best)
        actual = measured[best]
        matched.append({"midi": note["midi"], "hand": note["hand"], "finger": note["finger"],
                        "actual_on_seconds": actual["on"], "actual_off_seconds": actual["off"],
                        "onset_error_seconds": actual["on"] - note["on"],
                        "release_error_seconds": actual["off"] - note["off"],
                        "expected_hold_seconds": note["off"] - note["on"],
                        "measured_hold_seconds": actual["off"] - actual["on"]})
    note_count_match = not (missing or unmatched or malformed)
    timing_match = note_count_match and all(abs(n["onset_error_seconds"]) <= onset_tolerance and abs(n["release_error_seconds"]) <= release_tolerance for n in matched)
    chords = []
    for onset in sorted({n["on"] for n in expected}):
        pitches = {n["midi"] for n in expected if abs(n["on"] - onset) < 1e-8}
        if len(pitches) < 2:
            continue
        candidates = [n for n in measured if n["midi"] in pitches and abs(n["on"] - onset) <= onset_tolerance]
        intersection = max(0.0, min((n["off"] for n in candidates), default=0) - max((n["on"] for n in candidates), default=0))
        expected_overlap = min(n["off"] for n in expected if abs(n["on"] - onset) < 1e-8) - onset
        chords.append({"midi": sorted(pitches), "expected_onset_seconds": onset,
                       "expected_overlap_seconds": expected_overlap,
                       "measured_overlap_seconds": intersection,
                       "passed": {n["midi"] for n in candidates} == pitches and intersection >= max(0.06, expected_overlap - onset_tolerance - release_tolerance)})
    return {"note_count_match": note_count_match, "timing_match": timing_match,
            "onset_metrics": onset_metrics,
            "chords_match": all(c["passed"] for c in chords), "chords": chords,
            "matched": matched, "missing": missing, "unexpected": [measured[i] for i in unmatched],
            "malformed": malformed, "onset_tolerance_seconds": onset_tolerance,
            "release_tolerance_seconds": release_tolerance}
