"""Import a user-authorized performance with explicit adaptation accounting."""

import argparse
from collections import defaultdict, deque
import hashlib
import itertools
import json
import math
from pathlib import Path

from midi_score import load_midi
from piano_context import write_json
from polyphonic_score import FINGERS


def validate_song(score):
    from piano_context import number
    if not isinstance(score["title"], str) or not score["title"].strip():
        raise ValueError("Song title must be nonempty")
    number(score["tempo_bpm"], "tempo", 30, 180)
    if not 1 <= len(score["notes"]) <= 10000:
        raise ValueError("Song requires 1..10000 notes")
    previous, ends = -1., {}
    for n in score["notes"]:
        if type(n["midi"]) is not int or not 21 <= n["midi"] <= 108 or n["hand"] not in ("left", "right"):
            raise ValueError("Invalid pitch or hand")
        number(n["start_beat"], "onset", 0, 1800)
        number(n["duration_beats"], "duration", .05, 60)
        if "release_lead_seconds" in n:
            number(n["release_lead_seconds"], "note release lead", 0, .2)
        if n["start_beat"] < previous or n["start_beat"] < ends.get(n["midi"], -1) - 1e-8:
            raise ValueError("Unsorted or overlapping physical key events")
        previous = n["start_beat"]
        ends[n["midi"]] = previous + n["duration_beats"]
    duration = max(ends.values()) * 60 / score["tempo_bpm"]
    if duration > 900:
        raise ValueError("Song exceeds 15-minute simulation limit")
    return dict(note_count=len(score["notes"]), duration_seconds=duration)


def key_coordinate(pitch):
    return (pitch // 12) * 7 + (0, .5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6)[pitch % 12]


def performance_notes(path):
    active, notes, now = defaultdict(deque), [], 0.
    for event in load_midi(path):
        now += event.time
        if event.type == "control_change" and event.control in (64, 66, 67) and event.value:
            raise ValueError("Pedal-dependent MIDI requires a separate explicit arrangement")
        if event.type == "pitchwheel" and event.pitch:
            raise ValueError("Pitch-bent MIDI cannot be reproduced by discrete piano keys")
        if event.type not in ("note_on", "note_off"):
            continue
        if event.channel == 9:
            raise ValueError("Percussion is not a piano part")
        key = (event.channel, event.note)
        if event.type == "note_on" and event.velocity:
            active[key].append((now, event.velocity))
        else:
            if not active[key]:
                raise ValueError("Unmatched note-off")
            start, velocity = active[key].popleft()
            if now <= start:
                raise ValueError("Non-positive note duration")
            notes.append(dict(midi=event.note, start=start, end=now, velocity=velocity))
    if any(active.values()) or not notes:
        raise ValueError("Incomplete or empty MIDI")
    return sorted(notes, key=lambda n: (n["start"], n["midi"]))


def merge_same_keys(notes):
    """One physical key cannot sustain overlapping voices independently."""
    by_pitch = defaultdict(list)
    for note in notes:
        by_pitch[note["midi"]].append(dict(note))
    output, merged = [], 0
    for pitch, group in by_pitch.items():
        current = None
        for note in sorted(group, key=lambda n: n["start"]):
            if current is not None and note["start"] < current["end"] - 1e-8:
                current["end"] = max(current["end"], note["end"])
                merged += 1
            else:
                if current is not None:
                    output.append(current)
                current = note
        output.append(current)
    return sorted(output, key=lambda n: (n["start"], n["midi"])), merged


def assign_fingers(notes):
    result = []
    for side in ("left", "right"):
        part = [dict(n) for n in notes if n["hand"] == side]
        if not part:
            continue
        offsets = [4, 3, 2, 1, 0] if side == "left" else [0, 1, 2, 3, 4]
        base = min(key_coordinate(n["midi"]) for n in part[:5])
        beam = [(0., [None] * 5, base, [])]
        for _, batch in itertools.groupby(part, key=lambda n: round(n["start"], 4)):
            group = list(batch)
            start = group[0]["start"]
            candidates = []
            for cost, previous, base, history in beam:
                free = [i for i, n in enumerate(previous) if n is None or n["end"] + .055 <= start + 1e-8]
                held = [(i, n) for i, n in enumerate(previous) if n is not None and n["end"] > start]
                for choice in itertools.permutations(free, len(group)):
                    if any(previous[i] is not None
                           and previous[i]["end"] + .055 + min(.165, .035 * abs(
                               key_coordinate(previous[i]["midi"]) - key_coordinate(n["midi"])))
                           > start + 1e-8 for i, n in zip(choice, group)):
                        continue
                    placed = [*held, *zip(choice, group)]
                    order = sorted(placed, key=lambda item: offsets[item[0]])
                    if any(a[1]["midi"] >= b[1]["midi"] for a, b in zip(order, order[1:])):
                        continue
                    bases = [key_coordinate(n["midi"]) - offsets[i] for i, n in placed]
                    center = sum(bases) / len(bases)
                    spread = sum((b - center)**2 for b in bases)
                    travel = .2 * (center - base)**2
                    prev, added = list(previous), []
                    for index, n in zip(choice, group):
                        note = dict(n, finger=FINGERS[index])
                        prev[index] = note
                        added.append(note)
                    candidates.append((cost + spread * 4 + travel, prev, center, history + added))
            if not candidates:
                raise ValueError(f"No non-crossing fingering at {start:.3f}s for {side}; explicit rearrangement required")
            unique = {}
            for candidate in sorted(candidates, key=lambda item: item[0]):
                identity = tuple((n["midi"], n["end"]) if n and n["end"] + .055 > start else None
                                 for n in candidate[1])
                unique.setdefault(identity, candidate)
                if len(unique) >= 64:
                    break
            beam = list(unique.values())
        result.extend(min(beam, key=lambda item: item[0])[3])
    return sorted(result, key=lambda n: (n["start"], n["hand"], n["midi"]))


def prepare(path, title, stretch=1., seconds=None, fold_registers=False, compact=False):
    if not math.isfinite(stretch) or not 1 <= stretch <= 3:
        raise ValueError("Time stretch must be in [1, 3]")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Excerpt duration must be positive")
    original = performance_notes(path)
    selected, shifted, cut = [], 0, 0
    for source in original:
        if seconds is not None and source["start"] >= seconds:
            continue
        note = dict(source)
        if seconds is not None and note["end"] > seconds:
            note["end"] = seconds
            cut += 1
        note["hand"] = "left" if note["midi"] < 60 else "right"
        if fold_registers or compact:
            low, high = (48, 59) if note["hand"] == "left" else (60, 71 if compact else 79)
            before = note["midi"]
            while note["midi"] < low:
                note["midi"] += 12
            while note["midi"] > high:
                note["midi"] -= 12
            shifted += before != note["midi"]
        note["start"] *= stretch
        note["end"] *= stretch
        selected.append(note)
    if not selected:
        raise ValueError("No notes selected")
    merged, merges = merge_same_keys(selected)
    assigned = assign_fingers(merged)
    score = {"title": title, "tempo_bpm": 60, "notes": [
        dict(midi=n["midi"], hand=n["hand"], finger=n["finger"],
             start_beat=round(n["start"], 8), duration_beats=round(n["end"]-n["start"], 8))
        for n in assigned]}
    report = dict(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_note_count=len(original), selected_note_count=len(selected), output_note_count=len(assigned),
        octave_shifted_notes=shifted, overlapping_same_key_voices_merged=merges,
        register_adaptation="one octave per hand" if compact else ("bounded two-hand register" if fold_registers else "none"),
        excerpt_boundary_releases=cut, source_excerpt_seconds=seconds, time_stretch=stretch,
        hand_assignment="pitch below MIDI60 = left; not verified source fingering",
        dynamics="fixed velocity during physical playback, source velocity not reproduced",
        authorization="user states permission; tool does not independently establish redistribution rights",
        song_execution_verified=False)
    return score, report


def fingering_geometry(score, key_y, max_span=None):
    """Screen overlapping held notes, including human-timed (non-grid) chords.

    A lateral span is a necessary check, not a full robot reachability test.
    The caller must supply its own calibrated limit to request a rejection.
    """
    if max_span is not None and (not math.isfinite(max_span) or max_span <= 0):
        raise ValueError("Hand span limit must be positive and finite")
    beat = 60 / score["tempo_bpm"]
    result = dict(max_hand_span_m=max_span, robot_reachability_verified=False, hands={})
    for side in ("left", "right"):
        notes = [n for n in score["notes"] if n["hand"] == side]
        worst = dict(span_m=0., time_seconds=None, midi=[], fingers=[])
        violations = []
        for t in sorted({n["start_beat"] for n in notes}):
            held = [n for n in notes if n["start_beat"] <= t
                    < n["start_beat"] + n["duration_beats"] - 1e-9]
            values = [float(key_y[n["midi"]]) for n in held]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("Key coordinates must be finite")
            span = max(values) - min(values) if values else 0.
            row = dict(span_m=span, time_seconds=t*beat,
                       midi=[n["midi"] for n in held], fingers=[n["finger"] for n in held])
            if span > worst["span_m"]:
                worst = row
            if max_span is not None and span > max_span + 1e-9:
                violations.append(row)
        result["hands"][side] = dict(worst=worst, violations=violations)
    result["within_supplied_limit"] = (None if max_span is None else
        not any(hand["violations"] for hand in result["hands"].values()))
    return result


def staged_finger_target(time, notes, key_y, resting, beat, hover, pressed, release_lead=.09):
    """Separate lateral travel, attack, hold and release without skipping notes.

    This schedules command targets, not measured contact. Insufficient transition
    time delays a command instead of sliding a depressed finger across keys.
    """
    import numpy as np
    from physical_piano import smooth
    previous_xy = np.asarray(resting[:2], dtype=float)
    ready = -math.inf
    for note in notes:
        start = note["start_beat"] * beat
        end = start + note["duration_beats"] * beat
        black = note["midi"] % 12 not in {0, 2, 4, 5, 7, 9, 11}
        destination = np.array([.467 if black else resting[0], key_y[note["midi"]]])
        move_start = max(ready, start - .35)
        move_end = max(move_start + .02, start - .14)
        attack_end = max(move_end + .02, start)
        # Short attacks and wide shifts may need separate release calibration.
        note_release_lead = note.get("release_lead_seconds", release_lead)
        release_start = max(attack_end, end - note_release_lead)
        release_end = release_start + .08
        if time < release_end:
            origin = previous_xy
            # In a long idle gap, let the released finger follow its hand first.
            if math.isfinite(ready) and move_start - ready > .14:
                if time < move_start:
                    xy = origin + smooth((time - ready) / .14) * (resting[:2] - origin)
                    return np.array([*xy, hover])
                origin = np.asarray(resting[:2])
            progress = smooth((time - move_start) / (move_end - move_start))
            xy = origin + progress * (destination - origin)
            depth = (smooth((time - move_end) / (attack_end - move_end))
                     if time < release_start else 1 - smooth((time - release_start) / .08))
            return np.array([*xy, hover + depth * (pressed + (.012 if black else 0) - hover)])
        previous_xy, ready = destination, release_end
    xy = previous_xy + smooth((time - ready) / .14) * (resting[:2] - previous_xy) if notes else resting[:2]
    return np.array([*xy, hover])


class SongTargets:
    """Move idle fingers with the hand while holding active notes at their keys."""

    def __init__(self, score, key_y, hover=.955, pressed=.916, staged=False, release_lead=.09):
        if not math.isfinite(release_lead) or not 0 <= release_lead <= .2:
            raise ValueError("Release lead must be finite and within [0, 0.2] seconds")
        self.beat = 60 / score["tempo_bpm"]
        self.hover, self.pressed, self.key_y = hover, pressed, key_y
        self.parts = {(s, f): [n for n in score["notes"] if n["hand"] == s and n["finger"] == f]
                      for s in ("left", "right") for f in FINGERS}
        self.notes = score["notes"]
        self.centers = {}
        self.last_time = None
        self.staged = staged
        self.release_lead = release_lead

    def position_cost(self, side, finger, time):
        """Keep idle fingertips clear without locking their lateral spacing."""
        from physical_piano import smooth
        t = time - 1.
        engagement = 0.
        for note in self.parts[side, finger]:
            start = note["start_beat"] * self.beat
            end = start + note["duration_beats"] * self.beat
            approach = smooth((t - start + .35) / .21)
            release = 1. - smooth((t - end) / .14)
            engagement = max(engagement, approach * release)
        return [5. + 95. * engagement, 5. + 95. * engagement, 100.]

    def targets(self, time):
        import numpy as np
        from physical_piano import smooth
        t = time - 1.
        output = {}
        for side in ("left", "right"):
            near = [n for n in self.notes if n["hand"] == side
                    and (n["start_beat"] + n["duration_beats"]) * self.beat > t
                    and n["start_beat"] * self.beat < t + .45]
            if not near:
                past = [n for n in self.notes if n["hand"] == side and n["start_beat"] * self.beat <= t + .45]
                future = [n for n in self.notes if n["hand"] == side and n["start_beat"] * self.beat > t]
                near = past[-1:] or future[:1]
            offsets = [4, 3, 2, 1, 0] if side == "left" else [0, 1, 2, 3, 4]
            # White-key center spacing is 23.5 mm in this exported piano.
            bases = [self.key_y[n["midi"]] + .0235 * offsets[FINGERS.index(n["finger"])] for n in near]
            base = float(np.median(bases)) if bases else self.key_y[48 if side == "left" else 60]
            if self.last_time is not None and time >= self.last_time:
                maximum = .12 * (time - self.last_time)
                base = self.centers[side] + float(np.clip(base - self.centers[side], -maximum, maximum))
            self.centers[side] = base
            for i, finger in enumerate(FINGERS):
                resting = np.array([.409 if i == 0 else .436, base - .0235 * offsets[i], self.hover])
                part = self.parts[side, finger]
                if self.staged:
                    output[side, finger] = staged_finger_target(
                        t, part, self.key_y, resting, self.beat, self.hover, self.pressed, self.release_lead)
                    continue
                candidates = [n for n in part if n["start_beat"] * self.beat - .35 <= t
                              < (n["start_beat"] + n["duration_beats"]) * self.beat + .08]
                # An old release tail must not mask a new attack on the same finger.
                sounding = [n for n in candidates if n["start_beat"] * self.beat <= t
                            < (n["start_beat"] + n["duration_beats"]) * self.beat]
                active = (sounding or candidates)[-1] if candidates else None
                if active is not None:
                    start = active["start_beat"] * self.beat
                    end = start + active["duration_beats"] * self.beat
                    black = active["midi"] % 12 not in {0, 2, 4, 5, 7, 9, 11}
                    approach = smooth((t - start + .35) / .23)
                    if t > end + .08:
                        approach = 1 - smooth((t-end-.08)/.14)
                    resting[:2] += approach * (np.array([.467 if black else (.409 if i == 0 else .436),
                                                       self.key_y[active["midi"]]]) - resting[:2])
                    # Physical release trails the command by about 90 ms in this
                    # fixed R1/Shadow scene. Reserve the gap before a repeated key.
                    release_start = max(start, end - active.get("release_lead_seconds", self.release_lead))
                    depth = (smooth((t-start+.14)/.14) if t < release_start
                             else 1-smooth((t-release_start)/.08))
                    resting[2] = self.hover + depth * (self.pressed + (.012 if black else 0) - self.hover)
                output[side, finger] = resting
        self.last_time = time
        return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--midi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--time-stretch", type=float, default=1.)
    p.add_argument("--seconds", type=float)
    p.add_argument("--fold-registers", action="store_true")
    p.add_argument("--compact", action="store_true", help="Explicitly fold each hand into one octave")
    a = p.parse_args()
    score, report = prepare(a.midi, a.title, a.time_stretch, a.seconds, a.fold_registers, a.compact)
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out / "score.plan.json", score)
    write_json(a.out / "import-report.json", report)
    print(json.dumps(report, indent=2))
