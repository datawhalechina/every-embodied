"""Keep acoustic note offsets distinct from physical key releases."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_events(notes, pedals):
    if not isinstance(notes, list) or not notes or not isinstance(pedals, list):
        raise ValueError("Expected a nonempty note list and a pedal list")
    for event in [*notes, *pedals]:
        for key in ("onset_time", "offset_time"):
            value = event[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Expected finite event times")
        if event["onset_time"] < 0 or event["offset_time"] <= event["onset_time"]:
            raise ValueError("Expected positive event durations at nonnegative times")
    for note in notes:
        if type(note["midi_note"]) is not int or not 21 <= note["midi_note"] <= 108:
            raise ValueError("Expected an absolute piano MIDI pitch")
        if type(note["velocity"]) is not int or not 1 <= note["velocity"] <= 127:
            raise ValueError("Expected MIDI velocity in 1..127")
    end = -1.
    for pedal in sorted(pedals, key=lambda p: p["onset_time"]):
        if pedal["onset_time"] < end:
            raise ValueError("Overlapping sustain pedal intervals")
        end = pedal["offset_time"]


def reference_events(notes, pedals, offset_semantics):
    """A reference renderer, not a source of measured robot sound."""
    validate_events(notes, pedals)
    if offset_semantics not in ("acoustic", "key-release"):
        raise ValueError("Explicit acoustic or key-release offset semantics required")
    events = []
    for note in notes:
        events.extend([
            dict(type="NoteOn", midi=note["midi_note"], velocity=note["velocity"], time_seconds=note["onset_time"]),
            dict(type="NoteOff", midi=note["midi_note"], time_seconds=note["offset_time"]),
        ])
    if offset_semantics == "key-release":
        for pedal in pedals:
            events.extend([
                dict(type="SustainOn", time_seconds=pedal["onset_time"]),
                dict(type="SustainOff", time_seconds=pedal["offset_time"]),
            ])
    # Release the old pedal/key before the new press at a shared timestamp.
    return sorted(events, key=lambda e: (e["time_seconds"], e["type"] in ("NoteOn", "SustainOn")))


def polyphony(notes):
    points = sorted([(n["onset_time"], 1) for n in notes] + [(n["offset_time"], -1) for n in notes])
    active = peak = 0
    over_ten = previous = 0.
    for time, change in points:
        if active > 10:
            over_ten += time - previous
        active += change
        peak = max(peak, active)
        previous = time
    return dict(peak=peak, seconds_above_ten=over_ten)


def release_hypothesis(notes, pedals, minimum_hold=.16, tolerance=.12):
    """Explore early release under sustain; this is NOT recovered human fingering.

    Shorten only if the sound is explainable by a pedal release or a repeated
    strike of the same key. Keep other notes unchanged, even when infeasible.
    """
    validate_events(notes, pedals)
    if not math.isfinite(minimum_hold) or minimum_hold <= 0:
        raise ValueError("Minimum hold must be positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    ordered = sorted(enumerate(notes), key=lambda pair: (pair[1]["onset_time"], pair[0]))
    next_onset, following = {}, {}
    for index, note in reversed(ordered):
        next_onset[index] = following.get(note["midi_note"], math.inf)
        following[note["midi_note"]] = note["onset_time"]
    output, changes = [], []
    for index, note in enumerate(notes):
        candidate = dict(note)
        for pedal in pedals:
            release = max(note["onset_time"] + minimum_hold, pedal["onset_time"] + .001)
            sound_end = min(pedal["offset_time"], next_onset[index])
            if (release < note["offset_time"] and release < pedal["offset_time"]
                    and abs(sound_end - note["offset_time"]) <= tolerance):
                candidate["offset_time"] = release
                changes.append(dict(note_index=index, midi=note["midi_note"],
                    old_offset=note["offset_time"], proposed_key_release=release,
                    reconstructed_sound_end=sound_end,
                    sound_end_error_seconds=sound_end-note["offset_time"]))
                break
        output.append(candidate)
    return output, changes


def review(notes, pedals):
    validate_events(notes, pedals)
    extensions = []
    for note in notes:
        for pedal in pedals:
            if pedal["onset_time"] <= note["offset_time"] < pedal["offset_time"]:
                extensions.append(pedal["offset_time"] - note["offset_time"])
                break
    return dict(note_count=len(notes), predicted_pedal_intervals=len(pedals),
        acoustic_polyphony=polyphony(notes),
        note_intervals_under_50ms=sum(n["offset_time"]-n["onset_time"] < .05 for n in notes),
        potential_double_sustain=dict(note_count=len(extensions),
            over_100ms=sum(e > .1 for e in extensions),
            max_extension_seconds=max(extensions, default=0.),
            interpretation="Timing risk, not measured audible error; repeated strikes and decay also matter"),
        transcription_verified=False, physical_key_releases_verified=False,
        robot_execution_verified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notes", type=Path, required=True)
    p.add_argument("--pedals", type=Path, required=True)
    p.add_argument("--offset-semantics", choices=("acoustic", "key-release"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--release-hypothesis", action="store_true")
    args = p.parse_args()
    notes = json.loads(args.notes.read_text(encoding="utf-8"))
    pedals = json.loads(args.pedals.read_text(encoding="utf-8"))
    events = reference_events(notes, pedals, args.offset_semantics)
    report = review(notes, pedals)
    report.update(offset_semantics=args.offset_semantics,
        note_source_sha256=hashlib.sha256(args.notes.read_bytes()).hexdigest(),
        pedal_source_sha256=hashlib.sha256(args.pedals.read_bytes()).hexdigest(),
        applied_sustain_events=sum(e["type"].startswith("Sustain") for e in events))
    outputs = {"reference-events.json": events}
    if args.release_hypothesis:
        if args.offset_semantics != "acoustic":
            p.error("Release reconstruction applies only to acoustic offsets")
        keys, changes = release_hypothesis(notes, pedals)
        outputs.update({"key-release-hypothesis.json": keys, "release-changes.json": changes})
        report["release_hypothesis"] = dict(minimum_hold_seconds=.16, tolerance_seconds=.12,
            changed_notes=len(changes), note_count=len(keys), polyphony=polyphony(keys),
            robot_ready=False, exact_source_key_releases=False,
            interpretation="Planning hypothesis only; preserves pitch/onset/velocity, may alter sound ends")
    args.out.mkdir(parents=True, exist_ok=False)
    outputs["review.json"] = report
    for name, value in outputs.items():
        (args.out/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
