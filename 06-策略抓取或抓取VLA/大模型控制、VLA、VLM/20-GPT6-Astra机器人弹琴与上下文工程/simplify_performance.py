"""Make an explicitly simplified full-length study from a piano transcription.

Treble salience/continuity is a heuristic, not a verified melody transcription.
Every omitted note, octave shift, timing and articulation change is accounted for.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from piano_context import write_json
from polyphonic_score import validate_score
from song_score import assign_fingers
from transcription_review import validate_events


def groups(notes, window=.075):
    output = []
    for index, note in sorted(enumerate(notes), key=lambda pair: pair[1]["onset_time"]):
        if not output or note["onset_time"] - output[-1][0][1]["onset_time"] > window:
            output.append([])
        output[-1].append((index, note))
    return output


def spaced(events, gap):
    result = []
    for event in events:
        if result and event[1]["onset_time"] - result[-1][1]["onset_time"] < gap:
            if event[1]["velocity"] > result[-1][1]["velocity"]:
                result[-1] = event
        else:
            result.append(event)
    return result


def arrange(notes, stretch=1.2, melody_gap=.23, bass_gap=.85, bass_low=48,
            melody_indices=None, melody_octaves=0):
    validate_events(notes, [])
    if not math.isfinite(stretch) or not 1 <= stretch <= 2:
        raise ValueError("Study stretch must be within [1, 2]")
    if not .15 <= melody_gap <= 1 or not .5 <= bass_gap <= 3:
        raise ValueError("Invalid note spacing")
    if bass_low not in (36, 48):
        raise ValueError("Bass register must start at MIDI36 or MIDI48")
    if type(melody_octaves) is not int or melody_octaves not in (-1, 0, 1):
        raise ValueError("Only a uniform melody octave shift is supported")
    batches = groups(notes)
    candidates = [[(i, n) for i, n in batch if 60 <= n["midi_note"] <= 89 and n["velocity"] >= 40]
                  for batch in batches]
    candidates = [batch for batch in candidates if batch]
    if not candidates and melody_indices is None:
        raise ValueError("No salient treble candidates; manual melody selection required")
    # Retain the best continuous treble path, rather than always selecting the
    # highest pitch (high harmonics are common transcription false positives).
    paths = [(0., 76, [])]
    for batch in candidates:
        new = []
        for index, note in batch:
            pitch = note["midi_note"]
            emission = -note["velocity"] / 20 + .035 * abs(pitch-76) + .2 * max(0, pitch-86)
            options = []
            for cost, previous, history in paths:
                distance = abs(pitch-previous)
                options.append((cost + emission + .06*distance + .25*max(0, distance-12),
                                pitch, history+[(index, note)]))
            new.append(min(options, key=lambda x: x[0]))
        paths = new
    protected = melody_indices is not None
    if protected:
        if (not isinstance(melody_indices, list) or not melody_indices
                or any(type(i) is not int or not 0 <= i < len(notes) for i in melody_indices)
                or len(set(melody_indices)) != len(melody_indices)):
            raise ValueError("Melody selection must contain unique valid source indices")
        melody = sorted(((i, notes[i]) for i in melody_indices), key=lambda pair: pair[1]["onset_time"])
    else:
        melody = spaced(min(paths, key=lambda x: x[0])[2], melody_gap)
    bass = []
    melody_ids = {i for i, _ in melody}
    for batch in batches:
        low = [(i, n) for i, n in batch if i not in melody_ids
               and 28 <= n["midi_note"] < 60 and n["velocity"] >= 30]
        if low:
            bass.append(min(low, key=lambda pair: (pair[1]["midi_note"], -pair[1]["velocity"])))
    # Sparse bass anchors, without replacing earlier anchors using later notes.
    bass = [event for j, event in enumerate(bass)
            if j == 0 or event[1]["onset_time"] > bass[j-1][1]["onset_time"]]
    anchors = []
    for event in bass:
        if not anchors or event[1]["onset_time"]-anchors[-1][1]["onset_time"] >= bass_gap:
            anchors.append(event)
    selected, changes = [], []
    for hand, events in (("right", melody), ("left", anchors)):
        for position, (index, note) in enumerate(events):
            original = note["midi_note"]
            pitch = original - 12 if hand == "right" else original
            low, high = (60, 76) if hand == "right" else (bass_low, bass_low+11)
            if protected and hand == "right":
                pitch = original + 12 * melody_octaves
                if not 21 <= pitch <= 108:
                    raise ValueError("Protected melody exceeds keyboard; do not fold individual notes")
            else:
                while pitch < low:
                    pitch += 12
                while pitch > high:
                    pitch -= 12
            start = note["onset_time"] * stretch
            available = ((events[position+1][1]["onset_time"] * stretch - start - .075)
                         if position+1 < len(events) else 1.)
            hold = min(.22 if hand == "right" else .25, available)
            if protected and hand == "right":
                # Acoustic offsets are not physical releases: this is an explicit
                # monophonic release proposal, not recovered human articulation.
                duration = (note["offset_time"] - note["onset_time"]) * stretch
                hold = min(duration, available) if position+1 < len(events) else duration
            if hold < .05:
                raise ValueError(f"Insufficient physical release gap or hold for source note {index}: "
                                 f"{hold:.6f}s; review articulation or slow globally, do not delete melody")
            selected.append(dict(midi=pitch, start=start, end=start+hold, hand=hand))
            changes.append(dict(source_index=index, hand=hand, original_midi=original, midi=pitch,
                                original_onset=note["onset_time"], onset=start,
                                original_duration=note["offset_time"]-note["onset_time"], hold_seconds=hold))
    selected.sort(key=lambda n: (n["start"], n["midi"]))
    assigned = assign_fingers(selected)
    score = dict(title=("Protected melody selection with sparse bass" if protected else
                       "Full-length simplified piano study: heuristic melody and sparse bass"),
                 tempo_bpm=60, notes=[dict(midi=n["midi"], start_beat=n["start"],
                    duration_beats=n["end"]-n["start"], hand=n["hand"], finger=n["finger"])
                    for n in assigned])
    validate_score(score, song_mode=True)
    kept = {n["source_index"] for n in changes}
    report = dict(source_notes=len(notes), selected_notes=len(selected), melody_notes=len(melody),
        bass_notes=len(anchors), omitted_source_indices=[i for i in range(len(notes)) if i not in kept],
        source_end_seconds=max(n["offset_time"] for n in notes),
        source_last_selected_onset_seconds=max(n["original_onset"] for n in changes),
        time_stretch=stretch, melody_min_gap_seconds=None if protected else melody_gap,
        bass_min_gap_seconds=bass_gap,
        bass_register_midi=[bass_low, bass_low+11],
        melody_register_midi=[min(n['midi'] for n in changes if n['hand'] == 'right'),
                              max(n['midi'] for n in changes if n['hand'] == 'right')],
        octave_shifted_notes=sum(n["original_midi"] != n["midi"] for n in changes),
        changes=changes, source_melody_verified=False, source_pedal_reproduced=False,
        melody_selection_protected=protected,
        melody_uniform_octaves=melody_octaves if protected else None,
        source_dynamics_reproduced=False, physical_execution_verified=False,
        interpretation="Explicit simplified arrangement, not faithful transcription or unchanged-score benchmark")
    return score, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-stretch", type=float, default=1.2)
    parser.add_argument("--bass-low", type=int, choices=(36, 48), default=48)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--heuristic-study", action="store_true",
                      help="Legacy lossy experiment; not a verified song arrangement")
    mode.add_argument("--melody-selection", type=Path,
                      help="JSON with source_sha256 and melody_source_indices; preserve all selected melody notes")
    parser.add_argument("--melody-octaves", type=int, choices=(-1, 0, 1), default=0)
    args = parser.parse_args()
    source = args.notes.read_bytes()
    selection = None
    if args.melody_selection:
        selection = json.loads(args.melody_selection.read_text(encoding="utf-8"))
        if selection["source_sha256"] != hashlib.sha256(source).hexdigest():
            raise ValueError("Melody selection refers to a different source transcription")
    score, report = arrange(json.loads(source), args.time_stretch, bass_low=args.bass_low,
        melody_indices=selection["melody_source_indices"] if selection is not None else None,
        melody_octaves=args.melody_octaves)
    if selection is not None:
        report["melody_selection_sha256"] = hashlib.sha256(args.melody_selection.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "score.plan.json", score)
    report["source_sha256"] = hashlib.sha256(source).hexdigest()
    report["plan_sha256"] = hashlib.sha256((args.out / "score.plan.json").read_bytes()).hexdigest()
    write_json(args.out / "arrangement-report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("changes", "omitted_source_indices")}, indent=2))
