"""Inspect or import an authorized MIDI excerpt without playing audio or moving a robot."""

import argparse
import hashlib
import importlib.metadata
from pathlib import Path

from piano_context import number, write_json
from piano_score import validate_contact_score


def load_midi(path):
    import mido

    path = Path(path)
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Use a MIDI file no larger than 5 MiB")
    midi = mido.MidiFile(path)
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ValueError("Use synchronous type 0/1 MIDI with PPQN timing, not type 2 or SMPTE")
    return midi


def inspect_tracks(midi):
    result = []
    for index, track in enumerate(midi.tracks):
        notes = [msg.note for msg in track if msg.type == "note_on" and msg.velocity > 0]
        result.append({
            "index": index, "name": track.name, "note_count": len(notes),
            "midi_range": [min(notes), max(notes)] if notes else None,
            "channels": sorted({msg.channel for msg in track if not msg.is_meta and hasattr(msg, "channel")}),
            "sustain_pedal": any(msg.type == "control_change" and msg.control == 64 and msg.value >= 64 for msg in track),
        })
    return result


def read_notes(midi, track_hands):
    import mido

    if not track_hands or any(type(i) is not int or not 0 <= i < len(midi.tracks) for i in track_hands):
        raise ValueError("Select existing track indices shown by inspect")
    if any(side not in ("left", "right") for side in track_hands.values()) or len(set(track_hands.values())) != len(track_hands):
        raise ValueError("Use a separate track for each hand")
    # Preserve track identity while integrating the global tempo map, including conductor tracks.
    timeline = []
    for index, track in enumerate(midi.tracks):
        tick = 0
        for order, msg in enumerate(track):
            tick += msg.time
            if msg.type == "set_tempo" or index in track_hands:
                timeline.append((tick, index, order, msg))
    timeline.sort(key=lambda item: item[:3])
    previous_tick, seconds, tempo = 0, 0.0, 500000
    active, notes = {}, []
    for tick, track, _, msg in timeline:
        seconds += mido.tick2second(tick - previous_tick, midi.ticks_per_beat, tempo)
        previous_tick = tick
        if msg.type == "set_tempo":
            if msg.tempo <= 0:
                raise ValueError("Invalid MIDI tempo")
            tempo = msg.tempo
            continue
        if track not in track_hands:
            continue
        if msg.type == "control_change" and msg.control in (64, 66, 67) and msg.value > 0:
            raise ValueError("Pedal-dependent scores are not supported; provide a pedal-free arrangement")
        if msg.type == "pitchwheel" and msg.pitch != 0:
            raise ValueError("Pitch-bend expression cannot be reproduced by discrete piano keys")
        if msg.type not in ("note_on", "note_off"):
            continue
        if msg.channel == 9:
            raise ValueError("A selected track contains percussion, not piano notes")
        key = (track, msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            if key in active:
                raise ValueError("A repeated MIDI pitch has no intervening note-off")
            active[key] = seconds
        else:
            if key not in active:
                raise ValueError("A selected MIDI note-off has no matching note-on")
            start = active.pop(key)
            if seconds <= start:
                raise ValueError("MIDI notes must have positive duration")
            notes.append({"midi": msg.note, "start_seconds": start, "end_seconds": seconds, "hand": track_hands[track]})
    if active:
        raise ValueError("A selected MIDI note is missing its note-off")
    if not notes:
        raise ValueError("Selected tracks contain no notes")
    return sorted(notes, key=lambda note: (note["start_seconds"], note["hand"], note["midi"]))


def import_score(path, title, track_hands, start_seconds=0.0, end_seconds=None,
                 time_stretch=1.0, transpose=None, split_midi=None):
    number(start_seconds, "start_seconds", 0, 3600)
    number(time_stretch, "time_stretch", 1, 8)
    if end_seconds is not None:
        number(end_seconds, "end_seconds", 0, 3600)
        if end_seconds <= start_seconds:
            raise ValueError("end_seconds must be later than start_seconds")
    transpose = transpose or {}
    if any(side not in ("left", "right") or type(value) is not int or not -24 <= value <= 24 for side, value in transpose.items()):
        raise ValueError("Transpose values must be integer semitones in [-24, 24]")
    if split_midi is not None:
        if type(split_midi) is not int or not 21 <= split_midi <= 108 or list(track_hands.values()) != ["right"]:
            raise ValueError("Pitch splitting requires one right-hand source track and a MIDI boundary in [21, 108]")
    midi = load_midi(path)
    original = read_notes(midi, track_hands)
    end_seconds = end_seconds if end_seconds is not None else max(note["end_seconds"] for note in original)
    chosen = []
    for note in original:
        if note["end_seconds"] <= start_seconds or note["start_seconds"] >= end_seconds:
            continue
        if note["start_seconds"] < start_seconds - 1e-9 or note["end_seconds"] > end_seconds + 1e-9:
            raise ValueError("An excerpt boundary cuts a held note; move the boundary to a release")
        chosen.append(note)
    if not chosen:
        raise ValueError("The selected excerpt contains no notes")
    # 60 BPM is a seconds-based transport; source tempo changes remain in the event times.
    plan = {"title": title, "tempo_bpm": 60, "notes": [{
        "midi": note["midi"] + transpose.get(note["hand"], 0),
        "start_beat": round((note["start_seconds"] - start_seconds) * time_stretch, 9),
        "duration_beats": round((note["end_seconds"] - note["start_seconds"]) * time_stretch, 9),
        "hand": ("left" if note["midi"] + transpose.get(note["hand"], 0) < split_midi else "right") if split_midi is not None else note["hand"],
    } for note in chosen]}
    validation = validate_contact_score(plan)
    report = {
        **validation, "source_file": Path(path).name,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "parser": "mido", "parser_version": importlib.metadata.version("mido"),
        "selected_tracks": {str(index): side for index, side in track_hands.items()},
        "split_midi_after_transpose": split_midi,
        "source_selected_track_note_count": len(original), "excerpt_note_count": len(chosen),
        "excerpt_seconds": [start_seconds, end_seconds], "time_stretch": time_stretch,
        "transpose_semitones": {side: transpose.get(side, 0) for side in ("left", "right")},
        "velocity": "not_reproduced_by_current_controller", "pedals": "unsupported",
        "chords_removed": 0, "notes_changed_by_GPT": 0,
        "authorization": "user_supplied_file; importer_does_not_verify_rights",
    }
    return plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--midi", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--midi", type=Path, required=True)
    prepare.add_argument("--title", required=True)
    prepare.add_argument("--right-track", type=int, required=True)
    prepare.add_argument("--left-track", type=int)
    prepare.add_argument("--start-seconds", type=float, default=0)
    prepare.add_argument("--end-seconds", type=float)
    prepare.add_argument("--time-stretch", type=float, default=1)
    prepare.add_argument("--transpose-right", type=int, default=0)
    prepare.add_argument("--transpose-left", type=int, default=0)
    prepare.add_argument("--split-midi", type=int, help="For one melody track, assign transposed pitches below this boundary to the left hand")
    prepare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            import json

            print(json.dumps(inspect_tracks(load_midi(args.midi)), ensure_ascii=False, indent=2))
            return
        if args.left_track == args.right_track:
            raise ValueError("Left and right hands cannot share one track")
        tracks = {args.right_track: "right"}
        if args.left_track is not None:
            tracks[args.left_track] = "left"
        plan, report = import_score(
            args.midi, args.title, tracks, args.start_seconds, args.end_seconds,
            args.time_stretch, {"left": args.transpose_left, "right": args.transpose_right}, args.split_midi,
        )
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "score.plan.json", plan)
        report["plan_sha256"] = hashlib.sha256((args.out / "score.plan.json").read_bytes()).hexdigest()
        write_json(args.out / "import-report.json", report)
        print(f"Prepared {args.out.resolve()}; no model call, audio playback or robot motion")
    except (ValueError, OSError, ImportError, EOFError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
