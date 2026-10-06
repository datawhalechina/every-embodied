"""Plan a bounded two-hand fingering study without deleting or transposing notes.

Optional shortened holds are an explicit articulation experiment, not recovered
human fingering or a pedal-faithful transcription. No simulator is started here.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from piano_context import write_json
from polyphonic_score import FINGERS, validate_score
from song_score import fingering_geometry, key_coordinate


def reassign(score, max_span=.18, max_hold=None, beam_width=128, register_weight=4.):
    validate_score(score, song_mode=True)
    if not math.isfinite(max_span) or not .04 <= max_span <= .3:
        raise ValueError("Supply a hand-span screening limit in [0.04, 0.3] meters")
    if max_hold is not None and (not math.isfinite(max_hold) or not .05 <= max_hold <= 2):
        raise ValueError("Explicit hold cap must be in [0.05, 2] seconds")
    if type(beam_width) is not int or not 1 <= beam_width <= 512:
        raise ValueError("Beam width must be in 1..512")
    if not math.isfinite(register_weight) or register_weight < 0:
        raise ValueError("Register preference must be finite and nonnegative")
    beat = 60 / score["tempo_bpm"]
    notes = []
    shortened = []
    for index, original in enumerate(score["notes"]):
        start = original["start_beat"] * beat
        hold = original["duration_beats"] * beat
        duration = max_hold if max_hold is not None and hold > max_hold + 1e-9 else hold
        notes.append(dict(midi=original["midi"], start=start, end=start+duration,
                          duration=duration, changed=duration < hold, index=index))
        if duration < hold:
            shortened.append(dict(note_index=index, original_hold_seconds=hold, hold_seconds=duration))
    offsets = [4, 3, 2, 1, 0, 0, 1, 2, 3, 4]
    # A state retains all finger release times, including recently released notes.
    beam = [(0., [None]*10, [key_coordinate(48), key_coordinate(60)], [])]
    for note in notes:
        candidates = []
        for cost, previous, bases, history in beam:
            held = [(i, n) for i, n in enumerate(previous) if n is not None and n["end"] > note["start"] + 1e-9]
            for slot in range(10):
                last = previous[slot]
                if last is not None:
                    travel = abs(key_coordinate(note["midi"]) - key_coordinate(last["midi"]))
                    if last["end"] + .055 + min(.165, .035*travel) > note["start"] + 1e-9:
                        continue
                placed = [*held, (slot, note)]
                hands = [[(i, n) for i, n in placed if i//5 == hand] for hand in (0, 1)]
                if hands[0] and hands[1] and max(n["midi"] for _, n in hands[0]) >= min(n["midi"] for _, n in hands[1]):
                    continue
                hand = slot//5
                ordered = sorted(hands[hand], key=lambda pair: offsets[pair[0]])
                if any(a[1]["midi"] >= b[1]["midi"] for a, b in zip(ordered, ordered[1:])):
                    continue
                coords = [key_coordinate(n["midi"]) for _, n in ordered]
                if .0235*(max(coords)-min(coords)) > max_span + 1e-9:
                    continue
                required_bases = [key_coordinate(n["midi"])-offsets[i] for i, n in ordered]
                center = sum(required_bases)/len(required_bases)
                spread = sum((b-center)**2 for b in required_bases)
                travel_cost = .2*(center-bases[hand])**2
                # Register is a soft preference; unlike the old fixed MIDI60 split,
                # it may yield to a feasible assignment of a wide accompaniment.
                register_cost = register_weight*max(0, note["midi"]-59 if hand == 0 else 60-note["midi"])
                state, centers = list(previous), list(bases)
                state[slot], centers[hand] = note, center
                candidates.append((cost + 4*spread + travel_cost + register_cost,
                                   state, centers, history+[slot]))
        if not candidates:
            raise ValueError(f"No bounded two-hand fingering at {note['start']:.3f}s; "
                             "review releases, hand span or score. No notes were removed.")
        unique = {}
        for candidate in sorted(candidates, key=lambda c: c[0]):
            identity = tuple(n["index"] if n is not None else None for n in candidate[1])
            unique.setdefault(identity, candidate)
            if len(unique) >= beam_width:
                break
        beam = list(unique.values())
    winner = min(beam, key=lambda c: c[0])
    result = copy.deepcopy(score)
    if shortened:
        result["title"] = "Articulation study (not pedal-faithful): " + score["title"]
    for n, slot in zip(notes, winner[3]):
        output = result["notes"][n["index"]]
        output["hand"] = "left" if slot < 5 else "right"
        output["finger"] = FINGERS[slot % 5]
        if n["changed"]:
            output["duration_beats"] = n["duration"]/beat
    validate_score(result, song_mode=True)
    changed_hands = [i for i, (a, b) in enumerate(zip(score["notes"], result["notes"])) if a["hand"] != b["hand"]]
    geometry = fingering_geometry(result, {p: -.0235*key_coordinate(p) for p in range(21, 109)}, max_span)
    report = dict(note_count=len(notes), pitch_changes=0, onset_changes=0, removed_notes=0,
                  max_hold_seconds=max_hold, hold_changes=shortened, hand_changes=changed_hands,
                  max_hand_span_m=max_span, beam_width=beam_width, fingering_geometry=geometry,
                  register_preference_weight=register_weight,
                  keyboard_spacing_assumption_m=.0235, physical_execution_verified=False,
                  interpretation="Articulation study, not pedal-faithful music" if shortened else "Fingering search; physical reachability unverified")
    return result, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-hand-span", type=float, default=.18,
                        help="Lateral screening limit, not a measured robot workspace")
    parser.add_argument("--max-hold-seconds", type=float,
                        help="Explicitly shorten key holds; changes articulation, not pitch/onset")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--register-weight", type=float, default=4.)
    args = parser.parse_args()
    source = args.plan.read_bytes()
    score, report = reassign(json.loads(source), args.max_hand_span, args.max_hold_seconds, args.beam_width, args.register_weight)
    report["source_plan_sha256"] = hashlib.sha256(source).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "score.plan.json", score)
    report["output_plan_sha256"] = hashlib.sha256((args.out / "score.plan.json").read_bytes()).hexdigest()
    write_json(args.out / "fingering-report.json", report)
    print(json.dumps(report, indent=2))
