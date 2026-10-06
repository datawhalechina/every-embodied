"""Compare recorded note attacks with the quantized score, without filtering audio."""

import argparse
import json
from pathlib import Path
import wave


def score_events(expected, sustain, dt):
    import numpy as np

    expected = np.asarray(expected, dtype=bool)
    sustain = np.asarray(sustain)
    if expected.ndim != 2 or expected.shape[1] != 88 or sustain.shape != (len(expected),):
        raise ValueError("Expected an N x 88 score and N sustain values")
    if not np.isfinite(sustain).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid score timing or pedal")
    previous = np.zeros(88, dtype=bool)
    pedal = False
    events = []
    for step, active in enumerate(expected):
        down = sustain[step] >= 0.5
        if down != pedal:
            events.append({"type": "SustainOn" if down else "SustainOff", "time": step * dt, "note": None})
            pedal = down
        for key in np.flatnonzero(active != previous):
            events.append({"type": "NoteOn" if active[key] else "NoteOff", "time": step * dt, "note": int(key + 21)})
        previous = active
    return events


def compare_onsets(expected, measured, tolerance=0.10):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Expected a positive finite onset tolerance")
    reference = [e for e in expected if e["type"] == "NoteOn"]
    actual = [e for e in measured if e["type"] == "NoteOn"]
    matched_reference, matched_actual, differences = set(), set(), []
    for pitch in range(21, 109):
        ri = [i for i, e in enumerate(reference) if e["note"] == pitch]
        ai = [i for i, e in enumerate(actual) if e["note"] == pitch]
        if not ri or not ai:
            continue
        distance = np.abs(np.asarray([reference[i]["time"] for i in ri])[:, None]
                          - np.asarray([actual[i]["time"] for i in ai])[None, :])
        # A large penalty prioritizes maximum feasible matching, then timing error.
        cost = np.where(distance <= tolerance, distance, (len(ri) + len(ai) + 1) * tolerance)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if distance[row, col] <= tolerance:
                matched_reference.add(ri[row])
                matched_actual.add(ai[col])
                differences.append(actual[ai[col]]["time"] - reference[ri[row]]["time"])
    count = len(matched_reference)
    return {"scope": "pitch and onset only; no validation of fingers, note lengths or complete song",
        "tolerance_seconds": tolerance, "expected_onsets": len(reference), "measured_onsets": len(actual),
        "matched_onsets": count, "onset_precision": count / max(1, len(actual)),
        "onset_recall": count / max(1, len(reference)), "onset_f1": 2 * count / max(1, len(reference) + len(actual)),
        "missing": [e for i, e in enumerate(reference) if i not in matched_reference],
        "unexpected": [e for i, e in enumerate(actual) if i not in matched_actual],
        "matched_time_errors_seconds": differences}


def main(args):
    import numpy as np

    source_report = json.loads((args.source / "report.json").read_text())
    source = np.load(args.source / "trajectory.npz", allow_pickle=False)
    reference = score_events(source["expected"], source["sustain"], 1 / source_report["control_hz"])
    measured = json.loads((args.run / "events.json").read_text())
    raw = json.loads((args.run / "raw_robopianist_events.json").read_text())
    report = {"task": source_report["task"], "audio": compare_onsets(reference, measured),
              "raw_sensor": compare_onsets(reference, raw),
              "reference_is_robot_audio": False, "score_used_to_filter_actual_audio": False}
    (args.run / "audio_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.run / "reference-score-only.events.json").write_text(json.dumps(reference), encoding="utf-8")
    if args.reference_audio:
        from piano_events import midi_messages
        from robopianist.music.synthesizer import Synthesizer
        synth = Synthesizer(sample_rate=44100)
        try:
            samples = synth.get_samples(midi_messages(reference))
        finally:
            synth.stop()
        with wave.open(str(args.run / "reference-score-only.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(44100)
            stream.writeframes(samples.tobytes())
    print(json.dumps({"task": report["task"], **{k: v for k, v in report["audio"].items()
                                               if k not in ("missing", "unexpected", "matched_time_errors_seconds")}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-audio", action="store_true")
    main(parser.parse_args())
