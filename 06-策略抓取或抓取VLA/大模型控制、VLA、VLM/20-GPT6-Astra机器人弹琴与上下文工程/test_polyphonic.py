import copy
import unittest

from polyphonic_score import KeyCycles, evaluate, finger_target, performance_passed, validate_score


class KeyCycleTests(unittest.TestCase):
    def test_partial_rebound_is_not_a_new_press(self):
        sensor = KeyCycles(1)
        changes = [sensor.update([value]) for value in [0, 0.91, 0.85, 0.72, 0.55, 0.95, 0.49, 0.92]]
        self.assertEqual(changes, [[], [(0, True)], [], [], [], [], [(0, False)], [(0, True)]])

    def test_unexpected_pitch_is_not_filtered(self):
        sensor = KeyCycles(2)
        self.assertEqual(sensor.update([0.93, 0.95]), [(0, True), (1, True)])
        self.assertEqual(sensor.update([0.1, 0.7]), [(0, False)])

    def test_nonfinite_or_wrong_shape_readout_is_rejected(self):
        sensor = KeyCycles(2)
        for value in ([0], [float("nan"), 0], [0, float("inf")]):
            with self.assertRaises(ValueError):
                sensor.update(value)


class PolyphonicScoreTests(unittest.TestCase):
    def test_full_performance_requires_all_checks(self):
        good = dict.fromkeys(("note_count_match", "timing_match", "chords_match", "finger_matches"), True)
        good["interhand_contact_steps"] = 0
        self.assertTrue(performance_passed(good))
        for key in good:
            bad = dict(good)
            bad[key] = 1 if key == "interhand_contact_steps" else False
            self.assertFalse(performance_passed(bad))
        self.assertFalse(performance_passed({}))

    def test_malformed_note_container_is_rejected(self):
        for notes in (None, {}, [None], [3]):
            with self.assertRaises(ValueError):
                validate_score({"title": "Bad", "tempo_bpm": 80, "notes": notes})

    def setUp(self):
        self.score = {"title": "Chord", "tempo_bpm": 60, "notes": [
            {"midi": 48, "start_beat": 0, "duration_beats": 1, "hand": "left", "finger": "little"},
            {"midi": 52, "start_beat": 0, "duration_beats": 1, "hand": "left", "finger": "middle"},
            {"midi": 60, "start_beat": 0, "duration_beats": 1, "hand": "right", "finger": "thumb"}]}
        self.events = [{"type": kind, "midi": pitch, "time_seconds": time}
                       for kind, time in (("NoteOn", 1.02), ("NoteOff", 2.03)) for pitch in (48, 52, 60)]

    def test_independent_fingers_hold_a_chord(self):
        report = evaluate(self.score, self.events)
        self.assertTrue(report["note_count_match"] and report["timing_match"] and report["chords_match"])
        self.assertAlmostEqual(report["chords"][0]["measured_overlap_seconds"], 1.01)

    def test_same_finger_cannot_play_overlapping_notes(self):
        self.score["notes"][1]["finger"] = "little"
        with self.assertRaisesRegex(ValueError, "One finger"):
            validate_score(self.score)

    def test_missing_chord_tone_or_extra_note_fails(self):
        missing = [e for e in self.events if e["midi"] != 52]
        self.assertFalse(evaluate(self.score, missing)["note_count_match"])
        extra = self.events + [{"type": kind, "midi": 65, "time_seconds": time} for kind, time in (("NoteOn", 2.5), ("NoteOff", 3))]
        self.assertFalse(evaluate(self.score, extra)["note_count_match"])

    def test_staccato_hits_cannot_pass_as_sustained_chord(self):
        short = copy.deepcopy(self.events)
        for event in short:
            if event["type"] == "NoteOff":
                event["time_seconds"] = 1.1
        report = evaluate(self.score, short)
        self.assertTrue(report["note_count_match"])
        self.assertFalse(report["timing_match"])
        self.assertFalse(report["chords_match"])

    def test_broken_chord_is_not_simultaneous(self):
        delayed = copy.deepcopy(self.events)
        for event in delayed:
            if event["midi"] == 52:
                event["time_seconds"] += 1.1
        self.assertFalse(evaluate(self.score, delayed)["chords_match"])

    def test_target_holds_until_score_release(self):
        note = self.score["notes"][0]
        y = {48: 0.2}
        self.assertAlmostEqual(finger_target(1.5, [note], y, 48, 1)[1], 0.908)
        self.assertAlmostEqual(finger_target(1.99, [note], y, 48, 1)[1], 0.908)
        self.assertAlmostEqual(finger_target(2.1, [note], y, 48, 1)[1], 0.950)


if __name__ == "__main__":
    unittest.main()
