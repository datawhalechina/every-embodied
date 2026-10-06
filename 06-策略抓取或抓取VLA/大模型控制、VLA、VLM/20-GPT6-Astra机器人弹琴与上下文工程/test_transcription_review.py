import copy
import unittest

from transcription_review import polyphony, reference_events, release_hypothesis, review


def note(pitch=60, start=0., end=2.):
    return dict(midi_note=pitch, onset_time=start, offset_time=end, velocity=80)


class TranscriptionReviewTests(unittest.TestCase):
    def test_acoustic_offsets_never_apply_sustain_twice(self):
        notes = [note()]
        pedals = [dict(onset_time=.1, offset_time=2.1)]
        events = reference_events(notes, pedals, "acoustic")
        self.assertEqual([e["type"] for e in events], ["NoteOn", "NoteOff"])
        self.assertEqual(events[-1]["time_seconds"], 2.)
        self.assertEqual(review(notes, pedals)["potential_double_sustain"]["note_count"], 1)

    def test_key_release_semantics_keep_pedal(self):
        events = reference_events([note()], [dict(onset_time=0., offset_time=3.)], "key-release")
        self.assertEqual(sum(e["type"].startswith("Sustain") for e in events), 2)

    def test_repeated_notes_release_before_new_attack(self):
        events = reference_events([note(), note(start=2., end=3.)], [], "acoustic")
        self.assertEqual([e["type"] for e in events if e["time_seconds"] == 2.], ["NoteOff", "NoteOn"])

    def test_invalid_values_and_unknown_semantics(self):
        for bad in [note(start=float("nan")), note(end=-1.), note(pitch=0), dict(note(), velocity=128)]:
            with self.assertRaises(ValueError):
                reference_events([bad], [], "acoustic")
        with self.assertRaises(ValueError):
            reference_events([note()], [], "unknown")

    def test_overlapping_pedals_fail(self):
        with self.assertRaises(ValueError):
            reference_events([note()], [dict(onset_time=0., offset_time=2.),
                                       dict(onset_time=1., offset_time=3.)], "acoustic")

    def test_polyphony_uses_half_open_intervals(self):
        self.assertEqual(polyphony([note(), note(62, 2., 3.)])["peak"], 1)
        notes = [note(pitch=48+i) for i in range(11)]
        self.assertEqual(polyphony(notes), dict(peak=11, seconds_above_ten=2.))

    def test_reconstruction_is_explicit_and_preserves_inputs(self):
        notes = [note()]
        snapshot = copy.deepcopy(notes)
        result, changes = release_hypothesis(notes, [dict(onset_time=.2, offset_time=2.)])
        self.assertEqual(notes, snapshot)
        self.assertEqual(result[0]["offset_time"], .201)
        self.assertEqual(changes[0]["sound_end_error_seconds"], 0.)
        for field in ("midi_note", "onset_time", "velocity"):
            self.assertEqual(result[0][field], notes[0][field])

    def test_no_pedal_or_inconsistent_pedal_does_not_shorten(self):
        for pedals in ([], [dict(onset_time=.1, offset_time=3.)]):
            result, changes = release_hypothesis([note()], pedals)
            self.assertEqual(result, [note()])
            self.assertEqual(changes, [])

    def test_repeated_strike_can_end_acoustic_sustain(self):
        notes = [note(), note(start=2.01, end=3.)]
        result, changes = release_hypothesis(notes, [dict(onset_time=.1, offset_time=3.)])
        self.assertEqual(len(changes), 2)
        self.assertAlmostEqual(result[0]["offset_time"], .16)


if __name__ == "__main__":
    unittest.main()
