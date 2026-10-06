import copy
import unittest

from bimanual_fingering import reassign


def score(pitches, times=None, duration=.3):
    return dict(title="Original test study", tempo_bpm=60, notes=[
        dict(midi=p, hand="left" if p < 60 else "right", finger=("thumb", "index", "middle", "ring", "little")[i % 5],
             start_beat=times[i] if times else 0., duration_beats=duration)
        for i, p in enumerate(pitches)])


class BimanualTests(unittest.TestCase):
    def test_wide_accompaniment_can_use_both_hands(self):
        source = score([41, 45, 57])
        original = copy.deepcopy(source)
        result, report = reassign(source)
        self.assertEqual(source, original)
        self.assertEqual([n["midi"] for n in result["notes"]], [41, 45, 57])
        self.assertEqual({n["hand"] for n in result["notes"]}, {"left", "right"})
        self.assertTrue(report["fingering_geometry"]["within_supplied_limit"])
        self.assertEqual(report["hold_changes"], [])

    def test_articulation_change_is_explicit_and_counted(self):
        source = score([60, 64], times=[0., .02], duration=1.)
        result, report = reassign(source, max_hold=.16)
        self.assertEqual(len(report["hold_changes"]), 2)
        self.assertEqual(report["removed_notes"], 0)
        self.assertFalse(report["physical_execution_verified"])
        for before, after in zip(source["notes"], result["notes"]):
            self.assertEqual(before["midi"], after["midi"])
            self.assertEqual(before["start_beat"], after["start_beat"])
            self.assertAlmostEqual(after["duration_beats"], .16)

    def test_impossible_chord_fails_without_discarding(self):
        source = score([24, 48, 72])
        with self.assertRaisesRegex(ValueError, "No notes were removed"):
            reassign(source, max_span=.18)
        self.assertEqual(len(source["notes"]), 3)

    def test_parameters_and_fast_finger_reuse(self):
        source = score([60])
        for kwargs in (dict(max_span=float("nan")), dict(max_hold=0), dict(beam_width=0)):
            with self.assertRaises(ValueError):
                reassign(source, **kwargs)
        source = score([60, 60], times=[0., .11], duration=.1)
        result, _ = reassign(source)
        self.assertNotEqual([(n["hand"], n["finger"]) for n in result["notes"]][0],
                            [(n["hand"], n["finger"]) for n in result["notes"]][1])

    def test_uncapped_durations_are_exactly_preserved_at_validation_boundary(self):
        source = score([60], times=[100.1], duration=.05)
        source["tempo_bpm"] = 120
        for cap in (None, .16):
            result, report = reassign(source, max_hold=cap)
            self.assertEqual(result["notes"][0]["duration_beats"], .05)
            self.assertEqual(report["hold_changes"], [])

    def test_float_noise_is_not_counted_as_rearrangement(self):
        source = score([60], duration=.16000000000000014)
        result, report = reassign(source, max_hold=.16)
        self.assertEqual(report["hold_changes"], [])
        self.assertEqual(result["notes"][0]["duration_beats"], source["notes"][0]["duration_beats"])


if __name__ == "__main__":
    unittest.main()
