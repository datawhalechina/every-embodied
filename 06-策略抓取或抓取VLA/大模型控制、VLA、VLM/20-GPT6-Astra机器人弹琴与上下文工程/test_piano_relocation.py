import unittest

from piano_relocation import score_with_pause


class RelocationTests(unittest.TestCase):
    def test_pause_preserves_notes_and_does_not_cut_sustain(self):
        score = dict(title="Original test", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=.5, hand="right", finger="thumb"),
            dict(midi=64, start_beat=1., duration_beats=.5, hand="right", finger="middle")])
        actual = score_with_pause(score, 1.75, 4.4)
        self.assertEqual(actual["notes"][0], score["notes"][0])
        self.assertAlmostEqual(actual["notes"][1]["start_beat"], 5.4)
        self.assertEqual(actual["notes"][1]["duration_beats"], .5)
        self.assertEqual(score["notes"][1]["start_beat"], 1.)
        with self.assertRaises(ValueError):
            score_with_pause(score, 1.25, 4.4)


if __name__ == "__main__":
    unittest.main()
