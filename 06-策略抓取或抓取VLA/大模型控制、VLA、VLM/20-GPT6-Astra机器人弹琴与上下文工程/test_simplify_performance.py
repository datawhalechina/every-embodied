import copy
import unittest
from simplify_performance import arrange


class SimplifiedStudyTests(unittest.TestCase):
    def test_changes_and_omissions_account_for_every_source_note(self):
        notes = [dict(midi_note=p, onset_time=t, offset_time=t+.3, velocity=v)
                 for t, p, v in ((0.,48,70),(0.,60,80),(0.,72,100),(0.,61,20),
                                 (.5,74,90),(.5,50,60),(.5,86,55),(1.,76,90),(1.,52,70))]
        original = copy.deepcopy(notes)
        score, report = arrange(notes)
        kept = {n["source_index"] for n in report["changes"]}
        removed = set(report["omitted_source_indices"])
        self.assertFalse(kept & removed)
        self.assertEqual(kept | removed, set(range(len(notes))))
        self.assertEqual(notes, original)
        self.assertEqual(len(score["notes"]), len(kept))
        for n in report["changes"]:
            self.assertEqual(n["midi"] % 12, n["original_midi"] % 12)
            self.assertAlmostEqual(n["onset"], 1.2*n["original_onset"])
        self.assertFalse(report["source_melody_verified"])

    def test_missing_melody_and_invalid_speed_are_rejected(self):
        notes = [dict(midi_note=48,onset_time=0.,offset_time=1.,velocity=80)]
        with self.assertRaises(ValueError):
            arrange(notes)
        with self.assertRaises(ValueError):
            arrange(notes, stretch=float("nan"))

    def test_protected_melody_keeps_quiet_fast_notes_and_intervals(self):
        notes = [dict(midi_note=p, onset_time=t, offset_time=t+d, velocity=v)
                 for t, p, d, v in ((0.,72,.18,20),(.2,74,.4,100),(1.,77,.8,80))]
        score, report = arrange(notes, stretch=1., melody_indices=[0,1,2], melody_octaves=-1)
        self.assertEqual([n['midi'] for n in score['notes']], [60,62,65])
        self.assertEqual([n['start_beat'] for n in score['notes']], [0.,.2,1.])
        self.assertAlmostEqual(score['notes'][-1]['duration_beats'], .8)
        self.assertEqual(report['omitted_source_indices'], [])
        self.assertTrue(report['melody_selection_protected'])
        self.assertFalse(report['source_melody_verified'])

    def test_protected_melody_fails_instead_of_deleting_unplayable_note(self):
        notes = [dict(midi_note=72+i, onset_time=t, offset_time=t+.4, velocity=80)
                 for i,t in enumerate((0.,.02))]
        with self.assertRaisesRegex(ValueError, 'release gap'):
            arrange(notes, melody_indices=[0,1])
        with self.assertRaisesRegex(ValueError, 'unique valid'):
            arrange(notes, melody_indices=[0,0])
