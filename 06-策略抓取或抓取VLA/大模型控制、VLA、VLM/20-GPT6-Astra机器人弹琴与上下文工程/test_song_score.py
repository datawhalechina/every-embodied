import copy
from pathlib import Path
import tempfile
import unittest

import mido

from song_score import assign_fingers, merge_same_keys, performance_notes, prepare, SongTargets
from polyphonic_score import validate_score


class SongImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "original-study.mid"

    def save(self, events):
        midi = mido.MidiFile(type=0, ticks_per_beat=100)
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=1000000)] + events))
        midi.save(self.path)

    def test_overlapping_voices_are_paired_then_explicitly_merged(self):
        self.save([mido.Message("note_on", note=60, velocity=80),
                   mido.Message("note_on", note=60, velocity=80, time=50),
                   mido.Message("note_off", note=60, time=50),
                   mido.Message("note_off", note=60, time=50)])
        notes = performance_notes(self.path)
        self.assertEqual(len(notes), 2)
        merged, count = merge_same_keys(notes)
        self.assertEqual(count, 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0, 1.5))

    def test_octave_adaptation_is_explicit_and_counted(self):
        self.save([mido.Message("note_on", note=36, velocity=80),
                   mido.Message("note_off", note=36, time=100)])
        score, report = prepare(self.path, "Study", fold_registers=True)
        self.assertEqual(score["notes"][0]["midi"], 48)
        self.assertEqual(report["octave_shifted_notes"], 1)
        self.assertFalse(report["song_execution_verified"])
        unchanged, _ = prepare(self.path, "Study")
        self.assertEqual(unchanged["notes"][0]["midi"], 36)

    def test_missing_note_off_and_pedal_fail(self):
        for events in ([mido.Message("note_on", note=60, velocity=80)],
                       [mido.Message("control_change", control=64, value=127)]):
            self.save(events)
            with self.assertRaises(ValueError):
                performance_notes(self.path)

    def test_compact_register_is_explicit(self):
        self.save([mido.Message("note_on", note=79, velocity=80),
                   mido.Message("note_off", note=79, time=100)])
        score, report = prepare(self.path, "Study", compact=True)
        self.assertEqual(score["notes"][0]["midi"], 67)
        self.assertEqual(report["register_adaptation"], "one octave per hand")
        self.assertEqual(report["octave_shifted_notes"], 1)

    def test_pitch_bend_is_not_silently_discarded(self):
        self.save([mido.Message("pitchwheel", pitch=100)])
        with self.assertRaises(ValueError):
            performance_notes(self.path)

    def test_surface_control_uses_rotated_lowest_mesh_patch(self):
        import numpy as np
        from polyphonic_piano import contact_patch_point
        vertices = np.array([[0., 0., -1.], [0., 0., 1.]])
        original = vertices.copy()
        np.testing.assert_array_equal(contact_patch_point(vertices, np.eye(3)), vertices[0])
        np.testing.assert_array_equal(contact_patch_point(vertices, np.diag([1., -1., -1.])), vertices[1])
        np.testing.assert_array_equal(vertices, original)
        with self.assertRaises(ValueError):
            contact_patch_point(np.empty((0, 3)), np.eye(3))

    def test_old_release_does_not_mask_next_attack(self):
        score = dict(title="Repeated original study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=t, duration_beats=.5, hand="right", finger="thumb")
            for t in (0., .6)])
        target = SongTargets(score, {i: -.01*(i-60) for i in range(21, 109)})
        self.assertAlmostEqual(target.targets(1.61)["right", "thumb"][2], .916)

    def test_idle_priority_preserves_vertical_clearance(self):
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=1., duration_beats=1., hand="right", finger="thumb")])
        target = SongTargets(score, {i: -.01*(i-60) for i in range(21, 109)})
        self.assertEqual(target.position_cost("right", "index", 2.5), [5., 5., 100.])
        self.assertEqual(target.position_cost("right", "thumb", 2.5), [100., 100., 100.])
        self.assertEqual(target.position_cost("right", "thumb", 3.5), [5., 5., 100.])

    def test_ik_step_bounds_intersect_joint_limits(self):
        import numpy as np
        from polyphonic_piano import bounded_ik_limits
        low, high = bounded_ik_limits(np.array([-.02, -.5]), np.array([.5, .01]), .08)
        np.testing.assert_allclose(low, [-.02, -.08])
        np.testing.assert_allclose(high, [.08, .01])
        for bad in (0., -1., .3, float("nan")):
            with self.assertRaises(ValueError):
                bounded_ik_limits(-1., 1., bad)
        with self.assertRaises(ValueError):
            bounded_ik_limits(.1, .2, .08)

    def test_key_depth_feedback_is_bounded_and_unwinds(self):
        from polyphonic_piano import key_press_offset
        value = 0.
        for _ in range(100):
            value = key_press_offset(value, 0., .05)
        self.assertEqual(value, .004)
        self.assertLess(key_press_offset(value, 1.1, .05), value)
        self.assertEqual(key_press_offset(value, .2, 0.), 0.)
        self.assertEqual(key_press_offset(0., 1., .05), 0.)
        for gain in (-1., .2, float("nan")):
            with self.assertRaises(ValueError):
                key_press_offset(0., 0., gain)

    def test_staged_targets_lift_before_moving_and_do_not_skip_next_note(self):
        import numpy as np
        from song_score import staged_finger_target
        notes = [dict(midi=p, start_beat=t, duration_beats=.16)
                 for p, t in [(60, 0.), (62, .25), (64, .5)]]
        original = copy.deepcopy(notes)
        y = {60: 0., 62: -.025, 64: -.05}
        resting = np.array([.436, .015, .95])
        def sample(t):
            return staged_finger_target(t, notes, y, resting, 1., .95, .912)
        self.assertAlmostEqual(sample(.10)[1], y[60])
        self.assertLess(sample(.10)[2], .95)
        self.assertAlmostEqual(sample(.16)[2], .95)
        self.assertAlmostEqual(sample(.25)[1], y[62])
        self.assertAlmostEqual(sample(.25)[2], .912)
        self.assertAlmostEqual(sample(.5)[1], y[64])
        self.assertEqual(notes, original)

    def test_staged_idle_return_is_continuous(self):
        import numpy as np
        from song_score import staged_finger_target
        notes = [dict(midi=60, start_beat=0., duration_beats=.5)]
        y = {60: 0.}
        resting = np.array([.436, .1, .95])
        def sample(t):
            return staged_finger_target(t, notes, y, resting, 1., .95, .912)
        np.testing.assert_allclose(sample(.49-1e-6), sample(.49+1e-6), atol=1e-7)
        np.testing.assert_allclose(sample(.7), resting)

    def test_release_lead_is_configurable_without_changing_score(self):
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=.16, hand="right", finger="thumb")])
        original = copy.deepcopy(score)
        y = {i: -.01*(i-60) for i in range(21, 109)}
        default = SongTargets(score, y, staged=True)
        later = SongTargets(score, y, staged=True, release_lead=.03)
        self.assertGreater(default.targets(1.12)["right", "thumb"][2], later.targets(1.12)["right", "thumb"][2])
        self.assertEqual(score, original)
        for bad in (-.1, .3, float("nan")):
            with self.assertRaises(ValueError):
                SongTargets(score, y, release_lead=bad)

    def test_per_note_release_keeps_short_attack_without_changing_other_notes(self):
        from song_score import validate_song
        score = dict(title="Short attack", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=.15, hand="right", finger="thumb",
                 release_lead_seconds=.025)])
        original = copy.deepcopy(score)
        y = {i: -.01*(i-60) for i in range(21,109)}
        for staged in (False, True):
            target = SongTargets(score, y, staged=staged, pressed=.912)
            self.assertAlmostEqual(target.targets(1.10)["right", "thumb"][2], .912)
        self.assertEqual(score, original)
        for bad in (-.1, .3, float("nan")):
            invalid = copy.deepcopy(score)
            invalid["notes"][0]["release_lead_seconds"] = bad
            with self.assertRaises(ValueError):
                validate_song(invalid)

    def test_span_check_detects_rubato_overlap_not_just_equal_onsets(self):
        from song_score import fingering_geometry
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=.5, hand="right", finger="thumb"),
            dict(midi=81, start_beat=.01, duration_beats=.2, hand="right", finger="little")])
        original = copy.deepcopy(score)
        report = fingering_geometry(score, {60: 0., 81: -.282}, .18)
        self.assertFalse(report["within_supplied_limit"])
        self.assertAlmostEqual(report["hands"]["right"]["worst"]["span_m"], .282)
        self.assertEqual(report["hands"]["right"]["worst"]["midi"], [60, 81])
        self.assertEqual(score, original)
        self.assertIsNone(fingering_geometry(score, {60: 0., 81: -.282})["within_supplied_limit"])
        self.assertFalse(report["robot_reachability_verified"])

    def test_span_check_excludes_released_notes(self):
        from song_score import fingering_geometry
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=.5, hand="right", finger="thumb"),
            dict(midi=81, start_beat=.5, duration_beats=.2, hand="right", finger="little")])
        report = fingering_geometry(score, {60: 0., 81: -.282}, .18)
        self.assertTrue(report["within_supplied_limit"])
        for limit in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                fingering_geometry(score, {60: 0., 81: -.282}, limit)

    def test_release_command_leads_the_next_repeated_note(self):
        score = dict(title="Repeated original study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=t, duration_beats=.5, hand="right", finger="thumb")
            for t in (0., .575)])
        target = SongTargets(score, {i: -.01*(i-60) for i in range(21, 109)})
        self.assertGreater(target.targets(1.49)["right", "thumb"][2], .95)
        self.assertAlmostEqual(target.targets(1.58)["right", "thumb"][2], .916)

    def test_onset_metric_does_not_borrow_a_later_same_pitch(self):
        from polyphonic_score import evaluate
        score = dict(title="Repeated original study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=t, duration_beats=.5, hand="right", finger="thumb")
            for t in (0., 2.)])
        result = evaluate(score, [dict(type="NoteOn", midi=60, time_seconds=3.),
                                  dict(type="NoteOff", midi=60, time_seconds=3.5)], song_mode=True)
        self.assertEqual(result["onset_metrics"]["true_positive"], 1)
        self.assertEqual(result["onset_metrics"]["recall"], .5)

    def test_beam_fingering_keeps_chord_and_uses_distinct_fingers(self):
        notes = [dict(midi=p, start=0., end=1., hand="right") for p in (60, 64, 67)]
        assigned = assign_fingers(notes)
        self.assertEqual([n["midi"] for n in assigned], [60, 64, 67])
        self.assertEqual(len({n["finger"] for n in assigned}), 3)

    def test_six_simultaneous_fingers_fail_instead_of_dropping_notes(self):
        notes = [dict(midi=p, start=0., end=1., hand="right") for p in (60, 62, 64, 65, 67, 69)]
        with self.assertRaises(ValueError):
            assign_fingers(notes)

    def test_long_score_requires_explicit_song_mode(self):
        score = dict(title="Long original study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=float(i), duration_beats=.5, hand="right", finger="thumb")
            for i in range(300)])
        with self.assertRaises(ValueError):
            validate_score(score)
        self.assertEqual(validate_score(score, song_mode=True)["note_count"], 300)
        bad = copy.deepcopy(score)
        bad["notes"][1]["start_beat"] = .1
        with self.assertRaises(ValueError):
            validate_score(bad, song_mode=True)

    def test_song_targets_hold_then_lift_without_changing_pitch(self):
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=1., hand="right", finger="thumb")])
        y = {i: -.01*(i-60) for i in range(21, 109)}
        target = SongTargets(score, y)
        pressed = target.targets(1.5)["right", "thumb"]
        self.assertAlmostEqual(pressed[1], y[60])
        self.assertAlmostEqual(pressed[2], .916)
        self.assertGreater(target.targets(2.1)["right", "thumb"][2], .95)

    def test_pianomime_adapter_uses_midi_pitch_not_key_index(self):
        from pianomime_runner import score_frames
        score = dict(title="Study", tempo_bpm=60, notes=[
            dict(midi=60, start_beat=0., duration_beats=1., hand="right", finger="thumb")])
        frames = score_frames(score, lambda pitch, velocity, finger: (pitch, velocity, finger))
        self.assertEqual(frames[0], [])
        self.assertEqual(frames[20], [(60, 80, 0)])
        self.assertEqual(frames[40], [])


if __name__ == "__main__":
    unittest.main()
