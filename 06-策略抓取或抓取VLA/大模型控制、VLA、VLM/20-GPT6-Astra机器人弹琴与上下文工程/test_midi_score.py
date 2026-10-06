import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mido

from midi_score import import_score, inspect_tracks, load_midi, read_notes
from piano_score import ContactRelease, compare_timing, rest_lift, validate_contact_score

ROOT = Path(__file__).parent


class MidiImportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "exercise.mid"
        self.midi = mido.MidiFile(type=1, ticks_per_beat=480)
        conductor = mido.MidiTrack()
        conductor.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))
        self.midi.tracks.append(conductor)
        self.right = mido.MidiTrack()
        self.right.extend([
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_on", note=60, velocity=0, time=480),
            mido.Message("note_on", note=67, velocity=64, time=240),
            mido.Message("note_off", note=67, velocity=0, time=480),
        ])
        self.midi.tracks.append(self.right)

    def save(self):
        self.midi.save(self.path)

    def test_track_inspection_and_velocity_zero_release(self):
        self.save()
        tracks = inspect_tracks(load_midi(self.path))
        self.assertEqual(tracks[0]["note_count"], 0)
        self.assertEqual(tracks[1]["midi_range"], [60, 67])
        plan, report = import_score(self.path, "Exercise", {1: "right"})
        self.assertEqual([n["start_beat"] for n in plan["notes"]], [0, 1.5])
        self.assertEqual([n["duration_beats"] for n in plan["notes"]], [1, 1])
        self.assertEqual(report["chords_removed"], 0)
        self.assertEqual(report["robot_execution"], "not_performed")

    def test_conductor_tempo_change_inside_note_and_rest(self):
        self.midi.tracks[0].append(mido.MetaMessage("set_tempo", tempo=2000000, time=240))
        self.save()
        plan, _ = import_score(self.path, "Exercise", {1: "right"})
        self.assertEqual(plan["notes"][0]["duration_beats"], 1.5)
        self.assertEqual(plan["notes"][1]["start_beat"], 2.5)
        self.assertEqual(plan["notes"][1]["duration_beats"], 2)

    def test_excerpt_and_explicit_time_stretch_preserve_rest(self):
        self.save()
        plan, report = import_score(self.path, "Excerpt", {1: "right"}, start_seconds=1.2, end_seconds=2.5, time_stretch=2)
        self.assertAlmostEqual(plan["notes"][0]["start_beat"], 0.6)
        self.assertEqual(plan["notes"][0]["duration_beats"], 2)
        self.assertEqual(report["source_selected_track_note_count"], 2)
        self.assertEqual(report["excerpt_note_count"], 1)
        for start, end in ((0.2, 2.5), (0, 2.0), (1.1, 1.4)):
            with self.assertRaises(ValueError):
                import_score(self.path, "Bad boundary", {1: "right"}, start, end)

    def test_separate_left_track_can_play_simultaneously(self):
        left = mido.MidiTrack([
            mido.Message("note_on", note=48, velocity=64, time=0),
            mido.Message("note_off", note=48, velocity=0, time=960),
        ])
        self.midi.tracks.append(left)
        self.save()
        plan, _ = import_score(self.path, "Two hands", {1: "right", 2: "left"})
        self.assertEqual(plan["notes"][0]["hand"], "left")
        self.assertEqual(plan["notes"][1]["start_beat"], 0)

    def test_explicit_pitch_split_changes_hands_not_pitch_or_time(self):
        self.save()
        plan, report = import_score(self.path, "Split melody", {1: "right"}, split_midi=62)
        self.assertEqual([n["hand"] for n in plan["notes"]], ["left", "right"])
        self.assertEqual([n["midi"] for n in plan["notes"]], [60, 67])
        self.assertEqual([n["start_beat"] for n in plan["notes"]], [0, 1.5])
        self.assertEqual(report["split_midi_after_transpose"], 62)
        with self.assertRaises(ValueError):
            import_score(self.path, "Bad split", {0: "left", 1: "right"}, split_midi=62)

    def test_polyphony_is_rejected_without_dropping_notes(self):
        self.right.insert(1, mido.Message("note_on", note=64, velocity=64, time=0))
        self.right.insert(3, mido.Message("note_off", note=64, velocity=0, time=0))
        self.save()
        with self.assertRaisesRegex(ValueError, "polyphonic"):
            import_score(self.path, "Chord", {1: "right"})

    def test_missing_release_or_unmatched_release_is_rejected(self):
        for track in (
            mido.MidiTrack([mido.Message("note_on", note=60, velocity=64)]),
            mido.MidiTrack([mido.Message("note_off", note=60)]),
            mido.MidiTrack([mido.Message("note_on", note=60, velocity=64), mido.Message("note_on", note=60, velocity=64, time=480)]),
        ):
            self.midi.tracks[1] = track
            with self.assertRaises(ValueError):
                read_notes(self.midi, {1: "right"})

    def test_pedal_drums_and_black_keys_fail_closed(self):
        self.right.insert(0, mido.Message("control_change", control=64, value=127, time=0))
        with self.assertRaisesRegex(ValueError, "Pedal"):
            read_notes(self.midi, {1: "right"})
        self.right.pop(0)
        self.right.insert(0, mido.Message("pitchwheel", pitch=100, time=0))
        with self.assertRaisesRegex(ValueError, "Pitch-bend"):
            read_notes(self.midi, {1: "right"})
        self.right.pop(0)
        self.right[0].channel = 9
        with self.assertRaisesRegex(ValueError, "percussion"):
            read_notes(self.midi, {1: "right"})
        self.right[0].channel = 0
        self.save()
        with self.assertRaises(ValueError):
            import_score(self.path, "Black keys", {1: "right"}, transpose={"right": 1})

    def test_asynchronous_file_bad_track_and_nonfinite_options_are_rejected(self):
        self.midi.type = 2
        self.save()
        with self.assertRaises(ValueError):
            load_midi(self.path)
        self.midi.type = 1
        self.save()
        for options in ({"track_hands": {5: "right"}}, {"time_stretch": float("nan")}, {"transpose": {"right": True}}):
            arguments = {"track_hands": {1: "right"}, **options}
            with self.assertRaises(ValueError):
                import_score(self.path, "Bad options", **arguments)

    def test_cli_outputs_have_digest_and_refuse_overwrite(self):
        self.save()
        out = Path(self.folder.name) / "prepared"
        args = [sys.executable, str(ROOT / "midi_score.py"), "prepare", "--midi", str(self.path), "--title", "Exercise", "--right-track", "1", "--out", str(out)]
        self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
        report = json.loads((out / "import-report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["plan_sha256"]), 64)
        self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)


class ContactScoreTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "verified_whole_gpt6.plan.json").read_text(encoding="utf-8"))

    def test_current_exercise_passes_but_too_fast_or_out_of_register_fails(self):
        validate_contact_score(self.plan)
        for field, value in (("midi", 100), ("duration_beats", 0.5)):
            changed = copy.deepcopy(self.plan)
            changed["notes"][0][field] = value
            with self.assertRaises(ValueError):
                validate_contact_score(changed)

    def test_timing_does_not_ignore_duplicate_or_early_events(self):
        notes = [n for n in self.plan["notes"] if n["hand"] == "right"]
        events = [{"midi": n["midi"], "time_seconds": 1.62 + n["start_beat"]} for n in notes]
        self.assertTrue(compare_timing(events, notes, 60))
        self.assertFalse(compare_timing(events + [events[0]], notes, 60))
        events[0]["time_seconds"] = 1.0
        self.assertFalse(compare_timing(events, notes, 60))

    def test_shared_pitch_between_hands_is_rejected_before_execution(self):
        self.plan["notes"][-1]["midi"] = 60
        self.plan["notes"][-1]["hand"] = "left"
        with self.assertRaises(ValueError):
            validate_contact_score(self.plan)

    def test_release_needs_actual_activation_and_resets_for_repeated_note(self):
        notes = [n for n in self.plan["notes"] if n["hand"] == "right"]
        release = ContactRelease({"right": notes}, 60)
        release.observe(1.6, {61})
        self.assertEqual(release.height("right", 1.7, 0.008), 0.008)
        release.observe(1.6, {60})
        self.assertAlmostEqual(release.height("right", 1.72, 0.008), 0.080)
        self.assertEqual(release.height("right", 2.6, 0.008), 0.008)
        release.observe(2.6, {60})
        self.assertEqual(len(release.fired["right"]), 2)
        self.assertEqual(release.tail_lift("right", 3), 0)
        release.observe(7.62, {67})
        self.assertAlmostEqual(release.tail_lift("right", 8.12), 0.22)

    def test_idle_hand_lifts_and_returns_before_note_window(self):
        notes = [{"midi": 60, "start_beat": 3, "duration_beats": 1, "hand": "left"}]
        self.assertEqual(rest_lift(1, notes, 60), 0.22)
        self.assertGreater(rest_lift(3.75, notes, 60), 0)
        self.assertEqual(rest_lift(4, notes, 60), 0)
        self.assertEqual(rest_lift(4.7, notes, 60), 0)
        self.assertAlmostEqual(rest_lift(5.3, notes, 60), 0.22)
        notes[0]["start_beat"] = 0
        self.assertEqual(rest_lift(0, notes, 60), 0)


if __name__ == "__main__":
    unittest.main()
