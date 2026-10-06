import unittest
import numpy as np

from g05_piano_session import handoff_controls, shifted_score, verify_preparation, session_exit_code


class G05PianoSessionTests(unittest.TestCase):
    def setUp(self):
        self.report = dict(mode="g05_assisted", success=True, safety_stop=None,
            model_calls=[0], model_induced_command_distance_rad_seconds=.01)
        self.protocol = dict(source_scene_sha256="scene", source_trajectory_sha256="trace", control_hz=100, physics_hz=500)

    def test_preparation_requires_real_contribution(self):
        verify_preparation(self.report, self.protocol, "scene", "trace")
        for change in (dict(mode="ik_only"), dict(success=False), dict(safety_stop="contact"),
                       dict(model_calls=[]), dict(model_induced_command_distance_rad_seconds=0.),
                       dict(model_induced_command_distance_rad_seconds=float("nan"))):
            with self.assertRaises(ValueError):
                verify_preparation(dict(self.report, **change), self.protocol, "scene", "trace")

    def test_wrong_scene_or_clock_rejected(self):
        for shift in ([0, .06, 0], [0, float("nan"), 0], [0, 0]):
            with self.assertRaises(ValueError):
                verify_preparation(self.report, dict(self.protocol, piano_shift_world_m=shift), "scene", "trace")
            with self.assertRaises(ValueError):
                verify_preparation(dict(self.report, piano_shift_world_m=shift), self.protocol, "scene", "trace")
        with self.assertRaises(ValueError):
            verify_preparation(self.report, self.protocol, "different", "trace")
        with self.assertRaises(ValueError):
            verify_preparation(self.report, dict(self.protocol, control_hz=50), "scene", "trace")

    def test_handoff_is_continuous_and_bounded(self):
        controls = handoff_controls(np.zeros(2), np.array([.2,-.2]), 2., [0,1])
        self.assertEqual(controls.shape, (200,2))
        np.testing.assert_allclose(controls[-1], [.2,-.2])
        self.assertLess(np.max(np.abs(np.diff(np.vstack([np.zeros(2),controls]),axis=0))), .006)

    def test_large_or_invalid_handoff_fails(self):
        for target, seconds in (([4.], 1.), ([float("nan")], 2.), ([.2], 0.)):
            with self.assertRaises(ValueError):
                handoff_controls(np.zeros(1), np.array(target), seconds, [0])

    def test_score_shift_preserves_source_pitch_timing_and_fingers(self):
        score = dict(title="Synthetic test", tempo_bpm=120, notes=[dict(midi=60,
            start_beat=1., duration_beats=.5, hand="right", finger="thumb")])
        result = shifted_score(score, 6.)
        self.assertEqual(score["notes"][0]["start_beat"], 1.)
        self.assertEqual(result["notes"][0]["start_beat"], 13.)
        self.assertEqual(result["notes"][0]["duration_beats"], .5)
        self.assertEqual(result["notes"][0]["midi"], 60)

    def test_completion_mode_keeps_musical_failure_and_rejects_physical_failure(self):
        report = dict(performance_passed=False, playback_completed=True, safety_stop=None,
                      interhand_contact_steps=0, onset_metrics=dict(measured_count=3))
        self.assertEqual(session_exit_code(report), 2)
        self.assertEqual(session_exit_code(report, True), 0)
        self.assertFalse(report["performance_passed"])
        for change in (dict(playback_completed=False), dict(safety_stop="contact"),
                       dict(interhand_contact_steps=1), dict(onset_metrics=dict(measured_count=0))):
            self.assertEqual(session_exit_code(dict(report, **change), True), 2)


if __name__ == "__main__":
    unittest.main()
