import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from pianomime_galaxea import position_velocity_command, reduced_arm_problem, unique_suffix, validate_plan
from piano_events import PianoEventSensor


class PianoMimeTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"task": "Adieu_0", "steps": 120, "g05_prior_weight": 0.03,
                     "g05_phase": "preparation_only", "finger_controller": "pianomime_low_level",
                     "rationale": "Low-priority preparation prior only."}

    def test_valid_plan(self):
        self.assertEqual(validate_plan(self.plan), 0.03)
        actual = json.loads(Path(__file__).with_name("astra-pianomime-plan.json").read_text())
        self.assertEqual(validate_plan(actual), actual["g05_prior_weight"])

    def test_unsafe_or_nonfinite_plan_is_rejected(self):
        for key, value in (("g05_phase", "during_chords"), ("finger_controller", "g05"),
                           ("steps", 120.0), ("steps", True), ("task", "../other"),
                           ("g05_prior_weight", float("nan")), ("g05_prior_weight", True),
                           ("g05_prior_weight", 1.0), ("g05_prior_weight", 0)):
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_plan(plan)
        for plan in (None, {}, {**self.plan, "extra": 1}):
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_joint_names_are_mapped_not_guessed_by_offset(self):
        names = ["robot/right/rh_A_WRJ1", "robot/left/lh_A_WRJ1"]
        self.assertEqual(unique_suffix(names, "lh_A_WRJ1"), 1)
        for names, suffix in ((names, "missing"), (names + ["other/lh_A_WRJ1"], "lh_A_WRJ1")):
            with self.assertRaises(ValueError):
                unique_suffix(names, suffix)

    def test_custom_plan_must_match_verified_source_identity(self):
        plan = dict(self.plan, task="CustomSong", steps=630)
        self.assertEqual(validate_plan(plan, ("CustomSong", 630)), .03)
        for identity in (("OtherSong", 630), ("CustomSong", 631), ("../bad", 630), ("CustomSong", True)):
            with self.assertRaises(ValueError):
                validate_plan(plan, identity)

    def test_arm_qp_excludes_fingers_and_preserves_order(self):
        problem = SimpleNamespace(P=np.diag([4., 999., 2.]), q=np.array([1., 1e9, 3.]))
        P, q, low, high = reduced_arm_problem(problem, [2, 0], [3, 1],
            np.array([0., 0.2, 0., -0.1]), [[-1, 1], [-2, 2]])
        np.testing.assert_array_equal(P, np.diag([2., 4.]))
        np.testing.assert_array_equal(q, [3., 1.])
        np.testing.assert_allclose(low, 0.95 * np.array([-0.9, -2.2]))
        np.testing.assert_allclose(high, 0.95 * np.array([1.1, 1.8]))

    def test_invalid_arm_mapping_or_limits_is_rejected(self):
        problem = SimpleNamespace(P=np.eye(2), q=np.zeros(2))
        for velocities, positions, limits in (([0, 0], [0, 1], [[-1, 1]] * 2),
                ([0], [0, 1], [[-1, 1]]), ([0], [0], [[1, -1]]),
                ([0], [0], [[-1, float("nan")]])):
            with self.assertRaises(ValueError):
                reduced_arm_problem(problem, velocities, positions, np.zeros(2), limits)

    def test_velocity_feedforward_cancels_actuator_damping(self):
        position, velocity, kp, kv = np.array([0.2]), np.array([0.5]), np.array([1000.]), np.array([60.])
        command, clipped = position_velocity_command(position, velocity, kp, kv, [[-1, 1]])
        np.testing.assert_allclose(kp * (command - position) - kv * velocity, [0], atol=1e-12)
        self.assertEqual(clipped, 0)

    def test_feedforward_is_bounded_and_rejects_nan(self):
        command, clipped = position_velocity_command([0.99], [1], [1000], [60], [[-1, 1]])
        np.testing.assert_allclose(command, [1])
        self.assertEqual(clipped, 1)
        with self.assertRaises(ValueError):
            position_velocity_command([0], [float("nan")], [1000], [60], [[-1, 1]])


class PianoEventTests(unittest.TestCase):
    def test_bounce_does_not_retrigger_without_releasing_key(self):
        sensor = PianoEventSensor()
        for i, depth in enumerate([0, .92, .86, .92, .6, .49, .93]):
            keys = [0.] * 88
            keys[39] = depth
            sensor.update((i + 1) * .002, keys, 0)
        self.assertEqual([(e["type"], e["note"]) for e in sensor.events],
                         [("NoteOn", 60), ("NoteOff", 60), ("NoteOn", 60)])

    def test_extra_pitches_are_never_filtered(self):
        sensor = PianoEventSensor()
        keys = [0.] * 88
        keys[0] = keys[87] = .95
        sensor.update(.002, keys, 0)
        self.assertEqual([e["note"] for e in sensor.events], [21, 108])

    def test_pedal_is_preserved_and_does_not_fabricate_key_presses(self):
        sensor = PianoEventSensor()
        sensor.update(.002, [0.] * 88, 1)
        sensor.update(.004, [0.] * 88, 1)
        sensor.update(.006, [0.] * 88, 0)
        self.assertEqual([e["type"] for e in sensor.events], ["SustainOn", "SustainOff"])

    def test_invalid_or_repeated_time_is_rejected(self):
        sensor = PianoEventSensor()
        sensor.update(.01, [0.] * 88, 0)
        for time in (.01, -.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sensor.update(time, [0.] * 88, 0)


if __name__ == "__main__":
    unittest.main()
