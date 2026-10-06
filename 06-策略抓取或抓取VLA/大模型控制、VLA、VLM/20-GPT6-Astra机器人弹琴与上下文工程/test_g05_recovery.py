import unittest

import numpy as np

from g05_recovery import bounded_command, hover_offset, piano_shift, summarize, sustained_success


class RecoveryTests(unittest.TestCase):
    def test_piano_translation_is_horizontal_and_bounded(self):
        np.testing.assert_allclose(piano_shift([0, .06, 0]), [0, .06, 0])
        for shift in ([0, .09, 0], [.07, .07, 0], [0, 0, .01], [float("nan"), 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                piano_shift(shift)

    def test_hover_height_is_explicit_and_bounded(self):
        np.testing.assert_allclose(hover_offset(.04), [0, 0, .04])
        for height in (0, .5, float("nan")):
            with self.assertRaises(ValueError):
                hover_offset(height)

    def test_limit_slew_and_joint_range(self):
        np.testing.assert_allclose(bounded_command([1, -1], [0, 0], [0, 0], [[-.5, .5]] * 2), [.006, -.006])
        np.testing.assert_allclose(bounded_command([1], [.348], [0], [[-1, 1]]), [.35])

    def test_invalid_inputs_fail_closed(self):
        for target in ([float("nan")], [float("inf")], [[0]], [0, 0]):
            with self.assertRaises(ValueError):
                bounded_command(target, [0], [0], [[-1, 1]])
        for speed in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                bounded_command([0], [0], [0], [[-1, 1]], speed=speed)

    def test_infeasible_bound_rejected(self):
        with self.assertRaises(ValueError):
            bounded_command([0], [1], [0], [[-1, 1]])

    def test_success_requires_sustained_position_and_rotation(self):
        good = {"position_m": .002, "rotation_rad": .01}
        self.assertFalse(sustained_success([good] * 19))
        self.assertTrue(sustained_success([good] * 20))
        self.assertFalse(sustained_success([good] * 19 + [{**good, "rotation_rad": .1}]))
        self.assertFalse(sustained_success([good] * 19 + [{**good, "position_m": float("nan")}]))

    def pair(self):
        common = {"case": "small", "initial_qpos_sha256": "same", "success": True,
                  "safety_stop": None, "simulated_seconds": 4.}
        return [{**common, "mode": "ik_only", "integrated_position_error": .05},
                {**common, "mode": "g05_assisted", "integrated_position_error": .03}]

    def test_error_reduction_is_measured_not_call_count(self):
        rows = self.pair()
        self.assertTrue(summarize(rows)[0]["assisted_has_lower_error"])
        rows[1]["integrated_position_error"] = .1
        self.assertFalse(summarize(rows)[0]["assisted_has_lower_error"])

    def test_early_stop_cannot_look_like_improvement(self):
        rows = self.pair()
        rows[1].update(safety_stop="contact", simulated_seconds=.1, integrated_position_error=.001)
        summary = summarize(rows)[0]
        self.assertFalse(summary["assisted_has_lower_error"])
        self.assertIsNone(summary["assisted_minus_baseline_integrated_position_error"])

    def test_mismatched_starts_rejected(self):
        rows = self.pair()
        rows[1]["initial_qpos_sha256"] = "different"
        with self.assertRaises(ValueError):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
