"""Tests for the same-frame FK-residual data check (issue #105 follow-up).

Synthetic episodes on the real OMY model cover both shapes of the bug:

* broken   -- action is a copy of observation.state (the issue #105 bug);
              every motion frame must come out with a near-zero residual.
* healthy  -- action leads the state by a small per-frame target;
              motion-frame residuals must exceed the 1e-6 threshold.

Run:
    MUJOCO_GL=disable python -m unittest test_check_fk_residual -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

if os.environ.get("MUJOCO_GL", "") == "":
    os.environ["MUJOCO_GL"] = "disable"  # headless-safe default

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import check_fk_residual as cfr  # noqa: E402

ROBOT_XML = "robotis_omy/omy.xml"
N_ARM = 6
N_FRAMES = 30
# Per-frame arm travel (rad) at 20 Hz for a slow trajectory: ~0.007 rad,
# matching the observed median per-frame state travel in demo_data_example.
STEP = 0.007


def make_episode(lead=0.0):
    """Slow sinusoidal trajectory; action leads state by ``lead`` rad."""
    t = np.arange(N_FRAMES, dtype=np.float64)
    base = 0.1 + 0.002 * t + 0.05 * np.sin(0.1 * t)
    states = np.stack([base + 0.02 * i for i in range(N_ARM)], axis=1)
    actions = np.concatenate([states + lead, np.zeros((N_FRAMES, 1))], axis=1)
    return states, actions


class FkResidualCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = cfr.FkResidualChecker(robot_xml=ROBOT_XML, n_arm=N_ARM)

    def test_checker_binds_omy_model(self):
        self.assertEqual(len(self.checker.joint_names), N_ARM)
        self.assertTrue(self.checker.ee_body_name)

    def test_broken_episode_action_copy_of_state_has_all_near_zero_residuals(self):
        states, actions = make_episode(lead=0.0)
        residuals, motion_idx = self.checker.episode_residuals(states, actions)
        self.assertEqual(len(motion_idx), N_FRAMES - 1)
        self.assertEqual(len(residuals), N_FRAMES - 1)
        np.testing.assert_array_less(residuals, cfr.DEFAULT_RESIDUAL_THRESHOLD)
        ratio = float(np.mean(residuals < cfr.DEFAULT_RESIDUAL_THRESHOLD))
        self.assertGreater(ratio, cfr.BROKEN_RATIO_THRESHOLD)

    def test_healthy_episode_action_leads_state_has_no_near_zero_residuals(self):
        states, actions = make_episode(lead=STEP)
        residuals, _ = self.checker.episode_residuals(states, actions)
        self.assertGreater(float(np.min(residuals)), cfr.DEFAULT_RESIDUAL_THRESHOLD)
        ratio = float(np.mean(residuals < cfr.DEFAULT_RESIDUAL_THRESHOLD))
        self.assertEqual(ratio, 0.0)

    def test_stationary_frames_are_excluded(self):
        states, actions = make_episode(lead=STEP)
        states[5:] = states[4]  # freeze everything after frame 4
        residuals, motion_idx = self.checker.episode_residuals(states, actions)
        self.assertEqual(len(motion_idx), 4)
        self.assertEqual(len(residuals), 4)

    def test_action_dim_mismatch_is_rejected(self):
        states, actions = make_episode(lead=STEP)
        with self.assertRaises(ValueError):
            self.checker.episode_residuals(states[:, :4], actions)

    def test_report_flags_broken_dataset(self):
        broken_report = {
            "dataset": "synthetic", "robot": ROBOT_XML, "ee_body": "tcp_link",
            "fps": 20, "motion_frames": 100, "residual_threshold": 1e-6,
            "near_zero_motion_frames": 100, "near_zero_ratio": 1.0,
            "median_residual_m": 0.0, "action_is_copy_of_state": True,
            "verdict": "BROKEN (action appears to be a copy of observation.state)",
        }
        self.assertIn("BROKEN", cfr.format_report(broken_report))


if __name__ == "__main__":
    unittest.main()
