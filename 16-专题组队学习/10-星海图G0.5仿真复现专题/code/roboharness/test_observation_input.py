"""Check measured history selection, state provenance, and camera identity."""
import unittest

import numpy as np

from bridge import Calibration
from observation_input import build_input,validate


def snapshot(step, episode='episode-a'):
    q = [0.0]*65
    q[3], q[26], q[27], q[30], q[57] = .1, .02, .03, .2, .4
    return {'episode_id': episode, 'sequence': step + 1, 'control_step': step,
            'proprio': q, 'images': {camera: np.full((4, 6, 3), i + step, np.uint8)
            for i, camera in enumerate(('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'))}}


class ObservationTests(unittest.TestCase):
    def test_reset_uses_measured_state_and_marks_padding(self):
        data, binding = build_input([snapshot(0)], 'put decorations away')
        self.assertEqual(binding['history_is_pad'], [True]*5+[False])
        self.assertEqual(tuple(data['images']['head_rgb'].shape), (6, 3, 4, 6))
        self.assertAlmostEqual(data['state']['left_gripper'][-1, 0].item(), .02)
        self.assertAlmostEqual(data['state']['left_gripper'][-1, 1].item(), .03)
        self.assertAlmostEqual(data['state']['trunk_qpos'][-1, 0].item(), .4)
        self.assertTrue(data['action_is_pad'].all())

    def test_exact_history_keeps_camera_roles_and_latest_binding(self):
        data, binding = build_input([snapshot(i) for i in range(0, 151, 30)], 'put decorations away')
        self.assertEqual(binding['history_steps'], [0, 30, 60, 90, 120, 150])
        self.assertEqual(binding['sequence'], 151)
        self.assertFalse(any(binding['history_is_pad']))
        self.assertEqual(data['images']['left_wrist_rgb'][-1, 0, 0, 0].item(), 151)
        self.assertEqual(data['images']['right_wrist_rgb'][-1, 0, 0, 0].item(), 152)

    def test_future_frame_is_not_selected_for_earlier_target(self):
        data, binding = build_input([snapshot(0), snapshot(29), snapshot(31), snapshot(60)], 'task', history_count=3)
        self.assertEqual(binding['history_steps'], [0, 29, 60])

    def test_missing_interior_history_rejected(self):
        with self.assertRaises(ValueError):
            build_input([snapshot(0), snapshot(150)], 'task')

    def test_no_padding_from_unobserved_reset(self):
        with self.assertRaises(ValueError):
            build_input([snapshot(30)], 'task')

    def test_episode_boundary_rejected(self):
        with self.assertRaises(ValueError):
            build_input([snapshot(0), snapshot(30, 'another')], 'task')

    def test_duplicate_read_does_not_create_physical_history(self):
        with self.assertRaises(ValueError):
            build_input([snapshot(0), snapshot(0)], 'task')

    def test_j8_must_match_seven_joint_interface(self):
        row = snapshot(0)
        row['proprio'][10] = .2
        with self.assertRaises(ValueError):
            build_input([row], 'task')

    def test_raw_capture_keeps_measured_j8_while_policy_input_rejects_it(self):
        row = snapshot(0)
        invalid_roll = 1.5 * Calibration().tool_roll_tolerance_rad
        row['proprio'][10], row['proprio'][37] = invalid_roll, -invalid_roll
        state,images=validate(row,policy_state=False)
        self.assertIsNone(state);self.assertEqual(row['proprio'][10], invalid_roll)
        self.assertEqual(set(images),set(row['images']))
        with self.assertRaises(ValueError):build_input([row],'task')

    def test_missing_camera_rejected(self):
        row = snapshot(0)
        del row['images']['right_wrist_rgb']
        with self.assertRaises(ValueError):
            build_input([row], 'task')

    def test_float_images_rejected(self):
        row = snapshot(0)
        row['images']['head_rgb'] = row['images']['head_rgb'].astype(float)
        with self.assertRaises(ValueError):
            build_input([row], 'task')

    def test_input_arrays_are_owned(self):
        row = snapshot(0)
        data, binding = build_input([row], 'task')
        row['images']['head_rgb'][:] = 255
        self.assertEqual(data['images']['head_rgb'][-1, 0, 0, 0].item(), 0)

    def test_invalid_frequency_rejected(self):
        for frequency in (True, 0, float('nan')):
            with self.assertRaises(ValueError):
                build_input([snapshot(0)], 'task', control_hz=frequency)


if __name__ == '__main__':
    unittest.main()
