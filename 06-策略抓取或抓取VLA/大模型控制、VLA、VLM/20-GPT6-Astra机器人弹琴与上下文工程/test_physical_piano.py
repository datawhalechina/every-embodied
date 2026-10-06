import importlib.util
import json
import unittest
from pathlib import Path

from physical_piano import position_targets, smooth, target_at
from piano_context import validate_plan
from replay_windows import key_activation


class InterpolationTests(unittest.TestCase):
    def test_smooth_clamps_and_endpoints(self):
        self.assertEqual(smooth(-1), 0)
        self.assertEqual(smooth(2), 1)
        self.assertEqual(smooth(0.5), 0.5)

    def test_recorded_gpt_plan_matches_task_contract(self):
        root = Path(__file__).parent
        plan = json.loads((root / "verified_gpt6.plan.json").read_text(encoding="utf-8"))
        task = json.loads((root / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_plan(plan, task)["note_count"], 7)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required")
class ContactScheduleTests(unittest.TestCase):
    def setUp(self):
        self.notes = json.loads(Path(__file__).with_name("example.plan.json").read_text(encoding="utf-8"))["notes"]
        self.positions = {60: -0.05875, 67: 0.03525, 69: 0.05875}

    def target(self, time):
        return target_at(time, self.notes, self.positions, 1.0)

    def test_warmup_and_end_are_above_keys(self):
        self.assertAlmostEqual(self.target(0)[2], 0.080)
        self.assertAlmostEqual(self.target(9.2)[2], 0.080)

    def test_repeated_note_is_released(self):
        self.assertLess(self.target(1.7)[2], 0.01)
        self.assertGreater(self.target(1.99)[2], 0.079)
        self.assertGreater(self.target(2.3)[2], 0.079)
        self.assertLess(self.target(2.7)[2], 0.01)

    def test_cross_key_travel_happens_above_keyboard(self):
        midway = self.target(3.175)
        self.assertAlmostEqual(midway[1], (self.positions[60] + self.positions[67]) / 2)
        self.assertAlmostEqual(midway[2], 0.080)
        self.assertAlmostEqual(self.target(3.49)[1], self.positions[67])
        self.assertAlmostEqual(self.target(3.49)[2], 0.080)

    def test_soft_limit_overshoot_does_not_make_duplicate_onset(self):
        import numpy as np

        q = np.array([0.0, 0.060, 0.090])
        original = q.copy()
        active = key_activation(q, np.tile([0.0, 0.066], (3, 1)), 0.00872665)
        self.assertEqual(active.tolist(), [False, True, True])
        np.testing.assert_array_equal(q, original)

    def test_non_finite_readout_is_rejected(self):
        import numpy as np

        with self.assertRaises(ValueError):
            key_activation(np.array([float("nan")]), np.array([[0.0, 0.066]]), 0.00872665)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco required")
class ActuatorTransmissionTests(unittest.TestCase):
    def test_joint_and_tendon_targets_are_refreshed_without_moving_plant(self):
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="j1" axis="0 0 1"/><geom size=".1"/>
          <body pos=".3 0 0"><joint name="j2" axis="0 0 1"/><geom size=".1"/></body>
          </body></worldbody><tendon><fixed name="sum"><joint joint="j1" coef="1"/>
          <joint joint="j2" coef="1"/></fixed></tendon><actuator>
          <position joint="j1" ctrlrange="-1 1"/>
          <position tendon="sum" ctrlrange="-1 1"/></actuator></mujoco>''')
        plant, ik = mujoco.MjData(model), mujoco.MjData(model)
        ik.qpos[:] = [0.2, 0.3]
        mujoco.mj_kinematics(model, ik)
        np.testing.assert_allclose(position_targets(model, ik), [0.2, 0.5])
        np.testing.assert_array_equal(plant.qpos, [0, 0])


if __name__ == "__main__":
    unittest.main()
