import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from g05_piano_client import PianoPolicy


class ModelBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = PianoPolicy.__new__(PianoPolicy)
        self.client.folder = Path(self.temp.name)
        self.client.calls = []
        self.client.process = SimpleNamespace(stdin=io.StringIO())
        self.client._read = lambda: {"actions": "actions.npz", "report": "report.json", "call_index": 0}
        self.obs = {side + "_arm": np.zeros(7, np.float32) for side in ("left", "right")}
        self.actions = {side + "_arm": np.tile(np.arange(6)[:, None] * .1, (1, 7)) for side in ("left", "right")}
        self.report = {"calls": [{"observation_sha256": "fresh", "inference_seconds": 1., "actions_sha256": "output"}]}

    def infer(self, **kwargs):
        with patch("g05_piano_client.sha256", return_value="fresh"), patch("g05_piano_client.load_prediction",
                return_value=(self.actions, set(), self.report)):
            return self.client.infer_observation(0, self.obs, "task", "plan", **kwargs)

    def test_legacy_mean_is_preserved(self):
        np.testing.assert_allclose(self.infer()["left"], [.15] * 7)

    def test_chunks_keep_time_order_and_are_bounded(self):
        chunk = self.infer(chunk=True, seed=7)["left"]
        self.assertEqual(chunk.shape, (6, 7))
        np.testing.assert_allclose(chunk[:, 0], [0, .1, .2, .25, .25, .25])
        self.assertEqual(self.client.calls[0]["seed"], 7)
        self.assertIn('"seed": 7', self.client.process.stdin.getvalue())

    def test_absent_group_is_not_executed(self):
        with patch("g05_piano_client.sha256", return_value="fresh"), patch("g05_piano_client.load_prediction",
                return_value=(self.actions, {"left_control"}, self.report)):
            targets = self.client.infer_observation(0, self.obs, "task", "plan", chunk=True)
        self.assertNotIn("left", targets)
        self.assertIn("right", targets)

    def test_stale_observation_rejected(self):
        self.report["calls"][0]["observation_sha256"] = "stale"
        with self.assertRaises(ValueError):
            self.infer()

    def test_invalid_measured_state_rejected(self):
        self.obs["left_arm"][0] = np.nan
        with self.assertRaises(ValueError):
            self.infer()

    def test_invalid_seed_rejected(self):
        for seed in (True, -1, 2**32, .1):
            with self.assertRaises(ValueError):
                self.infer(seed=seed)


if __name__ == "__main__":
    unittest.main()
