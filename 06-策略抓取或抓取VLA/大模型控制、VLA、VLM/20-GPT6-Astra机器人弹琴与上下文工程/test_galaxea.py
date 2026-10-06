import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from g05_native_scene import load_prediction
from g05_piano_probe import sha256
from galaxea_piano import build_task
from audit_piano_contacts import audit


class ExportedGalaxeaTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run = Path(self.folder.name)
        with zipfile.ZipFile(Path(__file__).parent / "assets/galaxea-replay.zip") as archive:
            archive.extractall(self.run)

    def test_hands_attach_to_source_wrists_and_licenses_are_preserved(self):
        import xml.etree.ElementTree as ET

        xml = ET.parse(self.run / "scene/scene.xml").getroot()
        for side, prefix in (("left", "lh_"), ("right", "rh_")):
            wrist = next(body for body in xml.iter("body") if body.get("name", "").split("/")[-1] == side + "_arm_link7")
            self.assertTrue(any(body.get("name", "").endswith("/" + prefix + "palm") for body in wrist.iter("body")))
        self.assertFalse(list(xml.iter("freejoint")))
        self.assertFalse(any("forearm_" in joint.get("name", "") for joint in xml.iter("joint")))
        self.assertFalse(any(element.get("name", "").startswith("piano/") for element in xml.find("actuator")))
        for name in ("LICENSE-Galaxea-R1Pro.txt", "NOTICE-Galaxea.txt", "LICENSE-Shadow-Hand.txt"):
            self.assertTrue((self.run / "scene" / name).is_file())
        report = json.loads((self.run / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["base"], "fixed_chassis")
        self.assertEqual(report["g05_calls_during_execution"], 0)
        self.assertFalse(report["native_g05_hand_interface_compatible"])
        self.assertEqual(report["plan_sha256"], sha256(self.run / "source.plan.json"))

    def test_all_physical_substeps_match_eleven_reference_notes(self):
        report = audit(self.run)
        self.assertEqual(len(report["measured_onsets"]), 11)
        self.assertTrue(report["parts_match"] and report["onset_contact_hand_matches"])
        self.assertEqual(report["unexpected_onsets"], [])
        self.assertEqual(report["interhand_contact_physics_steps"], 0)
        self.assertFalse(report["joint_state_teleported_during_execution"])


class SceneArgumentsTests(unittest.TestCase):
    def test_invalid_keyboard_position_is_rejected_before_runtime_imports(self):
        for value in (float("nan"), float("inf"), True, -0.16, 0.16):
            with self.assertRaises(ValueError):
                build_task(Path("missing-model"), piano_y=value)


class NativePredictionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.actions = self.root / "actions-0.npz"
        self.report = self.root / "report.json"
        self.arrays = {f"{s}_{p}": np.zeros((1, 32, d), dtype=np.float32) for s in ("left", "right") for p, d in (("arm", 7), ("gripper", 1))}
        self.metadata = {"backend": "G0.5_Qwen35_autoregressive", "embodiment": "galaxea_r1pro", "calls": [{"actions_file": self.actions.name, "absent_action_groups": ["left_control", "left_gripper", "lower_body", "right_gripper"]}]}
        self.save()

    def save(self):
        np.savez_compressed(self.actions, **self.arrays)
        self.metadata["calls"][0]["actions_sha256"] = sha256(self.actions)
        self.report.write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_preserves_absent_group_mask_and_squeezes_batch(self):
        actions, absent, _ = load_prediction(self.actions, self.report, 0)
        self.assertEqual(actions["right_arm"].shape, (32, 7))
        self.assertIn("left_control", absent)
        self.assertIn("right_gripper", absent)

    def test_general_policy_backend_is_accepted(self):
        self.metadata["backend"] = "G0.5_Qwen35_policy"
        self.save()
        actions, _, _ = load_prediction(self.actions, self.report, 0)
        self.assertEqual(actions["left_arm"].shape, (32, 7))

    def test_changed_action_file_is_rejected(self):
        self.arrays["right_arm"] += 1
        np.savez_compressed(self.actions, **self.arrays)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_prediction(self.actions, self.report, 0)

    def test_wrong_model_or_embodiment_is_rejected(self):
        for key in ("backend", "embodiment"):
            original = self.metadata[key]
            self.metadata[key] = "other"
            self.save()
            with self.assertRaises(ValueError):
                load_prediction(self.actions, self.report, 0)
            self.metadata[key] = original

    def test_wrong_dimension_nonfinite_and_empty_chunk_are_rejected(self):
        for value in (np.zeros((32, 6)), np.full((32, 7), np.nan), np.zeros((0, 7)), np.zeros((65, 7))):
            self.arrays["right_arm"] = value
            self.save()
            with self.assertRaises(ValueError):
                load_prediction(self.actions, self.report, 0)

    def test_mismatched_horizons_are_rejected(self):
        self.arrays["left_gripper"] = np.zeros((31, 1))
        self.save()
        with self.assertRaisesRegex(ValueError, "horizons"):
            load_prediction(self.actions, self.report, 0)

    def test_unknown_absence_mask_is_rejected(self):
        self.metadata["calls"][0]["absent_action_groups"] = ["five_fingers"]
        self.save()
        with self.assertRaisesRegex(ValueError, "absent"):
            load_prediction(self.actions, self.report, 0)


if __name__ == "__main__":
    unittest.main()
