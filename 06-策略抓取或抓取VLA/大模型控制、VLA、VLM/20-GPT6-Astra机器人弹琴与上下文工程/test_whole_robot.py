import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

from replay_windows import compare_parts, replay
from whole_robot_piano import validate_bimanual_plan
from audit_piano_contacts import audit, main as audit_main

ROOT = Path(__file__).parent


class BimanualPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "verified_whole_gpt6.plan.json").read_text(encoding="utf-8"))
        self.expected = {s: [n["midi"] for n in self.plan["notes"] if n["hand"] == s] for s in ("left", "right")}

    def test_verified_plan_matches_both_task_parts(self):
        validate_bimanual_plan(self.plan)
        self.assertEqual(len(self.plan["notes"]), 11)

    def test_wrong_side_is_rejected(self):
        self.plan["notes"][0]["hand"] = "right"
        with self.assertRaises(ValueError):
            validate_bimanual_plan(self.plan)

    def test_changed_bass_or_tempo_is_rejected(self):
        for field, value in (("midi", 49), ("duration_beats", 1)):
            changed = copy.deepcopy(self.plan)
            changed["notes"][0][field] = value
            with self.assertRaises(ValueError):
                validate_bimanual_plan(changed)
        self.plan["tempo_bpm"] = 61
        with self.assertRaises(ValueError):
            validate_bimanual_plan(self.plan)

    def test_simultaneous_hands_need_not_have_same_event_order(self):
        events = [{"midi": n["midi"]} for n in self.plan["notes"]]
        events[0], events[1] = events[1], events[0]
        self.assertTrue(compare_parts(events, self.expected)[2])

    def test_extraneous_note_is_not_filtered_into_success(self):
        events = [{"midi": n["midi"]} for n in self.plan["notes"]] + [{"midi": 61}]
        _, unexpected, matches = compare_parts(events, self.expected)
        self.assertEqual(unexpected, [{"midi": 61}])
        self.assertFalse(matches)

    def test_missing_or_duplicate_repeated_note_fails(self):
        events = [{"midi": n["midi"]} for n in self.plan["notes"]]
        self.assertFalse(compare_parts(events[:2] + events[3:], self.expected)[2])
        self.assertFalse(compare_parts(events + [{"midi": 60}], self.expected)[2])

    def test_overlapping_registers_require_different_evaluation(self):
        with self.assertRaises(ValueError):
            compare_parts([], {"left": [60], "right": [60]})


class ContactAuditCLITests(unittest.TestCase):
    def test_wrong_notes_wrong_hand_or_interhand_contact_fail(self):
        passed = {"parts_match": True, "onset_contact_hand_matches": True, "interhand_contact_physics_steps": 0}
        for key, value in (("parts_match", False), ("onset_contact_hand_matches", False), ("interhand_contact_physics_steps", 1)):
            report = dict(passed, **{key: value})
            with self.subTest(key=key), patch("sys.argv", ["audit_piano_contacts.py", "--run", "unused"]), patch("audit_piano_contacts.audit", return_value=report), patch("builtins.print"):
                with self.assertRaises(SystemExit) as exit_status:
                    audit_main()
                self.assertEqual(exit_status.exception.code, 1)


class ExportedWholeRobotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run = Path(self.folder.name)
        with zipfile.ZipFile(ROOT / "assets/whole-robot-replay.zip") as archive:
            archive.extractall(self.run)

    def test_report_uses_reference_plan_not_measured_events_as_expectation(self):
        plan = (ROOT / "verified_whole_gpt6.plan.json").read_bytes()
        report = json.loads((self.run / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plan_sha256"], hashlib.sha256(plan).hexdigest())
        self.assertEqual(report["expected_onsets"], [n["midi"] for n in json.loads(plan)["notes"]])
        self.assertTrue(report["parts_match"] and report["audio_parts_match"])
        self.assertEqual(report["unexpected_onsets"], [])
        for side in ("left", "right"):
            self.assertGreater(report["hand_key_contact_frames"][side], 0)
            self.assertGreater(report["arm_joint_excursion_radians"][side][side + "_elbow_joint"], 0.01)

    def test_hands_are_descendants_of_robot_wrists_without_virtual_slides(self):
        import xml.etree.ElementTree as ET

        xml = ET.parse(self.run / "scene/scene.xml").getroot()
        for side, prefix in (("left", "lh_"), ("right", "rh_")):
            wrist = next(b for b in xml.iter("body") if b.get("name", "").endswith("/" + side + "_wrist_yaw_link"))
            self.assertTrue(any(b.get("name", "").endswith("/" + prefix + "palm") for b in wrist.iter("body")))
        self.assertFalse(any("forearm_" in j.get("name", "") for j in xml.iter("joint")))
        self.assertFalse(list(xml.iter("freejoint")))
        for element in xml.find("actuator"):
            self.assertFalse(element.get("name", "").startswith("piano/"))
        self.assertTrue((self.run / "scene/LICENSE-Unitree-G1.txt").exists())
        self.assertTrue((self.run / "scene/LICENSE-Shadow-Hand.txt").exists())

    def test_native_actuator_replay_matches_both_parts(self):
        report = replay(self.run)
        self.assertTrue(report["onset_sequence_matches"])
        self.assertEqual(report["unexpected_onsets"], [])

    def test_all_physical_substeps_preserve_notes_and_contact_ownership(self):
        report = audit(self.run)
        self.assertEqual(report["sample_hz"], 500)
        self.assertTrue(report["parts_match"])
        self.assertTrue(report["onset_contact_hand_matches"])
        self.assertEqual(report["interhand_contact_physics_steps"], 0)

    def test_no_visual_robot_column_but_fixed_pelvis_is_disclosed(self):
        import xml.etree.ElementTree as ET

        xml = ET.parse(self.run / "scene/scene.xml").getroot()
        names = {g.get("name", "") for g in xml.iter("geom")}
        self.assertNotIn("support_column", names)
        self.assertEqual(sum(name.startswith("piano_leg_") for name in names), 2)
        report = json.loads((self.run / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["base"], "fixed_pelvis")
        self.assertFalse(report["visible_robot_support"])


if __name__ == "__main__":
    unittest.main()
