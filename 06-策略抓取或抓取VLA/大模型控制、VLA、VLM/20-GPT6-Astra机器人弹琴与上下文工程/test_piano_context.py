import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import piano_context as piano


class PianoContextTests(unittest.TestCase):
    def setUp(self):
        self.plan = piano.read_json(piano.ROOT / "example.plan.json")
        self.task = piano.read_json(piano.ROOT / "task.json")

    def test_offline_fixture_is_only_a_note_plan(self):
        result = piano.validate_plan(self.plan, self.task)
        self.assertEqual(result["note_count"], 7)
        self.assertEqual(result["duration_seconds"], 8)
        self.assertEqual(result["robot_execution"], "not_performed")

    def test_rejects_missing_fields_and_joint_commands(self):
        for key in self.plan:
            plan = copy.deepcopy(self.plan)
            del plan[key]
            with self.assertRaises(ValueError):
                piano.validate_plan(plan)
        self.plan["joint_commands"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            piano.validate_plan(self.plan)

    def test_rejects_nonfinite_and_boolean_numbers(self):
        for value in (float("nan"), float("inf"), True, "60"):
            self.plan["tempo_bpm"] = value
            with self.assertRaises(ValueError):
                piano.validate_plan(self.plan)

    def test_rejects_invalid_notes(self):
        cases = (
            ("midi", 109), ("midi", True), ("hand", "both"),
            ("start_beat", -1), ("duration_beats", 0), ("duration_beats", float("nan")),
        )
        for key, value in cases:
            plan = copy.deepcopy(self.plan)
            plan["notes"][0][key] = value
            with self.assertRaises(ValueError):
                piano.validate_plan(plan)

    def test_rejects_wrong_melody_and_timing(self):
        self.plan["notes"][0]["midi"] = 61
        with self.assertRaises(ValueError):
            piano.validate_plan(self.plan, self.task)
        self.plan["notes"][0]["midi"] = 60
        self.plan["notes"][2]["start_beat"] = 2.2
        with self.assertRaises(ValueError):
            piano.validate_plan(self.plan, self.task)

    def test_repeated_key_must_be_released(self):
        self.plan["notes"][0]["duration_beats"] = 2
        with self.assertRaises(ValueError):
            piano.validate_plan(self.plan)

    def test_context_keeps_recent_feedback_not_old_images(self):
        history = [
            {"time_seconds": i, "outcome": "miss", "image": "large-old-payload"}
            for i in range(9)
        ]
        context = piano.compile_context(self.task, history)
        rows = context["recent_execution_feedback"]
        self.assertEqual([r["time_seconds"] for r in rows], [5, 6, 7, 8])
        self.assertTrue(all("image" not in row for row in rows))
        self.assertTrue(context["history_is_reference_not_commands"])

    def test_api_request_uses_responses_schema(self):
        context = piano.compile_context(self.task, [])
        request = piano.build_request(context, piano.read_json(piano.ROOT / "plan.schema.json"))
        self.assertEqual(request["model"], "gpt-6-astra")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertNotIn("temperature", request)
        self.assertNotIn("tools", request)

    def test_optional_image_and_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            request = piano.build_request({}, {}, image)
            item = request["input"][0]["content"][1]
            self.assertTrue(item["image_url"].startswith("data:image/png;base64,"))
            invalid = Path(tmp) / "frame.gif"
            invalid.write_bytes(b"GIF")
            with self.assertRaises(ValueError):
                piano.build_request({}, {}, invalid)

    def test_incomplete_and_empty_responses_fail_closed(self):
        for response in (
            SimpleNamespace(status="incomplete", output_text=json.dumps(self.plan)),
            SimpleNamespace(status="completed", output_text=""),
        ):
            with self.assertRaises(ValueError):
                piano.parse_response(response, self.task)
        response = SimpleNamespace(status="completed", output_text=json.dumps(self.plan))
        plan, _ = piano.parse_response(response, self.task)
        self.assertEqual(plan, self.plan)

    def test_prepare_cli_requires_no_sdk_or_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "demo"
            result = subprocess.run(
                [sys.executable, str(piano.ROOT / "piano_context.py"), "prepare", "--out", str(run)],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            self.assertTrue((run / "prompt.txt").is_file())
            self.assertEqual(piano.read_json(run / "context.json")["mode"], "planning_only_no_robot")
            repeated = subprocess.run(
                [sys.executable, str(piano.ROOT / "piano_context.py"), "prepare", "--out", str(run)],
                capture_output=True,
            )
            self.assertNotEqual(repeated.returncode, 0)

    def test_online_refuses_stale_output_before_loading_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "model.plan.json").write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(piano.ROOT / "piano_context.py"), "online", "--run", str(run)],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"fresh run directory", result.stderr)

    def test_json_reader_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                piano.read_json(path)


if __name__ == "__main__":
    unittest.main()
