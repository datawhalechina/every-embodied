"""Run one readable workspace tool locally; record its atomic inputs and results."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
from PIL import Image

from .coordinates import (arguments_to_interface, contains_uv, response_to_pixels,
                          latest_image_grounding_reminder)
from .evidence import _action_evidence_text, _compact_persistent_tracking
from .files import atomic_json, load_json
from .history import CallHistory, record_failure
from .media import MediaExtractor, safe_json
from .perception import freeze_camera, publish
from .robot import Robot
from .receipt import model_receipt_text

CAPTURE_ROLES = {f"capture_{role}_camera": role for role in ("head", "left_wrist", "right_wrist")}
PLAN_TOOLS = {"plan_grasp_point_filter_rgbd_lite", "plan_press_point"}


class LocalMedia:
    def get_media(self, *args, **kwargs):
        raise ValueError("Workspace observations must be local files; no image HTTP fallback")


def validate_pixels(robot, arguments):
    if not contains_uv(arguments):
        return
    state = load_json(robot.state_dir / "state.json", {})
    if arguments.get("image_id") != state.get("latest_image_id") or not state.get("latest_image_id"):
        raise ValueError("Point selection requires the latest head_image_id in perception/latest.json; inspect that actual image first")
    source = Path(state["latest_head_path"])
    with Image.open(source) as image:
        if image.size != (720, 720):
            raise ValueError(f"Clickable head view must be exactly 720 × 720; got {image.size}")
    if state.get("latest_sequence") != robot._official_adapter.current.sequence:
        raise ValueError("Robot state changed after this image. Capture and inspect a new head view before clicking")
    # Recursion validates every nested point, including multi-point tools.
    arguments_to_interface(arguments)


def capture_role(name, arguments, result):
    if name in CAPTURE_ROLES:
        return CAPTURE_ROLES[name]
    if name == "close_gripper":
        return str(arguments.get("arm", "right")) + "_wrist"
    if name == "exec_plan_pose":
        return str(result.get("plan_arm") or result.get("arm") or arguments.get("arm") or "right") + "_wrist"
    if "left_eef_pose_in_wrist" in name:
        return "left_wrist"
    if "right_eef_pose_in_wrist" in name:
        return "right_wrist"
    return "head"


def run(name, schema, actions, arguments, *, robot=None):
    robot = robot or Robot()
    with robot.exclusive():
        history = CallHistory(robot, name, arguments)
        robot.history = history
        result, images, perception = None, [], None
        error = None
        try:
            jsonschema.Draft202012Validator(schema).validate(arguments)
            validate_pixels(robot, arguments)
            converted = arguments_to_interface(arguments)
            if name in CAPTURE_ROLES:
                result = freeze_camera(robot, CAPTURE_ROLES[name])
            else:
                signature = inspect.signature(actions)
                if "session_id" in signature.parameters or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                    converted["session_id"] = robot.session_id
                state_path = robot.state_dir / "state.json"
                state = load_json(state_path, {})
                state["latest_image_id"] = ""
                atomic_json(state_path, state)
                result = robot.execute(actions, **converted)
            result = dict(result or {"ok": False, "error": "Tool completed without an explicit result"})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = {"ok": False, "error": error}
        try:
            robot.observe()
            # Latest perception is always complete, even when the baseline
            # model-visible image for this tool is a wrist or plan preview.
            head = result if name == "capture_head_camera" and result.get("ok") else freeze_camera(robot, "head")
            perception = publish(robot, history, head)
            role = capture_role(name, arguments, result)
            is_plan_preview = name in PLAN_TOOLS and result.get("ok") and (result.get("marked_image_url") or result.get("gripper_visualization"))
            if is_plan_preview:
                visible = result
            elif role == "head":
                visible = head
            elif name in CAPTURE_ROLES and result.get("ok"):
                visible = result
            else:
                visible = freeze_camera(robot, role)
            extractor = MediaExtractor(LocalMedia(), max_bytes=16 * 1024 * 1024, max_images=1,
                                       max_model_bytes=256 * 1024, max_model_edge=720, jpeg_quality=95)
            selected = extractor.extract(visible, tool_name=name)
            for index, media in enumerate(selected.selected):
                extension = "jpg" if media.mime_type == "image/jpeg" else "png"
                path = history.path / f"returned_image_{index}.{extension}"
                path.write_bytes(media.data)
                caption = (latest_image_grounding_reminder(perception["head_image_id"])
                           if role == "head" and not is_plan_preview else
                           f"{media.source_feed or role} observation only; click the latest head image "
                           f"{perception['head_image_id'] or 'after a new capture'}." )
                images.append({**media.public(), "path": str(path), "caption": caption,
                               "image_id": media.source_image_id or visible.get("image_id"), "feed": media.source_feed or role})
            if not images:
                raise RuntimeError("No model image could be selected: " + "; ".join(selected.warnings))
        except Exception as exc:
            # Do not conceal a successful action just because post-action
            # observation failed; mark the observation failure separately.
            error = (error + "; " if error else "") + f"Observation failed: {type(exc).__name__}: {exc}"
            state_path = robot.state_dir / "state.json"
            state = load_json(state_path, {})
            state["latest_image_id"] = ""
            atomic_json(state_path, state)
        converted_result = response_to_pixels(safe_json(result))
        try:
            if name in {"robot_step", "joint_control"}:
                # The atomic API's exact 27-vector and named joint feedback
                # must not inherit the macro result compactor's list limits.
                evidence = converted_result
            else:
                compact = _action_evidence_text(SimpleNamespace(data={"tool_name": name, "response": converted_result}))
                evidence = json.loads(compact.removeprefix("action_evidence="))
        except (TypeError, ValueError) as exc:
            error = (error + "; " if error else "") + f"Result presentation failed: {exc}"
            evidence = {"error": error}
        # Use the coherent last observation payload. A reset/disconnect after
        # an action must not prevent its input/output history from completing.
        tracking = response_to_pixels(_compact_persistent_tracking(robot.tracking))
        state = load_json(robot.state_dir / "state.json", {})
        receipt = {"tool_name": name,
            "response": evidence, "images": images, "persistent_tracking": tracking,
            "rollout_budget": robot.budget, "joint_commands_sent": robot.step_count,
            "latest_clickable_image_id": state.get("latest_image_id", ""),
            "perception": str(robot.workspace / "perception/latest.json") if perception else None,
            "task_skill_state": {"baseline": "behavior-v2-baseline", "active_skill": state.get("active_skill")},
            "observation_instruction": "Read attached pixels; if only a path is shown, use native Read. Use the latest head image for clicks."}
        if error:
            receipt["error"] = error
        return history.finish(receipt, {"operation_result": converted_result, "perception": perception, "error": error})


def main(name, schema, actions):
    parser = argparse.ArgumentParser(description=actions.__doc__)
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object; see SCHEMA in this file")
    parser.add_argument("--schema", action="store_true", help="Print input schema without connecting to a robot")
    options = parser.parse_args()
    if options.schema:
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        return
    arguments = options.arguments
    try:
        arguments = json.loads(options.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool input must be a JSON object")
        result = run(name, schema, actions, arguments)
    except Exception as exc:
        result = {"tool_name": name, "error": f"{type(exc).__name__}: {exc}", "images": []}
        result = record_failure(name, arguments, result)
    print(model_receipt_text(result), flush=True)
