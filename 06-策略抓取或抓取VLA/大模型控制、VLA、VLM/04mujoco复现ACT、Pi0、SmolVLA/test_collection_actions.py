"""Regression checks for command labels and keyboard collection episodes."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.y_env import SimpleEnv
from mujoco_env.y_env2 import SimpleEnv2


ROOT = Path(__file__).resolve().parent
COLLECTORS = (
    "1.collect_data.ipynb",
    "1.collect_data_nova5.ipynb",
    "1.collect_data_xarm6.ipynb",
    "1.collect_data_xarm7.ipynb",
    "5.language_env.ipynb",
)


def make_env(cls=SimpleEnv, n_arm=6):
    env = cls.__new__(cls)
    env.joint_names = [f"joint{i + 1}" for i in range(n_arm)]
    env.gripper_joint_names = ["rh_r1", "rh_r2", "lh_r1", "lh_r2"]
    env.gripper_joint_limits = {name: (0.0, 1.0) for name in env.gripper_joint_names}
    env.gripper_monitor_joint = "rh_r1"
    env.env = SimpleNamespace(
        get_qpos_joints=lambda joint_names: np.zeros(len(joint_names)),
        get_qpos_joint=lambda name: np.array([0.0]),
        get_body_names=lambda prefix: [],
        forward=lambda **kwargs: None,
        get_pR_body=lambda body_name: (np.zeros(3), np.eye(3)),
    )
    env.action_type = "joint_angle"
    env.state_type = "joint_angle"
    env.compute_q = np.zeros(n_arm)
    env.last_q = np.zeros(n_arm)
    env.gripper_cmd_scalar = 0.0
    env.gripper_rate_per_step = 0.03
    env.action_deadband = 1e-6
    env.xarm7_contact_stop_close = False
    env.nova5_gripper_mode = False
    env.use_actuator_ctrl_mode = False
    env.ee_body_name = "tcp_link"
    env.p0 = np.zeros(3)
    env.R0 = np.eye(3)
    env.ik_backend = "mujoco"
    env.ik_max_tick = 50
    env.ik_stepsize = 1.0
    env.ik_eps = 1e-2
    env.ik_trim_th = 0.1
    return env


def robot_parser(relative_xml):
    xml_path = ROOT / "asset" / relative_xml
    tree = ET.parse(xml_path)
    # Resolve mesh paths without copying assets into MuJoCo's size-limited VFS.
    for node in tree.iter():
        name = node.get("file")
        if name:
            candidates = (xml_path.parent / name, ROOT / "asset" / name)
            path = next(p for p in candidates if p.is_file())
            node.set("file", path.relative_to(ROOT).as_posix())
    # MuJoCo 3.1.6 on Windows needs ASCII resource paths, relative to this chapter.
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        with patch("mujoco_env.mujoco_parser.get_monitor_size", return_value=(1920, 1080)):
            return MuJoCoParserClass(
                xml_string=ET.tostring(tree.getroot(), encoding="unicode"),
                verbose=False,
            )
    finally:
        os.chdir(previous_cwd)


class CommandLabelTests(unittest.TestCase):
    def test_targets_are_distinct_from_measured_state_on_real_models(self):
        models = (
            ("robotis_omy/omy.xml", 6),
            ("nova5_description/nova5.xml", 6),
            ("xarm6_description/lite6.xml", 6),
            ("xarm7_description/xarm7.xml", 7),
        )
        for model_path, n_arm in models:
            with self.subTest(model=model_path):
                env = make_env(n_arm=n_arm)
                env.env = robot_parser(model_path)
                env._configure_robot_bindings()
                self.assertEqual(len(env.joint_names), n_arm)
                measured = env.get_joint_state()
                target = measured[:n_arm] + np.linspace(0.02, 0.08, n_arm)
                returned_state = env.step(np.r_[target, 1.0])
                label = env.get_commanded_joint_action()
                np.testing.assert_allclose(returned_state, measured)
                np.testing.assert_allclose(label[:-1], target)
                self.assertFalse(np.allclose(label[:-1], measured[:-1]))
                self.assertEqual(label.shape, (n_arm + 1,))
                self.assertEqual(label.dtype, np.float32)
                self.assertAlmostEqual(label[-1], 0.03)
                np.testing.assert_allclose(env.q[:n_arm], label[:-1])
                if env.use_actuator_ctrl_mode:
                    lo, hi = env.gripper_actuator_ctrlrange
                    index = env.ctrl_name_to_idx[env.gripper_actuator_name]
                    self.assertAlmostEqual(env.ctrl_cmd[index], lo + label[-1] * (hi - lo), places=5)

    def test_language_environment_returns_command_not_sensor_gripper(self):
        env = make_env(SimpleEnv2)
        target = np.linspace(0.1, 0.6, 6)
        measured = env.step(np.r_[target, 1.0])
        np.testing.assert_allclose(measured, np.zeros(7))
        np.testing.assert_allclose(env.get_commanded_joint_action(), np.r_[target, 1.0])

    def test_ik_result_is_the_recorded_joint_target(self):
        for cls, module in ((SimpleEnv, "mujoco_env.y_env"), (SimpleEnv2, "mujoco_env.y_env2")):
            with self.subTest(cls=cls.__name__):
                env = make_env(cls)
                env.action_type = "eef_pose"
                target = np.linspace(0.1, 0.6, 6)
                with patch(f"{module}.solve_ik", return_value=(target, None, None)):
                    measured = env.step(np.array([0.007, 0, 0, 0, 0, 0, 1.0]))
                np.testing.assert_allclose(measured, np.zeros(7))
                np.testing.assert_allclose(env.get_commanded_joint_action()[:-1], target)

    def test_labels_are_independent_snapshots(self):
        for cls in (SimpleEnv, SimpleEnv2):
            with self.subTest(cls=cls.__name__):
                env = make_env(cls)
                env.step(np.r_[np.ones(6), 1.0])
                label = env.get_commanded_joint_action()
                original = label.copy()
                env.compute_q[:] = -1.0
                env.gripper_cmd_scalar = 0.0
                np.testing.assert_array_equal(label, original)

    def test_contact_limited_gripper_label_matches_applied_command(self):
        env = make_env(n_arm=7)
        env.ee_body_name = "xarm_gripper_base_link"
        env.xarm7_contact_stop_close = True
        env._xarm7_has_gripper_contact = lambda: True
        env.gripper_cmd_scalar = 0.4
        env.step(np.r_[np.zeros(7), 1.0])
        self.assertAlmostEqual(env.get_commanded_joint_action()[-1], 0.4)
        self.assertAlmostEqual(env.q[7], 0.4)

    def test_reset_replaces_previous_episode_commands(self):
        for cls, module in ((SimpleEnv, "mujoco_env.y_env"), (SimpleEnv2, "mujoco_env.y_env2")):
            with self.subTest(cls=cls.__name__):
                env = make_env(cls)
                env.compute_q[:] = 2.0
                env.gripper_cmd_scalar = 1.0
                env.step_env = lambda: None
                poses = 2 if cls is SimpleEnv else 3
                env.get_obj_pose = lambda: tuple(np.zeros(3) for _ in range(poses))
                env.env.set_p_base_body = lambda **kwargs: None
                env.env.set_R_base_body = lambda **kwargs: None
                env.set_instruction = lambda: None
                home = np.linspace(0.1, 0.6, 6)
                with patch(f"{module}.solve_ik", return_value=(home, None, None)):
                    with patch(f"{module}.sample_xyzs", side_effect=lambda n, **kwargs: np.zeros((n, 3))):
                        env.reset(seed=0)
                        np.testing.assert_allclose(env.get_commanded_joint_action(), np.r_[home, 0.0])
                        env.action_type = "eef_pose"
                        env.step(np.zeros(7))
                        np.testing.assert_allclose(env.get_commanded_joint_action(), np.r_[home, 0.0])


class FrameSink:
    def __init__(self, shape):
        self.shape = shape
        self.frames = []
        self.episodes = []

    def add_frame(self, frame, task):
        if frame["action"].shape != self.shape:
            raise AssertionError(f"action shape mismatch: {frame['action'].shape} != {self.shape}")
        self.frames.append({**{key: value.copy() for key, value in frame.items()}, "task": task})

    def save_episode(self):
        if not self.frames:
            raise AssertionError("attempted to save an empty episode")
        self.episodes.append(self.frames)
        self.frames = []

    def clear_episode_buffer(self):
        self.frames = []


class TeleopFixture:
    def __init__(self, events, n_arm):
        self.events = iter(events)
        self.event = None
        self.n_arm = n_arm
        self.ticks = 0
        self.resets = 0
        self.instruction = "Place the red mug on the plate."
        self.obj_init_pose = np.zeros(6, dtype=np.float32)
        self.env = SimpleNamespace(is_viewer_alive=lambda: self.ticks < 20, loop_every=lambda HZ: True)
        self.pose = np.arange(6, dtype=np.float32)
        self.sensor = np.arange(n_arm + 1, dtype=np.float32) / 10
        self.target = np.r_[np.ones(n_arm), 0.4].astype(np.float32)

    def step_env(self):
        self.event = next(self.events)
        self.ticks += 1

    def check_success(self):
        return self.event == "success"

    def teleop_robot(self):
        action = np.zeros(7)
        if self.event == "move":
            action[:2] = [1.0, -1.0]
        return action, self.event == "reset"

    def reset(self, **kwargs):
        self.resets += 1

    def get_ee_pose(self):
        return self.pose.copy()

    def get_joint_state(self):
        return self.sensor.copy()

    def grab_image(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        return image, image.copy()

    def step(self, action):
        # Detect collectors that source observations after preparing the command.
        return self.sensor + 0.7

    def get_commanded_joint_action(self):
        return self.target.copy()

    def render(self, **kwargs):
        pass


class NotebookCollectionTests(unittest.TestCase):
    def collect(self, name, events, num_demo):
        notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        create_source = next(s for s in sources if "LeRobotDataset.create(" in s)
        create = next(
            n for n in ast.walk(ast.parse(create_source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create"
        )
        features = ast.literal_eval(next(k.value for k in create.keywords if k.arg == "features"))
        sink = FrameSink(features["action"]["shape"])
        env = TeleopFixture(events, sink.shape[0] - 1)
        source = next(s for s in sources if "dataset.add_frame(" in s)
        with patch("builtins.print"):
            exec(compile(source, name, "exec"), {
                "np": np, "Image": Image, "PnPEnv": env, "dataset": sink,
                "NUM_DEMO": num_demo, "SEED": 0, "TASK_NAME": "Put mug cup on the plate",
            })
        self.assertEqual(len(sink.episodes), num_demo)
        for episode in sink.episodes:
            for frame in episode:
                np.testing.assert_allclose(frame["action"], env.target)
                state = env.sensor[:6] if name == "5.language_env.ipynb" else env.pose
                np.testing.assert_allclose(frame["observation.state"], state)
        return sink, env

    def test_all_collectors_save_targets_and_opposing_inputs_start_recording(self):
        for name in COLLECTORS:
            with self.subTest(notebook=name):
                sink, env = self.collect(name, ["move", "hold", "success"], 1)
                self.assertEqual(len(sink.episodes[0]), 2)
                self.assertEqual(env.resets, 0)
                self.assertFalse(sink.frames)

    def test_next_episode_waits_for_new_movement(self):
        for name in COLLECTORS:
            with self.subTest(notebook=name):
                sink, env = self.collect(name, ["move", "success", "hold", "move", "success"], 2)
                self.assertEqual([len(e) for e in sink.episodes], [1, 1])
                self.assertEqual(env.resets, 1)

    def test_manual_reset_discards_unsaved_frames(self):
        for name in COLLECTORS:
            with self.subTest(notebook=name):
                sink, env = self.collect(name, ["move", "reset", "hold", "move", "success"], 1)
                self.assertEqual(len(sink.episodes[0]), 1)
                self.assertEqual(env.resets, 1)

    def test_success_before_recording_does_not_save_empty_episode(self):
        for name in COLLECTORS:
            with self.subTest(notebook=name):
                sink, env = self.collect(name, ["success", "move", "success"], 1)
                self.assertEqual(len(sink.episodes[0]), 1)
                self.assertEqual(env.resets, 0)


if __name__ == "__main__":
    unittest.main()
