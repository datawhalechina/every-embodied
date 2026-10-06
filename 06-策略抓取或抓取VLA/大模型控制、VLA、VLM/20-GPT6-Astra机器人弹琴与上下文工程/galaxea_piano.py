"""R1 Pro with simulation-only Shadow Hands; not a native dexterous G0.5 policy."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_galaxea import COMMIT
from piano_visuals import style_hands, style_stage
from piano_context import number
from whole_robot_piano import run


def build_task(model_dir, native_grippers=False, piano_y=None, policy_hands=False):
    if piano_y is not None:
        number(piano_y, "piano_y", -0.15, 0.15)
    import numpy as np
    from dm_control import composer, mjcf
    from dm_env import specs
    from robopianist.models.arenas import stage
    from robopianist.models.hands import HandSide, shadow_hand
    from robopianist.suite.tasks import base

    class R1Pro(composer.Entity):
        def _build(self):
            manifest = json.loads((model_dir / "source_manifest.json").read_text(encoding="utf-8"))
            if manifest["commit"] != COMMIT:
                raise ValueError("Unexpected Galaxea source commit")
            for name, record in manifest["files"].items():
                if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Changed model resource: " + name)
            self._mjcf_root = mjcf.from_path(str(model_dir / "r1_pro.xml"))
            self._mjcf_root.model = "galaxea_r1pro"
            self.seed = {}
            self.arm_joints = {}
            self.hands = {}
            self.hand_seed = []
            shell = self._mjcf_root.asset.add("material", name="robot_satin", rgba=(0.80, 0.83, 0.86, 1), specular=0.12, shininess=0.2)
            torso = self._mjcf_root.asset.add("material", name="robot_torso", rgba=(0.30, 0.33, 0.36, 1), specular=0.1, shininess=0.2)
            for geom in self._mjcf_root.find_all("geom"):
                if geom.group == 1:
                    geom.material = torso if geom.parent.name in ("torso_link4", "torso_link3") else shell
                    geom.rgba = None
                if geom.name and "_collision_" in geom.name:
                    # Collision meshes overlap the visual meshes; rendering both flickers.
                    geom.group = 3
            for body in self._mjcf_root.find_all("body"):
                body.gravcomp = 1
            for joint in list(self._mjcf_root.find_all("joint")):
                if joint.name.startswith(("steer_motor_", "wheel_motor_")):
                    joint.remove()
            for geom in self._mjcf_root.find_all("geom"):
                if geom.parent.name == "base_link" or geom.parent.name.startswith(("steer_motor_", "wheel_motor_")):
                    geom.contype = geom.conaffinity = 0
            for joint in self._mjcf_root.find_all("joint"):
                joint.damping = 2
                joint.armature = 0.01
                limits = joint.range if joint.limited else (-6.28319, 6.28319)
                self._mjcf_root.actuator.add("position", name=joint.name + "_act", joint=joint, kp=1000, kv=60, ctrllimited=True, ctrlrange=limits)
            for side, hand_side in (("left", HandSide.LEFT), ("right", HandSide.RIGHT)):
                self.arm_joints[side] = [self._mjcf_root.find("joint", f"{side}_arm_joint{i}") for i in range(1, 8)]
                values = (-0.4, 1.3, -0.7, -1.57, 1.3, -0.4, -0.8) if side == "left" else (-0.4, -1.3, 0.7, -1.57, -1.3, -0.4, 0.8)
                self.seed.update({j.name: value for j, value in zip(self.arm_joints[side], values)})
                if native_grippers:
                    self.seed.update({f"{side}_gripper_finger_joint1": 0.05, f"{side}_gripper_finger_joint2": -0.05})
                    self._mjcf_root.find("body", side + "_arm_link7").add("camera", name=side + "_wrist_rgb", pos=(0, 0.055, -0.01), quat=(1, 0, 0, 0), fovy=75)
                    continue
                hand = shadow_hand.ShadowHand(side=hand_side, forearm_dofs=())
                hand.root_body.pos = (0, 0, 0)
                hand.root_body.quat = (1, 0, 0, 0)
                for geom in list(hand.root_body.geom):
                    geom.remove()
                hand.root_body.inertial.mass = 0.001
                hand.root_body.inertial.pos = (0, 0, 0)
                hand.root_body.inertial.diaginertia = (1e-6, 1e-6, 1e-6)
                hand.mjcf_model.find("body", ("lh_" if side == "left" else "rh_") + "wrist").pos = (0, 0, 0)
                for body in hand.mjcf_model.find_all("body"):
                    body.gravcomp = 1
                wrist = self._mjcf_root.find("body", side + "_arm_link7")
                # The retained tool origin is beyond link7's visible flange.
                # Replace the removed gripper housing with a fixed adapter.
                adapter = wrist.add("body", name=side + "_shadow_adapter",
                                    pos=(-0.0295, 0, -0.124), gravcomp=1)
                adapter.add("inertial", pos=(0, 0, 0), mass=0.08,
                            diaginertia=(4e-5, 4e-5, 2e-5))
                for name, z, radius, half_length, rgba in (
                    ("arm_flange", .030, .027, .006, (.32, .35, .38, 1)),
                    ("sleeve", 0., .021, .030, (.72, .76, .79, 1)),
                    ("hand_flange", -.028, .024, .006, (.32, .35, .38, 1)),
                ):
                    adapter.add("geom", name=side + "_adapter_" + name,
                                type="cylinder", pos=(0, 0, z),
                                size=(radius, half_length), rgba=rgba,
                                group=1, contype=0, conaffinity=0, mass=0)
                adapter.add("geom", name=side + "_adapter_collision",
                            type="cylinder", size=(.022, .035), group=3,
                            contype=1, conaffinity=1, mass=0)
                # Original gripper's fixed-joint origin in new_robot.urdf, in link7.
                mount = wrist.add("site", name=side + "_shadow_mount", pos=(-0.0295, 0, -0.16065), quat=(0, 1, 0, 0), size=(0.004,), rgba=(0, 0, 0, 0))
                self.attach(hand, attach_site=mount)
                self.hands[side] = hand
                prefix = "lh_" if side == "left" else "rh_"
                # Teaching approximation for the two joints driven by one FF tendon.
                if not policy_hands:
                    hand.mjcf_model.equality.add("joint", name=prefix + "index_coupling", joint1=hand.mjcf_model.find("joint", prefix + "FFJ1"), joint2=hand.mjcf_model.find("joint", prefix + "FFJ2"), polycoef=(0, 1, 0, 0, 0))
                # Park unused fingers so they do not strike neighboring keys.
                for joint in hand.joints:
                    if not policy_hands and any(name in joint.name for name in ("MFJ", "RFJ", "LFJ")) and joint.name.endswith(("3", "2", "1")):
                        self.hand_seed.append((joint, 0.7 if joint.name.endswith("1") else 1.0))
                for actuator in hand.mjcf_model.find_all("actuator"):
                    if not policy_hands and "A_FFJ" in actuator.name:
                        gain = 4.0 if actuator.name.endswith("0") else 8.0
                        actuator.kp = gain
                        actuator.kv = 0.15
                self._mjcf_root.contact.add("exclude", body1=self._mjcf_root.find("body", side + "_arm_link5"), body2=wrist)
                wrist.add("camera", name=side + "_wrist_rgb", pos=(0, 0.055, -0.01), quat=(1, 0, 0, 0), fovy=75)
            self._mjcf_root.worldbody.add("camera", name="head_rgb", pos=(0.10, 0, 1.48), xyaxes=(0, -1, 0, 0.862, 0, 0.507), fovy=75)
            style_hands(self.hands.values())

        @property
        def mjcf_model(self):
            return self._mjcf_root

    class PianoTask(base.PianoOnlyTask):
        def __init__(self):
            super().__init__(arena=stage.Stage(), change_color_on_activation=True, control_timestep=0.02, physics_timestep=0.002)
            self.robot = R1Pro()
            self.arena.attach(self.robot)
            self.piano.root_body.pos = (0.47, piano_y if piano_y is not None else (-0.13 if native_grippers else -0.03), 0.90)
            self.piano.root_body.quat = (0, 0, 0, 1)
            if not native_grippers and not policy_hands:
                self.release_on_activation = True
                for joint in self.piano.joints:
                    joint.damping = 0.15

        def action_spec(self, physics):
            ranges = physics.model.actuator_ctrlrange
            return specs.BoundedArray((physics.model.nu,), np.float64, ranges[:, 0], ranges[:, 1])

        def before_step(self, physics, action, random_state):
            del random_state
            physics.data.ctrl[:] = action

    task = PianoTask()
    root = task.root_entity.mjcf_model
    root.option.integrator = "implicitfast"
    getattr(root.visual, "global").offwidth = 960
    getattr(root.visual, "global").offheight = 720
    center_y = float(task.piano.root_body.pos[1])
    for y in (center_y - 0.53, center_y + 0.53):
        root.worldbody.add("geom", name="piano_leg_" + str(y), type="box", pos=(0.47, y, 0.45), size=(0.025, 0.025, 0.45), rgba=(0.13, 0.15, 0.18, 1), contype=0, conaffinity=0)
    root.worldbody.add("camera", name="whole_robot", pos=(2.15, -2.50, 1.95), xyaxes=(0.758, 0.652, 0, -0.236, 0.274, 0.933), fovy=42)
    if native_grippers:
        for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
            camera = task.robot.mjcf_model.find("camera", name)
            camera.mode = "targetbody"
            camera.target = task.piano.root_body
            if name != "head_rgb":
                camera.pos = (0.08, 0.07, -0.17)
    style_stage(root)
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--custom-score", action="store_true", help="Use a constrained imported MIDI excerpt instead of the Twinkle exercise")
    parser.add_argument("--keyboard-y", type=float, help="Calibrated lateral piano position in meters; does not change robot joint limits")
    args = parser.parse_args()
    run(args, task_builder=lambda model: build_task(model, piano_y=args.keyboard_y), robot_info={"mode": "Galaxea_R1Pro_two_attached_Shadow_Hands_IK_baseline", "base": "fixed_chassis", "robot_source_commit": COMMIT, "g05_calls_during_execution": 0, "native_g05_hand_interface_compatible": False, "index_distal_equal_angle_coupling": True, "piano_key_damping": 0.15}, robot_license="Galaxea-R1Pro")
