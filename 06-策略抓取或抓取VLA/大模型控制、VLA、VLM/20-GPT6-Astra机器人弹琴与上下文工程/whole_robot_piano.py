"""A fixed-base full G1 body with two physically attached Shadow Hands at a piano."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

from physical_piano import position_targets, target_at
from piano_context import validate_plan
from piano_score import ContactRelease, compare_timing, rest_lift, validate_contact_score
from download_g1 import COMMIT
from replay_windows import compare_parts


def validate_bimanual_plan(plan):
    validate_plan(plan)
    task = json.loads(Path(__file__).with_name("whole_robot_task.json").read_text(encoding="utf-8"))
    if plan["tempo_bpm"] != task["tempo_bpm"]:
        raise ValueError("Expected the task's tempo")
    for side in ("left", "right"):
        actual = [{k: v for k, v in n.items() if k != "hand"} for n in plan["notes"] if n["hand"] == side]
        if actual != task[side]:
            raise ValueError("Plan differs from the supplied " + side + " part")


def build_task(model_dir):
    import numpy as np
    from dm_control import composer, mjcf
    from dm_env import specs
    from robopianist.models.arenas import stage
    from robopianist.models.hands import HandSide, shadow_hand
    from robopianist.suite.tasks import base

    class G1(composer.Entity):
        def _build(self):
            source = json.loads((model_dir / "source_manifest.json").read_text(encoding="utf-8"))
            if source["commit"] != COMMIT:
                raise ValueError("Use the pinned Menagerie model")
            for name, record in source["files"].items():
                if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Changed model resource: " + name)
            xml = ET.fromstring((model_dir / "g1.xml").read_bytes())
            # Fixing the pelvis is a simulation constraint, not a balance policy.
            pelvis = xml.find("./worldbody/body[@name='pelvis']")
            pelvis.remove(pelvis.find("freejoint"))
            for keyframe in xml.findall("keyframe"):
                xml.remove(keyframe)
            for parent in xml.iter():
                for geom in list(parent.findall("geom")):
                    if geom.get("mesh", "").endswith("rubber_hand"):
                        parent.remove(geom)
            self._mjcf_root = mjcf.from_xml_string(ET.tostring(xml, encoding="unicode"), model_dir=str(model_dir))
            for body in self._mjcf_root.find_all("body"):
                body.gravcomp = 1
            self.hands = {}
            for side, hand_side in (("left", HandSide.LEFT), ("right", HandSide.RIGHT)):
                hand = shadow_hand.ShadowHand(side=hand_side, forearm_dofs=())
                root = hand.root_body
                root.pos = (0, 0, 0)
                root.quat = (1, 0, 0, 0)
                # G1 already supplies the forearm; remove Shadow's separate forearm housing.
                for geom in list(root.geom):
                    geom.remove()
                root.inertial.mass = 0.001
                root.inertial.pos = (0, 0, 0)
                root.inertial.diaginertia = (1e-6, 1e-6, 1e-6)
                prefix = "lh_" if side == "left" else "rh_"
                hand.mjcf_model.find("body", prefix + "wrist").pos = (0, 0, 0)
                for body in hand.mjcf_model.find_all("body"):
                    body.gravcomp = 1
                wrist = self._mjcf_root.find("body", side + "_wrist_yaw_link")
                mount = wrist.add("site", name=side + "_shadow_mount", pos=(0.055, 0, 0), quat=(0.5, 0.5, 0.5, 0.5), size=(0.004,), rgba=(0, 0, 0, 0))
                self.attach(hand, attach_site=mount)
                self.hands[side] = hand

        @property
        def mjcf_model(self):
            return self._mjcf_root

    class WholeTask(base.PianoOnlyTask):
        def __init__(self):
            super().__init__(arena=stage.Stage(), change_color_on_activation=True, control_timestep=0.02, physics_timestep=0.002)
            self.robot = G1()
            self.arena.attach(self.robot)
            self.piano.root_body.pos = (0.47, -0.13, 0.90)
            self.piano.root_body.quat = (0, 0, 0, 1)

        def action_spec(self, physics):
            ranges = physics.model.actuator_ctrlrange
            return specs.BoundedArray((physics.model.nu,), np.float64, ranges[:, 0], ranges[:, 1])

        def before_step(self, physics, action, random_state):
            del random_state
            physics.data.ctrl[:] = action

    task = WholeTask()
    root = task.root_entity.mjcf_model
    getattr(root.visual, "global").offwidth = 960
    getattr(root.visual, "global").offheight = 720
    for y in (-0.66, 0.40):
        root.worldbody.add("geom", name="piano_leg_" + str(y), type="box", pos=(0.47, y, 0.45), size=(0.025, 0.025, 0.45), rgba=(0.13, 0.15, 0.18, 1), contype=0, conaffinity=0)
    root.worldbody.add("camera", name="whole_robot", pos=(1.95, -2.30, 1.65), xyaxes=(0.762, 0.647, 0, -0.188, 0.221, 0.957), fovy=40)
    return task


def run(args, task_builder=build_task, robot_info=None, robot_license="Unitree-G1"):
    import imageio.v2 as imageio
    import mink
    import mujoco
    import numpy as np
    import robopianist
    from dm_control import mjcf
    from mujoco_utils import composer_utils
    from robopianist.music import midi_message, synthesizer

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    custom_score = getattr(args, "custom_score", False)
    if custom_score:
        validate_contact_score(plan)
    else:
        validate_bimanual_plan(plan)
    beat_seconds = 60.0 / plan["tempo_bpm"]
    duration = 1.0 + max(n["start_beat"] + n["duration_beats"] for n in plan["notes"]) * beat_seconds + 0.5
    args.out.mkdir(parents=True, exist_ok=False)
    task = task_builder(args.model)
    env = composer_utils.Environment(task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False)
    writer = None
    try:
        env.reset()
        p, model = env.physics, env.physics.model.ptr
        configuration = mink.Configuration(model)
        seed = p.data.qpos.copy()
        if hasattr(task.robot, "seed"):
            for name, value in task.robot.seed.items():
                seed[int(p.bind(task.robot.mjcf_model.find("joint", name)).qposadr)] = value
        else:
            for side in ("left", "right"):
                for suffix, value in (("shoulder_pitch_joint", -0.8), ("shoulder_roll_joint", 0.25 if side == "left" else -0.25), ("elbow_joint", 1.0)):
                    joint = task.robot.mjcf_model.find("joint", side + "_" + suffix)
                    seed[int(p.bind(joint).qposadr)] = value
        for joint, value in getattr(task.robot, "hand_seed", []):
            seed[int(p.bind(joint).qposadr)] = value
        configuration.update(seed)
        parts = {side: [n for n in plan["notes"] if n["hand"] == side] for side in ("left", "right")}
        active_parts = {side: notes for side, notes in parts.items() if notes}
        contact_release = ContactRelease(active_parts, plan["tempo_bpm"]) if getattr(task, "release_on_activation", False) else None
        positions = {n["midi"]: float(p.bind(task.piano.keys[n["midi"] - 21]).xpos[1]) for n in plan["notes"]}
        weights = np.full(model.nv, 100.0)
        arm_joints = {}
        for side in active_parts:
            arm_joints[side] = task.robot.arm_joints[side] if hasattr(task.robot, "arm_joints") else [j for j in task.robot.mjcf_model.find_all("joint") if j.name.startswith(side + "_") and any(s in j.name for s in ("shoulder", "elbow", "wrist"))]
            for joint in arm_joints[side]:
                weights[int(p.bind(joint).dofadr)] = 0.001 if custom_score else 0.02
            for joint in task.robot.hands[side].joints:
                if "FFJ" in joint.name and not joint.name.endswith("4"):
                    weights[int(p.bind(joint).dofadr)] = getattr(task, "index_joint_cost", 0.001)
        posture = mink.PostureTask(model, weights)
        posture.set_target_from_configuration(configuration)
        tips, palms = {}, {}
        for side in active_parts:
            hand = task.robot.hands[side]
            tips[side] = mink.FrameTask(hand.fingertip_sites[1].full_identifier, "site", position_cost=100, orientation_cost=0, lm_damping=0.001)
            palm_name = ("lh_" if side == "left" else "rh_") + "palm"
            palms[side] = mink.FrameTask(hand.mjcf_model.find("body", palm_name).full_identifier, "body", position_cost=1, orientation_cost=1 if custom_score else 10, lm_damping=0.001)
        tasks = [posture] + list(tips.values()) + list(palms.values())
        if model.neq:
            tasks.append(mink.EqualityConstraintTask(model, cost=1000.0))
        torso_geoms, arm_geoms = [], []
        collision_hands = {"left": [], "right": []}
        for geom in range(model.ngeom):
            if not model.geom_contype[geom] and not model.geom_conaffinity[geom]:
                continue
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
            for side, prefix in (("left", "lh_"), ("right", "rh_")):
                if "/" + prefix + "shadow_hand/" in body_name:
                    collision_hands[side].append(geom)
            if body_name.endswith(("/torso_link", "/torso_link4")):
                torso_geoms.append(geom)
            elif "shadow_hand/" in body_name or any(body_name.endswith("/" + side + "_" + suffix) for side in parts for suffix in ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "arm_link3", "arm_link4", "arm_link5", "arm_link6", "arm_link7")):
                arm_geoms.append(geom)
        collision_pairs = [(torso_geoms, arm_geoms)]
        if custom_score and all(collision_hands.values()):
            collision_pairs.append((collision_hands["left"], collision_hands["right"]))
        limits = [mink.ConfigurationLimit(model), mink.CollisionAvoidanceLimit(model, collision_pairs, minimum_distance_from_collisions=0.003, collision_detection_distance=0.03)]
        target_positions = {}

        def set_targets(time):
            for side in active_parts:
                local = target_at(time, parts[side], positions, beat_seconds)
                if contact_release is not None:
                    local[2] = contact_release.height(side, time, local[2])
                if hasattr(task, "fingertip_clearance"):
                    hover, pressed = task.fingertip_clearance
                    local[2] = pressed + (local[2] - 0.008) * (hover - pressed) / (0.080 - 0.008)
                target = np.array([0.47 - local[0], local[1], 0.90 + local[2]])
                lift = rest_lift(time, parts[side], plan["tempo_bpm"]) if custom_score else 0.0
                if custom_score and contact_release is not None:
                    lift = max(lift, contact_release.tail_lift(side, time))
                if custom_score:
                    target[1] += (0.12 if side == "left" else -0.12) * lift / 0.22
                target[2] += lift
                target_positions[side] = target.tolist()
                tips[side].set_target(mink.SE3.from_translation(target))
                palm_position = np.array([0.245, target[1] + (0.033 if side == "right" else -0.033), 0.98 + lift])
                palms[side].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), palm_position))

        set_targets(0)
        for _ in range(120):
            velocity = mink.solve_ik(configuration, tasks, 0.02, "daqp", damping=1e-5, limits=limits)
            configuration.integrate_inplace(velocity, 0.02)
        # Initialize once at the solved hover pose; all subsequent motion uses actuators.
        p.data.qpos[:] = configuration.q
        p.data.qvel[:] = 0
        p.forward()
        initial_qpos, initial_qvel = p.data.qpos.copy(), p.data.qvel.copy()
        mjcf.export_with_assets(task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        package = Path(robopianist.__file__).parent
        shutil.copyfile(package / "models/hands/third_party/shadow_hand/LICENSE", args.out / "scene/LICENSE-Shadow-Hand.txt")
        shutil.copyfile(args.model / "LICENSE", args.out / ("scene/LICENSE-" + robot_license + ".txt"))
        if (args.model / "NOTICE").exists():
            shutil.copyfile(args.model / "NOTICE", args.out / "scene/NOTICE-Galaxea.txt")
        (args.out / "scene/NOTICE.txt").write_text(("G1 source: MuJoCo Menagerie, commit " + COMMIT + " (BSD-3-Clause).\n" if robot_info is None else "R1 Pro source: GalaxeaManipSim, commit " + robot_info["robot_source_commit"] + " (see LICENSE and NOTICE).\n") + "Shadow Hand/RoboPianist 1.0.10 (Apache-2.0).\nTeaching adaptation: fixed base, wrist-mounted hands, removed duplicate forearms, ideal gravity compensation. Not a stock hardware configuration or balance controller.\n", encoding="utf-8")
        (args.out / "scene_manifest.json").write_text(json.dumps({"scene": "scene/scene.xml", "mujoco_version": mujoco.__version__, "key_joints": [j.full_identifier for j in task.piano.joints], "minimum_midi": 21, "activation_tolerance_radians": 0.00872665, "physics_steps_per_control": 10, "control_timestep_seconds": 0.02, "camera": "whole_robot"}, indent=2) + "\n", encoding="utf-8")
        if not args.no_video:
            writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=25, codec="libx264", macro_block_size=1)
        previous = np.zeros(88, dtype=bool)
        onsets, frames, controls, qpos = [], [], [], []
        key_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g.full_identifier) for g in task.piano.mjcf_model.find_all("geom") if g.name and g.name.startswith(("white_key_geom_", "black_key_geom_"))}
        key_geom_by_midi = {int(g.name.rsplit("_", 1)[1]) + 21: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g.full_identifier) for g in task.piano.mjcf_model.find_all("geom") if g.name and g.name.startswith(("white_key_geom_", "black_key_geom_"))}
        hand_geoms = {s: {i for i in range(model.ngeom) if ("/" + ("lh_" if s == "left" else "rh_") + "shadow_hand/") in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")} for s in parts}
        contact_frames = {s: 0 for s in parts}
        interhand_contact_frames = 0
        for step in range(math.ceil(duration / 0.02)):
            set_targets(step * 0.02)
            if custom_score:
                configuration.update(p.data.qpos.copy())
            for _ in range(5):
                velocity = mink.solve_ik(configuration, tasks, 0.02, "daqp", damping=1e-5, limits=limits)
                configuration.integrate_inplace(velocity, 0.02)
            control = position_targets(model, configuration.data)
            env.step(control)
            if not np.isfinite(p.data.qpos).all() or not np.isfinite(p.data.qvel).all():
                raise ValueError("Non-finite physical state")
            hand_contacts = {s: any((c.geom1 in key_geoms and c.geom2 in hand_geoms[s]) or (c.geom2 in key_geoms and c.geom1 in hand_geoms[s]) for c in p.data.contact) for s in parts}
            for side in parts:
                contact_frames[side] += int(hand_contacts[side])
            interhand_contact_frames += int(any((c.geom1 in hand_geoms["left"] and c.geom2 in hand_geoms["right"]) or (c.geom2 in hand_geoms["left"] and c.geom1 in hand_geoms["right"]) for c in p.data.contact))
            active = task.piano.activation.copy()
            if contact_release is not None:
                contact_release.observe(float(p.data.time), set((np.flatnonzero(active) + 21).tolist()))
            for key in np.flatnonzero(active & ~previous):
                midi = int(key + 21)
                geom = key_geom_by_midi[midi]
                owners = [side for side in parts if any((c.geom1 == geom and c.geom2 in hand_geoms[side]) or (c.geom2 == geom and c.geom1 in hand_geoms[side]) for c in p.data.contact)]
                onsets.append({"midi": midi, "time_seconds": round(float(p.data.time), 4), "contact_hands": owners})
            previous = active
            frames.append({"time_seconds": float(p.data.time), "active_midi": (np.flatnonzero(active) + 21).tolist(), "fingertips": {s: p.bind(task.robot.hands[s].fingertip_sites[1]).xpos.tolist() for s in parts}, "hand_key_contacts": hand_contacts, "contacts": int(p.data.ncon)})
            frames[-1]["target_fingertips"] = {side: list(target) for side, target in target_positions.items()}
            frames[-1]["IK_fingertips"] = {side: configuration.data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, task.robot.hands[side].fingertip_sites[1].full_identifier)].tolist() for side in active_parts}
            controls.append(control)
            qpos.append(p.data.qpos.copy())
            if writer is not None and step % 2 == 0:
                pixels = p.render(height=720, width=960, camera_id="whole_robot")
                writer.append_data(pixels)
                if step in (0, 86, 186, 286):
                    imageio.imwrite(args.out / f"frame-{step:04d}.png", pixels)
        if writer is not None:
            writer.close()
            writer = None
        events = task.piano.midi_module.get_all_midi_messages()
        midi_events = [{"type": type(e).__name__, "midi": int(e.note), "time_seconds": float(e.time)} for e in events if isinstance(e, (midi_message.NoteOn, midi_message.NoteOff))]
        (args.out / "measured_midi_events.json").write_text(json.dumps(midi_events, indent=2), encoding="utf-8")
        expected = {s: [n["midi"] for n in parts[s]] for s in parts}
        by_part, unexpected, matches = compare_parts(onsets, expected)
        _, audio_unexpected, audio_matches = compare_parts([e for e in midi_events if e["type"] == "NoteOn"], expected)
        timing_matches = all(compare_timing([event for event in onsets if event["midi"] in expected[side]], parts[side], plan["tempo_bpm"]) for side in parts)
        audio_timing_matches = all(compare_timing([event for event in midi_events if event["type"] == "NoteOn" and event["midi"] in expected[side]], parts[side], plan["tempo_bpm"]) for side in parts)
        states = np.asarray(qpos)
        arm_ranges = {s: {j.name: float(np.ptp(states[:, int(p.bind(j).qposadr)])) for j in arm_joints[s]} for s in active_parts}
        report = {"mode": "whole_G1_two_attached_Shadow_Hands", "expected_parts": expected, "measured_parts": by_part, "measured_onsets": onsets, "expected_onsets": [n["midi"] for n in plan["notes"]], "parts_match": matches, "audio_parts_match": audio_matches, "unexpected_onsets": unexpected, "audio_unexpected_onsets": audio_unexpected, "hand_key_contact_frames": contact_frames, "arm_joint_excursion_radians": arm_ranges, "base": "fixed_pelvis", "visible_robot_support": False, "gravity_compensation": "ideal", "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "gpt_calls_during_execution": 0, "key_state_teleported": False, "joint_state_teleported_during_execution": False, "duration_seconds": 9.5, "timing_note": "1s warmup; 0.5s approach and 0.18s release within each scheduled note window. Measured holds are shorter than nominal score durations.", "versions": {"mujoco": mujoco.__version__, "robopianist": robopianist.__version__}}
        if robot_info is not None:
            report.update(robot_info)
        report.update({"duration_seconds": float(p.data.time), "score_mode": "imported_excerpt" if custom_score else "verified_twinkle_exercise", "timing_windows_match": timing_matches, "audio_timing_windows_match": audio_timing_matches, "full_MIDI_sustain_and_velocity_reproduced": False, "active_hands": list(active_parts)})
        report["release_on_measured_activation"] = contact_release is not None
        report["idle_hand_lift_and_bimanual_collision_limit"] = custom_score
        report["IK_feedback_uses_measured_joints"] = custom_score
        report["interhand_contact_frames"] = interhand_contact_frames
        report["piano_position_meters"] = task.piano.root_body.pos.tolist()
        expected_hand = {note["midi"]: note["hand"] for note in plan["notes"]}
        report["onset_contact_hand_matches"] = all(event["contact_hands"] == [expected_hand.get(event["midi"])] for event in onsets)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.out / "measurements.json").write_text(json.dumps(frames), encoding="utf-8")
        np.savez_compressed(args.out / "trajectory.npz", controls=controls, qpos=qpos, initial_qpos=initial_qpos, initial_qvel=initial_qvel)
        if not args.no_video and any(e["type"] == "NoteOn" for e in midi_events):
            synth = synthesizer.Synthesizer()
            try:
                samples = synth.get_samples(events)
            finally:
                synth.stop()
            with wave.open(str(args.out / "measured.wav"), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(44100)
                output.writeframes(samples.tobytes())
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(args.out / "silent.mp4"), "-i", str(args.out / "measured.wav"), "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out / "demo.mp4")], check=True)
        print(json.dumps(report, indent=2), flush=True)
        if not matches or not audio_matches or (custom_score and not (timing_matches and audio_timing_matches)) or not all(contact_frames[s] for s in active_parts) or not all(any(v > 0.01 for k, v in arm_ranges[s].items() if "shoulder" in k or "elbow" in k or "arm_joint" in k) for s in active_parts):
            raise SystemExit("Physical score validation failed; inspect report and measurements")
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--custom-score", action="store_true", help="Use a constrained imported MIDI excerpt instead of the Twinkle exercise")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
