"""Execute a validated note plan using Shadow Hand actuators and physical keys."""

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

from piano_context import validate_plan


def smooth(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def position_targets(model, ik_data):
    import mujoco
    import numpy as np

    # Mink updates kinematics, but actuator transmissions also need refreshing.
    mujoco.mj_fwdPosition(model, ik_data)
    return np.clip(ik_data.actuator_length.copy(), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def target_at(time, notes, key_positions, beat_seconds):
    """Reserve approach/release time; repeated notes must leave the key."""
    import numpy as np

    hover, pressed = 0.080, 0.008
    point = np.array([0.040, key_positions[notes[0]["midi"]], hover])
    if time < 1.0:
        return point
    time -= 1.0
    for i, note in enumerate(notes):
        start = note["start_beat"] * beat_seconds
        end = start + note["duration_beats"] * beat_seconds
        if time < end:
            point[1] = key_positions[note["midi"]]
            elapsed = time - start
            if elapsed < 0.50:
                previous = notes[max(0, i - 1)]["midi"]
                point[1] = key_positions[previous] + smooth(elapsed / 0.35) * (
                    point[1] - key_positions[previous]
                )
            elif time < end - 0.18:
                point[2] = hover + smooth((elapsed - 0.50) / 0.12) * (pressed - hover)
            else:
                point[2] = pressed + smooth((time - (end - 0.18)) / 0.18) * (hover - pressed)
            return point
    point[1] = key_positions[notes[-1]["midi"]]
    return point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    task_contract = json.loads(Path(__file__).with_name("task.json").read_text(encoding="utf-8"))
    validate_plan(plan, task_contract)
    args.out.mkdir(parents=True, exist_ok=False)

    import imageio.v2 as imageio
    import mink
    import mujoco
    import numpy as np
    import robopianist
    from dm_control import mjcf
    from dm_env import specs
    from mujoco_utils import composer_utils
    from robopianist.music import midi_message, synthesizer
    from robopianist.models.arenas import stage
    from robopianist.suite.tasks import base

    class ContactTask(base.PianoTask):
        def action_spec(self, physics):
            ranges = physics.model.actuator_ctrlrange
            return specs.BoundedArray((physics.model.nu,), np.float64, ranges[:, 0], ranges[:, 1])

        def before_step(self, physics, action, random_state):
            del random_state
            physics.data.ctrl[:] = action

    task = ContactTask(
        arena=stage.Stage(), gravity_compensation=True,
        change_color_on_activation=True, control_timestep=0.02, physics_timestep=0.002,
        forearm_dofs=("forearm_tx", "forearm_ty", "forearm_tz"),
    )
    task.right_hand.root_body.pos = (0.46, 0.15, 0.13)
    task.left_hand.root_body.pos = (0.46, -1.4, 0.23)
    visual_global = getattr(task.root_entity.mjcf_model.visual, "global")
    visual_global.offwidth = 960
    visual_global.offheight = 480
    env = composer_utils.Environment(task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False)
    writer = None
    try:
        env.reset()
        physics = env.physics
        model = physics.model.ptr
        mjcf.export_with_assets(task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        package = Path(robopianist.__file__).parent
        shutil.copyfile(package / "models/hands/third_party/shadow_hand/LICENSE", args.out / "scene/LICENSE-Shadow-Hand.txt")
        (args.out / "scene/NOTICE.txt").write_text(
            "Generated from RoboPianist 1.0.10 (Apache-2.0), Google Research.\n"
            "Shadow Hand E3M5 assets: Shadow Robot Company / MuJoCo Menagerie (Apache-2.0).\n"
            "Teaching scene modifications: mount positions, added forearm_tz, timesteps.\n",
            encoding="utf-8",
        )
        initial_qpos = physics.data.qpos.copy()
        initial_qvel = physics.data.qvel.copy()
        key_joints = [j.full_identifier for j in task.piano.joints]
        (args.out / "scene_manifest.json").write_text(json.dumps({
            "scene": "scene/scene.xml", "mujoco_version": mujoco.__version__,
            "key_joints": key_joints, "minimum_midi": 21,
            "activation_tolerance_radians": 0.00872665,
            "physics_steps_per_control": 10, "control_timestep_seconds": 0.02,
            "camera": "piano/closeup",
        }, indent=2) + "\n", encoding="utf-8")
        configuration = mink.Configuration(model)
        configuration.update(physics.data.qpos.copy())
        site = task.right_hand.fingertip_sites[1]
        frame = mink.FrameTask(site.full_identifier, "site", position_cost=100.0, orientation_cost=0.0, lm_damping=0.001)
        weights = np.full(model.nv, 10.0)
        for joint in task.right_hand.joints:
            if joint.name in ("rh_FFJ3", "rh_FFJ2", "rh_FFJ1") or joint.name.startswith("forearm_"):
                weights[int(physics.bind(joint).dofadr)] = 0.001
        posture = mink.PostureTask(model, cost=weights)
        posture.set_target_from_configuration(configuration)
        positions = {n["midi"]: float(physics.bind(task.piano.keys[n["midi"] - 21]).xpos[1]) for n in plan["notes"]}
        beat_seconds = 60.0 / plan["tempo_bpm"]
        duration = 1.0 + max(n["start_beat"] + n["duration_beats"] for n in plan["notes"]) * beat_seconds + 0.5
        previous = np.zeros(88, dtype=bool)
        key_geoms = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.full_identifier)
            for geom in task.piano.mjcf_model.find_all("geom")
            if geom.name and geom.name.startswith(("white_key_geom_", "black_key_geom_"))
        }
        hand_geoms = {
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rh_shadow_hand/")
        }
        onsets, releases, frames, controls, joint_states = [], [], [], [], []
        contact_frames = 0
        if not args.no_video:
            writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=25, codec="libx264", macro_block_size=1)
        for step in range(round(duration / 0.02)):
            time = step * 0.02
            target = target_at(time, plan["notes"], positions, beat_seconds)
            frame.set_target(mink.SE3.from_translation(target))
            for _ in range(5):
                velocity = mink.solve_ik(configuration, [frame, posture], 0.02, "daqp", damping=1e-5)
                configuration.integrate_inplace(velocity, 0.02)
            # Map joint AND coupled-tendon position transmissions to actuator targets.
            # configuration.data is only an IK copy, never the physical piano state.
            control = position_targets(model, configuration.data)
            env.step(control)
            if not np.isfinite(physics.data.qpos).all():
                raise ValueError("Non-finite physical state")
            active = task.piano.activation.copy()
            for key in np.flatnonzero(active & ~previous):
                onsets.append({"time_seconds": round(float(physics.data.time), 4), "midi": int(key + 21)})
            for key in np.flatnonzero(previous & ~active):
                releases.append({"time_seconds": round(float(physics.data.time), 4), "midi": int(key + 21)})
            previous = active
            hand_key_contact = any(
                (contact.geom1 in key_geoms and contact.geom2 in hand_geoms)
                or (contact.geom2 in key_geoms and contact.geom1 in hand_geoms)
                for contact in physics.data.contact
            )
            contact_frames += int(hand_key_contact)
            frames.append({"time_seconds": round(float(physics.data.time), 4), "active_midi": (np.flatnonzero(active) + 21).tolist(), "target": target.tolist(), "fingertip": physics.bind(site).xpos.tolist(), "contacts": int(physics.data.ncon)})
            controls.append(control)
            joint_states.append(physics.data.qpos.copy())
            if writer is not None and step % 2 == 0:
                image = physics.render(height=480, width=960, camera_id="piano/closeup")
                writer.append_data(image)
                if step in (0, 86, 186, 286):
                    imageio.imwrite(args.out / f"frame-{step:04d}.png", image)
        if writer is not None:
            writer.close()
            writer = None
        events = task.piano.midi_module.get_all_midi_messages()
        midi_events = [
            {"type": type(event).__name__, "time_seconds": float(event.time), "midi": int(event.note)}
            for event in events if isinstance(event, (midi_message.NoteOn, midi_message.NoteOff))
        ]
        (args.out / "measured_midi_events.json").write_text(json.dumps(midi_events, indent=2) + "\n", encoding="utf-8")
        # Render audio ONLY from measured key events, never from the reference plan.
        if any(isinstance(e, midi_message.NoteOn) for e in events) and not args.no_video:
            synth = synthesizer.Synthesizer()
            try:
                waveform = synth.get_samples(events)
            finally:
                synth.stop()
            with wave.open(str(args.out / "measured.wav"), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(44100)
                output.writeframes(waveform.tobytes())
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(args.out / "silent.mp4"), "-i", str(args.out / "measured.wav"), "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out / "demo.mp4")], check=True)
        expected = [n["midi"] for n in plan["notes"]]
        actual = [n["midi"] for n in onsets]
        report = {
            "mode": "note_plan_plus_local_ik_contact_control",
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "expected_onsets": expected,
            "measured_onsets": onsets,
            "measured_releases": releases,
            "onset_sequence_matches": actual == expected,
            "audio_onset_sequence_matches": [e["midi"] for e in midi_events if e["type"] == "NoteOn"] == expected,
            "hand_key_contact_frames": contact_frames,
            "control_timestep_seconds": 0.02,
            "duration_seconds": duration,
            "gpt_calls_during_execution": 0,
            "key_state_teleported": False,
            "joint_state_teleported_during_execution": False,
            "versions": {"mujoco": mujoco.__version__, "robopianist": robopianist.__version__},
            "timing_note": "1s warmup; each beat reserves 0.5s approach/settle and 0.18s release. Nominal duration is a scheduling window, not the measured key-down duration.",
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.out / "measurements.json").write_text(json.dumps(frames) + "\n", encoding="utf-8")
        np.savez_compressed(args.out / "trajectory.npz", controls=controls, qpos=joint_states, initial_qpos=initial_qpos, initial_qvel=initial_qvel)
        print(json.dumps(report, indent=2), flush=True)
        if not report["onset_sequence_matches"] or not report["audio_onset_sequence_matches"]:
            raise SystemExit("Physical note sequence differs from the plan; inspect measurements")
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
