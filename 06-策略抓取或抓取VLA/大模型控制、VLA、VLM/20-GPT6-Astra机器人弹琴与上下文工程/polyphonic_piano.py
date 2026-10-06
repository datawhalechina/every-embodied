"""Finger-assigned sustained polyphony on a fixed-base Galaxea R1 Pro."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

from galaxea_piano import build_task
from physical_piano import position_targets
from piano_context import write_json
from polyphonic_score import FINGERS, evaluate, finger_target, performance_passed, validate_score


HOME = {"right": {"thumb": 60, "index": 62, "middle": 64, "ring": 65, "little": 67},
        "left": {"thumb": 55, "index": 53, "middle": 52, "ring": 50, "little": 48}}


def contact_patch_point(local, rotation):
    """Return a body-local point in the bottom 1 mm of the collision mesh."""
    import numpy as np
    local, rotation = np.asarray(local), np.asarray(rotation)
    if (local.ndim != 2 or local.shape[1] != 3 or not len(local)
            or rotation.shape != (3, 3) or not np.isfinite(local).all()
            or not np.isfinite(rotation).all()):
        raise ValueError("Expected finite collision vertices and a 3x3 rotation")
    height = (local @ rotation.T)[:, 2]
    return local[height <= height.min() + .001].mean(axis=0)


def bounded_ik_limits(low, high, max_step):
    """Intersect joint-limit deltas with a local nonlinear IK trust region."""
    import numpy as np
    if not math.isfinite(max_step) or not 0 < max_step <= .2:
        raise ValueError("IK step must be finite and within (0, 0.2] radians")
    low, high = np.maximum(low, -max_step), np.minimum(high, max_step)
    if np.any(low > high):
        raise ValueError("Current configuration is outside the bounded IK region")
    return low, high


def key_press_offset(previous, travel, gain, dt=.02, limit=.004):
    """Bounded integral correction of a fingertip target, never of a piano key."""
    if (not all(math.isfinite(v) for v in (previous, travel, gain, dt, limit))
            or not 0 <= gain <= .1 or dt <= 0 or limit <= 0):
        raise ValueError("Invalid key-press feedback inputs")
    if gain == 0:
        return 0.
    return min(limit, max(0., previous + gain * dt * (.95-travel)))


def polyphonic_task(model_dir, little_abduction_limit=None):
    task = build_task(model_dir, piano_y=0)
    task.robot.hand_seed.clear()
    task.pad_sites = {}
    for side, hand in task.robot.hands.items():
        task.pad_sites[side] = [site.parent.add("site", name="pad_control_" + finger,
            pos=(0, 0, 0.0195 if finger == "thumb" else 0.0176), size=(0.003,), group=4)
            for finger, site in zip(FINGERS, hand.fingertip_sites)]
        prefix = "lh_" if side == "left" else "rh_"
        if little_abduction_limit is not None:
            joint = hand.mjcf_model.find("joint", prefix + "LFJ4")
            joint.range = (-little_abduction_limit, little_abduction_limit)
        for finger in ("MF", "RF", "LF"):
            hand.mjcf_model.equality.add("joint", name=prefix + finger + "_coupling",
                joint1=hand.mjcf_model.find("joint", prefix + finger + "J1"),
                joint2=hand.mjcf_model.find("joint", prefix + finger + "J2"), polycoef=(0, 1, 0, 0, 0))
        for joint in hand.joints:
            if joint.name.endswith(("J1", "J2", "J3")) and "TH" not in joint.name and "WR" not in joint.name:
                task.robot.hand_seed.append((joint, 0.25))
        for actuator in hand.mjcf_model.find_all("actuator"):
            if "_A_WR" in actuator.name:
                actuator.kp = 120.0
                actuator.kv = 2.0
            else:
                actuator.kp = 8.0 if actuator.name.endswith("0") else 16.0
                actuator.kv = 0.3
    for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        camera = task.robot.mjcf_model.find("camera", name)
        camera.mode = "targetbody"
        camera.target = task.piano.root_body
        if name != "head_rgb":
            camera.pos = (0.08, 0.07, -0.17)
    return task


def run(args):
    if not math.isfinite(args.pressed_z) or not 0.90 <= args.pressed_z <= 0.93:
        raise ValueError("Pressed target must be finite and within the calibrated keyboard height")
    if not math.isfinite(args.g05_prior_weight) or not 0 < args.g05_prior_weight <= 1:
        raise ValueError("G0.5 prior weight must be finite and within (0, 1]")
    if not math.isfinite(args.hand_feedback_gain) or not 0 <= args.hand_feedback_gain <= 4:
        raise ValueError("Hand feedback gain must be within [0, 4]")
    if not math.isfinite(args.hover_z) or not .95 <= args.hover_z <= 1.02:
        raise ValueError("Hover height must be within [0.95, 1.02]")
    if args.video_fps not in (10, 25, 50):
        raise ValueError("Video FPS must divide the 50 Hz control clock")
    if args.ik_max_step is not None:
        bounded_ik_limits(0., 0., args.ik_max_step)
        if not args.fixed_torso:
            raise ValueError("Bounded IK currently requires --fixed-torso")
    if (args.song_active_priority or args.song_staged_motion) and not args.song_mode:
        raise ValueError("Song trajectory options require --song-mode")
    if args.max_hand_span is not None and (not math.isfinite(args.max_hand_span) or args.max_hand_span <= 0):
        raise ValueError("Hand span limit must be positive and finite")
    if not math.isfinite(args.song_release_lead) or not 0 <= args.song_release_lead <= .2:
        raise ValueError("Release lead must be finite and within [0, 0.2] seconds")
    key_press_offset(0., 0., args.key_press_feedback)
    if args.key_press_feedback and not args.song_mode:
        raise ValueError("Key-press feedback requires --song-mode")
    model_paths = (args.g05_python, args.g05_repo, args.g05_models, args.g05_checkpoint)
    if any(model_paths) and not all(model_paths):
        raise ValueError("Supply all four G0.5 paths, or none for the control-only baseline")
    import imageio.v2 as imageio
    import mink
    import mujoco
    import numpy as np
    import robopianist
    import qpsolvers
    from dm_control import mjcf
    from mujoco_utils import composer_utils
    from robopianist.music import midi_message, synthesizer

    score = json.loads(args.plan.read_text(encoding="utf-8"))
    validation = validate_score(score, song_mode=args.song_mode)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.little_abduction_limit is not None and not 0.05 <= args.little_abduction_limit <= 0.34:
        raise ValueError("Little finger abduction limit must be within [0.05, 0.34]")
    task = polyphonic_task(args.model, args.little_abduction_limit)
    env = composer_utils.Environment(task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False)
    policy, writer = None, None
    try:
        env.reset()
        p, model = env.physics, env.physics.model.ptr
        configuration = mink.Configuration(model)
        seed = p.data.qpos.copy()
        for name, value in task.robot.seed.items():
            seed[int(p.bind(task.robot.mjcf_model.find("joint", name)).qposadr)] = value
        for joint, value in task.robot.hand_seed:
            seed[int(p.bind(joint).qposadr)] = value
        configuration.update(seed)
        weights = np.full(model.nv, 100.0)
        for side in HOME:
            for joint in task.robot.arm_joints[side]:
                weights[int(p.bind(joint).dofadr)] = 0.01
            for joint in task.robot.hands[side].joints:
                weights[int(p.bind(joint).dofadr)] = 0.002 if "WR" not in joint.name else 0.02
        posture = mink.PostureTask(model, cost=weights)
        posture.set_target_from_configuration(configuration)
        tips, palms, parts = {}, {}, {}
        for side, hand in task.robot.hands.items():
            for i, finger in enumerate(FINGERS):
                key = (side, finger)
                tips[key] = mink.FrameTask(task.pad_sites[side][i].full_identifier, "site", position_cost=100, orientation_cost=0, lm_damping=0.001)
                parts[key] = [n for n in score["notes"] if (n["hand"], n["finger"]) == key]
            prefix = "lh_" if side == "left" else "rh_"
            palms[side] = mink.FrameTask(hand.mjcf_model.find("body", prefix + "palm").full_identifier, "body", position_cost=0.1, orientation_cost=2.0, lm_damping=0.001)
        tasks = [posture, *tips.values(), *palms.values(), mink.EqualityConstraintTask(model, cost=1000.0)]
        limits = [mink.ConfigurationLimit(model)]
        owned_joints = [j for side, hand in task.robot.hands.items()
                        for j in [*task.robot.arm_joints[side], *hand.joints]]
        owned_q = np.array([int(p.bind(j).qposadr) for j in owned_joints])
        owned_v = np.array([int(p.bind(j).dofadr) for j in owned_joints])
        owned_ranges = model.jnt_range[[int(p.bind(j).element_id) for j in owned_joints]]
        contact_sites = []
        if args.contact_surface:
            for side in HOME:
                for site in task.pad_sites[side]:
                    sid = int(p.bind(site).element_id)
                    body = int(model.site_bodyid[sid])
                    vertices = []
                    for geom in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]):
                        if not model.geom_contype[geom] or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                            continue
                        mesh = model.geom_dataid[geom]
                        local = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                        rotation = np.empty(9)
                        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                        vertices.append(local @ rotation.reshape(3, 3).T + model.geom_pos[geom])
                    if not vertices:
                        raise ValueError("No distal collision mesh for contact-surface control")
                    contact_sites.append((sid, body, np.concatenate(vertices)))

        def integrate_ik(initializing=False):
            for sid, body, local in contact_sites:
                rotation = configuration.data.xmat[body].reshape(3, 3)
                # Move the measurement site only, never geometry, keys or joints.
                model.site_pos[sid] = contact_patch_point(local, rotation)
            if contact_sites:
                configuration.update(configuration.q.copy())
            if not args.fixed_torso:
                velocity = mink.solve_ik(configuration, tasks, 0.02, "daqp", damping=1e-5, limits=limits)
                configuration.integrate_inplace(velocity, 0.02)
                return
            problem = mink.build_ik(configuration, tasks, .02, damping=1e-5, limits=[])
            from pianomime_galaxea import reduced_arm_problem
            hessian, gradient, low, high = reduced_arm_problem(
                problem, owned_v, owned_q, configuration.q, owned_ranges)
            if args.ik_max_step is not None and not initializing:
                try:
                    low, high = bounded_ik_limits(low, high, args.ik_max_step)
                except ValueError as error:
                    outside = np.flatnonzero((low > args.ik_max_step) | (high < -args.ik_max_step))
                    details = {owned_joints[i].full_identifier: dict(
                        qpos=float(configuration.q[owned_q[i]]), range=owned_ranges[i].tolist())
                        for i in outside}
                    raise ValueError(f"{error}; time={p.data.time:.3f}s; joints={details}") from error
            delta = qpsolvers.solve_qp(hessian, gradient, lb=low, ub=high, solver="daqp")
            if delta is None or not np.isfinite(delta).all():
                raise RuntimeError("Owned-joint IK failed")
            velocity = np.zeros(model.nv)
            velocity[owned_v] = delta / .02
            configuration.integrate_inplace(velocity, .02)
        key_y = {k: float(p.bind(task.piano.keys[k - 21]).xpos[1]) for k in range(21, 109)}
        from song_score import fingering_geometry
        geometry = fingering_geometry(score, key_y, args.max_hand_span)
        write_json(args.out / "fingering-geometry.json", geometry)
        if geometry["within_supplied_limit"] is False:
            raise ValueError("Held notes exceed --max-hand-span; inspect fingering-geometry.json before execution")
        beat = 60.0 / score["tempo_bpm"]
        targets = {}
        from song_score import SongTargets
        song_targets = SongTargets(score, key_y, args.hover_z, args.pressed_z,
                                   staged=args.song_staged_motion,
                                   release_lead=args.song_release_lead) if args.song_mode else None
        press_offsets = {key: 0. for key in tips}
        press_notes = {key: None for key in tips}

        def set_targets(time):
            planned = song_targets.targets(time) if song_targets else None
            for (side, finger), frame in tips.items():
                y, z = finger_target(time, parts[side, finger], key_y, HOME[side][finger], beat,
                                     hover=args.hover_z, pressed=args.pressed_z)
                target = np.array([0.409 if finger == "thumb" else 0.436, y, z])
                if planned is not None:
                    target = planned[side, finger]
                if args.contact_surface and finger == "thumb":
                    depth = np.clip((args.hover_z - target[2]) / (args.hover_z - args.pressed_z), 0., 1.)
                    target[2] -= .002 * depth
                if args.key_press_feedback:
                    active = [n for n in parts[side, finger]
                              if n["start_beat"]*beat <= time-1. < (n["start_beat"]+n["duration_beats"])*beat]
                    note = active[0] if active else None
                    identity = (note["midi"], note["start_beat"]) if note else None
                    if identity != press_notes[side, finger]:
                        press_offsets[side, finger] = 0.
                    press_notes[side, finger] = identity
                    black = note is not None and note["midi"] % 12 not in {0, 2, 4, 5, 7, 9, 11}
                    full_depth = args.pressed_z + (.012 if black else 0.) - (.002 if args.contact_surface and finger == "thumb" else 0.)
                    if note and abs(target[1]-key_y[note["midi"]]) < .002 and target[2] <= full_depth + .001:
                        joint = p.bind(task.piano.joints[note["midi"]-21])
                        low, high = joint.range
                        travel = float((joint.qpos[0]-low)/(high-low))
                        press_offsets[side, finger] = key_press_offset(
                            press_offsets[side, finger], travel, args.key_press_feedback)
                        target[2] -= press_offsets[side, finger]
                    else:
                        press_offsets[side, finger] = 0.
                targets[side, finger] = target
                frame.set_target(mink.SE3.from_translation(target))
                if args.song_active_priority:
                    frame.set_position_cost(song_targets.position_cost(side, finger, time))
            for side, frame in palms.items():
                center = np.mean([targets[side, finger][1] for finger in FINGERS])
                frame.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), np.array([0.29, center, 1.035])))

        set_targets(0)
        for _ in range(250):
            integrate_ik(initializing=True)
        p.data.qpos[:] = configuration.q
        p.data.qvel[:] = 0
        p.forward()
        errors = {side + "/" + finger: float(np.linalg.norm(p.bind(task.pad_sites[side][FINGERS.index(finger)]).xpos - target)) for (side, finger), target in targets.items()}
        print("initial_tip_errors", json.dumps(errors), flush=True)
        imageio.imwrite(args.out / "initial.png", p.render(height=720, width=960, camera_id="whole_robot"))
        if args.pose_only:
            return
        controls, states, frames = [], [], []
        initial = p.data.qpos.copy()
        mjcf.export_with_assets(task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        package = Path(robopianist.__file__).parent
        shutil.copyfile(package / "models/hands/third_party/shadow_hand/LICENSE", args.out / "scene/LICENSE-Shadow-Hand.txt")
        shutil.copyfile(args.model / "LICENSE", args.out / "scene/LICENSE-Galaxea-R1Pro.txt")
        shutil.copyfile(args.model / "NOTICE", args.out / "scene/NOTICE-Galaxea.txt")
        shutil.copyfile(args.plan, args.out / "source.plan.json")
        write_json(args.out / "scene_manifest.json", {"scene": "scene/scene.xml", "mujoco_version": mujoco.__version__,
            "key_joints": [j.full_identifier for j in task.piano.joints], "minimum_midi": 21,
            "activation_tolerance_radians": 0.00872665, "physics_steps_per_control": 10,
            "control_timestep_seconds": 0.02, "camera": "whole_robot"})
        if args.video:
            writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=args.video_fps, codec="libx264", macro_block_size=1)
        prior, prior_target, prior_goal = None, initial.copy(), initial.copy()
        arm_addresses = {side: [int(p.bind(j).qposadr) for j in joints] for side, joints in task.robot.arm_joints.items()}
        if args.g05_python:
            from g05_piano_client import PianoPolicy

            policy = PianoPolicy(args, task, p)
            model_weights = np.zeros(model.nv)
            for joints in task.robot.arm_joints.values():
                for joint in joints:
                    model_weights[int(p.bind(joint).dofadr)] = args.g05_prior_weight
            prior = mink.PostureTask(model, cost=model_weights)
            prior.set_target(prior_target)
            tasks.append(prior)
        duration = validation["duration_seconds"] + 1.5
        hand_actuators = [int(p.bind(a).element_id) for hand in task.robot.hands.values()
                          for a in hand.mjcf_model.find_all("actuator")]
        fixed_control = position_targets(model, configuration.data)
        owned_actuators = set(hand_actuators)
        for side in HOME:
            owned_actuators.update(int(p.bind(task.robot.mjcf_model.find("actuator", f"{side}_arm_joint{i}_act")).element_id)
                                   for i in range(1, 8))
        held_actuators = [i for i in range(model.nu) if i not in owned_actuators]
        for step in range(round(duration / 0.02)):
            if policy is not None and step % 75 == 0:
                proposals = policy.infer(float(p.data.time), score)
                prior_goal = p.data.qpos.copy()
                for side, values in proposals.items():
                    prior_goal[arm_addresses[side]] = values
                print("g05_proposal", json.dumps(policy.calls[-1]), flush=True)
            if prior is not None:
                prior_target += np.clip(prior_goal - prior_target, -0.01, 0.01)
                prior.set_target(prior_target)
            set_targets(step * 0.02)
            configuration.update(p.data.qpos.copy())
            for _ in range(3):
                integrate_ik()
            control = position_targets(model, configuration.data)
            error = configuration.data.actuator_length[hand_actuators] - p.data.actuator_length[hand_actuators]
            control[hand_actuators] = np.clip(
                control[hand_actuators] + np.clip(args.hand_feedback_gain * error, -0.15, 0.15),
                model.actuator_ctrlrange[hand_actuators, 0], model.actuator_ctrlrange[hand_actuators, 1])
            if args.fixed_torso:
                control[held_actuators] = fixed_control[held_actuators]
            env.step(control)
            if not np.isfinite(p.data.qpos).all():
                raise ValueError("Non-finite physical state")
            controls.append(control)
            states.append(p.data.qpos.copy())
            frames.append({"time": float(p.data.time), "active_midi": (np.flatnonzero(task.piano.activation) + 21).tolist(),
                "tips": {side: [p.bind(site).xpos.tolist() for site in hand.fingertip_sites] for side, hand in task.robot.hands.items()},
                "ik_errors": {side + "/" + finger: (configuration.data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, task.pad_sites[side][FINGERS.index(finger)].full_identifier)] - target).tolist() for (side, finger), target in targets.items()},
                "key_press_offsets_m": {side+"/"+finger: offset for (side, finger), offset in press_offsets.items()},
                "wrist_tracking_error": {side: [float(control[int(p.bind(a).element_id)] - p.data.actuator_length[int(p.bind(a).element_id)]) for a in hand.mjcf_model.find_all("actuator") if "_A_WR" in a.name] for side, hand in task.robot.hands.items()}})
            if step % 250 == 0:
                print(f"physical_playback {step * .02:.1f}/{duration:.1f}s", flush=True)
            if writer is not None and step % (50 // args.video_fps) == 0:
                pixels = p.render(height=720, width=960, camera_id="whole_robot")
                writer.append_data(pixels)
                if step in (60, 120, 200, 275):
                    imageio.imwrite(args.out / f"frame-{step:04d}.png", pixels)
        if writer is not None:
            writer.close()
            writer = None
        messages = task.piano.midi_module.get_all_midi_messages()
        events = [{"type": type(e).__name__, "midi": int(e.note), "time_seconds": float(e.time)} for e in messages if isinstance(e, (midi_message.NoteOn, midi_message.NoteOff))]
        report = evaluate(score, events, song_mode=args.song_mode)
        report.update({"mode": "Galaxea_R1Pro_finger_assigned_polyphony", "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "pressed_target_z_m": args.pressed_z, "base": "fixed_chassis", "gravity_compensation": "ideal",
            "hand_reference_feedback_gain": args.hand_feedback_gain,
            "key_press_feedback_gain_m_per_second": args.key_press_feedback,
            "key_press_feedback_limit_m": .004,
            "little_abduction_limit_rad": args.little_abduction_limit,
            "hover_target_z_m": args.hover_z,
            "fixed_torso_controls": args.fixed_torso,
            "contact_surface_control": args.contact_surface,
            "song_mode": args.song_mode,
            "song_active_priority": args.song_active_priority,
            "song_staged_motion": args.song_staged_motion,
            "fingering_geometry": geometry,
            "ik_max_step_radians_per_iteration": args.ik_max_step,
            "song_attack_lead_seconds": .14 if args.song_mode else None,
            "song_release_lead_seconds": args.song_release_lead if args.song_mode else None,
            "finger_distal_equal_angle_coupling": True, "finger_position_controller": "IK with native position/tendon actuators",
            "g05_calls": policy.calls if policy is not None else [], "g05_prior_weight": args.g05_prior_weight if policy is not None else 0,
            "g05_role": "fresh visual/state observations to bounded arm posture priors" if policy is not None else "disabled",
            "g05_finger_outputs": False, "simulation_clock_pauses_for_model": policy is not None,
            "arm_qpos_addresses": arm_addresses, "joint_state_teleported_during_execution": False,
            "versions": {"mujoco": mujoco.__version__, "robopianist": robopianist.__version__}})
        write_json(args.out / "report.json", report)
        write_json(args.out / "raw_robopianist_events.json", events)
        write_json(args.out / "measurements.json", frames)
        np.savez_compressed(args.out / "trajectory.npz", controls=controls, qpos=states, initial_qpos=initial, initial_qvel=np.zeros(model.nv))
        from audit_polyphonic_contacts import audit

        audit_result = audit(args.out)
        report["raw_robopianist_sensor_validation"] = {key: report.pop(key) for key in ("note_count_match", "timing_match", "chords_match", "matched", "missing", "unexpected", "malformed", "chords")}
        report.update({key: audit_result[key] for key in ("note_count_match", "timing_match", "chords_match", "finger_matches", "matched", "missing", "unexpected", "malformed", "chords", "interhand_contact_steps", "key_sensor", "onset_metrics")})
        write_json(args.out / "report.json", report)
        events = audit_result["events"]
        write_json(args.out / "events.json", events)
        messages = [midi_message.NoteOn(note=e["midi"], velocity=80, time=e["time_seconds"]) if e["type"] == "NoteOn" else midi_message.NoteOff(note=e["midi"], time=e["time_seconds"]) for e in events]
        if args.video and any(e["type"] == "NoteOn" for e in events):
            synth = synthesizer.Synthesizer()
            try:
                samples = synth.get_samples(messages)
            finally:
                synth.stop()
            with wave.open(str(args.out / "measured.wav"), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(44100)
                stream.writeframes(samples.tobytes())
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(args.out / "silent.mp4"),
                "-i", str(args.out / "measured.wav"), "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-shortest", str(args.out / "demo.mp4")], check=True)
        report["performance_passed"] = performance_passed(report)
        write_json(args.out / "report.json", report)
        print(json.dumps({"performance_passed": report["performance_passed"], "note_count_match": report["note_count_match"], "timing_match": report["timing_match"], "chords_match": report["chords_match"], "missing": [n["midi"] for n in report["missing"]], "unexpected_count": len(report["unexpected"]), "matched": report["matched"]}, indent=2), flush=True)
        if not report["performance_passed"]:
            raise RuntimeError("Polyphonic performance failed; diagnostic video is not a successful demonstration")
    except (ValueError, RuntimeError) as error:
        write_json(args.out / "failure.json", dict(error=str(error), performance_passed=False,
            simulation_time_seconds=float(env.physics.data.time)))
        raise
    finally:
        if policy is not None:
            policy.close()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pose-only", action="store_true")
    parser.add_argument("--pressed-z", type=float, default=0.916)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--g05-python", type=Path)
    parser.add_argument("--g05-repo", type=Path)
    parser.add_argument("--g05-models", type=Path)
    parser.add_argument("--g05-checkpoint", type=Path)
    parser.add_argument("--g05-prior-weight", type=float, default=0.5)
    parser.add_argument("--hand-feedback-gain", type=float, default=0.0)
    parser.add_argument("--little-abduction-limit", type=float)
    parser.add_argument("--hover-z", type=float, default=.950)
    parser.add_argument("--fixed-torso", action="store_true")
    parser.add_argument("--contact-surface", action="store_true", help="Control the distal collision-mesh contact patch")
    parser.add_argument("--song-mode", action="store_true")
    parser.add_argument("--song-active-priority", action="store_true",
                        help="Relax idle fingertip XY tasks, keeping vertical clearance")
    parser.add_argument("--song-staged-motion", action="store_true",
                        help="Experimental release-before-travel finger targets")
    parser.add_argument("--ik-max-step", type=float,
                        help="Experimental per-iteration joint delta bound, at most 0.2 rad")
    parser.add_argument("--max-hand-span", type=float,
                        help="Reject held-note lateral spans above this calibrated limit in meters")
    parser.add_argument("--song-release-lead", type=float, default=.09,
                        help="Seconds to anticipate key release; default 0.09, calibrated per study")
    parser.add_argument("--key-press-feedback", type=float, default=0.,
                        help="Experimental key-travel to fingertip-depth integral gain (0..0.1 m/s)")
    run(parser.parse_args())
