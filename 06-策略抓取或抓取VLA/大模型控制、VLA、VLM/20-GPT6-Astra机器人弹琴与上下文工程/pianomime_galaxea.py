"""Experimental physical transfer of recorded PianoMime actions to a full R1 Pro.

The low-level policy was closed-loop in the source environment. This transfer
replays its actions; it is NOT closed-loop PianoMime on the new embodiment.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

from galaxea_piano import build_task
from physical_piano import position_targets


def validate_plan(plan, source_identity=None):
    required = {"task", "steps", "g05_prior_weight", "g05_phase", "finger_controller", "rationale"}
    if not isinstance(plan, dict) or set(plan) != required or not isinstance(plan["rationale"], str):
        raise ValueError("Invalid orchestration plan fields")
    task, steps = source_identity if source_identity is not None else ("Adieu_0", 120)
    if (not isinstance(task, str) or not task.replace("_", "").isalnum()
            or type(steps) is not int or not 1 <= steps <= 12000):
        raise ValueError("Invalid source rollout identity")
    if (plan.get("task") != task or type(plan.get("steps")) is not int or plan["steps"] != steps
            or plan.get("g05_phase") != "preparation_only"
            or plan.get("finger_controller") != "pianomime_low_level"):
        raise ValueError("Unsupported or unsafe orchestration plan")
    weight = plan.get("g05_prior_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.01 <= weight <= 0.15:
        raise ValueError("G0.5 arm prior weight must be within [0.01, 0.15]")
    return weight


def unique_suffix(names, suffix):
    matches = [i for i, name in enumerate(names) if name and name.rsplit("/", 1)[-1] == suffix]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source {suffix}, found {len(matches)}")
    return matches[0]


def reduced_arm_problem(problem, velocities, positions, qpos, ranges):
    """Freeze unowned DOFs exactly before solving, not after optimization."""
    import numpy as np

    if len(velocities) != len(positions) or len(set(velocities)) != len(velocities):
        raise ValueError("Arm mapping must have unique DOFs with corresponding positions")
    limits = np.asarray(ranges)
    if limits.shape != (len(velocities), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid arm position limits")
    lower = 0.95 * (limits[:, 0] - qpos[positions])
    upper = 0.95 * (limits[:, 1] - qpos[positions])
    return problem.P[np.ix_(velocities, velocities)], problem.q[velocities], lower, upper


def position_velocity_command(position, velocity, kp, kv, ranges):
    """Compensate a position actuator's damping using reference velocity."""
    import numpy as np

    inputs = [np.asarray(v, dtype=float) for v in (position, velocity, kp, kv, ranges)]
    position, velocity, kp, kv, ranges = inputs
    if (not all(np.isfinite(v).all() for v in inputs) or np.any(kp <= 0) or np.any(kv < 0)
            or position.shape != velocity.shape or kp.shape != position.shape
            or kv.shape != position.shape or ranges.shape != position.shape + (2,)):
        raise ValueError("Invalid position/velocity controller inputs")
    raw = position + kv / kp * velocity
    clipped = np.clip(raw, ranges[:, 0], ranges[:, 1])
    return clipped, int(np.count_nonzero(raw != clipped))


def run(args):
    if not math.isfinite(args.hand_feedback_gain) or not 0 <= args.hand_feedback_gain <= 4:
        raise ValueError("Hand reference feedback gain must be finite and within [0, 4]")
    import imageio.v2 as imageio
    import mink
    import mujoco
    import numpy as np
    import robopianist
    import qpsolvers
    from scipy.interpolate import PchipInterpolator
    from dm_control import mjcf
    from mujoco_utils import composer_utils

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source_report = json.loads((args.source / "report.json").read_text())
    weight = validate_plan(plan, (source_report["task"], source_report["steps"]))
    paths = (args.g05_python, args.g05_repo, args.g05_models, args.g05_checkpoint)
    if any(paths) and not all(paths):
        raise ValueError("Supply all four G0.5 paths, or none for an ablation")
    if source_report["task"] != plan["task"] or source_report["steps"] != plan["steps"]:
        raise ValueError("Plan does not match the actual source rollout")
    manifest = json.loads((args.source / "source_manifest.json").read_text())
    data = np.load(args.source / "trajectory.npz", allow_pickle=False)
    required_arrays = ("qpos", "actions", "initial_qpos", "actuator_controls", "sustain", "physics_times", "physics_qpos")
    if any(key not in data for key in required_arrays):
        raise ValueError("Regenerate the PianoMime source: actual actuator controls, sustain and substeps are required")
    for key in required_arrays:
        if not np.isfinite(data[key]).all():
            raise ValueError("Non-finite source trajectory")
    source = mujoco.MjModel.from_xml_path(str(args.source / "scene/scene.xml"))
    reference = mujoco.MjData(source)
    reference.qpos[:] = data["initial_qpos"]
    mujoco.mj_forward(source, reference)
    source_piano = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, manifest["piano_root"])
    if source_piano < 0:
        raise ValueError("Source piano root is missing")
    source_rotation = reference.xmat[source_piano].reshape(3, 3).copy()
    source_origin = reference.xpos[source_piano].copy()
    args.out.mkdir(parents=True, exist_ok=False)
    task = build_task(args.model, piano_y=args.keyboard_y, policy_hands=True)
    task.set_timesteps(control_timestep=0.01, physics_timestep=0.002)
    # Set MJCF as well as physics: exported replay scenes must use the same
    # key contact softness as the policy's original environment.
    for geom in task.piano.mjcf_model.find_all("geom"):
        src = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_GEOM, geom.full_identifier)
        if src >= 0:
            geom.solref = source.geom_solref[src]
    for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        camera = task.robot.mjcf_model.find("camera", name)
        camera.mode, camera.target = "targetbody", task.piano.root_body
    env = composer_utils.Environment(task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False)
    writer = policy = None
    try:
        env.reset()
        p, model = env.physics, env.physics.model.ptr
        from piano_events import PianoEventSensor, midi_messages

        audio_sensor = PianoEventSensor()
        key_ranges = p.model.jnt_range[np.asarray(p.bind(task.piano.joints).element_id)]
        original_substep = task.after_substep

        def record_key_substep(current, random_state):
            original_substep(current, random_state)
            travel = (current.bind(task.piano.joints).qpos - key_ranges[:, 0]) / np.diff(key_ranges, axis=1).ravel()
            audio_sensor.update(float(current.data.time), travel, float(task.piano.sustain_activation[0]))

        task.after_substep = record_key_substep
        destination = p.bind(task.piano.root_body)
        rotation = destination.xmat.reshape(3, 3) @ source_rotation.T
        origin = destination.xpos - rotation @ source_origin
        configuration = mink.Configuration(model)
        seed = p.data.qpos.copy()
        for name, value in task.robot.seed.items():
            seed[int(p.bind(task.robot.mjcf_model.find("joint", name)).qposadr)] = value
        source_joints = [mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(source.njnt)]
        source_bodies = [mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(source.nbody)]
        hand_mapping, action_mapping, palm_ids, frames, mounts = [], [], {}, {}, {}
        arm_q, arm_v = {}, []
        for side, hand in task.robot.hands.items():
            prefix = "lh_" if side == "left" else "rh_"
            palm_ids[side] = unique_suffix(source_bodies, prefix + "palm")
            source_wrist = unique_suffix(source_bodies, prefix + "wrist")
            mounts[side] = (int(source.body_parentid[source_wrist]), source.body_pos[source_wrist].copy())
            if not np.allclose(source.body_quat[source_wrist], [1, 0, 0, 0]):
                raise ValueError("Unsupported fixed wrist rotation in source hand")
            frames[side] = mink.FrameTask(hand.root_body.full_identifier, "body", position_cost=100,
                orientation_cost=10, lm_damping=0.001)
            arm_q[side] = [int(p.bind(j).qposadr) for j in task.robot.arm_joints[side]]
            arm_v.extend(int(p.bind(j).dofadr) for j in task.robot.arm_joints[side])
            for joint in hand.joints:
                src = int(source.jnt_qposadr[unique_suffix(source_joints, joint.name)])
                dst = int(p.bind(joint).qposadr)
                hand_mapping.append((dst, src))
                seed[dst] = data["initial_qpos"][src]
            for actuator in hand.mjcf_model.find_all("actuator"):
                src = unique_suffix(manifest["actuators"], actuator.name)
                action_mapping.append((int(p.bind(actuator).element_id), src))
        if len(action_mapping) != 40:
            raise ValueError("Expected 20 named hand actuators on each hand")
        costs = np.full(model.nv, 10000.0)
        costs[arm_v] = 0.001
        configuration.update(seed)
        posture = mink.PostureTask(model, cost=costs)
        posture.set_target(seed)
        tasks = [posture, *frames.values()]
        arm_positions = [address for addresses in arm_q.values() for address in addresses]
        arm_ranges = np.asarray([joint.range for side in arm_q for joint in task.robot.arm_joints[side]])
        goals = {}

        def set_reference(qpos):
            reference.qpos[:] = qpos
            mujoco.mj_forward(source, reference)
            for side, frame in frames.items():
                pos = origin + rotation @ reference.xpos[palm_ids[side]]
                rot = rotation @ reference.xmat[palm_ids[side]].reshape(3, 3)
                goals[side] = (pos, rot)
                # The R1 mount replaces the source forearm translations. The hand's
                # two actuated wrist joints must not be controlled a second time by IK.
                parent, offset = mounts[side]
                mount_rot = rotation @ reference.xmat[parent].reshape(3, 3)
                mount_pos = origin + rotation @ reference.xpos[parent] + mount_rot @ offset
                frame.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(mount_rot), mount_pos))

        def solve(iterations, step_origin=None):
            for _ in range(iterations):
                # Do not ask IK to return moving fingers to their initial pose:
                # their velocity is masked below and supplied by the learned policy.
                posture_target = configuration.q.copy()
                for addresses in arm_q.values():
                    posture_target[addresses] = seed[addresses]
                posture.set_target(posture_target)
                problem = mink.build_ik(configuration, tasks, 0.01, damping=1e-5, limits=[])
                # Solve a reduced QP, rather than solving for fingers and throwing
                # those velocities away. Piano/hand physical limits remain active
                # in MuJoCo; the arm optimizer owns only these 14 position limits.
                hessian, gradient, lower, upper = reduced_arm_problem(
                    problem, arm_v, arm_positions, configuration.q, arm_ranges)
                if step_origin is not None:
                    # Bound total displacement per control tick, not per IK iteration.
                    lower = np.maximum(lower, step_origin - 0.02 - configuration.q[arm_positions])
                    upper = np.minimum(upper, step_origin + 0.02 - configuration.q[arm_positions])
                delta = qpsolvers.solve_qp(hessian, gradient, lb=lower, ub=upper, solver="daqp")
                if delta is None or not np.isfinite(delta).all():
                    raise RuntimeError("Arm-only inverse kinematics failed")
                masked = np.zeros(model.nv)
                masked[arm_v] = delta / 0.01
                configuration.integrate_inplace(masked, 0.01)

        def palm_errors():
            errors = {}
            for side, hand in task.robot.hands.items():
                prefix = "lh_" if side == "left" else "rh_"
                palm = p.bind(hand.mjcf_model.find("body", prefix + "palm"))
                pos, rot = goals[side]
                angle = np.arccos(np.clip((np.trace(rot.T @ palm.xmat.reshape(3, 3)) - 1) / 2, -1, 1))
                errors[side] = {"position_m": float(np.linalg.norm(palm.xpos - pos)), "rotation_rad": float(angle)}
            return errors

        set_reference(data["initial_qpos"])
        solve(250)
        p.data.qpos[:] = configuration.q
        p.data.qvel[:] = 0
        p.forward()
        preparation = {"before": palm_errors(), "g05_adopted": False}
        if args.g05_python:
            from g05_piano_client import PianoPolicy
            policy = PianoPolicy(args, task, p)
            proposals = policy.infer(0, {"tempo_bpm": 60, "notes": []})
            prior_goal = configuration.q.copy()
            for side, values in proposals.items():
                prior_goal[arm_q[side]] = values
            weights = np.zeros(model.nv)
            weights[arm_v] = weight
            prior = mink.PostureTask(model, cost=weights)
            prior.set_target(prior_goal)
            before = configuration.q.copy()
            tasks.append(prior)
            solve(150)
            p.data.qpos[:] = configuration.q
            p.forward()
            errors = palm_errors()
            adopted = all(e["position_m"] <= 0.003 and e["rotation_rad"] <= 0.03 for e in errors.values())
            preparation.update({"candidate_errors": errors, "g05_adopted": adopted,
                "max_arm_delta_rad": float(np.max(np.abs(configuration.q - before)))})
            if not adopted:
                tasks.remove(prior)
                configuration.update(before)
                p.data.qpos[:] = before
                p.forward()
        initial = p.data.qpos.copy()
        initial_error = palm_errors()
        print("preparation", json.dumps(preparation), flush=True)
        if any(e["position_m"] > 0.003 or e["rotation_rad"] > 0.03 for e in initial_error.values()):
            raise RuntimeError("Source hand pose is not reachable within preparation tolerances")
        mjcf.export_with_assets(task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        shutil.copyfile(args.model / "LICENSE", args.out / "scene/LICENSE-Galaxea.txt")
        shutil.copyfile(args.model / "NOTICE", args.out / "scene/NOTICE-Galaxea.txt")
        shutil.copyfile(Path(robopianist.__file__).parent / "models/hands/third_party/shadow_hand/LICENSE", args.out / "scene/LICENSE-Shadow.txt")
        imageio.imwrite(args.out / "initial.png", p.render(height=720, width=960, camera_id="whole_robot"))
        if args.video:
            writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=20, codec="libx264")
        controls, states, measured, errors, interhand = [], [], [], [], 0
        fixed_controls = position_targets(model, configuration.data)
        moving_actuators = {dst for dst, _ in action_mapping}
        arm_actuators = []
        for side in arm_q:
            arm_actuators.extend(int(p.bind(task.robot.mjcf_model.find("actuator", f"{side}_arm_joint{i}_act")).element_id)
                                 for i in range(1, 8))
        moving_actuators.update(arm_actuators)
        fixed_actuators = [i for i in range(model.nu) if i not in moving_actuators]
        source_times = np.r_[0.0, data["physics_times"]]
        source_states = np.vstack((data["initial_qpos"], data["physics_qpos"]))
        if np.any(np.diff(source_times) <= 0) or len(source_states) != len(source_times):
            raise ValueError("Source substeps must have strictly increasing timestamps")
        reference_times = np.clip(np.arange(len(data["actions"]) * 5 + 1) * 0.01, 0, source_times[-1])
        dense_reference = PchipInterpolator(source_times, source_states, axis=0)(reference_times)
        planned_arms = [initial[arm_positions].copy()]
        if args.arm_tracking != "reactive":
            for qpos in dense_reference[1:]:
                set_reference(qpos)
                solve(12, configuration.q[arm_positions].copy())
                planned_arms.append(configuration.q[arm_positions].copy())
            planned_arms = np.asarray(planned_arms)
            planned_velocity = np.gradient(planned_arms, 0.01, axis=0)
            configuration.update(initial)
            np.savez_compressed(args.out / "arm_reference.npz", qpos=planned_arms, qvel=planned_velocity,
                                times=reference_times, qpos_addresses=arm_positions)
        ff_clipped = 0
        for index, action in enumerate(data["actuator_controls"]):
            # The upstream task overwrites the policy's sustain slot with the score.
            # Carry the actual pedal state, not the unused policy slot, to this piano.
            task.piano.apply_sustain(p, float(data["sustain"][index]), None)
            for substep in range(5):
                tick = index * 5 + substep + 1
                set_reference(dense_reference[tick])
                if args.arm_tracking == "predictive":
                    control = fixed_controls.copy()
                    command, clipped = position_velocity_command(planned_arms[tick], planned_velocity[tick],
                        model.actuator_gainprm[arm_actuators, 0], -model.actuator_biasprm[arm_actuators, 2],
                        model.actuator_ctrlrange[arm_actuators])
                    control[arm_actuators] = command
                    ff_clipped += clipped
                else:
                    configuration.update(p.data.qpos.copy())
                    solve(3, configuration.q[arm_positions].copy())
                    control = position_targets(model, configuration.data)
                # Stabilize the torso instead of feeding back its displaced angles.
                control[fixed_actuators] = fixed_controls[fixed_actuators]
                for dst, src in action_mapping:
                    error = reference.actuator_length[src] - p.data.actuator_length[dst]
                    correction = np.clip(args.hand_feedback_gain * error, -0.15, 0.15)
                    control[dst] = np.clip(action[src] + correction, *model.actuator_ctrlrange[dst])
                env.step(control)
                if not np.isfinite(p.data.qpos).all():
                    raise RuntimeError("Non-finite full-body physics")
                controls.append(control.copy())
                states.append(p.data.qpos.copy())
                for contact in p.data.contact[:p.data.ncon]:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or "" for i in contact.geom]
                    if ("lh_shadow_hand" in names[0] and "rh_shadow_hand" in names[1]) or ("rh_shadow_hand" in names[0] and "lh_shadow_hand" in names[1]):
                        interhand += 1
            measured.append(task.piano.activation.copy())
            errors.append(palm_errors())
            if writer:
                pixels = p.render(height=720, width=960, camera_id="whole_robot")
                writer.append_data(pixels)
                if index in (19, 59, 99):
                    imageio.imwrite(args.out / f"frame-{index + 1:04d}.png", pixels)
            if (index + 1) % 20 == 0:
                print(f"Full robot {index + 1}/{len(data['actions'])}", flush=True)
        if writer:
            writer.close()
            writer = None
        expected, actual = data["expected"].astype(bool), np.asarray(measured, dtype=bool)
        tp, fp, fn = int((expected & actual).sum()), int((~expected & actual).sum()), int((expected & ~actual).sum())
        report = {"mode": "pianomime_action_transfer_to_R1Pro", "closed_loop_pianomime_on_R1": False,
            "source_report": source_report, "plan": plan,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "preparation": preparation, "g05_calls": policy.calls if policy else [],
            "g05_role": "preparation arm posture prior, fixed throughout playback",
            "joint_state_teleported_during_execution": False, "control_hz": 100,
            "arm_tracking": args.arm_tracking, "arm_feedforward_clipped_coordinates": ff_clipped,
            "hand_controls": "recorded MuJoCo actuator ctrl, matched by name",
            "hand_reference_feedback_gain": args.hand_feedback_gain,
            "audio_sensor": "all 88 keys, each physics substep, 90% press / 50% release; no score filtering",
            "physics_timestep": 0.002, "arm_reference_speed_limit_rad_s": 2.0,
            "pedal": "virtual sustain copied from source; not a robot foot controller",
            "keyboard_y_m": args.keyboard_y,
            "arm_tracking_frame": "hand mount, not actuated wrist/palm",
            "micro_precision": tp / max(1, tp + fp), "micro_recall": tp / max(1, tp + fn),
            "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_positive_key_frames": fp,
            "false_negative_key_frames": fn, "interhand_contact_samples": interhand,
            "max_palm_position_error_m": max(e[s]["position_m"] for e in errors for s in frames),
            "max_palm_rotation_error_rad": max(e[s]["rotation_rad"] for e in errors for s in frames),
            "key_frame_check_passed": fp == 0 and fn == 0 and interhand == 0,
            "complete_performance_validated": False,
            "evaluation_scope": f"{len(data['actions']) * .05:.2f}-second recorded source; finger ownership and complete song are not audited",
            "limitations": ["fixed base", "ideal gravity compensation", "simulation-only Shadow mounting",
                            "no G0.5 finger action outputs", "recorded policy actions, not online retargeted feedback"]}
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.out / "palm_errors.json").write_text(json.dumps(errors), encoding="utf-8")
        np.savez_compressed(args.out / "trajectory.npz", controls=controls, qpos=states,
            initial_qpos=initial, initial_qvel=np.zeros(model.nv), expected=expected, measured=actual)
        messages = task.piano.midi_module.get_all_midi_messages()
        (args.out / "raw_robopianist_events.json").write_text(json.dumps([
            {"type": type(e).__name__, "time": float(e.time), "note": int(e.note) if hasattr(e, "note") else None}
            for e in messages]), encoding="utf-8")
        (args.out / "events.json").write_text(json.dumps(audio_sensor.events), encoding="utf-8")
        messages = midi_messages(audio_sensor.events)
        if args.video and any(type(e).__name__ == "NoteOn" for e in messages):
            from robopianist.music.synthesizer import Synthesizer
            synth = Synthesizer()
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
                "-i", str(args.out / "measured.wav"), "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-shortest",
                str(args.out / "diagnostic.mp4")], check=True)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if writer:
            writer.close()
        if policy:
            policy.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source", "plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--arm-tracking", choices=("predictive", "reactive"), default="predictive")
    parser.add_argument("--hand-feedback-gain", type=float, default=2.0)
    parser.add_argument("--keyboard-y", type=float, default=0.0,
                        help="Keyboard lateral placement in meters, restricted to [-0.15, 0.15]")
    for name in ("python", "repo", "models", "checkpoint"):
        parser.add_argument("--g05-" + name, type=Path)
    run(parser.parse_args())
