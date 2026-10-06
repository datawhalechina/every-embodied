"""One physical session: play, lift, translate piano, query G0.5, resume."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import wave

import numpy as np

from g05_recovery import bounded_command, piano_shift, sustained_success
from g05_piano_session import digest
from piano_context import write_json
from piano_relocation import PianoTranslation, score_with_pause
from pianomime_galaxea import unique_suffix, reduced_arm_problem
from physical_piano import position_targets
from polyphonic_score import KeyCycles, performance_passed


def ease(t):
    t = np.clip(t, 0, 1)
    return 10*t**3 - 15*t**4 + 6*t**5


def run(args):
    import imageio.v2 as imageio
    import mink
    import mujoco
    import qpsolvers
    from dm_control import mujoco as dm_mujoco
    from g05_piano_client import PianoPolicy
    from audit_polyphonic_contacts import audit
    from robopianist.music import midi_message, synthesizer

    shift = piano_shift(args.piano_shift)
    if not np.any(shift):
        raise ValueError("Relocation needs nonzero translation")
    if not .02 <= args.lift_height <= .10:
        raise ValueError("Invalid lift height")
    if not .01 <= args.model_weight <= 2.:
        raise ValueError("Model weight must be in [0.01, 2]")
    durations = dict(lift=1., move=.8, recover=1.6, lower=1.)
    pause = sum(durations.values())
    if abs(args.boundary / .02 - round(args.boundary / .02)) > 1e-8:
        raise ValueError("Boundary must align with 50 Hz controls")
    original = json.loads((args.reference / "source.plan.json").read_text())
    score = score_with_pause(original, args.boundary, pause)
    manifest = json.loads((args.reference / "scene_manifest.json").read_text())
    if manifest["control_timestep_seconds"] != .02 or manifest["physics_steps_per_control"] != 10:
        raise ValueError("Expected original 50 Hz performance")
    if manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("Source MuJoCo version mismatch")
    with np.load(args.reference / "trajectory.npz", allow_pickle=False) as trace:
        source_controls = trace["controls"].copy()
        source_states = trace["qpos"].copy()
        initial = trace["initial_qpos"].copy()
    cut = round(args.boundary / .02)
    if not 0 < cut < len(source_controls):
        raise ValueError("Boundary outside performance")
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.reference / "scene", args.out / "scene")
    write_json(args.out / "source.plan.json", score)
    manifest.update(control_timestep_seconds=.01, physics_steps_per_control=5,
                    piano_translation="trajectory.npz:piano_offsets; applied before each control")
    write_json(args.out / "scene_manifest.json", manifest)
    p = dm_mujoco.Physics.from_xml_path(str(args.out / "scene/scene.xml"))
    model = p.model.ptr
    mover = PianoTranslation(model)
    names = lambda kind, count: [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    jnames, anames, bnames = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt), names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu), names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    arm_j = [unique_suffix(jnames, f"{s}_arm_joint{i}") for s in ("left", "right") for i in range(1, 8)]
    qids, vids = model.jnt_qposadr[arm_j], model.jnt_dofadr[arm_j]
    aids = [unique_suffix(anames, f"{s}_arm_joint{i}_act") for s in ("left", "right") for i in range(1, 8)]
    if any(model.actuator_trnid[a, 0] != j or model.actuator_gear[a, 0] != 1 for a, j in zip(aids, arm_j)):
        raise ValueError("Expected direct unit-gear arm actuators")
    if any((n or "").startswith("piano/") for n in anames):
        raise ValueError("Piano actuators prohibited")
    limits = model.jnt_range[arm_j].copy()
    mounts = [unique_suffix(bnames, prefix + "forearm") for prefix in ("lh_", "rh_")]
    config = mink.Configuration(model)
    frames = [mink.FrameTask(bnames[i], "body", position_cost=100, orientation_cost=10, lm_damping=.001) for i in mounts]
    weights = np.zeros(model.nv)
    weights[vids] = args.model_weight
    prior = mink.PostureTask(model, cost=weights)

    def poses(data):
        return [mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(data.xmat[i].reshape(3, 3)), data.xpos[i].copy()) for i in mounts]

    def shifted(goals, delta):
        return [mink.SE3.from_rotation_and_translation(g.rotation(), g.translation() + delta) for g in goals]

    def solve(qpos, goals, proposal=None, bound=.02, command=None, anchor=None):
        config.update(qpos)
        for f, g in zip(frames, goals):
            f.set_target(g)
        tasks = list(frames)
        if proposal is not None:
            prior.set_target(proposal)
            tasks.append(prior)
        problem = mink.build_ik(config, tasks, .01, damping=1e-5, limits=[])
        P, q, lo, hi = reduced_arm_problem(problem, vids, qids, config.q, limits)
        if command is not None:
            # Solve inside executable limits so per-joint clipping cannot undo wrist alignment.
            if anchor is None:
                raise ValueError("Bounded IK requires a recovery anchor")
            lo = np.maximum.reduce([lo, anchor - .35 - qpos[qids], command - .006 - qpos[qids]])
            hi = np.minimum.reduce([hi, anchor + .35 - qpos[qids], command + .006 - qpos[qids]])
        dq = qpsolvers.solve_qp(P, q, lb=np.maximum(lo, -bound), ub=np.minimum(hi, bound), solver="daqp")
        if dq is None or not np.isfinite(dq).all():
            raise RuntimeError("Arm solve failed")
        target = qpos.copy()
        target[qids] += dq
        return target

    # Retarget commanded arm mounts; finger commands, torso and note timing stay intact.
    fk = mujoco.MjData(model)
    retargeted = source_controls.copy()
    seed = initial[qids].copy()
    max_target_error = 0.
    for i in range(cut, len(source_controls)):
        fk.qpos[:] = source_states[max(0, i-1)]
        fk.qpos[qids] = source_controls[i, aids]
        mujoco.mj_forward(model, fk)
        goals = shifted(poses(fk), shift)
        q = fk.qpos.copy()
        q[qids] = seed if i > cut else fk.qpos[qids]
        for iteration in range(60):
            q = solve(q, goals, bound=.04)
            config.update(q)
            residual = max(np.linalg.norm(config.data.xpos[k]-g.translation()) for k, g in zip(mounts, goals))
            if iteration >= 4 and residual < .0001:
                break
        if residual > .003:
            alternate = fk.qpos.copy()
            for _ in range(100):
                alternate = solve(alternate, goals, bound=.04)
            config.update(alternate)
            alternate_error = max(np.linalg.norm(config.data.xpos[k]-g.translation()) for k,g in zip(mounts,goals))
            if alternate_error < residual:
                q, residual = alternate, alternate_error
        seed = q[qids].copy()
        retargeted[i, aids] = seed
        config.update(q)
        max_target_error = max(max_target_error, residual)
        if residual > .003:
            raise ValueError(f"Retarget error {residual:.5f} m at source time {i*.02:.2f}s; per arm="
                             f"{[float(np.linalg.norm(config.data.xpos[k]-g.translation())) for k,g in zip(mounts,goals)]}")
    if max_target_error > .003:
        raise ValueError(f"Retarget error {max_target_error:.5f} m")
    np.savez_compressed(args.out / "retargeted_controls.npz", controls=retargeted, piano_shift=shift)
    policy, writer = None, None
    states, controls, offsets, phases = [], [], [], []
    stop, calls, recovery_errors, influence = None, [], [], 0.
    sensor = KeyCycles()
    key_j = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in manifest["key_joints"]]
    robot = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("galaxea_r1pro/") for i in range(model.ngeom)])
    gbody = model.geom_bodyid
    key_bodies = set(model.jnt_bodyid[key_j])
    p.data.qpos[:] = initial
    p.data.qvel[:] = 0
    p.forward()
    current_shift = np.zeros(3)
    stage_ranges = []

    def tick(control, phase, offset=None):
        nonlocal current_shift
        if offset is not None:
            current_shift = offset.copy()
        mover.apply(current_shift)
        if np.any(control < model.actuator_ctrlrange[:, 0]) or np.any(control > model.actuator_ctrlrange[:, 1]):
            raise RuntimeError("Actuator range exceeded")
        for _ in range(5):
            p.set_control(control)
            p.step()
            if not np.isfinite(p.data.qpos).all() or not np.isfinite(p.data.qvel).all():
                raise RuntimeError("Nonfinite state")
            ranges = model.jnt_range[key_j]
            changes = sensor.update((p.data.qpos[model.jnt_qposadr[key_j]]-ranges[:, 0])/(ranges[:, 1]-ranges[:, 0]))
            if phase in ("move", "recover"):
                if any(active for _, active in changes) or any(sensor.active):
                    raise RuntimeError("Key pressed during relocation")
                for c in p.data.contact:
                    if (robot[c.geom1] and gbody[c.geom2] in key_bodies) or (robot[c.geom2] and gbody[c.geom1] in key_bodies):
                        raise RuntimeError("Hand/key contact during relocation")
            if np.any(p.data.qpos[qids] < limits[:, 0]-.02) or np.any(p.data.qpos[qids] > limits[:, 1]+.02):
                raise RuntimeError("Physical arm joint limit exceeded")
        states.append(p.data.qpos.copy())
        controls.append(control.copy())
        offsets.append(current_shift.copy())
        phases.append(phase)
        if len(states) % 5 == 0:
            writer.append_data(p.render(width=960, height=720, camera_id="whole_robot"))

    def error(goals):
        return dict(time_seconds=float(p.data.time),
                    position_m=max(float(np.linalg.norm(p.data.xpos[k]-g.translation())) for k, g in zip(mounts, goals)),
                    rotation_rad=max(float(np.arccos(np.clip((np.trace(g.rotation().as_matrix().T @ p.data.xmat[k].reshape(3,3))-1)/2,-1,1))) for k,g in zip(mounts, goals)))

    def observe():
        obs = {name: p.render(width=size[0], height=size[1], camera_id="galaxea_r1pro/"+name).transpose(2,0,1)
               for name,size in (("head_rgb",(640,360)),("left_wrist_rgb",(640,480)),("right_wrist_rgb",(640,480)))}
        for i, side in enumerate(("left", "right")):
            obs[side+"_arm"] = p.data.qpos[qids[i*7:i*7+7]].astype(np.float32).copy()
            prefix = "lh_" if i == 0 else "rh_"
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"galaxea_r1pro/{prefix}shadow_hand/{f}distal_site") for f in ("th", "ff")]
            if min(ids) < 0:
                raise ValueError("Missing gripper proxy sites")
            obs[side+"_gripper"] = np.array([np.clip(np.linalg.norm(p.data.site_xpos[ids[0]]-p.data.site_xpos[ids[1]])*1000,0,100)],dtype=np.float32)
        return obs

    began = time.monotonic()
    try:
        policy = PianoPolicy(args, None, None)
        writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=20, codec="libx264")
        for c in np.repeat(source_controls[:cut], 2, axis=0):
            tick(c, "play_before")
        print("prefix complete", float(p.data.time), flush=True)
        hold = position_targets(model, p.data.ptr)
        original_goals = poses(p.data)
        lift_goals = shifted(original_goals, np.array([0, 0, args.lift_height]))
        anchor = p.data.qpos[qids].copy()
        control = hold.copy()
        start_time = float(p.data.time)
        for i in range(100):
            goals = shifted(original_goals, np.array([0,0,args.lift_height*ease((i+1)/100)]))
            target = solve(p.data.qpos.copy(), goals)
            control[aids] = bounded_command(target[qids], control[aids], anchor, limits)
            tick(control, "lift")
        if error(lift_goals)["position_m"] > .005 or any(sensor.active):
            raise RuntimeError("Hands not lifted clear of keys")
        stage_ranges.append(dict(phase="lift", start=start_time, end=float(p.data.time)))
        start_time = float(p.data.time)
        for i in range(80):
            tick(control, "move", shift * ease((i+1)/80))
        stage_ranges.append(dict(phase="move", start=start_time, end=float(p.data.time)))
        print("piano moved", current_shift.tolist(), flush=True)
        goals = shifted(lift_goals, shift)
        anchor = p.data.qpos[qids].copy()
        start_time = float(p.data.time)
        chunk = {}
        for i in range(160):
            if i % 40 == 0:
                targets = "; ".join(f"{side} xyz={np.round(g.translation(),3).tolist()} wxyz={np.round(g.rotation().wxyz,3).tolist()}" for side,g in zip(("left","right"),goals))
                chunk = policy.infer_observation(float(p.data.time), observe(),
                    "The piano has moved sideways. Reposition both arms above the relocated keyboard. World-frame hover targets: " + targets,
                    "Keep fingers still and above the keys. Propose arm actions for re-alignment; a dedicated finger controller will resume the next phrase.", chunk=True, seed=2300+i//40)
                calls.append(policy.calls[-1])
            measured = p.data.qpos.copy()
            proposal = measured.copy()
            row = int((i%40)*.01*15)
            for h,side in enumerate(("left","right")):
                if side in chunk:
                    proposal[qids[h*7:h*7+7]] = chunk[side][min(row,len(chunk[side])-1)]
            target = solve(measured, goals, proposal, command=control[aids], anchor=anchor)
            commanded = bounded_command(target[qids], control[aids], anchor, limits)
            baseline = bounded_command(solve(measured, goals, command=control[aids], anchor=anchor)[qids],
                                       control[aids], anchor, limits)
            influence += float(np.linalg.norm(commanded-baseline))*.01
            control[aids] = commanded
            tick(control, "recover")
            recovery_errors.append(error(goals))
        stage_ranges.append(dict(phase="recover", start=start_time, end=float(p.data.time)))
        if not sustained_success(recovery_errors) or influence <= 1e-8:
            raise RuntimeError("Model-assisted recovery did not reach its target")
        print("recovery complete", recovery_errors[-1], flush=True)
        # Blend to the newly retargeted next command, never reset qpos/qvel.
        target_control = retargeted[cut]
        from g05_piano_session import handoff_controls
        # Quintic easing peaks at 1.875; extend the pause to respect the 0.6 rad/s arm limit.
        arm_distance = float(np.max(np.abs(target_control[aids] - control[aids])))
        durations["lower"] = max(1., float(np.ceil(1.875 * arm_distance / .6 * 100) / 100))
        pause = sum(durations.values())
        write_json(args.out / "source.plan.json", score_with_pause(original, args.boundary, pause))
        start_time = float(p.data.time)
        for c in handoff_controls(control, target_control, durations["lower"], aids):
            tick(c, "lower")
        stage_ranges.append(dict(phase="lower", start=start_time, end=float(p.data.time)))
        for i,c in enumerate(np.repeat(retargeted[cut:], 2, axis=0)):
            tick(c, "play_after")
            if i % 2000 == 0:
                print("resumed", float(p.data.time), flush=True)
    except (RuntimeError, ValueError) as exc:
        stop = str(exc)
    finally:
        if writer:
            writer.close()
        if policy:
            policy.close()
    if not controls:
        p.free()
        raise RuntimeError(stop or "No controls executed")
    np.savez_compressed(args.out / "trajectory.npz", initial_qpos=initial, initial_qvel=np.zeros(model.nv),
                        controls=controls, qpos=states, piano_offsets=offsets)
    report = dict(mode="g05_mid_phrase_piano_relocation", song_mode=True,
                  plan_sha256=digest(args.out/"source.plan.json"), original_plan_sha256=digest(args.reference/"source.plan.json"),
                  original_trajectory_sha256=digest(args.reference/"trajectory.npz"),
                  executed_seconds=len(controls)*.01, planned_seconds=len(source_controls)*.02+pause,
                  playback_completed=stop is None, safety_stop=stop, phases=stage_ranges,
                  insertion_original_seconds=args.boundary, insertion_duration_seconds=pause,
                  performance_start_seconds=0., piano_shift_world_m=shift.tolist(),
                  model_calls_during_session=len(calls), model_calls=calls,
                  g05_prior_weight=args.model_weight,
                  model_induced_command_distance_rad_seconds=influence, recovery_final_error=recovery_errors[-1] if recovery_errors else None,
                  retarget_max_position_error_m=float(max_target_error), qpos_teleported_during_execution=False,
                  wall_seconds=time.monotonic()-began, simulation_only=True, inference_pauses_simulation=True,
                  piano_motion="external kinematic translation while hands are clear; not force-driven pushing",
                  audio_source="all physical key events; no soundtrack substitution",
                  g05_role="fresh camera/state observations after translation; arm posture proposals combined with IK",
                  target_source="known simulator geometry", finger_controller="original finger controls; Cartesian-retargeted arm controls")
    write_json(args.out / "report.json", report)
    result = audit(args.out)
    write_json(args.out / "events.json", result["events"])
    report.update({k:result[k] for k in ("onset_metrics","note_count_match","timing_match","chords_match","finger_matches","interhand_contact_steps")})
    report["performance_passed"] = stop is None and performance_passed(result)
    report["audit_final_qpos_error"] = result.get("replay_final_qpos_error")
    write_json(args.out / "report.json", report)
    if any(e["type"] == "NoteOn" for e in result["events"]):
        messages = [midi_message.NoteOn(note=e["midi"],velocity=80,time=e["time_seconds"]) if e["type"]=="NoteOn"
                    else midi_message.NoteOff(note=e["midi"],time=e["time_seconds"]) for e in result["events"]]
        synth = synthesizer.Synthesizer()
        try:
            samples = synth.get_samples(messages)
        finally:
            synth.stop()
        with wave.open(str(args.out/"measured.wav"),"wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(44100); f.writeframes(samples.tobytes())
        subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-i",str(args.out/"silent.mp4"),"-i",str(args.out/"measured.wav"),
                        "-c:v","copy","-c:a","aac","-af","apad","-shortest",str(args.out/"demo.mp4")],check=True)
    imageio.imwrite(args.out/"final.png",p.render(width=960,height=720,camera_id="whole_robot"))
    p.free()
    print(json.dumps({k:report[k] for k in ("playback_completed","safety_stop","onset_metrics","model_calls_during_session","recovery_final_error","audit_final_qpos_error")}),flush=True)
    return 0 if stop is None and result["interhand_contact_steps"] == 0 else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary", type=float, default=18.68)
    parser.add_argument("--piano-shift", type=float, nargs=3, default=[0., .06, 0.])
    parser.add_argument("--lift-height", type=float, default=.055)
    parser.add_argument("--model-weight", type=float, default=.5,
                        help="G0.5 posture cost alongside wrist alignment tasks")
    parser.add_argument("--g05-config-mode", default="checkpoint")
    for name in ("python", "repo", "models", "checkpoint"):
        parser.add_argument("--g05-"+name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
