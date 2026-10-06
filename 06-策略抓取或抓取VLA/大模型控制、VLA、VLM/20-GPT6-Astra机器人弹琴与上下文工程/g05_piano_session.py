"""Execute audited G0.5 preparation, a bounded handoff, and piano controls.

This is offline execution of recorded model-assisted controls, not live policy
inference. The physics state is initialized once and never reset at a boundary.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np

from piano_context import write_json
from polyphonic_score import KeyCycles, performance_passed, validate_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preparation(report, protocol, scene_sha, trajectory_sha):
    for record in (report, protocol):
        shift = np.asarray(record.get("piano_shift_world_m", [0., 0., 0.]), dtype=float)
        if shift.shape != (3,) or not np.isfinite(shift).all() or np.any(shift):
            raise ValueError("Relocated piano preparation needs a matching relocated performance, not original controls")
    if report.get("mode") != "g05_assisted" or report.get("success") is not True or report.get("safety_stop"):
        raise ValueError("A successful, stopped-free G0.5 assisted preparation is required")
    influence = report.get("model_induced_command_distance_rad_seconds", 0.)
    if not report.get("model_calls") or not np.isfinite(influence) or influence <= 1e-8:
        raise ValueError("No recorded model contribution to the executed controls")
    if protocol.get("source_scene_sha256") != scene_sha or protocol.get("source_trajectory_sha256") != trajectory_sha:
        raise ValueError("Preparation belongs to a different source scene or trajectory")
    if protocol.get("control_hz") != 100 or protocol.get("physics_hz") != 500:
        raise ValueError("Expected 100 Hz preparation and 500 Hz physics")


def handoff_controls(start, target, seconds, arm_ids, speed=.6):
    start, target = np.asarray(start), np.asarray(target)
    if (start.ndim != 1 or target.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(target).all()
            or not np.isfinite(seconds) or not 1 <= seconds <= 10
            or not np.isfinite(speed) or speed <= 0):
        raise ValueError("Invalid handoff inputs")
    steps = round(seconds * 100)
    u = np.arange(1, steps + 1) / steps
    ease = 10*u**3 - 15*u**4 + 6*u**5
    controls = start + ease[:, None] * (target-start)
    increments = np.diff(np.vstack([start, controls]), axis=0)
    if np.max(np.abs(increments[:, arm_ids])) > speed * .01 + 1e-9:
        raise ValueError("Handoff exceeds the arm command speed limit; increase duration")
    return controls


def shifted_score(score, seconds):
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid preparation duration")
    result = copy.deepcopy(score)
    for note in result["notes"]:
        note["start_beat"] += seconds * score["tempo_bpm"] / 60
    validate_score(result, song_mode=True)
    return result


def session_exit_code(report, allow_note_errors=False):
    if report.get("performance_passed") is True:
        return 0
    if (allow_note_errors and report.get("playback_completed") is True
            and report.get("safety_stop") is None
            and report.get("interhand_contact_steps") == 0
            and report.get("onset_metrics", {}).get("measured_count", 0) > 0):
        return 0
    return 2


def run(args):
    import imageio.v2 as imageio
    import mujoco
    from dm_control import mujoco as dm_mujoco
    from physical_piano import position_targets
    from pianomime_galaxea import unique_suffix
    from audit_polyphonic_contacts import audit
    from robopianist.music import midi_message, synthesizer

    reference, preparation = args.reference.resolve(), args.preparation.resolve()
    prep_report = json.loads((preparation / "report.json").read_text())
    protocol = json.loads((preparation.parent / "protocol.json").read_text())
    source_report = json.loads((reference / "report.json").read_text())
    manifest = json.loads((reference / "scene_manifest.json").read_text())
    scene = reference / "scene/scene.xml"
    verify_preparation(prep_report, protocol, digest(scene), digest(reference / "trajectory.npz"))
    if source_report.get("performance_passed") is not True and not args.diagnostic_source:
        raise ValueError("Source performance failed; --diagnostic-source is required for a failure investigation")
    if manifest["control_timestep_seconds"] != .02 or manifest["physics_steps_per_control"] != 10:
        raise ValueError("Expected a 50 Hz source performance")
    if manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("Use the source MuJoCo version")
    score_bytes = (reference / "source.plan.json").read_bytes()
    if hashlib.sha256(score_bytes).hexdigest() != source_report["plan_sha256"]:
        raise ValueError("Source score was changed")
    with np.load(preparation / "trajectory.npz", allow_pickle=False) as trace:
        start = trace["initial_qpos"].copy()
        prep_controls = trace["controls"].copy()
        expected_ready = trace["qpos"][-1].copy()
    with np.load(reference / "trajectory.npz", allow_pickle=False) as trace:
        performance_controls = trace["controls"].copy()
        performance_initial = trace["initial_qpos"].copy()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference / "scene", args.out / "scene")
    p = dm_mujoco.Physics.from_xml_path(str(args.out / "scene/scene.xml"))
    writer = None
    try:
        model = p.model.ptr
        if not np.isclose(model.opt.timestep, .002):
            raise ValueError("Expected 500 Hz physics")
        for controls in (prep_controls, performance_controls):
            if (controls.ndim != 2 or controls.shape[1] != model.nu or not len(controls)
                    or not np.isfinite(controls).all()):
                raise ValueError("Invalid source control sequence")
        if any(q.shape != (model.nq,) or not np.isfinite(q).all() for q in (start, expected_ready, performance_initial)):
            raise ValueError("Invalid source states")
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        if any((name or "").startswith("piano/") for name in names):
            raise ValueError("Piano actuators are prohibited")
        arms = [unique_suffix(names, f"{side}_arm_joint{i}_act") for side in ("left", "right") for i in range(1,8)]
        target_data = mujoco.MjData(model)
        target_data.qpos[:] = performance_initial
        mujoco.mj_forward(model, target_data)
        target_control = position_targets(model, target_data)
        handoff = handoff_controls(prep_controls[-1], target_control, args.handoff_seconds, arms)
        # Repeat 50 Hz controls at 100 Hz; this preserves their physical duration.
        combined = np.concatenate([prep_controls, handoff, np.repeat(performance_controls, 2, axis=0)])
        if (combined < model.actuator_ctrlrange[:,0]).any() or (combined > model.actuator_ctrlrange[:,1]).any():
            raise ValueError("Control outside actuator range")
        boundary = len(prep_controls) + len(handoff)
        performance_offset = boundary * .01
        shifted = shifted_score(json.loads(score_bytes), performance_offset)
        write_json(args.out / "source.plan.json", shifted)
        manifest.update(control_timestep_seconds=.01, physics_steps_per_control=5)
        write_json(args.out / "scene_manifest.json", manifest)
        p.data.qpos[:] = start
        p.data.qvel[:] = 0
        p.forward()
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest["key_joints"]]
        if len(joints) != 88 or min(joints) < 0:
            raise ValueError("Expected 88 piano joints")
        sensor = KeyCycles()
        robot = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("galaxea_r1pro/") for i in range(model.ngeom)])

        def contacts():
            return {tuple(sorted((int(c.geom1), int(c.geom2)))): float(c.dist)
                for c in p.data.contact[:p.data.ncon]
                if c.dist < -.001 and (robot[c.geom1] or robot[c.geom2])}

        baseline_contacts = contacts()
        states, executed, stop = [], [], None
        ready_error = handoff_error = None
        writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=args.video_fps, codec="libx264")
        for tick, control in enumerate(combined):
            for _ in range(5):
                p.set_control(control)
                p.step()
                if not np.isfinite(p.data.qpos).all() or not np.isfinite(p.data.qvel).all():
                    raise ValueError("Non-finite physical state")
                ranges = model.jnt_range[joints]
                changes = sensor.update((p.data.qpos[model.jnt_qposadr[joints]]-ranges[:,0])/(ranges[:,1]-ranges[:,0]))
                if tick < boundary:
                    if any(active for _, active in changes):
                        stop = "key_pressed_before_performance"
                    if any(depth < baseline_contacts.get(pair, 0.)-.001 for pair, depth in contacts().items()):
                        stop = "new_or_deeper_preparation_contact"
            executed.append(control.copy())
            states.append(p.data.qpos.copy())
            if (tick + 1) % (100 // args.video_fps) == 0:
                writer.append_data(p.render(width=960, height=720, camera_id="whole_robot"))
            if tick % 2000 == 0:
                print(f"session_playback {tick*.01:.1f}/{len(combined)*.01:.1f}s", flush=True)
            if tick + 1 == len(prep_controls):
                ready_error = float(np.max(np.abs(p.data.qpos-expected_ready)))
                if ready_error > 1e-5:
                    stop = "preparation_replay_diverged"
            if tick + 1 == boundary:
                handoff_error = float(np.max(np.abs(p.data.actuator_length[arms]-target_data.actuator_length[arms])))
                if handoff_error > .03:
                    stop = "handoff_arm_error_above_0.03rad"
            if stop:
                break
        writer.close()
        writer = None
        np.savez_compressed(args.out / "trajectory.npz", initial_qpos=start, initial_qvel=np.zeros(model.nv),
            controls=executed, qpos=states)
        report = dict(mode="offline_g05_preparation_handoff_performance", song_mode=True,
            plan_sha256=digest(args.out / "source.plan.json"), original_plan_sha256=source_report["plan_sha256"],
            g05_preparation_report_sha256=digest(preparation / "report.json"),
            g05_preparation_trajectory_sha256=digest(preparation / "trajectory.npz"),
            model_call_indices=prep_report["model_calls"], model_calls_during_session=0,
            model_induced_command_distance_rad_seconds=prep_report["model_induced_command_distance_rad_seconds"],
            source_performance_passed=source_report.get("performance_passed"),
            diagnostic_source=args.diagnostic_source, preparation_seconds=len(prep_controls)*.01,
            handoff_seconds=len(handoff)*.01, performance_start_seconds=performance_offset,
            preparation_replay_max_qpos_error=ready_error, handoff_max_arm_error_rad=handoff_error,
            handoff_passed=stop is None and len(executed) >= boundary,
            playback_completed=stop is None and len(executed) == len(combined),
            executed_seconds=len(executed)*.01, planned_seconds=len(combined)*.01,
            allow_note_errors=args.allow_note_errors, video_fps=args.video_fps,
            safety_stop=stop, qpos_teleported_during_execution=False, simulation_only=True,
            audio_source="all independently replayed physical key events; no soundtrack substitution",
            g05_role="recorded real model-assisted arm preparation; not finger control",
            limitations=["offline action execution", "fixed chassis", "no physical pedal", "fixed playback velocity", "no new Astra invocation"])
        write_json(args.out / "report.json", report)
        result = audit(args.out)
        report.update({k:result[k] for k in ("onset_metrics", "note_count_match", "timing_match", "chords_match", "finger_matches", "interhand_contact_steps")})
        report["performance_passed"] = stop is None and performance_passed(result)
        write_json(args.out / "events.json", result["events"])
        messages = [midi_message.NoteOn(note=e["midi"], velocity=80, time=e["time_seconds"])
            if e["type"] == "NoteOn" else midi_message.NoteOff(note=e["midi"], time=e["time_seconds"]) for e in result["events"]]
        if any(e["type"] == "NoteOn" for e in result["events"]):
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
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(args.out/"silent.mp4"),
                "-i", str(args.out/"measured.wav"), "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-shortest", str(args.out/"demo.mp4")], check=True)
        imageio.imwrite(args.out / "final.png", p.render(width=960, height=720, camera_id="whole_robot"))
        write_json(args.out / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return session_exit_code(report, args.allow_note_errors)
    finally:
        if writer is not None:
            writer.close()
        p.free()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--handoff-seconds", type=float, default=2.)
    parser.add_argument("--diagnostic-source", action="store_true")
    parser.add_argument("--video-fps", type=int, choices=(10, 20), default=20)
    parser.add_argument("--allow-note-errors", action="store_true",
                        help="Allow musical mismatches after a full playback, without overriding stops or interhand contacts")
    raise SystemExit(run(parser.parse_args()))
