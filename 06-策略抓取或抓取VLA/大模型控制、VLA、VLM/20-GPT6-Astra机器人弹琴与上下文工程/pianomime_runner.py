"""Reproduce PianoMime's two-stage inference without editing its source checkout."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
import wave


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def score_frames(score, note_factory, dt=.05):
    import math
    from polyphonic_score import validate_score, FINGERS
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Score frame period must be positive and finite")
    validation = validate_score(score, song_mode=True)
    beat = 60 / score["tempo_bpm"]
    frames = []
    for step in range(math.ceil((validation["duration_seconds"] + 1.5) / dt)):
        t = step * dt - 1.
        frames.append([note_factory(n["midi"], 80,
            FINGERS.index(n["finger"]) + (5 if n["hand"] == "left" else 0))
            for n in score["notes"] if n["start_beat"] * beat <= t <
            (n["start_beat"] + n["duration_beats"]) * beat])
    return frames


def corpus_stats(repo):
    import numpy as np
    import zarr

    result = {}
    for level in ("hl", "ll"):
        group = zarr.open(str(repo / f"dataset_{level}.zarr"), mode="r")
        result[level] = {}
        for key, source in (("obs", "state"), ("action", "action")):
            array = group["data"][source]
            low = np.full(array.shape[1], np.inf, dtype=np.float32)
            high = -low.copy()
            for start in range(0, len(array), 32768):
                block = array[start:start + 32768]
                if not np.isfinite(block).all():
                    raise ValueError("Non-finite official normalization corpus")
                low = np.minimum(low, block.min(axis=0))
                high = np.maximum(high, block.max(axis=0))
            result[level][key] = {"min": low, "max": high}
    return result


def main(args):
    if not 1 <= args.steps <= 12000 or not args.task.replace("_", "").isalnum():
        raise ValueError("Use a simple task name and 1..12000 evaluation steps")
    score_path = args.score.resolve() if args.score else None
    args.repo, args.models, args.dataset, args.out = [p.resolve() for p in
        (args.repo, args.models, args.dataset, args.out)]
    args.out.mkdir(parents=True, exist_ok=False)
    work = args.out / "work"
    trajectories = work / "pianomime/multi_task/trajectories"
    trajectories.mkdir(parents=True)
    if score_path is None:
        (work / "dataset").symlink_to(args.dataset, target_is_directory=True)
    else:
        (work / "dataset/notes").mkdir(parents=True)
    os.chdir(work)
    sys.path.insert(0, str(args.repo))

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import torch
    from dm_control import mjcf
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from goal_auto_encoder.network import Autoencoder
    from multi_task.network import ConditionalUnet1D, ConvEncoder, VariationalConvMlpEncoder
    from multi_task.dataset import normalize_data, unnormalize_data
    from multi_task.utils import get_env_hl, get_env_ll, get_flattend_obs, adjust_ft_fingering
    if score_path is not None:
        from robopianist.music.midi_file import NoteTrajectory, PianoNote
        score = json.loads(score_path.read_text(encoding="utf-8"))
        frames = score_frames(score, PianoNote.create)
        # Serialize only our validated data; never load an external user pickle.
        with (work / "dataset/notes" / (args.task + ".pkl")).open("wb") as stream:
            pickle.dump(NoteTrajectory(dt=.05, notes=frames, sustains=[0] * len(frames)), stream)
        (args.out / "source.plan.json").write_bytes(score_path.read_bytes())

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(0.20)
    device = "cuda"
    stats = corpus_stats(args.repo)
    np.savez_compressed(args.out / "normalization.npz", **{
        f"{level}_{key}_{bound}": values for level, group in stats.items()
        for key, limits in group.items() for bound, values in limits.items()})
    checkpoints = {}

    def load(model, filename):
        path = args.models / filename
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        checkpoints[filename] = digest(path)
        # Match the original evaluation scripts, including their BatchNorm behavior.
        return model

    ae = load(Autoencoder(latent_dim=16, cond_dim=64).to(device), "checkpoint_ae.ckpt")
    high = load(ConditionalUnet1D(input_dim=36, global_cond_dim=212, midi_dim=212,
        midi_cond_dim=36, midi_encoder=lambda: VariationalConvMlpEncoder(in_channels=16,
        mid_channels=32, out_channels=64, latent_dim=32, noise=0.08).to(device)).to(device),
        "checkpoint_high_level.ckpt")
    low = load(ConditionalUnet1D(input_dim=46, global_cond_dim=404, midi_dim=208,
        midi_cond_dim=0, midi_encoder=lambda: ConvEncoder(in_channels=52, mid_channels=64,
        out_channels=128, horizon=4, noise_fingering=0, noise_ft=0).to(device),
        freeze_encoder=False).to(device), "checkpoint_low_level.ckpt")
    print("Three official models loaded", flush=True)

    def sample(model, obs, dimensions, iterations):
        scheduler = DDPMScheduler(num_train_timesteps=iterations,
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
        scheduler.set_timesteps(iterations)
        values = torch.randn((1, 4, dimensions), device=device)
        conditioning = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(1, -1)
        for tick in scheduler.timesteps:
            noise = model(sample=values, timestep=tick, global_cond=conditioning)
            values = scheduler.step(model_output=noise, timestep=tick, sample=values).prev_sample
        result = values[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Non-finite diffusion prediction")
        return result

    started = time.perf_counter()
    env, total = get_env_hl(args.task, lookahead=10)
    left, right = [], []
    try:
        ts = env.reset()
        lh, rh = env.task.get_fingertip_pos(env.physics)
        previous = np.concatenate((lh, rh)).flatten()
        last_keys = last_lh = last_rh = last_fingering = None
        with torch.no_grad():
            for step in range(min(total, args.steps + 11)):
                goal = get_flattend_obs(ts, lookahead=10,
                    exclude_keys=["fingering", "hand", "demo", "prior_action", "q_piano"],
                    encoder=ae.encoder, sampling=False)
                obs = normalize_data(np.concatenate((goal, previous)), stats["hl"]["obs"])
                action = sample(high, obs, 36, 100)
                action = np.concatenate((action, np.zeros((4, 10))), axis=1).flatten()
                action = unnormalize_data(action, stats["hl"]["action"]).reshape(4, -1)
                keys = np.nonzero(ts.observation["goal"][:88])
                lh, rh, fingering = adjust_ft_fingering(env, keys,
                    action[0, :18].reshape(6, 3).T, action[0, 18:36].reshape(6, 3).T,
                    last_keys, last_lh, last_rh, last_fingering)
                left.append(lh.copy())
                right.append(rh.copy())
                previous = np.concatenate((lh.T.flatten(), rh.T.flatten()))
                last_keys, last_lh, last_rh, last_fingering = keys, lh, rh, fingering
                ts = env.step(np.zeros(47))
                if step % 20 == 0:
                    print(f"High-level {step + 1}/{min(total, args.steps + 11)}", flush=True)
                if ts.last():
                    break
    finally:
        env.close()
    for side, values in (("left", left), ("right", right)):
        # Upstream requires full-song array lengths even for a prefix evaluation.
        # Only generated frames (including the lookahead) are used by this run.
        padded = np.asarray(values)
        if len(padded) < total:
            padded = np.concatenate((padded, np.repeat(padded[-1:], total - len(padded), axis=0)))
        np.save(trajectories / f"{args.task}_{side}_hand_action_list.npy", padded)

    env = get_env_ll(args.task, enable_ik=False, lookahead=10,
        record_dir=None, use_fingering_emb=False, use_midi=False)
    writer = None
    actions, states, desired, measured, palms = [], [], [], [], []
    actuator_controls, sustains, physics_times, physics_states = [], [], [], []
    try:
        ts = env.reset()
        physics = env.physics
        original_substep = env.task.after_substep

        def record_substep(current_physics, random_state):
            original_substep(current_physics, random_state)
            physics_times.append(float(current_physics.data.time))
            physics_states.append(current_physics.data.qpos.copy())

        env.task.after_substep = record_substep
        initial_qpos, initial_qvel = physics.data.qpos.copy(), physics.data.qvel.copy()
        manifest = {"piano_root": env.task.piano.root_body.full_identifier,
            "actuators": [mujoco.mj_id2name(physics.model.ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(physics.model.nu)],
            "hand_joints": {side: [j.full_identifier for j in getattr(env.task, side + "_hand").joints]
                            for side in ("left", "right")}}
        (args.out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        mjcf.export_with_assets(env.task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        if args.video:
            writer = imageio.get_writer(str(args.out / "silent.mp4"), fps=20, codec="libx264")
        with torch.no_grad():
            while len(actions) < args.steps and not ts.last():
                obs = get_flattend_obs(ts, lookahead=3, exclude_keys=["fingering", "prior_action"],
                    encoder=ae.encoder, sampling=False, concatenate_keys=["goal", "demo"])
                action_chunk = sample(low, normalize_data(obs, stats["ll"]["obs"]), 46, 50)
                for action in action_chunk:
                    target = unnormalize_data(action, stats["ll"]["action"])
                    control = np.append(target, 0)
                    wanted = ts.observation["goal"][:88].astype(bool)
                    ts = env.step(control)
                    if not np.isfinite(physics.data.qpos).all():
                        raise ValueError("Non-finite simulated state")
                    actions.append(control.copy())
                    actuator_controls.append(physics.data.ctrl.copy())
                    sustains.append(float(env.task.piano.sustain_activation[0]))
                    states.append(physics.data.qpos.copy())
                    desired.append(wanted)
                    measured.append(env.task.piano.activation.copy())
                    row = {}
                    for side in ("left", "right"):
                        hand = getattr(env.task, side + "_hand")
                        prefix = "lh_" if side == "left" else "rh_"
                        palm = hand.mjcf_model.find("body", prefix + "palm")
                        row[side] = {"position": physics.bind(palm).xpos.tolist(),
                                     "rotation": physics.bind(palm).xmat.reshape(3, 3).tolist()}
                    palms.append(row)
                    if writer is not None:
                        pixels = physics.render(height=480, width=640, camera_id="piano/back")
                        writer.append_data(pixels)
                        if len(actions) in (1, 20, 60, 100):
                            imageio.imwrite(args.out / f"frame-{len(actions):04d}.png", pixels)
                    if len(actions) % 20 == 0:
                        print(f"Low-level {len(actions)}/{args.steps}", flush=True)
                    if ts.last() or len(actions) >= args.steps:
                        break
        if writer is not None:
            writer.close()
            writer = None
        np.savez_compressed(args.out / "trajectory.npz", actions=actions, qpos=states,
            initial_qpos=initial_qpos, initial_qvel=initial_qvel, expected=desired, measured=measured,
            actuator_controls=actuator_controls, sustain=sustains,
            physics_times=physics_times, physics_qpos=physics_states)
        (args.out / "palms.json").write_text(json.dumps(palms), encoding="utf-8")
        expected, actual = np.asarray(desired), np.asarray(measured)
        tp = int((expected & actual).sum())
        fp, fn = int((~expected & actual).sum()), int((expected & ~actual).sum())
        report = {"mode": "official_pianomime_two_stage_baseline", "task": args.task,
            "custom_score_sha256": digest(score_path) if score_path else None,
            "source_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint_sha256": checkpoints, "normalization_sha256": digest(args.out / "normalization.npz"),
            "steps": len(actions), "control_hz": 20, "episode_complete": bool(ts.last()),
            "high_level_generated_frames": len(left), "source_total_frames": total,
            "micro_precision": tp / max(1, tp + fp), "micro_recall": tp / max(1, tp + fn),
            "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_positive_key_frames": fp,
            "false_negative_key_frames": fn, "true_positive_key_frames": tp,
            "hand_collisions_disabled_upstream": True, "robot": "upstream floating Shadow hands",
            "g05_calls": 0, "astra_calls": 0, "model_mode": "upstream default train mode; no gradients or optimizer",
            "elapsed_seconds": time.perf_counter() - started, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
            "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__}}
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        messages = env.task.piano.midi_module.get_all_midi_messages()
        (args.out / "events.json").write_text(json.dumps([
            {"type": type(e).__name__, "time": float(e.time),
             "note": int(e.note) if hasattr(e, "note") else None}
            for e in messages], indent=2), encoding="utf-8")
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
                str(args.out / "demo.mp4")], check=True)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="Adieu_0")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--score", type=Path, help="Validated user-authorized polyphonic score")
    main(parser.parse_args())
