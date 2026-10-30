#!/usr/bin/env python3
"""Scripted oracle collection for randomized red-mug pick-and-place.

The script first drives the environment with end-effector delta commands, then
stores the resulting joint-angle actions in the same format as the teleop
collector. This keeps generated episodes replayable by the existing ACT data
audit and training code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image


os.environ.setdefault("SUCCESS_REQUIRE_EE_HEIGHT", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TASK_NAME = "Pick up the red mug and place it on the white plate."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1013,1017,1022,1024")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/Data14TB/datasets/every_embodied/omy_pnp_scripted_oracle_v0"),
    )
    parser.add_argument("--repo-id", default="datawhale_eai_pnp_scripted_oracle")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/data/Data14TB/videos/every_embodied/pnp_scripted_oracle_v0"),
    )
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-failed", action="store_true")
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--physics-steps-per-action",
        type=int,
        default=25,
        help="MuJoCo substeps per recorded control action; 25 matches 500Hz physics / 20Hz control.",
    )
    parser.add_argument("--move-step", type=float, default=0.004)
    parser.add_argument(
        "--record-action-source",
        choices=["commanded", "observed"],
        default="commanded",
        help="Store commanded IK joints or observed joint state as the supervised action.",
    )
    parser.add_argument("--place-step-scale", type=float, default=0.65)
    parser.add_argument("--pos-tol", type=float, default=0.008)
    parser.add_argument("--max-move-steps", type=int, default=260)
    parser.add_argument("--max-total-actions", type=int, default=900)
    parser.add_argument("--initial-hold", type=int, default=12)
    parser.add_argument("--close-hold", type=int, default=64)
    parser.add_argument("--pre-release-hold", type=int, default=12)
    parser.add_argument("--open-hold", type=int, default=70)
    parser.add_argument("--final-hold", type=int, default=35)
    parser.add_argument("--success-hold-frames", type=int, default=6)
    parser.add_argument("--grasp-offset-x", type=float, default=0.006)
    parser.add_argument("--grasp-offset-y", type=float, default=0.086)
    parser.add_argument("--grasp-offset-z", type=float, default=0.026)
    parser.add_argument("--pregrasp-z", type=float, default=0.135)
    parser.add_argument("--lift-z", type=float, default=0.145)
    parser.add_argument("--release-offset-x", type=float, default=0.008)
    parser.add_argument("--release-offset-y", type=float, default=0.026)
    parser.add_argument("--release-offset-z", type=float, default=0.095)
    parser.add_argument("--retreat-z", type=float, default=0.145)
    parser.add_argument("--physical-min-lift", type=float, default=0.06)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.85)
    parser.add_argument(
        "--policy-prefix-path",
        type=Path,
        default=None,
        help="Optional ACT checkpoint. If set, roll this policy for --policy-prefix-steps before scripted recovery.",
    )
    parser.add_argument("--policy-prefix-repo-id", default=None)
    parser.add_argument("--policy-prefix-dataset-root", type=Path, default=None)
    parser.add_argument("--policy-prefix-device", default="cuda")
    parser.add_argument("--policy-prefix-act-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-prefix-act-clamp-timestamp", action="store_true")
    parser.add_argument(
        "--policy-prefix-act-episode-timestamp-offsets",
        default="",
        help="Episode timestamp offsets used by the ACT prefix checkpoint, e.g. '49-71:2.0'.",
    )
    parser.add_argument(
        "--policy-prefix-act-episode-source-flags",
        default="",
        help="Episode source flags used by the ACT prefix checkpoint, e.g. '49-71:1.0'.",
    )
    parser.add_argument("--policy-prefix-steps", type=int, default=0)
    parser.add_argument("--policy-prefix-hz", type=float, default=20.0)
    parser.add_argument("--policy-prefix-task", default="Put mug cup on the plate")
    return parser.parse_args()


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed_start is not None and args.num_demos is not None:
        return [int(args.seed_start + i) for i in range(args.num_demos)]
    seeds: list[int] = []
    for part in str(args.seeds).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if args.num_demos is not None:
        seeds = seeds[: int(args.num_demos)]
    if not seeds:
        raise ValueError("No seeds requested")
    return seeds


def dataset_features() -> dict:
    return {
        "observation.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "obj_init": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["obj_init"],
        },
    }


def make_dataset(args: argparse.Namespace):
    if not args.write_dataset:
        return None
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.root.exists() and args.overwrite:
        shutil.rmtree(args.root)
    if args.root.exists() and not args.resume:
        raise SystemExit(f"Dataset root exists. Use --resume or --overwrite: {args.root}")
    if args.root.exists():
        return LeRobotDataset(args.repo_id, root=args.root)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        robot_type="omy",
        fps=args.fps,
        features=dataset_features(),
        image_writer_threads=10,
        image_writer_processes=5,
    )


def resize_rgb(image: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((256, 256)))


def round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(values).reshape(-1)]


def make_writer(path: Path, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.video_fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def draw_overlay(frame: np.ndarray, seed: int, action_step: int, phase: str, debug: dict) -> np.ndarray:
    out = frame.copy()
    physical_success = bool(debug.get("physical_success", False))
    lines = [
        f"scripted oracle seed={seed} | step {action_step} | {phase}",
        (
            f"success={bool(debug.get('success', False))} "
            f"physical_success={physical_success} "
            f"lift={float(debug.get('max_mug_lift', 0.0)):.3f} "
            f"upright={float(debug.get('final_mug_upright_cos', 0.0)):.3f}"
        ),
        (
            f"xy={float(debug.get('xy_dist', 0.0)):.3f} "
            f"z={float(debug.get('z_dist', 0.0)):.3f} "
            f"ee_z={float(debug.get('ee_z', 0.0)):.3f} "
            f"grip_open={float(debug.get('gripper_open_norm', 0.0)):.3f}"
        ),
    ]
    pad = 10
    line_h = 28
    cv2.rectangle(out, (0, 0), (out.shape[1], pad * 2 + line_h * len(lines)), (0, 0, 0), -1)
    color = (130, 255, 130) if physical_success else (255, 255, 255)
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (pad, pad + 20 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def write_video_frame(writer, image: np.ndarray, seed: int, step: int, phase: str, debug: dict, args: argparse.Namespace) -> None:
    frame = cv2.resize(np.asarray(image), (args.width, args.height), interpolation=cv2.INTER_AREA)
    frame = draw_overlay(frame, seed, step, phase, debug)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def clip_delta(delta: np.ndarray, step: float) -> np.ndarray:
    return np.clip(delta, -float(step), float(step)).astype(np.float32)


def target_positions(env, args: argparse.Namespace) -> dict[str, np.ndarray]:
    mug, plate = env.get_obj_pose()
    mug = np.asarray(mug, dtype=np.float32)
    plate = np.asarray(plate, dtype=np.float32)
    grasp = mug + np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float32)
    pregrasp = grasp.copy()
    pregrasp[2] = mug[2] + float(args.pregrasp_z)
    lift = grasp.copy()
    lift[2] = mug[2] + float(args.lift_z)
    preplace = plate + np.array([args.release_offset_x, args.release_offset_y, args.lift_z], dtype=np.float32)
    release = plate + np.array([args.release_offset_x, args.release_offset_y, args.release_offset_z], dtype=np.float32)
    retreat = release.copy()
    retreat[2] = plate[2] + float(args.retreat_z)
    return {
        "mug": mug,
        "plate": plate,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
        "preplace": preplace,
        "release": release,
        "retreat": retreat,
    }


def control_tick(
    env,
    dataset,
    writer,
    seed: int,
    phase: str,
    action: np.ndarray,
    state: dict,
    args: argparse.Namespace,
):
    from eval_policy_success import get_act_physical_debug, update_act_physical_tracker

    for _ in range(max(int(args.physics_steps_per_action), 1)):
        env.step_env()
        state["sim_steps"] += 1
        update_act_physical_tracker(env, state["tracker"], state["physical_args"])
    debug = env.get_success_debug() if hasattr(env, "get_success_debug") else {"success": bool(env.check_success())}
    debug = get_act_physical_debug(env, state["tracker"], state["physical_args"], debug)
    state["last_debug"] = debug
    if env.check_success():
        state["success_hold"] += 1
    else:
        state["success_hold"] = 0
    if debug["physical_success"]:
        state["physical_success_ever"] = True
        if state["first_physical_success_step"] is None:
            state["first_physical_success_step"] = state["action_steps"]

    ee_pose = env.get_ee_pose()
    agent_image, wrist_image = env.grab_image()
    if writer is not None and (state["action_steps"] % max(int(args.frame_every), 1) == 0):
        write_video_frame(writer, agent_image, seed, state["action_steps"], phase, debug, args)
        state["video_frames"] += 1

    joint_q = np.asarray(env.step(action), dtype=np.float32)
    if args.record_action_source == "commanded":
        arm_q = np.asarray(getattr(env, "compute_q", joint_q[:-1]), dtype=np.float32).reshape(-1)
        if arm_q.shape[0] != 6:
            raise ValueError(f"Expected 6-D commanded arm action, got shape {arm_q.shape}")
        recorded_action = np.concatenate([arm_q, np.asarray([float(action[-1])], dtype=np.float32)])
    else:
        if joint_q.shape[0] != 7:
            raise ValueError(f"Expected 7-D observed action, got shape {joint_q.shape}")
        recorded_action = joint_q.copy()
    recorded_action[-1] = float(action[-1])
    state["actions"].append(recorded_action.copy())
    state["phases"].append(phase)

    if dataset is not None:
        dataset.add_frame(
            {
                "observation.image": resize_rgb(agent_image),
                "observation.wrist_image": resize_rgb(wrist_image),
                "observation.state": np.asarray(ee_pose, dtype=np.float32),
                "action": recorded_action,
                "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
            },
            task=args.task,
        )

    state["action_steps"] += 1
    if state["action_steps"] > args.max_total_actions:
        raise RuntimeError(f"Exceeded --max-total-actions={args.max_total_actions}")


def hold(env, dataset, writer, seed: int, phase: str, gripper: float, steps: int, state: dict, args: argparse.Namespace) -> None:
    action = np.zeros(7, dtype=np.float32)
    action[-1] = float(gripper)
    for _ in range(max(int(steps), 0)):
        control_tick(env, dataset, writer, seed, phase, action, state, args)


def move_to(
    env,
    dataset,
    writer,
    seed: int,
    phase: str,
    target: np.ndarray,
    gripper: float,
    state: dict,
    args: argparse.Namespace,
    step_scale: float = 1.0,
) -> None:
    target = np.asarray(target, dtype=np.float32)
    for _ in range(int(args.max_move_steps)):
        current = np.asarray(env.get_ee_pose()[:3], dtype=np.float32)
        delta = target - current
        if float(np.linalg.norm(delta)) <= float(args.pos_tol):
            return
        action = np.zeros(7, dtype=np.float32)
        action[:3] = clip_delta(delta, float(args.move_step) * float(step_scale))
        action[-1] = float(gripper)
        control_tick(env, dataset, writer, seed, phase, action, state, args)
    raise RuntimeError(f"Failed to reach target for phase={phase}: target={round_list(target)}")


def make_policy_prefix(args: argparse.Namespace):
    if args.policy_prefix_path is None or int(args.policy_prefix_steps) <= 0:
        return None
    if args.policy_prefix_repo_id is None:
        raise ValueError("--policy-prefix-repo-id is required when --policy-prefix-path is set")
    if args.policy_prefix_dataset_root is None:
        raise ValueError("--policy-prefix-dataset-root is required when --policy-prefix-path is set")
    import torch
    from eval_policy_success import make_act_policy

    if args.policy_prefix_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Policy prefix requested cuda, but torch.cuda.is_available() is false")
    policy = make_act_policy(
        args.policy_prefix_device,
        args.policy_prefix_path,
        args.policy_prefix_repo_id,
        args.policy_prefix_dataset_root,
        n_action_steps=args.policy_prefix_act_n_action_steps,
        episode_timestamp_offsets=args.policy_prefix_act_episode_timestamp_offsets,
        episode_source_flags=args.policy_prefix_act_episode_source_flags,
    )
    policy.clamp_timestamp_to_dataset = bool(args.policy_prefix_act_clamp_timestamp)
    return policy


def run_policy_prefix(env, policy, seed: int, args: argparse.Namespace) -> dict:
    """Roll ACT into its own visited state, then hand control back to eef oracle."""
    import torch
    from eval_policy_success import get_act_state, make_timestamp_env_feature, to_tensor_image

    if policy is None or int(args.policy_prefix_steps) <= 0:
        return {"enabled": False, "steps": 0}

    old_action_type = env.action_type
    env.action_type = "joint_angle"
    policy.reset()
    rows: list[dict] = []
    try:
        for action_step in range(int(args.policy_prefix_steps)):
            for _ in range(max(int(args.physics_steps_per_action), 1)):
                env.step_env()
            batch = {
                "task": [args.policy_prefix_task],
                "timestamp": torch.tensor(
                    [action_step / float(args.policy_prefix_hz)],
                    dtype=torch.float32,
                    device=args.policy_prefix_device,
                ),
            }
            state = None
            if "observation.state" in policy.config.input_features:
                state = get_act_state(env, policy, action_step, float(args.policy_prefix_hz))
                batch["observation.state"] = torch.tensor(
                    np.asarray([state]), dtype=torch.float32, device=args.policy_prefix_device
                )
            if "observation.environment_state" in policy.config.input_features:
                env_state = make_timestamp_env_feature(
                    action_step / float(args.policy_prefix_hz),
                    getattr(policy, "timestamp_env_max_timestamp", action_step / float(args.policy_prefix_hz)),
                    getattr(policy, "timestamp_env_frequencies", 0),
                )
                batch["observation.environment_state"] = torch.tensor(
                    env_state[None, :], dtype=torch.float32, device=args.policy_prefix_device
                )
            if "observation.state" not in batch:
                batch["observation.state"] = torch.zeros((1, 1), dtype=torch.float32, device=args.policy_prefix_device)
            if policy.config.image_features:
                image, wrist_image = env.grab_image()
                if "observation.image" in policy.config.input_features:
                    batch["observation.image"] = to_tensor_image(image).unsqueeze(0).to(args.policy_prefix_device)
                if "observation.wrist_image" in policy.config.input_features:
                    batch["observation.wrist_image"] = to_tensor_image(wrist_image).unsqueeze(0).to(args.policy_prefix_device)
            with torch.no_grad():
                action = policy.select_action(batch)[0].detach().cpu().numpy().astype(np.float32)
            env.step(action)
            if action_step < 5 or action_step == int(args.policy_prefix_steps) - 1:
                rows.append(
                    {
                        "step": int(action_step),
                        "action": round_list(action),
                        "tcp_pos": round_list(env.env.get_p_body(env.ee_body_name)),
                        "mug_pos": round_list(env.env.get_p_body("body_obj_mug_5")),
                        "state": round_list(state) if state is not None else None,
                    }
                )
    finally:
        env.action_type = old_action_type
        env.p0, env.R0 = env.env.get_pR_body(body_name=env.ee_body_name)

    return {
        "enabled": True,
        "policy_path": str(args.policy_prefix_path),
        "steps": int(args.policy_prefix_steps),
        "act_clamp_timestamp": bool(args.policy_prefix_act_clamp_timestamp),
        "act_episode_timestamp_offsets": args.policy_prefix_act_episode_timestamp_offsets,
        "act_episode_source_flags": args.policy_prefix_act_episode_source_flags,
        "sampled_steps": rows,
        "final_tcp_pos": round_list(env.env.get_p_body(env.ee_body_name)),
        "final_mug_pos": round_list(env.env.get_p_body("body_obj_mug_5")),
    }


def run_episode(env, dataset, seed: int, args: argparse.Namespace, policy_prefix=None) -> dict:
    from eval_policy_success import get_act_physical_debug, init_act_physical_tracker

    env.reset(seed=seed)
    prefix_info = run_policy_prefix(env, policy_prefix, seed, args)
    points = target_positions(env, args)
    physical_args = SimpleNamespace(
        physical_min_lift=args.physical_min_lift,
        physical_min_lift_steps=args.physical_min_lift_steps,
        physical_final_upright_cos=args.physical_final_upright_cos,
    )
    tracker = init_act_physical_tracker(env)
    debug = env.get_success_debug() if hasattr(env, "get_success_debug") else {"success": bool(env.check_success())}
    debug = get_act_physical_debug(env, tracker, physical_args, debug)
    state = {
        "tracker": tracker,
        "physical_args": physical_args,
        "action_steps": 0,
        "sim_steps": 0,
        "video_frames": 0,
        "success_hold": 0,
        "physical_success_ever": False,
        "first_physical_success_step": None,
        "last_debug": debug,
        "actions": [],
        "phases": [],
    }

    args.video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = args.video_dir / f"scripted_oracle_seed{seed}.mp4"
    writer = make_writer(mp4_path, args)
    try:
        hold(env, dataset, writer, seed, "initial_open_hold", 0.0, args.initial_hold, state, args)
        move_to(env, dataset, writer, seed, "move_pregrasp", points["pregrasp"], 0.0, state, args)
        move_to(env, dataset, writer, seed, "move_grasp", points["grasp"], 0.0, state, args)
        hold(env, dataset, writer, seed, "close_gripper", 1.0, args.close_hold, state, args)
        move_to(env, dataset, writer, seed, "lift_mug", points["lift"], 1.0, state, args)
        move_to(env, dataset, writer, seed, "move_preplace", points["preplace"], 1.0, state, args)
        move_to(
            env,
            dataset,
            writer,
            seed,
            "lower_to_plate",
            points["release"],
            1.0,
            state,
            args,
            step_scale=args.place_step_scale,
        )
        hold(env, dataset, writer, seed, "pre_release_hold", 1.0, args.pre_release_hold, state, args)
        hold(env, dataset, writer, seed, "open_gripper", 0.0, args.open_hold, state, args)
        move_to(env, dataset, writer, seed, "retreat", points["retreat"], 0.0, state, args)
        hold(env, dataset, writer, seed, "final_open_hold", 0.0, args.final_hold, state, args)
    finally:
        writer.release()

    actions = np.stack(state["actions"]).astype(np.float32) if state["actions"] else np.zeros((0, 7), dtype=np.float32)
    npz_path = args.video_dir / f"scripted_oracle_seed{seed}_actions.npz"
    np.savez_compressed(npz_path, recorded=actions, phases=np.asarray(state["phases"], dtype=object))
    debug = state["last_debug"]
    physical_success = bool(state["physical_success_ever"] or debug.get("physical_success", False))
    success = bool(env.check_success() or debug.get("success", False))
    row = {
        "seed": seed,
        "success": success,
        "physical_success": physical_success,
        "action_steps": int(state["action_steps"]),
        "sim_steps": int(state["sim_steps"]),
        "video_frames": int(state["video_frames"]),
        "first_physical_success_step": state["first_physical_success_step"],
        "obj_init": round_list(env.obj_init_pose),
        "targets": {key: round_list(value) for key, value in points.items()},
        "policy_prefix": prefix_info,
        "debug": debug,
        "video": str(mp4_path),
        "actions_npz": str(npz_path),
        "record_action_source": args.record_action_source,
    }
    return row


def episode_count(root: Path) -> int:
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    from mujoco_env.y_env import SimpleEnv

    args = parse_args()
    seeds = parse_seeds(args)
    dataset = make_dataset(args)
    manifest_path = args.root / "collection_manifest.jsonl"
    if args.write_dataset:
        args.root.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.video_dir / "scripted_oracle_summary.jsonl"
    if summary_path.exists() and args.overwrite:
        summary_path.unlink()

    env = SimpleEnv("./asset/example_scene_y.xml", action_type="eef_pose")
    policy_prefix = make_policy_prefix(args)
    rows: list[dict] = []
    try:
        for seed in seeds:
            started_episode_count = episode_count(args.root) if args.write_dataset else None
            start = time.time()
            try:
                row = run_episode(env, dataset, seed, args, policy_prefix=policy_prefix)
                row["elapsed_s"] = round(time.time() - start, 3)
                save_episode = bool(args.write_dataset and (row["physical_success"] or args.save_failed))
                row["dataset_saved"] = save_episode
                if save_episode:
                    dataset.save_episode()
                    row["episode_index"] = started_episode_count
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "episode_in_run": started_episode_count,
                                    "seed": seed,
                                    "frames": row["action_steps"],
                                    "obj_init": row["obj_init"],
                                    "success_debug": row["debug"],
                                    "source": "scripted_oracle",
                                    "video": row["video"],
                                    "actions_npz": row["actions_npz"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                elif dataset is not None:
                    dataset.clear_episode_buffer()
                rows.append(row)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
            except Exception as exc:
                if dataset is not None:
                    dataset.clear_episode_buffer()
                row = {
                    "seed": seed,
                    "error": repr(exc),
                    "physical_success": False,
                    "dataset_saved": False,
                    "elapsed_s": round(time.time() - start, 3),
                }
                rows.append(row)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)

    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass

    summary = {
        "seeds": seeds,
        "episodes": len(rows),
        "physical_success_count": sum(1 for row in rows if row.get("physical_success", False)),
        "dataset_saved_count": sum(1 for row in rows if row.get("dataset_saved", False)),
        "root": str(args.root),
        "video_dir": str(args.video_dir),
        "summary_jsonl": str(summary_path),
    }
    summary_json = args.video_dir / "scripted_oracle_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
