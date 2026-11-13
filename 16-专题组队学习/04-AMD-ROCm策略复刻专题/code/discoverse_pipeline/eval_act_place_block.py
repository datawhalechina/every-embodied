#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mediapy as media


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop ACT evaluation for DISCOVERSE place_block")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--control-fps", type=float, default=20.0)
    parser.add_argument(
        "--image-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help="Color order expected by the checkpoint; legacy checkpoints use BGR.",
    )
    parser.add_argument(
        "--temporal-agg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ACT temporal ensembling (default: enabled).",
    )
    return parser.parse_args()


def build_policy(checkpoint, max_steps, temporal_agg):
    from policies.act.act import ACTPolicy

    config = {
        "kl_weight": 10,
        "chunk_size": 25,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "temporal_agg": temporal_agg,
        "num_queries": 25,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "lr": -1,
        "camera_names": ["0", "1"],
        "state_dim": 7,
        "action_dim": 7,
        "max_timesteps": max_steps,
    }
    policy = ACTPolicy(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(state)
    policy.cuda().eval()
    return policy


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "policies" / "act"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from examples.tasks_airbot_play.place_block import SimNode, cfg as task_cfg
    with args.stats.open("rb") as file:
        import pickle

        stats = pickle.load(file)

    qpos_mean = np.asarray(stats["qpos_mean"])
    qpos_std = np.asarray(stats["qpos_std"])
    action_mean = np.asarray(stats["action_mean"])
    action_std = np.asarray(stats["action_std"])
    policy = build_policy(
        args.checkpoint.resolve(), args.max_steps, args.temporal_agg
    )

    eval_cfg = copy.deepcopy(task_cfg)
    eval_cfg.headless = True
    eval_cfg.sync = False
    eval_cfg.use_gaussian_renderer = False
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = SimNode(eval_cfg)
    physics_steps_per_action = max(1, round(1.0 / (args.control_fps * sim.delta_t)))
    results = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        rollout_dir = args.output_dir / f"seed_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        video_frames = {cam_id: [] for cam_id in eval_cfg.obs_rgb_cam_id}
        obs = sim.reset()
        policy.reset()
        trace = []
        success = False
        steps = 0
        for steps in range(1, args.max_steps + 1):
            frames = [obs["img"][cam_id] for cam_id in eval_cfg.obs_rgb_cam_id]
            if args.image_color_order == "bgr":
                frames = [np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames]
            images = np.stack(frames, axis=0)
            image_tensor = (
                torch.from_numpy(images.copy())
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
                .unsqueeze(0)
                .cuda()
            )
            qpos = (np.asarray(obs["jq"]) - qpos_mean) / qpos_std
            qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).cuda()
            with torch.inference_mode():
                normalized_actions = policy(qpos_tensor, image_tensor)
            action = normalized_actions[0, 0].cpu().numpy() * action_std + action_mean
            action = np.clip(
                action,
                sim.mj_model.actuator_ctrlrange[: action.shape[0], 0],
                sim.mj_model.actuator_ctrlrange[: action.shape[0], 1],
            )
            block_pos = sim.mj_data.body("block_green").xpos.copy()
            bowl_pos = sim.mj_data.body("bowl_pink").xpos.copy()
            trace.append(
                {
                    "step": steps,
                    "sim_time": float(sim.mj_data.time),
                    "qpos": np.asarray(obs["jq"]).tolist(),
                    "action": action.tolist(),
                    "block_pos": block_pos.tolist(),
                    "bowl_pos": bowl_pos.tolist(),
                }
            )
            for _ in range(physics_steps_per_action):
                obs, _, _, _, _ = sim.step(action)
                if sim.check_success():
                    success = True
                    break
            for cam_id in eval_cfg.obs_rgb_cam_id:
                video_frames[cam_id].append(obs["img"][cam_id].copy())
            if success:
                break

        for cam_id, frames in video_frames.items():
            media.write_video(rollout_dir / f"cam_{cam_id}.mp4", frames, fps=20)

        result = {
            "seed": seed,
            "success": success,
            "steps": steps,
            "sim_time": float(sim.mj_data.time),
            "control_fps": args.control_fps,
            "physics_steps_per_action": physics_steps_per_action,
            "videos": [str(rollout_dir / f"cam_{cam_id}.mp4") for cam_id in eval_cfg.obs_rgb_cam_id],
        }
        (rollout_dir / "action_trace.json").write_text(
            json.dumps(trace), encoding="utf-8"
        )
        (rollout_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps(result))

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "temporal_agg": args.temporal_agg,
        "image_color_order": args.image_color_order,
        "successes": sum(result["success"] for result in results),
        "episodes": len(results),
        "success_rate": sum(result["success"] for result in results) / len(results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
