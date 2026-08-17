#!/usr/bin/env python3
"""Replay GT or Pi0-predicted actions for one language dataset episode.

The Pi0 prediction mode is teacher-forced on dataset observations: the policy
sees recorded dataset frames/states/tasks, but the MuJoCo environment receives
the predicted actions open-loop.  This separates action accuracy from closed-loop
observation drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "pi0"], required=True)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--repo-id", default="datawhale_eai_pnp_language")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data_language"))
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--settle-actions", type=int, default=20)
    parser.add_argument("--max-episode-frames", type=int, default=0)
    parser.add_argument(
        "--env-reset-seed",
        type=int,
        default=0,
        help=(
            "Seed passed to env.reset before replay. Default keeps historical "
            "behavior; set this to the data-collection seed when auditing "
            "whether recorded actions are exactly replayable."
        ),
    )
    parser.add_argument(
        "--hard-reset-sim-data",
        action="store_true",
        help="Call the inner MuJoCo reset before env.reset(...), matching collection runs that used this cleanup.",
    )
    parser.add_argument(
        "--headless-no-viewer",
        action="store_true",
        help="Run MuJoCo physics without creating a GLFW viewer; useful for SSH/headless evaluation.",
    )
    parser.add_argument(
        "--stop-on-physical-success",
        action="store_true",
        help="Diagnostic/deployment option: stop replay as soon as the stricter physical success criterion is reached.",
    )
    parser.add_argument("--reset-policy-each-frame", action="store_true")
    parser.add_argument(
        "--pi0-action-mode",
        choices=["absolute", "joint_delta", "eef_delta", "eef_abs"],
        default="absolute",
        help=(
            "How to bridge replay actions into MuJoCo. absolute uses joint_angle, "
            "joint_delta adds action[:6] to current joint state, and eef_delta uses "
            "SimpleEnv2 eef_pose with action[:3] as TCP delta. eef_abs interprets "
            "action[:3] as the next TCP xyz target and converts it to a bounded delta."
        ),
    )
    parser.add_argument(
        "--eef-abs-max-step",
        type=float,
        default=0.004,
        help="Maximum per-axis TCP delta used when --pi0-action-mode=eef_abs.",
    )
    parser.add_argument("--clamp-action-to-episode-gt", action="store_true")
    parser.add_argument(
        "--clip-gripper",
        action="store_true",
        help="Fair actuator postprocess: clip the gripper command to the executable [0, 1] range.",
    )
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-until-step", type=int, default=-1)
    parser.add_argument(
        "--gripper-open-tail",
        type=int,
        default=0,
        help="Diagnostic: force the last N replay gripper actions to 0.0/open-release.",
    )
    parser.add_argument(
        "--replace-tail-with-gt",
        type=int,
        default=0,
        help="Diagnostic: replace the last N replay actions with dataset GT actions.",
    )
    parser.add_argument(
        "--replace-prefix-with-gt",
        type=int,
        default=0,
        help="Diagnostic: replace the first N replay actions with dataset GT actions.",
    )
    parser.add_argument(
        "--replace-arm-with-gt",
        action="store_true",
        help="Diagnostic: use dataset GT for action[:6] and policy prediction for gripper.",
    )
    parser.add_argument(
        "--replace-gripper-with-gt",
        action="store_true",
        help="Diagnostic: use policy prediction for action[:6] and dataset GT for gripper.",
    )
    parser.add_argument(
        "--append-template-tail",
        choices=["none", "all", "task"],
        default="none",
        help=(
            "Diagnostic: append a dataset-derived mean tail trajectory after the policy actions. "
            "'task' averages only episodes with the same language instruction."
        ),
    )
    parser.add_argument(
        "--template-tail-steps",
        type=int,
        default=0,
        help="Number of final dataset actions used for --append-template-tail.",
    )
    parser.add_argument(
        "--template-blend-steps",
        type=int,
        default=0,
        help="Interpolate this many actions from the final policy action to the first template action.",
    )
    parser.add_argument(
        "--template-force-open-gripper",
        action="store_true",
        help="Force the appended template tail gripper command to 0.0/open.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-actions-npz",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic archive containing raw policy actions, GT actions, "
            "postprocessed actions, bridged env actions, and per-step replay metrics."
        ),
    )
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-every-n-actions", type=int, default=1)
    parser.add_argument("--video-title", default="")
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    return parser.parse_args()


def round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(values).reshape(-1)]


def put_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    frame = frame_rgb.copy()
    pad = 8
    line_h = 24
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_h), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (pad, pad + line_h * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def make_video_frame(renderer: Any, env: Any, debug: dict, action_step: int, phase: str, title: str) -> np.ndarray:
    renderer.update_scene(env.env.data, camera=renderer._codex_camera_name)
    frame = renderer.render()
    tcp_pos = debug.get("tcp_pos", [np.nan, np.nan, np.nan])
    lines = [
        title or f"episode replay {phase}",
        (
            f"step={action_step} phase={phase} final_ok={debug.get('physical_success', False)} "
            f"legacy={debug.get('success', False)}"
        ),
        (
            f"xy={float(debug.get('xy_dist', np.nan)):.3f} "
            f"lift={float(debug.get('max_target_lift', np.nan)):.3f} "
            f"tcp_z={float(tcp_pos[2]):.3f} "
            f"grip={float(debug.get('gripper', np.nan)):.3f} "
            f"upright={float(debug.get('final_target_upright_cos', np.nan)):.3f}"
        ),
    ]
    return put_overlay(frame, lines)


def to_device_batch(item: dict[str, Any], device: str) -> dict[str, Any]:
    batch: dict[str, Any] = {"task": [item["task"]]}
    for key in ("observation.state", "observation.image", "observation.wrist_image"):
        value = item[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        batch[key] = value.unsqueeze(0).to(device=device, dtype=torch.float32)
    return batch


def tensor_to_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def maybe_hard_reset_sim_data(env: Any, enabled: bool) -> None:
    if not enabled:
        return
    inner = getattr(env, "env", None)
    if inner is None:
        return
    try:
        inner.reset(step=False)
    except TypeError:
        inner.reset()


def build_template_tail_actions(
    dataset: Any,
    episode_column: np.ndarray,
    current_task: str,
    selector: str,
    tail_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if selector == "none" or tail_steps <= 0:
        raise ValueError("--template-tail-steps must be > 0 when --append-template-tail is enabled")

    tails: list[np.ndarray] = []
    used_episodes: list[int] = []
    skipped_episodes: list[int] = []
    for episode_index in sorted({int(x) for x in episode_column.tolist()}):
        raw_indices = np.where(episode_column == int(episode_index))[0]
        if raw_indices.size < tail_steps:
            skipped_episodes.append(int(episode_index))
            continue
        episode_task = str(dataset[int(raw_indices[0])]["task"])
        if selector == "task" and episode_task != current_task:
            continue
        tail_indices = raw_indices[-tail_steps:]
        tail = np.stack(
            [
                tensor_to_np(dataset[int(raw_idx)]["action"]).reshape(-1)[:7].astype(np.float32)
                for raw_idx in tail_indices
            ]
        )
        tails.append(tail)
        used_episodes.append(int(episode_index))

    if not tails:
        raise ValueError(
            f"No episodes available for template tail selector={selector!r}, "
            f"task={current_task!r}, tail_steps={tail_steps}"
        )

    template_tail = np.mean(np.stack(tails, axis=0), axis=0).astype(np.float32)
    meta = {
        "selector": selector,
        "tail_steps": int(tail_steps),
        "used_episodes": used_episodes,
        "skipped_episodes": skipped_episodes,
        "num_template_episodes": int(len(used_episodes)),
    }
    return template_tail, meta


def init_tracker(env) -> dict:
    target_body = env.obj_target
    p_target = env.env.get_p_body(target_body)
    p_plate = env.env.get_p_body("body_obj_plate_11")
    return {
        "target_body": target_body,
        "initial_target_pos": round_list(p_target),
        "initial_plate_pos": round_list(p_plate),
        "initial_target_z": float(p_target[2]),
        "max_target_z": float(p_target[2]),
        "max_target_lift": 0.0,
        "lifted_steps": 0,
    }


def update_tracker(env, tracker: dict, args: argparse.Namespace) -> None:
    p_target = env.env.get_p_body(tracker["target_body"])
    lift = float(p_target[2]) - float(tracker["initial_target_z"])
    tracker["max_target_z"] = max(float(tracker["max_target_z"]), float(p_target[2]))
    tracker["max_target_lift"] = max(float(tracker["max_target_lift"]), lift)
    if lift >= float(args.physical_min_lift):
        tracker["lifted_steps"] += 1


def physical_debug(env, tracker: dict, args: argparse.Namespace) -> dict:
    from eval_policy_success import get_body_upright_cos, get_smolvla_debug

    base = get_smolvla_debug(env)
    target_body = tracker["target_body"]
    p_target = env.env.get_p_body(target_body)
    upright_cos = get_body_upright_cos(env, target_body)
    lifted_enough = int(tracker["lifted_steps"]) >= int(args.physical_min_lift_steps)
    final_upright = bool(np.isfinite(upright_cos) and upright_cos >= float(args.physical_final_upright_cos))
    base.update(
        {
            "physical_success": bool(base["success"] and lifted_enough and final_upright),
            "physical_lifted_enough": bool(lifted_enough),
            "physical_final_upright": bool(final_upright),
            "target_body": target_body,
            "initial_target_pos": tracker["initial_target_pos"],
            "initial_plate_pos": tracker["initial_plate_pos"],
            "final_target_pos": round_list(p_target),
            "max_target_z": float(tracker["max_target_z"]),
            "max_target_lift": float(tracker["max_target_lift"]),
            "lifted_steps": int(tracker["lifted_steps"]),
            "final_target_upright_cos": float(upright_cos),
        }
    )
    return base


def action_for_environment(action: np.ndarray, env: Any, args: argparse.Namespace) -> np.ndarray:
    env_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if args.pi0_action_mode in ("absolute", "eef_delta"):
        return env_action
    if args.pi0_action_mode == "eef_abs":
        if env_action.shape[0] < 7:
            raise ValueError(f"Expected 7-D action, got {env_action.shape}")
        current_tcp = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32).reshape(3)
        target_tcp = env_action[:3].copy()
        bridged = np.zeros(7, dtype=np.float32)
        bridged[:3] = np.clip(
            target_tcp - current_tcp,
            -float(args.eef_abs_max_step),
            float(args.eef_abs_max_step),
        )
        bridged[6] = env_action[6]
        return bridged
    if args.pi0_action_mode == "joint_delta":
        state = np.asarray(env.get_joint_state()[:6], dtype=np.float32).reshape(-1)
        if env_action.shape[0] < 7 or state.shape[0] != 6:
            raise ValueError(f"Expected 7-D action and 6-D joint state, got {env_action.shape}, {state.shape}")
        bridged = env_action[:7].copy()
        bridged[:6] = state + env_action[:6]
        return bridged.astype(np.float32)
    raise ValueError(f"Unsupported pi0_action_mode: {args.pi0_action_mode}")


def configure_env_action_space(env: Any, action_mode: str) -> None:
    if action_mode in ("eef_delta", "eef_abs"):
        from mujoco_env.transforms import rpy2r

        env.action_type = "eef_pose"
        env.p0, _ = env.env.get_pR_body(body_name="tcp_link")
        env.R0 = rpy2r(np.deg2rad([90.0, 0.0, 90.0]))
    elif action_mode in ("absolute", "joint_delta"):
        env.action_type = "joint_angle"
    else:
        raise ValueError(f"Unsupported pi0_action_mode: {action_mode}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.headless_no_viewer or args.output_video is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.headless_no_viewer and "pyautogui" not in sys.modules:
        sys.modules["pyautogui"] = SimpleNamespace(size=lambda: (1920, 1080))

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.pi0 import PI0Policy
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from mujoco_env.y_env2 import SimpleEnv2
    import cv2
    import mujoco
    if args.headless_no_viewer:
        SimpleEnv2.init_viewer = lambda self: self.env.reset()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    episode_column = tensor_to_np(dataset.hf_dataset["episode_index"]).astype(int)
    raw_indices = np.where(episode_column == int(args.episode))[0]
    if raw_indices.size == 0:
        raise ValueError(f"Episode {args.episode} not found")
    if args.max_episode_frames > 0:
        raw_indices = raw_indices[: int(args.max_episode_frames)]

    first_item = dataset[int(raw_indices[0])]
    task = str(first_item["task"])
    obj_init = tensor_to_np(first_item["obj_init"]).astype(np.float32).reshape(-1)
    if obj_init.size < 9:
        raise ValueError(f"Expected obj_init with 9 values, got {obj_init.shape}")

    policy = None
    if args.mode == "pi0":
        if args.policy_path is None:
            raise ValueError("--policy-path is required for --mode pi0")
        metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
        policy = PI0Policy.from_pretrained(args.policy_path, dataset_stats=metadata.stats)
        policy.to(args.device)
        policy.eval()
        policy.reset()

    actions = []
    gt_actions = []
    pred_errors = []
    dataset_timestamps = []
    dataset_states = []
    for raw_idx in raw_indices:
        item = dataset[int(raw_idx)]
        gt = tensor_to_np(item["action"]).reshape(-1)[:7].astype(np.float32)
        gt_actions.append(gt)
        if "timestamp" in item:
            dataset_timestamps.append(float(np.asarray(tensor_to_np(item["timestamp"])).reshape(-1)[0]))
        if "observation.state" in item:
            dataset_states.append(tensor_to_np(item["observation.state"]).reshape(-1).astype(np.float32))
        if args.mode == "gt":
            actions.append(gt)
            continue
        if args.reset_policy_each_frame:
            policy.reset()
        batch = to_device_batch(item, args.device)
        with torch.no_grad():
            pred = policy.select_action(batch)[0, :7].detach().cpu().numpy().astype(np.float32)
        actions.append(pred)
        pred_errors.append(np.abs(pred - gt))
    actions_np = np.stack(actions).astype(np.float32)
    gt_np = np.stack(gt_actions).astype(np.float32)
    raw_actions_np = actions_np.copy()
    postprocess = {
        "clamp_action_to_episode_gt": bool(args.clamp_action_to_episode_gt),
        "clip_gripper": bool(args.clip_gripper),
        "binarize_gripper": bool(args.binarize_gripper),
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_open_until_step": int(args.gripper_open_until_step),
        "gripper_open_tail": int(args.gripper_open_tail),
        "replace_tail_with_gt": int(args.replace_tail_with_gt),
        "replace_prefix_with_gt": int(args.replace_prefix_with_gt),
        "replace_arm_with_gt": bool(args.replace_arm_with_gt),
        "replace_gripper_with_gt": bool(args.replace_gripper_with_gt),
        "append_template_tail": args.append_template_tail,
        "template_tail_steps": int(args.template_tail_steps),
        "template_blend_steps": int(args.template_blend_steps),
        "template_force_open_gripper": bool(args.template_force_open_gripper),
        "stop_on_physical_success": bool(args.stop_on_physical_success),
        "pi0_action_mode": args.pi0_action_mode,
        "template_tail_meta": None,
    }
    if args.clamp_action_to_episode_gt:
        actions_np = np.clip(actions_np, gt_np.min(axis=0), gt_np.max(axis=0))
    if args.clip_gripper:
        actions_np[:, 6] = np.clip(actions_np[:, 6], 0.0, 1.0)
    if args.binarize_gripper:
        actions_np[:, 6] = (actions_np[:, 6] >= float(args.gripper_threshold)).astype(np.float32)
    if int(args.gripper_open_until_step) >= 0:
        open_steps = min(int(args.gripper_open_until_step), len(actions_np))
        actions_np[:open_steps, 6] = 0.0
    if int(args.gripper_open_tail) > 0:
        tail_open_steps = min(int(args.gripper_open_tail), len(actions_np))
        actions_np[-tail_open_steps:, 6] = 0.0
    if int(args.replace_tail_with_gt) > 0:
        tail_steps = min(int(args.replace_tail_with_gt), len(actions_np))
        actions_np[-tail_steps:] = gt_np[-tail_steps:]
    if int(args.replace_prefix_with_gt) > 0:
        prefix_steps = min(int(args.replace_prefix_with_gt), len(actions_np))
        actions_np[:prefix_steps] = gt_np[:prefix_steps]
    if args.replace_arm_with_gt:
        actions_np[:, :6] = gt_np[:, :6]
    if args.replace_gripper_with_gt:
        actions_np[:, 6] = gt_np[:, 6]
    if args.append_template_tail != "none":
        template_tail, template_meta = build_template_tail_actions(
            dataset=dataset,
            episode_column=episode_column,
            current_task=task,
            selector=args.append_template_tail,
            tail_steps=int(args.template_tail_steps),
        )
        if args.template_force_open_gripper:
            template_tail[:, 6] = 0.0
        if int(args.template_blend_steps) > 0:
            blend_steps = int(args.template_blend_steps)
            start = actions_np[-1]
            end = template_tail[0]
            alphas = np.linspace(
                1.0 / float(blend_steps + 1),
                float(blend_steps) / float(blend_steps + 1),
                blend_steps,
                dtype=np.float32,
            ).reshape(-1, 1)
            blend_tail = (1.0 - alphas) * start.reshape(1, -1) + alphas * end.reshape(1, -1)
            actions_np = np.concatenate([actions_np, blend_tail.astype(np.float32), template_tail], axis=0)
        else:
            actions_np = np.concatenate([actions_np, template_tail], axis=0)
        postprocess["template_tail_meta"] = template_meta

    env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    start = time.time()
    action_steps = 0
    sim_steps = 0
    success_ever = False
    physical_success_ever = False
    first_physical_success_step = None
    first_legacy_success_step = None
    replay_step_indices = []
    replay_phase_codes = []
    replay_actions = []
    replay_env_actions = []
    replay_tcp_pos = []
    replay_target_pos = []
    replay_xy_dist = []
    replay_max_lift = []
    replay_upright_cos = []
    replay_gripper = []
    replay_legacy_success = []
    replay_physical_success = []
    renderer = None
    video_writer = None
    video_frame_count = 0
    try:
        maybe_hard_reset_sim_data(env, bool(args.hard_reset_sim_data))
        env.reset(seed=int(args.env_reset_seed))
        env.set_instruction(task)
        env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])
        configure_env_action_space(env, args.pi0_action_mode)
        tracker = init_tracker(env)
        last_debug = physical_debug(env, tracker, args)
        if args.output_video is not None:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(env.env.model, height=int(args.video_height), width=int(args.video_width))
            renderer._codex_camera_name = str(args.video_camera)
            video_writer = cv2.VideoWriter(
                str(args.output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.video_fps),
                (int(args.video_width), int(args.video_height)),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {args.output_video}")
            frame = make_video_frame(renderer, env, last_debug, action_step=0, phase="reset", title=args.video_title)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_frame_count += 1
        total_actions = len(actions_np) + max(int(args.settle_actions), 0)
        last_action = actions_np[0]
        while action_steps < total_actions and (args.headless_no_viewer or env.env.is_viewer_alive()):
            env.step_env()
            sim_steps += 1
            if not env.env.loop_every(HZ=args.hz):
                continue
            if action_steps < len(actions_np):
                action = actions_np[action_steps]
                last_action = action
                phase = "replay"
            else:
                action = last_action
                phase = "settle"
            env_action = action_for_environment(action, env, args)
            env.step(env_action)
            update_tracker(env, tracker, args)
            last_debug = physical_debug(env, tracker, args)
            replay_step_indices.append(int(action_steps))
            replay_phase_codes.append(0 if phase == "replay" else 1)
            replay_actions.append(np.asarray(action, dtype=np.float32).reshape(-1)[:7])
            replay_env_actions.append(np.asarray(env_action, dtype=np.float32).reshape(-1)[:7])
            replay_tcp_pos.append(np.asarray(last_debug.get("tcp_pos", [np.nan, np.nan, np.nan]), dtype=np.float32))
            replay_target_pos.append(
                np.asarray(last_debug.get("final_target_pos", [np.nan, np.nan, np.nan]), dtype=np.float32)
            )
            replay_xy_dist.append(float(last_debug.get("xy_dist", np.nan)))
            replay_max_lift.append(float(last_debug.get("max_target_lift", np.nan)))
            replay_upright_cos.append(float(last_debug.get("final_target_upright_cos", np.nan)))
            replay_gripper.append(float(last_debug.get("gripper", np.nan)))
            replay_legacy_success.append(bool(last_debug.get("success", False)))
            replay_physical_success.append(bool(last_debug.get("physical_success", False)))
            if last_debug["success"] and first_legacy_success_step is None:
                first_legacy_success_step = action_steps
            if last_debug["physical_success"] and first_physical_success_step is None:
                first_physical_success_step = action_steps
            success_ever = bool(success_ever or last_debug["success"])
            physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])
            if video_writer is not None and action_steps % max(int(args.video_every_n_actions), 1) == 0:
                frame = make_video_frame(
                    renderer,
                    env,
                    last_debug,
                    action_step=action_steps,
                    phase=phase,
                    title=args.video_title,
                )
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                video_frame_count += 1
            action_steps += 1
            if args.stop_on_physical_success and last_debug["physical_success"]:
                break
        last_debug = physical_debug(env, tracker, args)
    finally:
        if video_writer is not None:
            video_writer.release()
        if renderer is not None:
            renderer.close()
        try:
            env.env.close_viewer()
        except Exception:
            pass

    error_summary = None
    if pred_errors:
        err = np.stack(pred_errors)
        error_summary = {
            "mae": float(err.mean()),
            "joint_mae": float(err[:, :6].mean()),
            "gripper_abs": float(err[:, 6].mean()),
            "max_abs": float(err.max()),
        }
    summary = {
        "mode": args.mode,
        "episode": int(args.episode),
        "task": task,
        "num_episode_frames": int(len(raw_indices)),
        "num_actions_after_postprocess": int(len(actions_np)),
        "settle_actions": int(args.settle_actions),
        "action_steps": int(action_steps),
        "sim_steps": int(sim_steps),
        "elapsed_s": round(time.time() - start, 3),
        "video": str(args.output_video) if args.output_video is not None else None,
        "video_frame_count": int(video_frame_count),
        "success": bool(success_ever or last_debug["success"]),
        "physical_success": bool(physical_success_ever or last_debug["physical_success"]),
        "legacy_success_ever": bool(success_ever),
        "physical_success_ever": bool(physical_success_ever),
        "final_legacy_success": bool(last_debug["success"]),
        "final_physical_success": bool(last_debug["physical_success"]),
        "first_legacy_success_step": first_legacy_success_step,
        "first_physical_success_step": first_physical_success_step,
        "prediction_error": error_summary,
        "actions_npz": str(args.output_actions_npz) if args.output_actions_npz is not None else None,
        "postprocess": postprocess,
        "action_stats": {
            "pred_or_replay_min": round_list(actions_np.min(axis=0), 5),
            "pred_or_replay_max": round_list(actions_np.max(axis=0), 5),
            "raw_pred_or_replay_min": round_list(raw_actions_np.min(axis=0), 5),
            "raw_pred_or_replay_max": round_list(raw_actions_np.max(axis=0), 5),
            "gt_min": round_list(gt_np.min(axis=0), 5),
            "gt_max": round_list(gt_np.max(axis=0), 5),
        },
        "debug": last_debug,
    }
    if args.output_actions_npz is not None:
        args.output_actions_npz.parent.mkdir(parents=True, exist_ok=True)
        pred_error_np = np.stack(pred_errors).astype(np.float32) if pred_errors else np.empty((0, 7), dtype=np.float32)
        timestamp_np = (
            np.asarray(dataset_timestamps, dtype=np.float32)
            if dataset_timestamps
            else np.empty((0,), dtype=np.float32)
        )
        state_np = (
            np.stack(dataset_states).astype(np.float32)
            if dataset_states
            else np.empty((0, 0), dtype=np.float32)
        )
        np.savez_compressed(
            args.output_actions_npz,
            raw_actions=raw_actions_np,
            postprocessed_actions=actions_np,
            gt_actions=gt_np,
            pred_abs_error=pred_error_np,
            dataset_timestamps=timestamp_np,
            dataset_states=state_np,
            raw_indices=np.asarray(raw_indices, dtype=np.int64),
            replay_step_indices=np.asarray(replay_step_indices, dtype=np.int64),
            replay_phase_codes=np.asarray(replay_phase_codes, dtype=np.int8),
            replay_actions=np.asarray(replay_actions, dtype=np.float32),
            replay_env_actions=np.asarray(replay_env_actions, dtype=np.float32),
            replay_tcp_pos=np.asarray(replay_tcp_pos, dtype=np.float32),
            replay_target_pos=np.asarray(replay_target_pos, dtype=np.float32),
            replay_xy_dist=np.asarray(replay_xy_dist, dtype=np.float32),
            replay_max_lift=np.asarray(replay_max_lift, dtype=np.float32),
            replay_upright_cos=np.asarray(replay_upright_cos, dtype=np.float32),
            replay_gripper=np.asarray(replay_gripper, dtype=np.float32),
            replay_legacy_success=np.asarray(replay_legacy_success, dtype=np.bool_),
            replay_physical_success=np.asarray(replay_physical_success, dtype=np.bool_),
            task=np.asarray(task),
            episode=np.asarray(int(args.episode), dtype=np.int64),
            mode=np.asarray(args.mode),
            postprocess_json=np.asarray(json.dumps(postprocess, ensure_ascii=False)),
            summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
