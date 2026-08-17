#!/usr/bin/env python3
"""Batch replay GT or Pi0-predicted actions over multiple dataset episodes.

This is a thin, tutorial-friendly wrapper around
``replay_pi0_dataset_predictions.py``.  It keeps one Pi0 policy instance loaded
while evaluating many teacher-forced dataset episodes, then writes per-episode
JSON files plus one JSONL/summary pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import replay_pi0_dataset_predictions as single


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def parse_episode_spec(spec: str, available: list[int]) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return available
    episodes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            episodes.extend(range(start, end + step, step))
        else:
            episodes.append(int(part))
    missing = sorted(set(episodes) - set(available))
    if missing:
        raise ValueError(f"Episodes not found in dataset: {missing}")
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "pi0"], required=True)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--repo-id", default="datawhale_eai_pnp_language")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data_language"))
    parser.add_argument(
        "--position-profile",
        default="",
        help="SimpleEnv2 random-position profile used during data collection.",
    )
    parser.add_argument("--stats-repo-id", default=None)
    parser.add_argument("--stats-dataset-root", type=Path, default=None)
    parser.add_argument("--episodes", default="all", help="'all', '0-19', or comma-separated episode ids.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--settle-actions", type=int, default=20)
    parser.add_argument("--max-episode-frames", type=int, default=0)
    parser.add_argument(
        "--env-reset-seed",
        type=int,
        default=0,
        help=(
            "Seed passed to env.reset before replay when --env-reset-seeds is "
            "not provided. Default keeps historical behavior."
        ),
    )
    parser.add_argument(
        "--env-reset-seeds",
        default="",
        help=(
            "Comma-separated env.reset seeds aligned with --episodes order. "
            "Useful for GT replay audits of data collected from specific seeds."
        ),
    )
    parser.add_argument(
        "--hard-reset-sim-data",
        action="store_true",
        help="Call the inner MuJoCo reset before env.reset(...), matching collection runs that used this cleanup.",
    )
    parser.add_argument(
        "--use-env-reset-object-pose",
        action="store_true",
        help=(
            "Keep the settled object pose produced by env.reset(seed). Use this when replay seeds are the "
            "original collection seeds; dataset obj_init stores spawn coordinates, not settled body poses."
        ),
    )
    parser.add_argument(
        "--reuse-environment",
        action="store_true",
        help=(
            "Reuse one SimpleEnv2 instance across episodes while still applying hard reset and seeded reset. "
            "This avoids file-descriptor/asset-provider exhaustion in long replay panels."
        ),
    )
    parser.add_argument(
        "--prefix-replay-map-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping episode ids to saved prefix NPZ files and action modes. "
            "Use it to reconstruct recovery episodes before replaying their oracle suffix."
        ),
    )
    parser.add_argument("--headless-no-viewer", action="store_true")
    parser.add_argument("--stop-on-physical-success", action="store_true")
    parser.add_argument("--reset-policy-each-frame", action="store_true")
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=-1,
        help=(
            "If >=0, reset torch/numpy RNG at the start of each episode to "
            "policy_seed + episode_id. This makes stochastic Pi0 action sampling "
            "independent of batch order."
        ),
    )
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
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-until-step", type=int, default=-1)
    parser.add_argument("--gripper-open-tail", type=int, default=0)
    parser.add_argument("--replace-tail-with-gt", type=int, default=0)
    parser.add_argument("--replace-prefix-with-gt", type=int, default=0)
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
        "--gripper-head-path",
        type=Path,
        default=None,
        help=(
            "Diagnostic/raw-compatible auxiliary head: replace action[6] with a "
            "learned logistic gripper head saved by train_pi0_gripper_progress_head.py."
        ),
    )
    parser.add_argument("--gripper-head-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-head-blue-threshold", type=float, default=None)
    parser.add_argument("--gripper-head-red-threshold", type=float, default=None)
    parser.add_argument(
        "--gripper-head-continuous",
        action="store_true",
        help="Use the head probability directly instead of binarizing at --gripper-head-threshold.",
    )
    parser.add_argument("--append-template-tail", choices=["none", "all", "task"], default="none")
    parser.add_argument("--template-tail-steps", type=int, default=0)
    parser.add_argument("--template-blend-steps", type=int, default=0)
    parser.add_argument("--template-force-open-gripper", action="store_true")
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    parser.add_argument(
        "--env-create-retries",
        type=int,
        default=3,
        help="Retry MuJoCo environment creation if asset loading hits a transient filesystem/provider error.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def prepare_runtime(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.headless_no_viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.headless_no_viewer and "pyautogui" not in sys.modules:
        sys.modules["pyautogui"] = SimpleNamespace(size=lambda: (1920, 1080))


def compute_actions(
    args: argparse.Namespace,
    dataset: Any,
    raw_indices: np.ndarray,
    policy: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, float] | None]:
    actions = []
    gt_actions = []
    pred_errors = []
    for raw_idx in raw_indices:
        item = dataset[int(raw_idx)]
        gt = single.tensor_to_np(item["action"]).reshape(-1)[:7].astype(np.float32)
        gt_actions.append(gt)
        if args.mode == "gt":
            actions.append(gt)
            continue
        if args.reset_policy_each_frame:
            policy.reset()
        batch = single.to_device_batch(item, args.device)
        with torch.no_grad():
            pred = policy.select_action(batch)[0, :7].detach().cpu().numpy().astype(np.float32)
        actions.append(pred)
        pred_errors.append(np.abs(pred - gt))

    actions_np = np.stack(actions).astype(np.float32)
    gt_np = np.stack(gt_actions).astype(np.float32)
    error_summary = None
    if pred_errors:
        err = np.stack(pred_errors)
        error_summary = {
            "mae": float(err.mean()),
            "joint_mae": float(err[:, :6].mean()),
            "gripper_abs": float(err[:, 6].mean()),
            "max_abs": float(err.max()),
        }
    return actions_np, gt_np, error_summary


def load_gripper_head(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    required = {"weight", "bias", "feature_mean", "feature_std"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Gripper head {path} is missing fields: {missing}")
    return {
        "path": str(path),
        "weight": np.asarray(data["weight"], dtype=np.float32),
        "bias": float(np.asarray(data["bias"]).reshape(())),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
        "feature_mode": str(
            np.asarray(data["feature_mode"] if "feature_mode" in data.files else "state_progress_task").reshape(())
        ),
    }


def gripper_head_features(
    dataset: Any,
    raw_indices: np.ndarray,
    task: str,
    actions_np: np.ndarray,
) -> np.ndarray:
    """Build non-oracle features for a learned gripper head.

    The feature vector uses recorded observation.state, episode progress, task
    color parsed from language, and Pi0's own predicted arm target. It does not
    use object/plate pose or GT gripper labels at inference time.
    """

    n = int(len(raw_indices))
    denom = float(max(n - 1, 1))
    is_blue = 1.0 if "blue" in task.lower() else 0.0
    is_red = 1.0 if "red" in task.lower() else 0.0
    rows = []
    for i, raw_idx in enumerate(raw_indices):
        item = dataset[int(raw_idx)]
        state = single.tensor_to_np(item["observation.state"]).astype(np.float32).reshape(-1)[:6]
        progress = float(i) / denom
        pows = np.asarray([progress, progress**2, progress**3, progress**4, progress**5], dtype=np.float32)
        pred_arm = np.asarray(actions_np[i, :6], dtype=np.float32).reshape(-1)
        rows.append(np.concatenate([state, pred_arm, pows, np.asarray([is_blue, is_red], dtype=np.float32)]))
    return np.stack(rows).astype(np.float32)


def apply_gripper_head(
    args: argparse.Namespace,
    dataset: Any,
    raw_indices: np.ndarray,
    task: str,
    processed: np.ndarray,
    postprocess: dict[str, Any],
    head: dict[str, Any] | None,
) -> np.ndarray:
    if head is None:
        postprocess["gripper_head_path"] = None
        return processed
    features = gripper_head_features(dataset, raw_indices, task, processed)
    mean = head["feature_mean"]
    std = np.where(head["feature_std"] < 1e-6, 1.0, head["feature_std"])
    if features.shape[1] != mean.shape[0] or mean.shape != std.shape or mean.shape != head["weight"].shape:
        raise ValueError(
            "Gripper head feature shape mismatch: "
            f"features={features.shape}, mean={mean.shape}, weight={head['weight'].shape}"
        )
    probs = sigmoid_np(((features - mean) / std) @ head["weight"] + head["bias"]).astype(np.float32)
    effective_threshold = float(args.gripper_head_threshold)
    task_lower = task.lower()
    if args.gripper_head_blue_threshold is not None and "blue" in task_lower:
        effective_threshold = float(args.gripper_head_blue_threshold)
    if args.gripper_head_red_threshold is not None and "red" in task_lower:
        effective_threshold = float(args.gripper_head_red_threshold)
    if args.gripper_head_continuous:
        processed[:, 6] = probs
    else:
        processed[:, 6] = (probs >= effective_threshold).astype(np.float32)
    postprocess["gripper_head_path"] = head["path"]
    postprocess["gripper_head_threshold"] = float(args.gripper_head_threshold)
    postprocess["gripper_head_blue_threshold"] = (
        None if args.gripper_head_blue_threshold is None else float(args.gripper_head_blue_threshold)
    )
    postprocess["gripper_head_red_threshold"] = (
        None if args.gripper_head_red_threshold is None else float(args.gripper_head_red_threshold)
    )
    postprocess["gripper_head_effective_threshold"] = float(effective_threshold)
    postprocess["gripper_head_continuous"] = bool(args.gripper_head_continuous)
    postprocess["gripper_head_prob_min"] = float(probs.min())
    postprocess["gripper_head_prob_max"] = float(probs.max())
    postprocess["gripper_head_closed_steps"] = int(np.sum(probs >= effective_threshold))
    return processed


def apply_postprocess(
    args: argparse.Namespace,
    dataset: Any,
    episode_column: np.ndarray,
    raw_indices: np.ndarray,
    task: str,
    actions_np: np.ndarray,
    gt_np: np.ndarray,
    gripper_head: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    postprocess = {
        "clamp_action_to_episode_gt": bool(args.clamp_action_to_episode_gt),
        "binarize_gripper": bool(args.binarize_gripper),
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_open_until_step": int(args.gripper_open_until_step),
        "gripper_open_tail": int(args.gripper_open_tail),
        "replace_tail_with_gt": int(args.replace_tail_with_gt),
        "replace_prefix_with_gt": int(args.replace_prefix_with_gt),
        "replace_arm_with_gt": bool(args.replace_arm_with_gt),
        "replace_gripper_with_gt": bool(args.replace_gripper_with_gt),
        "gripper_head_path": str(args.gripper_head_path) if args.gripper_head_path is not None else None,
        "append_template_tail": args.append_template_tail,
        "template_tail_steps": int(args.template_tail_steps),
        "template_blend_steps": int(args.template_blend_steps),
        "template_force_open_gripper": bool(args.template_force_open_gripper),
        "stop_on_physical_success": bool(args.stop_on_physical_success),
        "pi0_action_mode": args.pi0_action_mode,
        "template_tail_meta": None,
    }

    processed = actions_np.copy()
    if args.clamp_action_to_episode_gt:
        processed = np.clip(processed, gt_np.min(axis=0), gt_np.max(axis=0))
    if args.binarize_gripper:
        processed[:, 6] = (processed[:, 6] >= float(args.gripper_threshold)).astype(np.float32)
    if int(args.gripper_open_until_step) >= 0:
        open_steps = min(int(args.gripper_open_until_step), len(processed))
        processed[:open_steps, 6] = 0.0
    if int(args.gripper_open_tail) > 0:
        tail_open_steps = min(int(args.gripper_open_tail), len(processed))
        processed[-tail_open_steps:, 6] = 0.0
    if int(args.replace_tail_with_gt) > 0:
        tail_steps = min(int(args.replace_tail_with_gt), len(processed))
        processed[-tail_steps:] = gt_np[-tail_steps:]
    if int(args.replace_prefix_with_gt) > 0:
        prefix_steps = min(int(args.replace_prefix_with_gt), len(processed))
        processed[:prefix_steps] = gt_np[:prefix_steps]
    if args.replace_arm_with_gt:
        processed[:, :6] = gt_np[:, :6]
    if args.replace_gripper_with_gt:
        processed[:, 6] = gt_np[:, 6]
    processed = apply_gripper_head(args, dataset, raw_indices, task, processed, postprocess, gripper_head)
    if args.append_template_tail != "none":
        template_tail, template_meta = single.build_template_tail_actions(
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
            start = processed[-1]
            end = template_tail[0]
            alphas = np.linspace(
                1.0 / float(blend_steps + 1),
                float(blend_steps) / float(blend_steps + 1),
                blend_steps,
                dtype=np.float32,
            ).reshape(-1, 1)
            blend_tail = (1.0 - alphas) * start.reshape(1, -1) + alphas * end.reshape(1, -1)
            processed = np.concatenate([processed, blend_tail.astype(np.float32), template_tail], axis=0)
        else:
            processed = np.concatenate([processed, template_tail], axis=0)
        postprocess["template_tail_meta"] = template_meta
    return processed, postprocess


def make_simple_env(args: argparse.Namespace):
    from mujoco_env.y_env2 import SimpleEnv2

    last_error: Exception | None = None
    for attempt in range(max(1, int(args.env_create_retries))):
        try:
            kwargs = {}
            if str(args.position_profile).strip():
                kwargs["position_profile"] = str(args.position_profile).strip()
            return SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle", **kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            if attempt + 1 >= max(1, int(args.env_create_retries)):
                break
            print(
                f"MuJoCo env creation failed on attempt {attempt + 1}; retrying: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.5)
    raise RuntimeError(f"Failed to create MuJoCo env after {args.env_create_retries} attempts") from last_error


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


def load_prefix_replay_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    for episode_text, spec in raw.items():
        episode = int(episode_text)
        mode = str(spec["action_mode"])
        if mode not in {"eef_abs", "eef_delta"}:
            raise ValueError(f"Unsupported prefix action mode for episode {episode}: {mode}")
        npz_path = Path(spec["npz_path"])
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        result[episode] = {
            "npz_path": npz_path,
            "action_mode": mode,
            "eef_abs_max_step": float(spec.get("eef_abs_max_step", 0.004)),
            "post_physics_steps": int(spec.get("post_physics_steps", 25)),
            "expected_prefix_steps": int(spec.get("expected_prefix_steps", -1)),
        }
    return result


def replay_saved_prefix(
    env: Any,
    tracker: dict[str, Any],
    spec: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if spec is None:
        return None
    with np.load(spec["npz_path"], allow_pickle=False) as data:
        actions = np.asarray(data["prefix"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Invalid prefix action shape in {spec['npz_path']}: {actions.shape}")
    if spec["expected_prefix_steps"] >= 0 and actions.shape[0] != spec["expected_prefix_steps"]:
        raise ValueError(
            f"Prefix length mismatch in {spec['npz_path']}: "
            f"{actions.shape[0]} != {spec['expected_prefix_steps']}"
        )

    mode = str(spec["action_mode"])
    post_physics_steps = max(int(spec["post_physics_steps"]), 0)
    for action in actions[:, :7]:
        env_action = action.copy()
        if mode == "eef_abs":
            current_tcp = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)
            env_action = np.zeros(7, dtype=np.float32)
            env_action[:3] = np.clip(
                action[:3] - current_tcp,
                -float(spec["eef_abs_max_step"]),
                float(spec["eef_abs_max_step"]),
            )
            env_action[6] = action[6]
        env.step(env_action)
        for _ in range(post_physics_steps):
            env.step_env()
            single.update_tracker(env, tracker, args)

    return {
        "npz_path": str(spec["npz_path"]),
        "action_mode": mode,
        "prefix_actions": int(actions.shape[0]),
        "post_physics_steps": post_physics_steps,
    }


def replay_episode(
    args: argparse.Namespace,
    dataset: Any,
    episode_column: np.ndarray,
    episode: int,
    policy: Any,
    gripper_head: dict[str, Any] | None,
    env: Any | None = None,
) -> dict[str, Any]:
    raw_indices = np.where(episode_column == int(episode))[0]
    if raw_indices.size == 0:
        raise ValueError(f"Episode {episode} not found")
    if args.max_episode_frames > 0:
        raw_indices = raw_indices[: int(args.max_episode_frames)]

    first_item = dataset[int(raw_indices[0])]
    task = str(first_item["task"])
    obj_init = single.tensor_to_np(first_item["obj_init"]).astype(np.float32).reshape(-1)
    if obj_init.size < 9:
        raise ValueError(f"Expected obj_init with 9 values, got {obj_init.shape}")

    if policy is not None:
        policy.reset()
    if int(args.policy_seed) >= 0:
        episode_seed = int(args.policy_seed) + int(episode)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
    raw_actions_np, gt_np, error_summary = compute_actions(args, dataset, raw_indices, policy)
    actions_np, postprocess = apply_postprocess(
        args,
        dataset,
        episode_column,
        raw_indices,
        task,
        raw_actions_np,
        gt_np,
        gripper_head,
    )

    owns_env = env is None
    if env is None:
        env = make_simple_env(args)
    start = time.time()
    action_steps = 0
    sim_steps = 0
    success_ever = False
    physical_success_ever = False
    first_physical_success_step = None
    first_legacy_success_step = None
    try:
        seed_map = getattr(args, "_env_reset_seed_map", {})
        reset_seed = int(seed_map.get(int(episode), int(args.env_reset_seed)))
        maybe_hard_reset_sim_data(env, bool(args.hard_reset_sim_data))
        env.reset(seed=reset_seed)
        env.set_instruction(task)
        if not args.use_env_reset_object_pose:
            env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])
        single.configure_env_action_space(env, args.pi0_action_mode)
        tracker = single.init_tracker(env)
        prefix_map = getattr(args, "_prefix_replay_map", {})
        prefix_replay = replay_saved_prefix(env, tracker, prefix_map.get(int(episode)), args)
        last_debug = single.physical_debug(env, tracker, args)
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
            else:
                action = last_action
            env_action = single.action_for_environment(action, env, args)
            env.step(env_action)
            single.update_tracker(env, tracker, args)
            last_debug = single.physical_debug(env, tracker, args)
            if last_debug["success"] and first_legacy_success_step is None:
                first_legacy_success_step = action_steps
            if last_debug["physical_success"] and first_physical_success_step is None:
                first_physical_success_step = action_steps
            success_ever = bool(success_ever or last_debug["success"])
            physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])
            action_steps += 1
            if args.stop_on_physical_success and last_debug["physical_success"]:
                break
        last_debug = single.physical_debug(env, tracker, args)
    finally:
        if owns_env:
            try:
                env.env.close_viewer()
            except Exception:
                pass

    return {
        "label": args.label,
        "mode": args.mode,
        "episode": int(episode),
        "task": task,
        "num_episode_frames": int(len(raw_indices)),
        "num_actions_after_postprocess": int(len(actions_np)),
        "settle_actions": int(args.settle_actions),
        "action_steps": int(action_steps),
        "sim_steps": int(sim_steps),
        "elapsed_s": round(time.time() - start, 3),
        "success": bool(success_ever or last_debug["success"]),
        "physical_success": bool(physical_success_ever or last_debug["physical_success"]),
        "legacy_success_ever": bool(success_ever),
        "physical_success_ever": bool(physical_success_ever),
        "final_legacy_success": bool(last_debug["success"]),
        "final_physical_success": bool(last_debug["physical_success"]),
        "first_legacy_success_step": first_legacy_success_step,
        "first_physical_success_step": first_physical_success_step,
        "policy_seed": int(args.policy_seed),
        "episode_policy_seed": int(args.policy_seed) + int(episode) if int(args.policy_seed) >= 0 else None,
        "env_reset_seed": int(reset_seed),
        "position_profile": str(args.position_profile),
        "use_env_reset_object_pose": bool(args.use_env_reset_object_pose),
        "reuse_environment": bool(args.reuse_environment),
        "prefix_replay": prefix_replay,
        "prediction_error": error_summary,
        "postprocess": postprocess,
        "action_stats": {
            "pred_or_replay_min": single.round_list(actions_np.min(axis=0), 5),
            "pred_or_replay_max": single.round_list(actions_np.max(axis=0), 5),
            "raw_pred_or_replay_min": single.round_list(raw_actions_np.min(axis=0), 5),
            "raw_pred_or_replay_max": single.round_list(raw_actions_np.max(axis=0), 5),
            "gt_min": single.round_list(gt_np.min(axis=0), 5),
            "gt_max": single.round_list(gt_np.max(axis=0), 5),
        },
        "debug": last_debug,
    }


def summarize(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    final_ok = sum(1 for row in results if row["final_physical_success"])
    ever_ok = sum(1 for row in results if row["physical_success_ever"])
    legacy_final_ok = sum(1 for row in results if row["final_legacy_success"])
    task_summary: dict[str, dict[str, Any]] = {}
    for row in results:
        task_row = task_summary.setdefault(
            row["task"],
            {"total": 0, "final_physical_success": 0, "physical_success_ever": 0},
        )
        task_row["total"] += 1
        task_row["final_physical_success"] += int(bool(row["final_physical_success"]))
        task_row["physical_success_ever"] += int(bool(row["physical_success_ever"]))

    error_keys = ["mae", "joint_mae", "gripper_abs", "max_abs"]
    mean_error = {}
    for key in error_keys:
        values = [
            row["prediction_error"][key]
            for row in results
            if row.get("prediction_error") is not None and key in row["prediction_error"]
        ]
        if values:
            mean_error[key] = float(np.mean(values))

    return {
        "label": args.label,
        "mode": args.mode,
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "policy_path": str(args.policy_path) if args.policy_path is not None else None,
        "episodes": [int(row["episode"]) for row in results],
        "total": len(results),
        "final_physical_success": final_ok,
        "physical_success_ever": ever_ok,
        "final_legacy_success": legacy_final_ok,
        "final_physical_success_text": f"{final_ok}/{len(results)}",
        "physical_success_ever_text": f"{ever_ok}/{len(results)}",
        "task_summary": task_summary,
        "mean_prediction_error": mean_error,
        "postprocess": results[0]["postprocess"] if results else None,
    }


def main() -> int:
    args = parse_args()
    prepare_runtime(args)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.pi0 import PI0Policy
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from mujoco_env.y_env2 import SimpleEnv2

    if args.headless_no_viewer:
        SimpleEnv2.init_viewer = lambda self: self.env.reset()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    episode_column = single.tensor_to_np(dataset.hf_dataset["episode_index"]).astype(int)
    available = sorted({int(x) for x in episode_column.tolist()})
    episodes = parse_episode_spec(args.episodes, available)
    if str(args.env_reset_seeds).strip():
        reset_seeds = [int(part.strip()) for part in str(args.env_reset_seeds).split(",") if part.strip()]
        if len(reset_seeds) != len(episodes):
            raise ValueError(
                f"--env-reset-seeds length ({len(reset_seeds)}) must match selected episodes ({len(episodes)})"
            )
        args._env_reset_seed_map = {int(ep): int(seed) for ep, seed in zip(episodes, reset_seeds)}
    else:
        args._env_reset_seed_map = {}
    args._prefix_replay_map = load_prefix_replay_map(args.prefix_replay_map_json)

    policy = None
    gripper_head = load_gripper_head(args.gripper_head_path)
    if args.mode == "pi0":
        if args.policy_path is None:
            raise ValueError("--policy-path is required for --mode pi0")
        stats_repo_id = args.stats_repo_id or args.repo_id
        stats_dataset_root = args.stats_dataset_root or args.dataset_root
        metadata = LeRobotDatasetMetadata(stats_repo_id, root=stats_dataset_root)
        policy = PI0Policy.from_pretrained(args.policy_path, dataset_stats=metadata.stats)
        policy.to(args.device)
        policy.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    results = []
    shared_env = make_simple_env(args) if args.reuse_environment else None
    try:
        with args.output_jsonl.open("w", encoding="utf-8") as jsonl:
            for episode in episodes:
                result = replay_episode(
                    args,
                    dataset,
                    episode_column,
                    episode,
                    policy,
                    gripper_head,
                    env=shared_env,
                )
                results.append(result)
                episode_json = args.output_dir / f"ep{int(episode):02d}_{args.mode}.json"
                episode_json.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
                jsonl.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if shared_env is not None:
            try:
                shared_env.env.close_viewer()
            except Exception:
                pass

    summary = summarize(args, results)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
