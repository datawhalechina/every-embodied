#!/usr/bin/env python3
"""Run SmolVLA rollouts with physical-success auditing.

The project eval script originally reported SmolVLA's environment success only.
This helper keeps that metric but also requires target-mug lift history and final
uprightness, matching the stricter ACT postmortem convention.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from eval_policy_success import (
    get_body_upright_cos,
    get_smolvla_debug,
    make_smolvla_policy,
    to_tensor_image,
)

PHASES = [
    "initial_open_hold",
    "move_pregrasp",
    "move_grasp",
    "close_gripper",
    "lift_mug",
    "move_preplace",
    "lower_to_plate",
    "pre_release_hold",
    "open_gripper",
    "retreat",
    "final_open_hold",
]
PHASE_SCRIPTED_CLOSED = {
    "close_gripper",
    "lift_mug",
    "move_preplace",
    "lower_to_plate",
    "pre_release_hold",
}


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(values).reshape(-1)]


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
        "stable_place_steps": 0,
    }


def update_tracker(env, tracker: dict, args: argparse.Namespace) -> None:
    p_target = env.env.get_p_body(tracker["target_body"])
    target_z = float(p_target[2])
    tracker["max_target_z"] = max(float(tracker["max_target_z"]), target_z)
    lift = target_z - float(tracker["initial_target_z"])
    tracker["max_target_lift"] = max(float(tracker["max_target_lift"]), lift)
    if lift >= float(args.physical_min_lift):
        tracker["lifted_steps"] += 1


def physical_debug(env, tracker: dict, args: argparse.Namespace, *, count_stability: bool = False) -> dict:
    base = get_smolvla_debug(env)
    target_body = tracker["target_body"]
    p_target = env.env.get_p_body(target_body)
    p_plate = env.env.get_p_body("body_obj_plate_11")
    upright_cos = get_body_upright_cos(env, target_body)
    lifted_enough = int(tracker["lifted_steps"]) >= int(args.physical_min_lift_steps)
    final_upright = bool(np.isfinite(upright_cos) and upright_cos >= float(args.physical_final_upright_cos))
    plate_z_gap = float(abs(float(p_target[2]) - float(p_plate[2])))
    placed_height = plate_z_gap <= float(args.physical_max_plate_z_gap)
    initial_plate = np.asarray(tracker["initial_plate_pos"], dtype=np.float64)
    plate_xy_displacement = float(np.linalg.norm(np.asarray(p_plate[:2]) - initial_plate[:2]))
    plate_stable = plate_xy_displacement <= float(args.physical_max_plate_xy_displacement)
    place_candidate = bool(
        base["success"] and lifted_enough and final_upright and placed_height and plate_stable
    )
    if count_stability:
        if place_candidate:
            tracker["stable_place_steps"] = int(tracker.get("stable_place_steps", 0)) + 1
        else:
            tracker["stable_place_steps"] = 0
    physical_success = bool(
        place_candidate
        and int(tracker.get("stable_place_steps", 0)) >= int(args.physical_stable_place_steps)
    )
    base.update(
        {
            "physical_success": physical_success,
            "physical_lifted_enough": bool(lifted_enough),
            "physical_final_upright": bool(final_upright),
            "physical_min_lift": float(args.physical_min_lift),
            "physical_min_lift_steps": int(args.physical_min_lift_steps),
            "physical_final_upright_cos_threshold": float(args.physical_final_upright_cos),
            "physical_place_candidate": place_candidate,
            "physical_plate_z_gap": plate_z_gap,
            "physical_max_plate_z_gap": float(args.physical_max_plate_z_gap),
            "physical_plate_xy_displacement": plate_xy_displacement,
            "physical_max_plate_xy_displacement": float(args.physical_max_plate_xy_displacement),
            "physical_plate_stable": plate_stable,
            "physical_stable_place_steps": int(tracker.get("stable_place_steps", 0)),
            "physical_required_stable_place_steps": int(args.physical_stable_place_steps),
            "target_body": target_body,
            "initial_target_pos": tracker["initial_target_pos"],
            "initial_plate_pos": tracker["initial_plate_pos"],
            "final_target_pos": round_list(p_target),
            "max_target_z": float(tracker["max_target_z"]),
            "max_target_lift": float(tracker["max_target_lift"]),
            "lifted_steps": int(tracker["lifted_steps"]),
            "final_target_upright_cos": upright_cos,
        }
    )
    return base


def close_env(env) -> None:
    inner = getattr(env, "env", None)
    if inner is not None:
        try:
            inner.close_viewer()
        except Exception:
            pass
    gc.collect()


def write_four_view_frame(env, agent: np.ndarray, wrist: np.ndarray, writer) -> None:
    import cv2

    views = [
        ("Agent", agent),
        ("Egocentric", wrist),
        ("Top", env.env.get_fixed_cam_rgb(cam_name="topview")),
        ("Side", env.env.get_fixed_cam_rgb(cam_name="sideview")),
    ]
    panels = []
    for label, image in views:
        panel = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.rectangle(panel, (0, 0), (150, 28), (20, 20, 20), thickness=-1)
        cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))


def hard_reset_sim_data(env) -> None:
    inner = getattr(env, "env", None)
    if inner is None:
        return
    try:
        inner.reset(step=False)
    except TypeError:
        inner.reset()


def load_action_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    if not args.clamp_action_to_dataset:
        return None
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    action_stats = metadata.stats["action"]
    low = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)[:7]
    high = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)[:7]
    return low, high


def make_pi0_policy_for_dataset(device: str, policy_path: Path, repo_id: str, dataset_root: Path):
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets_pi0"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    policy = PI0Policy.from_pretrained(policy_path, dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()
    return policy


class Pi05PolicyAdapter:
    def __init__(self, policy, preprocessor, postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.config = policy.config

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        processed = self.preprocessor(batch)
        action = self.policy.select_action(processed)
        return self.postprocessor(action)


def make_pi05_policy_for_dataset(
    device: str,
    policy_path: Path,
    repo_id: str,
    dataset_root: Path,
    n_action_steps: int | None = None,
):
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets_pi05"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    config = PreTrainedConfig.from_pretrained(policy_path)
    if config.type != "pi05":
        raise ValueError(f"Expected a Pi0.5 checkpoint, got policy type {config.type!r}")
    config.device = device
    if n_action_steps is not None:
        if n_action_steps <= 0 or n_action_steps > config.chunk_size:
            raise ValueError(
                f"Pi0.5 execution horizon must be in 1..{config.chunk_size}, got {n_action_steps}"
            )
        config.n_action_steps = n_action_steps
    policy = PI05Policy.from_pretrained(
        policy_path,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
    )
    return Pi05PolicyAdapter(policy, preprocessor, postprocessor)


def expected_state_dim(policy) -> int | None:
    config = getattr(policy, "config", None)
    input_features = getattr(config, "input_features", None)
    if input_features is None:
        return None
    feature = input_features.get("observation.state") if hasattr(input_features, "get") else None
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return None
    return int(tuple(shape)[0])


def assert_fair_vla_policy(args: argparse.Namespace, policy) -> None:
    if not args.fair_vla:
        return
    if args.policy_type not in {"pi0", "pi05"}:
        raise ValueError("--fair-vla is only defined for Pi0/Pi0.5 VLA evaluation")
    dim = expected_state_dim(policy)
    if dim != 6:
        raise ValueError(
            f"--fair-vla requires observation.state dim 6 (robot proprioception only), got {dim}. "
            "Do not evaluate phase/timestamp/object-pose conditioned checkpoints as raw VLA."
        )
    blocked: list[str] = []
    if args.pi0_phase_state != "auto":
        blocked.append(f"--pi0-phase-state={args.pi0_phase_state}")
    if args.clamp_action_to_dataset:
        blocked.append("--clamp-action-to-dataset")
    if args.binarize_gripper:
        blocked.append("--binarize-gripper")
    if args.gripper_open_until_step is not None:
        blocked.append("--gripper-open-until-step")
    if args.gripper_hysteresis:
        blocked.append("--gripper-hysteresis")
    if args.pi0_gripper_head_latch:
        blocked.append("--pi0-gripper-head-latch")
    if args.phase_scripted_gripper:
        blocked.append("--phase-scripted-gripper")
    if args.pi0_gripper_head_path is not None:
        blocked.append("--pi0-gripper-head-path")
    if args.pi0_gripper_head_latch_release_head_path is not None:
        blocked.append("--pi0-gripper-head-latch-release-head-path")
    if args.pi0_eef_residual_head_path is not None:
        blocked.append("--pi0-eef-residual-head-path")
    if args.pi0_eef_contact_residual_head_path is not None:
        blocked.append("--pi0-eef-contact-residual-head-path")
    if args.pi0_eef_tail_residual_head_path is not None:
        blocked.append("--pi0-eef-tail-residual-head-path")
    if args.reset_policy_each_action:
        blocked.append("--reset-policy-each-action")
    if blocked:
        raise ValueError(
            "--fair-vla forbids diagnostic/scaffold options that inject non-policy behavior: "
            + ", ".join(blocked)
        )


def load_pi0_gripper_head(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    required = {"weight", "bias", "feature_mean", "feature_std"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Pi0 gripper head {path} is missing fields: {missing}")
    return {
        "path": str(path),
        "weight": np.asarray(data["weight"], dtype=np.float32),
        "bias": float(np.asarray(data["bias"]).reshape(())),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
    }


def pi0_aux_features(action: np.ndarray, args: argparse.Namespace, progress_denom: float) -> tuple[np.ndarray, float]:
    state = np.asarray(getattr(args, "current_observation_state", np.zeros(6)), dtype=np.float32).reshape(-1)[:6]
    if state.size < 6:
        state = np.pad(state, (0, 6 - state.size), mode="constant")
    pred_arm = np.asarray(action[:6], dtype=np.float32).reshape(-1)
    step = float(getattr(args, "current_action_step", 0))
    denom = max(float(progress_denom), 1.0)
    progress = float(np.clip(step / denom, 0.0, 1.0))
    pows = np.asarray([progress, progress**2, progress**3, progress**4, progress**5], dtype=np.float32)
    instruction = str(getattr(args, "current_instruction", "") or "").lower()
    color = np.asarray([1.0 if "blue" in instruction else 0.0, 1.0 if "red" in instruction else 0.0], dtype=np.float32)
    features = np.concatenate([state, pred_arm, pows, color]).astype(np.float32)
    return features, progress


def load_pi0_eef_residual_head(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    required = {"weight", "bias", "feature_mean", "feature_std"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Pi0 EEF residual head {path} is missing fields: {missing}")
    output_mode = "residual"
    if "output_mode" in data.files:
        output_mode = str(np.asarray(data["output_mode"]).reshape(()))
    return {
        "path": str(path),
        "weight": np.asarray(data["weight"], dtype=np.float32),
        "bias": np.asarray(data["bias"], dtype=np.float32).reshape(-1),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
        "output_mode": output_mode,
    }


def load_pi0_visual_contact_head(path: Path | None, device: str) -> dict | None:
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    required = {
        "feature_mean",
        "feature_std",
        "projection",
        "prototypes",
        "distance_threshold",
        "output_min",
        "output_max",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Pi0 visual contact head {path} is missing fields: {missing}")
    projection = np.asarray(data["projection"], dtype=np.float32)
    if projection.ndim != 3 or projection.shape[0] != 2:
        raise ValueError(f"Unexpected visual head projection shape: {projection.shape}")
    output_mode = str(np.asarray(data["output_mode"] if "output_mode" in data.files else "").reshape(()))
    if output_mode not in {"direct_eef_abs_and_gripper", "mlp_direct_eef_abs_and_gripper"}:
        raise ValueError(f"Unsupported Pi0 visual contact head output mode: {output_mode!r}")
    head = {
        "path": str(path),
        "feature_mean": np.asarray(data["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(data["feature_std"], dtype=np.float32),
        "projection": torch.as_tensor(projection, device=device),
        "prototypes": np.asarray(data["prototypes"], dtype=np.float32),
        "distance_threshold": float(np.asarray(data["distance_threshold"]).reshape(())),
        "output_min": np.asarray(data["output_min"], dtype=np.float32).reshape(-1),
        "output_max": np.asarray(data["output_max"], dtype=np.float32).reshape(-1),
        "output_mode": output_mode,
        "include_prev_action": bool(
            np.asarray(data["include_prev_action"]).reshape(()) if "include_prev_action" in data.files else False
        ),
    }
    if output_mode == "direct_eef_abs_and_gripper":
        for key in ("weight", "bias"):
            if key not in data.files:
                raise ValueError(f"Linear Pi0 visual head {path} is missing {key}")
        head["weight"] = np.asarray(data["weight"], dtype=np.float32)
        head["bias"] = np.asarray(data["bias"], dtype=np.float32).reshape(-1)
        head["model_type"] = "linear"
    else:
        mlp_required = {
            "mlp_weight0",
            "mlp_bias0",
            "mlp_weight1",
            "mlp_bias1",
            "mlp_weight2",
            "mlp_bias2",
            "target_mean",
            "target_std",
            "activation",
        }
        mlp_missing = sorted(mlp_required - set(data.files))
        if mlp_missing:
            raise ValueError(f"MLP Pi0 visual head {path} is missing fields: {mlp_missing}")
        if str(np.asarray(data["activation"]).reshape(())) != "gelu_tanh":
            raise ValueError(f"Unsupported MLP activation in {path}")
        for index in range(3):
            head[f"mlp_weight{index}"] = np.asarray(data[f"mlp_weight{index}"], dtype=np.float32)
            head[f"mlp_bias{index}"] = np.asarray(data[f"mlp_bias{index}"], dtype=np.float32).reshape(-1)
        head["target_mean"] = np.asarray(data["target_mean"], dtype=np.float32).reshape(3)
        head["target_std"] = np.asarray(data["target_std"], dtype=np.float32).reshape(3)
        head["model_type"] = "mlp"
    head["gripper_sequence_head"] = None
    if "gripper_sequence_head_type" in data.files:
        sequence_type = str(np.asarray(data["gripper_sequence_head_type"]).reshape(()))
        if sequence_type != "gru":
            raise ValueError(f"Unsupported visual gripper sequence head: {sequence_type!r}")
        sequence_required = {
            "gripper_input_mode",
            "gripper_feature_mean",
            "gripper_feature_std",
            "gripper_input_weight",
            "gripper_input_bias",
            "gripper_norm_weight",
            "gripper_norm_bias",
            "gripper_gru_weight_ih",
            "gripper_gru_weight_hh",
            "gripper_gru_bias_ih",
            "gripper_gru_bias_hh",
            "gripper_output_weight",
            "gripper_output_bias",
            "gripper_threshold",
        }
        sequence_missing = sorted(sequence_required - set(data.files))
        if sequence_missing:
            raise ValueError(f"Pi0 visual gripper GRU {path} is missing fields: {sequence_missing}")
        head["gripper_sequence_head"] = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in sequence_required
            if key not in {"gripper_input_mode"}
        }
        head["gripper_sequence_head"]["gripper_input_mode"] = str(
            np.asarray(data["gripper_input_mode"]).reshape(())
        )
    return head


def gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(coefficient * (x + 0.044715 * x**3)))


def pi0_visual_gripper_gru_step(head: dict, features: np.ndarray, args: argparse.Namespace) -> float | None:
    sequence = head.get("gripper_sequence_head")
    if sequence is None:
        return None
    if sequence["gripper_input_mode"] != "base_plus_prev_action6":
        raise ValueError(f"Unsupported gripper GRU input mode: {sequence['gripper_input_mode']!r}")
    if not head["include_prev_action"] or features.shape[1] < 7:
        raise ValueError("Gripper GRU requires a visual head with previous 7-D action features")

    raw = features[0, :-1]
    mean = sequence["gripper_feature_mean"]
    std = np.where(sequence["gripper_feature_std"] < 1e-6, 1.0, sequence["gripper_feature_std"])
    if raw.shape != mean.shape:
        raise ValueError(f"Gripper GRU feature mismatch: {raw.shape} vs {mean.shape}")
    normalized = (raw - mean) / std
    encoded = normalized @ sequence["gripper_input_weight"].T + sequence["gripper_input_bias"]
    encoded_mean = encoded.mean()
    encoded_var = np.mean((encoded - encoded_mean) ** 2)
    encoded = (encoded - encoded_mean) / np.sqrt(encoded_var + 1e-5)
    encoded = encoded * sequence["gripper_norm_weight"] + sequence["gripper_norm_bias"]
    encoded = gelu_tanh_np(encoded)

    hidden_size = int(sequence["gripper_gru_weight_hh"].shape[1])
    previous_hidden = getattr(args, "pi0_visual_gripper_hidden", None)
    if previous_hidden is None:
        previous_hidden = np.zeros(hidden_size, dtype=np.float32)
    input_gates = encoded @ sequence["gripper_gru_weight_ih"].T + sequence["gripper_gru_bias_ih"]
    hidden_gates = (
        previous_hidden @ sequence["gripper_gru_weight_hh"].T + sequence["gripper_gru_bias_hh"]
    )
    input_reset, input_update, input_new = np.split(input_gates, 3)
    hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3)
    reset = sigmoid_np(input_reset + hidden_reset)
    update = sigmoid_np(input_update + hidden_update)
    candidate = np.tanh(input_new + reset * hidden_new)
    hidden = (1.0 - update) * candidate + update * previous_hidden
    args.pi0_visual_gripper_hidden = hidden.astype(np.float32)
    logit = float(sequence["gripper_output_weight"].reshape(-1) @ hidden) + float(
        sequence["gripper_output_bias"].reshape(-1)[0]
    )
    return float(sigmoid_np(np.asarray([logit], dtype=np.float32))[0])


def pi0_visual_spatial_moments(tokens: torch.Tensor) -> torch.Tensor:
    batch, token_count, _ = tokens.shape
    side = int(round(float(token_count) ** 0.5))
    if side * side != token_count:
        raise ValueError(f"Expected square Pi0 image token grid, got {token_count}")
    coords = torch.linspace(-1.0, 1.0, side, device=tokens.device, dtype=tokens.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx.reshape(1, token_count, 1)
    yy = yy.reshape(1, token_count, 1)
    mean = tokens.mean(dim=1)
    std = tokens.float().std(dim=1, unbiased=False).to(tokens.dtype)
    x_moment = (tokens * xx).mean(dim=1)
    y_moment = (tokens * yy).mean(dim=1)
    return torch.stack([mean, std, x_moment, y_moment], dim=1).reshape(batch, -1)


@torch.inference_mode()
def pi0_visual_contact_features(policy, batch: dict, head: dict, args: argparse.Namespace) -> np.ndarray:
    normalized = policy.normalize_inputs(dict(batch))
    images, image_masks = policy.prepare_images(normalized)
    lang_tokens, lang_masks = policy.prepare_language(normalized)
    image_projection = head["projection"][0]
    image_parts = []
    for image, mask in zip(images, image_masks, strict=False):
        tokens = policy.model.paligemma_with_expert.embed_image(image).float()
        moments = pi0_visual_spatial_moments(tokens).reshape(tokens.shape[0], 4, tokens.shape[-1])
        projected = torch.matmul(moments, image_projection).reshape(tokens.shape[0], -1)
        image_parts.append(projected * mask[:, None].float())

    language = policy.model.paligemma_with_expert.embed_language_tokens(lang_tokens).float()
    language_mask = lang_masks[:, :, None].float()
    language = (language * language_mask).sum(dim=1) / language_mask.sum(dim=1).clamp_min(1.0)
    language = language @ head["projection"][1]
    state = batch["observation.state"].float()[:, :6]
    feature_parts = [*image_parts, language, state]
    if head["include_prev_action"]:
        previous = torch.as_tensor(
            np.asarray(getattr(args, "pi0_prev_action", np.zeros(7)), dtype=np.float32).reshape(1, 7),
            device=state.device,
        )
        feature_parts.append(previous.expand(state.shape[0], -1))
    features = torch.cat(feature_parts, dim=1)
    return features.detach().cpu().numpy().astype(np.float32)


def apply_pi0_visual_contact_head(
    policy,
    batch: dict,
    action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    head = getattr(args, "pi0_visual_contact_head", None)
    if head is None:
        return action, {}
    features = pi0_visual_contact_features(policy, batch, head, args)
    mean = head["feature_mean"]
    std = np.where(head["feature_std"] < 1e-6, 1.0, head["feature_std"])
    if features.shape[1] != mean.shape[0]:
        raise ValueError(f"Pi0 visual head feature mismatch: {features.shape} vs {mean.shape}")
    sequence_gripper_probability = pi0_visual_gripper_gru_step(head, features, args)
    normalized = (features[0] - mean) / std
    nearest = float(np.sqrt(np.mean((head["prototypes"] - normalized[None, :]) ** 2, axis=1).min()))
    threshold = float(head["distance_threshold"]) * float(args.pi0_visual_contact_head_distance_scale)
    active = nearest <= threshold
    if head["model_type"] == "linear":
        pred = normalized @ head["weight"] + head["bias"]
    else:
        hidden = gelu_tanh_np(normalized @ head["mlp_weight0"].T + head["mlp_bias0"])
        hidden = gelu_tanh_np(hidden @ head["mlp_weight1"].T + head["mlp_bias1"])
        raw = hidden @ head["mlp_weight2"].T + head["mlp_bias2"]
        pred = np.concatenate(
            [
                raw[:3] * head["target_std"] + head["target_mean"],
                sigmoid_np(np.asarray(raw[3:4], dtype=np.float32)),
            ]
        )
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    pred[:3] = np.clip(pred[:3], head["output_min"][:3] - 0.02, head["output_max"][:3] + 0.02)
    pred[3] = np.clip(pred[3], 0.0, 1.0)
    info = {
        "pi0_visual_contact_head_path": head["path"],
        "pi0_visual_contact_head_model_type": head["model_type"],
        "pi0_visual_contact_head_include_prev_action": bool(head["include_prev_action"]),
        "pi0_visual_contact_head_nearest_rms": round(nearest, 6),
        "pi0_visual_contact_head_threshold": round(threshold, 6),
        "pi0_visual_contact_head_active": bool(active),
        "pi0_visual_contact_head_monitor_only": bool(args.pi0_visual_contact_head_monitor_only),
        "pi0_visual_contact_head_prediction": round_list(pred, 5),
        "pi0_visual_gripper_sequence_head": "gru" if sequence_gripper_probability is not None else None,
        "pi0_visual_gripper_sequence_probability": round(sequence_gripper_probability, 6)
        if sequence_gripper_probability is not None
        else None,
    }
    if active:
        args.pi0_visual_contact_head_active_steps += 1
    if not active or args.pi0_visual_contact_head_monitor_only:
        return action, info

    blend = float(np.clip(args.pi0_visual_contact_head_blend, 0.0, 1.0))
    adjusted = action.copy()
    adjusted[:3] = (1.0 - blend) * adjusted[:3] + blend * pred[:3]
    adjusted[6] = pred[3] if sequence_gripper_probability is None else sequence_gripper_probability
    if args.pi0_visual_contact_head_binarize_gripper:
        gripper_threshold = float(args.pi0_visual_contact_head_gripper_threshold)
        if sequence_gripper_probability is not None:
            gripper_threshold = float(head["gripper_sequence_head"]["gripper_threshold"].reshape(()))
            if args.pi0_visual_gripper_sequence_threshold is not None:
                gripper_threshold = float(args.pi0_visual_gripper_sequence_threshold)
        adjusted[6] = 1.0 if adjusted[6] >= gripper_threshold else 0.0
        info["pi0_visual_gripper_threshold"] = gripper_threshold
    info["pi0_visual_contact_head_blend"] = blend
    info["pi0_visual_contact_head_raw_action"] = round_list(action, 5)
    info["pi0_visual_contact_head_adjusted_action"] = round_list(adjusted, 5)
    return adjusted.astype(np.float32), info


def apply_loaded_pi0_eef_residual_head(
    action: np.ndarray,
    args: argparse.Namespace,
    info: dict,
    *,
    head: dict,
    progress_denom: float,
    scale: float,
    max_abs: float,
    info_prefix: str,
) -> tuple[np.ndarray, float]:
    features, progress = pi0_aux_features(action, args, progress_denom)
    mean = head["feature_mean"]
    std = np.where(head["feature_std"] < 1e-6, 1.0, head["feature_std"])
    weight = head["weight"]
    bias = head["bias"]
    if features.shape[0] != mean.shape[0] or mean.shape != std.shape or weight.shape[0] != mean.shape[0]:
        raise ValueError(
            "Pi0 EEF residual head feature shape mismatch: "
            f"features={features.shape}, mean={mean.shape}, weight={weight.shape}"
        )
    pred = ((features - mean) / std) @ weight + bias
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if pred.shape[0] < 3:
        raise ValueError(f"Pi0 EEF residual head must output at least 3 values, got {pred.shape}")
    raw_xyz = action[:3].copy()
    if str(head["output_mode"]) == "direct":
        adjusted = pred[:3]
    elif str(head["output_mode"]) == "residual":
        residual = np.clip(pred[:3], -float(max_abs), float(max_abs))
        adjusted = raw_xyz + float(scale) * residual
    else:
        raise ValueError(f"Unsupported Pi0 EEF residual head output_mode={head['output_mode']!r}")
    action[:3] = adjusted.astype(np.float32)
    info[f"{info_prefix}_path"] = head["path"]
    info[f"{info_prefix}_output_mode"] = head["output_mode"]
    info[f"{info_prefix}_progress"] = round(progress, 6)
    info[f"{info_prefix}_scale"] = float(scale)
    info[f"{info_prefix}_max_abs"] = float(max_abs)
    info[f"{info_prefix}_raw_xyz"] = round_list(raw_xyz, 5)
    info[f"{info_prefix}_pred"] = round_list(pred[:3], 5)
    info[f"{info_prefix}_adjusted_xyz"] = round_list(action[:3], 5)
    return action, progress


def apply_pi0_eef_residual_head(action: np.ndarray, args: argparse.Namespace, info: dict) -> np.ndarray:
    head = getattr(args, "pi0_eef_residual_head", None)
    if head is None:
        return action
    action, _progress = apply_loaded_pi0_eef_residual_head(
        action,
        args,
        info,
        head=head,
        progress_denom=args.pi0_eef_residual_head_progress_denom,
        scale=args.pi0_eef_residual_head_scale,
        max_abs=args.pi0_eef_residual_head_max_abs,
        info_prefix="pi0_eef_residual_head",
    )
    return action


def apply_pi0_eef_contact_residual_head(action: np.ndarray, args: argparse.Namespace, info: dict) -> np.ndarray:
    head = getattr(args, "pi0_eef_contact_residual_head", None)
    if head is None:
        return action
    _features, progress = pi0_aux_features(action, args, args.pi0_eef_contact_residual_head_progress_denom)
    info["pi0_eef_contact_residual_head_progress"] = round(progress, 6)
    if progress < float(args.pi0_eef_contact_residual_head_start_progress):
        info["pi0_eef_contact_residual_head_skipped"] = "before_start_progress"
        return action
    end_progress = args.pi0_eef_contact_residual_head_end_progress
    if end_progress is not None and progress > float(end_progress):
        info["pi0_eef_contact_residual_head_skipped"] = "after_end_progress"
        return action
    if (
        args.pi0_eef_contact_residual_head_require_open
        and float(action[6]) >= float(args.pi0_eef_contact_residual_head_open_threshold)
    ):
        info["pi0_eef_contact_residual_head_skipped"] = "gripper_not_open"
        return action
    action, _progress = apply_loaded_pi0_eef_residual_head(
        action,
        args,
        info,
        head=head,
        progress_denom=args.pi0_eef_contact_residual_head_progress_denom,
        scale=args.pi0_eef_contact_residual_head_scale,
        max_abs=args.pi0_eef_contact_residual_head_max_abs,
        info_prefix="pi0_eef_contact_residual_head",
    )
    return action


def apply_pi0_eef_tail_residual_head(action: np.ndarray, args: argparse.Namespace, info: dict) -> np.ndarray:
    head = getattr(args, "pi0_eef_tail_residual_head", None)
    if head is None:
        return action
    _features, progress = pi0_aux_features(action, args, args.pi0_eef_tail_residual_head_progress_denom)
    info["pi0_eef_tail_residual_head_progress"] = round(progress, 6)
    if progress < float(args.pi0_eef_tail_residual_head_start_progress):
        info["pi0_eef_tail_residual_head_skipped"] = "before_start_progress"
        return action
    if (
        args.pi0_eef_tail_residual_head_require_closed
        and float(action[6]) < float(args.pi0_eef_tail_residual_head_closed_threshold)
    ):
        info["pi0_eef_tail_residual_head_skipped"] = "gripper_not_closed"
        return action
    action, _progress = apply_loaded_pi0_eef_residual_head(
        action,
        args,
        info,
        head=head,
        progress_denom=args.pi0_eef_tail_residual_head_progress_denom,
        scale=args.pi0_eef_tail_residual_head_scale,
        max_abs=args.pi0_eef_tail_residual_head_max_abs,
        info_prefix="pi0_eef_tail_residual_head",
    )
    return action


def apply_pi0_gripper_head(action: np.ndarray, args: argparse.Namespace, info: dict) -> np.ndarray:
    head = getattr(args, "pi0_gripper_head", None)
    if head is None:
        return action
    prob, progress = predict_pi0_gripper_head_prob(
        head,
        action,
        args,
        progress_denom=args.pi0_gripper_head_progress_denom,
    )
    threshold = float(args.pi0_gripper_head_threshold)
    instruction = str(getattr(args, "current_instruction", "") or "").lower()
    if args.pi0_gripper_head_blue_threshold is not None and "blue" in instruction:
        threshold = float(args.pi0_gripper_head_blue_threshold)
    if args.pi0_gripper_head_red_threshold is not None and "red" in instruction:
        threshold = float(args.pi0_gripper_head_red_threshold)
    if args.pi0_gripper_head_continuous:
        action[6] = prob
    else:
        action[6] = 1.0 if prob >= threshold else 0.0
    info["pi0_gripper_head_path"] = head["path"]
    info["pi0_gripper_head_prob"] = round(prob, 6)
    info["pi0_gripper_head_threshold"] = threshold
    info["pi0_gripper_head_progress"] = round(progress, 6)
    info["pi0_gripper_head_continuous"] = bool(args.pi0_gripper_head_continuous)
    return action


def predict_pi0_gripper_head_prob(
    head: dict,
    action: np.ndarray,
    args: argparse.Namespace,
    *,
    progress_denom: float,
) -> tuple[float, float]:
    features, progress = pi0_aux_features(action, args, progress_denom)
    mean = head["feature_mean"]
    std = np.where(head["feature_std"] < 1e-6, 1.0, head["feature_std"])
    if features.shape[0] != mean.shape[0] or mean.shape != std.shape or mean.shape != head["weight"].shape:
        raise ValueError(
            "Pi0 gripper head feature shape mismatch: "
            f"features={features.shape}, mean={mean.shape}, weight={head['weight'].shape}"
        )
    prob = float(sigmoid_np(np.asarray([((features - mean) / std) @ head["weight"] + head["bias"]]))[0])
    return prob, progress


def apply_pi0_gripper_head_latch(action: np.ndarray, args: argparse.Namespace, info: dict) -> np.ndarray:
    if not getattr(args, "pi0_gripper_head_latch", False):
        return action
    step = int(getattr(args, "current_action_step", 0))
    release_step = getattr(args, "pi0_gripper_head_latch_release_step", None)
    release_by_step = release_step is not None and step >= int(release_step)
    release_by_prob = False
    release_prob_below = getattr(args, "pi0_gripper_head_latch_release_prob_below", None)
    release_min_step = int(getattr(args, "pi0_gripper_head_latch_release_min_step", 0))
    if release_prob_below is not None and step >= release_min_step and "pi0_gripper_head_prob" in info:
        release_by_prob = float(info["pi0_gripper_head_prob"]) <= float(release_prob_below)
    release_head = getattr(args, "pi0_gripper_head_latch_release_head", None)
    release_by_head = False
    if release_head is not None and step >= release_min_step:
        release_head_prob, release_head_progress = predict_pi0_gripper_head_prob(
            release_head,
            action,
            args,
            progress_denom=args.pi0_gripper_head_latch_release_head_progress_denom,
        )
        info["pi0_gripper_head_latch_release_head_path"] = release_head["path"]
        info["pi0_gripper_head_latch_release_head_prob"] = round(release_head_prob, 6)
        info["pi0_gripper_head_latch_release_head_progress"] = round(release_head_progress, 6)
        release_by_head = release_head_prob <= float(args.pi0_gripper_head_latch_release_head_prob_below)
    if release_by_step or release_by_prob or release_by_head:
        args.pi0_gripper_head_latch_active = False
        info["pi0_gripper_head_latch_released"] = True
        if release_by_step:
            reason = "step"
        elif release_by_prob:
            reason = "prob"
        else:
            reason = "release_head"
        info["pi0_gripper_head_latch_release_reason"] = reason
    elif float(action[6]) >= float(args.pi0_gripper_head_latch_close_threshold):
        args.pi0_gripper_head_latch_active = True
    if getattr(args, "pi0_gripper_head_latch_active", False):
        action[6] = float(args.pi0_gripper_head_latch_closed_value)
    info["pi0_gripper_head_latch"] = True
    info["pi0_gripper_head_latch_active"] = bool(getattr(args, "pi0_gripper_head_latch_active", False))
    info["pi0_gripper_head_latch_release_step"] = release_step
    info["pi0_gripper_head_latch_release_prob_below"] = release_prob_below
    info["pi0_gripper_head_latch_release_min_step"] = release_min_step
    return action


def phase_feature(phase_index: int) -> np.ndarray:
    phase_index = int(np.clip(phase_index, 0, len(PHASES) - 1))
    onehot = np.zeros(len(PHASES), dtype=np.float32)
    onehot[phase_index] = 1.0
    denom = max(len(PHASES) - 1, 1)
    return np.concatenate([np.asarray([phase_index / denom], dtype=np.float32), onehot])


def phase_index_from_state(row: np.ndarray) -> int | None:
    state = np.asarray(row, dtype=np.float32).reshape(-1)
    if state.shape[0] >= 8 + len(PHASES):
        onehot = state[8 : 8 + len(PHASES)]
        if onehot.shape[0] == len(PHASES) and float(np.max(onehot)) >= 0.5:
            return int(np.argmax(onehot))
    if state.shape[0] >= 8:
        denom = max(len(PHASES) - 1, 1)
        return int(np.clip(round(float(state[7]) * denom), 0, len(PHASES) - 1))
    return None


def set_current_phase(args: argparse.Namespace, phase_index: int | None) -> None:
    if phase_index is None:
        args.current_phase_index = None
        args.current_phase_name = None
        return
    phase_index = int(np.clip(phase_index, 0, len(PHASES) - 1))
    args.current_phase_index = phase_index
    args.current_phase_name = PHASES[phase_index]


def parse_episode_list(spec: str | None, seeds: list[int]) -> list[int]:
    if spec is None or not str(spec).strip():
        return list(range(len(seeds)))
    episode_ids = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(episode_ids) != len(seeds):
        raise ValueError(
            f"--pi0-phase-schedule-episodes has {len(episode_ids)} entries, "
            f"but {len(seeds)} seeds were requested"
        )
    return episode_ids


def load_pi0_phase_schedule_states(
    args: argparse.Namespace,
    seeds: list[int],
    policy,
) -> dict[int, np.ndarray]:
    if args.policy_type != "pi0" or args.pi0_phase_state != "dataset_schedule":
        return {}

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    episode_ids = parse_episode_list(args.pi0_phase_schedule_episodes, seeds)
    expected_dim_value = expected_state_dim(policy)
    state_column = dataset.hf_dataset["observation.state"]
    schedules: dict[int, np.ndarray] = {}
    for seed, episode_id in zip(seeds, episode_ids):
        if episode_id < 0 or episode_id >= dataset.num_episodes:
            raise ValueError(f"Episode {episode_id} is outside dataset range 0..{dataset.num_episodes - 1}")
        start = int(dataset.episode_data_index["from"][episode_id].item())
        end = int(dataset.episode_data_index["to"][episode_id].item())
        states = [
            np.asarray(state_column[idx], dtype=np.float32).reshape(-1)
            for idx in range(start, end)
        ]
        if not states:
            raise ValueError(f"Episode {episode_id} has no frames")
        schedule = np.stack(states).astype(np.float32)
        if expected_dim_value is not None and schedule.shape[1] != int(expected_dim_value):
            raise ValueError(
                f"Episode {episode_id} state dim {schedule.shape[1]} does not match "
                f"policy expected dim {expected_dim_value}"
            )
        schedules[int(seed)] = schedule
    return schedules


class ScriptedPhaseTracker:
    """Small finite-state phase tracker matching the scripted reset oracle."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.phase_index = 0
        self.phases = [
            ("initial_open_hold", "hold", int(args.initial_hold)),
            ("move_pregrasp", "move", 0),
            ("move_grasp", "move", 0),
            ("close_gripper", "hold", int(args.close_hold)),
            ("lift_mug", "move", 0),
            ("move_preplace", "move", 0),
            ("lower_to_plate", "move", 0),
            ("pre_release_hold", "hold", int(args.pre_release_hold)),
            ("open_gripper", "hold", int(args.open_hold)),
            ("retreat", "move", 0),
            ("final_open_hold", "hold", int(args.final_hold)),
        ]
        self.hold_remaining = self.phases[0][2]

    def _target_for_phase(self, points: dict, phase: str):
        mapping = {
            "move_pregrasp": "pregrasp",
            "move_grasp": "grasp",
            "lift_mug": "lift",
            "move_preplace": "preplace",
            "lower_to_plate": "release",
            "retreat": "retreat",
        }
        key = mapping.get(phase)
        return None if key is None else points[key]

    def observe(self, env) -> int:
        try:
            from collect_pi0_dagger_tail import target_positions
        except Exception as exc:
            raise RuntimeError("Phase-conditioned Pi0 eval requires collect_pi0_dagger_tail.py on PYTHONPATH") from exc

        points = target_positions(env, self.args)
        transitions = 0
        while transitions < 20:
            if self.phase_index >= len(self.phases):
                return len(PHASES) - 1
            phase, kind, hold_steps = self.phases[self.phase_index]
            if kind == "hold":
                if hold_steps <= 0:
                    self.phase_index += 1
                    if self.phase_index < len(self.phases):
                        self.hold_remaining = self.phases[self.phase_index][2]
                    transitions += 1
                    continue
                current_index = self.phase_index
                self.hold_remaining -= 1
                if self.hold_remaining <= 0:
                    self.phase_index += 1
                    if self.phase_index < len(self.phases):
                        self.hold_remaining = self.phases[self.phase_index][2]
                return current_index

            target = self._target_for_phase(points, phase)
            current = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)
            dist = float(np.linalg.norm(np.asarray(target, dtype=np.float32) - current))
            if dist <= float(self.args.pos_tol):
                self.phase_index += 1
                if self.phase_index < len(self.phases):
                    self.hold_remaining = self.phases[self.phase_index][2]
                transitions += 1
                continue
            return self.phase_index
        raise RuntimeError("Phase tracker transition loop did not settle")


def observation_state_for_policy(
    env,
    policy,
    action_step: int,
    hz: float,
    args: argparse.Namespace,
    phase_tracker: ScriptedPhaseTracker | None,
    scheduled_states: np.ndarray | None = None,
) -> np.ndarray:
    base_state = np.asarray(env.get_joint_state()[:6], dtype=np.float32).reshape(-1)
    dim = expected_state_dim(policy)
    if dim is None or dim == base_state.shape[0]:
        set_current_phase(args, None)
        return base_state
    if dim == base_state.shape[0] + 1 and args.pi0_state_extra == "gripper":
        debug = get_smolvla_debug(env)
        gripper = np.asarray([np.clip(float(debug.get("gripper", 0.0)), 0.0, 1.0)], dtype=np.float32)
        set_current_phase(args, None)
        return np.concatenate([base_state, gripper])
    if dim == base_state.shape[0] + 7 and args.pi0_state_extra == "prev_action":
        prev_action = np.asarray(getattr(args, "pi0_prev_action", np.zeros(7, dtype=np.float32)), dtype=np.float32)
        set_current_phase(args, None)
        return np.concatenate([base_state, prev_action.reshape(7)])
    if dim == base_state.shape[0] + 8 and args.pi0_state_extra == "gripper_prev_action":
        debug = get_smolvla_debug(env)
        gripper = np.asarray([np.clip(float(debug.get("gripper", 0.0)), 0.0, 1.0)], dtype=np.float32)
        prev_action = np.asarray(getattr(args, "pi0_prev_action", np.zeros(7, dtype=np.float32)), dtype=np.float32)
        set_current_phase(args, None)
        return np.concatenate([base_state, gripper, prev_action.reshape(7)])
    if dim == base_state.shape[0] + 2 and args.pi0_state_extra == "gripper_time":
        debug = get_smolvla_debug(env)
        gripper = np.asarray([np.clip(float(debug.get("gripper", 0.0)), 0.0, 1.0)], dtype=np.float32)
        timestamp = np.asarray([float(action_step) / float(hz)], dtype=np.float32)
        set_current_phase(args, None)
        return np.concatenate([base_state, gripper, timestamp])
    if args.pi0_phase_state == "dataset_schedule":
        if scheduled_states is None or len(scheduled_states) == 0:
            raise ValueError("--pi0-phase-state=dataset_schedule requires per-seed scheduled states")
        row = np.asarray(scheduled_states[min(int(action_step), len(scheduled_states) - 1)], dtype=np.float32).reshape(-1)
        if row.shape[0] != dim:
            raise ValueError(f"Scheduled state dim {row.shape[0]} does not match policy expected dim {dim}")
        set_current_phase(args, phase_index_from_state(row))
        if dim <= base_state.shape[0]:
            return row
        return np.concatenate([base_state, row[base_state.shape[0]:]])
    timestamp = np.asarray([float(action_step) / float(hz)], dtype=np.float32)
    with_timestamp = np.concatenate([base_state, timestamp])
    if dim == with_timestamp.shape[0]:
        set_current_phase(args, None)
        return with_timestamp
    with_zero_phase = np.concatenate([with_timestamp, phase_feature(0)])
    if dim == with_zero_phase.shape[0]:
        if args.pi0_phase_state == "zeros":
            set_current_phase(args, 0)
            return with_zero_phase
        if phase_tracker is None:
            raise ValueError("Phase-conditioned state requested, but no phase tracker is available")
        phase_index = phase_tracker.observe(env)
        set_current_phase(args, phase_index)
        return np.concatenate([with_timestamp, phase_feature(phase_index)])
    raise ValueError(
        f"Unsupported observation.state dim {dim}; available base dim is "
        f"{base_state.shape[0]}, timestamp dim is {with_timestamp.shape[0]}, "
        f"and timestamp+phase dim is {with_zero_phase.shape[0]}"
    )


def postprocess_action(
    action: np.ndarray,
    args: argparse.Namespace,
    action_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)[:7].copy()
    processed = raw.copy()
    info: dict = {}
    if action_bounds is not None:
        low, high = action_bounds
        processed = np.clip(processed, low, high)
        info["clamped"] = True
    if args.clip_gripper:
        processed[6] = np.clip(processed[6], 0.0, 1.0)
        info["clip_gripper"] = True
    if args.binarize_gripper:
        processed[6] = 1.0 if processed[6] >= float(args.gripper_threshold) else 0.0
        info["binarize_gripper"] = True
        info["gripper_threshold"] = float(args.gripper_threshold)
    if args.gripper_open_until_step is not None and args.current_action_step < int(args.gripper_open_until_step):
        processed[6] = 0.0
        info["gripper_open_until_step"] = int(args.gripper_open_until_step)
    if args.gripper_hysteresis:
        if processed[6] >= float(args.gripper_hysteresis_close_threshold):
            args.gripper_hysteresis_active = True
        if (
            args.gripper_hysteresis_release_step is not None
            and args.current_action_step >= int(args.gripper_hysteresis_release_step)
        ):
            args.gripper_hysteresis_active = False
        if getattr(args, "gripper_hysteresis_active", False):
            processed[6] = float(args.gripper_hysteresis_closed_value)
        info["gripper_hysteresis"] = True
        info["gripper_hysteresis_active"] = bool(getattr(args, "gripper_hysteresis_active", False))
    if args.phase_scripted_gripper:
        phase_name = getattr(args, "current_phase_name", None)
        if phase_name is None:
            raise ValueError("--phase-scripted-gripper requires a phase-conditioned Pi0 state")
        processed[6] = 1.0 if str(phase_name) in PHASE_SCRIPTED_CLOSED else 0.0
        info["phase_scripted_gripper"] = True
        info["phase_name"] = str(phase_name)
    if args.policy_type == "pi0":
        processed = apply_pi0_eef_residual_head(processed, args, info)
        processed = apply_pi0_gripper_head(processed, args, info)
        processed = apply_pi0_eef_contact_residual_head(processed, args, info)
        processed = apply_pi0_gripper_head_latch(processed, args, info)
        processed = apply_pi0_eef_tail_residual_head(processed, args, info)
    info["raw_action"] = round_list(raw, 5)
    info["processed_action"] = round_list(processed, 5)
    return processed, info


def action_for_environment(action: np.ndarray, env, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    policy_action = np.asarray(action, dtype=np.float32).reshape(-1)[:7].copy()
    env_action = policy_action.copy()
    info: dict = {"pi0_action_mode": args.pi0_action_mode}
    if args.policy_type in {"pi0", "pi05"} and args.pi0_action_mode == "joint_delta":
        state = np.asarray(env.get_joint_state()[:6], dtype=np.float32).reshape(-1)
        env_action[:6] = state[:6] + policy_action[:6]
        env_action[6] = policy_action[6]
        info["joint_state"] = round_list(state, 5)
        info["policy_delta_action"] = round_list(policy_action, 5)
        info["env_action"] = round_list(env_action, 5)
    elif args.policy_type in {"pi0", "pi05"} and args.pi0_action_mode == "eef_delta":
        info["policy_eef_delta_action"] = round_list(policy_action, 5)
        if args.eef_delta_max_step is not None:
            if args.eef_delta_max_step <= 0:
                raise ValueError("--eef-delta-max-step must be positive")
            env_action[:3] = np.clip(
                env_action[:3],
                -float(args.eef_delta_max_step),
                float(args.eef_delta_max_step),
            )
            info["eef_delta_max_step"] = float(args.eef_delta_max_step)
        info["env_action"] = round_list(env_action, 5)
    elif args.policy_type in {"pi0", "pi05"} and args.pi0_action_mode == "eef_abs":
        current_tcp = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32).reshape(3)
        target_tcp = policy_action[:3].copy()
        env_action = np.zeros(7, dtype=np.float32)
        env_action[:3] = np.clip(
            target_tcp - current_tcp,
            -float(args.eef_abs_max_step),
            float(args.eef_abs_max_step),
        )
        env_action[6] = policy_action[6]
        info["current_tcp"] = round_list(current_tcp, 5)
        info["policy_eef_abs_target"] = round_list(policy_action, 5)
        info["env_eef_delta_action"] = round_list(env_action, 5)
    elif args.pi0_action_mode != "absolute":
        raise ValueError(f"Unsupported pi0 action mode: {args.pi0_action_mode}")
    return env_action, info


def resolve_pi0_sample_reducer(args: argparse.Namespace, instruction: str | None = None) -> str:
    reducer = str(args.pi0_action_sample_reducer)
    if reducer == "instruction_color_blue_sample1_gmean_gate":
        text = (instruction or "").lower()
        if "blue" in text:
            return "sample1_gmean_gate"
        if "red" in text:
            return "mean"
        return "mean"
    if reducer.startswith("instruction_color_blue_sample"):
        text = (instruction or "").lower()
        if "blue" in text:
            return reducer.replace("instruction_color_blue_", "", 1)
        if "red" in text:
            return "mean"
        return "mean"
    if reducer != "instruction_color":
        return reducer
    text = (instruction or "").lower()
    if "blue" in text:
        return "median"
    if "red" in text:
        return "mean"
    return "mean"


def write_pi0_sample_log(
    args: argparse.Namespace,
    batch: dict,
    stacked: torch.Tensor,
    actions: torch.Tensor,
    reducer: str,
    info: dict,
    instruction: str | None,
) -> None:
    log_path = getattr(args, "pi0_sample_log_jsonl", None)
    if log_path is None:
        return

    chunks = stacked[:, 0].detach().float().cpu().numpy()
    selected = actions[0].detach().float().cpu().numpy()
    offset = max(int(getattr(args, "pi0_action_chunk_offset", 0)), 0)
    if offset >= chunks.shape[1]:
        raise ValueError(
            f"--pi0-action-chunk-offset={offset} is outside the sampled chunk length {chunks.shape[1]}"
        )
    window = max(int(getattr(args, "pi0_exec_chunk_steps", 0)), 0)
    available = chunks.shape[1] - offset
    if window <= 0 or window > available:
        window = available
    window_chunks = chunks[:, offset : offset + window, :]
    window_selected = selected[offset : offset + window, :]
    center_mean = window_chunks.mean(axis=0)
    center_median = np.median(window_chunks, axis=0)
    flat = window_chunks.reshape(window_chunks.shape[0], -1)
    pairwise = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    state_tensor = batch.get("observation.state")
    if state_tensor is None:
        obs_state = []
    else:
        obs_state = state_tensor[0].detach().float().cpu().numpy()

    candidates = []
    for sample_index, chunk in enumerate(window_chunks):
        gripper = chunk[:, 6] if chunk.shape[1] >= 7 else np.asarray([], dtype=np.float32)
        close_hits = np.flatnonzero(gripper >= 0.5) if gripper.size else np.asarray([], dtype=np.int64)
        xyz = chunk[:, :3] if chunk.shape[1] >= 3 else np.zeros((chunk.shape[0], 0), dtype=np.float32)
        if xyz.shape[1] == 3 and xyz.shape[0] > 1:
            xyz_path_len = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            xyz_total_delta = float(np.linalg.norm(xyz[-1] - xyz[0]))
        else:
            xyz_path_len = 0.0
            xyz_total_delta = 0.0
        candidates.append(
            {
                "sample_index": int(sample_index),
                "first_action": round_list(chunk[0, : min(7, chunk.shape[1])], 5),
                "last_action": round_list(chunk[-1, : min(7, chunk.shape[1])], 5),
                "chunk_mean": round_list(chunk[:, : min(7, chunk.shape[1])].mean(axis=0), 5),
                "chunk_std": round_list(chunk[:, : min(7, chunk.shape[1])].std(axis=0), 5),
                "gripper_min": round(float(gripper.min()), 5) if gripper.size else None,
                "gripper_max": round(float(gripper.max()), 5) if gripper.size else None,
                "gripper_mean": round(float(gripper.mean()), 5) if gripper.size else None,
                "first_close_offset": int(close_hits[0]) if close_hits.size else None,
                "xyz_path_len": round(xyz_path_len, 6),
                "xyz_total_delta": round(xyz_total_delta, 6),
                "dist_to_mean": round(float(np.linalg.norm(chunk - center_mean)), 6),
                "dist_to_median": round(float(np.linalg.norm(chunk - center_median)), 6),
                "pairwise_mean_dist": round(float(pairwise[sample_index].mean()), 6),
                "pairwise_min_dist": round(float(np.min(np.delete(pairwise[sample_index], sample_index))), 6)
                if window_chunks.shape[0] > 1
                else 0.0,
            }
        )

    selected_gripper = window_selected[:, 6] if window_selected.shape[1] >= 7 else np.asarray([], dtype=np.float32)
    selected_close_hits = np.flatnonzero(selected_gripper >= 0.5) if selected_gripper.size else np.asarray([], dtype=np.int64)
    row = {
        "event": "pi0_sample_chunk",
        "seed": int(getattr(args, "current_seed", -1)),
        "action_step": int(getattr(args, "current_action_step", -1)),
        "instruction": instruction,
        "pi0_action_mode": args.pi0_action_mode,
        "pi0_state_extra": args.pi0_state_extra,
        "pi0_action_samples": int(getattr(args, "pi0_action_samples", 1)),
        "pi0_action_sample_reducer": str(args.pi0_action_sample_reducer),
        "pi0_action_sample_reducer_resolved": reducer,
        "pi0_exec_chunk_steps": int(getattr(args, "pi0_exec_chunk_steps", 0)),
        "pi0_action_chunk_offset": int(offset),
        "logged_chunk_steps": int(window),
        "observation_state": round_list(obs_state, 5),
        "selected_first_action": round_list(window_selected[0, : min(7, window_selected.shape[1])], 5),
        "selected_chunk_mean": round_list(window_selected[:, : min(7, window_selected.shape[1])].mean(axis=0), 5),
        "selected_gripper_min": round(float(selected_gripper.min()), 5) if selected_gripper.size else None,
        "selected_gripper_max": round(float(selected_gripper.max()), 5) if selected_gripper.size else None,
        "selected_gripper_mean": round(float(selected_gripper.mean()), 5) if selected_gripper.size else None,
        "selected_first_close_offset": int(selected_close_hits[0]) if selected_close_hits.size else None,
        "sample_fixed_index": info.get("sample_fixed_index"),
        "sample1_gmean_gate_decision": info.get("sample1_gmean_gate_decision"),
        "sample1_gmean_gate_first_value": info.get("sample1_gmean_gate_first_value"),
        "sample1_gmean_gate_threshold": info.get("sample1_gmean_gate_threshold"),
        "sample_medoid_indices": info.get("sample_medoid_indices"),
        "sample_medoid_dist": info.get("sample_medoid_dist"),
        "candidates": candidates,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def select_pi0_action(
    policy,
    batch: dict,
    args: argparse.Namespace,
    instruction: str | None = None,
) -> tuple[torch.Tensor, dict]:
    samples = int(getattr(args, "pi0_action_samples", 1))
    exec_chunk_steps = int(getattr(args, "pi0_exec_chunk_steps", 0))
    chunk_offset = int(getattr(args, "pi0_action_chunk_offset", 0))
    if chunk_offset < 0:
        raise ValueError(f"--pi0-action-chunk-offset must be non-negative, got {chunk_offset}")
    if samples <= 1 and exec_chunk_steps <= 0 and chunk_offset <= 0:
        return policy.select_action(batch), {"pi0_action_samples": 1}

    policy.eval()
    reducer = resolve_pi0_sample_reducer(args, instruction)
    info = {
        "pi0_action_samples": samples,
        "pi0_action_sample_reducer": str(args.pi0_action_sample_reducer),
        "pi0_action_sample_reducer_resolved": reducer,
        "pi0_exec_chunk_steps": exec_chunk_steps,
        "pi0_action_chunk_offset": chunk_offset,
    }

    if len(policy._action_queue) > 0:
        info["sampled_new_chunk"] = False
        return policy._action_queue.popleft(), info

    from lerobot.common.constants import OBS_STATE

    if policy.config.adapt_to_pi_aloha:
        batch = dict(batch)
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])

    normalized = policy.normalize_inputs(batch)
    images, img_masks = policy.prepare_images(normalized)
    state = policy.prepare_state(normalized)
    lang_tokens, lang_masks = policy.prepare_language(normalized)

    chunks = []
    for _ in range(max(samples, 1)):
        actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None)
        original_action_dim = policy.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = policy.unnormalize_outputs({"action": actions})["action"]
        if policy.config.adapt_to_pi_aloha:
            actions = policy._pi_aloha_encode_actions(actions)
        chunks.append(actions)

    stacked = torch.stack(chunks, dim=0)
    if samples <= 1:
        actions = stacked[0]
    elif reducer == "mean":
        actions = stacked.mean(dim=0)
    elif reducer == "median":
        actions = stacked.median(dim=0).values
    elif reducer == "sample1_gmean_gate":
        if stacked.shape[0] <= 1:
            raise ValueError("Reducer 'sample1_gmean_gate' requires --pi0-action-samples >= 2")
        if stacked.shape[-1] < 7:
            raise ValueError("Reducer 'sample1_gmean_gate' requires a gripper action at index 6")
        window = max(exec_chunk_steps, 0)
        if window <= 0 or window > stacked.shape[2]:
            window = stacked.shape[2]
        threshold = float(getattr(args, "pi0_sample1_gmean_threshold", 0.01696))
        action_step = int(getattr(args, "current_action_step", 0))
        if action_step <= 0 or getattr(args, "pi0_sample1_gmean_gate_decision", None) is None:
            sample1_gmean = float(stacked[1, 0, :window, 6].detach().float().mean().cpu().item())
            decision = "sample1" if sample1_gmean > threshold else "median"
            args.pi0_sample1_gmean_gate_decision = decision
            args.pi0_sample1_gmean_gate_first_value = sample1_gmean
            args.pi0_sample1_gmean_gate_first_step = action_step
        else:
            decision = str(args.pi0_sample1_gmean_gate_decision)
            sample1_gmean = float(getattr(args, "pi0_sample1_gmean_gate_first_value", float("nan")))
        if decision == "sample1":
            actions = stacked[1]
            info["sample_fixed_index"] = 1
        elif decision == "median":
            actions = stacked.median(dim=0).values
        else:
            raise ValueError(f"Unsupported sample1_gmean_gate decision: {decision!r}")
        info["sample1_gmean_gate_decision"] = decision
        info["sample1_gmean_gate_first_value"] = round(sample1_gmean, 7)
        info["sample1_gmean_gate_threshold"] = threshold
        info["sample1_gmean_gate_first_step"] = int(getattr(args, "pi0_sample1_gmean_gate_first_step", action_step))
    elif reducer.startswith("sample"):
        sample_index = int(reducer.replace("sample", "", 1))
        if sample_index < 0 or sample_index >= stacked.shape[0]:
            raise ValueError(
                f"Reducer {reducer!r} requires --pi0-action-samples > {sample_index}, "
                f"got {stacked.shape[0]}"
            )
        actions = stacked[sample_index]
        info["sample_fixed_index"] = sample_index
    elif reducer in {"medoid_mean", "medoid_median"}:
        if reducer == "medoid_mean":
            center = stacked.mean(dim=0)
        else:
            center = stacked.median(dim=0).values
        # Pick one sampled chunk closest to the center, preserving a coherent Pi0 trajectory.
        dist = (stacked.float() - center.unsqueeze(0).float()).pow(2).flatten(2).mean(dim=2)
        best = dist.argmin(dim=0)
        batch_indices = torch.arange(stacked.shape[1], device=stacked.device)
        actions = stacked[best, batch_indices]
        info["sample_medoid_indices"] = [int(x) for x in best.detach().cpu().tolist()]
        info["sample_medoid_dist"] = [round(float(x), 7) for x in dist[best, batch_indices].detach().cpu().tolist()]
    else:
        raise ValueError(f"Unsupported --pi0-action-sample-reducer={args.pi0_action_sample_reducer}")

    first_actions = stacked[:, 0, 0, : min(7, stacked.shape[-1])].detach().float().cpu().numpy()
    info["sampled_new_chunk"] = True
    if samples > 1:
        info["sample_first_action_std"] = round_list(first_actions.std(axis=0), 5)
        info["sample_first_action_min"] = round_list(first_actions.min(axis=0), 5)
        info["sample_first_action_max"] = round_list(first_actions.max(axis=0), 5)
    write_pi0_sample_log(args, batch, stacked, actions, reducer, info, instruction)

    queued_actions = actions.transpose(0, 1)
    if chunk_offset >= queued_actions.shape[0]:
        raise ValueError(
            f"--pi0-action-chunk-offset={chunk_offset} is outside the sampled chunk length "
            f"{queued_actions.shape[0]}"
        )
    queued_actions = queued_actions[chunk_offset:]
    if exec_chunk_steps > 0:
        queued_actions = queued_actions[:exec_chunk_steps]
    if queued_actions.shape[0] <= 0:
        raise ValueError("Pi0 action queue is empty after applying chunk offset/window")
    policy._action_queue.extend(queued_actions)
    return policy._action_queue.popleft(), info


def configure_env_action_space(env, args: argparse.Namespace) -> None:
    if args.policy_type in {"pi0", "pi05"} and args.pi0_action_mode in ("eef_delta", "eef_abs"):
        from mujoco_env.transforms import rpy2r

        env.action_type = "eef_pose"
        env.p0, _ = env.env.get_pR_body(body_name="tcp_link")
        env.R0 = rpy2r(np.deg2rad([90.0, 0.0, 90.0]))
    else:
        env.action_type = "joint_angle"


def rollout(
    args: argparse.Namespace,
    policy,
    seed: int,
    env=None,
    scheduled_states: np.ndarray | None = None,
) -> dict:
    from mujoco_env.y_env2 import SimpleEnv2

    owns_env = env is None
    video_writer = None
    video_path: Path | None = None
    if env is None:
        env = SimpleEnv2(
            "./asset/example_scene_y2.xml",
            action_type="joint_angle",
            position_profile=args.position_profile,
        )
    try:
        if args.hard_reset_sim_data:
            hard_reset_sim_data(env)
        env_seed = int(seed) if args.fixed_env_seed is None else int(args.fixed_env_seed)
        env.reset(seed=env_seed)
        if args.instruction:
            env.set_instruction(args.instruction)
        policy_seed = int(seed) + int(args.policy_seed_offset)
        random.seed(policy_seed)
        np.random.seed(policy_seed)
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        configure_env_action_space(env, args)
        policy.reset()
        phase_tracker = (
            ScriptedPhaseTracker(args)
            if args.policy_type == "pi0" and args.pi0_phase_state in ("auto", "dynamic_oracle")
            else None
        )
        tracker = init_tracker(env)
        action_steps = 0
        sim_steps = 0
        success_ever = False
        physical_success_ever = False
        first_success_step = None
        first_physical_success_step = None
        args.current_seed = int(seed)
        args.pi0_prev_action = np.zeros(7, dtype=np.float32)
        args.pi0_visual_gripper_hidden = None
        args.gripper_hysteresis_active = False
        args.pi0_gripper_head_latch_active = False
        args.pi0_visual_contact_head_active_steps = 0
        last_debug = physical_debug(env, tracker, args)
        start = time.time()
        action_bounds = load_action_bounds(args)

        while action_steps < args.max_action_steps and env.env.is_viewer_alive():
            fixed_physics_steps = max(int(args.fixed_physics_steps_per_action), 0)
            tracker_already_updated = False
            if fixed_physics_steps > 0:
                for _ in range(fixed_physics_steps):
                    env.step_env()
                    sim_steps += 1
                    update_tracker(env, tracker, args)
                tracker_already_updated = True
            else:
                env.step_env()
                sim_steps += 1
                if not env.env.loop_every(HZ=args.hz):
                    continue

                pre_action_physics_steps = max(int(args.pre_action_physics_steps), 0)
                for _ in range(pre_action_physics_steps):
                    env.step_env()
                    sim_steps += 1
                    update_tracker(env, tracker, args)
                tracker_already_updated = pre_action_physics_steps > 0

            if not tracker_already_updated:
                update_tracker(env, tracker, args)
            last_debug = physical_debug(env, tracker, args)
            if last_debug["success"]:
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            if last_debug["physical_success"]:
                break

            args.current_action_step = action_steps
            state = observation_state_for_policy(
                env,
                policy,
                action_steps,
                args.hz,
                args,
                phase_tracker,
                scheduled_states=scheduled_states,
            )
            image, wrist_image = env.grab_image()
            if args.four_view_video_dir is not None:
                import cv2

                if video_writer is None:
                    args.four_view_video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = args.four_view_video_dir / f"seed{seed}_env{env_seed}_four_view.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(args.four_view_video_fps),
                        (640, 480),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Could not open four-view video writer: {video_path}")
                write_four_view_frame(env, image, wrist_image, video_writer)
            batch = {
                "observation.state": torch.tensor(np.asarray([state]), dtype=torch.float32, device=args.device),
                "observation.image": to_tensor_image(image).unsqueeze(0).to(args.device),
                "observation.wrist_image": to_tensor_image(wrist_image).unsqueeze(0).to(args.device),
                "task": [env.instruction],
            }
            if args.reset_policy_each_action:
                policy.reset()
            with torch.no_grad():
                if args.policy_type == "pi0":
                    action_tensor, sampling_info = select_pi0_action(policy, batch, args, instruction=env.instruction)
                else:
                    action_tensor = policy.select_action(batch)
                    sampling_info = {}
                action = action_tensor[0, :7].detach().cpu().numpy()
            if args.policy_type == "pi0":
                action, visual_contact_info = apply_pi0_visual_contact_head(policy, batch, action, args)
            else:
                visual_contact_info = {}
            args.current_observation_state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
            args.current_instruction = env.instruction
            action, action_info = postprocess_action(action, args, action_bounds)
            action_info.update(sampling_info)
            action_info.update(visual_contact_info)
            env_action, bridge_info = action_for_environment(action, env, args)
            action_info.update(bridge_info)
            args.pi0_prev_action = np.asarray(action, dtype=np.float32).reshape(7).copy()

            if args.log_steps_jsonl and (action_steps < 10 or action_steps % args.log_every == 0):
                row = {
                    "event": "step",
                    "seed": seed,
                    "action_step": action_steps,
                    "instruction": env.instruction,
                    "action": round_list(env_action, 5),
                    "action_info": action_info,
                    "debug": last_debug,
                }
                with args.log_steps_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            env.step(env_action)
            action_steps += 1
            if args.render:
                env.render()

            update_tracker(env, tracker, args)
            last_debug = physical_debug(env, tracker, args, count_stability=True)
            if last_debug["success"]:
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            if last_debug["physical_success"]:
                break

        last_debug = physical_debug(env, tracker, args)
        success = bool(success_ever or last_debug["success"])
        physical_success = bool(physical_success_ever or last_debug["physical_success"])
        last_debug.update(
            {
                "legacy_success_ever": success,
                "first_legacy_success_step": first_success_step,
                "physical_success_ever": physical_success,
                "first_physical_success_step": first_physical_success_step,
            }
        )
        return {
            "policy": args.policy_type,
            "seed": seed,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
            "success": success,
            "physical_success": physical_success,
            "action_steps": action_steps,
            "sim_steps": sim_steps,
            "elapsed_s": round(time.time() - start, 3),
            "pi0_visual_contact_head_active_steps": int(args.pi0_visual_contact_head_active_steps),
            "instruction": getattr(env, "instruction", None),
            "four_view_video": str(video_path) if video_path is not None else None,
            "debug": last_debug,
        }
    finally:
        if video_writer is not None:
            video_writer.release()
        if owns_env:
            close_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-type", choices=["smolvla", "pi0", "pi05"], default="smolvla")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--position-profile",
        choices=["legacy", "pnp_generalization_v1"],
        default="legacy",
        help="Object and plate placement distribution. Default preserves the original environment.",
    )
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--max-action-steps", type=int, default=300)
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    parser.add_argument("--physical-max-plate-z-gap", type=float, default=0.08)
    parser.add_argument("--physical-max-plate-xy-displacement", type=float, default=0.05)
    parser.add_argument("--physical-stable-place-steps", type=int, default=5)
    parser.add_argument("--reset-policy-each-action", action="store_true")
    parser.add_argument(
        "--hard-reset-sim-data",
        action="store_true",
        help="Call the inner MuJoCo reset before env.reset(seed=...) to clear residual physics state.",
    )
    parser.add_argument(
        "--fresh-env-per-episode",
        action="store_true",
        help="Construct a new SimpleEnv2 for every seed. Slower, but useful for clean eval audits.",
    )
    parser.add_argument(
        "--policy-seed-offset",
        type=int,
        default=0,
        help="Add this offset to the environment seed before stochastic Pi0 action sampling.",
    )
    parser.add_argument(
        "--fixed-env-seed",
        type=int,
        default=None,
        help="Use one explicit environment seed while --seeds continues to control Pi0 sampling seeds.",
    )
    parser.add_argument(
        "--pi0-action-samples",
        type=int,
        default=1,
        help=(
            "For Pi0/Pi0.5: sample this many full action chunks from the same observation when the "
            "policy action queue is empty, then reduce them before execution. This is a fair "
            "inference-time denoising diagnostic because it uses no target/plate/phase state."
        ),
    )
    parser.add_argument(
        "--pi0-action-sample-reducer",
        choices=[
            "mean",
            "median",
            "medoid_mean",
            "medoid_median",
            "instruction_color",
            "sample0",
            "sample1",
            "sample2",
            "sample3",
            "sample4",
            "sample5",
            "sample6",
            "sample7",
            "instruction_color_blue_sample0",
            "instruction_color_blue_sample1",
            "instruction_color_blue_sample2",
            "instruction_color_blue_sample3",
            "instruction_color_blue_sample1_gmean_gate",
        ],
        default="mean",
        help=(
            "Reducer used with --pi0-action-samples. instruction_color is a strict-input "
            "diagnostic: it reads the language instruction only, using mean for red mug "
            "and median for blue mug. instruction_color_blue_sampleN keeps red mug on "
            "mean and uses a fixed sampled chunk index for blue mug. "
            "instruction_color_blue_sample1_gmean_gate is a diagnostic selector that keeps "
            "red mug on mean and, for blue mug, chooses sample1 or median from the first "
            "fresh action chunk's sample1 gripper mean."
        ),
    )
    parser.add_argument(
        "--pi0-sample1-gmean-threshold",
        type=float,
        default=0.01696,
        help=(
            "Diagnostic threshold used only by instruction_color_blue_sample1_gmean_gate. "
            "It compares the first fresh chunk's sample1 gripper mean over the executed "
            "window and does not read target/plate/phase state."
        ),
    )
    parser.add_argument(
        "--pi0-exec-chunk-steps",
        type=int,
        default=0,
        help=(
            "For Pi0/Pi0.5: after sampling a full action chunk, execute at most this many "
            "actions before forcing a new visual observation/action chunk. 0 keeps the "
            "policy checkpoint's native n_action_steps."
        ),
    )
    parser.add_argument(
        "--pi0-action-chunk-offset",
        type=int,
        default=0,
        help=(
            "For Pi0/Pi0.5: skip this many leading actions from each freshly sampled chunk before "
            "execution. This is a strict-input waypoint-horizon diagnostic; it uses only the "
            "policy's own action chunk. 0 preserves the historical behavior."
        ),
    )
    parser.add_argument(
        "--pre-action-physics-steps",
        type=int,
        default=0,
        help=(
            "Run this many raw MuJoCo physics steps immediately before each audited "
            "observation/action. 0 keeps the original 20Hz audit protocol; 25 matches "
            "the collector's settled-control prefix loop."
        ),
    )
    parser.add_argument(
        "--fixed-physics-steps-per-action",
        type=int,
        default=0,
        help=(
            "If greater than 0, bypass env.loop_every(...) and run exactly this many "
            "raw MuJoCo physics steps before each observation/action. This reproduces "
            "the collector prefix loop more closely than stacking extra steps on top "
            "of the original 20Hz audit loop."
        ),
    )
    parser.add_argument(
        "--fair-vla",
        action="store_true",
        help=(
            "Reject Pi0 eval settings that use privileged target/plate/phase/schedule "
            "signals or scripted action postprocessing."
        ),
    )
    parser.add_argument(
        "--pi0-action-mode",
        choices=["absolute", "joint_delta", "eef_delta", "eef_abs"],
        default="absolute",
        help=(
            "For Pi0/Pi0.5: interpret policy action as absolute joint target, "
            "joint delta plus absolute gripper, EEF/TCP delta plus absolute gripper, "
            "or next TCP xyz target plus absolute gripper."
        ),
    )
    parser.add_argument(
        "--eef-abs-max-step",
        type=float,
        default=0.004,
        help="Maximum per-axis TCP delta used when --pi0-action-mode=eef_abs.",
    )
    parser.add_argument(
        "--eef-delta-max-step",
        type=float,
        default=None,
        help=(
            "Optional fixed per-axis controller limit for eef_delta actions. "
            "This does not read target/plate state or dataset statistics."
        ),
    )
    parser.add_argument(
        "--pi0-phase-state",
        choices=["auto", "dynamic_oracle", "zeros", "dataset_schedule"],
        default="auto",
        help=(
            "For Pi0 phase-conditioned checkpoints: build phase state from a scripted oracle tracker, "
            "zeros, or the recorded dataset episode schedule while keeping live joint state."
        ),
    )
    parser.add_argument(
        "--pi0-state-extra",
        choices=["auto", "gripper", "gripper_time", "prev_action", "gripper_prev_action"],
        default="auto",
        help=(
            "For robot-proprio checkpoints, append current gripper state, gripper plus "
            "control timestamp, previous 7-D policy action, or current gripper plus "
            "previous action to the 6-D joint state. "
            "This is separate from the strict 6-D fair-vla protocol."
        ),
    )
    parser.add_argument(
        "--pi0-phase-schedule-episodes",
        default=None,
        help=(
            "Comma-separated dataset episode ids aligned with --seeds when "
            "--pi0-phase-state=dataset_schedule. Defaults to 0,1,2,..."
        ),
    )
    parser.add_argument("--clamp-action-to-dataset", action="store_true")
    parser.add_argument(
        "--clip-gripper",
        action="store_true",
        help="Fair actuator postprocess: clip the gripper command to the executable [0, 1] range.",
    )
    parser.add_argument("--dataset-repo-id", default="datawhale_eai_pnp_language")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data_language"))
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gripper-open-until-step",
        type=int,
        default=None,
        help="Diagnostic only: force gripper action to 0 before this action step.",
    )
    parser.add_argument(
        "--gripper-hysteresis",
        action="store_true",
        help=(
            "Diagnostic only: once the gripper command crosses "
            "--gripper-hysteresis-close-threshold, hold it closed until "
            "--gripper-hysteresis-release-step if provided."
        ),
    )
    parser.add_argument("--gripper-hysteresis-close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-hysteresis-closed-value", type=float, default=1.0)
    parser.add_argument("--gripper-hysteresis-release-step", type=int, default=None)
    parser.add_argument(
        "--phase-scripted-gripper",
        action="store_true",
        help="Diagnostic only: use the current phase to choose gripper open/close while keeping Pi0 arm/EEF output.",
    )
    parser.add_argument(
        "--pi0-gripper-head-path",
        type=Path,
        default=None,
        help=(
            "Pi0 auxiliary gripper timing head. This uses robot state, Pi0's own action, "
            "progress, and instruction color; it does not use target/plate oracle state."
        ),
    )
    parser.add_argument("--pi0-gripper-head-threshold", type=float, default=0.5)
    parser.add_argument("--pi0-gripper-head-blue-threshold", type=float, default=None)
    parser.add_argument("--pi0-gripper-head-red-threshold", type=float, default=None)
    parser.add_argument("--pi0-gripper-head-progress-denom", type=float, default=240.0)
    parser.add_argument("--pi0-gripper-head-continuous", action="store_true")
    parser.add_argument(
        "--pi0-gripper-head-latch",
        action="store_true",
        help=(
            "Diagnostic only: after the learned Pi0 gripper head closes, hold the "
            "gripper closed until --pi0-gripper-head-latch-release-step. This is "
            "applied after the learned head, so it can test early-release failures."
        ),
    )
    parser.add_argument("--pi0-gripper-head-latch-close-threshold", type=float, default=0.5)
    parser.add_argument("--pi0-gripper-head-latch-closed-value", type=float, default=1.0)
    parser.add_argument("--pi0-gripper-head-latch-release-step", type=int, default=None)
    parser.add_argument(
        "--pi0-gripper-head-latch-release-prob-below",
        type=float,
        default=None,
        help="Release the post-head latch once the learned gripper-head close probability falls below this value.",
    )
    parser.add_argument(
        "--pi0-gripper-head-latch-release-min-step",
        type=int,
        default=0,
        help="Minimum action step before --pi0-gripper-head-latch-release-prob-below may release the latch.",
    )
    parser.add_argument(
        "--pi0-gripper-head-latch-release-head-path",
        type=Path,
        default=None,
        help=(
            "Optional separate learned gripper head used only to release the latch. "
            "It does not control early close; it only decides when a held-close "
            "latch may open after --pi0-gripper-head-latch-release-min-step."
        ),
    )
    parser.add_argument("--pi0-gripper-head-latch-release-head-progress-denom", type=float, default=240.0)
    parser.add_argument(
        "--pi0-gripper-head-latch-release-head-prob-below",
        type=float,
        default=0.5,
        help="Release the latch when the separate release head close-probability is below this threshold.",
    )
    parser.add_argument(
        "--pi0-visual-contact-head-path",
        type=Path,
        default=None,
        help=(
            "Learned Pi0 contact adapter using only both camera images, language, and 6-D robot "
            "proprioception. It predicts a direct EEF target and gripper value and is gated by "
            "distance to held-out-calibrated training features. No object/plate/phase oracle is read."
        ),
    )
    parser.add_argument(
        "--pi0-visual-contact-head-monitor-only",
        action="store_true",
        help="Compute and log visual-head OOD decisions without changing the Pi0 action.",
    )
    parser.add_argument("--pi0-visual-contact-head-distance-scale", type=float, default=1.0)
    parser.add_argument("--pi0-visual-contact-head-blend", type=float, default=1.0)
    parser.add_argument("--pi0-visual-contact-head-binarize-gripper", action="store_true")
    parser.add_argument("--pi0-visual-contact-head-gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--pi0-visual-gripper-sequence-threshold",
        type=float,
        default=None,
        help="Optional deployment threshold override for an embedded learned visual gripper sequence head.",
    )
    parser.add_argument(
        "--pi0-eef-residual-head-path",
        type=Path,
        default=None,
        help=(
            "Pi0 auxiliary TCP target residual head. It uses robot state, Pi0's own "
            "action, progress, and instruction color; it does not read target/plate "
            "oracle state. Diagnostic/adapter protocol, not raw fair-vla."
        ),
    )
    parser.add_argument("--pi0-eef-residual-head-scale", type=float, default=1.0)
    parser.add_argument("--pi0-eef-residual-head-max-abs", type=float, default=0.04)
    parser.add_argument("--pi0-eef-residual-head-progress-denom", type=float, default=240.0)
    parser.add_argument(
        "--pi0-eef-contact-residual-head-path",
        type=Path,
        default=None,
        help=(
            "Optional second-stage EEF residual head for early contact correction. "
            "It is applied after the primary residual and learned gripper head, "
            "then gated by rollout progress and optionally by open gripper state."
        ),
    )
    parser.add_argument("--pi0-eef-contact-residual-head-scale", type=float, default=1.0)
    parser.add_argument("--pi0-eef-contact-residual-head-max-abs", type=float, default=0.04)
    parser.add_argument("--pi0-eef-contact-residual-head-progress-denom", type=float, default=240.0)
    parser.add_argument("--pi0-eef-contact-residual-head-start-progress", type=float, default=0.0)
    parser.add_argument("--pi0-eef-contact-residual-head-end-progress", type=float, default=None)
    parser.add_argument("--pi0-eef-contact-residual-head-require-open", action="store_true")
    parser.add_argument("--pi0-eef-contact-residual-head-open-threshold", type=float, default=0.5)
    parser.add_argument(
        "--pi0-eef-tail-residual-head-path",
        type=Path,
        default=None,
        help=(
            "Optional second Pi0 EEF residual head for the carry/release tail. "
            "It is gated by progress and optionally gripper closure, so it can be "
            "kept out of the pregrasp/contact phase."
        ),
    )
    parser.add_argument("--pi0-eef-tail-residual-head-scale", type=float, default=1.0)
    parser.add_argument("--pi0-eef-tail-residual-head-max-abs", type=float, default=0.04)
    parser.add_argument("--pi0-eef-tail-residual-head-progress-denom", type=float, default=240.0)
    parser.add_argument("--pi0-eef-tail-residual-head-start-progress", type=float, default=0.0)
    parser.add_argument("--pi0-eef-tail-residual-head-require-closed", action="store_true")
    parser.add_argument("--pi0-eef-tail-residual-head-closed-threshold", type=float, default=0.5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--four-view-video-dir", type=Path, default=None)
    parser.add_argument("--four-view-video-fps", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-steps-jsonl", type=Path, default=None)
    parser.add_argument(
        "--pi0-sample-log-jsonl",
        type=Path,
        default=None,
        help=(
            "For Pi0 diagnostics: write one JSONL row whenever a fresh sampled action chunk "
            "is produced. Logs only policy candidates, language, and robot/state timing; it "
            "does not change the executed action."
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--place-step-scale", type=float, default=0.65)
    parser.add_argument("--pos-tol", type=float, default=0.008)
    parser.add_argument("--initial-hold", type=int, default=0)
    parser.add_argument("--close-hold", type=int, default=64)
    parser.add_argument("--pre-release-hold", type=int, default=12)
    parser.add_argument("--open-hold", type=int, default=70)
    parser.add_argument("--final-hold", type=int, default=35)
    parser.add_argument("--grasp-offset-x", type=float, default=0.006)
    parser.add_argument("--grasp-offset-y", type=float, default=0.060)
    parser.add_argument("--grasp-offset-z", type=float, default=0.000)
    parser.add_argument("--pregrasp-z", type=float, default=0.135)
    parser.add_argument("--lift-z", type=float, default=0.145)
    parser.add_argument("--release-offset-x", type=float, default=0.008)
    parser.add_argument("--release-offset-y", type=float, default=0.026)
    parser.add_argument("--release-offset-z", type=float, default=0.095)
    parser.add_argument("--retreat-z", type=float, default=0.145)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()
    if args.log_steps_jsonl:
        args.log_steps_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.log_steps_jsonl.exists():
            args.log_steps_jsonl.unlink()
    if args.pi0_sample_log_jsonl:
        args.pi0_sample_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.pi0_sample_log_jsonl.exists():
            args.pi0_sample_log_jsonl.unlink()
    args.pi0_gripper_head = load_pi0_gripper_head(args.pi0_gripper_head_path)
    args.pi0_gripper_head_latch_release_head = load_pi0_gripper_head(args.pi0_gripper_head_latch_release_head_path)
    args.pi0_eef_residual_head = load_pi0_eef_residual_head(args.pi0_eef_residual_head_path)
    args.pi0_eef_contact_residual_head = load_pi0_eef_residual_head(args.pi0_eef_contact_residual_head_path)
    args.pi0_eef_tail_residual_head = load_pi0_eef_residual_head(args.pi0_eef_tail_residual_head_path)
    args.pi0_visual_contact_head = load_pi0_visual_contact_head(args.pi0_visual_contact_head_path, args.device)
    if args.pi0_visual_contact_head is not None:
        if args.policy_type != "pi0" or args.pi0_action_mode != "eef_abs":
            raise ValueError("--pi0-visual-contact-head-path requires Pi0 with --pi0-action-mode=eef_abs")
        incompatible = [
            name
            for name, value in (
                ("--pi0-gripper-head-path", args.pi0_gripper_head_path),
                ("--pi0-eef-residual-head-path", args.pi0_eef_residual_head_path),
                ("--pi0-eef-contact-residual-head-path", args.pi0_eef_contact_residual_head_path),
                ("--pi0-eef-tail-residual-head-path", args.pi0_eef_tail_residual_head_path),
            )
            if value is not None
        ]
        if args.phase_scripted_gripper:
            incompatible.append("--phase-scripted-gripper")
        if incompatible:
            raise ValueError("Pi0 visual contact head cannot be mixed with: " + ", ".join(incompatible))
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.episodes))

    from mujoco_env.y_env2 import SimpleEnv2

    if args.policy_type == "pi0":
        policy = make_pi0_policy_for_dataset(
            args.device,
            args.policy_path,
            args.dataset_repo_id,
            args.dataset_root,
        )
    elif args.policy_type == "pi05":
        policy = make_pi05_policy_for_dataset(
            args.device,
            args.policy_path,
            args.dataset_repo_id,
            args.dataset_root,
            n_action_steps=args.pi0_exec_chunk_steps if args.pi0_exec_chunk_steps > 0 else None,
        )
    else:
        policy = make_smolvla_policy(args.device, args.policy_path)
    assert_fair_vla_policy(args, policy)
    phase_schedule_states = load_pi0_phase_schedule_states(args, seeds, policy)
    results = []
    env = None
    try:
        for seed in seeds:
            if env is None or args.fresh_env_per_episode:
                if env is not None:
                    close_env(env)
                env = SimpleEnv2(
                    "./asset/example_scene_y2.xml",
                    action_type="joint_angle",
                    position_profile=args.position_profile,
                )
            row = rollout(args, policy, seed, env=env, scheduled_states=phase_schedule_states.get(int(seed)))
            results.append(row)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            close_env(env)

    summary = {
        "policy": args.policy_type,
        "episodes": len(results),
        "success_count": sum(1 for row in results if row["success"]),
        "physical_success_count": sum(1 for row in results if row["physical_success"]),
        "success_rate": sum(1 for row in results if row["success"]) / max(len(results), 1),
        "physical_success_rate": sum(1 for row in results if row["physical_success"]) / max(len(results), 1),
        "seeds": seeds,
        "max_action_steps": args.max_action_steps,
        "physical_min_lift": args.physical_min_lift,
        "physical_min_lift_steps": args.physical_min_lift_steps,
        "physical_final_upright_cos": args.physical_final_upright_cos,
        "physical_max_plate_z_gap": args.physical_max_plate_z_gap,
        "physical_max_plate_xy_displacement": args.physical_max_plate_xy_displacement,
        "physical_stable_place_steps": args.physical_stable_place_steps,
        "fixed_env_seed": args.fixed_env_seed,
        "position_profile": args.position_profile,
        "fair_vla": bool(args.fair_vla),
        "hard_reset_sim_data": bool(args.hard_reset_sim_data),
        "fresh_env_per_episode": bool(args.fresh_env_per_episode),
        "four_view_video_dir": str(args.four_view_video_dir) if args.four_view_video_dir else None,
        "pi0_action_mode": args.pi0_action_mode,
        "pi0_action_samples": int(args.pi0_action_samples),
        "pi0_action_sample_reducer": args.pi0_action_sample_reducer,
        "pi0_exec_chunk_steps": int(args.pi0_exec_chunk_steps),
        "eef_delta_max_step": args.eef_delta_max_step,
        "pi0_action_chunk_offset": int(args.pi0_action_chunk_offset),
        "pi0_sample_log_jsonl": str(args.pi0_sample_log_jsonl) if args.pi0_sample_log_jsonl else None,
        "pre_action_physics_steps": int(args.pre_action_physics_steps),
        "fixed_physics_steps_per_action": int(args.fixed_physics_steps_per_action),
        "clip_gripper": bool(args.clip_gripper),
        "pi0_gripper_head_path": str(args.pi0_gripper_head_path) if args.pi0_gripper_head_path else None,
        "pi0_gripper_head_threshold": float(args.pi0_gripper_head_threshold),
        "pi0_gripper_head_blue_threshold": args.pi0_gripper_head_blue_threshold,
        "pi0_gripper_head_red_threshold": args.pi0_gripper_head_red_threshold,
        "pi0_gripper_head_progress_denom": float(args.pi0_gripper_head_progress_denom),
        "pi0_gripper_head_continuous": bool(args.pi0_gripper_head_continuous),
        "pi0_gripper_head_latch_release_head_path": str(args.pi0_gripper_head_latch_release_head_path)
        if args.pi0_gripper_head_latch_release_head_path
        else None,
        "pi0_gripper_head_latch_release_head_progress_denom": float(
            args.pi0_gripper_head_latch_release_head_progress_denom
        ),
        "pi0_gripper_head_latch_release_head_prob_below": float(
            args.pi0_gripper_head_latch_release_head_prob_below
        ),
        "pi0_eef_residual_head_path": str(args.pi0_eef_residual_head_path) if args.pi0_eef_residual_head_path else None,
        "pi0_eef_residual_head_scale": float(args.pi0_eef_residual_head_scale),
        "pi0_eef_residual_head_max_abs": float(args.pi0_eef_residual_head_max_abs),
        "pi0_eef_residual_head_progress_denom": float(args.pi0_eef_residual_head_progress_denom),
        "pi0_visual_contact_head_path": str(args.pi0_visual_contact_head_path)
        if args.pi0_visual_contact_head_path
        else None,
        "pi0_visual_contact_head_monitor_only": bool(args.pi0_visual_contact_head_monitor_only),
        "pi0_visual_contact_head_distance_scale": float(args.pi0_visual_contact_head_distance_scale),
        "pi0_visual_contact_head_blend": float(args.pi0_visual_contact_head_blend),
        "pi0_visual_contact_head_binarize_gripper": bool(args.pi0_visual_contact_head_binarize_gripper),
        "pi0_visual_gripper_sequence_threshold": args.pi0_visual_gripper_sequence_threshold,
        "pi0_visual_contact_head_active_steps": int(
            sum(row.get("pi0_visual_contact_head_active_steps", 0) for row in results)
        ),
        "pi0_eef_contact_residual_head_path": str(args.pi0_eef_contact_residual_head_path)
        if args.pi0_eef_contact_residual_head_path
        else None,
        "pi0_eef_contact_residual_head_scale": float(args.pi0_eef_contact_residual_head_scale),
        "pi0_eef_contact_residual_head_max_abs": float(args.pi0_eef_contact_residual_head_max_abs),
        "pi0_eef_contact_residual_head_progress_denom": float(args.pi0_eef_contact_residual_head_progress_denom),
        "pi0_eef_contact_residual_head_start_progress": float(args.pi0_eef_contact_residual_head_start_progress),
        "pi0_eef_contact_residual_head_end_progress": args.pi0_eef_contact_residual_head_end_progress,
        "pi0_eef_contact_residual_head_require_open": bool(args.pi0_eef_contact_residual_head_require_open),
        "pi0_eef_contact_residual_head_open_threshold": float(args.pi0_eef_contact_residual_head_open_threshold),
        "pi0_eef_tail_residual_head_path": str(args.pi0_eef_tail_residual_head_path)
        if args.pi0_eef_tail_residual_head_path
        else None,
        "pi0_eef_tail_residual_head_scale": float(args.pi0_eef_tail_residual_head_scale),
        "pi0_eef_tail_residual_head_max_abs": float(args.pi0_eef_tail_residual_head_max_abs),
        "pi0_eef_tail_residual_head_progress_denom": float(args.pi0_eef_tail_residual_head_progress_denom),
        "pi0_eef_tail_residual_head_start_progress": float(args.pi0_eef_tail_residual_head_start_progress),
        "pi0_eef_tail_residual_head_require_closed": bool(args.pi0_eef_tail_residual_head_require_closed),
        "pi0_eef_tail_residual_head_closed_threshold": float(args.pi0_eef_tail_residual_head_closed_threshold),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
