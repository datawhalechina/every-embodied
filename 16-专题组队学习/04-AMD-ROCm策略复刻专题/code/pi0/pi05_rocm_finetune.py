#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the PI0.5 action expert on ROCm.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-start-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Trade additional unified memory for faster backward passes.",
    )
    parser.add_argument(
        "--train-mode",
        choices=("expert_only", "expert_vision", "full"),
        default="expert_only",
        help=(
            "Parameters to update: action expert only; action expert plus the vision tower and "
            "multimodal projector; or all parameters. The unused action-expert LM head stays frozen."
        ),
    )
    parser.add_argument(
        "--action-loss-weights",
        default=None,
        help="Optional comma-separated weights for the real action dimensions, for example 1,1,1,0,0,0,2.",
    )
    parser.add_argument("--open-frame-sample-weight", type=float, default=1.0)
    parser.add_argument("--episode-start-frame-sample-weight", type=float, default=1.0)
    parser.add_argument("--episode-start-window", type=int, default=0)
    parser.add_argument(
        "--episode-start-episode-count",
        type=int,
        default=0,
        help=(
            "Apply episode-start weighting only to the first N episodes. Zero means all episodes. "
            "This avoids treating recovery suffixes as genuine task starts."
        ),
    )
    parser.add_argument("--tail-episode-frame-sample-weight", type=float, default=1.0)
    parser.add_argument("--tail-episode-count", type=int, default=0)
    parser.add_argument("--transition-frame-sample-weight", type=float, default=1.0)
    parser.add_argument("--transition-radius", type=int, default=0)
    parser.add_argument("--skip-save", action="store_true")
    return parser.parse_args()


def make_eval_indices(dataset: LeRobotDataset, count: int, start_count: int) -> list[int]:
    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise ValueError("The dataset is empty.")
    count = max(1, min(int(count), dataset_size))
    episode_ids = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    starts = [int(np.flatnonzero(episode_ids == episode)[0]) for episode in range(dataset.num_episodes)]
    start_count = max(0, min(int(start_count), len(starts), count))
    selected: list[int] = []
    if start_count > 0:
        for episode in np.linspace(0, len(starts) - 1, start_count):
            candidate = starts[int(round(float(episode)))]
            if candidate not in selected:
                selected.append(candidate)
    for index in np.linspace(0, dataset_size - 1, max(count * 4, 1)):
        candidate = int(index)
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= count:
            break
    return sorted(selected)


def validate_dataset_alignment(dataset: LeRobotDataset) -> dict:
    """Reject datasets whose physical row order disagrees with LeRobot global indices."""
    indices = np.asarray(dataset.hf_dataset["index"], dtype=np.int64)
    episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    expected_indices = np.arange(len(dataset), dtype=np.int64)
    mismatch = np.flatnonzero(indices != expected_indices)
    if mismatch.size:
        first = int(mismatch[0])
        raise ValueError(
            "Dataset physical row order is incompatible with action chunks: "
            f"{mismatch.size} rows have index != physical position; "
            f"first position={first}, stored index={int(indices[first])}. "
            "Canonicalize the LeRobot dataset before training."
        )

    probe_positions: set[int] = set()
    for episode in range(dataset.num_episodes):
        positions = np.flatnonzero(episodes == episode)
        if positions.size == 0:
            raise ValueError(f"Episode {episode} is missing from the physical dataset rows.")
        start = int(positions[0])
        end = int(positions[-1] + 1)
        if not np.array_equal(positions, np.arange(start, end, dtype=np.int64)):
            raise ValueError(f"Episode {episode} rows are not physically contiguous.")
        if not np.array_equal(frame_indices[positions], np.arange(len(positions), dtype=np.int64)):
            raise ValueError(f"Episode {episode} frame_index is not contiguous from zero.")
        metadata = dataset.meta.episodes[episode]
        if int(metadata["dataset_from_index"]) != start or int(metadata["dataset_to_index"]) != end:
            raise ValueError(
                f"Episode {episode} metadata boundary [{metadata['dataset_from_index']}, "
                f"{metadata['dataset_to_index']}) != physical boundary [{start}, {end})."
            )
        probe_positions.update({start, (start + end - 1) // 2, end - 1})

    if dataset.delta_indices is None or dataset.delta_indices.get(ACTION, [None])[0] != 0:
        raise ValueError("The action chunk must begin at delta index zero.")
    ordered_probes = sorted(probe_positions)
    current = torch.stack(
        [torch.as_tensor(value) for value in dataset.hf_dataset[ordered_probes][ACTION]]
    ).float()
    queried = dataset._query_hf_dataset({ACTION: ordered_probes})[ACTION].float()
    chunk0_max_abs_error = float((current - queried).abs().max().item())
    if chunk0_max_abs_error != 0.0:
        raise ValueError(
            "Action chunk query mismatch across episode boundary probes: "
            f"max_abs={chunk0_max_abs_error}."
        )

    return {
        "status": "passed",
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "physical_index_mismatch_rows": 0,
        "chunk0_probes": len(ordered_probes),
        "chunk0_max_abs_error": chunk0_max_abs_error,
    }


def configure_policy(args: argparse.Namespace, metadata: LeRobotDatasetMetadata):
    features = dataset_to_policy_features(metadata.features)
    config = PreTrainedConfig.from_pretrained(args.model_dir)
    if config.type != "pi05":
        raise RuntimeError(f"Expected a PI0.5 config, got {config.type!r}.")
    config.device = "cuda"
    config.dtype = args.dtype
    config.chunk_size = args.chunk_size
    config.n_action_steps = args.chunk_size
    config.gradient_checkpointing = not args.disable_gradient_checkpointing
    config.compile_model = False
    config.input_features = {
        key: feature
        for key, feature in features.items()
        if feature.type in {FeatureType.VISUAL, FeatureType.STATE}
    }
    config.output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    return config


def configure_trainable_parameters(policy: PI05Policy, train_mode: str) -> list[torch.nn.Parameter]:
    model = policy.model.paligemma_with_expert
    policy.requires_grad_(True)

    if train_mode == "expert_only":
        model.paligemma.requires_grad_(False)
    elif train_mode == "expert_vision":
        model.paligemma.requires_grad_(False)
        model.paligemma.model.vision_tower.requires_grad_(True)
        model.paligemma.model.multi_modal_projector.requires_grad_(True)
    elif train_mode != "full":
        raise ValueError(f"Unsupported train mode: {train_mode}")

    # The action expert's vocabulary head is not used by flow matching.
    model.gemma_expert.lm_head.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"Train mode {train_mode!r} selected no parameters.")
    return trainable


def parse_action_loss_weights(spec: str | None, action_dim: int) -> torch.Tensor | None:
    if spec is None:
        return None
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(values) != action_dim:
        raise ValueError(f"Expected {action_dim} action loss weights, got {len(values)}: {values}")
    weights = torch.tensor(values, dtype=torch.float32)
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("Action loss weights must be non-negative and have a positive sum.")
    return weights


def compute_training_loss(
    policy: PI05Policy,
    batch: dict,
    action_loss_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    if action_loss_weights is None:
        return policy(batch)

    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    losses = policy.model.forward(images, image_masks, tokens, token_masks, actions)
    action_dim = policy.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :action_dim]
    weights = action_loss_weights.to(device=losses.device, dtype=losses.dtype)
    loss = (losses * weights.view(1, 1, -1)).sum(dim=-1).mean() / weights.sum()
    return loss, {"loss_per_dim": losses.mean(dim=(0, 1)).detach().cpu().tolist()}


def make_frame_sampler(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler | None, dict]:
    if (
        args.open_frame_sample_weight <= 0
        or args.episode_start_frame_sample_weight <= 0
        or args.tail_episode_frame_sample_weight <= 0
        or args.transition_frame_sample_weight <= 0
    ):
        raise ValueError("Frame sample weights must be positive.")
    if (
        args.transition_radius < 0
        or args.episode_start_window < 0
        or args.episode_start_episode_count < 0
        or args.tail_episode_count < 0
    ):
        raise ValueError(
            "Transition radius, episode-start window/count, and tail-episode count must be non-negative."
        )

    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    gripper = actions[:, 6]
    weights = np.ones(len(dataset), dtype=np.float64)
    open_mask = gripper < 0.5
    weights[open_mask] *= args.open_frame_sample_weight

    episode_start_mask = np.zeros(len(dataset), dtype=bool)
    episode_start_episode_count = (
        dataset.num_episodes
        if args.episode_start_episode_count == 0
        else min(int(args.episode_start_episode_count), dataset.num_episodes)
    )
    for episode in range(episode_start_episode_count):
        indices = np.flatnonzero(episodes == episode)
        if args.episode_start_window > 0:
            episode_start_mask[indices[: args.episode_start_window]] = True
    weights[episode_start_mask] *= args.episode_start_frame_sample_weight

    tail_episode_mask = np.zeros(len(dataset), dtype=bool)
    tail_episode_count = min(int(args.tail_episode_count), dataset.num_episodes)
    if tail_episode_count > 0:
        tail_start = dataset.num_episodes - tail_episode_count
        tail_episode_mask = episodes >= tail_start
    weights[tail_episode_mask] *= args.tail_episode_frame_sample_weight

    transition_mask = np.zeros(len(dataset), dtype=bool)
    transition_centers: list[int] = []
    for episode in range(dataset.num_episodes):
        indices = np.flatnonzero(episodes == episode)
        binary = gripper[indices] >= 0.5
        rising = np.flatnonzero((~binary[:-1]) & binary[1:]) + 1
        falling = np.flatnonzero(binary[:-1] & (~binary[1:])) + 1
        centers: list[int] = []
        if rising.size:
            centers.append(int(indices[int(rising[0])]))
        if falling.size:
            centers.append(int(indices[int(falling[-1])]))
        for center in centers:
            transition_centers.append(center)
            episode_start = int(indices[0])
            episode_end = int(indices[-1])
            low = max(center - args.transition_radius, episode_start)
            high = min(center + args.transition_radius, episode_end)
            transition_mask[low : high + 1] = True

    weights[transition_mask] *= args.transition_frame_sample_weight
    info = {
        "open_frame_sample_weight": args.open_frame_sample_weight,
        "episode_start_frame_sample_weight": args.episode_start_frame_sample_weight,
        "episode_start_window": args.episode_start_window,
        "episode_start_episode_count": episode_start_episode_count,
        "episode_start_window_frames": int(episode_start_mask.sum()),
        "tail_episode_frame_sample_weight": args.tail_episode_frame_sample_weight,
        "tail_episode_count": tail_episode_count,
        "tail_episode_frames": int(tail_episode_mask.sum()),
        "transition_frame_sample_weight": args.transition_frame_sample_weight,
        "transition_radius": args.transition_radius,
        "open_frames": int(open_mask.sum()),
        "transition_window_frames": int(transition_mask.sum()),
        "transition_centers": transition_centers,
        "effective_open_probability": float(weights[open_mask].sum() / weights.sum()),
        "effective_episode_start_window_probability": float(
            weights[episode_start_mask].sum() / weights.sum()
        ),
        "effective_tail_episode_probability": float(weights[tail_episode_mask].sum() / weights.sum()),
    }
    if np.allclose(weights, 1.0):
        return None, info
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, info


@torch.no_grad()
def evaluate_policy(
    policy: PI05Policy,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    indices: list[int],
    seed: int,
) -> dict:
    policy.eval()
    losses: list[float] = []
    action_errors: list[torch.Tensor] = []
    first_action_probes: list[dict] = []

    for ordinal, index in enumerate(indices):
        raw = dataset[index]
        batch = preprocessor(default_collate([raw]))

        torch.manual_seed(seed + ordinal)
        torch.cuda.manual_seed_all(seed + ordinal)
        loss, _ = policy(batch)
        losses.append(float(loss.detach().cpu().item()))

        torch.manual_seed(seed + 10_000 + ordinal)
        torch.cuda.manual_seed_all(seed + 10_000 + ordinal)
        predicted = policy.predict_action_chunk(batch)
        predicted = postprocessor(predicted).detach().float().cpu()[0]
        target = raw["action"].detach().float().cpu()
        valid = ~raw["action_is_pad"].detach().bool().cpu()
        if valid.any():
            action_errors.append((predicted[valid] - target[valid]).abs())
            first_action_probes.append(
                {
                    "index": int(index),
                    "episode": int(raw["episode_index"].item()),
                    "frame": int(raw["frame_index"].item()),
                    "target": target[0].tolist(),
                    "predicted": predicted[0].tolist(),
                }
            )

    if not action_errors:
        raise RuntimeError("No non-padded validation actions were found.")
    errors = torch.cat(action_errors, dim=0)
    return {
        "indices": indices,
        "flow_loss_mean": float(np.mean(losses)),
        "flow_loss_std": float(np.std(losses)),
        "action_mae": float(errors.mean().item()),
        "xyz_mae": float(errors[:, :3].mean().item()),
        "gripper_mae": float(errors[:, 6].mean().item()),
        "valid_action_rows": int(errors.shape[0]),
        "first_action_probes": first_action_probes,
    }


def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    delta_timestamps = {"action": [index / metadata.fps for index in range(args.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    dataset_alignment = validate_dataset_alignment(dataset)
    print(json.dumps({"event": "dataset_alignment", **dataset_alignment}), flush=True)
    config = configure_policy(args, metadata)
    action_dim = config.output_features[ACTION].shape[0]
    action_loss_weights = parse_action_loss_weights(args.action_loss_weights, action_dim)
    policy = PI05Policy.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
        strict=True,
    )
    trainable = configure_trainable_parameters(policy, args.train_mode)
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
    )

    eval_indices = make_eval_indices(dataset, args.eval_samples, args.eval_start_samples)
    baseline_metrics = evaluate_policy(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        eval_indices,
        seed=args.seed + 20_000,
    )
    print(json.dumps({"event": "baseline_eval", **baseline_metrics}), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    sampler, sampler_info = make_frame_sampler(dataset, args, generator)
    print(json.dumps({"event": "sampler", **sampler_info}), flush=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=0,
        drop_last=False,
    )
    iterator = iter(dataloader)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    trainable_probe = policy.model.action_out_proj.bias.detach().clone()
    frozen_probe = (
        policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
        .patch_embedding.weight
        .detach()
        .clone()
    )
    language_probe = (
        policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0]
        .input_layernorm.weight.detach()
        .clone()
    )
    recent_losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    policy.train()

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = preprocessor(batch)
        lr = learning_rate_at_step(step, args.steps, args.warmup_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss, output = compute_training_loss(policy, batch, action_loss_weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        recent_losses.append(float(loss.detach().cpu().item()))

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            window = recent_losses[-min(len(recent_losses), args.log_every) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": recent_losses[-1],
                        "loss_window_mean": float(np.mean(window)),
                        "grad_norm": float(grad_norm.detach().cpu().item()),
                        "learning_rate": lr,
                        "loss_per_dim": output["loss_per_dim"],
                        "memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "memory_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                ),
                flush=True,
            )

    final_metrics = evaluate_policy(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        eval_indices,
        seed=args.seed + 20_000,
    )
    print(json.dumps({"event": "final_eval", **final_metrics}), flush=True)

    trainable_delta = (
        (policy.model.action_out_proj.bias - trainable_probe).abs().max().detach().cpu().item()
    )
    frozen_delta = (
        (
            policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
            .patch_embedding.weight
            - frozen_probe
        )
        .abs()
        .max()
        .detach()
        .cpu()
        .item()
    )
    language_delta = (
        (
            policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .input_layernorm.weight
            - language_probe
        )
        .abs()
        .max()
        .detach()
        .cpu()
        .item()
    )
    if args.steps > 0 and trainable_delta == 0.0:
        raise RuntimeError("The trainable action head did not update.")
    if args.train_mode == "expert_only" and frozen_delta != 0.0:
        raise RuntimeError(f"The frozen vision tower changed by {frozen_delta}.")
    if args.steps > 0 and args.train_mode in {"expert_vision", "full"} and frozen_delta == 0.0:
        raise RuntimeError("The trainable FP32 vision patch embedding did not update.")
    if args.train_mode in {"expert_only", "expert_vision"} and language_delta != 0.0:
        raise RuntimeError(f"The frozen language model changed by {language_delta}.")

    summary = {
        "status": "finetune_passed",
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": config.gradient_checkpointing,
        "train_mode": args.train_mode,
        "steps": args.steps,
        "dataset_frames": len(dataset),
        "dataset_episodes": dataset.num_episodes,
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_head_max_delta": trainable_delta,
        "vision_probe_max_delta": frozen_delta,
        "language_probe_max_delta": language_delta,
        "frozen_vlm_max_delta": (
            max(frozen_delta, language_delta) if args.train_mode == "expert_only" else None
        ),
        "memory_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "baseline_eval": baseline_metrics,
        "final_eval": final_metrics,
        "checkpoint_saved": not args.skip_save,
        "output_dir": str(args.output_dir),
        "action_loss_weights": action_loss_weights.tolist() if action_loss_weights is not None else None,
        "eval_samples": len(eval_indices),
        "eval_start_samples_requested": int(args.eval_start_samples),
        "dataset_alignment": dataset_alignment,
        "sampler": sampler_info,
    }

    if not args.skip_save:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        policy.eval()
        policy.save_pretrained(args.output_dir)
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
