#!/usr/bin/env python3
"""Train a visual/proprioceptive Pi0 contact-action head.

The head only consumes rollout-available inputs: frozen Pi0/SigLIP image
features, the language instruction, and the robot state.  Dataset-only object
metadata is intentionally ignored.  Episodes are split by policy seed so the
validation set contains unseen Pi0 rollout perturbations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Source:
    repo_id: str
    root: Path
    summary_jsonl: Path


def parse_int_spec(spec: str) -> set[int]:
    values: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(x) for x in token.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if not values:
        raise ValueError(f"Empty integer specification: {spec!r}")
    return values


def parse_source(values: list[str]) -> Source:
    if len(values) != 3:
        raise ValueError("--source expects REPO_ID ROOT SUMMARY_JSONL")
    return Source(values[0], Path(values[1]), Path(values[2]))


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_episode_seed_map(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("dataset_saved", False) or "episode_index" not in row:
            continue
        episode = int(row["episode_index"])
        seed = int(row.get("policy_seed", row["seed"]))
        if episode in mapping:
            raise ValueError(f"Duplicate episode {episode} in {path}")
        mapping[episode] = seed
    if not mapping:
        raise ValueError(f"No saved episode mapping found in {path}")
    return mapping


def selected_local_frames(
    actions: np.ndarray,
    first_frames: int,
    transition_pre: int,
    transition_post: int,
    transition_stride: int,
    bridge_stride: int,
    close_threshold: float,
) -> list[int]:
    count = int(actions.shape[0])
    selected = set(range(min(first_frames, count)))
    close_hits = np.flatnonzero(actions[:, 6] >= close_threshold)
    if len(close_hits):
        center = int(close_hits[0])
        start = max(0, center - transition_pre)
        stop = min(count, center + transition_post + 1)
        if bridge_stride > 0:
            selected.update(range(0, stop, bridge_stride))
        selected.update(range(start, stop, max(transition_stride, 1)))
        if stop > start:
            selected.add(stop - 1)
    return sorted(selected)


def load_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    train_seeds = parse_int_spec(args.train_seeds)
    val_seeds = parse_int_spec(args.val_seeds)
    overlap = train_seeds & val_seeds
    if overlap:
        raise ValueError(f"Train/validation seeds overlap: {sorted(overlap)}")

    rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_index, raw_source in enumerate(args.source):
        source = parse_source(raw_source)
        dataset = LeRobotDataset(source.repo_id, root=source.root)
        episode_column = tensor_to_numpy(dataset.hf_dataset["episode_index"]).astype(int)
        action_column = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
        episode_seed = load_episode_seed_map(source.summary_jsonl)
        selected_by_episode: dict[int, int] = {}
        for episode, seed in sorted(episode_seed.items()):
            split = "train" if seed in train_seeds else "val" if seed in val_seeds else None
            if split is None:
                continue
            raw_indices = np.flatnonzero(episode_column == episode)
            if not len(raw_indices):
                raise ValueError(f"Source {source.root} has no rows for mapped episode {episode}")
            episode_actions = action_column[raw_indices, :7]
            local_frames = selected_local_frames(
                episode_actions,
                first_frames=args.first_frames,
                transition_pre=args.transition_pre,
                transition_post=args.transition_post,
                transition_stride=args.transition_stride,
                bridge_stride=args.bridge_stride,
                close_threshold=args.close_threshold,
            )
            for local_frame in local_frames:
                rows.append(
                    {
                        "source_index": source_index,
                        "dataset": dataset,
                        "raw_index": int(raw_indices[local_frame]),
                        "episode": episode,
                        "seed": seed,
                        "local_frame": int(local_frame),
                        "split": split,
                        "action": episode_actions[local_frame].copy(),
                        "prev_action": episode_actions[max(local_frame - 1, 0)].copy(),
                    }
                )
            selected_by_episode[episode] = len(local_frames)
        source_summaries.append(
            {
                "repo_id": source.repo_id,
                "root": str(source.root),
                "summary_jsonl": str(source.summary_jsonl),
                "episodes": len(selected_by_episode),
                "selected_frames": int(sum(selected_by_episode.values())),
                "selected_by_episode": selected_by_episode,
            }
        )
    if not rows:
        raise ValueError("No samples selected")
    return rows, {
        "train_seeds": sorted(train_seeds),
        "val_seeds": sorted(val_seeds),
        "sources": source_summaries,
    }


def load_policy(args: argparse.Namespace):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    metadata = LeRobotDatasetMetadata(args.stats_repo_id, root=args.stats_dataset_root)
    policy = PI0Policy.from_pretrained(args.policy_path, dataset_stats=metadata.stats)
    policy.to(args.device)
    policy.eval()
    return policy


def make_projection(input_dim: int, output_dim: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    projection /= math.sqrt(float(output_dim))
    return projection.to(device=device)


def spatial_moments(tokens: torch.Tensor) -> torch.Tensor:
    batch, token_count, _ = tokens.shape
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"Expected square image token grid, got {token_count} tokens")
    coords = torch.linspace(-1.0, 1.0, side, device=tokens.device, dtype=tokens.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx = xx.reshape(1, token_count, 1)
    yy = yy.reshape(1, token_count, 1)
    mean = tokens.mean(dim=1)
    std = tokens.float().std(dim=1, unbiased=False).to(tokens.dtype)
    x_moment = (tokens * xx).mean(dim=1)
    y_moment = (tokens * yy).mean(dim=1)
    return torch.stack([mean, std, x_moment, y_moment], dim=1).reshape(batch, -1)


def collate_rows(rows: list[dict[str, Any]], image_keys: list[str], device: str) -> dict[str, Any]:
    items = [row["dataset"][row["raw_index"]] for row in rows]
    batch: dict[str, Any] = {
        "observation.state": torch.stack([item["observation.state"] for item in items]).to(device),
        "task": [str(item["task"]) for item in items],
    }
    for key in image_keys:
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items]).to(device)
    return batch


@torch.inference_mode()
def extract_features(
    args: argparse.Namespace, policy: Any, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    image_keys = list(policy.config.image_features.keys())
    image_projection: torch.Tensor | None = None
    language_projection: torch.Tensor | None = None
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        batch = collate_rows(chunk, image_keys, args.device)
        raw_state = batch["observation.state"].float()
        normalized = policy.normalize_inputs(batch)
        images, image_masks = policy.prepare_images(normalized)
        lang_tokens, lang_masks = policy.prepare_language(normalized)

        image_parts: list[torch.Tensor] = []
        for image, mask in zip(images, image_masks, strict=False):
            tokens = policy.model.paligemma_with_expert.embed_image(image).float()
            if image_projection is None:
                image_projection = make_projection(
                    tokens.shape[-1], args.projection_dim, args.projection_seed, args.device
                )
            moments = spatial_moments(tokens)
            moments = moments.reshape(tokens.shape[0], 4, tokens.shape[-1])
            projected = torch.matmul(moments, image_projection).reshape(tokens.shape[0], -1)
            projected = projected * mask[:, None].float()
            image_parts.append(projected)

        language = policy.model.paligemma_with_expert.embed_language_tokens(lang_tokens).float()
        language_mask = lang_masks[:, :, None].float()
        language = (language * language_mask).sum(dim=1) / language_mask.sum(dim=1).clamp_min(1.0)
        if language_projection is None:
            language_projection = make_projection(
                language.shape[-1], args.projection_dim, args.projection_seed + 1, args.device
            )
        language = language @ language_projection
        feature = torch.cat([*image_parts, language, raw_state[:, :6]], dim=1)

        for local_index, row in enumerate(chunk):
            action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
            label = np.concatenate([action[:3], action[6:7]]).astype(np.float32)
            labels.append(label)
            metadata.append({k: row[k] for k in ("source_index", "episode", "seed", "local_frame", "split")})
        features.append(feature.detach().cpu().numpy().astype(np.float32))
        print(
            f"[extract] {min(start + len(chunk), len(rows))}/{len(rows)}",
            flush=True,
        )

    assert image_projection is not None and language_projection is not None
    projection = torch.stack([image_projection.cpu(), language_projection.cpu()], dim=0).numpy()
    return np.concatenate(features), np.stack(labels), metadata, projection.astype(np.float32)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = pred - y
    target_close = y[:, 3] >= 0.5
    pred_close = pred[:, 3] >= 0.5
    positives = int(target_close.sum())
    negatives = int((~target_close).sum())
    true_positive_rate = float((pred_close[target_close]).mean()) if positives else None
    true_negative_rate = float((~pred_close[~target_close]).mean()) if negatives else None
    balanced = (
        float((true_positive_rate + true_negative_rate) / 2.0)
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    return {
        "samples": int(len(y)),
        "xyz_rmse_m": [float(v) for v in np.sqrt(np.mean(error[:, :3] ** 2, axis=0))],
        "xyz_mae_m": [float(v) for v in np.mean(np.abs(error[:, :3]), axis=0)],
        "xyz_sign_accuracy": [
            float(v)
            for v in np.mean(np.sign(pred[:, :3]) == np.sign(y[:, :3]), axis=0)
        ],
        "gripper_mae": float(np.mean(np.abs(error[:, 3]))),
        "gripper_accuracy": float(np.mean(pred_close == target_close)),
        "gripper_balanced_accuracy": balanced,
        "gripper_positive_count": positives,
        "gripper_negative_count": negatives,
        "gripper_true_positive_rate": true_positive_rate,
        "gripper_true_negative_rate": true_negative_rate,
    }


def fit_ridge_sweep(
    x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, alphas: list[float]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x_train = x[train_mask]
    y_train = y[train_mask]
    feature_mean = x_train.mean(axis=0).astype(np.float32)
    feature_std = x_train.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    xn = ((x - feature_mean) / feature_std).astype(np.float64)
    y_mean = y_train.mean(axis=0).astype(np.float64)
    xtr = xn[train_mask]
    ytr = y_train.astype(np.float64) - y_mean
    gram = xtr.T @ xtr
    rhs = xtr.T @ ytr
    identity = np.eye(gram.shape[0], dtype=np.float64)

    sweep: list[dict[str, Any]] = []
    candidates: list[dict[str, np.ndarray]] = []
    for alpha in alphas:
        weight = np.linalg.solve(gram + float(alpha) * identity, rhs)
        bias = y_mean
        pred = xn @ weight + bias
        xyz_min = y_train[:, :3].min(axis=0) - 0.02
        xyz_max = y_train[:, :3].max(axis=0) + 0.02
        pred[:, :3] = np.clip(pred[:, :3], xyz_min, xyz_max)
        pred[:, 3] = np.clip(pred[:, 3], 0.0, 1.0)
        train_metrics = regression_metrics(y[train_mask], pred[train_mask])
        val_metrics = regression_metrics(y[val_mask], pred[val_mask])
        balanced = val_metrics["gripper_balanced_accuracy"]
        if balanced is None:
            balanced = val_metrics["gripper_accuracy"]
        score = float(np.mean(val_metrics["xyz_rmse_m"]) / 0.02 + (1.0 - balanced))
        sweep.append(
            {
                "alpha": float(alpha),
                "selection_score": score,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        candidates.append(
            {
                "weight": weight.astype(np.float32),
                "bias": bias.astype(np.float32),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
            }
        )
    best_index = int(np.argmin([row["selection_score"] for row in sweep]))
    return candidates[best_index], sweep


def nearest_rms_distance(queries: np.ndarray, prototypes: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        squared = np.mean((query[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(distances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", nargs=3, required=True, metavar=("REPO", "ROOT", "JSONL"))
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--stats-repo-id", required=True)
    parser.add_argument("--stats-dataset-root", type=Path, required=True)
    parser.add_argument("--train-seeds", default="21-32")
    parser.add_argument("--val-seeds", default="33-36")
    parser.add_argument("--first-frames", type=int, default=16)
    parser.add_argument("--transition-pre", type=int, default=12)
    parser.add_argument("--transition-post", type=int, default=24)
    parser.add_argument("--transition-stride", type=int, default=2)
    parser.add_argument(
        "--bridge-stride",
        type=int,
        default=0,
        help="If positive, also sample continuously from takeover through the post-close window.",
    )
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260710)
    parser.add_argument(
        "--append-prev-action",
        action="store_true",
        help="Append the previous recorded 7-D action as legal one-step action history.",
    )
    parser.add_argument(
        "--stationary-action-threshold",
        type=float,
        default=0.0,
        help="If positive, thin consecutive rows whose action change norm is below this threshold.",
    )
    parser.add_argument("--stationary-keep", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--prototype-stride", type=int, default=1)
    parser.add_argument("--distance-quantile", type=float, default=0.99)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument(
        "--extra-feature-cache",
        action="append",
        type=Path,
        default=None,
        help="Append a compatible feature cache before fitting; projection matrices must match exactly.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows, manifest = load_manifest(args)
    if args.feature_cache is not None and args.feature_cache.exists():
        cached = np.load(args.feature_cache, allow_pickle=False)
        x = cached["features"].astype(np.float32)
        # Labels are rebuilt from the manifest.  This also makes old feature
        # caches safe after changes to the target representation.
        y = np.stack(
            [
                np.concatenate(
                    [
                        np.asarray(row["action"], dtype=np.float32)[:3],
                        np.asarray(row["action"], dtype=np.float32)[6:7],
                    ]
                )
                for row in rows
            ]
        ).astype(np.float32)
        split = cached["split"].astype(str)
        projection = cached["projection"].astype(np.float32)
        seed = cached["seed"].astype(np.int64)
        metadata = [
            {"split": str(split[i]), "seed": int(seed[i])}
            for i in range(len(split))
        ]
    else:
        policy = load_policy(args)
        x, y, metadata, projection = extract_features(args, policy, rows)
        if args.feature_cache is not None:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.feature_cache,
                features=x,
                labels=y,
                split=np.asarray([row["split"] for row in metadata]),
                seed=np.asarray([row["seed"] for row in metadata], dtype=np.int64),
                projection=projection,
            )

    samples_before_temporal_filter = int(len(x))
    if len(rows) != len(x):
        raise ValueError(f"Manifest/cache row mismatch: rows={len(rows)} features={len(x)}")
    if args.append_prev_action:
        previous = np.stack([np.asarray(row["prev_action"], dtype=np.float32)[:7] for row in rows])
        x = np.concatenate([x, previous], axis=1).astype(np.float32)
    if args.stationary_action_threshold > 0:
        keep_mask = np.zeros(len(rows), dtype=bool)
        current_key: tuple[int, int] | None = None
        stationary_run = 0
        for index, row in enumerate(rows):
            key = (int(row["source_index"]), int(row["episode"]))
            if key != current_key:
                current_key = key
                stationary_run = 0
            action = np.asarray(row["action"], dtype=np.float32)[:7]
            previous = np.asarray(row["prev_action"], dtype=np.float32)[:7]
            if float(np.linalg.norm(action - previous)) < float(args.stationary_action_threshold):
                stationary_run += 1
                keep_mask[index] = stationary_run <= max(int(args.stationary_keep), 1)
            else:
                stationary_run = 0
                keep_mask[index] = True
        x = x[keep_mask]
        y = y[keep_mask]
        metadata = [row for row, keep in zip(metadata, keep_mask, strict=True) if keep]

    extra_cache_summaries: list[dict[str, Any]] = []
    for extra_path in args.extra_feature_cache or []:
        extra = np.load(extra_path, allow_pickle=False)
        extra_projection = extra["projection"].astype(np.float32)
        if extra_projection.shape != projection.shape or not np.array_equal(extra_projection, projection):
            raise ValueError(f"Feature projection mismatch in extra cache {extra_path}")
        extra_x = extra["features"].astype(np.float32)
        extra_y = extra["labels"].astype(np.float32)
        extra_split = extra["split"].astype(str)
        extra_seed = extra["seed"].astype(np.int64)
        if len(extra_x) != len(extra_y) or len(extra_x) != len(extra_split):
            raise ValueError(f"Feature cache row mismatch in {extra_path}")
        x = np.concatenate([x, extra_x], axis=0)
        y = np.concatenate([y, extra_y], axis=0)
        metadata.extend(
            {"split": str(extra_split[i]), "seed": int(extra_seed[i]), "extra_cache": str(extra_path)}
            for i in range(len(extra_x))
        )
        extra_cache_summaries.append(
            {
                "path": str(extra_path),
                "samples": int(len(extra_x)),
                "train_samples": int(np.sum(extra_split == "train")),
                "val_samples": int(np.sum(extra_split == "val")),
            }
        )

    train_mask = np.asarray([row["split"] == "train" for row in metadata], dtype=bool)
    val_mask = np.asarray([row["split"] == "val" for row in metadata], dtype=bool)
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"Need non-empty train and validation sets: train={train_mask.sum()} val={val_mask.sum()}")
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    best, sweep = fit_ridge_sweep(x, y, train_mask, val_mask, alphas)
    best_row = min(sweep, key=lambda row: row["selection_score"])

    normalized = (x - best["feature_mean"]) / best["feature_std"]
    prototypes = normalized[train_mask][:: max(int(args.prototype_stride), 1)].astype(np.float32)
    val_nearest = nearest_rms_distance(normalized[val_mask], prototypes)
    distance_threshold = float(np.quantile(val_nearest, float(args.distance_quantile)))

    constant_pred = np.broadcast_to(y[train_mask].mean(axis=0), y.shape).copy()
    summary = {
        **manifest,
        "feature_dim": int(x.shape[1]),
        "output_layout": ["tcp_target_x", "tcp_target_y", "tcp_target_z", "gripper_closed"],
        "projection_dim": int(args.projection_dim),
        "append_prev_action": bool(args.append_prev_action),
        "stationary_action_threshold": float(args.stationary_action_threshold),
        "stationary_keep": int(args.stationary_keep),
        "samples_before_temporal_filter": samples_before_temporal_filter,
        "samples_after_temporal_filter": int(len(x)),
        "extra_feature_caches": extra_cache_summaries,
        "selection": {
            "first_frames": int(args.first_frames),
            "transition_pre": int(args.transition_pre),
            "transition_post": int(args.transition_post),
            "transition_stride": int(args.transition_stride),
            "bridge_stride": int(args.bridge_stride),
            "close_threshold": float(args.close_threshold),
        },
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "constant_mean_baseline_val": regression_metrics(y[val_mask], constant_pred[val_mask]),
        "alpha_sweep": sweep,
        "best": best_row,
        "ood_gate": {
            "prototype_count": int(len(prototypes)),
            "prototype_stride": int(args.prototype_stride),
            "distance_quantile": float(args.distance_quantile),
            "distance_threshold": distance_threshold,
            "val_nearest_rms_quantiles": {
                str(q): float(np.quantile(val_nearest, q))
                for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
            },
        },
        "fair_input_contract": [
            "observation.image",
            "observation.wrist_image",
            "language instruction",
            "observation.state[:6]",
        ],
        "forbidden_inputs_used": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **best,
        projection=projection,
        prototypes=prototypes,
        distance_threshold=np.asarray(distance_threshold, dtype=np.float32),
        projection_dim=np.asarray(args.projection_dim, dtype=np.int64),
        projection_seed=np.asarray(args.projection_seed, dtype=np.int64),
        feature_mode=np.asarray("siglip_spatial_moments_language_state"),
        output_mode=np.asarray("direct_eef_abs_and_gripper"),
        include_prev_action=np.asarray(bool(args.append_prev_action)),
        output_min=y[train_mask].min(axis=0).astype(np.float32),
        output_max=y[train_mask].max(axis=0).astype(np.float32),
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
