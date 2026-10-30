#!/usr/bin/env python3
"""Train ACT with a configurable action chunk and print chunk diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
import os
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms


def patch_lerobot_image_transform() -> None:
    """Handle parquet image rows decoded as dict payloads on some platforms."""
    import lerobot.common.datasets.lerobot_dataset as ds_mod
    import lerobot.common.datasets.utils as ds_utils

    def hf_transform_to_torch_patched(items_dict):
        to_tensor = T.ToTensor()
        for key in items_dict:
            first_item = items_dict[key][0]
            if isinstance(first_item, Image.Image):
                items_dict[key] = [to_tensor(img) for img in items_dict[key]]
            elif isinstance(first_item, dict) and ("bytes" in first_item or "path" in first_item):
                out = []
                for item in items_dict[key]:
                    if isinstance(item, dict) and item.get("bytes") is not None:
                        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                        out.append(to_tensor(img))
                    else:
                        out.append(item)
                items_dict[key] = out
            elif first_item is None:
                pass
            else:
                items_dict[key] = [
                    item if isinstance(item, str) else torch.tensor(item) for item in items_dict[key]
                ]
        return items_dict

    ds_utils.hf_transform_to_torch = hf_transform_to_torch_patched
    ds_mod.hf_transform_to_torch = hf_transform_to_torch_patched


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class Clamp01:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp(0, 1)


def tensor_to_device_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def _action_array(action: torch.Tensor) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy()
    return np.asarray(action)


def _first_action_vector(action: torch.Tensor) -> np.ndarray:
    action_np = _action_array(action)
    if action_np.ndim == 2:
        return action_np[0]
    return action_np


def _action_motion_norm(action: torch.Tensor) -> float:
    action_np = _action_array(action)
    if action_np.ndim == 2 and action_np.shape[0] > 1:
        return float(np.linalg.norm(action_np[-1, :6] - action_np[0, :6]))
    return 0.0


def _base_dataset_object(dataset):
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    return base


def _base_hf_dataset(dataset):
    base = _base_dataset_object(dataset)
    return getattr(base, "hf_dataset", None)


def _hf_column(dataset, name: str):
    base = _base_dataset_object(dataset)
    hf_dataset = getattr(base, "hf_dataset", None)
    if hf_dataset is None:
        return None
    cache = getattr(base, "_codex_hf_column_cache", None)
    if cache is None:
        cache = {}
        setattr(base, "_codex_hf_column_cache", cache)
    if name not in cache:
        cache[name] = hf_dataset[name]
    return cache[name]


def _has_timestamp_offsets(dataset) -> bool:
    base = dataset
    while True:
        if getattr(base, "timestamp_offsets", None):
            return True
        if not hasattr(base, "dataset"):
            return False
        base = base.dataset


def _timestamp_offsets(dataset) -> dict[int, float]:
    base = dataset
    while True:
        offsets = getattr(base, "timestamp_offsets", None)
        if offsets:
            return {int(k): float(v) for k, v in offsets.items()}
        if not hasattr(base, "dataset"):
            return {}
        base = base.dataset


def _raw_dataset_index(dataset, idx: int) -> int:
    if hasattr(dataset, "indices"):
        idx = int(dataset.indices[int(idx)])
    if hasattr(dataset, "dataset"):
        return _raw_dataset_index(dataset.dataset, idx)
    return int(idx)


def _raw_first_action_vector(dataset, idx: int) -> np.ndarray:
    hf_dataset = _base_hf_dataset(dataset)
    if hf_dataset is not None:
        return np.asarray(_hf_column(dataset, "action")[_raw_dataset_index(dataset, idx)], dtype=np.float32)
    return _first_action_vector(dataset[idx]["action"])


def _episode_ranges(dataset) -> list[tuple[int, int, int]]:
    hf_dataset = _base_hf_dataset(dataset)
    episode_column = _hf_column(dataset, "episode_index") if hf_dataset is not None and "episode_index" in hf_dataset.column_names else None
    episode_ids: list[int] = []
    for idx in range(len(dataset)):
        if episode_column is not None:
            episode_ids.append(int(episode_column[_raw_dataset_index(dataset, idx)]))
            continue
        item = dataset[idx]
        if "episode_index" in item:
            episode_ids.append(int(item["episode_index"]))
        else:
            episode_ids.append(0)

    if not episode_ids:
        return []
    ranges: list[tuple[int, int, int]] = []
    start = 0
    current = episode_ids[0]
    for idx, episode_id in enumerate(episode_ids[1:], start=1):
        if episode_id != current:
            ranges.append((start, idx, current))
            start = idx
            current = episode_id
    ranges.append((start, len(episode_ids), current))
    return ranges


def make_sampler(dataset, args: argparse.Namespace):
    if args.sample_mode == "uniform":
        return None

    weights = np.ones(len(dataset), dtype=np.float64)
    sampler_debug = {}
    episode_sample_weights = parse_episode_float_map(
        getattr(args, "episode_sample_weights", ""),
        "--episode-sample-weights",
    )
    if episode_sample_weights:
        episode_ranges = _episode_ranges(dataset)
        weighted_frames = 0
        matched_episodes: list[int] = []
        for start, end, episode_id in episode_ranges:
            factor = float(episode_sample_weights.get(int(episode_id), 1.0))
            if int(episode_id) in episode_sample_weights:
                matched_episodes.append(int(episode_id))
                weighted_frames += end - start
            weights[start:end] *= max(factor, 1e-9)
        sampler_debug.update(
            {
                "episode_sample_weights": episode_sample_weights,
                "episode_sample_weight_matched": matched_episodes,
                "episode_sample_weighted_frames": int(weighted_frames),
            }
        )
    if args.sample_mode in ("early", "early-motion", "trajectory"):
        n = max(args.early_frame_count, 0)
        if args.per_episode_early and n > 0:
            early_weighted = 0
            episode_ranges = _episode_ranges(dataset)
            for start, end, _episode_id in episode_ranges:
                hi = min(start + n, end)
                if hi > start:
                    weights[start:hi] *= max(args.early_frame_weight, 1e-6)
                    early_weighted += hi - start
            sampler_debug.update(
                {
                    "per_episode_early": True,
                    "episode_count": int(len(episode_ranges)),
                    "early_weighted_frames": int(early_weighted),
                }
            )
        else:
            n = min(n, len(weights))
            weights[:n] *= max(args.early_frame_weight, 1e-6)
            sampler_debug.update({"per_episode_early": False, "early_weighted_frames": int(n)})

    if args.sample_mode in ("motion", "early-motion"):
        motions = np.zeros(len(dataset), dtype=np.float64)
        for idx in range(len(dataset)):
            if _base_hf_dataset(dataset) is not None:
                if idx > 0:
                    motions[idx] = float(
                        np.linalg.norm(
                            _raw_first_action_vector(dataset, idx)[:6]
                            - _raw_first_action_vector(dataset, idx - 1)[:6]
                        )
                    )
            else:
                motions[idx] = _action_motion_norm(dataset[idx]["action"])
        scale = float(np.percentile(motions, 95)) if np.any(motions > 0) else 0.0
        if scale > 1e-9:
            weights *= 1.0 + args.motion_frame_weight * np.clip(motions / scale, 0.0, 1.0)

    if args.sample_mode == "trajectory":
        actions = np.stack([_raw_first_action_vector(dataset, idx) for idx in range(len(dataset))])
        gripper = actions[:, 6]
        closed_mask = gripper > 0.5
        weights[closed_mask] *= max(args.gripper_closed_weight, 1e-6)

        transitions = np.flatnonzero(np.abs(np.diff(gripper, prepend=gripper[0])) > args.gripper_transition_threshold)
        window = max(args.gripper_transition_window, 0)
        for transition_idx in transitions:
            lo = max(int(transition_idx) - window, 0)
            hi = min(int(transition_idx) + window + 1, len(weights))
            weights[lo:hi] *= max(args.gripper_transition_weight, 1e-6)
        tail_count = min(max(args.tail_frame_count, 0), len(weights))
        if tail_count:
            weights[-tail_count:] *= max(args.tail_frame_weight, 1e-6)

        one_step_motion = np.zeros(len(dataset), dtype=np.float64)
        if len(dataset) > 1:
            one_step_motion[1:] = np.linalg.norm(actions[1:, :6] - actions[:-1, :6], axis=1)
        scale = float(np.percentile(one_step_motion, 95)) if np.any(one_step_motion > 0) else 0.0
        if scale > 1e-9:
            weights *= 1.0 + args.motion_frame_weight * np.clip(one_step_motion / scale, 0.0, 1.0)

        sampler_debug.update(
            {
                "gripper_closed_frames": int(closed_mask.sum()),
                "gripper_transition_count": int(len(transitions)),
                "gripper_transition_indexes": [int(x) for x in transitions],
                "gripper_closed_weight": args.gripper_closed_weight,
                "gripper_transition_weight": args.gripper_transition_weight,
                "gripper_transition_window": args.gripper_transition_window,
                "tail_frame_count": args.tail_frame_count,
                "tail_frame_weight": args.tail_frame_weight,
            }
        )

    weights = np.clip(weights, 1e-9, None)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    print(
        json.dumps(
            {
                "event": "sampler",
                "sample_mode": args.sample_mode,
                "weight_min": round(float(weights.min()), 4),
                "weight_max": round(float(weights.max()), 4),
                "weight_mean": round(float(weights.mean()), 4),
                "early_frame_count": args.early_frame_count,
                "early_frame_weight": args.early_frame_weight,
                "motion_frame_weight": args.motion_frame_weight,
                **sampler_debug,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return sampler


class TimestampStateDataset(torch.utils.data.Dataset):
    """Append the scalar LeRobot timestamp to observation.state."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[idx])
        state = item["observation.state"]
        timestamp = item["timestamp"]
        if not isinstance(timestamp, torch.Tensor):
            timestamp = torch.tensor(timestamp)
        timestamp = timestamp.reshape(1).to(dtype=state.dtype)
        item["observation.state"] = torch.cat([state, timestamp])
        return item


class EpisodeSubsetDataset(torch.utils.data.Dataset):
    """Index into a subset of LeRobot episodes without copying parquet data."""

    def __init__(self, dataset, episode_indexes: list[int]):
        self.dataset = dataset
        self.episode_indexes = episode_indexes
        self.indices: list[int] = []
        for episode_index in episode_indexes:
            start = int(dataset.episode_data_index["from"][episode_index].item())
            end = int(dataset.episode_data_index["to"][episode_index].item())
            self.indices.extend(range(start, end))
        if not self.indices:
            raise ValueError(f"No frames selected by episode indexes: {episode_indexes}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[int(idx)]]


class EpisodeTimestampOffsetDataset(torch.utils.data.Dataset):
    """Add a timestamp offset for selected episodes without copying parquet data."""

    def __init__(self, dataset, timestamp_offsets: dict[int, float]):
        self.dataset = dataset
        self.timestamp_offsets = {int(k): float(v) for k, v in timestamp_offsets.items() if abs(float(v)) > 1e-9}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[int(idx)])
        episode_index = item.get("episode_index")
        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.detach().cpu().reshape(-1)[0])
        elif episode_index is not None:
            episode_index = int(episode_index)
        offset = self.timestamp_offsets.get(episode_index, 0.0)
        if offset:
            timestamp = item["timestamp"]
            if isinstance(timestamp, torch.Tensor):
                item["timestamp"] = timestamp + timestamp.new_tensor(offset)
            else:
                item["timestamp"] = np.asarray(timestamp, dtype=np.float32) + np.float32(offset)
        return item


class EpisodeSourceFlagDataset(torch.utils.data.Dataset):
    """Append a per-episode source/correction flag to observation.state."""

    def __init__(self, dataset, source_flags: dict[int, float]):
        self.dataset = dataset
        self.source_flags = {int(k): float(v) for k, v in source_flags.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[int(idx)])
        state = item["observation.state"]
        episode_index = item.get("episode_index")
        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.detach().cpu().reshape(-1)[0])
        elif episode_index is not None:
            episode_index = int(episode_index)
        flag = self.source_flags.get(episode_index, 0.0)
        flag_tensor = torch.as_tensor([flag], dtype=state.dtype, device=state.device)
        item["observation.state"] = torch.cat([state, flag_tensor])
        return item


class TimestampOnlyStateDataset(torch.utils.data.Dataset):
    """Use only the scalar LeRobot timestamp as observation.state."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[idx])
        state = item["observation.state"]
        timestamp = item["timestamp"]
        if not isinstance(timestamp, torch.Tensor):
            timestamp = torch.tensor(timestamp)
        item["observation.state"] = timestamp.reshape(1).to(dtype=state.dtype)
        return item


def _timestamp_float(timestamp) -> float:
    if isinstance(timestamp, torch.Tensor):
        return float(timestamp.detach().cpu().reshape(-1)[0])
    return float(np.asarray(timestamp).reshape(-1)[0])


def make_timestamp_env_feature(timestamp, max_timestamp: float, frequencies: int) -> np.ndarray:
    ts = _timestamp_float(timestamp)
    if frequencies <= 0:
        return np.asarray([ts], dtype=np.float32)
    phase = np.clip(ts / max(max_timestamp, 1e-6), 0.0, 1.0)
    values = [phase]
    for k in range(1, frequencies + 1):
        angle = 2.0 * math.pi * k * phase
        values.extend([math.sin(angle), math.cos(angle)])
    return np.asarray(values, dtype=np.float32)


class TimestampEnvStateDataset(torch.utils.data.Dataset):
    """Use the scalar LeRobot timestamp as observation.environment_state."""

    def __init__(self, dataset, max_timestamp: float, frequencies: int):
        self.dataset = dataset
        self.max_timestamp = max_timestamp
        self.frequencies = frequencies

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[idx])
        state = item["observation.state"]
        timestamp = item["timestamp"]
        env_state = make_timestamp_env_feature(timestamp, self.max_timestamp, self.frequencies)
        item["observation.environment_state"] = torch.as_tensor(env_state, dtype=state.dtype)
        item.pop("observation.image", None)
        item.pop("observation.wrist_image", None)
        return item


class ObjInitStateDataset(torch.utils.data.Dataset):
    """Append mug/plate initial pose to observation.state."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[idx])
        state = item["observation.state"]
        obj_init = item["obj_init"]
        if not isinstance(obj_init, torch.Tensor):
            obj_init = torch.tensor(obj_init)
        item["observation.state"] = torch.cat([state, obj_init.reshape(-1).to(dtype=state.dtype)])
        return item


class BinarizeGripperActionDataset(torch.utils.data.Dataset):
    """Replace the gripper target with a hard open/closed label."""

    def __init__(self, dataset, threshold: float):
        self.dataset = dataset
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.dataset[idx])
        action = item["action"].clone()
        action[..., 6] = (action[..., 6] > self.threshold).to(dtype=action.dtype)
        item["action"] = action
        return item


def _timestamp_stats_from_values(values: np.ndarray, template: dict) -> dict:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute timestamp stats from an empty dataset")
    out = {}
    for key in template:
        if key == "min":
            value = values.min()
        elif key == "max":
            value = values.max()
        elif key == "mean":
            value = values.mean()
        elif key == "std":
            value = values.std()
        elif key == "count":
            value = values.size
        else:
            value = np.asarray(template[key]).reshape(-1)[0]
        out[key] = np.asarray([value], dtype=np.float32)
    return out


def append_timestamp_state_stats(stats: dict, timestamps: np.ndarray | None = None) -> dict:
    patched = deepcopy(stats)
    state_stats = patched["observation.state"]
    timestamp_stats = patched["timestamp"]
    if timestamps is not None:
        timestamp_stats = _timestamp_stats_from_values(timestamps, timestamp_stats)
    for key in ("min", "max", "mean", "std"):
        state_stats[key] = np.concatenate([state_stats[key], timestamp_stats[key].reshape(1)])
    state_stats["std"][-1] = max(float(state_stats["std"][-1]), 1e-6)
    return patched


def append_obj_init_state_stats(stats: dict) -> dict:
    patched = deepcopy(stats)
    state_stats = patched["observation.state"]
    obj_stats = patched["obj_init"]
    for key in ("min", "max", "mean", "std"):
        state_stats[key] = np.concatenate([state_stats[key], np.asarray(obj_stats[key]).reshape(-1)])
    state_stats["std"] = np.maximum(state_stats["std"], 1e-6)
    return patched


def append_episode_source_flag_state_stats(stats: dict, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute source flag stats from an empty dataset")
    patched = deepcopy(stats)
    state_stats = patched["observation.state"]
    for key in ("min", "max", "mean", "std"):
        if key == "min":
            value = values.min()
        elif key == "max":
            value = values.max()
        elif key == "mean":
            value = values.mean()
        else:
            value = max(float(values.std()), 1e-6)
        state_stats[key] = np.concatenate([state_stats[key], np.asarray([value], dtype=np.float32)])
    state_stats["std"][-1] = max(float(state_stats["std"][-1]), 1e-6)
    return patched


def timestamp_only_state_stats(stats: dict) -> dict:
    patched = deepcopy(stats)
    timestamp_stats = patched["timestamp"]
    patched["observation.state"] = {}
    for key in ("min", "max", "mean", "std"):
        patched["observation.state"][key] = np.asarray(timestamp_stats[key], dtype=np.float32).reshape(1)
    patched["observation.state"]["std"][0] = max(float(patched["observation.state"]["std"][0]), 1e-6)
    return patched


def timestamp_env_state_stats(
    stats: dict,
    timestamps: np.ndarray | None = None,
    max_timestamp: float | None = None,
    frequencies: int = 0,
) -> dict:
    patched = deepcopy(stats)
    if frequencies <= 0:
        timestamp_stats = patched["timestamp"]
        patched["observation.environment_state"] = {}
        for key in ("min", "max", "mean", "std"):
            patched["observation.environment_state"][key] = np.asarray(timestamp_stats[key], dtype=np.float32).reshape(1)
        patched["observation.environment_state"]["std"][0] = max(
            float(patched["observation.environment_state"]["std"][0]), 1e-6
        )
        return patched

    if timestamps is None or max_timestamp is None:
        raise ValueError("timestamps and max_timestamp are required when timestamp env frequencies are enabled")
    features = np.stack(
        [make_timestamp_env_feature(timestamp, max_timestamp, frequencies) for timestamp in timestamps]
    ).astype(np.float32)
    patched["observation.environment_state"] = {}
    for key in ("min", "max", "mean", "std"):
        if key == "min":
            patched["observation.environment_state"][key] = features.min(axis=0)
        elif key == "max":
            patched["observation.environment_state"][key] = features.max(axis=0)
        elif key == "mean":
            patched["observation.environment_state"][key] = features.mean(axis=0)
        else:
            patched["observation.environment_state"][key] = np.maximum(features.std(axis=0), 1e-6)
    return patched


def dataset_timestamps(dataset) -> np.ndarray:
    hf_dataset = _base_hf_dataset(dataset)
    if hf_dataset is not None:
        offsets = _timestamp_offsets(dataset)
        timestamp_column = _hf_column(dataset, "timestamp")
        episode_column = _hf_column(dataset, "episode_index") if "episode_index" in hf_dataset.column_names else None
        values = []
        for idx in range(len(dataset)):
            raw_idx = _raw_dataset_index(dataset, idx)
            timestamp = _timestamp_float(timestamp_column[raw_idx])
            episode_index = int(episode_column[raw_idx]) if episode_column is not None else -1
            values.append(timestamp + float(offsets.get(episode_index, 0.0)))
        return np.asarray(values, dtype=np.float32)
    return np.asarray([_timestamp_float(dataset[idx]["timestamp"]) for idx in range(len(dataset))], dtype=np.float32)


def parse_episode_indexes(spec: str) -> list[int]:
    indexes = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            indexes.append(int(part))
    return indexes


def parse_episode_range(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        start = int(lo)
        end = int(hi)
        if end < start:
            raise ValueError(f"Invalid episode range: {spec}")
        return list(range(start, end + 1))
    return [int(spec)]


def parse_episode_timestamp_offsets(spec: str) -> dict[int, float]:
    offsets: dict[int, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Expected EPISODES:SECONDS in --episode-timestamp-offsets, got {part!r}")
        episode_spec, offset_spec = part.split(":", 1)
        offset = float(offset_spec)
        for episode_index in parse_episode_range(episode_spec):
            offsets[int(episode_index)] = offset
    return offsets


def parse_episode_float_map(spec: str, arg_name: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Expected EPISODES:VALUE in {arg_name}, got {part!r}")
        episode_spec, value_spec = part.split(":", 1)
        value = float(value_spec)
        for episode_index in parse_episode_range(episode_spec):
            values[int(episode_index)] = value
    return values


def dataset_episode_source_values(dataset, source_flags: dict[int, float]) -> np.ndarray:
    hf_dataset = _base_hf_dataset(dataset)
    if hf_dataset is not None:
        episode_column = _hf_column(dataset, "episode_index") if "episode_index" in hf_dataset.column_names else None
        values = []
        for idx in range(len(dataset)):
            raw_idx = _raw_dataset_index(dataset, idx)
            episode_index = int(episode_column[raw_idx]) if episode_column is not None else -1
            values.append(float(source_flags.get(episode_index, 0.0)))
        return np.asarray(values, dtype=np.float32)
    values = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        episode_index = item.get("episode_index", -1)
        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.detach().cpu().reshape(-1)[0])
        values.append(float(source_flags.get(int(episode_index), 0.0)))
    return np.asarray(values, dtype=np.float32)


def make_policy_and_data(args: argparse.Namespace, device: torch.device):
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    patch_lerobot_image_transform()

    enabled_phase_modes = [
        args.append_timestamp_to_state,
        args.timestamp_only_state,
        args.timestamp_env_state,
    ]
    if sum(bool(x) for x in enabled_phase_modes) > 1:
        raise ValueError(
            "--append-timestamp-to-state, --timestamp-only-state and --timestamp-env-state are mutually exclusive"
        )

    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    if not args.keep_wrist_image:
        input_features.pop("observation.wrist_image", None)
    dataset_stats = dataset_metadata.stats
    if args.timestamp_env_state:
        env_state_dim = 1 + 2 * max(args.timestamp_env_frequencies, 0)
        input_features = {
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(env_state_dim,)),
        }
    elif args.timestamp_only_state:
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        dataset_stats = timestamp_only_state_stats(dataset_stats)
    elif args.append_timestamp_to_state:
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    if args.append_episode_source_flag_to_state:
        if "observation.state" not in input_features:
            raise ValueError("--append-episode-source-flag-to-state requires observation.state input")
        state_dim = int(input_features["observation.state"].shape[0]) + 1
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    if args.append_obj_init_to_state:
        if "observation.state" not in input_features:
            raise ValueError("--append-obj-init-to-state requires observation.state input")
        state_dim = int(input_features["observation.state"].shape[0]) + 6
        input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps or args.chunk_size,
        use_vae=not args.disable_vae,
        kl_weight=args.kl_weight,
        dropout=args.dropout,
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    image_transform = transforms.Compose([AddGaussianNoise(std=args.image_noise_std), Clamp01()])
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps=delta_timestamps,
        root=args.root,
        image_transforms=image_transform if args.image_noise_std > 0 else None,
    )
    if args.episode_indexes:
        episode_indexes = parse_episode_indexes(args.episode_indexes)
        dataset = EpisodeSubsetDataset(dataset, episode_indexes)
        print(
            json.dumps(
                {
                    "event": "episode_subset",
                    "episode_indexes": episode_indexes,
                    "frames": len(dataset),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    timestamp_offsets = parse_episode_timestamp_offsets(args.episode_timestamp_offsets)
    if timestamp_offsets:
        dataset = EpisodeTimestampOffsetDataset(dataset, timestamp_offsets)
        print(
            json.dumps(
                {
                    "event": "episode_timestamp_offsets",
                    "episode_offsets": timestamp_offsets,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.append_timestamp_to_state:
        timestamp_values = dataset_timestamps(dataset) if timestamp_offsets else None
        dataset_stats = append_timestamp_state_stats(dataset_stats, timestamps=timestamp_values)
        dataset = TimestampStateDataset(dataset)
    elif args.timestamp_only_state:
        dataset = TimestampOnlyStateDataset(dataset)
    elif args.timestamp_env_state:
        timestamps = dataset_timestamps(dataset)
        max_timestamp = float(timestamps.max()) if len(timestamps) else 1.0
        dataset_stats = timestamp_env_state_stats(
            dataset_stats,
            timestamps=timestamps,
            max_timestamp=max_timestamp,
            frequencies=args.timestamp_env_frequencies,
        )
        dataset = TimestampEnvStateDataset(dataset, max_timestamp, args.timestamp_env_frequencies)
    episode_source_flags = parse_episode_float_map(args.episode_source_flags, "--episode-source-flags")
    if args.append_episode_source_flag_to_state:
        source_values = dataset_episode_source_values(dataset, episode_source_flags)
        dataset_stats = append_episode_source_flag_state_stats(dataset_stats, source_values)
        dataset = EpisodeSourceFlagDataset(dataset, episode_source_flags)
        print(
            json.dumps(
                {
                    "event": "episode_source_flags",
                    "episode_source_flags": episode_source_flags,
                    "source_flag_mean": round(float(source_values.mean()), 6) if len(source_values) else 0.0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.append_obj_init_to_state:
        dataset_stats = append_obj_init_state_stats(dataset_stats)
        dataset = ObjInitStateDataset(dataset)
    if args.binarize_gripper_target:
        dataset = BinarizeGripperActionDataset(dataset, args.gripper_binary_threshold)
    policy = ACTPolicy(cfg, dataset_stats=dataset_stats)
    policy.to(device)
    return policy, dataset, dataset_metadata


def normalized_gripper_threshold(policy, threshold: float, device: torch.device) -> torch.Tensor:
    """Map a raw gripper threshold into ACT's normalized action space."""
    from lerobot.configs.types import FeatureType
    from lerobot.common.policies.normalize import NormalizationMode

    feature = policy.config.output_features["action"]
    norm_mode = (
        policy.config.normalization_mapping.get(feature.type)
        or policy.config.normalization_mapping.get(FeatureType.ACTION)
        or policy.config.normalization_mapping.get("ACTION")
        or NormalizationMode.IDENTITY
    )
    threshold_t = torch.tensor(threshold, dtype=torch.float32, device=device)
    if norm_mode is NormalizationMode.IDENTITY:
        return threshold_t

    buffer = getattr(policy.normalize_targets, "buffer_action")
    if norm_mode is NormalizationMode.MEAN_STD:
        mean = buffer["mean"].to(device)
        std = buffer["std"].to(device)
        return (threshold_t - mean[..., 6]) / (std[..., 6] + 1e-8)
    if norm_mode is NormalizationMode.MIN_MAX:
        min_v = buffer["min"].to(device)
        max_v = buffer["max"].to(device)
        return ((threshold_t - min_v[..., 6]) / (max_v[..., 6] - min_v[..., 6] + 1e-8)) * 2 - 1
    raise ValueError(f"Unsupported action normalization mode: {norm_mode}")


def act_forward_with_optional_gripper_loss(
    policy,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """ACT forward pass plus an optional binary gripper classification loss."""
    if args.gripper_bce_weight <= 0:
        return policy.forward(batch)

    raw_action = batch["action"]
    batch = policy.normalize_inputs(batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = [batch[key] for key in policy.config.image_features]

    batch = policy.normalize_targets(batch)
    actions_hat, (mu_hat, log_sigma_x2_hat) = policy.model(batch)

    valid_mask = ~batch["action_is_pad"]
    l1_loss = (F.l1_loss(batch["action"], actions_hat, reduction="none") * valid_mask.unsqueeze(-1)).mean()

    loss_dict = {"l1_loss": l1_loss.item()}
    if policy.config.use_vae:
        mean_kld = (
            (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
        )
        loss_dict["kld_loss"] = mean_kld.item()
        loss = l1_loss + mean_kld * policy.config.kl_weight
    else:
        loss = l1_loss

    target_bin = (raw_action[..., 6] > args.gripper_binary_threshold).to(dtype=actions_hat.dtype, device=device)
    threshold_norm = normalized_gripper_threshold(policy, args.gripper_binary_threshold, device)
    gripper_logits = (actions_hat[..., 6] - threshold_norm) * args.gripper_bce_logit_scale
    bce = F.binary_cross_entropy_with_logits(gripper_logits, target_bin, reduction="none")
    valid_float = valid_mask.to(dtype=bce.dtype)
    gripper_bce = (bce * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    gripper_acc = (((gripper_logits > 0) == (target_bin > 0.5)) * valid_mask).sum().to(torch.float32)
    gripper_acc = gripper_acc / valid_float.sum().clamp_min(1.0)

    loss = loss + args.gripper_bce_weight * gripper_bce
    loss_dict["gripper_bce_loss"] = gripper_bce.item()
    loss_dict["gripper_bin_acc"] = gripper_acc.item()
    return loss, loss_dict


@torch.no_grad()
def chunk_diagnostics(policy, dataset, device: torch.device, indexes: list[int]) -> list[dict]:
    policy.eval()
    rows = []
    for idx in indexes:
        if idx >= len(dataset):
            continue
        sample = dataset[idx]
        batch = {"task": ["Put mug cup on the plate"]}
        if "observation.state" in policy.config.input_features:
            batch["observation.state"] = sample["observation.state"].unsqueeze(0).to(device)
        elif "observation.state" in sample:
            batch["observation.state"] = sample["observation.state"].unsqueeze(0).to(device)
        if "observation.environment_state" in policy.config.input_features:
            batch["observation.environment_state"] = sample["observation.environment_state"].unsqueeze(0).to(device)
        if "observation.image" in policy.config.input_features:
            batch["observation.image"] = sample["observation.image"].unsqueeze(0).to(device)
        if "observation.wrist_image" in policy.config.input_features:
            batch["observation.wrist_image"] = sample["observation.wrist_image"].unsqueeze(0).to(device)
        norm_batch = policy.normalize_inputs(dict(batch))
        if policy.config.image_features:
            norm_batch = dict(norm_batch)
            norm_batch["observation.images"] = [norm_batch[key] for key in policy.config.image_features]
        pred = policy.model(norm_batch)[0][:, : policy.config.n_action_steps]
        pred = policy.unnormalize_outputs({"action": pred})["action"][0].detach().cpu().numpy()
        gt_action = sample["action"].numpy()
        if gt_action.ndim == 2:
            gt = gt_action[: policy.config.n_action_steps]
        else:
            gt = []
            for j in range(policy.config.n_action_steps):
                gt.append(dataset[min(idx + j, len(dataset) - 1)]["action"].numpy())
            gt = np.stack(gt)
        rows.append(
            {
                "idx": int(idx),
                "timestamp": float(sample["timestamp"]),
                "mae": round(float(np.abs(pred - gt).mean()), 6),
                "pred_motion": round(float(np.linalg.norm(pred[-1, :6] - pred[0, :6])), 6),
                "gt_motion": round(float(np.linalg.norm(gt[-1, :6] - gt[0, :6])), 6),
                "pred_first": np.round(pred[0], 4).tolist(),
                "pred_last": np.round(pred[-1], 4).tolist(),
                "gt_first": np.round(gt[0], 4).tolist(),
                "gt_last": np.round(gt[-1], 4).tolist(),
            }
        )
    policy.train()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="datawhale_eai_pnp")
    parser.add_argument("--root", default="./demo_data")
    parser.add_argument("--output-dir", default="./ckpt/act_chunk100_rocm")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--disable-vae", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-noise-std", type=float, default=0.02)
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "early", "motion", "early-motion", "trajectory"],
        default="uniform",
        help="Weighted sampling mode for one-demo ACT overfit diagnostics.",
    )
    parser.add_argument("--early-frame-count", type=int, default=250)
    parser.add_argument("--early-frame-weight", type=float, default=6.0)
    parser.add_argument(
        "--per-episode-early",
        action="store_true",
        help="Apply --early-frame-count/weight at the start of every episode instead of only dataset index 0.",
    )
    parser.add_argument("--motion-frame-weight", type=float, default=8.0)
    parser.add_argument("--gripper-closed-weight", type=float, default=6.0)
    parser.add_argument("--gripper-transition-weight", type=float, default=25.0)
    parser.add_argument("--gripper-transition-window", type=int, default=24)
    parser.add_argument("--gripper-transition-threshold", type=float, default=0.5)
    parser.add_argument("--tail-frame-count", type=int, default=0)
    parser.add_argument("--tail-frame-weight", type=float, default=1.0)
    parser.add_argument(
        "--append-timestamp-to-state",
        action="store_true",
        help="Append timestamp seconds to observation.state as a phase feature.",
    )
    parser.add_argument(
        "--timestamp-only-state",
        action="store_true",
        help="Replace observation.state with timestamp seconds to test exact one-demo trajectory alignment.",
    )
    parser.add_argument(
        "--timestamp-env-state",
        action="store_true",
        help="Use timestamp seconds as observation.environment_state and drop images/state for open-loop replay.",
    )
    parser.add_argument(
        "--timestamp-env-frequencies",
        type=int,
        default=0,
        help="Number of sinusoidal phase frequencies to append when using --timestamp-env-state.",
    )
    parser.add_argument(
        "--append-obj-init-to-state",
        action="store_true",
        help="Append mug/plate initial pose (obj_init, 6D) to observation.state.",
    )
    parser.add_argument(
        "--keep-wrist-image",
        action="store_true",
        help="Keep observation.wrist_image as an ACT visual input.",
    )
    parser.add_argument(
        "--episode-indexes",
        default="",
        help="Comma-separated LeRobot episode indexes to train on, for overfit diagnostics.",
    )
    parser.add_argument(
        "--episode-timestamp-offsets",
        default="",
        help="Comma-separated EPISODES:SECONDS timestamp offsets, e.g. '50-71:2.0'.",
    )
    parser.add_argument(
        "--episode-sample-weights",
        default="",
        help="Comma-separated EPISODES:WEIGHT sampler multipliers, e.g. '49-71:0.25'. Unlisted episodes use 1.",
    )
    parser.add_argument(
        "--append-episode-source-flag-to-state",
        action="store_true",
        help="Append a scalar per-episode source/correction flag to observation.state.",
    )
    parser.add_argument(
        "--episode-source-flags",
        default="",
        help="Comma-separated EPISODES:VALUE source flags, e.g. '49-71:1.0'. Unlisted episodes use 0.",
    )
    parser.add_argument(
        "--binarize-gripper-target",
        action="store_true",
        help="Replace gripper action targets with hard 0/1 labels before ACT target normalization.",
    )
    parser.add_argument(
        "--gripper-binary-threshold",
        type=float,
        default=0.5,
        help="Raw gripper action threshold used for binary target/loss diagnostics.",
    )
    parser.add_argument(
        "--gripper-bce-weight",
        type=float,
        default=0.0,
        help="Optional auxiliary BCE weight for classifying gripper open/closed in normalized action space.",
    )
    parser.add_argument(
        "--gripper-bce-logit-scale",
        type=float,
        default=4.0,
        help="Scale applied to normalized gripper distance from threshold before BCE.",
    )
    parser.add_argument("--diag-indexes", default="0,10,50,80,100,200,900")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested accelerator is unavailable")
    else:
        device = torch.device(args.device)
    print(json.dumps({"event": "start", "args": vars(args), "device": str(device)}, ensure_ascii=False), flush=True)

    policy, dataset, _ = make_policy_and_data(args, device)
    print(json.dumps({"event": "model_contract", "torch": torch.__version__,
                      "hip": torch.version.hip, "device": str(device),
                      "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                      "frozen_parameters": sum(p.numel() for p in policy.parameters() if not p.requires_grad),
                      "gradient_accumulation": 1, "effective_batch_size": args.batch_size}), flush=True)
    sampler = make_sampler(dataset, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    data_iter = cycle(dataloader)

    start = time.time()
    policy.train()
    for step in range(1, args.steps + 1):
        batch = tensor_to_device_batch(next(data_iter), device)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = act_forward_with_optional_gripper_loss(policy, batch, args, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at update {step}")
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            payload = {
                "event": "train",
                "step": step,
                "loss": round(float(loss.item()), 6),
                "elapsed_s": round(time.time() - start, 1),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else 0,
            }
            payload.update({key: round(float(value), 6) for key, value in loss_dict.items()})
            print(json.dumps(payload, ensure_ascii=False), flush=True)

        if args.eval_freq > 0 and (step % args.eval_freq == 0 or step == args.steps):
            indexes = [int(x) for x in args.diag_indexes.split(",") if x.strip()]
            for row in chunk_diagnostics(policy, dataset, device, indexes):
                row["event"] = "chunk_diag"
                row["step"] = step
                print(json.dumps(row, ensure_ascii=False), flush=True)

        if args.save_every > 0 and step % args.save_every == 0:
            step_output_dir = Path(args.output_dir) / f"step_{step}"
            step_output_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(step_output_dir)
            print(json.dumps({"event": "saved_step", "step": step, "output_dir": str(step_output_dir)}, ensure_ascii=False), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    print(json.dumps({"event": "saved", "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
