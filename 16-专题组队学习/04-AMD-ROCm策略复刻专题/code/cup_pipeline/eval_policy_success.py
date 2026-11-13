#!/usr/bin/env python3
"""Roll out migrated LeRobot policies and report MuJoCo success rate.

This script intentionally reuses the tutorial environment success checks:
`SimpleEnv.check_success()` for ACT and `SimpleEnv2.check_success()` for
language-conditioned SmolVLA/Pi0-style policies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["act", "smolvla", "pi0"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-action-steps", type=int, default=400)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--record-all-episodes", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--act-n-action-steps",
        type=int,
        default=None,
        help="Override ACT config.n_action_steps at evaluation time without changing the checkpoint.",
    )
    parser.add_argument(
        "--act-force-dataset-gripper",
        action="store_true",
        help="For ACT diagnostics, replace the policy gripper command with the dataset action at the same timestep.",
    )
    parser.add_argument(
        "--act-clamp-timestamp",
        action="store_true",
        help="Clamp ACT timestamp state/env features to the maximum timestamp seen in the training dataset.",
    )
    parser.add_argument(
        "--reset-policy-each-action",
        action="store_true",
        help="For chunked policies, discard queued actions and replan from the latest observation each action step.",
    )
    parser.add_argument("--act-policy-path", type=Path, default=Path("./ckpt/act_y"))
    parser.add_argument("--act-repo-id", default="datawhale_eai_pnp")
    parser.add_argument("--act-dataset-root", type=Path, default=Path("./demo_data"))
    parser.add_argument(
        "--act-episode-timestamp-offsets",
        default="",
        help="Comma-separated EPISODES:SECONDS offsets used when the ACT checkpoint was trained, e.g. '50-71:2.0'.",
    )
    parser.add_argument(
        "--act-episode-source-flags",
        default="",
        help="Comma-separated EPISODES:VALUE source flags used when the ACT checkpoint was trained, e.g. '49-71:1.0'.",
    )
    parser.add_argument(
        "--physical-success",
        action="store_true",
        help="For ACT, require a physically plausible pick/place in addition to the environment terminal rule.",
    )
    parser.add_argument(
        "--physical-min-lift",
        type=float,
        default=0.06,
        help="Minimum mug center lift above its reset height for physical_success.",
    )
    parser.add_argument(
        "--physical-min-lift-steps",
        type=int,
        default=3,
        help="Number of control ticks the mug must stay lifted for physical_success.",
    )
    parser.add_argument(
        "--physical-final-upright-cos",
        type=float,
        default=0.85,
        help="Minimum final mug local z-axis/world z cosine for physical_success.",
    )
    parser.add_argument(
        "--smolvla-policy-path",
        type=Path,
        default=Path("./ckpt/smolvla_rocm_smoke5/checkpoints/last/pretrained_model"),
    )
    parser.add_argument(
        "--pi0-policy-path",
        type=Path,
        default=Path("./ckpt/pi0_datawhale_eai/checkpoints/last/pretrained_model"),
    )
    return parser.parse_args()


def _capture_panel(frames, image, wrist_image):
    if frames is None:
        return
    left = np.asarray(image).astype(np.uint8)
    right = np.asarray(wrist_image).astype(np.uint8)
    if left.ndim != 3 or right.ndim != 3:
        return
    height = max(left.shape[0], right.shape[0])
    width = max(left.shape[1], right.shape[1])
    canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], width : width + right.shape[1]] = right
    frames.append(canvas)


def _write_episode_media(args, seed, frames, trace):
    if args.video_dir is None and args.trace_dir is None:
        return {}
    payload = {}
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_dir / f"episode_seed{int(seed)}.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
        payload["_trace_path"] = str(trace_path)
    if args.video_dir is not None and frames:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.video_dir / f"episode_seed{int(seed)}.tmp.mp4"
        try:
            import imageio.v2 as imageio
            imageio.mimsave(video_path, frames, fps=args.hz, quality=7)
            payload["_video_path"] = str(video_path)
        except Exception as exc:
            raise RuntimeError(f"Cannot save evaluation video for seed {seed}") from exc
    return payload


def finalize_case_media(args, row, selected_cases):
    media = row.pop("_media", {}) or {}
    kind = "success" if bool(row.get("physical_success", row.get("success"))) else "failure"
    if kind not in selected_cases or args.record_all_episodes:
        for key in ("_video_path", "_trace_path"):
            path = media.get(key)
            if not path:
                continue
            source = Path(path)
            if not source.exists():
                continue
            suffix = "video_path" if key == "_video_path" else "trace_path"
            extension = ".mp4" if key == "_video_path" else ".json"
            target = source.with_name(f"{args.policy}_full14_{kind}_seed{int(row['seed'])}{extension}")
            source.replace(target)
            row[suffix] = str(target)
        selected_cases.add(kind)
    else:
        for path in media.values():
            candidate = Path(path)
            if candidate.exists():
                candidate.unlink()
    return row


def to_tensor_image(image: np.ndarray) -> torch.Tensor:
    transform = transforms.ToTensor()
    return transform(Image.fromarray(image).resize((256, 256)))


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
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Expected EPISODES:SECONDS in timestamp offsets, got {part!r}")
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


def make_act_policy(
    device: str,
    policy_path: Path,
    repo_id: str,
    dataset_root: Path,
    n_action_steps: int | None = None,
    episode_timestamp_offsets: str = "",
    episode_source_flags: str = "",
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset_stats = dataset_metadata.stats
    timestamp_offsets = parse_episode_timestamp_offsets(episode_timestamp_offsets)
    source_flags = parse_episode_float_map(episode_source_flags, "--act-episode-source-flags")
    offset_timestamps = None
    dataset_for_stats = None
    config_path = policy_path / "config.json"
    desired_state_dim = None
    desired_env_state_dim = None
    base_state_dim = 6
    timestamp_env_frequencies = 0
    timestamp_env_max = float(np.asarray(dataset_stats["timestamp"]["max"]).reshape(-1)[0])
    if timestamp_offsets or source_flags:
        dataset_for_stats = LeRobotDataset(repo_id, root=dataset_root)
    if timestamp_offsets:
        offset_timestamps = dataset_timestamps(dataset_for_stats, timestamp_offsets)
        timestamp_env_max = float(offset_timestamps.max()) if len(offset_timestamps) else timestamp_env_max
    if config_path.exists():
        config_data = json.loads(config_path.read_text())
        desired_state_dim = (
            config_data.get("input_features", {})
            .get("observation.state", {})
            .get("shape", [None])[0]
        )
        desired_env_state_dim = (
            config_data.get("input_features", {})
            .get("observation.environment_state", {})
            .get("shape", [None])[0]
        )
    if desired_env_state_dim == 1:
        dataset_stats = timestamp_env_state_stats(dataset_stats)
    elif desired_env_state_dim and desired_env_state_dim > 1:
        if (desired_env_state_dim - 1) % 2 != 0:
            raise ValueError(f"Unsupported timestamp env_state dim: {desired_env_state_dim}")
        timestamp_env_frequencies = (desired_env_state_dim - 1) // 2
        if dataset_for_stats is None:
            dataset_for_stats = LeRobotDataset(repo_id, root=dataset_root)
        timestamps = offset_timestamps if offset_timestamps is not None else dataset_timestamps(dataset_for_stats)
        timestamp_env_max = float(timestamps.max()) if len(timestamps) else timestamp_env_max
        dataset_stats = timestamp_env_state_stats(
            dataset_stats,
            timestamps=timestamps,
            max_timestamp=timestamp_env_max,
            frequencies=timestamp_env_frequencies,
        )
    elif desired_state_dim == 1:
        dataset_stats = timestamp_only_state_stats(dataset_stats)
    elif desired_state_dim == base_state_dim + 1:
        dataset_stats = append_timestamp_state_stats(dataset_stats, timestamps=offset_timestamps)
    elif desired_state_dim == base_state_dim + 2:
        if dataset_for_stats is None:
            dataset_for_stats = LeRobotDataset(repo_id, root=dataset_root)
        source_values = dataset_episode_source_values(dataset_for_stats, source_flags)
        dataset_stats = append_timestamp_state_stats(dataset_stats, timestamps=offset_timestamps)
        dataset_stats = append_episode_source_flag_state_stats(dataset_stats, source_values)
    elif desired_state_dim == base_state_dim + 6:
        dataset_stats = append_obj_init_state_stats(dataset_stats)
    elif desired_state_dim == base_state_dim + 1 + 6:
        dataset_stats = append_timestamp_state_stats(dataset_stats, timestamps=offset_timestamps)
        dataset_stats = append_obj_init_state_stats(dataset_stats)
    elif desired_state_dim == base_state_dim + 1 + 1 + 6:
        if dataset_for_stats is None:
            dataset_for_stats = LeRobotDataset(repo_id, root=dataset_root)
        source_values = dataset_episode_source_values(dataset_for_stats, source_flags)
        dataset_stats = append_timestamp_state_stats(dataset_stats, timestamps=offset_timestamps)
        dataset_stats = append_episode_source_flag_state_stats(dataset_stats, source_values)
        dataset_stats = append_obj_init_state_stats(dataset_stats)
    policy = ACTPolicy.from_pretrained(
        policy_path,
        dataset_stats=dataset_stats,
    )
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError("--act-n-action-steps must be >= 1")
        if n_action_steps > policy.config.chunk_size:
            raise ValueError(
                f"--act-n-action-steps={n_action_steps} exceeds checkpoint chunk_size={policy.config.chunk_size}"
            )
        policy.config.n_action_steps = n_action_steps
        policy.reset()
    policy.to(device)
    policy.eval()
    policy.timestamp_env_frequencies = timestamp_env_frequencies
    policy.timestamp_env_max_timestamp = timestamp_env_max
    policy.act_source_flag = 0.0
    return policy


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


def timestamp_phase_env_state_stats(
    stats: dict, timestamps: np.ndarray, max_timestamp: float, frequencies: int
) -> dict:
    patched = deepcopy(stats)
    features = np.stack(
        [make_timestamp_env_feature(timestamp, max_timestamp, frequencies) for timestamp in timestamps]
    ).astype(np.float32)
    patched["observation.environment_state"] = {
        "min": features.min(axis=0),
        "max": features.max(axis=0),
        "mean": features.mean(axis=0),
        "std": np.maximum(features.std(axis=0), 1e-6),
    }
    return patched


def dataset_timestamps(dataset, episode_timestamp_offsets: dict[int, float] | None = None) -> np.ndarray:
    offsets = episode_timestamp_offsets or {}
    if not offsets:
        return np.asarray([_timestamp_float(x) for x in dataset.hf_dataset["timestamp"]], dtype=np.float32)
    timestamp_column = dataset.hf_dataset["timestamp"]
    episode_column = dataset.hf_dataset["episode_index"] if "episode_index" in dataset.hf_dataset.column_names else None
    values = []
    for raw_idx in range(len(timestamp_column)):
        timestamp = _timestamp_float(timestamp_column[raw_idx])
        episode_index = int(episode_column[raw_idx]) if episode_column is not None else -1
        values.append(timestamp + float(offsets.get(episode_index, 0.0)))
    return np.asarray(values, dtype=np.float32)


def dataset_episode_source_values(dataset, source_flags: dict[int, float]) -> np.ndarray:
    episode_column = dataset.hf_dataset["episode_index"] if "episode_index" in dataset.hf_dataset.column_names else None
    values = []
    for raw_idx in range(len(dataset.hf_dataset)):
        episode_index = int(episode_column[raw_idx]) if episode_column is not None else -1
        values.append(float(source_flags.get(episode_index, 0.0)))
    return np.asarray(values, dtype=np.float32)


def timestamp_env_state_stats(
    stats: dict,
    timestamps: np.ndarray | None = None,
    max_timestamp: float | None = None,
    frequencies: int = 0,
) -> dict:
    if frequencies <= 0:
        return _timestamp_env_state_stats_1d(stats)
    if timestamps is None or max_timestamp is None:
        raise ValueError("timestamps and max_timestamp are required when timestamp env frequencies are enabled")
    return timestamp_phase_env_state_stats(stats, timestamps, max_timestamp, frequencies)


def _timestamp_env_state_stats_1d(stats: dict) -> dict:
    patched = deepcopy(stats)
    timestamp_stats = patched["timestamp"]
    patched["observation.environment_state"] = {}
    for key in ("min", "max", "mean", "std"):
        patched["observation.environment_state"][key] = np.asarray(timestamp_stats[key], dtype=np.float32).reshape(1)
    patched["observation.environment_state"]["std"][0] = max(
        float(patched["observation.environment_state"]["std"][0]), 1e-6
    )
    return patched


def get_act_state(env, policy, action_steps: int, hz: float) -> np.ndarray:
    state = np.asarray(env.get_ee_pose(), dtype=np.float32)
    obj_init = np.asarray(getattr(env, "obj_init_pose", []), dtype=np.float32).reshape(-1)
    expected_dim = policy.config.input_features["observation.state"].shape[0]
    timestamp = action_steps / hz
    source_flag = float(getattr(policy, "act_source_flag", 0.0))
    if getattr(policy, "clamp_timestamp_to_dataset", False):
        timestamp = min(timestamp, float(getattr(policy, "timestamp_env_max_timestamp", timestamp)))
    if expected_dim == 1:
        return np.asarray([timestamp], dtype=np.float32)
    if expected_dim == state.shape[0]:
        return state
    if expected_dim == state.shape[0] + 1:
        return np.concatenate([state, np.asarray([timestamp], dtype=np.float32)])
    if expected_dim == state.shape[0] + 2:
        return np.concatenate([state, np.asarray([timestamp, source_flag], dtype=np.float32)])
    if obj_init.size == 6 and expected_dim == state.shape[0] + obj_init.size:
        return np.concatenate([state, obj_init])
    if obj_init.size == 6 and expected_dim == state.shape[0] + 1 + obj_init.size:
        return np.concatenate([state, np.asarray([timestamp], dtype=np.float32), obj_init])
    if obj_init.size == 6 and expected_dim == state.shape[0] + 2 + obj_init.size:
        return np.concatenate([state, np.asarray([timestamp, source_flag], dtype=np.float32), obj_init])
    raise ValueError(f"ACT policy expects state dim {expected_dim}, env produced {state.shape[0]}")


def make_smolvla_policy(device: str, policy_path: Path):
    os.environ.setdefault("HF_HOME", "/tmp/aup_hf_smolvla")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/aup_hf_datasets_smolvla")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dataset_metadata = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root="./demo_data_language")
    policy_path = Path(policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"SmolVLA checkpoint directory not found: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(
        str(policy_path),
        local_files_only=True,
        dataset_stats=dataset_metadata.stats,
    )
    policy.to(device)
    policy.eval()
    return policy


def make_pi0_policy(device: str, policy_path: Path):
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets_pi0"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    dataset_metadata = LeRobotDatasetMetadata("datawhale_eai_pnp_language", root="./demo_data_language")
    policy_path = Path(policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Pi0 checkpoint directory not found: {policy_path}")
    policy = PI0Policy.from_pretrained(
        str(policy_path),
        local_files_only=True,
        dataset_stats=dataset_metadata.stats,
    )
    policy.to(device)
    policy.eval()
    return policy


def get_smolvla_debug(env) -> dict:
    p_mug = env.env.get_p_body(env.obj_target)
    p_plate = env.env.get_p_body("body_obj_plate_11")
    p_tcp = env.env.get_p_body("tcp_link")
    return {
        "xy_dist": float(np.linalg.norm(p_mug[:2] - p_plate[:2])),
        "target_pos": [round(float(x), 4) for x in p_mug],
        "plate_pos": [round(float(x), 4) for x in p_plate],
        "tcp_pos": [round(float(x), 4) for x in p_tcp],
        "gripper": float(env.env.get_qpos_joint("rh_r1")[0]),
        "success": bool(env.check_success()),
    }


def _round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(x), ndigits) for x in values]


def get_body_upright_cos(env, body_name: str) -> float:
    """Return local z-axis alignment with world +z."""
    try:
        R = env.env.get_R_body(body_name)
        return float(R[2, 2])
    except Exception:
        return float("nan")


def has_finger_mug_contact(env) -> bool:
    if hasattr(env, "_xarm7_has_gripper_contact"):
        try:
            return bool(env._xarm7_has_gripper_contact())
        except Exception:
            pass
    try:
        contact_pairs = env.env.get_contact_body_names()
    except Exception:
        return False
    finger_tokens = ("finger", "knuckle", "gripper", "rh_")
    mug_bodies = {"object_mug_5", "body_obj_mug_5"}
    for b1, b2 in contact_pairs:
        s = {str(b1), str(b2)}
        has_mug = bool(s & mug_bodies)
        has_finger = any(any(tok in body for tok in finger_tokens) for body in s)
        if has_mug and has_finger:
            return True
    return False


def init_act_physical_tracker(env) -> dict:
    p_mug = env.env.get_p_body("body_obj_mug_5")
    p_plate = env.env.get_p_body("body_obj_plate_11")
    return {
        "initial_mug_pos": _round_list(p_mug),
        "initial_plate_pos": _round_list(p_plate),
        "initial_mug_z": float(p_mug[2]),
        "max_mug_z": float(p_mug[2]),
        "max_mug_lift": 0.0,
        "lifted_steps": 0,
        "finger_mug_contact_steps": 0,
        "closed_finger_mug_contact_steps": 0,
    }


def update_act_physical_tracker(env, tracker: dict, args: argparse.Namespace) -> None:
    p_mug = env.env.get_p_body("body_obj_mug_5")
    mug_z = float(p_mug[2])
    tracker["max_mug_z"] = max(float(tracker["max_mug_z"]), mug_z)
    tracker["max_mug_lift"] = max(
        float(tracker["max_mug_lift"]),
        mug_z - float(tracker["initial_mug_z"]),
    )
    if mug_z - float(tracker["initial_mug_z"]) >= float(args.physical_min_lift):
        tracker["lifted_steps"] += 1
    contact = has_finger_mug_contact(env)
    if contact:
        tracker["finger_mug_contact_steps"] += 1
    if contact and hasattr(env, "_get_gripper_open_norm"):
        try:
            if float(env._get_gripper_open_norm()) < 0.5:
                tracker["closed_finger_mug_contact_steps"] += 1
        except Exception:
            pass


def get_act_physical_debug(env, tracker: dict, args: argparse.Namespace, base_debug: dict | None = None) -> dict:
    debug = dict(base_debug or {})
    p_mug = env.env.get_p_body("body_obj_mug_5")
    final_upright_cos = get_body_upright_cos(env, "body_obj_mug_5")
    lifted_enough = int(tracker["lifted_steps"]) >= int(args.physical_min_lift_steps)
    final_upright = bool(np.isfinite(final_upright_cos) and final_upright_cos >= float(args.physical_final_upright_cos))
    terminal_success = bool(debug.get("success", env.check_success()))
    debug.update(
        {
            "physical_success": bool(terminal_success and lifted_enough and final_upright),
            "physical_lifted_enough": bool(lifted_enough),
            "physical_final_upright": bool(final_upright),
            "physical_min_lift": float(args.physical_min_lift),
            "physical_min_lift_steps": int(args.physical_min_lift_steps),
            "physical_final_upright_cos_threshold": float(args.physical_final_upright_cos),
            "initial_mug_pos": tracker["initial_mug_pos"],
            "initial_plate_pos": tracker["initial_plate_pos"],
            "final_mug_pos": _round_list(p_mug),
            "max_mug_z": float(tracker["max_mug_z"]),
            "max_mug_lift": float(tracker["max_mug_lift"]),
            "lifted_steps": int(tracker["lifted_steps"]),
            "finger_mug_contact_steps": int(tracker["finger_mug_contact_steps"]),
            "closed_finger_mug_contact_steps": int(tracker["closed_finger_mug_contact_steps"]),
            "final_mug_upright_cos": final_upright_cos,
        }
    )
    return debug


def act_stop_success(env, tracker: dict, args: argparse.Namespace, debug: dict) -> bool:
    if not env.check_success():
        return False
    if not args.physical_success:
        return True
    physical_debug = get_act_physical_debug(env, tracker, args, debug)
    return bool(physical_debug["physical_success"])


def rollout_act(args: argparse.Namespace, policy, seed: int) -> dict:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from mujoco_env.y_env import SimpleEnv

    dataset = None
    if args.act_force_dataset_gripper:
        dataset = LeRobotDataset(args.act_repo_id, root=args.act_dataset_root)

    env = SimpleEnv("./asset/example_scene_y.xml", action_type="joint_angle")
    try:
        env.reset(seed=seed)
        policy.reset()
        action_steps = 0
        sim_steps = 0
        last_debug = {}
        physical_tracker = init_act_physical_tracker(env)
        success_ever = False
        first_success_step = None
        physical_success_ever = False
        first_physical_success_step = None
        frames = [] if args.video_dir is not None else None
        trace = []
        start = time.time()
        while action_steps < args.max_action_steps and env.env.is_viewer_alive():
            env.step_env()
            sim_steps += 1
            if not env.env.loop_every(HZ=args.hz):
                continue
            update_act_physical_tracker(env, physical_tracker, args)
            if hasattr(env, "get_success_debug"):
                last_debug = env.get_success_debug()
            last_debug = get_act_physical_debug(env, physical_tracker, args, last_debug)
            if env.check_success():
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            if act_stop_success(env, physical_tracker, args, last_debug):
                break

            state = None
            batch = {
                "task": ["Put mug cup on the plate"],
                "timestamp": torch.tensor([action_steps / args.hz], dtype=torch.float32, device=args.device),
            }
            if "observation.state" in policy.config.input_features:
                state = get_act_state(env, policy, action_steps, args.hz)
                batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=args.device)
            if "observation.environment_state" in policy.config.input_features:
                env_state = make_timestamp_env_feature(
                    action_steps / args.hz,
                    getattr(policy, "timestamp_env_max_timestamp", action_steps / args.hz),
                    getattr(policy, "timestamp_env_frequencies", 0),
                )
                batch["observation.environment_state"] = torch.tensor(
                    env_state[None, :], dtype=torch.float32, device=args.device
                )
            if "observation.state" not in batch:
                batch["observation.state"] = torch.zeros((1, 1), dtype=torch.float32, device=args.device)
            if policy.config.image_features:
                image, wrist_image = env.grab_image()
                _capture_panel(frames, image, wrist_image)
                if "observation.image" in policy.config.input_features:
                    batch["observation.image"] = to_tensor_image(image).unsqueeze(0).to(args.device)
                if "observation.wrist_image" in policy.config.input_features:
                    batch["observation.wrist_image"] = to_tensor_image(wrist_image).unsqueeze(0).to(args.device)
            if args.reset_policy_each_action:
                policy.reset()
            with torch.no_grad():
                action = policy.select_action(batch)[0].detach().cpu().numpy()
            if dataset is not None:
                gt_action = dataset[min(action_steps, len(dataset) - 1)]["action"].numpy()
                action[6] = gt_action[6]
            env.step(action)
            action_steps += 1
            if args.render:
                env.render()
            update_act_physical_tracker(env, physical_tracker, args)
            if hasattr(env, "get_success_debug"):
                last_debug = env.get_success_debug()
            last_debug = get_act_physical_debug(env, physical_tracker, args, last_debug)
            if env.check_success():
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            trace.append({
                "step": int(action_steps),
                "action": [float(x) for x in action],
                "debug": last_debug,
            })
            if act_stop_success(env, physical_tracker, args, last_debug):
                break
        success = bool(success_ever or env.check_success())
        if hasattr(env, "get_success_debug"):
            last_debug = env.get_success_debug()
        last_debug = get_act_physical_debug(env, physical_tracker, args, last_debug)
        physical_success = bool(physical_success_ever or last_debug["physical_success"])
        last_debug.update(
            {
                "legacy_success_ever": bool(success),
                "first_legacy_success_step": first_success_step,
                "physical_success_ever": bool(physical_success),
                "first_physical_success_step": first_physical_success_step,
            }
        )
        result = {
            "policy": "act",
            "seed": seed,
            "success": success,
            "physical_success": physical_success,
            "action_steps": action_steps,
            "sim_steps": sim_steps,
            "elapsed_s": round(time.time() - start, 3),
            "act_force_dataset_gripper": bool(args.act_force_dataset_gripper),
            "debug": last_debug,
        }
        result["_media"] = _write_episode_media(args, seed, frames, trace)
        return result
    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass


def rollout_language_policy(args: argparse.Namespace, policy, seed: int) -> dict:
    from mujoco_env.y_env2 import SimpleEnv2

    env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    try:
        env.reset(seed=seed)
        policy.reset()
        action_steps = 0
        sim_steps = 0
        frames = [] if args.video_dir is not None else None
        trace = []
        start = time.time()
        while action_steps < args.max_action_steps and env.env.is_viewer_alive():
            env.step_env()
            sim_steps += 1
            if not env.env.loop_every(HZ=args.hz):
                continue
            if env.check_success():
                break

            state = env.get_joint_state()[:6]
            image, wrist_image = env.grab_image()
            _capture_panel(frames, image, wrist_image)
            batch = {
                "observation.state": torch.tensor([state], dtype=torch.float32, device=args.device),
                "observation.image": to_tensor_image(image).unsqueeze(0).to(args.device),
                "observation.wrist_image": to_tensor_image(wrist_image).unsqueeze(0).to(args.device),
                "task": [env.instruction],
            }
            if args.reset_policy_each_action:
                policy.reset()
            with torch.no_grad():
                action = policy.select_action(batch)[0, :7].detach().cpu().numpy()
            env.step(action)
            action_steps += 1
            trace.append({
                "step": int(action_steps),
                "state": [float(x) for x in state],
                "action": [float(x) for x in action],
            })
            if args.render:
                env.render()
            if env.check_success():
                break
        result = {
            "policy": getattr(args, "policy", "smolvla"),
            "seed": seed,
            "success": bool(env.check_success()),
            "action_steps": action_steps,
            "sim_steps": sim_steps,
            "elapsed_s": round(time.time() - start, 3),
            "instruction": getattr(env, "instruction", None),
            "debug": get_smolvla_debug(env),
        }
        result["_media"] = _write_episode_media(args, seed, frames, trace)
        return result
    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass


rollout_smolvla = rollout_language_policy


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but torch.cuda.is_available() is false")

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.policy == "act":
        policy = make_act_policy(
            args.device,
            args.act_policy_path,
            args.act_repo_id,
            args.act_dataset_root,
            args.act_n_action_steps,
            args.act_episode_timestamp_offsets,
            args.act_episode_source_flags,
        )
        policy.clamp_timestamp_to_dataset = bool(args.act_clamp_timestamp)
        rollout = rollout_act
    elif args.policy == "smolvla":
        policy = make_smolvla_policy(args.device, args.smolvla_policy_path)
        rollout = rollout_language_policy
    else:
        policy = make_pi0_policy(args.device, args.pi0_policy_path)
        rollout = rollout_language_policy

    results = []
    selected_cases = set()
    for offset in range(args.episodes):
        seed = args.seed_start + offset
        result = rollout(args, policy, seed)
        result = finalize_case_media(args, result, selected_cases)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.output_jsonl is not None:
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    success_count = sum(1 for r in results if r["success"])
    physical_success_count = sum(1 for r in results if r.get("physical_success", r["success"]))
    summary = {
        "policy": args.policy,
        "episodes": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / max(args.episodes, 1),
        "physical_success_count": physical_success_count,
        "physical_success_rate": physical_success_count / max(args.episodes, 1),
        "seed_start": args.seed_start,
        "max_action_steps": args.max_action_steps,
        "act_n_action_steps": args.act_n_action_steps,
        "act_force_dataset_gripper": bool(args.act_force_dataset_gripper),
        "physical_success_enabled": bool(args.physical_success),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
