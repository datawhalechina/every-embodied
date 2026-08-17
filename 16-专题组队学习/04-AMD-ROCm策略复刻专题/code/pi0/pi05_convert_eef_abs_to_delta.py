#!/usr/bin/env python3
"""Convert LeRobot EEF-absolute labels to exact state-relative EEF deltas."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_stats
from lerobot.datasets.v30.augment_dataset_quantile_stats import (
    compute_quantile_stats_for_dataset,
    has_quantile_stats,
)
from mujoco_env.y_env2 import SimpleEnv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--scene", type=Path, default=Path("asset/example_scene_y2.xml"))
    parser.add_argument("--position-profile", default="pnp_generalization_v1")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=4, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def hardlink_copy(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            return dst
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                raise
            return shutil.copy2(src, dst)

    shutil.copytree(source, output, copy_function=link_or_copy)


def make_fk_environment(scene: Path, position_profile: str) -> SimpleEnv2:
    return SimpleEnv2(str(scene), action_type="joint_angle", position_profile=position_profile)


def tcp_from_joint_state(env: SimpleEnv2, state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < 6:
        raise ValueError(f"Expected at least 6 joint values, got {state.shape}")
    env.env.data.qpos[:6] = state[:6]
    env.env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.env.model, env.env.data)
    return np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float64)


def convert_parquet(path: Path, env: SimpleEnv2) -> tuple[int, float, float]:
    table = pq.read_table(path)
    states = table.column("observation.state").to_pylist()
    actions = table.column("action").to_pylist()
    converted: list[list[float]] = []
    max_spatial_abs = 0.0
    max_reconstruction_error = 0.0
    for state, action in zip(states, actions, strict=True):
        action_abs = np.asarray(action, dtype=np.float64).reshape(7)
        tcp = tcp_from_joint_state(env, np.asarray(state, dtype=np.float64))
        action_delta = np.zeros(7, dtype=np.float32)
        action_delta[:3] = (action_abs[:3] - tcp).astype(np.float32)
        action_delta[6] = np.float32(action_abs[6])
        reconstructed = tcp + action_delta[:3].astype(np.float64)
        max_spatial_abs = max(max_spatial_abs, float(np.max(np.abs(action_delta[:3]))))
        max_reconstruction_error = max(
            max_reconstruction_error,
            float(np.max(np.abs(reconstructed - action_abs[:3]))),
        )
        converted.append(action_delta.tolist())

    action_index = table.schema.get_field_index("action")
    action_type = table.schema.field(action_index).type
    table = table.set_column(action_index, "action", pa.array(converted, type=action_type))
    temp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temp, compression="snappy")
    os.replace(temp, path)
    return len(converted), max_spatial_abs, max_reconstruction_error


def update_feature_name(output: Path) -> None:
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"]["names"] = ["eef_delta_xyz_gripper"]
    atomic_json(info_path, info)


def recompute_stats(repo_id: str, output: Path) -> dict[str, Any]:
    metadata = LeRobotDatasetMetadata(repo_id, root=output)
    dataset = LeRobotDataset(repo_id, root=output)
    stats = compute_quantile_stats_for_dataset(dataset)
    stats_path = output / "meta" / "stats.json"
    if stats_path.exists():
        stats_path.unlink()
    write_stats(stats, metadata.root)
    metadata = LeRobotDatasetMetadata(repo_id, root=output)
    return {
        "episodes": int(dataset.num_episodes),
        "frames": int(dataset.num_frames),
        "fps": int(dataset.fps),
        "has_quantile_stats": bool(has_quantile_stats(metadata.stats)),
        "action_stats": metadata.stats["action"],
    }


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    source_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No parquet files under {source / 'data'}")
    source_probe_hashes = {str(path.relative_to(source)): sha256(path) for path in source_files[:3]}

    hardlink_copy(source, output)
    env = make_fk_environment(args.scene.resolve(), args.position_profile)
    total_frames = 0
    max_spatial_abs = 0.0
    max_reconstruction_error = 0.0
    for path in sorted((output / "data").glob("chunk-*/*.parquet")):
        frames, spatial_abs, reconstruction_error = convert_parquet(path, env)
        total_frames += frames
        max_spatial_abs = max(max_spatial_abs, spatial_abs)
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)

    update_feature_name(output)
    stats_summary = recompute_stats(args.repo_id, output)
    source_probe_hashes_after = {
        str(path.relative_to(source)): sha256(path) for path in source_files[:3]
    }
    if source_probe_hashes_after != source_probe_hashes:
        raise RuntimeError("Source parquet changed during hard-link conversion")
    if total_frames != stats_summary["frames"]:
        raise RuntimeError(f"Frame mismatch: converted={total_frames}, dataset={stats_summary['frames']}")
    if max_spatial_abs > 0.005:
        raise RuntimeError(f"Unexpected EEF delta magnitude: {max_spatial_abs}")

    first_table = pq.read_table(
        sorted((output / "data").glob("chunk-*/*.parquet"))[0],
        columns=["action"],
    )
    first_action = first_table.column("action")[0].as_py()
    summary = {
        "repo_id": args.repo_id,
        "source_root": str(source),
        "output_root": str(output),
        "conversion": "eef_abs_xyz_to_fk_exact_eef_delta_xyz; gripper unchanged",
        "source_probe_hashes_unchanged": True,
        "parquet_files": len(source_files),
        "converted_frames": total_frames,
        "max_spatial_abs": max_spatial_abs,
        "max_reconstruction_error": max_reconstruction_error,
        "first_action": first_action,
        **stats_summary,
    }
    summary_path = output.parent / f"{output.name}_conversion_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
