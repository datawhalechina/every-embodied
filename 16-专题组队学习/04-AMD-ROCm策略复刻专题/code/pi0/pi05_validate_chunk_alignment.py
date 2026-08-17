#!/usr/bin/env python3

"""Validate LeRobot action chunks against canonical parquet trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hf_transform_to_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--uniform-samples", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def load_canonical_columns(root: Path) -> dict[str, np.ndarray]:
    names = ["action", "episode_index", "frame_index", "index"]
    batches: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for path in sorted(root.glob("data/chunk-*/*.parquet")):
        table = pq.read_table(path, columns=names)
        for name in names:
            batches[name].append(np.asarray(table[name].to_pylist()))
    if not batches["index"]:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    values = {name: np.concatenate(parts, axis=0) for name, parts in batches.items()}
    expected = np.arange(len(values["index"]), dtype=values["index"].dtype)
    if not np.array_equal(values["index"], expected):
        bad = np.flatnonzero(values["index"] != expected)
        raise ValueError(
            f"Physical parquet order is not canonical: {len(bad)} mismatches, "
            f"first position={int(bad[0])}, stored index={int(values['index'][bad[0]])}"
        )
    return values


def episode_bounds(episode_indices: np.ndarray) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    for episode in np.unique(episode_indices):
        positions = np.flatnonzero(episode_indices == episode)
        start, end = int(positions[0]), int(positions[-1] + 1)
        if not np.array_equal(positions, np.arange(start, end)):
            raise ValueError(f"Episode {episode} is not physically contiguous")
        bounds[int(episode)] = (start, end)
    return bounds


def sample_positions(
    total: int, bounds: dict[int, tuple[int, int]], max_horizon: int, uniform_samples: int
) -> list[int]:
    positions: set[int] = set(np.linspace(0, total - 1, min(total, uniform_samples), dtype=int).tolist())
    for start, end in bounds.values():
        positions.update({start, (start + end - 1) // 2, end - 1})
        for offset in {1, 2, 5, 9, 10, max_horizon - 1}:
            if offset >= 0:
                positions.add(max(start, end - 1 - offset))
    return sorted(position for position in positions if 0 <= position < total)


def validate_horizon(
    root: Path,
    repo_id: str,
    fps: int,
    horizon: int,
    positions: list[int],
    values: dict[str, np.ndarray],
    bounds: dict[int, tuple[int, int]],
    atol: float,
) -> dict[str, int | float]:
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={"action": [step / fps for step in range(horizon)]},
    )
    dataset._ensure_hf_dataset_loaded()
    required = ["action", "episode_index", "timestamp", "task_index", "frame_index", "index"]
    dataset.hf_dataset = dataset.hf_dataset.select_columns(required)
    dataset.hf_dataset.set_transform(hf_transform_to_torch)

    max_abs_error = 0.0
    checked_values = 0
    padded_values = 0
    for position in positions:
        episode = int(values["episode_index"][position])
        start, end = bounds[episode]
        query = np.minimum(np.arange(position, position + horizon), end - 1)
        expected = values["action"][query]
        expected_pad = np.arange(position, position + horizon) >= end

        item = dataset[position]
        actual = item["action"].detach().cpu().numpy()
        actual_pad = item["action_is_pad"].detach().cpu().numpy().astype(bool)
        error = float(np.max(np.abs(actual - expected)))
        max_abs_error = max(max_abs_error, error)
        checked_values += int(actual.size)
        padded_values += int(expected_pad.sum())
        if not np.allclose(actual, expected, rtol=0.0, atol=atol):
            mismatch = np.argwhere(np.abs(actual - expected) > atol)[0]
            raise AssertionError(
                f"horizon={horizon} position={position} episode={episode} chunk mismatch "
                f"at {mismatch.tolist()}: actual={actual[tuple(mismatch)]}, "
                f"expected={expected[tuple(mismatch)]}, max_abs={error}"
            )
        if not np.array_equal(actual_pad, expected_pad):
            raise AssertionError(
                f"horizon={horizon} position={position} episode={episode} padding mismatch"
            )
        if not np.allclose(actual[0], values["action"][position], rtol=0.0, atol=atol):
            raise AssertionError(f"horizon={horizon} position={position} chunk[0] != current action")
        if not (start <= position < end):
            raise AssertionError("Invalid episode boundary calculation")

    return {
        "horizon": horizon,
        "positions_checked": len(positions),
        "action_values_checked": checked_values,
        "padded_steps_checked": padded_values,
        "max_abs_error": max_abs_error,
    }


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    values = load_canonical_columns(root)
    bounds = episode_bounds(values["episode_index"])
    positions = sample_positions(
        len(values["index"]), bounds, max(args.horizons), args.uniform_samples
    )
    results = [
        validate_horizon(
            root,
            args.repo_id,
            int(info["fps"]),
            horizon,
            positions,
            values,
            bounds,
            args.atol,
        )
        for horizon in args.horizons
    ]
    summary = {
        "root": str(root),
        "repo_id": args.repo_id,
        "episodes": len(bounds),
        "frames": len(values["index"]),
        "physical_index_mismatch_rows": 0,
        "horizon_results": results,
        "status": "PASS",
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
