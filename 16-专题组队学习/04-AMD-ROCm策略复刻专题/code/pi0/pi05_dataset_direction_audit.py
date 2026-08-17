#!/usr/bin/env python3
"""Audit phase-dependent direction statistics in a LeRobot EEF dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-object", choices=["red", "blue"], default="blue")
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--early-frames", type=int, default=80)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def cosine_alignment(vectors: np.ndarray, goals: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(goals, axis=1)
    valid = denom > 1e-9
    result = np.full(vectors.shape[0], np.nan, dtype=np.float64)
    result[valid] = np.sum(vectors[valid] * goals[valid], axis=1) / denom[valid]
    return result


def load_episodes(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes under {root / 'data'}")
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "obj_init"])
        for row in table.to_pylist():
            grouped_rows.setdefault(int(row["episode_index"]), []).append(row)
    episodes: list[dict[str, Any]] = []
    for episode_index, rows in grouped_rows.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        episodes.append(
            {
                "episode": episode_index,
                "action": np.asarray([row["action"] for row in rows], dtype=np.float64),
                "obj_init": np.asarray(rows[0]["obj_init"], dtype=np.float64),
            }
        )
    episodes.sort(key=lambda item: item["episode"])
    return episodes


def load_final_probe(log_path: Path | None) -> dict[str, Any] | None:
    if log_path is None:
        return None
    final_eval = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "final_eval":
                final_eval = row
    if final_eval is None:
        return None
    probes = final_eval.get("first_action_probes", [])
    if not probes:
        return None
    target_y = np.asarray([row["target"][1] for row in probes], dtype=np.float64)
    predicted_y = np.asarray([row["predicted"][1] for row in probes], dtype=np.float64)
    return {
        "count": len(probes),
        "label_target_y": quantiles(target_y),
        "predicted_y": quantiles(predicted_y),
        "mean_y_bias": float(np.mean(predicted_y - target_y)),
        "predicted_negative_fraction": float(np.mean(predicted_y < 0.0)),
    }


def main() -> int:
    args = parse_args()
    episodes = load_episodes(args.dataset_root)
    target_slice = slice(0, 3) if args.target_object == "red" else slice(3, 6)
    all_actions = []
    early_abs_y = []
    late_abs_y = []
    early_delta = []
    late_delta = []
    early_goal = []
    late_goal = []
    start_rows = []
    progress_buckets: list[list[np.ndarray]] = [[] for _ in range(5)]

    for item in episodes:
        action = item["action"]
        target = item["obj_init"][target_slice]
        plate = item["obj_init"][6:9]
        n = action.shape[0]
        if n < 2:
            continue
        delta = np.diff(action[:, :3], axis=0)
        early_n = min(max(int(args.early_frames), 2), n)
        early_delta_n = min(early_n - 1, delta.shape[0])
        all_actions.append(action)
        early_abs_y.append(action[:early_n, 1])
        late_abs_y.append(action[early_n:, 1])
        early_delta.append(delta[:early_delta_n])
        late_delta.append(delta[early_delta_n:])
        early_goal.append(np.repeat((target - action[0, :3])[None, :], early_delta_n, axis=0))
        late_count = max(delta.shape[0] - early_delta_n, 0)
        late_goal.append(np.repeat((plate - action[early_n - 1, :3])[None, :], late_count, axis=0))
        start_rows.append(
            {
                "episode": item["episode"],
                "frames": n,
                "target_y": float(target[1]),
                "plate_y": float(plate[1]),
                "first_action_y": float(action[0, 1]),
                "early80_mean_action_y": float(np.mean(action[:early_n, 1])),
                "early80_end_action_y": float(action[early_n - 1, 1]),
                "early80_net_delta_y": float(action[early_n - 1, 1] - action[0, 1]),
            }
        )
        progress = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        bucket_index = np.minimum((progress * 5).astype(np.int64), 4)
        for bucket in range(5):
            progress_buckets[bucket].append(action[bucket_index == bucket])

    action_all = np.concatenate(all_actions, axis=0)
    early_delta_all = np.concatenate([x for x in early_delta if x.size], axis=0)
    late_delta_all = np.concatenate([x for x in late_delta if x.size], axis=0)
    early_goal_all = np.concatenate([x for x in early_goal if x.size], axis=0)
    late_goal_all = np.concatenate([x for x in late_goal if x.size], axis=0)
    start_net_y = np.asarray([row["early80_net_delta_y"] for row in start_rows])
    target_minus_start_y = np.asarray([row["target_y"] - row["first_action_y"] for row in start_rows])
    desired_start_sign = np.sign(target_minus_start_y)

    report: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "episodes": len(episodes),
        "frames": int(action_all.shape[0]),
        "target_object": args.target_object,
        "early_frames": int(args.early_frames),
        "absolute_action_y_all": quantiles(action_all[:, 1]),
        "absolute_action_y_early": quantiles(np.concatenate(early_abs_y)),
        "absolute_action_y_late": quantiles(np.concatenate([x for x in late_abs_y if x.size])),
        "delta_action_y_early": quantiles(early_delta_all[:, 1]),
        "delta_action_y_late": quantiles(late_delta_all[:, 1]),
        "early_delta_goal_cosine": quantiles(cosine_alignment(early_delta_all, early_goal_all)),
        "late_delta_goal_cosine": quantiles(cosine_alignment(late_delta_all, late_goal_all)),
        "early_net_y_toward_target_fraction": float(
            np.mean(np.sign(start_net_y) == desired_start_sign)
        ),
        "target_y_positive_fraction": float(np.mean(np.asarray([r["target_y"] for r in start_rows]) > 0.0)),
        "plate_y_negative_fraction": float(np.mean(np.asarray([r["plate_y"] for r in start_rows]) < 0.0)),
        "progress_action_y": {
            f"{bucket * 20}-{(bucket + 1) * 20}%": quantiles(np.concatenate(rows, axis=0)[:, 1])
            for bucket, rows in enumerate(progress_buckets)
            if rows
        },
        "episode_start_rows": start_rows,
        "model_final_probe": load_final_probe(args.train_log),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
