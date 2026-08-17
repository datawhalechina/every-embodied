#!/usr/bin/env python3
"""Audit whether a visual action head shortcuts through previous gripper state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_pi0_visual_contact_head import load_manifest, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", nargs=3, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--train-seeds", default="21-32")
    parser.add_argument("--val-seeds", default="33-36")
    parser.add_argument("--stationary-action-threshold", type=float, default=0.0005)
    parser.add_argument("--stationary-keep", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_args = SimpleNamespace(
        source=args.source,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        first_frames=16,
        transition_pre=12,
        transition_post=1000,
        transition_stride=2,
        bridge_stride=2,
        close_threshold=0.5,
    )
    rows, _ = load_manifest(manifest_args)
    cache = np.load(args.feature_cache, allow_pickle=False)
    base = cache["features"].astype(np.float32)
    if len(base) != len(rows):
        raise ValueError(f"Manifest/cache mismatch: {len(rows)} != {len(base)}")

    previous = np.stack([np.asarray(row["prev_action"], dtype=np.float32)[:7] for row in rows])
    action = np.stack([np.asarray(row["action"], dtype=np.float32)[:7] for row in rows])
    split = np.asarray([row["split"] for row in rows])
    x = np.concatenate([base, previous], axis=1)
    y = np.concatenate([action[:, :3], action[:, 6:7]], axis=1)

    keep = np.zeros(len(rows), dtype=bool)
    current_key: tuple[int, int] | None = None
    stationary_run = 0
    for index, row in enumerate(rows):
        key = (int(row["source_index"]), int(row["episode"]))
        if key != current_key:
            current_key = key
            stationary_run = 0
        if np.linalg.norm(action[index] - previous[index]) < args.stationary_action_threshold:
            stationary_run += 1
            keep[index] = stationary_run <= max(args.stationary_keep, 1)
        else:
            stationary_run = 0
            keep[index] = True
    x, y, action, previous, split = x[keep], y[keep], action[keep], previous[keep], split[keep]

    head = np.load(args.head, allow_pickle=False)
    normalized = (x - head["feature_mean"]) / head["feature_std"]
    pred = normalized @ head["weight"] + head["bias"]
    pred[:, 3] = np.clip(pred[:, 3], 0.0, 1.0)

    ablated = normalized.copy()
    ablated[:, -1] = 0.0
    pred_ablated = ablated @ head["weight"] + head["bias"]
    pred_ablated[:, 3] = np.clip(pred_ablated[:, 3], 0.0, 1.0)

    val = split == "val"
    release = (previous[:, 6] >= 0.5) & (action[:, 6] < 0.5)
    close = (previous[:, 6] < 0.5) & (action[:, 6] >= 0.5)

    def transition_report(mask: np.ndarray, values: np.ndarray) -> dict[str, object]:
        chosen = mask & val
        return {
            "count": int(chosen.sum()),
            "pred_mean": float(values[chosen, 3].mean()) if chosen.any() else None,
            "binary_accuracy": float(np.mean((values[chosen, 3] >= 0.5) == (y[chosen, 3] >= 0.5)))
            if chosen.any()
            else None,
        }

    report = {
        "samples": int(len(x)),
        "val": regression_metrics(y[val], pred[val]),
        "val_without_previous_gripper": regression_metrics(y[val], pred_ablated[val]),
        "val_release_transition": transition_report(release, pred),
        "val_release_transition_without_previous_gripper": transition_report(release, pred_ablated),
        "val_close_transition": transition_report(close, pred),
        "val_close_transition_without_previous_gripper": transition_report(close, pred_ablated),
        "normalized_previous_gripper_weight": float(head["weight"][-1, 3]),
        "normalized_previous_eef_weight_l2": float(np.linalg.norm(head["weight"][-7:-1, 3])),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
