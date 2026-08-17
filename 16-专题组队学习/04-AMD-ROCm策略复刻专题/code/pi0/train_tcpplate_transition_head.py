#!/usr/bin/env python3
"""Train a tiny transition head for the 22-D tcp-to-plate finisher state.

The head predicts when a held ``move_preplace`` schedule should be released into
the ``lower_to_plate`` tail.  It intentionally avoids phase one-hot features so
it cannot simply read the answer from the dataset schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


PHASES = [
    "reset",
    "move_pregrasp",
    "move_grasp",
    "close_gripper",
    "lift_mug",
    "move_preplace",
    "lower_to_plate",
    "pre_release_hold",
    "open_gripper",
    "retreat",
    "done",
]
PHASE_TO_INDEX = {name: index for index, name in enumerate(PHASES)}
FULL_FEATURE_NAMES = [
    "tcp_to_plate_x",
    "tcp_to_plate_y",
    "tcp_to_plate_z",
    "tcp_to_plate_xy",
    "abs_tcp_to_plate_z",
    "local_step_norm",
]
FEATURE_MODES = {
    "full": FULL_FEATURE_NAMES,
    "xy_z_step": ["tcp_to_plate_xy", "abs_tcp_to_plate_z", "local_step_norm"],
    "xy_step": ["tcp_to_plate_xy", "local_step_norm"],
}


def tensor_to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def phase_index_from_state(state: np.ndarray) -> int:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] >= 8 + len(PHASES):
        onehot = state[8 : 8 + len(PHASES)]
        if onehot.shape[0] == len(PHASES) and float(np.max(onehot)) >= 0.5:
            return int(np.argmax(onehot))
    return int(np.clip(round(float(state[7]) * max(len(PHASES) - 1, 1)), 0, len(PHASES) - 1))


def transition_feature_map(state: np.ndarray, local_step: int, step_scale: float) -> dict[str, float]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 22:
        raise ValueError(f"Expected 22-D tcpplate state, got {state.shape[0]}")
    rel = state[19:22].astype(np.float32)
    xy = float(np.linalg.norm(rel[:2]))
    return {
        "tcp_to_plate_x": float(rel[0]),
        "tcp_to_plate_y": float(rel[1]),
        "tcp_to_plate_z": float(rel[2]),
        "tcp_to_plate_xy": xy,
        "abs_tcp_to_plate_z": abs(float(rel[2])),
        "local_step_norm": float(local_step) / max(float(step_scale), 1.0),
    }


def transition_features(
    state: np.ndarray,
    local_step: int,
    step_scale: float,
    feature_names: list[str],
) -> np.ndarray:
    values = transition_feature_map(state, local_step, step_scale)
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def load_examples(
    repo_id: str,
    root: Path,
    step_scale: float,
    positive_lookahead: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    dataset = LeRobotDataset(repo_id, root=root)
    episodes: dict[int, list[np.ndarray]] = {}
    for idx in range(len(dataset.hf_dataset)):
        item = dataset.hf_dataset[idx]
        episode_index = int(tensor_to_np(item["episode_index"]).reshape(-1)[0])
        state = tensor_to_np(item["observation.state"]).astype(np.float32).reshape(-1)
        episodes.setdefault(episode_index, []).append(state)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    reports: list[dict] = []
    move_index = PHASE_TO_INDEX["move_preplace"]
    lower_index = PHASE_TO_INDEX["lower_to_plate"]
    open_index = PHASE_TO_INDEX["open_gripper"]

    for episode_index, states_list in sorted(episodes.items()):
        states = np.stack(states_list).astype(np.float32)
        phases = np.asarray([phase_index_from_state(row) for row in states], dtype=np.int64)
        move_positions = np.where(phases == move_index)[0]
        lower_positions = np.where(phases >= lower_index)[0]
        if len(move_positions) == 0 or len(lower_positions) == 0:
            reports.append(
                {
                    "episode_index": int(episode_index),
                    "used": False,
                    "reason": "missing move_preplace or lower tail",
                }
            )
            continue

        move_start = int(move_positions[0])
        lower_start = int(lower_positions[0])
        usable_positions = np.where((phases == move_index) | ((phases >= lower_index) & (phases <= open_index)))[0]
        used = 0
        positive = 0
        for pos in usable_positions:
            if int(pos) < move_start:
                continue
            local_step = int(pos) - move_start
            label = float(int(pos) >= lower_start - int(positive_lookahead))
            x_rows.append(transition_features(states[int(pos)], local_step, step_scale, feature_names))
            y_rows.append(label)
            used += 1
            positive += int(label)

        reports.append(
            {
                "episode_index": int(episode_index),
                "used": True,
                "move_preplace_frames": int(len(move_positions)),
                "move_start": move_start,
                "lower_start": lower_start,
                "examples": used,
                "positive_labels": positive,
                "phase_counts": {
                    PHASES[int(phase)]: int(np.sum(phases == phase))
                    for phase in sorted(set(int(p) for p in phases.tolist()))
                },
            }
        )

    if not x_rows:
        raise RuntimeError("No transition-head examples were collected")
    return np.stack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.float32), reports


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[dict]]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    xn = ((x - mean) / std).astype(np.float32)
    w = np.zeros(xn.shape[1], dtype=np.float32)
    prior = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    b = float(np.log(prior / (1.0 - prior)))
    history: list[dict] = []
    n = float(len(y))
    for epoch in range(1, int(epochs) + 1):
        logits = xn @ w + b
        p = sigmoid(logits)
        grad = (p - y).astype(np.float32)
        grad_w = (xn.T @ grad) / n + float(l2) * w
        grad_b = float(grad.mean())
        w -= float(lr) * grad_w.astype(np.float32)
        b -= float(lr) * grad_b
        if epoch == 1 or epoch % max(int(epochs) // 10, 1) == 0 or epoch == int(epochs):
            eps = 1e-7
            loss = -float(np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            pred = (p >= 0.5).astype(np.float32)
            history.append({"epoch": epoch, "loss": loss, "accuracy": float(np.mean(pred == y))})
    return w.astype(np.float32), float(b), mean, std, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--step-scale", type=float, default=180.0)
    parser.add_argument("--feature-mode", choices=sorted(FEATURE_MODES), default="full")
    parser.add_argument(
        "--positive-lookahead",
        type=int,
        default=0,
        help="Mark the final N move_preplace frames as ready too. Default keeps labels strict.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_names = list(FEATURE_MODES[str(args.feature_mode)])
    x, y, episode_reports = load_examples(
        args.dataset_repo_id,
        args.dataset_root,
        args.step_scale,
        args.positive_lookahead,
        feature_names,
    )
    w, b, mean, std, history = train_logistic(x, y, args.epochs, args.lr, args.l2)
    probs = sigmoid(((x - mean) / std) @ w + b)
    pred = (probs >= 0.5).astype(np.float32)
    summary = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "frames": int(len(y)),
        "feature_mode": str(args.feature_mode),
        "feature_names": feature_names,
        "positive_labels": int(y.sum()),
        "negative_labels": int(len(y) - y.sum()),
        "train_accuracy": float(np.mean(pred == y)),
        "positive_lookahead": int(args.positive_lookahead),
        "step_scale": float(args.step_scale),
        "history": history,
        "weights": {name: float(value) for name, value in zip(feature_names, w.tolist())},
        "bias": float(b),
        "episode_reports": episode_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        state_mean=mean.astype(np.float32),
        state_std=std.astype(np.float32),
        weight=w.astype(np.float32),
        bias=np.asarray(b, dtype=np.float32),
        threshold=np.asarray(0.5, dtype=np.float32),
        step_scale=np.asarray(float(args.step_scale), dtype=np.float32),
        feature_names=np.asarray(feature_names),
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
