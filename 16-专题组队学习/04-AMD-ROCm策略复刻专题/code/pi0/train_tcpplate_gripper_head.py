#!/usr/bin/env python3
"""Train a tiny logistic gripper head for the 22-D tcp-to-plate finisher state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def phase_index_from_state(state: np.ndarray) -> int:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] >= 8 + len(PHASES):
        onehot = state[8 : 8 + len(PHASES)]
        if onehot.shape[0] == len(PHASES) and float(np.max(onehot)) >= 0.5:
            return int(np.argmax(onehot))
    return int(np.clip(round(float(state[7]) * max(len(PHASES) - 1, 1)), 0, len(PHASES) - 1))


def load_arrays(repo_id: str, root: Path, label_source: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    dataset = LeRobotDataset(repo_id, root=root)
    states: list[np.ndarray] = []
    labels: list[float] = []
    phases: list[int] = []
    for idx in range(len(dataset.hf_dataset)):
        state = np.asarray(dataset.hf_dataset[idx]["observation.state"], dtype=np.float32).reshape(-1)
        action = np.asarray(dataset.hf_dataset[idx]["action"], dtype=np.float32).reshape(-1)
        phase_index = phase_index_from_state(state)
        if label_source == "action":
            label = float(action[6] > 0.5)
        elif label_source == "phase":
            label = float(PHASES[phase_index] in {"close_gripper", "lift_mug", "move_preplace", "lower_to_plate"})
        else:
            raise ValueError(f"Unknown label source {label_source!r}")
        states.append(state)
        labels.append(label)
        phases.append(phase_index)
    return np.stack(states).astype(np.float32), np.asarray(labels, dtype=np.float32), phases


def feature_indices_for_mode(state_dim: int, feature_mode: str) -> np.ndarray:
    if feature_mode == "full":
        return np.arange(state_dim, dtype=np.int64)
    if feature_mode == "phase":
        # timestamp + phase_index_norm + phase_onehot11
        return np.arange(6, 19, dtype=np.int64)
    raise ValueError(f"Unknown feature mode {feature_mode!r}")


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
            acc = float(np.mean(pred == y))
            history.append({"epoch": epoch, "loss": loss, "accuracy": acc})
    return w.astype(np.float32), float(b), mean, std, history


def phase_report(
    x: np.ndarray,
    features: np.ndarray,
    y: np.ndarray,
    phases: list[int],
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
):
    probs = sigmoid(((x - mean) / std) @ w + b)
    pred = (probs >= 0.5).astype(np.float32)
    rows = []
    for phase_index in sorted(set(phases)):
        mask = np.asarray(phases) == phase_index
        rows.append(
            {
                "phase_index": int(phase_index),
                "phase": PHASES[int(phase_index)],
                "count": int(mask.sum()),
                "positive_labels": int(y[mask].sum()),
                "predicted_closed": int(pred[mask].sum()),
                "accuracy": float(np.mean(pred[mask] == y[mask])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--label-source", choices=["action", "phase"], default="action")
    parser.add_argument("--feature-mode", choices=["full", "phase"], default="full")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    states, y, phases = load_arrays(args.dataset_repo_id, args.dataset_root, args.label_source)
    feature_indices = feature_indices_for_mode(states.shape[1], args.feature_mode)
    x = states[:, feature_indices]
    w, b, mean, std, history = train_logistic(x, y, args.epochs, args.lr, args.l2)
    probs = sigmoid(((x - mean) / std) @ w + b)
    pred = (probs >= 0.5).astype(np.float32)
    summary = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "label_source": args.label_source,
        "frames": int(len(y)),
        "state_dim": int(states.shape[1]),
        "feature_mode": args.feature_mode,
        "feature_dim": int(x.shape[1]),
        "feature_indices": [int(i) for i in feature_indices],
        "positive_labels": int(y.sum()),
        "negative_labels": int(len(y) - y.sum()),
        "train_accuracy": float(np.mean(pred == y)),
        "history": history,
        "phase_report": phase_report(x, states, y, phases, w, b, mean, std),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        state_mean=mean.astype(np.float32),
        state_std=std.astype(np.float32),
        weight=w.astype(np.float32),
        bias=np.asarray(b, dtype=np.float32),
        threshold=np.asarray(0.5, dtype=np.float32),
        feature_indices=feature_indices.astype(np.int64),
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
