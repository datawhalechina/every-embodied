#!/usr/bin/env python3
"""Train tiny transition heads for the target-relative Pi0 contact primitive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_smolvla_physical import PHASES, phase_index_from_state


FEATURE_MODES = {
    "count_step": [
        "local_step_norm",
        "phase_elapsed_norm",
        "pregrasp_count_norm",
        "descend_count_norm",
        "close_count_norm",
        "lift_count_norm",
    ],
    "xy_z_count_step": [
        "tcp_to_target_xy",
        "tcp_to_grasp_xy",
        "abs_tcp_to_pregrasp_z",
        "tcp_to_floor_z",
        "abs_tcp_to_floor_z",
        "abs_tcp_to_target_z",
        "local_step_norm",
        "phase_elapsed_norm",
        "pregrasp_count_norm",
        "descend_count_norm",
        "close_count_norm",
        "lift_count_norm",
    ],
    "grasp_geom_count_step": [
        "tcp_to_grasp_xy",
        "abs_tcp_to_pregrasp_z",
        "tcp_to_floor_z",
        "abs_tcp_to_floor_z",
        "local_step_norm",
        "phase_elapsed_norm",
        "pregrasp_count_norm",
        "descend_count_norm",
        "close_count_norm",
        "lift_count_norm",
    ],
}
TASK_PHASES = [
    ("pregrasp_to_descend", "move_pregrasp"),
    ("descend_to_close", "move_grasp"),
    ("close_to_lift", "close_gripper"),
    ("lift_to_hold", "lift_mug"),
]
TAIL_TASK_NAMES = ["close_to_lift", "lift_to_hold"]
ALL_TASK_NAMES = [name for name, _phase in TASK_PHASES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--step-scale", type=float, default=180.0)
    parser.add_argument("--count-scale", type=float, default=64.0)
    parser.add_argument("--grasp-offset-x", type=float, default=0.006)
    parser.add_argument("--grasp-offset-y", type=float, default=0.060)
    parser.add_argument("--target-z-offset", type=float, default=0.0)
    parser.add_argument("--pregrasp-z", type=float, default=0.135)
    parser.add_argument("--pregrasp-xy-tol", type=float, default=0.040)
    parser.add_argument("--pregrasp-z-tol", type=float, default=0.020)
    parser.add_argument("--descend-xy-tol", type=float, default=0.025)
    parser.add_argument("--descend-z-tol", type=float, default=0.012)
    parser.add_argument("--feature-mode", choices=sorted(FEATURE_MODES), default="xy_z_count_step")
    parser.add_argument(
        "--early-label-mode",
        choices=["phase_tail", "geometry", "pregrasp_geometry"],
        default="phase_tail",
        help=(
            "How to label early transition positives. pregrasp_geometry uses a "
            "geometry label for pregrasp_to_descend and phase-tail labels for "
            "descend_to_close."
        ),
    )
    parser.add_argument(
        "--task-set",
        choices=["tail", "all"],
        default="tail",
        help="tail trains close/lift transitions only; all also trains pregrasp/descend transitions.",
    )
    parser.add_argument(
        "--positive-lookahead",
        type=int,
        default=8,
        help="Mark the final N frames of each selected phase as ready for the next phase.",
    )
    return parser.parse_args()


def tensor_to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def feature_map(
    state: np.ndarray,
    local_step: int,
    phase_elapsed: int,
    task_name: str,
    step_scale: float,
    count_scale: float,
    grasp_offset_x: float,
    grasp_offset_y: float,
    target_z_offset: float,
    pregrasp_z: float,
) -> dict[str, float]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 22:
        raise ValueError(f"Expected 22-D target-relative state, got {state.shape[0]}")
    rel = state[19:22].astype(np.float32)
    xy = float(np.linalg.norm(rel[:2]))
    grasp_xy = float(
        np.linalg.norm(
            rel[:2]
            - np.asarray([float(grasp_offset_x), float(grasp_offset_y)], dtype=np.float32)
        )
    )
    pregrasp_z_err = float(rel[2]) - float(pregrasp_z)
    floor_z = float(rel[2]) - float(target_z_offset)
    pregrasp_count = int(phase_elapsed) if task_name == "pregrasp_to_descend" else 0
    descend_count = int(phase_elapsed) if task_name == "descend_to_close" else 0
    close_count = int(phase_elapsed) if task_name == "close_to_lift" else 0
    lift_count = int(phase_elapsed) if task_name == "lift_to_hold" else 0
    return {
        "tcp_to_target_x": float(rel[0]),
        "tcp_to_target_y": float(rel[1]),
        "tcp_to_target_z": float(rel[2]),
        "tcp_to_target_xy": xy,
        "tcp_to_grasp_xy": grasp_xy,
        "tcp_to_pregrasp_z": pregrasp_z_err,
        "abs_tcp_to_pregrasp_z": abs(pregrasp_z_err),
        "tcp_to_floor_z": floor_z,
        "abs_tcp_to_floor_z": abs(floor_z),
        "abs_tcp_to_target_z": abs(float(rel[2])),
        "local_step_norm": float(local_step) / max(float(step_scale), 1.0),
        "phase_elapsed_norm": float(phase_elapsed) / max(float(count_scale), 1.0),
        "pregrasp_count_norm": float(pregrasp_count) / max(float(count_scale), 1.0),
        "descend_count_norm": float(descend_count) / max(float(count_scale), 1.0),
        "close_count_norm": float(close_count) / max(float(count_scale), 1.0),
        "lift_count_norm": float(lift_count) / max(float(count_scale), 1.0),
    }


def features(
    state: np.ndarray,
    local_step: int,
    phase_elapsed: int,
    task_name: str,
    step_scale: float,
    count_scale: float,
    grasp_offset_x: float,
    grasp_offset_y: float,
    target_z_offset: float,
    pregrasp_z: float,
    feature_names: list[str],
) -> np.ndarray:
    values = feature_map(
        state,
        local_step,
        phase_elapsed,
        task_name,
        step_scale,
        count_scale,
        grasp_offset_x,
        grasp_offset_y,
        target_z_offset,
        pregrasp_z,
    )
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def selected_tasks(task_set: str) -> list[str]:
    return list(ALL_TASK_NAMES if str(task_set) == "all" else TAIL_TASK_NAMES)


def transition_label(
    state: np.ndarray,
    phase_elapsed: int,
    phase_len: int,
    task_name: str,
    args: argparse.Namespace,
) -> float:
    early_label_mode = str(args.early_label_mode)
    if (
        (early_label_mode == "geometry" and task_name in {"pregrasp_to_descend", "descend_to_close"})
        or (early_label_mode == "pregrasp_geometry" and task_name == "pregrasp_to_descend")
    ):
        rel = np.asarray(state, dtype=np.float32).reshape(-1)[19:22]
        grasp_xy = float(
            np.linalg.norm(
                rel[:2]
                - np.asarray(
                    [float(args.grasp_offset_x), float(args.grasp_offset_y)],
                    dtype=np.float32,
                )
            )
        )
        if task_name == "pregrasp_to_descend":
            return float(
                grasp_xy <= float(args.pregrasp_xy_tol)
                and abs(float(rel[2]) - float(args.pregrasp_z)) <= float(args.pregrasp_z_tol)
            )
        return float(
            grasp_xy <= float(args.descend_xy_tol)
            and float(rel[2]) <= float(args.target_z_offset) + float(args.descend_z_tol)
        )
    return float(phase_elapsed > max(int(phase_len) - int(args.positive_lookahead), 0))


def load_examples(
    args: argparse.Namespace,
    feature_names: list[str],
    task_names: list[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict]]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    episodes: dict[int, list[np.ndarray]] = {}
    for idx in range(len(dataset.hf_dataset)):
        item = dataset.hf_dataset[idx]
        episode_index = int(tensor_to_np(item["episode_index"]).reshape(-1)[0])
        state = tensor_to_np(item["observation.state"]).astype(np.float32).reshape(-1)
        episodes.setdefault(episode_index, []).append(state)

    by_task: dict[str, list[tuple[np.ndarray, float]]] = {name: [] for name in task_names}
    reports: list[dict] = []
    task_to_phase_index = {
        task_name: int(PHASES.index(phase_name))
        for task_name, phase_name in TASK_PHASES
        if task_name in task_names
    }

    for episode_index, states_list in sorted(episodes.items()):
        states = np.stack(states_list).astype(np.float32)
        phases = np.asarray([phase_index_from_state(row) for row in states], dtype=np.int64)
        report = {
            "episode_index": int(episode_index),
            "phase_counts": {
                PHASES[int(phase)]: int(np.sum(phases == phase))
                for phase in sorted(set(int(p) for p in phases.tolist()))
            },
        }
        for task_name in task_names:
            phase_index = int(task_to_phase_index[task_name])
            positions = np.where(phases == phase_index)[0]
            if len(positions) == 0:
                report[f"{task_name}_used"] = False
                continue
            phase_start = int(positions[0])
            phase_len = int(len(positions))
            positives = 0
            for phase_elapsed, pos in enumerate(positions.tolist(), start=1):
                label = transition_label(states[int(pos)], phase_elapsed, phase_len, task_name, args)
                x = features(
                    states[int(pos)],
                    int(pos) - phase_start,
                    int(phase_elapsed),
                    task_name,
                    float(args.step_scale),
                    float(args.count_scale),
                    float(args.grasp_offset_x),
                    float(args.grasp_offset_y),
                    float(args.target_z_offset),
                    float(args.pregrasp_z),
                    feature_names,
                )
                by_task[task_name].append((x, label))
                positives += int(label)
            report[f"{task_name}_used"] = True
            report[f"{task_name}_frames"] = phase_len
            report[f"{task_name}_positive_labels"] = positives
        reports.append(report)

    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task_name, rows in by_task.items():
        if not rows:
            raise RuntimeError(f"No examples collected for {task_name}")
        x_rows, y_rows = zip(*rows, strict=True)
        packed[task_name] = (
            np.stack(x_rows).astype(np.float32),
            np.asarray(y_rows, dtype=np.float32),
        )
    return packed, reports


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, list[dict]]:
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
            history.append({"epoch": int(epoch), "loss": loss, "accuracy": float(np.mean(pred == y))})
    return w.astype(np.float32), float(b), history


def main() -> int:
    args = parse_args()
    feature_names = list(FEATURE_MODES[str(args.feature_mode)])
    task_names = selected_tasks(str(args.task_set))
    examples, reports = load_examples(args, feature_names, task_names)
    all_x = np.concatenate([examples[name][0] for name in task_names], axis=0)
    mean = all_x.mean(axis=0).astype(np.float32)
    std = all_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    weights: list[np.ndarray] = []
    biases: list[float] = []
    task_metrics: dict[str, dict] = {}
    for task_name in task_names:
        x, y = examples[task_name]
        w, b, history = train_logistic(x, y, mean, std, args.epochs, args.lr, args.l2)
        probs = sigmoid(((x - mean) / std) @ w + b)
        pred = (probs >= 0.5).astype(np.float32)
        weights.append(w)
        biases.append(float(b))
        task_metrics[task_name] = {
            "frames": int(len(y)),
            "positive_labels": int(y.sum()),
            "negative_labels": int(len(y) - y.sum()),
            "train_accuracy": float(np.mean(pred == y)),
            "history": history,
            "weights": {name: float(value) for name, value in zip(feature_names, w.tolist())},
            "bias": float(b),
        }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        state_mean=mean.astype(np.float32),
        state_std=std.astype(np.float32),
        weight=np.stack(weights).astype(np.float32),
        bias=np.asarray(biases, dtype=np.float32),
        threshold=np.asarray(0.5, dtype=np.float32),
        step_scale=np.asarray(float(args.step_scale), dtype=np.float32),
        count_scale=np.asarray(float(args.count_scale), dtype=np.float32),
        feature_names=np.asarray(feature_names),
        task_names=np.asarray(task_names),
    )
    summary = {
        "output_npz": str(args.output_npz),
        "dataset_repo_id": str(args.dataset_repo_id),
        "dataset_root": str(args.dataset_root),
        "feature_mode": str(args.feature_mode),
        "feature_names": feature_names,
        "task_set": str(args.task_set),
        "task_names": task_names,
        "early_label_mode": str(args.early_label_mode),
        "grasp_offset_x": float(args.grasp_offset_x),
        "grasp_offset_y": float(args.grasp_offset_y),
        "target_z_offset": float(args.target_z_offset),
        "pregrasp_z": float(args.pregrasp_z),
        "pregrasp_xy_tol": float(args.pregrasp_xy_tol),
        "pregrasp_z_tol": float(args.pregrasp_z_tol),
        "descend_xy_tol": float(args.descend_xy_tol),
        "descend_z_tol": float(args.descend_z_tol),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "l2": float(args.l2),
        "step_scale": float(args.step_scale),
        "count_scale": float(args.count_scale),
        "positive_lookahead": int(args.positive_lookahead),
        "tasks": task_metrics,
        "episode_reports": reports,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
