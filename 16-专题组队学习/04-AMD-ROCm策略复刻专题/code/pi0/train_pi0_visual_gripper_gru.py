#!/usr/bin/env python3
"""Train a strict-input recurrent gripper head on frozen Pi0 visual features."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_pi0_visual_contact_head import load_manifest


class GripperGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = F.gelu(self.norm(self.input(x)), approximate="tanh")
        hidden, _ = self.gru(encoded)
        return self.output(hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", nargs=3, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--direct-head", type=Path, required=True)
    parser.add_argument("--train-seeds", default="21-32")
    parser.add_argument("--val-seeds", default="33-36")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transition-weight", type=float, default=20.0)
    parser.add_argument("--release-weight", type=float, default=40.0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def make_manifest_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
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


def densify_sequence(
    features: np.ndarray,
    labels: np.ndarray,
    local_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat sparse samples so one recurrent update approximates one control frame."""
    dense_x: list[np.ndarray] = []
    dense_y: list[float] = []
    for index in range(len(features)):
        if index + 1 < len(features):
            repeat = max(int(local_frames[index + 1] - local_frames[index]), 1)
        else:
            repeat = 1
        repeat = min(repeat, 4)
        dense_x.extend([features[index]] * repeat)
        dense_y.extend([float(labels[index])] * repeat)
    return np.stack(dense_x).astype(np.float32), np.asarray(dense_y, dtype=np.float32)


def build_sequences(
    rows: list[dict[str, Any]],
    base_features: np.ndarray,
) -> list[dict[str, Any]]:
    previous_arm = np.stack(
        [np.asarray(row["prev_action"], dtype=np.float32)[:6] for row in rows]
    )
    features = np.concatenate([base_features, previous_arm], axis=1).astype(np.float32)
    labels = np.asarray(
        [float(np.asarray(row["action"], dtype=np.float32)[6] >= 0.5) for row in rows],
        dtype=np.float32,
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((int(row["source_index"]), int(row["episode"])), []).append(index)

    sequences: list[dict[str, Any]] = []
    for key, indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda index: int(rows[index]["local_frame"]))
        local_frames = np.asarray([rows[index]["local_frame"] for index in indices], dtype=np.int64)
        dense_x, dense_y = densify_sequence(features[indices], labels[indices], local_frames)
        row0 = rows[indices[0]]
        sequences.append(
            {
                "key": key,
                "seed": int(row0["seed"]),
                "split": str(row0["split"]),
                "x": dense_x,
                "y": dense_y,
            }
        )
    return sequences


def pad_sequences(
    sequences: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence["y"]) for sequence in sequences)
    input_dim = int(sequences[0]["x"].shape[1])
    x = np.zeros((len(sequences), max_length, input_dim), dtype=np.float32)
    y = np.zeros((len(sequences), max_length), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence["y"])
        x[index, :length] = (sequence["x"] - mean) / std
        y[index, :length] = sequence["y"]
        mask[index, :length] = True
    return (
        torch.as_tensor(x, device=device),
        torch.as_tensor(y, device=device),
        torch.as_tensor(mask, device=device),
    )


def event_masks(y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    previous = torch.cat([y[:, :1], y[:, :-1]], dim=1)
    valid_previous = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
    transition = mask & valid_previous & (y != previous)
    release = transition & (previous >= 0.5) & (y < 0.5)
    return transition, release


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    target = y >= 0.5
    pred = probability >= threshold
    positive = target
    negative = ~target
    tpr = float(pred[positive].mean()) if positive.any() else None
    tnr = float((~pred[negative]).mean()) if negative.any() else None
    return {
        "accuracy": float(np.mean(pred == target)),
        "balanced_accuracy": float((tpr + tnr) / 2.0) if tpr is not None and tnr is not None else None,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


@torch.inference_mode()
def evaluate(
    model: GripperGRU,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, Any], float]:
    probability = torch.sigmoid(model(x))
    transition, release = event_masks(y, mask)
    previous = torch.cat([y[:, :1], y[:, :-1]], dim=1)
    close = transition & (previous < 0.5) & (y >= 0.5)

    y_np = y[mask].detach().cpu().numpy()
    p_np = probability[mask].detach().cpu().numpy()
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.1, 0.9, 33):
        metrics = binary_metrics(y_np, p_np, float(threshold))
        release_acc = float((probability[release] < threshold).float().mean().item()) if release.any() else 0.0
        close_acc = float((probability[close] >= threshold).float().mean().item()) if close.any() else 0.0
        balanced = metrics["balanced_accuracy"] or 0.0
        score = 2.0 * release_acc + close_acc + balanced
        candidates.append(
            {
                "threshold": float(threshold),
                "score": score,
                "release_accuracy": release_acc,
                "close_accuracy": close_acc,
                **metrics,
            }
        )
    best = max(candidates, key=lambda row: (row["score"], row["balanced_accuracy"] or 0.0))
    report = {
        "samples": int(mask.sum().item()),
        "release_events": int(release.sum().item()),
        "close_events": int(close.sum().item()),
        "best_threshold": best,
        "threshold_sweep": candidates,
        "release_probability_mean": float(probability[release].mean().item()) if release.any() else None,
        "close_probability_mean": float(probability[close].mean().item()) if close.any() else None,
    }
    return report, float(best["score"])


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows, manifest = load_manifest(make_manifest_args(args))
    cache = np.load(args.feature_cache, allow_pickle=False)
    base = cache["features"].astype(np.float32)
    if len(rows) != len(base):
        raise ValueError(f"Manifest/cache mismatch: {len(rows)} != {len(base)}")
    sequences = build_sequences(rows, base)
    train_sequences = [sequence for sequence in sequences if sequence["split"] == "train"]
    val_sequences = [sequence for sequence in sequences if sequence["split"] == "val"]
    if not train_sequences or not val_sequences:
        raise ValueError("Both train and validation sequences are required")

    train_frames = np.concatenate([sequence["x"] for sequence in train_sequences], axis=0)
    feature_mean = train_frames.mean(axis=0).astype(np.float32)
    feature_std = train_frames.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x_train, y_train, mask_train = pad_sequences(
        train_sequences, feature_mean, feature_std, args.device
    )
    x_val, y_val, mask_val = pad_sequences(val_sequences, feature_mean, feature_std, args.device)

    model = GripperGRU(x_train.shape[-1], args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_transition, train_release = event_masks(y_train, mask_train)
    train_target = y_train[mask_train]
    positive_count = train_target.sum().clamp_min(1.0)
    negative_count = (1.0 - train_target).sum().clamp_min(1.0)
    positive_weight = 0.5 * train_target.numel() / positive_count
    negative_weight = 0.5 * train_target.numel() / negative_count

    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_report: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss_raw = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        sample_weight = torch.where(y_train >= 0.5, positive_weight, negative_weight)
        sample_weight = torch.where(
            train_transition,
            sample_weight * float(args.transition_weight),
            sample_weight,
        )
        sample_weight = torch.where(
            train_release,
            sample_weight * float(args.release_weight) / max(float(args.transition_weight), 1.0),
            sample_weight,
        )
        loss = (loss_raw[mask_train] * sample_weight[mask_train]).sum() / sample_weight[mask_train].sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_report, score = evaluate(model, x_val, y_val, mask_val)
            history.append({"epoch": epoch, "train_loss": float(loss.item()), "val": val_report})
            if score > best_score + 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_report = val_report
                stale = 0
            else:
                stale += 5
            print(
                f"epoch={epoch} loss={loss.item():.6f} score={score:.4f} "
                f"release={val_report['best_threshold']['release_accuracy']:.3f} "
                f"close={val_report['best_threshold']['close_accuracy']:.3f}",
                flush=True,
            )
            if stale >= args.patience:
                break

    if best_state is None or best_report is None:
        raise RuntimeError("No valid GRU checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    train_report, _ = evaluate(model, x_train, y_train, mask_train)
    val_report, _ = evaluate(model, x_val, y_val, mask_val)
    threshold = float(val_report["best_threshold"]["threshold"])

    direct = np.load(args.direct_head, allow_pickle=False)
    payload = {key: np.asarray(direct[key]) for key in direct.files}
    state = model.state_dict()
    payload.update(
        {
            "gripper_sequence_head_type": np.asarray("gru"),
            "gripper_input_mode": np.asarray("base_plus_prev_action6"),
            "gripper_feature_mean": feature_mean,
            "gripper_feature_std": feature_std,
            "gripper_input_weight": state["input.weight"].detach().cpu().numpy().astype(np.float32),
            "gripper_input_bias": state["input.bias"].detach().cpu().numpy().astype(np.float32),
            "gripper_norm_weight": state["norm.weight"].detach().cpu().numpy().astype(np.float32),
            "gripper_norm_bias": state["norm.bias"].detach().cpu().numpy().astype(np.float32),
            "gripper_gru_weight_ih": state["gru.weight_ih_l0"].detach().cpu().numpy().astype(np.float32),
            "gripper_gru_weight_hh": state["gru.weight_hh_l0"].detach().cpu().numpy().astype(np.float32),
            "gripper_gru_bias_ih": state["gru.bias_ih_l0"].detach().cpu().numpy().astype(np.float32),
            "gripper_gru_bias_hh": state["gru.bias_hh_l0"].detach().cpu().numpy().astype(np.float32),
            "gripper_output_weight": state["output.weight"].detach().cpu().numpy().astype(np.float32),
            "gripper_output_bias": state["output.bias"].detach().cpu().numpy().astype(np.float32),
            "gripper_threshold": np.asarray(threshold, dtype=np.float32),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)

    summary = {
        **manifest,
        "input_dim": int(x_train.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "input_contract": [
            "observation.image",
            "observation.wrist_image",
            "language instruction",
            "observation.state[:6]",
            "previous executed EEF action[:6]",
        ],
        "forbidden_inputs_used": [],
        "previous_gripper_used": False,
        "sequence_densification": "repeat selected feature until next local control frame, capped at 4",
        "transition_weight": float(args.transition_weight),
        "release_weight": float(args.release_weight),
        "best_epoch": int(best_epoch),
        "selected_threshold": threshold,
        "train": train_report,
        "val": val_report,
        "history": history,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
