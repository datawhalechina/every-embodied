#!/usr/bin/env python3
"""Train a nonlinear strict-input action head from cached Pi0 visual features."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VisualActionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden1, hidden2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden2, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"features", "labels", "split", "seed", "projection"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Feature cache {path} is missing fields: {missing}")
    return {key: np.asarray(data[key]) for key in required}


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = pred - y
    target_close = y[:, 3] >= 0.5
    pred_close = pred[:, 3] >= 0.5
    positives = int(target_close.sum())
    negatives = int((~target_close).sum())
    tpr = float(pred_close[target_close].mean()) if positives else None
    tnr = float((~pred_close[~target_close]).mean()) if negatives else None
    balanced = float((tpr + tnr) / 2.0) if tpr is not None and tnr is not None else None
    return {
        "samples": int(len(y)),
        "xyz_rmse_m": [float(v) for v in np.sqrt(np.mean(error[:, :3] ** 2, axis=0))],
        "xyz_mae_m": [float(v) for v in np.mean(np.abs(error[:, :3]), axis=0)],
        "gripper_mae": float(np.mean(np.abs(error[:, 3]))),
        "gripper_accuracy": float(np.mean(pred_close == target_close)),
        "gripper_balanced_accuracy": balanced,
        "gripper_positive_count": positives,
        "gripper_negative_count": negatives,
        "gripper_true_positive_rate": tpr,
        "gripper_true_negative_rate": tnr,
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    x: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        raw = model(torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device))
        xyz = raw[:, :3].cpu().numpy() * target_std + target_mean
        gripper = torch.sigmoid(raw[:, 3]).cpu().numpy()[:, None]
        rows.append(np.concatenate([xyz, gripper], axis=1).astype(np.float32))
    return np.concatenate(rows, axis=0)


def nearest_rms_distance(queries: np.ndarray, prototypes: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        squared = np.mean((query[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(distances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-cache", type=Path, required=True)
    parser.add_argument("--extra-cache", type=Path, default=None)
    parser.add_argument("--extra-train-repeat", type=int, default=4)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gripper-loss-weight", type=float, default=0.5)
    parser.add_argument("--prototype-stride", type=int, default=4)
    parser.add_argument("--distance-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    primary = load_cache(args.primary_cache)
    extra = load_cache(args.extra_cache) if args.extra_cache is not None else None
    if extra is not None and (
        primary["projection"].shape != extra["projection"].shape
        or not np.array_equal(primary["projection"], extra["projection"])
    ):
        raise ValueError("Primary and extra feature caches use different projections")

    p_train = primary["split"].astype(str) == "train"
    p_val = primary["split"].astype(str) == "val"
    repeat = max(int(args.extra_train_repeat), 1) if extra is not None else 0
    train_parts_x = [primary["features"][p_train]]
    train_parts_y = [primary["labels"][p_train]]
    val_parts_x = [primary["features"][p_val]]
    val_parts_y = [primary["labels"][p_val]]
    if extra is not None:
        e_train = extra["split"].astype(str) == "train"
        e_val = extra["split"].astype(str) == "val"
        train_parts_x.extend([extra["features"][e_train]] * repeat)
        train_parts_y.extend([extra["labels"][e_train]] * repeat)
        val_parts_x.append(extra["features"][e_val])
        val_parts_y.append(extra["labels"][e_val])
    else:
        e_train = np.zeros(0, dtype=bool)
        e_val = np.zeros(0, dtype=bool)
    train_x = np.concatenate(train_parts_x).astype(np.float32)
    train_y = np.concatenate(train_parts_y).astype(np.float32)
    val_x = np.concatenate(val_parts_x).astype(np.float32)
    val_y = np.concatenate(val_parts_y).astype(np.float32)

    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    train_xn = ((train_x - feature_mean) / feature_std).astype(np.float32)
    val_xn = ((val_x - feature_mean) / feature_std).astype(np.float32)
    target_mean = train_y[:, :3].mean(axis=0).astype(np.float32)
    target_std = train_y[:, :3].std(axis=0).astype(np.float32)
    target_std[target_std < 1e-6] = 1.0
    train_target = train_y.copy()
    train_target[:, :3] = (train_target[:, :3] - target_mean) / target_std

    close_count = float(np.sum(train_y[:, 3] >= 0.5))
    open_count = float(len(train_y) - close_count)
    pos_weight = open_count / max(close_count, 1.0)
    dataset = TensorDataset(torch.from_numpy(train_xn), torch.from_numpy(train_target))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, drop_last=False)
    model = VisualActionMLP(train_xn.shape[1], args.hidden1, args.hidden2).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            output = model(xb)
            xyz_loss = nn.functional.smooth_l1_loss(output[:, :3], yb[:, :3])
            gripper_loss = bce(output[:, 3], yb[:, 3])
            loss = xyz_loss + float(args.gripper_loss_weight) * gripper_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)

        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            val_pred = predict(model, val_xn, target_mean, target_std, args.device, args.batch_size)
            metrics = regression_metrics(val_y, val_pred)
            balanced = metrics["gripper_balanced_accuracy"]
            if balanced is None:
                balanced = metrics["gripper_accuracy"]
            score = float(np.mean(metrics["xyz_rmse_m"]) / 0.02 + (1.0 - balanced))
            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "selection_score": score,
                "val": metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best_score - 1e-5:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 5
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No MLP checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    train_pred = predict(model, train_xn, target_mean, target_std, args.device, args.batch_size)
    val_pred = predict(model, val_xn, target_mean, target_std, args.device, args.batch_size)

    unique_parts_x = [primary["features"][p_train]]
    unique_parts_y = [primary["labels"][p_train]]
    if extra is not None:
        unique_parts_x.append(extra["features"][e_train])
        unique_parts_y.append(extra["labels"][e_train])
    unique_train = np.concatenate(unique_parts_x).astype(np.float32)
    unique_y = np.concatenate(unique_parts_y).astype(np.float32)
    unique_train_norm = (unique_train - feature_mean) / feature_std
    prototypes = unique_train_norm[:: max(int(args.prototype_stride), 1)].astype(np.float32)
    val_nearest = nearest_rms_distance(val_xn, prototypes)
    distance_threshold = float(np.quantile(val_nearest, args.distance_quantile))

    linear_layers = [module for module in model.layers if isinstance(module, nn.Linear)]
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(linear_layers):
        arrays[f"mlp_weight{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"mlp_bias{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)

    summary = {
        "primary_cache": str(args.primary_cache),
        "extra_cache": str(args.extra_cache) if args.extra_cache is not None else None,
        "extra_train_repeat": repeat,
        "train_samples_weighted": int(len(train_x)),
        "train_samples_unique": int(len(unique_train)),
        "val_samples": int(len(val_x)),
        "feature_dim": int(train_x.shape[1]),
        "hidden_dims": [int(args.hidden1), int(args.hidden2)],
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "train": regression_metrics(train_y, train_pred),
        "val": regression_metrics(val_y, val_pred),
        "val_primary": regression_metrics(primary["labels"][p_val], val_pred[: int(p_val.sum())]),
        "val_extra": regression_metrics(extra["labels"][e_val], val_pred[int(p_val.sum()) :])
        if extra is not None
        else None,
        "ood_gate": {
            "prototype_count": int(len(prototypes)),
            "distance_quantile": float(args.distance_quantile),
            "distance_threshold": distance_threshold,
            "val_nearest_rms_quantiles": {
                str(q): float(np.quantile(val_nearest, q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
            },
        },
        "history": history,
        "fair_input_contract": [
            "observation.image",
            "observation.wrist_image",
            "language instruction",
            "observation.state[:6]",
        ],
        "forbidden_inputs_used": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        projection=primary["projection"].astype(np.float32),
        prototypes=prototypes,
        distance_threshold=np.asarray(distance_threshold, dtype=np.float32),
        output_min=unique_y.min(axis=0).astype(np.float32),
        output_max=unique_y.max(axis=0).astype(np.float32),
        output_mode=np.asarray("mlp_direct_eef_abs_and_gripper"),
        activation=np.asarray("gelu_tanh"),
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
