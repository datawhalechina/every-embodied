#!/usr/bin/env python3
"""Compare exported NumPy GRU inference with the training-time PyTorch module."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_smolvla_physical import load_pi0_visual_contact_head, pi0_visual_gripper_gru_step
from train_pi0_visual_gripper_gru import GripperGRU


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    archive = np.load(args.head, allow_pickle=False)
    input_dim = int(archive["gripper_feature_mean"].shape[0])
    hidden_dim = int(archive["gripper_gru_weight_hh"].shape[1])
    model = GripperGRU(input_dim, hidden_dim)
    state = model.state_dict()
    mapping = {
        "input.weight": "gripper_input_weight",
        "input.bias": "gripper_input_bias",
        "norm.weight": "gripper_norm_weight",
        "norm.bias": "gripper_norm_bias",
        "gru.weight_ih_l0": "gripper_gru_weight_ih",
        "gru.weight_hh_l0": "gripper_gru_weight_hh",
        "gru.bias_ih_l0": "gripper_gru_bias_ih",
        "gru.bias_hh_l0": "gripper_gru_bias_hh",
        "output.weight": "gripper_output_weight",
        "output.bias": "gripper_output_bias",
    }
    for torch_key, archive_key in mapping.items():
        state[torch_key] = torch.as_tensor(archive[archive_key]).to(state[torch_key].dtype)
    model.load_state_dict(state)
    model.eval()

    generator = np.random.default_rng(20260711)
    normalized = generator.normal(size=(args.steps, input_dim)).astype(np.float32)
    raw = normalized * archive["gripper_feature_std"] + archive["gripper_feature_mean"]
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.as_tensor(normalized[None])))[0].numpy()

    head = load_pi0_visual_contact_head(args.head, "cpu")
    runtime = SimpleNamespace(pi0_visual_gripper_hidden=None)
    observed = []
    for row in raw:
        full_feature = np.concatenate([row, np.zeros(1, dtype=np.float32)])[None]
        observed.append(pi0_visual_gripper_gru_step(head, full_feature, runtime))
    observed_array = np.asarray(observed, dtype=np.float32)
    max_abs = float(np.max(np.abs(expected - observed_array)))
    print(f"steps={args.steps} max_abs={max_abs:.9g}")
    if max_abs > 5e-5:
        raise SystemExit(f"GRU parity failed: max_abs={max_abs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
