#!/usr/bin/env python3
"""Run one deterministic MuJoCo-Warp variant and emit its final state."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--variant", choices=("native", "eager"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["WP_HIP_CONDITIONAL_NATIVE"] = "1" if args.variant == "native" else "0"
    os.environ["WP_HIP_CONDITIONAL_EMULATION"] = "0"

    import mujoco  # type: ignore
    import mujoco_warp as mjw  # type: ignore
    import warp as wp  # type: ignore

    mjm = mujoco.MjModel.from_xml_path(str(args.model))
    model = mjw.put_model(mjm)
    data = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=args.worlds, nconmax=128, njmax=128)

    for _ in range(4):
        mjw.step(model, data)
        wp.synchronize()

    started = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(model, data)
    wp.synchronize()

    result = {
        "schema": "mujoco-warp-amd-native-state-v1",
        "variant": args.variant,
        "device": args.device,
        "python": platform.python_version(),
        "worlds": args.worlds,
        "steps": args.steps,
        "elapsed_s": time.perf_counter() - started,
        "qpos": data.qpos.numpy().tolist(),
        "qvel": data.qvel.numpy().tolist(),
        "time": data.time.numpy().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"qpos", "qvel", "time"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
