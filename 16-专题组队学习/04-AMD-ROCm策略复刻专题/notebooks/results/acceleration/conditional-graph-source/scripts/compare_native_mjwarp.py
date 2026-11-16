#!/usr/bin/env python3
"""Compare eager and native MuJoCo-Warp state dumps and benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def maximum_error(left, right) -> float:
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def flatten(values):
    for value in values:
        if isinstance(value, list):
            yield from flatten(value)
        else:
            yield float(value)


def read_json_line(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON object found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-state", type=Path, required=True)
    parser.add_argument("--eager-state", type=Path, required=True)
    parser.add_argument("--native-benchmark", type=Path, required=True)
    parser.add_argument("--eager-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_state = json.loads(args.native_state.read_text(encoding="utf-8"))
    eager_state = json.loads(args.eager_state.read_text(encoding="utf-8"))
    native_benchmark = read_json_line(args.native_benchmark)
    eager_benchmark = read_json_line(args.eager_benchmark)

    qpos_error = maximum_error(list(flatten(native_state["qpos"])), list(flatten(eager_state["qpos"])))
    qvel_error = maximum_error(list(flatten(native_state["qvel"])), list(flatten(eager_state["qvel"])))
    time_error = maximum_error(native_state["time"], eager_state["time"])
    native_runtime = float(native_benchmark["run_time"])
    eager_runtime = float(eager_benchmark["run_time"])
    result = {
        "schema": "mujoco-warp-amd-native-conditional-v1",
        "execution": "patched_hip_clr_runtime",
        "native_runtime_s": native_runtime,
        "eager_runtime_s": eager_runtime,
        "native_worlds_per_second": float(native_benchmark["steps_per_second"]),
        "eager_worlds_per_second": float(eager_benchmark["steps_per_second"]),
        "throughput_speedup": float(native_benchmark["steps_per_second"]) / float(eager_benchmark["steps_per_second"]),
        "runtime_reduction": 1.0 - native_runtime / eager_runtime,
        "max_abs_qpos_error": qpos_error,
        "max_abs_qvel_error": qvel_error,
        "max_abs_time_error": time_error,
        "numerically_equivalent": max(qpos_error, qvel_error, time_error) <= 1e-5,
        "native_state": str(args.native_state),
        "eager_state": str(args.eager_state),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
