#!/usr/bin/env python3
"""Verify the exact LeRobot/ROCm runtime used by the AMD notebooks."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "python": "3.10",
    "torch": "2.11.0+rocm7.13.0a20260425",
    "torchvision": "0.26.0+rocm7.13.0a20260425",
    "transformers": "4.53.3",
    "huggingface-hub": "0.35.3",
    "safetensors": "0.7.0",
    "draccus": "0.10.0",
    "datasets": "2.19.0",
    "numpy": "2.2.6",
    "h5py": "3.16.0",
    "mujoco": "3.10.0",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def collect() -> dict:
    result: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "lerobot_src": os.environ.get("LEROBOT_SRC", ""),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "packages": {name: package_version(name) for name in EXPECTED if name != "python"},
    }

    try:
        import torch

        result["torch_hip"] = torch.version.hip
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        result["torch_error"] = repr(exc)

    for module_name in (
        "lerobot",
        "lerobot.common.datasets.factory",
        "lerobot.common.policies.act.configuration_act",
        "lerobot.common.policies.pi0.modeling_pi0",
    ):
        try:
            module = importlib.import_module(module_name)
            result.setdefault("imports", {})[module_name] = str(getattr(module, "__file__", "ok"))
        except Exception as exc:
            result.setdefault("imports", {})[module_name] = f"ERROR: {exc!r}"

    root = Path(result["lerobot_src"] or ".")
    result["lerobot_git"] = git_revision(root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = collect()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    failures = []
    if not result["python"].startswith(EXPECTED["python"]):
        failures.append(f"Python {result['python']} != {EXPECTED['python']}.x")
    for name, expected in EXPECTED.items():
        if name == "python":
            continue
        actual = result["packages"].get(name)
        if actual != expected:
            failures.append(f"{name}={actual!r} != {expected!r}")
    if result.get("torch_hip") != "7.13.26162":
        failures.append(f"torch HIP={result.get('torch_hip')!r} != '7.13.26162'")
    if not result.get("cuda_available"):
        failures.append("torch.cuda.is_available() is false")
    for module_name, location in result.get("imports", {}).items():
        if location.startswith("ERROR:"):
            failures.append(f"import failed: {module_name}: {location}")
    if result.get("lerobot_git") != "10b7b3532543b4adfb65760f02a49b4c537afde7":
        failures.append(f"LeRobot commit mismatch: {result.get('lerobot_git')}")

    if failures:
        print("ENVIRONMENT_MISMATCH")
        for item in failures:
            print("-", item)
        return 2
    print("ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
