from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_POWERSHELL_QUERY = r"""
Get-PnpDevice -PresentOnly |
  Where-Object { $_.InstanceId -like 'USB*' } |
  Select-Object Status, Class, FriendlyName, InstanceId |
  Sort-Object InstanceId |
  ConvertTo-Json -Compress
"""


def snapshot() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        raise RuntimeError("USB PnP snapshot currently supports Windows only")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _POWERSHELL_QUERY],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8-sig",
    )
    if not result.stdout.strip():
        return []
    data = json.loads(result.stdout)
    return data if isinstance(data, list) else [data]


def save_snapshot(path: Path) -> int:
    devices = snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(devices, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(devices)


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Snapshot must contain a JSON array: {path}")
    return data


def added_devices(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    old_ids = {device.get("InstanceId") for device in before}
    return [device for device in after if device.get("InstanceId") not in old_ids]
