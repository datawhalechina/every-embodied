"""Explicit feature helpers for the source-verified daemon API."""
from __future__ import annotations

import array
import ast
import base64
import json
import math
import sys
import time
import wave
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4


def audio_summary(response):
    """Validate PCM, excluding the first second's startup transient in level stats."""
    raw = base64.b64decode(response.get("audio") or "", validate=True)
    rate = response.get("sample_rate", 16000)
    if not isinstance(rate, int) or not 8000 <= rate <= 192000:
        raise ValueError("Invalid sample rate")
    if len(raw) % 2:
        raise ValueError("PCM16 byte count must be even")
    samples = array.array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    stable = samples[rate:] if len(samples) > rate else samples
    rms = math.sqrt(sum(v * v for v in stable) / len(stable)) if stable else 0
    return {"bytes": len(raw), "sample_rate": rate, "seconds": len(samples) / rate,
            "rms_after_startup": round(rms, 3),
            "peak_after_startup": max(map(abs, stable), default=0),
            "status": "empty" if not raw else "near_silent" if rms < 10 else "signal_present",
            "note": "Signal level does not prove speech intelligibility"}


def save_wav(response, output):
    summary = audio_summary(response)
    if summary["status"] == "empty":
        raise ValueError("Daemon returned empty audio; no WAV written")
    with Path(output).open("xb") as file:
        with wave.open(file, "wb") as wav:
            wav.setparams((1, 2, summary["sample_rate"], 0, "NONE", "not compressed"))
            wav.writeframes(base64.b64decode(response["audio"], validate=True))
    return summary


def app_action(client, name, action, *, confirmed=False, offset=0):
    if not name or action not in {"logs", "start", "stop"}:
        raise ValueError("Expected named app and logs/start/stop")
    if action != "logs" and not confirmed:
        raise ValueError("App start/stop requires --confirm-change")
    path = f"/api/apps/{quote(name, safe='')}/{action}"
    return client.http(path + f"?offset={max(0, offset)}") if action == "logs" else client.http(path, {})


def management(client, group, action, *, confirmed=False, name=None):
    allowed = {"daemon": {"start", "stop", "restart"}, "media": {"acquire", "release"},
               "skill": {"activate", "deactivate"}}
    if action not in allowed.get(group, set()):
        raise ValueError("Unsupported management action")
    if not confirmed:
        raise ValueError("This changes robot state; use --confirm-change")
    if group == "skill" and action == "activate" and not name:
        raise ValueError("Skill activation requires --name")
    return client.http(f"/api/{group}/{action}", {"name": name} if name else {})


def deploy_app(client, script, name, *, confirmed=False):
    if not confirmed:
        raise ValueError("Deploy executes Python on robot; requires --confirm-change")
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError("A simple app name is required")
    content = Path(script).read_text(encoding="utf-8")
    tree = ast.parse(content)
    literal_names = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "name" for t in n.targets)
        and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    if literal_names != [name]:
        raise ValueError("Use one literal name matching --name; trusted code only (not a sandbox)")
    if any(app["name"] == name for app in client.http("/api/apps")):
        raise ValueError("App already registered; refusing overwrite. Use apps start or a new name")
    return client.http("/api/apps/deploy_skill", {"filename": name + "-" + uuid4().hex[:8] + ".py",
                       "content": content, "replace": False})


def telemetry_samples(client, count=5):
    if not 1 <= count <= 20:
        raise ValueError("Sample count must be 1..20")
    rows = []
    for _ in range(count):
        rows.append(client.http("/api/motion/telemetry"))
        time.sleep(0.1)
    return {"samples": rows, "count": count,
            "imu_received": sum(r.get("imu") is not None for r in rows),
            "wheel_received": sum(r.get("wheel_speed") is not None for r in rows),
            "note": "Zero acceleration or missing frames need investigation; no motion test"}


def deploy_readonly_test(client):
    """Deploy only our unique bounded status reader, never an arbitrary script."""
    name = "aura-lab-status-" + uuid4().hex[:12]
    source = '''import asyncio, json
from urllib.request import build_opener, ProxyHandler
from aura.apps.app import AuraApp
class StatusTest(AuraApp):
    name = NAME
    description = "Lab read-only status probe; no motion, media or cloud"
    async def run(self, context):
        opener = build_opener(ProxyHandler({}))
        url = f"http://{context.daemon_host}:{context.daemon_port}/api/daemon/status"
        with opener.open(url, timeout=5) as response:
            data = json.load(response)
        print("LAB_STATUS " + json.dumps({"state": data.get("state")}), flush=True)
        await asyncio.sleep(15)
APP = StatusTest()
'''.replace("NAME", repr(name))
    result = {"name": name, "status": "fail", "source": source}
    attempted = False
    try:
        attempted = True
        result["deploy"] = client.http("/api/apps/deploy_skill", {
            "filename": name + ".py", "content": source, "replace": False})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            logs = app_action(client, name, "logs")
            if any('LAB_STATUS {"state": "running"}' in line for line in logs.get("lines", [])):
                result["logs"] = logs
                result["status"] = "pass"
                break
            time.sleep(0.5)
        else:
            result["logs"] = logs
            result["error"] = "No running-state evidence in app logs"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if attempted:
            try:
                result["stop"] = app_action(client, name, "stop", confirmed=True)
                result["final_state"] = next((a for a in client.http("/api/apps") if a["name"] == name), None)
                if not result["final_state"] or result["final_state"]["state"] != "stopped":
                    result["status"] = "fail"
            except Exception as exc:
                result["status"] = "fail"
                result["cleanup_error"] = str(exc)
    result["note"] = "Own app stopped; source remains under ~/.aura/deployed-skills. No delete API."
    return result
