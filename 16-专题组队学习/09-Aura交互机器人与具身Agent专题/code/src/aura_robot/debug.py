"""Source-verified AuraOS diagnostics; no automatic motion or cloud calls."""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from websockets.sync.client import connect
from .features import (audio_summary, save_wav, app_action, management,
                       deploy_readonly_test, deploy_app, telemetry_samples)


READS = {
    "daemon": "/api/daemon/status",
    "media": "/api/media/status",
    "motion": "/api/motion/status",
    "telemetry": "/api/motion/telemetry",
    "apps": "/api/apps",
}
WS_COMMANDS = ["get_status", "get_version", "capture", "transcribe", "synthesize",
               "say", "chat", "generate", "shutdown"]


class AuraClient:
    def __init__(self, host="192.168.4.1", port=8765, timeout=8.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws/sdk"
        self.timeout = timeout
        # Robot traffic must not pass through the PC's Internet proxy.
        self.opener = build_opener(ProxyHandler({}))

    def http(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.base + path, data=data,
                      headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError("response exceeds 32 MiB")
            if "json" in response.headers.get("Content-Type", ""):
                return json.loads(raw)
            return raw

    def ws(self, command, **payload):
        if command not in WS_COMMANDS or command == "shutdown":
            raise ValueError("Unsupported or disabled command")
        with connect(self.ws_url, proxy=None, open_timeout=self.timeout,
                     close_timeout=2, max_size=32 * 1024 * 1024) as socket:
            initial = json.loads(socket.recv(timeout=self.timeout))
            if initial.get("type") != "daemon_status":
                raise RuntimeError(f"Unexpected handshake: {initial.get('type')}")
            request_id = str(uuid4())
            socket.send(json.dumps(dict(payload, type=command, request_id=request_id)))
            if command == "say":
                return {"status": "sent_unconfirmed", "note": "No acknowledgement in vendor protocol"}
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SDK response timeout")
                result = json.loads(socket.recv(timeout=remaining))
                if result.get("type") == "error" and result.get("request_id") in (None, request_id):
                    raise RuntimeError(result.get("message", "Daemon error"))
                if command in ("get_status", "get_version"):
                    # Status broadcasts and command replies have no IDs in this firmware.
                    if result.get("type") == "daemon_status":
                        return result
                elif result.get("request_id") == request_id:
                    expected = f"{command}_response"
                    if result.get("type") != expected:
                        raise RuntimeError(f"Expected {expected}")
                    return result

    def drive_for(self, linear, angular, duration, *, confirmed=False):
        if not confirmed:
            raise ValueError("Motion requires explicit clear-area confirmation")
        if not all(math.isfinite(v) for v in (linear, angular, duration)):
            raise ValueError("Motion arguments must be finite")
        if abs(linear) > 0.05 or abs(angular) > 0.3 or not 0 < duration <= 1:
            raise ValueError("Lab limits: 0.05 m/s, 0.3 rad/s, duration <= 1 s")
        try:
            result = self.http("/api/motion/drive_for", {
                "linear_x": linear, "angular_z": angular, "duration": duration})
            time.sleep(duration)
            return result
        finally:
            # Best effort only: network loss can prevent stop delivery.
            self.http("/api/motion/stop", {})


def assess(name, value):
    if name == "apps":
        return ("pass", "List returned") if isinstance(value, list) else ("fail", "Not a list")
    if not isinstance(value, dict):
        return "fail", "Not a JSON object"
    if value.get("error"):
        return "fail", str(value["error"])
    if name == "telemetry":
        missing = [key for key in ("imu", "wheel_speed") if value.get(key) is None]
        return ("partial", "Missing: " + ", ".join(missing)) if missing else ("pass", "IMU and wheel feedback present")
    if name in ("daemon", "ws_status") and value.get("state") != "running":
        return "fail", "Daemon not running"
    if name == "media":
        if not value.get("running") or value.get("released"):
            return "fail", "Media not owned/running"
        return "pass", "Ownership/configuration only; not sensor validation"
    if name == "ws_version" and not value.get("version"):
        return "partial", "Firmware returns null version; reply indistinguishable from status broadcast"
    if name == "motion" and not value.get("connected", False):
        return "partial", "Connection not confirmed by connected field"
    return "pass", "Read succeeded; no actuator test"


def probe(client):
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "target": client.base, "tests": [], "inventory": []}
    try:
        schema = client.http("/openapi.json")
        for path, methods in schema["paths"].items():
            for method, spec in methods.items():
                if method not in {"get", "post", "put", "delete", "patch"}:
                    continue
                report["inventory"].append({"transport": "http", "method": method.upper(),
                    "path": path, "summary": spec.get("summary"), "status": "not_tested"})
    except Exception as exc:
        report["tests"].append({"name": "openapi", "status": "fail", "detail": str(exc)})
    cases = [(name, lambda path=path: client.http(path)) for name, path in READS.items()]
    cases += [("ws_status", lambda: client.ws("get_status")),
              ("ws_version", lambda: client.ws("get_version"))]
    for name, operation in cases:
        started = time.monotonic()
        try:
            value = operation()
            status, detail = assess(name, value)
            row = {"name": name, "status": status, "detail": detail, "response": value}
        except Exception as exc:
            row = {"name": name, "status": "fail", "detail": str(exc)}
        row["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        report["tests"].append(row)
        for item in report["inventory"]:
            if item["method"] == "GET" and item["path"] == READS.get(name):
                item["status"] = row["status"]
    for command in WS_COMMANDS:
        row = next((r for r in report["tests"] if r["name"] == {
            "get_status": "ws_status", "get_version": "ws_version"}.get(command)), None)
        report["inventory"].append({"transport": "websocket", "command": command,
            "status": row["status"] if row else "not_tested"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=8)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("probe", help="Read-only HTTP/WS suite and full API inventory")
    p.add_argument("--output", type=Path, default=Path("artifacts/sdk-probe.json"))
    p = sub.add_parser("camera", help="Capture one local JPEG; no cloud upload")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("battery", help="Stationary voltage diagnostic; percentage/charging may be unknown")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("display")
    p.add_argument("text")
    p = sub.add_parser("ws", help="Explicit cloud/audio API call; never part of probe")
    p.add_argument("command", choices=["synthesize", "transcribe", "capture"])
    p.add_argument("--text", default="Aura SDK test")
    p.add_argument("--source", choices=["camera", "microphone"], default="camera")
    p.add_argument("--input", type=Path, help="Raw PCM16 mono 16 kHz for transcribe")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-cloud", action="store_true")
    p = sub.add_parser("record", help="SDK microphone to WAV; rejects empty audio")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audio-check", help="Offline quality check of a saved SDK response")
    p.add_argument("input", type=Path)
    p = sub.add_parser("apps", help="App list/logs/start/stop; mutation requires confirmation")
    p.add_argument("operation", choices=["list", "logs", "start", "stop"])
    p.add_argument("--name")
    p.add_argument("--confirm-change", action="store_true")
    p = sub.add_parser("deploy", help="Run a trusted AuraApp; refuses registered name overwrite")
    p.add_argument("script", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--confirm-change", action="store_true")
    p = sub.add_parser("telemetry", help="Repeated read-only IMU/wheel measurements")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("app-selftest", help="Deploy unique read-only app, verify logs, stop it")
    p.add_argument("--confirm-deploy", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("manage", help="Explicit lifecycle/ownership/skill controls")
    p.add_argument("group", choices=["daemon", "media", "skill"])
    p.add_argument("operation")
    p.add_argument("--name")
    p.add_argument("--confirm-change", action="store_true")
    p = sub.add_parser("drive", help="Only after physical clearance; SI units")
    p.add_argument("--linear", type=float, default=0)
    p.add_argument("--angular", type=float, default=0)
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--confirm-clear-area", action="store_true")
    sub.add_parser("stop", help="Best-effort software stop, not a hardware emergency stop")
    args = parser.parse_args()
    client = AuraClient(args.host, args.port, args.timeout)
    try:
        if args.action == "battery":
            if not 1 <= args.count <= 30:
                raise ValueError("count must be 1..30")
            with args.output.open("x", encoding="utf-8") as file:
                samples = []
                for index in range(args.count):
                    if index:
                        time.sleep(2)
                    value = client.http("/api/motion/battery")
                    value["observed_at"] = datetime.now(timezone.utc).isoformat()
                    samples.append(value)
                    print(json.dumps(value, ensure_ascii=False), flush=True)
                json.dump({"target": client.base, "samples": samples}, file, indent=2)
            if any(s.get("error") or s.get("voltage_v") is None for s in samples):
                raise SystemExit(1)
            return
        if args.action == "deploy":
            print(json.dumps(deploy_app(client, args.script, args.name,
                                       confirmed=args.confirm_change), indent=2))
            return
        if args.action == "telemetry":
            with args.output.open("x", encoding="utf-8") as file:
                result = telemetry_samples(client, args.count)
                json.dump(result, file, indent=2)
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
            return
        if args.action == "audio-check":
            result = audio_summary(json.loads(args.input.read_text(encoding="utf-8-sig")))
            print(json.dumps(result, indent=2))
            if result["status"] != "signal_present":
                raise SystemExit(1)
            return
        if args.action == "record":
            result = save_wav(client.ws("capture", source="microphone"), args.output)
            print(json.dumps(result, indent=2))
            if result["status"] != "signal_present":
                raise SystemExit(1)
            return
        if args.action == "apps":
            result = client.http("/api/apps") if args.operation == "list" else app_action(
                client, args.name, args.operation, confirmed=args.confirm_change)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.action == "manage":
            result = management(client, args.group, args.operation,
                                confirmed=args.confirm_change, name=args.name)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.action == "app-selftest":
            if not args.confirm_deploy:
                raise ValueError("Requires --confirm-deploy; uploads and runs our read-only test app")
            # Reserve evidence file before performing any external writes.
            with args.output.open("x", encoding="utf-8") as file:
                result = deploy_readonly_test(client)
                json.dump(result, file, ensure_ascii=False, indent=2)
            print(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2))
            if result["status"] != "pass":
                raise SystemExit(1)
            return
        if args.action == "probe":
            result = probe(client)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            for row in result["tests"]:
                print(f"{row['status']:8} {row['name']}: {row['detail']}")
            print(args.output.resolve())
            if any(r["status"] in {"fail", "partial"} for r in result["tests"]):
                raise SystemExit(1)
            return
        if args.action == "camera":
            result = client.http("/api/media/camera/snapshot")
            if not isinstance(result, bytes) or not result.startswith(b"\xff\xd8"):
                raise ValueError("Camera did not return a JPEG")
            with args.output.open("xb") as output:
                output.write(result)
            print(f"JPEG saved: {args.output.resolve()} ({len(result)} bytes)")
            return
        if args.action == "display":
            result = client.http("/api/media/display/text", {"text": args.text})
        elif args.action == "stop":
            result = client.http("/api/motion/stop", {})
        elif args.action == "drive":
            result = client.drive_for(args.linear, args.angular, args.duration,
                                      confirmed=args.confirm_clear_area)
        else:
            if args.command in {"synthesize", "transcribe"} and not args.allow_cloud:
                raise ValueError("Cloud APIs require --allow-cloud; audio may leave the device")
            payload = {"text": args.text} if args.command == "synthesize" else {"source": args.source}
            if args.command == "transcribe":
                if args.input is None:
                    raise ValueError("transcribe requires --input PCM16 mono 16 kHz")
                payload = {"audio": base64.b64encode(args.input.read_bytes()).decode()}
            result = client.ws(args.command, **payload)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2)
            print(f"Response saved: {args.output.resolve()}; content/quality needs validation")
            return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
