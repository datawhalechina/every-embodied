"""Stationary Aura -> lab LightNav-0 inference. Never sends actuator commands."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from http.client import HTTPConnection
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4
from urllib.request import Request, build_opener, ProxyHandler

from websockets.sync.client import connect
from .debug import AuraClient

MAX_WS_MESSAGE_BYTES = 64 * 1024 * 1024  # Official LightNav protocol client limit.


def validate_prediction(response, seq):
    if not isinstance(response, dict) or response.get("action") != "next":
        raise ValueError("Unexpected LightNav response action")
    data = response.get("data")
    if not isinstance(data, dict) or data.get("rc") != 0 or data.get("seq") != seq:
        raise ValueError("Failed or mismatched LightNav response")
    if type(data.get("stop")) is not bool:
        raise ValueError("Missing boolean stop flag")
    if data["stop"]:
        return {"linear_x": 0.0, "angular_z": 0.0, "reason": "model_stop"}
    actions = data.get("actions")
    rows = actions.get("actions") if isinstance(actions, dict) else None
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("Missing or excessive waypoint chunk")
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in row
        ):
            raise ValueError("Waypoints must be finite [forward_m, lateral_m, yaw_rad]")
    fwd, lateral, yaw = rows[0]
    # Official normalized reference mapping, reduced to project lab speed limits.
    # No lateral actuator exists on Aura. This is a preview, NOT a path tracker.
    return {"linear_x": max(-1.0, min(1.0, fwd / 0.375)) * 0.05,
            "angular_z": max(-1.0, min(1.0, yaw / (math.pi / 20))) * 0.3,
            "untracked_lateral_m": lateral, "reason": "preview_only"}


def fetch_stream_frame(stream_url: str, timeout: float = 2.0) -> tuple[bytes, dict[str, str]]:
    opener = build_opener(ProxyHandler({}))
    request = Request(stream_url, headers={"Cache-Control": "no-cache"})
    with opener.open(request, timeout=timeout) as response:
        jpeg = response.read(8 * 1024 * 1024 + 1)
        if len(jpeg) > 8 * 1024 * 1024 or not jpeg.startswith(b"\xff\xd8\xff"):
            raise ValueError("camera stream did not return a JPEG")
        return jpeg, {key.lower(): value for key, value in response.headers.items()}


class PersistentStreamClient:
    """Reuse one HTTP connection so each frame is not a TCP handshake."""

    def __init__(self, stream_url: str, timeout: float = 2.0) -> None:
        parsed = urlsplit(stream_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("stream URL must be a plain http URL without query/fragment")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path or "/frame.jpg"
        self.timeout = timeout
        self._connection: HTTPConnection | None = None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def frame(self) -> tuple[bytes, dict[str, str]]:
        for attempt in range(2):
            try:
                if self._connection is None:
                    self._connection = HTTPConnection(self.host, self.port, timeout=self.timeout)
                self._connection.request("GET", self.path,
                                         headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
                response = self._connection.getresponse()
                raw = response.read(8 * 1024 * 1024 + 1)
                headers = {key.lower(): value for key, value in response.getheaders()}
                if response.status != 200:
                    raise RuntimeError(f"camera stream HTTP {response.status}")
                if len(raw) > 8 * 1024 * 1024 or not raw.startswith(b"\xff\xd8\xff"):
                    raise ValueError("camera stream did not return a JPEG")
                return raw, headers
            except Exception:
                self.close()
                if attempt:
                    raise
        raise RuntimeError("unreachable")


def stationary_inference(client, server, instruction, output_dir, stream_url=None):
    if not instruction.strip():
        raise ValueError("Nonempty navigation instruction required")
    output_dir.mkdir(parents=True, exist_ok=False)
    motion = client.http("/api/motion/status")
    if motion.get("linear_x") != 0 or motion.get("angular_z") != 0:
        raise RuntimeError("Stationary inference requires stopped robot")
    started = time.monotonic()
    headers = {}
    if stream_url:
        jpeg, headers = fetch_stream_frame(stream_url)
    else:
        jpeg = client.http("/api/media/camera/snapshot")
    capture_s = time.monotonic() - started
    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("Camera did not return JPEG")
    (output_dir / "camera.jpg").write_bytes(jpeg)
    session_id = "aura-stationary-" + uuid4().hex[:12]
    with connect(server, proxy=None, open_timeout=10, close_timeout=2,
                 max_size=MAX_WS_MESSAGE_BYTES) as ws:
        for action, data in [("login", {"clientId": session_id}), ("reset", {})]:
            ws.send(json.dumps({"action": action, "data": data}))
            reply = json.loads(ws.recv(timeout=15))
            if reply.get("action") != action or reply.get("data", {}).get("rc") != 0:
                raise RuntimeError(f"LightNav {action} failed")
        started = time.monotonic()
        ws.send(json.dumps({"action": "next", "data": {"seq": 0,
            "image": base64.b64encode(jpeg).decode(), "instruction": instruction}}))
        prediction = json.loads(ws.recv(timeout=45))
        round_trip_s = time.monotonic() - started
    # Preserve raw evidence even if schema validation fails.
    (output_dir / "prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    preview = validate_prediction(prediction, 0)
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "robot": client.base, "server": server, "session_id": session_id,
              "instruction": instruction, "capture_s": capture_s,
              "inference_round_trip_s": round_trip_s, "preview": preview,
              "motion_sent": False, "frames": 1,
              "camera_source": stream_url or client.base + "/api/media/camera/snapshot",
              "camera_headers": headers,
              "limitations": "Single stationary frame; no temporal navigation, odometry or obstacle safety validation"}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def stream_preview(client, server, instruction, output_dir, stream_url, frames=12, fps=4.0):
    """Run a bounded real-camera LightNav loop without sending motion."""
    if not stream_url:
        raise ValueError("stream-preview requires --stream-url")
    if not instruction.strip() or not 1 <= frames <= 120 or not 0.2 <= fps <= 10:
        raise ValueError("instruction, frames or fps out of bounds")
    output_dir.mkdir(parents=False, exist_ok=False)
    motion = client.http("/api/motion/status")
    if motion.get("linear_x") != 0 or motion.get("angular_z") != 0:
        raise RuntimeError("Stream preview requires stopped robot")
    session_id = "aura-preview-" + uuid4().hex[:12]
    rows = []
    next_tick = time.monotonic()
    stream = PersistentStreamClient(stream_url)
    try:
        with connect(server, proxy=None, open_timeout=10, close_timeout=2,
                     max_size=MAX_WS_MESSAGE_BYTES) as ws:
            for action, data in [("login", {"clientId": session_id}), ("reset", {})]:
                ws.send(json.dumps({"action": action, "data": data}))
                reply = json.loads(ws.recv(timeout=15))
                if reply.get("action") != action or reply.get("data", {}).get("rc") != 0:
                    raise RuntimeError(f"LightNav {action} failed")
            for seq in range(frames):
                next_tick += 1.0 / fps
                fetched = time.monotonic()
                jpeg, headers = stream.frame()
                fetch_s = time.monotonic() - fetched
                if seq in (0, frames - 1):
                    (output_dir / f"camera-{seq:04d}.jpg").write_bytes(jpeg)
                sent = time.monotonic()
                ws.send(json.dumps({"action": "next", "data": {
                    "seq": seq, "image": base64.b64encode(jpeg).decode(),
                    "instruction": instruction}}))
                response = json.loads(ws.recv(timeout=45))
                inference_s = time.monotonic() - sent
                preview = validate_prediction(response, seq)
                data = response["data"]
                rows.append({"seq": seq, "frame_sequence": headers.get("x-frame-sequence"),
                             "frame_age_ms": headers.get("x-frame-age-ms"),
                             "fetch_ms": round(fetch_s * 1000, 1),
                             "round_trip_ms": round(inference_s * 1000, 1),
                             "server_latency_ms": data.get("latency_ms"),
                             "vit_cache_hits": data.get("timings_ms", {}).get("vit_cache_hits"),
                             "stop": data.get("stop"), "preview": preview})
                if data.get("stop"):
                    break
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        stream.close()
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "robot": client.base, "server": server, "stream_url": stream_url,
              "session_id": session_id, "instruction": instruction,
              "frames_requested": frames, "frames_processed": len(rows),
              "target_fps": fps, "rows": rows, "motion_sent": False,
              "limitations": "Preview only; no actuator, odometry, obstacle safety or camera pose correction"}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--server", default="ws://127.0.0.1:18050")
    parser.add_argument("--stream-url", help="Persistent camera bridge URL, e.g. http://127.0.0.1:18076/frame.jpg")
    parser.add_argument("--frames", type=int, default=1, help="1 for stationary inference; >1 for bounded stream preview")
    parser.add_argument("--fps", type=float, default=4.0, help="Stream preview target rate")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        client = AuraClient(args.host, timeout=30)
        if args.frames == 1:
            result = stationary_inference(client, args.server, args.instruction,
                                          args.output_dir, args.stream_url)
        else:
            result = stream_preview(client, args.server, args.instruction,
                                    args.output_dir, args.stream_url, args.frames, args.fps)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
