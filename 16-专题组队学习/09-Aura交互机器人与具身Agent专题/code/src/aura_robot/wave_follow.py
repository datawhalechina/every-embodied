"""Wave-and-come-here interaction preview for the fixed Aura face camera.

The default mode observes a waving/calling person and prints bounded velocity
recommendations. It never sends motion. Actuator execution is intentionally a
separate, guarded follow-up once the camera view, battery and clear area are
accepted in person.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from websockets.sync.client import connect

from .debug import AuraClient
from .lightnav import PersistentStreamClient, validate_prediction
from .person_follow import decide_follow, target_from_lightnav_response


WAVE_INSTRUCTION = (
    "After an open-palm wake gesture, a person is waving and calling the robot to come here. "
    "Find that person. "
    "If the person is off-center, turn toward them; turn right when they are on "
    "the robot's right and turn left when they are on its left. Once facing the "
    "person and the open space, approach slowly. If no person is visible, stop."
)


def bounded_interaction_command(preview: dict[str, float], *, max_linear: float = 0.02,
                                max_angular: float = 0.12) -> dict[str, float | str]:
    """Apply a deliberately lower interaction speed limit than lab teleop."""
    linear = float(preview.get("linear_x", 0.0))
    angular = float(preview.get("angular_z", 0.0))
    if not all(math.isfinite(value) for value in (linear, angular)):
        raise ValueError("non-finite preview command")
    return {"linear_x": max(-max_linear, min(max_linear, linear)),
            "angular_z": max(-max_angular, min(max_angular, angular)),
            "mode": "wave_preview"}


def run_preview(client: AuraClient, server: str, stream_url: str, instruction: str,
                output_dir: Path, frames: int = 20, fps: float = 4.0) -> dict:
    if not instruction.strip() or not 1 <= frames <= 120 or not 0.2 <= fps <= 10:
        raise ValueError("instruction, frames or fps out of bounds")
    output_dir.mkdir(parents=False, exist_ok=False)
    motion = client.http("/api/motion/status")
    if motion.get("linear_x") != 0 or motion.get("angular_z") != 0:
        raise RuntimeError("Wave preview requires a stopped robot")
    session_id = "aura-wave-preview-" + uuid4().hex[:12]
    stream = PersistentStreamClient(stream_url)
    rows = []
    next_tick = time.monotonic()
    try:
        with connect(server, proxy=None, open_timeout=10, close_timeout=2,
                     max_size=4 * 1024 * 1024) as ws:
            for action, data in [("login", {"clientId": session_id}), ("reset", {})]:
                ws.send(json.dumps({"action": action, "data": data}))
                reply = json.loads(ws.recv(timeout=15))
                if reply.get("action") != action or reply.get("data", {}).get("rc") != 0:
                    raise RuntimeError(f"LightNav {action} failed")
            for seq in range(frames):
                next_tick += 1.0 / fps
                fetched = time.monotonic()
                jpeg, headers = stream.frame()
                fetch_ms = (time.monotonic() - fetched) * 1000
                if seq in (0, frames - 1):
                    (output_dir / f"camera-{seq:04d}.jpg").write_bytes(jpeg)
                sent = time.monotonic()
                ws.send(json.dumps({"action": "next", "data": {
                    "seq": seq, "image": base64.b64encode(jpeg).decode(),
                    "instruction": instruction}}))
                response = json.loads(ws.recv(timeout=45))
                round_trip_ms = (time.monotonic() - sent) * 1000
                raw_data = response.get("data") if isinstance(response, dict) else None
                # Validate the complete trajectory response for evidence, but
                # derive the interaction command from the tracking point.  A
                # vln server may produce a plausible waypoint while saying it
                # cannot see the person; that must not become a turn command.
                model_preview = validate_prediction(response, seq)
                target = target_from_lightnav_response(response)
                pointing = raw_data.get("pointing") if isinstance(raw_data, dict) else None
                frame_size = pointing.get("frame_size") if isinstance(pointing, dict) else None
                image_width = frame_size[0] if isinstance(frame_size, list) and frame_size else 960
                decision = decide_follow(target, image_width=float(image_width))
                command = bounded_interaction_command({
                    "linear_x": decision.linear_x,
                    "angular_z": decision.angular_z,
                })
                rows.append({"seq": seq, "frame_sequence": headers.get("x-frame-sequence"),
                             "frame_age_ms": headers.get("x-frame-age-ms"),
                             "fetch_ms": round(fetch_ms, 1),
                             "round_trip_ms": round(round_trip_ms, 1),
                             "server_latency_ms": raw_data.get("latency_ms") if isinstance(raw_data, dict) else None,
                             "visible": raw_data.get("visible") if isinstance(raw_data, dict) else None,
                             "stop": raw_data.get("stop") if isinstance(raw_data, dict) else None,
                             "model_preview": model_preview,
                             "target_center_x_px": target.center_x_px if target else None,
                             "tracking_state": decision.state.value,
                             "tracking_reason": decision.reason,
                             "command": command})
                if isinstance(raw_data, dict) and raw_data.get("stop"):
                    break
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        stream.close()
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "robot": client.base, "server": server, "stream_url": stream_url,
              "session_id": session_id, "instruction": instruction,
              "frames_requested": frames, "frames_processed": len(rows),
              "target_fps": fps, "rows": rows, "motion_sent": False,
              "interaction_speed_limit": {"linear_m_s": 0.02, "angular_rad_s": 0.12},
              "limitations": (
                  "Preview only; gesture semantics are supplied by the instruction, "
                  "not independently certified hand-pose detection. The LightNav "
                  "pointing channel has no calibrated target distance, so this "
                  "preview turns/holds but never authorizes approach. No odometry, "
                  "obstacle safety or actuator execution.")}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--server", default="ws://127.0.0.1:18050")
    parser.add_argument("--stream-url", default="http://127.0.0.1:18076/frame.jpg")
    parser.add_argument("--instruction", default=WAVE_INSTRUCTION)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_preview(AuraClient(args.host, timeout=30), args.server, args.stream_url,
                             args.instruction, args.output_dir, args.frames, args.fps)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
