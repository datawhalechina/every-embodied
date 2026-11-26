#!/usr/bin/env python3
"""Persistent ultra-low-latency HTTP audio + WS2812B LED daemon running on D-Robotics RDK X5 Magic Box.
Stores pre-rendered 48kHz WAV clips in RAM disk (/dev/shm/tuntun_voice) and plays directly via ALSA hw:0,0
in <30ms with zero SSH or ROS2 startup delay.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CACHE_DIR = Path("/dev/shm/tuntun_voice")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PORT = int(os.getenv("TUNTUN_VOICE_PORT", "8095"))

_play_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None

# Optional WS2812B LED ring controller
_led_strip = None
try:
    sys.path.insert(0, "/userdata/magicbox/launch/lamp")
    from ws2812b import WS2812B
    _led_strip = WS2812B(num_leds=4)
except Exception:
    _led_strip = None

LED_PRESETS = {
    "off":     (0, 0, 0),
    "red":     (255, 0, 0),
    "green":   (0, 255, 110),
    "blue":    (0, 120, 255),
    "white":   (255, 240, 230),
    "warm":    (255, 175, 50),
    "yellow":  (255, 210, 0),
    "cyan":    (0, 235, 255),
    "magenta": (220, 40, 255),
}


def set_led(preset: str) -> bool:
    if _led_strip is None:
        return False
    rgb = LED_PRESETS.get(preset, LED_PRESETS["cyan"])
    try:
        if preset == "off":
            _led_strip.clear()
        else:
            _led_strip.set_all_same_color(*rgb)
        return True
    except Exception:
        return False


def get_volume() -> int:
    try:
        out = subprocess.check_output(["amixer", "-c", "1", "sget", "DAC"], text=True, stderr=subprocess.DEVNULL)
        import re
        m = re.search(r"\[(\d+)%\]", out)
        return int(m.group(1)) if m else 85
    except Exception:
        return 85


def set_volume(percent: int) -> int:
    pct = max(20, min(100, int(percent)))
    for ctl in ("DAC", "HPL", "HPR", "SPKL", "SPKR"):
        subprocess.run(["amixer", "-q", "-c", "1", "sset", ctl, f"{pct}%"], check=False)
    return pct


def play_wav(wav_path: Path, led: str | None = None, interrupt: bool = True) -> None:
    global _current_proc
    with _play_lock:
        if interrupt and _current_proc is not None and _current_proc.poll() is None:
            try:
                _current_proc.terminate()
                _current_proc.wait(timeout=0.5)
            except Exception:
                try:
                    _current_proc.kill()
                except Exception:
                    pass
        if led:
            set_led(led)
        # Use plughw:0,0 (I2S1 HiFi DAC) for instant hardware playback
        _current_proc = subprocess.Popen(
            ["aplay", "-q", "-D", "plughw:0,0", str(wav_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "ok": True,
                "service": "tuntunclaw_magicbox_voice",
                "volume": get_volume(),
                "led_ready": _led_strip is not None,
                "cached_files": len(list(CACHE_DIR.glob("*.wav"))),
            })
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""

        if self.path.startswith("/cache/"):
            key = os.path.basename(self.path[len("/cache/"):])
            if not key:
                self._send_json(400, {"ok": False, "error": "empty_key"})
                return
            dst = CACHE_DIR / f"{key}.wav"
            dst.write_bytes(body)
            self._send_json(200, {"ok": True, "key": key, "bytes": len(body)})
            return

        data = json.loads(body.decode("utf-8")) if body else {}

        if self.path == "/play":
            key = os.path.basename(str(data.get("key", "")))
            wav = CACHE_DIR / f"{key}.wav"
            if not wav.exists():
                self._send_json(404, {"ok": False, "error": "cache_miss", "key": key})
                return
            play_wav(wav, led=data.get("led"), interrupt=bool(data.get("interrupt", True)))
            self._send_json(200, {"ok": True, "key": key, "playing": True})
            return

        if self.path == "/volume":
            pct = set_volume(int(data.get("percent", 88)))
            self._send_json(200, {"ok": True, "volume": pct})
            return

        if self.path == "/led":
            ok = set_led(str(data.get("preset", "cyan")))
            self._send_json(200, {"ok": ok, "preset": data.get("preset")})
            return

        if self.path == "/stop":
            global _current_proc
            with _play_lock:
                if _current_proc is not None and _current_proc.poll() is None:
                    _current_proc.terminate()
            self._send_json(200, {"ok": True})
            return

        self._send_json(404, {"ok": False, "error": "not_found"})


if __name__ == "__main__":
    set_volume(88)
    set_led("cyan")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"TuntunClaw Magic Box Voice Daemon listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
