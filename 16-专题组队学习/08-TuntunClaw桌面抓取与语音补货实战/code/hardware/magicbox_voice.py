"""TuntunClaw (囤囤钳) Low-Latency Bright Voice Engine for D-Robotics RDK X5 Magic Box.
- Synthesizes studio-clear Chinese neural speech via Microsoft Edge TTS (zh-CN-XiaoyiNeural / XiaoxiaoNeural / YunxiNeural)
- Applies small-speaker exhibition DSP EQ via ffmpeg:
    highpass=140Hz (removes boxy mud) + 3.2kHz presence boost (+4dB) + 6kHz air (+2.5dB) + dynaudnorm + alimiter
- Pushes pre-rendered 48kHz stereo WAVs to Magic Box RAM disk (/dev/shm/tuntun_voice) and triggers playback
  over persistent LAN HTTP daemon (port 8095) in <40ms (zero SSH cold-start, zero ROS2 topic pub delay!).
- Synchronizes Magic Box WS2812B RGB LED ring with every booth stage.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import edge_tts

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "hardware" / "voice_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_EXE = (
    shutil.which("ffmpeg")
    or r"/path/to/user\micromamba\envs\py311\Library\bin\ffmpeg.exe"
)

VOICE_PRESETS = {
    "xiaoyi":   {"voice": "zh-CN-XiaoyiNeural",   "rate": "+8%",  "pitch": "+4Hz", "label": "清亮元气女声 (小艺·推荐)"},
    "xiaoxiao": {"voice": "zh-CN-XiaoxiaoNeural", "rate": "+6%",  "pitch": "+2Hz", "label": "温柔甜美女声 (晓晓)"},
    "yunxi":    {"voice": "zh-CN-YunxiNeural",    "rate": "+7%",  "pitch": "+2Hz", "label": "清澈阳光男声 (云希)"},
}

# Small-speaker clarity & brightness DSP chain (tailored for Magic Box I2S speaker in noisy exhibition halls)
BRIGHT_EQ_FILTER = (
    "highpass=f=145,"
    "equalizer=f=3200:t=q:w=1.2:g=4.2,"
    "treble=g=2.8:f=6200,"
    "dynaudnorm=f=120:g=11:p=0.92,"
    "alimiter=limit=0.95"
)


class MagicBoxVoiceEngine:
    def __init__(
        self,
        host: str = "192.168.1.101",
        daemon_port: int = 8095,
        voice_preset: str = "xiaoyi",
    ) -> None:
        self.host = host
        self.daemon_port = daemon_port
        self.voice_preset = voice_preset if voice_preset in VOICE_PRESETS else "xiaoyi"
        self.enabled = True
        self.volume = 88
        self.last_text = ""
        self.last_stage = "idle"
        self.last_latency_ms = 0
        self.last_error = ""
        self.daemon_online = False
        self._lock = threading.RLock()
        self._uploaded_keys: set[str] = set()
        # Direct LAN opener (bypass Windows system proxy for 192.168.x.x!)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.daemon_port}"

    def _cache_key(self, text: str, preset: str | None = None) -> str:
        p = preset or self.voice_preset
        h = hashlib.md5(f"{p}|{text}".encode("utf-8")).hexdigest()[:16]
        return f"{p}_{h}"

    def _synthesize_wav(self, text: str, preset: str | None = None) -> Path:
        p = preset or self.voice_preset
        cfg = VOICE_PRESETS.get(p, VOICE_PRESETS["xiaoyi"])
        key = self._cache_key(text, p)
        wav_path = CACHE_DIR / f"{key}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 2000:
            return wav_path

        mp3_path = CACHE_DIR / f"{key}.mp3"

        async def _gen():
            comm = edge_tts.Communicate(
                text=text,
                voice=cfg["voice"],
                rate=cfg["rate"],
                pitch=cfg["pitch"],
            )
            await comm.save(str(mp3_path))

        # Run edge-tts cleanly regardless of whether caller is inside an event loop
        def _runner():
            asyncio.run(_gen())

        t = threading.Thread(target=_runner)
        t.start()
        t.join(timeout=18.0)
        if not mp3_path.exists() or mp3_path.stat().st_size < 100:
            raise RuntimeError("Edge TTS 合成失败或超时")

        # Convert MP3 -> bright studio-EQ 48kHz 16-bit stereo WAV (native Magic Box I2S format)
        cmd = [
            FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mp3_path),
            "-af", BRIGHT_EQ_FILTER,
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(wav_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        try:
            mp3_path.unlink(missing_ok=True)
        except Exception:
            pass
        if res.returncode != 0 or not wav_path.exists():
            raise RuntimeError(f"ffmpeg 转码失败: {res.stderr.strip()}")
        return wav_path

    def check_health(self) -> dict[str, Any]:
        try:
            r = self._opener.open(f"{self.base_url}/health", timeout=2.0)
            data = json.loads(r.read().decode("utf-8"))
            self.daemon_online = True
            self.last_error = ""
            if "volume" in data and isinstance(data["volume"], int):
                self.volume = data["volume"]
            return {"online": True, "host": self.host, **data}
        except Exception as exc:
            self.daemon_online = False
            self.last_error = str(exc)
            return {"online": False, "host": self.host, "error": str(exc)}

    def set_volume(self, percent: int) -> dict[str, Any]:
        pct = max(20, min(100, int(percent)))
        self.volume = pct
        try:
            req = urllib.request.Request(
                f"{self.base_url}/volume",
                data=json.dumps({"percent": pct}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = self._opener.open(req, timeout=3.0)
            return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            self.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

    def set_led(self, preset: str) -> None:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/led",
                data=json.dumps({"preset": preset}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            self._opener.open(req, timeout=2.0).read()
        except Exception:
            pass

    def _upload_if_needed(self, key: str, wav_path: Path) -> None:
        if key in self._uploaded_keys:
            return
        wav_bytes = wav_path.read_bytes()
        req = urllib.request.Request(
            f"{self.base_url}/cache/{key}",
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        self._opener.open(req, timeout=8.0).read()
        self._uploaded_keys.add(key)

    def prewarm_phrases(self, phrases: list[str]) -> int:
        """Pre-synthesize and pre-upload all common booth stage phrases to Magic Box RAM disk."""
        ok_count = 0
        for text in phrases:
            try:
                key = self._cache_key(text)
                wav_path = self._synthesize_wav(text)
                if self.daemon_online or self.check_health().get("online"):
                    self._upload_if_needed(key, wav_path)
                ok_count += 1
            except Exception as exc:
                self.last_error = str(exc)
        return ok_count

    def speak(
        self,
        text: str,
        *,
        stage: str = "info",
        led: str = "cyan",
        interrupt: bool = True,
        async_play: bool = True,
    ) -> dict[str, Any]:
        """Play bright neural speech on Magic Box with <40ms latency when cached."""
        if not self.enabled or not text.strip():
            return {"ok": False, "reason": "disabled_or_empty"}

        self.last_text = text
        self.last_stage = stage

        def _do_speak() -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                key = self._cache_key(text)
                wav_path = self._synthesize_wav(text)
                # Fast path: play directly from Magic Box RAM cache if uploaded; else upload+play
                try:
                    if key not in self._uploaded_keys:
                        self._upload_if_needed(key, wav_path)
                    req = urllib.request.Request(
                        f"{self.base_url}/play",
                        data=json.dumps({
                            "key": key,
                            "led": led,
                            "interrupt": interrupt,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    resp = json.loads(self._opener.open(req, timeout=5.0).read().decode("utf-8"))
                    if not resp.get("ok") and resp.get("error") == "cache_miss":
                        self._uploaded_keys.discard(key)
                        self._upload_if_needed(key, wav_path)
                        resp = json.loads(self._opener.open(req, timeout=5.0).read().decode("utf-8"))
                    self.daemon_online = True
                    self.last_error = ""
                    self.last_latency_ms = int((time.perf_counter() - t0) * 1000)
                    return {"ok": True, "latency_ms": self.last_latency_ms, "key": key, **resp}
                except Exception as exc:
                    self.daemon_online = False
                    self.last_error = str(exc)
                    return {"ok": False, "error": str(exc)}
            except Exception as exc:
                self.last_error = str(exc)
                return {"ok": False, "error": str(exc)}

        if async_play:
            threading.Thread(target=_do_speak, name="magicbox-speak", daemon=True).start()
            return {"ok": True, "queued": True, "text": text, "stage": stage}
        return _do_speak()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "daemon_port": self.daemon_port,
            "daemon_online": self.daemon_online,
            "voice_preset": self.voice_preset,
            "voice_presets": VOICE_PRESETS,
            "volume": self.volume,
            "last_text": self.last_text,
            "last_stage": self.last_stage,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
            "cached_clips": len(self._uploaded_keys),
        }


# Standard booth staged speech templates (TuntunClaw 囤囤钳展台分阶段语音剧本)
def phrase_welcome() -> str:
    return "你好呀，欢迎来到地瓜机器人展台，我是囤囤钳！请问你想要点什么东西呢？"


def phrase_picking(item_name: str) -> str:
    return f"好的，囤囤钳正在为你取送{item_name}，请稍候一下！"


def phrase_delivered(item_name: str, remaining: int, unit: str, low_stock: bool) -> str:
    stock_part = (
        f"目前{item_name}还剩{remaining}{unit}，已经触发补货提醒，我帮你加入补货清单啦。"
        if low_stock
        else f"目前{item_name}还剩{remaining}{unit}，库存充足。"
    )
    return (
        f"{item_name}已经取送完毕！{stock_part}"
        "温馨提示，请把物资放回取货区，本展台物资仅供现场演示使用哦！"
    )


def phrase_recycling(item_name: str) -> str:
    return f"机械臂现在正在准备回收{item_name}，请稍微注意安全距离。"


def phrase_recycled(item_name: str, current_qty: int, unit: str) -> str:
    return f"{item_name}已经回收完成，当前库存恢复为{current_qty}{unit}。感谢体验囤囤钳，欢迎下次再来玩呀！"


def phrase_inventory_report(items: list[dict[str, Any]]) -> str:
    parts = [f"{i['name']}还剩{i['quantity']}{i['unit']}" for i in items]
    return "桌面盘点完成！目前" + "，".join(parts) + "。"
