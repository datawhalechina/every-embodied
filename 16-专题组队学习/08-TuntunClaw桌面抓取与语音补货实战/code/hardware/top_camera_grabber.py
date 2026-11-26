"""On-demand overhead Orbbec RGB grabber with automatic idle release.

Keeps Orbbec (cam index 3) open while live clients poll /api/camera/top.jpg,
then automatically releases the USB device handle after `idle_release_s` so
ROKAE HALM pipelines or the local Tk popup never hit a locked camera.
"""
from __future__ import annotations

import threading
import time
import urllib.request

import cv2


class TopOrbbecGrabber:
    """Thread-safe on-demand grabber for the overhead Orbbec Astra Pro 2 RGB stream."""

    def __init__(
        self,
        index: int = 3,
        width: int = 640,
        height: int = 480,
        idle_release_s: float = 4.0,
        fallback_url: str | None = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.idle_release_s = idle_release_s
        self.fallback_url = fallback_url
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._last_req_ts = 0.0
        self._last_frame_ts = 0.0
        self._last_jpg: bytes | None = None
        self._watchdog_started = False

    def _ensure_watchdog(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True
        t = threading.Thread(target=self._idle_loop, name="top-cam-idle-release", daemon=True)
        t.start()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(1.0)
            with self._lock:
                if self._cap is not None and (time.monotonic() - self._last_req_ts) > self.idle_release_s:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None

    def release(self) -> None:
        """Immediately release the underlying OpenCV capture handle."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def get_jpeg(self, max_age: float = 0.12) -> bytes | None:
        """Capture or return a cached JPEG frame from the overhead Orbbec camera."""
        now = time.monotonic()
        self._last_req_ts = now
        self._ensure_watchdog()

        if self._last_jpg is not None and (now - self._last_frame_ts) < max_age:
            return self._last_jpg

        with self._lock:
            now = time.monotonic()
            self._last_req_ts = now
            if self._last_jpg is not None and (now - self._last_frame_ts) < max_age:
                return self._last_jpg

            fresh_open = False
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = None
                    if self.fallback_url:
                        try:
                            with self._opener.open(self.fallback_url, timeout=2.5) as resp:
                                data = resp.read()
                                if data and len(data) > 1000:
                                    self._last_jpg = data
                                    self._last_frame_ts = time.monotonic()
                                    return self._last_jpg
                        except Exception:
                            pass
                    return self._last_jpg
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                fresh_open = True

            frame = None
            reads = 4 if fresh_open else 2
            for _ in range(reads):
                ok, f = self._cap.read()
                if ok and f is not None:
                    frame = f

            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    self._last_jpg = buf.tobytes()
                    self._last_frame_ts = time.monotonic()
            return self._last_jpg
