"""Standalone live dual-camera 1:1 alignment web server (port 8091).

Runs independently of any existing TuntunClaw process so we never disrupt
a running booth server, proxies C931 from 8090 (fallback to cam index 2),
and uses TopOrbbecGrabber with 4s idle auto-release so USB cams stay free.
"""
from __future__ import annotations

import asyncio
import sys
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hardware.top_camera_grabber import TopOrbbecGrabber

WEB_DIR = ROOT_DIR / "web"
PORT = 8091
C931_UPSTREAM = "http://127.0.0.1:8090/api/camera/c931.jpg"

top_cam = TopOrbbecGrabber(index=3, idle_release_s=4.0)
_last_c931_jpg: bytes | None = None
app = FastAPI(title="TuntunClaw Live Dual-Camera 1:1 Alignment", version="1.0.0")


def _fetch_c931() -> bytes | None:
    global _last_c931_jpg
    try:
        with urllib.request.urlopen(C931_UPSTREAM, timeout=6.0) as resp:
            data = resp.read()
            if data:
                _last_c931_jpg = data
            return data
    except Exception:
        return _last_c931_jpg


@app.get("/", response_class=HTMLResponse)
@app.get("/align", response_class=HTMLResponse)
async def align_page() -> FileResponse:
    return FileResponse(WEB_DIR / "align.html")


@app.get("/api/camera/top.jpg")
async def api_top() -> Response:
    jpg = await asyncio.to_thread(top_cam.get_jpeg)
    if not jpg:
        raise HTTPException(status_code=503, detail="Orbbec 顶部相机暂不可用")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/camera/c931.jpg")
async def api_c931() -> Response:
    jpg = await asyncio.to_thread(_fetch_c931)
    if not jpg:
        raise HTTPException(status_code=503, detail="C931 上游暂不可用")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.on_event("shutdown")
def on_shutdown() -> None:
    top_cam.release()


if __name__ == "__main__":
    print(f"Live dual-cam align server: http://127.0.0.1:{PORT}/align")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
