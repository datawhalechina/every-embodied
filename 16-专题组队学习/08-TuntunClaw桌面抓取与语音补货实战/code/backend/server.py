"""TuntunClaw (囤囤钳) Unified FastAPI Backend
Bridges:
  - 1:1 myCobot 280 Pi MuJoCo digital twin (scan + pick + physical recycle)
  - Real myCobot 280 Pi hardware (192.168.1.100, SSH key auth + wrist RGB camera)
  - D-Robotics RDK X5 Magic Box (<40ms studio-bright neural voice + WS2812B LED ring)
  - Local GESOBYTE C931 desk camera
  - SQLite persistent inventory & replenishment checklist
  - Mobile-first Web Mini-Program & Native WeChat Mini-Program (REST + SSE + WebSocket + QR)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import qrcode
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import TuntunDB
from hardware.magicbox_voice import (
    MagicBoxVoiceEngine,
    VOICE_PRESETS,
    phrase_delivered,
    phrase_inventory_report,
    phrase_picking,
    phrase_recycled,
    phrase_recycling,
    phrase_welcome,
)
from hardware.real_mycobot_bridge import RealMyCobot280Bridge
from hardware.real_pick_controller import CLEAR_VIEW_POSE_DEG, RealPickController
from hardware.top_camera_grabber import TopOrbbecGrabber
from sim.sim_engine import MyCobot280SimEngine


WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
DB_PATH = ROOT_DIR / "backend" / "tuntunclaw.db"
PORT = int(os.getenv("TUNTUN_PORT", "8090"))
PUBLIC_BASE_URL = os.getenv("TUNTUN_PUBLIC_URL", "").strip()


def detect_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


LAN_IP = detect_lan_ip()
ACCESS_URL = PUBLIC_BASE_URL or f"http://{LAN_IP}:{PORT}"


class C931CameraGrabber:
    """Crash-isolated frame grabber for local GESOBYTE C931 webcam.
    Runs OpenCV capture in a short-lived subprocess so DirectShow/USB conflicts
    with other lab apps can NEVER crash the main FastAPI server.
    """

    def __init__(self, index: int = 2) -> None:
        self.index = index
        self._lock = threading.Lock()
        self._last_ts = 0.0
        self._last_jpg: bytes | None = None
        self._tmp_path = ROOT_DIR / "backend" / "_c931_latest.jpg"

    def get_jpeg(self, max_age: float = 2.0) -> bytes | None:
        import subprocess

        now = time.time()
        if self._last_jpg and (now - self._last_ts) < max_age:
            return self._last_jpg
        with self._lock:
            now = time.time()
            if self._last_jpg and (now - self._last_ts) < max_age:
                return self._last_jpg
            code = (
                "import cv2, time, sys; "
                f"cap = cv2.VideoCapture({self.index}, cv2.CAP_DSHOW); "
                "cap.set(3, 960); cap.set(4, 540); "
                "f = None\n"
                "for _ in range(3):\n"
                "  r, img = cap.read()\n"
                "  if r and img is not None: f = img\n"
                "cap.release()\n"
                "if f is not None:\n"
                "  cv2.putText(f, f'REAL DESK | GESOBYTE C931 | {time.strftime(\"%H:%M:%S\")}', (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 248, 246), 2, cv2.LINE_AA)\n"
                f"  cv2.imwrite({str(self._tmp_path)!r}, f, [int(cv2.IMWRITE_JPEG_QUALITY), 82])\n"
            )
            try:
                subprocess.run(
                    [sys.executable, "-c", code],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    check=False,
                )
                if self._tmp_path.exists() and self._tmp_path.stat().st_size > 1000:
                    self._last_jpg = self._tmp_path.read_bytes()
                    self._last_ts = time.time()
            except Exception:
                pass
            return self._last_jpg


db = TuntunDB(DB_PATH)
c931 = C931CameraGrabber(index=int(os.getenv("C931_INDEX", "2")))
top_cam = TopOrbbecGrabber(
    index=int(os.getenv("TOP_CAM_INDEX", "3")),
    idle_release_s=4.0,
    fallback_url="http://127.0.0.1:8091/api/camera/top.jpg",
)
real_arm = RealMyCobot280Bridge(
    host=os.getenv("MYCOBOT_HOST", "192.168.1.100"),
    user=os.getenv("MYCOBOT_USER", "er"),
    password=os.getenv("MYCOBOT_PASS", ""),
)
voice = MagicBoxVoiceEngine(
    host=os.getenv("MAGICBOX_HOST", "192.168.1.101"),
    daemon_port=int(os.getenv("MAGICBOX_VOICE_PORT", "8095")),
    voice_preset=os.getenv("TUNTUN_VOICE_PRESET", "xiaoyi"),
)
EXECUTION_MODE = os.getenv("TUNTUN_EXEC_MODE", "simulation")  # simulation | real_hardware | dual_sync

_event_loop: asyncio.AbstractEventLoop | None = None
_sse_queues: set[asyncio.Queue[str]] = set()
_ws_clients: set[WebSocket] = set()
_clients_lock = threading.Lock()


def build_full_state() -> dict[str, Any]:
    snap = db.snapshot()
    ws_report_path = DOCS_DIR / "workspace_report.json"
    workspace = json.loads(ws_report_path.read_text(encoding="utf-8")) if ws_report_path.exists() else {}
    tel = sim.telemetry()
    if EXECUTION_MODE in ("real_hardware", "dual_sync"):
        tel["tray_items"] = list(real_ctrl.delivered_stack)
    return {
        "brand": "TuntunClaw",
        "brand_zh": "囤囤钳",
        "execution_mode": EXECUTION_MODE,
        "strategy": sim.strategy,
        "robot": tel,
        "hardware": {
            "connected": real_arm.connected,
            "host": real_arm.host,
            "user": real_arm.user,
            "last_error": real_arm.last_error,
        },
        "voice": voice.status(),
        "inventory": snap["items"],
        "replenishment": snap["replenishment"],
        "events": snap["events"],
        "summary": snap["summary"],
        "workspace": workspace.get("recommended_layout_mm", {}),
        "access_url": ACCESS_URL,
        "lan_ip": LAN_IP,
        "port": PORT,
        "server_time": snap["server_time"],
    }


def broadcast_state() -> None:
    if _event_loop is None or _event_loop.is_closed():
        return
    payload = json.dumps(build_full_state(), ensure_ascii=False)

    def _dispatch() -> None:
        with _clients_lock:
            stale_q = []
            for q in _sse_queues:
                try:
                    if q.full():
                        q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    stale_q.append(q)
            for q in stale_q:
                _sse_queues.discard(q)

            for ws in list(_ws_clients):
                asyncio.create_task(_safe_ws_send(ws, payload))

    try:
        _event_loop.call_soon_threadsafe(_dispatch)
    except Exception:
        pass


async def _safe_ws_send(ws: WebSocket, payload: str) -> None:
    try:
        await ws.send_text(payload)
    except Exception:
        with _clients_lock:
            _ws_clients.discard(ws)


sim = MyCobot280SimEngine(on_state_change=broadcast_state)
real_ctrl = RealPickController(real_arm, top_cam=top_cam, sim_engine=sim)

app = FastAPI(title="TuntunClaw (囤囤钳) · 地瓜黑客松智能物资与语音工作台", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/docs-assets", StaticFiles(directory=str(DOCS_DIR)), name="docs-assets")


def _prewarm_booth_voice_cache() -> None:
    """Background pre-synthesis & RAM-disk upload of all high-frequency booth lines for <40ms response."""
    voice.check_health()
    snap = db.snapshot()
    phrases = [phrase_welcome(), phrase_inventory_report(snap["items"])]
    for item in snap["items"]:
        name, unit = item["name"], item["unit"]
        phrases.append(phrase_picking(name))
        phrases.append(phrase_recycling(name))
        for rem in range(0, 3):
            phrases.append(phrase_delivered(name, rem, unit, rem <= item["threshold"]))
            phrases.append(phrase_recycled(name, rem + 1, unit))
    voice.prewarm_phrases(phrases)
    broadcast_state()


@app.on_event("startup")
async def on_startup() -> None:
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    db.update_from_scan(sim.last_scan_counts, {})
    threading.Thread(target=_prewarm_booth_voice_cache, name="voice-prewarm", daemon=True).start()


# ------------------------------------------------------------------
# Request Models
# ------------------------------------------------------------------
class PickRequest(BaseModel):
    item_key: str


class RecycleRequest(BaseModel):
    item_key: str | None = None


class JogRequest(BaseModel):
    joint_idx: int
    delta_deg: float


class ReplenishAddRequest(BaseModel):
    item_name: str
    qty: int = 2
    unit: str = "件"
    reason: str = "展台手动登记"


class ReplenishStatusRequest(BaseModel):
    status: str


class ThresholdRequest(BaseModel):
    threshold: int


class HardwareConfigRequest(BaseModel):
    host: str
    user: str = "er"
    password: str = ""


class VoiceSpeakRequest(BaseModel):
    text: str
    stage: str = "custom"
    led: str = "cyan"


class VoiceConfigRequest(BaseModel):
    enabled: bool | None = None
    voice_preset: str | None = None
    volume: int | None = None
    host: str | None = None


class RobotConfigRequest(BaseModel):
    execution_mode: str | None = None
    strategy: str | None = None


# ------------------------------------------------------------------
# Pages & QR Code
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/align", response_class=HTMLResponse)
async def align_page() -> FileResponse:
    return FileResponse(WEB_DIR / "align.html")


@app.get("/api/qrcode.png")
async def qrcode_png(url: str | None = None) -> Response:
    target = (url or ACCESS_URL).strip()
    qr = qrcode.QRCode(version=2, box_size=8, border=2)
    qr.add_data(target)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#17191c", back_color="#ffffff")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ------------------------------------------------------------------
# Live State, SSE & WebSocket
# ------------------------------------------------------------------
@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(build_full_state())


@app.get("/api/events")
async def api_events() -> StreamingResponse:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
    with _clients_lock:
        _sse_queues.add(q)

    async def gen():
        try:
            yield f"data: {json.dumps(build_full_state(), ensure_ascii=False)}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                _sse_queues.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    with _clients_lock:
        _ws_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(build_full_state(), ensure_ascii=False))
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": time.time()}))
    except WebSocketDisconnect:
        pass
    finally:
        with _clients_lock:
            _ws_clients.discard(websocket)


# ------------------------------------------------------------------
# Live Camera Feeds
# ------------------------------------------------------------------
def _get_top_cam_jpeg() -> bytes | None:
    import urllib.request
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8091/api/camera/top.jpg", timeout=4.5) as resp:
            data = resp.read()
            if data:
                return data
    except Exception:
        pass
    return top_cam.get_jpeg()


@app.get("/api/camera/{cam_name}.jpg")
async def camera_snapshot(cam_name: str) -> Response:
    if cam_name == "top":
        jpg = await asyncio.to_thread(_get_top_cam_jpeg)
        if not jpg:
            raise HTTPException(status_code=503, detail="Orbbec 顶部摄像头暂不可用")
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if cam_name == "c931":
        jpg = await asyncio.to_thread(c931.get_jpeg)
        if not jpg:
            jpg = sim.get_frame("c931_view")
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if cam_name == "real_wrist":
        jpg = await asyncio.to_thread(real_arm.capture_wrist_jpeg)
        if not jpg:
            raise HTTPException(status_code=503, detail="真机腕部相机未连接")
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if cam_name not in ("booth_overview", "wrist_rgb", "overhead_top", "c931_view", "magicbox_cam"):
        raise HTTPException(status_code=404, detail=f"未知相机: {cam_name}")
    jpg = sim.get_frame(cam_name)
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------
# Magic Box Low-Latency Bright Voice Endpoints
# ------------------------------------------------------------------
@app.get("/api/voice/status")
async def voice_status() -> JSONResponse:
    health = await asyncio.to_thread(voice.check_health)
    broadcast_state()
    return JSONResponse({**voice.status(), "health": health})


@app.post("/api/voice/welcome")
async def voice_welcome() -> JSONResponse:
    txt = phrase_welcome()
    res = await asyncio.to_thread(voice.speak, txt, stage="welcome", led="cyan", async_play=False)
    db.add_event("voice", "Magic Box 迎宾问候", txt)
    broadcast_state()
    return JSONResponse({"ok": True, "text": txt, "result": res})


@app.post("/api/voice/inventory")
async def voice_inventory() -> JSONResponse:
    snap = db.snapshot()
    txt = phrase_inventory_report(snap["items"])
    res = await asyncio.to_thread(voice.speak, txt, stage="inventory", led="cyan", async_play=False)
    db.add_event("voice", "Magic Box 语音播报库存", txt)
    broadcast_state()
    return JSONResponse({"ok": True, "text": txt, "result": res})


@app.post("/api/voice/speak")
async def voice_custom_speak(req: VoiceSpeakRequest) -> JSONResponse:
    res = await asyncio.to_thread(voice.speak, req.text, stage=req.stage, led=req.led, async_play=False)
    db.add_event("voice", "Magic Box 自定义播报", req.text)
    broadcast_state()
    return JSONResponse({"ok": True, "result": res})


@app.post("/api/voice/config")
async def voice_config(req: VoiceConfigRequest) -> JSONResponse:
    if req.enabled is not None:
        voice.enabled = req.enabled
    if req.host is not None and req.host.strip():
        voice.host = req.host.strip()
        voice._uploaded_keys.clear()
        await asyncio.to_thread(voice.check_health)
    if req.voice_preset is not None and req.voice_preset in VOICE_PRESETS:
        voice.voice_preset = req.voice_preset
        threading.Thread(target=_prewarm_booth_voice_cache, daemon=True).start()
    if req.volume is not None:
        await asyncio.to_thread(voice.set_volume, req.volume)
    broadcast_state()
    return JSONResponse({"ok": True, "voice": voice.status()})


# ------------------------------------------------------------------
# Robot Actions with 5-Stage Synchronized Magic Box Voice + LED
# ------------------------------------------------------------------
def _run_bg(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


def _execute_pick_with_voice(item_key: str, source: str = "TuntunClaw 抓取") -> dict[str, Any]:
    item_before = db.get_item(item_key)
    item_name = item_before["name"] if item_before else item_key
    # Stage 2: Instant speech (<40ms) right as the arm starts moving!
    voice.speak(phrase_picking(item_name), stage="picking", led="blue", async_play=True)
    db.add_event("voice", f"阶段2 · 开始取送播报 ({item_name})", phrase_picking(item_name))
    broadcast_state()

    if EXECUTION_MODE in ("real_hardware", "dual_sync"):
        res = real_ctrl.run_real_pick(item_key, dual_sync=(EXECUTION_MODE == "dual_sync"))
        if not res.get("ok"):
            db.add_event("warning", f"真机抓取安全拦截 ({item_name})", str(res.get("reason") or res))
            broadcast_state()
            return res
    else:
        res = sim.run_pick_and_place(item_key, strategy=sim.strategy)
    if res.get("ok"):
        db.record_delivery(item_key, source=source)
        item_after = db.get_item(item_key)
        if item_after:
            # Stage 3: Delivered + remaining stock + demo-only notice!
            deliv_txt = phrase_delivered(
                item_after["name"],
                item_after["quantity"],
                item_after["unit"],
                item_after["low_stock"],
            )
            led_col = "yellow" if item_after["low_stock"] else "green"
            voice.speak(deliv_txt, stage="delivered", led=led_col, async_play=True)
            db.add_event("voice", f"阶段3 · 取送完毕与余量播报 ({item_name})", deliv_txt)
    else:
        db.add_event("warning", f"抓取未执行 ({item_key})", res.get("error") or res.get("reason") or "")
    broadcast_state()
    return res


def _execute_recycle_with_voice(item_key: str | None = None) -> dict[str, Any]:
    is_real = EXECUTION_MODE in ("real_hardware", "dual_sync")
    tray = list(real_ctrl.delivered_stack) if is_real else sim.telemetry().get("tray_items", [])
    if not tray:
        db.add_event("warning", "取货区暂无待回收物资", "请先执行一次取送演示")
        broadcast_state()
        return {"ok": False, "error": "取货区暂无待回收物资"}

    target = tray[-1]
    if item_key:
        for t in reversed(tray):
            if t["key"] == item_key:
                target = t
                break
    item_name = target["label"]

    # Stage 4: Preparing to recycle!
    rec_start_txt = phrase_recycling(item_name)
    voice.speak(rec_start_txt, stage="recycling", led="warm", async_play=True)
    db.add_event("voice", f"阶段4 · 准备回收物资播报 ({item_name})", rec_start_txt)
    broadcast_state()

    if is_real:
        res = real_ctrl.run_real_recycle(target["key"], dual_sync=(EXECUTION_MODE == "dual_sync"))
    else:
        res = sim.run_recycle_from_tray(target["key"])
    if res.get("ok"):
        db.record_recycle(target["key"])
        item_after = db.get_item(target["key"])
        qty = item_after["quantity"] if item_after else 1
        unit = item_after["unit"] if item_after else "件"
        # Stage 5: Recycle completed + goodbye!
        rec_done_txt = phrase_recycled(item_name, qty, unit)
        voice.speak(rec_done_txt, stage="recycled", led="green", async_play=True)
        db.add_event("voice", f"阶段5 · 回收完成欢送播报 ({item_name})", rec_done_txt)
    else:
        db.add_event("warning", f"回收未完成 ({item_name})", str(res.get("reason") or res.get("error") or res))
    broadcast_state()
    return res


@app.post("/api/robot/config")
async def robot_config(req: RobotConfigRequest) -> JSONResponse:
    global EXECUTION_MODE
    if req.execution_mode in ("simulation", "real_hardware", "dual_sync"):
        EXECUTION_MODE = req.execution_mode
    if req.strategy in ("servo_closed_loop", "oriented", "baseline"):
        sim.strategy = req.strategy
    db.add_event("system", "切换运行策略/模式", f"模式={EXECUTION_MODE}, 抓取策略={sim.strategy}")
    broadcast_state()
    return JSONResponse({"ok": True, "execution_mode": EXECUTION_MODE, "strategy": sim.strategy})


@app.post("/api/robot/scan")
async def robot_scan() -> JSONResponse:
    def _job() -> None:
        voice.speak("好的，囤囤钳正在扫描桌面物资。", stage="scanning", led="cyan", async_play=True)
        if EXECUTION_MODE in ("real_hardware", "dual_sync"):
            res = real_ctrl.run_real_scan(dual_sync=(EXECUTION_MODE == "dual_sync"))
            if res.get("ok"):
                pos_by_key: dict[str, list[list[float]]] = {}
                for det in res.get("detections", []):
                    pos_by_key.setdefault(det["key"], []).append(det.get("mc_mm", []))
                snap = db.update_from_scan(res["counts"], pos_by_key)
                voice.speak(phrase_inventory_report(snap["items"]), stage="inventory", led="green", async_play=True)
        else:
            res = sim.run_scene_scan()
            if res.get("ok"):
                snap = db.update_from_scan(res["counts"], res["positions"])
                voice.speak(phrase_inventory_report(snap["items"]), stage="inventory", led="green", async_play=True)
        broadcast_state()

    _run_bg(_job)
    return JSONResponse({"ok": True, "message": "已启动桌面场景视觉扫描与语音盘点"})


@app.post("/api/robot/pick")
async def robot_pick(req: PickRequest) -> JSONResponse:
    _run_bg(_execute_pick_with_voice, req.item_key, "TuntunClaw 抓取递送")
    return JSONResponse({"ok": True, "message": f"已派发抓取递送与语音播报: {req.item_key}"})


@app.post("/api/robot/recycle")
async def robot_recycle(req: RecycleRequest) -> JSONResponse:
    _run_bg(_execute_recycle_with_voice, req.item_key)
    return JSONResponse({"ok": True, "message": "已启动取货区物资自动回收与语音播报"})


@app.post("/api/robot/deliver_demo")
@app.post("/api/robot/full_booth_show")
async def robot_full_booth_show() -> JSONResponse:
    """Complete 5-stage hackathon booth showcase:
    1. Welcome speech -> 2. Pick available item + speech -> 3. Delivered + stock + demo-only notice ->
    4. Pause for visitor inspection -> 5. Recycle back to picking zone + farewell speech!
    """
    def _job() -> None:
        db.add_event("demo", "启动 TuntunClaw 五阶段展台全流程", "迎宾 → 取送 → 报库存/演示提示 → 回收 → 欢送")
        broadcast_state()

        # Stage 1: Welcome
        voice.speak(phrase_welcome(), stage="welcome", led="cyan", async_play=False)
        time.sleep(3.8)

        # Stage 2 & 3: Pick available item + announce delivery & remaining count
        if EXECUTION_MODE in ("real_hardware", "dual_sync"):
            scan_r = real_ctrl.run_real_scan(dual_sync=(EXECUTION_MODE == "dual_sync"))
            counts = scan_r.get("counts", {}) if scan_r.get("ok") else {}
        else:
            counts = sim.last_scan_counts
        pick_key = next((k for k in ("cuicuisha", "toothpaste", "coffee", "oreo") if counts.get(k, 0) > 0), None)
        if pick_key:
            res_p = _execute_pick_with_voice(pick_key, source="五阶段展台演示")
            if res_p.get("ok"):
                # Wait for Stage 3 voice & simulated visitor inspection
                time.sleep(8.5)
                # Stage 4 & 5: Recycle back to picking zone & farewell
                _execute_recycle_with_voice(pick_key)
        broadcast_state()

    _run_bg(_job)
    return JSONResponse({"ok": True, "message": "已启动 TuntunClaw 五阶段展台语音+机械臂全流程"})


@app.post("/api/robot/home")
async def robot_home() -> JSONResponse:
    def _job() -> None:
        if EXECUTION_MODE in ("real_hardware", "dual_sync"):
            real_ctrl.move_to_clear_view(dual_sync=(EXECUTION_MODE == "dual_sync"))
        else:
            sim.move_home()
        broadcast_state()

    _run_bg(_job)
    return JSONResponse({"ok": True})


@app.post("/api/robot/stop")
async def robot_stop() -> JSONResponse:
    tel = sim.stop_motion()
    if EXECUTION_MODE in ("real_hardware", "dual_sync"):
        await asyncio.to_thread(real_ctrl.stop_motion)
    voice.set_led("red")
    db.add_event("stop", "触发急停指令", "已立即中止当前机械臂轨迹")
    broadcast_state()
    return JSONResponse({"ok": True, "robot": tel})


@app.post("/api/robot/jog")
async def robot_jog(req: JogRequest) -> JSONResponse:
    _run_bg(sim.jog_joint, req.joint_idx, req.delta_deg)
    return JSONResponse({"ok": True})


@app.post("/api/inventory/restock")
async def inventory_restock() -> JSONResponse:
    def _job() -> None:
        sim.reset_scene()
        db.reset_scene_stock()
        voice.speak("展台取料区已完成补货复位，随时可以开始新一轮体验！", stage="restock", led="green", async_play=True)
        broadcast_state()

    _run_bg(_job)
    return JSONResponse({"ok": True, "message": "已一键补货并复位展台桌面"})


# ------------------------------------------------------------------
# Replenishment & Inventory Management
# ------------------------------------------------------------------
@app.post("/api/replenishment/add")
async def replenishment_add(req: ReplenishAddRequest) -> JSONResponse:
    if not req.item_name.strip():
        raise HTTPException(status_code=400, detail="物品名称不能为空")
    db.add_custom_replenishment(req.item_name, req.qty, req.unit, req.reason)
    broadcast_state()
    return JSONResponse({"ok": True, "state": build_full_state()})


@app.post("/api/replenishment/{rep_id}/status")
async def replenishment_status(rep_id: int, req: ReplenishStatusRequest) -> JSONResponse:
    if req.status not in ("pending", "ordered", "done"):
        raise HTTPException(status_code=400, detail="无效状态")
    db.set_replenishment_status(rep_id, req.status)
    broadcast_state()
    return JSONResponse({"ok": True, "state": build_full_state()})


@app.post("/api/inventory/{item_key}/threshold")
async def inventory_threshold(item_key: str, req: ThresholdRequest) -> JSONResponse:
    db.update_threshold(item_key, req.threshold)
    broadcast_state()
    return JSONResponse({"ok": True, "state": build_full_state()})


# ------------------------------------------------------------------
# Real myCobot 280 Pi Hardware Probe & Sync
# ------------------------------------------------------------------
@app.get("/api/hardware/probe")
async def hardware_probe() -> JSONResponse:
    info = await asyncio.to_thread(real_arm.probe)
    broadcast_state()
    return JSONResponse(info)


@app.get("/api/hardware/safety")
async def hardware_safety() -> JSONResponse:
    info = await asyncio.to_thread(real_ctrl.check_pre_flight_safety)
    return JSONResponse(info)


@app.get("/api/hardware/scan")
async def hardware_scan_readonly() -> JSONResponse:
    info = await asyncio.to_thread(real_ctrl.run_real_scan, False)
    return JSONResponse(info)


@app.post("/api/hardware/clear_view")
async def hardware_clear_view() -> JSONResponse:
    info = await asyncio.to_thread(real_ctrl.move_to_clear_view, True, 18)
    broadcast_state()
    return JSONResponse(info)


@app.post("/api/hardware/connect")
async def hardware_connect(req: HardwareConfigRequest) -> JSONResponse:
    real_arm.host = req.host.strip()
    real_arm.user = req.user.strip() or "er"
    real_arm.password = req.password
    info = await asyncio.to_thread(real_arm.probe)
    if info.get("connected"):
        db.add_event("hardware", f"已连接真机 myCobot 280 Pi ({real_arm.host})", f"当前关节角: {info.get('angles')}")
    broadcast_state()
    return JSONResponse(info)


if __name__ == "__main__":
    import uvicorn

    print(f"============================================================")
    print(f"  TuntunClaw (囤囤钳) · 地瓜机器人黑客松展台服务启动")
    print(f"  本机访问: http://127.0.0.1:{PORT}/")
    print(f"  手机扫码: {ACCESS_URL}/")
    print(f"  MagicBox: http://{voice.host}:{voice.daemon_port}/health")
    print(f"============================================================")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", access_log=False)
