"""MuJoCo physics + computer-vision digital twin engine for TuntunClaw.

Implements 1:1 tabletop kinematics & physics aligned with the real myCobot 280 Pi booth:
- Overhead vision detection (Red-tray exclusion + bright-cap isolation + wafer PCA)
- Symmetric shoulder-forward top-down IK with 180-deg parallel-jaw wrap
- Three selectable research grasp policies:
    * baseline          : fixed yaw=0 + raw bounding-box centroid
    * oriented          : overhead PCA jaw-yaw alignment + cap-head targeting
    * servo_closed_loop : oriented + wrist-cam hover residual servo + lift verification
- Single dedicated GL worker thread for lock-free offscreen rendering across FastAPI endpoints.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import mujoco
import numpy as np

from sim.tray_vision import detect_tray_skus_unified
from sim.workspace_analysis import load_model, solve_ik_topdown

SCENE_XML_PATH = Path(__file__).resolve().parent / "mycobot_280_scene.xml"

# Matches verified unobstructed front-gap park pose CLEAR_VIEW_POSE_DEG
HOME_Q = np.deg2rad([30.0, -35.0, -70.0, 15.0, 0.0, 0.0]).astype(np.float64)

SCAN_WAYPOINTS = [
    ("归位扫描起点", HOME_Q.copy()),
    ("俯瞰左侧红盘取料区", np.deg2rad([-25.0, -28.0, -82.0, 20.0, 0.0, 0.0])),
    ("居中平扫过渡区", np.deg2rad([20.0, -24.0, -86.0, 20.0, 0.0, 0.0])),
    ("俯瞰右侧核桃木放料区", np.deg2rad([62.0, -28.0, -80.0, 20.0, 0.0, 0.0])),
]

GRIPPER_OPEN = 0.08
GRIPPER_CLOSE_WAFER = -0.50
GRIPPER_CLOSE_CAP = -0.64
GRIPPER_CLOSE = -0.52

OBJECT_SPECS = [
    {
        "body": "obj_cuicuisha_1",
        "fj": "fj_cuicuisha_1",
        "key": "cuicuisha",
        "label": "雀巢脆脆鲨",
        "pos": [0.022, -0.170, 0.0128],
        "half_h": 0.0085,
        "close_q": GRIPPER_CLOSE_WAFER,
        "pick_z": 0.0095,
    },
    {
        "body": "obj_cuicuisha_2",
        "fj": "fj_cuicuisha_2",
        "key": "cuicuisha",
        "label": "雀巢脆脆鲨",
        "pos": [0.064, -0.168, 0.0128],
        "half_h": 0.0085,
        "close_q": GRIPPER_CLOSE_WAFER,
        "pick_z": 0.0095,
    },
    {
        "body": "obj_toothpaste_1",
        "fj": "fj_toothpaste_1",
        "key": "toothpaste",
        "label": "舒客+小牙膏",
        "pos": [0.096, -0.112, 0.0116],
        "half_h": 0.0075,
        "close_q": GRIPPER_CLOSE_CAP,
        "pick_z": 0.0085,
    },
    {
        "body": "obj_toothpaste_2",
        "fj": "fj_toothpaste_2",
        "key": "toothpaste",
        "label": "舒客+小牙膏",
        "pos": [0.132, -0.112, 0.0116],
        "half_h": 0.0075,
        "close_q": GRIPPER_CLOSE_CAP,
        "pick_z": 0.0085,
    },
    {
        "body": "obj_toothpaste_3",
        "fj": "fj_toothpaste_3",
        "key": "toothpaste",
        "label": "舒客+小牙膏",
        "pos": [0.168, -0.112, 0.0116],
        "half_h": 0.0075,
        "close_q": GRIPPER_CLOSE_CAP,
        "pick_z": 0.0085,
    },
]

# Reachable drop slots inside rear-right walnut wood tray (all inside empirical trace r=164..235mm)
TRAY_SLOTS = [
    [0.036, 0.162, 0.018],
    [0.168, 0.148, 0.018],
    [0.032, 0.212, 0.018],
    [0.105, 0.205, 0.018],
    [0.095, 0.172, 0.018],
]

# Delivery tray bounds in world frame (center=(0.091, 0.190), size=(0.30, 0.196))
WOOD_TRAY_X_MIN, WOOD_TRAY_X_MAX = -0.058, 0.240
WOOD_TRAY_Y_MIN, WOOD_TRAY_Y_MAX = 0.094, 0.286

# Legacy key aliases -> current booth SKUs
SKU_ALIASES = {
    "coffee": "cuicuisha",
    "wafer": "cuicuisha",
    "oreo": "toothpaste",
    "paste": "toothpaste",
}

HSV_CLASSES = {
    "cuicuisha": {"label": "雀巢脆脆鲨", "bgr": (42, 46, 175)},
    "toothpaste": {"label": "舒客+小牙膏", "bgr": (225, 175, 45)},
}


def _solve_ik_symmetric_topdown(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    jaw_az_rad: float = 0.0,
    q_ref: np.ndarray | None = None,
    max_iter: int = 150,
) -> tuple[bool, np.ndarray, float, float]:
    """Solve top-down IK avoiding rear-shoulder flips & selecting minimal J6 twist across 180-deg jaw symmetry."""
    j1_guess = math.atan2(float(target_pos[1]), float(target_pos[0]))
    seeds = [
        np.array([j1_guess, -0.55, -1.85, 0.85, 0.0, 0.0], dtype=np.float64),
        np.array([j1_guess, -0.28, -2.42, 1.15, 0.0, 0.0], dtype=np.float64),
    ]
    if q_ref is not None:
        seeds.insert(0, np.asarray(q_ref, dtype=np.float64).copy())

    best = None
    best_cost = 1e9
    for y_off in (0.0, math.pi):
        yaw = jaw_az_rad + y_off
        for seed in seeds:
            ok, q, ep, eo = solve_ik_topdown(
                model, data, target_pos, yaw_rad=yaw, q_init=seed, max_iter=max_iter, tol_pos=0.0012, tol_ori=0.05
            )
            # Reject backwards flipped shoulder solutions (J2 > +0.35 rad)
            if not ok or q[1] > 0.35:
                continue
            j6_pen = abs(float(q[5]))
            ref_pen = float(np.linalg.norm(q - q_ref)) if q_ref is not None else 0.0
            cost = ep * 100.0 + 0.15 * j6_pen + 0.05 * ref_pen
            if cost < best_cost:
                best_cost = cost
                best = (True, q, ep, eo)
    if best is not None:
        return best
    # Fallback unrestricted solve
    return solve_ik_topdown(model, data, target_pos, yaw_rad=jaw_az_rad, q_init=seeds[0], max_iter=max_iter)


class MyCobot280SimEngine:
    """Thread-safe MuJoCo digital twin with vision, physics pick/recycle, and hardware mirroring."""

    def __init__(self, on_state_change: Callable[[dict[str, Any]], None] | None = None):
        self.on_state_change = on_state_change
        self.model, self.data = load_model()
        self._ik_data = mujoco.MjData(self.model)
        self.width = 640
        self.height = 480

        self._gl_queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._stop_requested = False

        self.mode = "idle"  # idle | scanning | picking | recycling | homing | stopped | real_mirror
        self.strategy = "servo_closed_loop"  # baseline | oriented | servo_closed_loop
        self.progress = 0.0
        self.status_text = "高保真数字孪生待命中（真实 1:1 展台桌面同步完成）"
        self.last_detections: list[dict[str, Any]] = []
        self.last_scan_counts: dict[str, int] = {k: 0 for k in HSV_CLASSES}
        self.last_pick_metrics: dict[str, Any] = {}
        self.tray_next_slot = 0
        self._frames: dict[str, bytes] = {}

        # Cache IDs
        self._body_ids = {s["body"]: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s["body"]) for s in OBJECT_SPECS}
        self._fj_qadr = {
            s["body"]: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s["fj"])]
            for s in OBJECT_SPECS
        }
        self._fj_dadr = {
            s["body"]: self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s["fj"])]
            for s in OBJECT_SPECS
        }
        self._tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self._init_qpos = self.model.qpos0.copy()

        # Start dedicated OpenGL rendering & physics thread
        self._gl_thread = threading.Thread(target=self._gl_worker_loop, daemon=True)
        self._gl_thread.start()

        # Initialize robot at CLEAR_VIEW_POSE & run initial detection
        self.reset_scene()

    # ------------------------------------------------------------------
    # Dedicated GL Worker Thread
    # ------------------------------------------------------------------
    def _gl_worker_loop(self) -> None:
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        while True:
            fn, args, kwargs, done_ev, result_box = self._gl_queue.get()
            try:
                result_box["value"] = fn(*args, **kwargs)
            except Exception as exc:
                result_box["error"] = exc
            finally:
                done_ev.set()

    def _run_on_gl(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if threading.current_thread() is self._gl_thread:
            return fn(*args, **kwargs)
        done_ev = threading.Event()
        box: dict[str, Any] = {}
        self._gl_queue.put((fn, args, kwargs, done_ev, box))
        done_ev.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _render_raw_rgb(self, cam_name: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_name)
        return self._renderer.render().copy()

    def _refresh_all_frames(self, hud: bool = True) -> dict[str, np.ndarray]:
        cams = {}
        for cname in ("booth_overview", "overhead_top", "c931_view", "wrist_rgb"):
            rgb = self._render_raw_rgb(cname)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if hud and cname == "overhead_top":
                bgr = self._draw_overhead_hud(bgr)
            elif hud and cname == "booth_overview":
                cv2.putText(
                    bgr,
                    f"TuntunClaw 1:1 Twin | {self.mode.upper()} | {self.strategy}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (245, 250, 255),
                    2,
                )
            _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
            with self._lock:
                self._frames[cname] = enc.tobytes()
            cams[cname] = rgb
        return cams

    def get_frame(self, cam_name: str = "booth_overview") -> bytes:
        with self._lock:
            if cam_name in self._frames:
                return self._frames[cam_name]
            # fallback alias
            if cam_name == "top" and "overhead_top" in self._frames:
                return self._frames["overhead_top"]
            if cam_name == "c931" and "c931_view" in self._frames:
                return self._frames["c931_view"]
        return b""

    # ------------------------------------------------------------------
    # Camera Geometry & Red-Tray Exclusion Vision Detector
    # ------------------------------------------------------------------
    def _pixel_to_table_xy(self, u: float, v: float, plane_z: float = 0.012) -> tuple[float, float]:
        """Exact projection for overhead_top (pos=[0.225, 0.137, 0.96], xyaxes=[0 1 0 -1 0 0], fovy=48)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_top")
        cam_pos = self.model.cam_pos[cam_id]
        fovy = float(self.model.cam_fovy[cam_id])
        fy = (self.height / 2.0) / math.tan(math.radians(fovy / 2.0))
        fx = fy
        cx = self.width / 2.0
        cy = self.height / 2.0
        depth = float(cam_pos[2]) - plane_z
        # Image +u -> +Y_world, Image +v -> +X_world
        wx = float(cam_pos[0]) + ((v - cy) / fy) * depth
        wy = float(cam_pos[1]) + ((u - cx) / fx) * depth
        return wx, wy

    def _wrist_pixel_offset_xy(self, rgb_wrist: np.ndarray, target_key: str) -> tuple[float, float, bool]:
        """Closed-loop wrist visual servo estimator using actual MuJoCo wrist_rgb cam_xmat Jacobian!"""
        bgr = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        H, W = bgr.shape[:2]
        roi = np.zeros((H, W), dtype=np.uint8)
        roi[30:445, 90:560] = 255
        b_c, g_c, r_c = cv2.split(bgr.astype(np.int16))
        if target_key == "toothpaste":
            mask = ((v > 185) & (s < 48) & (roi > 0)).astype(np.uint8) * 255
        else:
            # Dark cocoa/maroon & warm caramel wafer wrapper excluding neutral black EVA pads & bright scarlet tray
            mask = ((r_c - b_c >= 10) & (r_c < 160) & (v < 155) & (roi > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb")
        cam_pos = self.data.cam_xpos[cam_id]
        cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)  # cols: X_right, Y_up, Z_back
        tcp_xy = self.data.site_xpos[self._tcp_id][:2]
        target_p = np.array([tcp_xy[0], tcp_xy[1], 0.013], dtype=np.float64)
        # Project TCP point at tabletop height into wrist_rgb to get exact reference pixel!
        rel_c = cam_mat.T @ (target_p - cam_pos)
        if rel_c[2] >= -1e-4:
            return 0.0, 0.0, False
        fovy = float(self.model.cam_fovy[cam_id])
        fy = (H / 2.0) / math.tan(math.radians(fovy / 2.0))
        ref_u = W / 2.0 + fy * (rel_c[0] / (-rel_c[2]))
        ref_v = H / 2.0 - fy * (rel_c[1] / (-rel_c[2]))

        yy, xx = np.ogrid[:H, :W]
        r_gate = 32.0 if target_key == "toothpaste" else 48.0
        gate_mask = (((xx - ref_u) ** 2 + (yy - ref_v) ** 2) <= (r_gate ** 2)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, gate_mask)
        n_l, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        best_c = None
        best_dist = 999.0
        for i in range(1, n_l):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 40 <= area <= 9000:
                pts = np.column_stack(np.where(labels == i))[:, ::-1]
                cu, cv = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                d_px = math.hypot(cu - ref_u, cv - ref_v)
                if d_px < 95.0 and d_px < best_dist:
                    best_dist = d_px
                    best_c = (cu, cv)
        if best_c is None:
            return 0.0, 0.0, False
        # Raycast detected blob pixel cleanly onto Z=0.013 table plane!
        cu, cv = best_c
        ray_cam = np.array([(cu - W / 2.0) / fy, -(cv - H / 2.0) / fy, -1.0], dtype=np.float64)
        ray_w = cam_mat @ ray_cam
        if abs(ray_w[2]) < 1e-4:
            return 0.0, 0.0, False
        t_hit = (0.013 - cam_pos[2]) / ray_w[2]
        hit_xy = cam_pos[:2] + t_hit * ray_w[:2]
        dx = float(np.clip(hit_xy[0] - tcp_xy[0], -0.005, 0.005))
        dy = float(np.clip(hit_xy[1] - tcp_xy[1], -0.005, 0.005))
        return dx, dy, True

    def _detect_items_from_rgb(self, rgb: np.ndarray) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Segment Red Picking Tray interior using the shared Real/Sim Red-Tray CV detector."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return detect_tray_skus_unified(
            bgr,
            lambda cu, cv: self._pixel_to_table_xy(cu, cv, plane_z=0.012),
            is_sim=True,
        )

    def _draw_overhead_hud(self, bgr: np.ndarray) -> np.ndarray:
        out = bgr.copy()
        for det in self.last_detections:
            x1, y1, x2, y2 = det["bbox"]
            cu, cv = int(det["pixel_center"][0]), int(det["pixel_center"][1])
            col = HSV_CLASSES.get(det["key"], {}).get("bgr", (0, 255, 180))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.circle(out, (cu, cv), 4, (255, 0, 255), -1)
            # Draw perpendicular jaw orientation line
            jaw_az = det.get("jaw_az_rad", 0.0)
            # world_az -> image vector (du=sin(az), dv=cos(az))
            du = int(round(14 * math.sin(jaw_az)))
            dv = int(round(14 * math.cos(jaw_az)))
            cv2.line(out, (cu - du, cv - dv), (cu + du, cv + dv), (0, 255, 255), 2)
            cv2.putText(out, det["key"], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        cv2.putText(out, "RED PICKING TRAY", (64, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 40, 210), 2)
        cv2.putText(out, "WALNUT DELIVERY TRAY", (252, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 225, 200), 2)
        return out

    # ------------------------------------------------------------------
    # Physics Motion Helpers
    # ------------------------------------------------------------------
    def _emit(self) -> None:
        if self.on_state_change:
            try:
                self.on_state_change(self.telemetry())
            except Exception:
                pass

    def _set_joints_immediate(self, q_arm: np.ndarray, grip: float) -> None:
        self.data.qpos[:6] = q_arm
        self.data.ctrl[:6] = q_arm
        self.data.qpos[6:12] = [grip, -grip, grip, -grip, -grip, grip]
        self.data.ctrl[6] = grip
        self.data.qvel[:12] = 0.0
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)

    def _move_smooth(
        self,
        q_target: np.ndarray,
        grip_target: float,
        steps: int = 18,
        dt: float = 0.018,
        substeps: int = 12,
        render: bool = True,
    ) -> bool:
        def _step_batch(q_cmd: np.ndarray, g_cmd: float) -> None:
            self.data.ctrl[:6] = q_cmd
            self.data.ctrl[6] = g_cmd
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.data)
            if render:
                self._refresh_all_frames(hud=True)

        q_start = self.data.qpos[:6].copy()
        g_start = float(self.data.ctrl[6])
        for i in range(1, steps + 1):
            if self._stop_requested:
                self.mode = "stopped"
                self.status_text = "急停生效：机械臂已保持悬停"
                self._emit()
                return False
            alpha = 0.5 - 0.5 * math.cos(math.pi * i / steps)
            q_cmd = (1.0 - alpha) * q_start + alpha * q_target
            g_cmd = (1.0 - alpha) * g_start + alpha * grip_target
            self._run_on_gl(_step_batch, q_cmd, g_cmd)
            self._emit()
            if render and dt > 0:
                time.sleep(dt)
        # Settle until PD actuators converge within ~0.7 deg so vertical descent & wrist servo never start mid-slew
        max_settle = 35 if not render else 8
        for _ in range(max_settle):
            self._run_on_gl(_step_batch, q_target, grip_target)
            if float(np.max(np.abs(self.data.qpos[:6] - q_target))) < 0.012:
                break
        return True

    def _body_world_pos(self, body_name: str) -> np.ndarray:
        return self.data.xpos[self._body_ids[body_name]].copy()

    def _items_in_picking_zone(self) -> list[dict[str, Any]]:
        out = []
        for s in OBJECT_SPECS:
            p = self._body_world_pos(s["body"])
            if p[1] < 0.045 and p[2] < 0.08:
                out.append({**s, "world_pos": p})
        return out

    def _items_in_delivery_tray(self) -> list[dict[str, Any]]:
        out = []
        for s in OBJECT_SPECS:
            p = self._body_world_pos(s["body"])
            if WOOD_TRAY_X_MIN <= p[0] <= WOOD_TRAY_X_MAX and WOOD_TRAY_Y_MIN <= p[1] <= WOOD_TRAY_Y_MAX:
                out.append({**s, "world_pos": p})
        return out

    # ------------------------------------------------------------------
    # Public Control Operations (Server + Research Evaluation)
    # ------------------------------------------------------------------
    def reset_scene(self) -> dict[str, Any]:
        with self._motion_lock:
            self._stop_requested = False

            def _do_reset() -> None:
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:] = self._init_qpos.copy()
                self.data.qvel[:] = 0.0
                self._set_joints_immediate(HOME_Q, GRIPPER_OPEN)
                cams = self._refresh_all_frames(hud=False)
                counts, dets = self._detect_items_from_rgb(cams["overhead_top"])
                self.last_scan_counts = counts
                self.last_detections = dets
                self._refresh_all_frames(hud=True)

            self._run_on_gl(_do_reset)
            self.tray_next_slot = 0
            self.mode = "idle"
            self.progress = 1.0
            self.status_text = "1:1 展台已复位（脆脆鲨 ×2，舒客小牙膏 ×3）"
            self._emit()
            return self.telemetry()

    def move_home(self) -> dict[str, Any]:
        with self._motion_lock:
            self._stop_requested = False
            self.mode = "homing"
            self.status_text = "机械臂归位至 CLEAR_VIEW_POSE..."
            self._emit()
            self._move_smooth(HOME_Q, GRIPPER_OPEN, steps=16, dt=0.015)
            self.mode = "idle"
            self.status_text = "已安全停泊至高空无遮挡俯瞰位"
            self._emit()
            return self.telemetry()

    def stop_motion(self) -> dict[str, Any]:
        self._stop_requested = True
        self.mode = "stopped"
        self.status_text = "急停指令已触发"
        self._emit()
        return self.telemetry()

    def jog_joint(self, joint_idx: int, delta_deg: float) -> dict[str, Any]:
        with self._motion_lock:
            self._stop_requested = False
            q = self.data.qpos[:6].copy()
            j = int(np.clip(joint_idx, 0, 5))
            q[j] = float(np.clip(q[j] + math.radians(delta_deg), self.model.jnt_range[j, 0], self.model.jnt_range[j, 1]))
            self._move_smooth(q, float(self.data.ctrl[6]), steps=8, dt=0.01)
            self._emit()
            return self.telemetry()

    def sync_real_joints_deg(self, angles_deg: list[float], gripper_0_to_100: float = 80.0) -> None:
        """Mirror real hardware joint angles & gripper aperture directly onto the 3D digital twin."""
        def _apply() -> None:
            q_rad = np.deg2rad(angles_deg[:6])
            for j in range(6):
                q_rad[j] = float(np.clip(q_rad[j], self.model.jnt_range[j, 0], self.model.jnt_range[j, 1]))
            ratio = float(np.clip(gripper_0_to_100 / 100.0, 0.0, 1.0))
            gj = -0.68 + ratio * (0.08 - (-0.68))
            self.data.qpos[:6] = q_rad
            self.data.ctrl[:6] = q_rad
            self.data.qpos[6:12] = [gj, -gj, gj, -gj, -gj, gj]
            self.data.ctrl[6] = gj
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self._refresh_all_frames(hud=True)

        self.mode = "real_mirror"
        self.status_text = "真实机械臂遥测同步中（Digital Twin 镜像）"
        self._run_on_gl(_apply)
        self._emit()

    def run_scene_scan(self) -> dict[str, Any]:
        with self._motion_lock:
            self._stop_requested = False
            self.mode = "scanning"
            self.progress = 0.0
            self.status_text = "正在扫描红盘取料区与核桃木放料区..."
            self._emit()

            for i, (label, q_wp) in enumerate(SCAN_WAYPOINTS):
                self.status_text = label
                self.progress = (i + 0.5) / len(SCAN_WAYPOINTS)
                if not self._move_smooth(q_wp, GRIPPER_OPEN, steps=14, dt=0.015):
                    return {"ok": False, "error": "stopped"}

            def _scan_detect() -> tuple[dict[str, int], list[dict[str, Any]]]:
                cams = self._refresh_all_frames(hud=False)
                counts, dets = self._detect_items_from_rgb(cams["overhead_top"])
                self.last_scan_counts = counts
                self.last_detections = dets
                self._refresh_all_frames(hud=True)
                return counts, dets

            counts, dets = self._run_on_gl(_scan_detect)
            self._move_smooth(HOME_Q, GRIPPER_OPEN, steps=12, dt=0.012)
            self.mode = "idle"
            self.progress = 1.0
            self.status_text = f"扫描完成：脆脆鲨 {counts.get('cuicuisha', 0)} 个，小牙膏 {counts.get('toothpaste', 0)} 支"
            self._emit()
            return {"ok": True, "counts": counts, "positions": dets}

    def run_pick_and_place(
        self,
        item_key: str,
        strategy: str | None = None,
        fast: bool = False,
    ) -> dict[str, Any]:
        """Execute pick-and-place in MuJoCo physics using selected research strategy."""
        norm_key = SKU_ALIASES.get(item_key, item_key)
        strat = strategy or self.strategy
        dt = 0.0 if fast else 0.016
        step_scale = 1.0  # preserve full physics substeps even when fast=True (only remove wall-clock sleep)

        with self._motion_lock:
            self._stop_requested = False
            self.mode = "picking"
            self.strategy = strat
            self.progress = 0.05
            self.status_text = f"[{strat}] 正在识别目标 ({norm_key})..."
            self._emit()

            # Fresh overhead vision scan from CLEAR_VIEW_POSE
            self._move_smooth(HOME_Q, GRIPPER_OPEN, steps=max(4, int(10 * step_scale)), dt=dt, render=not fast)

            def _scan() -> list[dict[str, Any]]:
                cams = self._refresh_all_frames(hud=False)
                counts, dets = self._detect_items_from_rgb(cams["overhead_top"])
                self.last_scan_counts = counts
                self.last_detections = dets
                return dets

            dets = self._run_on_gl(_scan)
            matches = [d for d in dets if d["key"] == norm_key]
            # Ground-truth body reference for post-run accuracy measurement ONLY
            gt_candidates = [s for s in self._items_in_picking_zone() if s["key"] == norm_key]
            if not matches or not gt_candidates:
                self.mode = "idle"
                self.status_text = f"未检测到可抓取的 {norm_key}"
                self._emit()
                return {"ok": False, "reason": "not_found", "item": norm_key, "strategy": strat}

            det = matches[0]
            # Match closest ground-truth body cleanly for evaluation telemetry
            gt_spec = min(
                gt_candidates,
                key=lambda g: math.hypot(g["world_pos"][0] - det["world_xy"][0], g["world_pos"][1] - det["world_xy"][1]),
            )
            body_name = gt_spec["body"]

            if strat == "baseline":
                # Baseline uses naive bounding-box centroid and zero yaw rotation (yaw=0)
                tx, ty = det["centroid_world_xy"]
                jaw_az = 0.0
            else:
                tx, ty = det["world_xy"]
                jaw_az = det["jaw_az_rad"]

            pick_z = gt_spec["pick_z"]
            close_q = gt_spec["close_q"]
            hover_z = 0.095
            pre_z = 0.095

            slot_idx = self.tray_next_slot % len(TRAY_SLOTS)
            px, py, pz = TRAY_SLOTS[slot_idx]
            drop_jaw_az = 0.0 if norm_key == "toothpaste" else math.pi / 2.0

            def _solve_all(cur_tx: float, cur_ty: float, cur_jaw: float):
                ok_h, q_hov, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([cur_tx, cur_ty, hover_z]), cur_jaw, HOME_Q)
                ok_p, q_pck, ep_p, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([cur_tx, cur_ty, pick_z]), cur_jaw, q_hov)
                ok_dh, q_dhi, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([px, py, pre_z]), drop_jaw_az, HOME_Q)
                ok_dl, q_dlo, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([px, py, pz]), drop_jaw_az, q_dhi)
                return ok_h and ok_p and ok_dh and ok_dl, q_hov, q_pck, q_dhi, q_dlo, ep_p

            ok_all, q_hover, q_pick, q_drop_hi, q_drop_lo, ik_err = self._run_on_gl(_solve_all, tx, ty, jaw_az)
            if not ok_all:
                self._run_on_gl(lambda: self._refresh_all_frames(hud=True))
                self.mode = "idle"
                self.status_text = f"目标 ({tx:.3f}, {ty:.3f}) 超出安全逆解工作空间"
                self._emit()
                return {"ok": False, "reason": "ik_unreachable", "item": norm_key, "strategy": strat}

            self.status_text = f"[{strat}] 前往高空对位点 ({tx*1000:.0f}, {ty*1000:.0f})mm..."
            self.progress = 0.25
            if not self._move_smooth(q_hover, GRIPPER_OPEN, steps=max(6, int(18 * step_scale)), dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            # Strategy 3: Closed-loop wrist visual servoing offset refinement at hover
            servo_dx = servo_dy = 0.0
            if strat == "servo_closed_loop":
                def _wrist_check():
                    rgb_w = self._render_raw_rgb("wrist_rgb")
                    return self._wrist_pixel_offset_xy(rgb_w, norm_key)

                dx_m, dy_m, locked = self._run_on_gl(_wrist_check)
                if locked and (abs(dx_m) > 0.0005 or abs(dy_m) > 0.0005):
                    if norm_key == "cuicuisha":
                        # Project along cross-bar jaw axis only so gripper shadow along length never pulls TCP off center
                        jx, jy = math.cos(jaw_az), math.sin(jaw_az)
                        proj = dx_m * jx + dy_m * jy
                        servo_dx, servo_dy = proj * jx, proj * jy
                    else:
                        servo_dx, servo_dy = dx_m, dy_m
                    tx_ref = float(np.clip(tx + servo_dx, tx - 0.005, tx + 0.005))
                    ty_ref = float(np.clip(ty + servo_dy, ty - 0.005, ty + 0.005))
                    ok_ref, q_h2, q_p2, q_dh2, q_dl2, _ = self._run_on_gl(_solve_all, tx_ref, ty_ref, jaw_az)
                    if ok_ref:
                        q_hover, q_pick, q_drop_hi, q_drop_lo = q_h2, q_p2, q_dh2, q_dl2
                        self._move_smooth(q_hover, GRIPPER_OPEN, steps=6, substeps=15, dt=dt * 0.5, render=not fast)

            self.status_text = f"[{strat}] 垂直下降对准 ({gt_spec['label']})"
            self.progress = 0.42
            if not self._move_smooth(q_pick, GRIPPER_OPEN, steps=16, substeps=15, dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            self.status_text = f"[{strat}] 泡沫柔性指垫闭合夹紧"
            self.progress = 0.55
            if not self._move_smooth(q_pick, close_q, steps=20, substeps=22, dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            self.status_text = f"[{strat}] 垂直抬升脱离红盘..."
            self.progress = 0.68
            if not self._move_smooth(q_hover, close_q, steps=18, substeps=15, dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            # Verify whether physics contact actually lifted the object cleanly (>= 0.045m Z)!
            p_lifted = self._run_on_gl(self._body_world_pos, body_name)
            lift_ok = bool(p_lifted[2] >= 0.045)

            self.status_text = f"[{strat}] 平稳转运至中央木质放料盘"
            self.progress = 0.80
            if not self._move_smooth(q_drop_hi, close_q, steps=max(6, int(18 * step_scale)), dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            if not self._move_smooth(q_drop_lo, close_q, steps=max(4, int(10 * step_scale)), dt=dt, render=not fast):
                return {"ok": False, "reason": "stopped"}

            self.progress = 0.90
            self._move_smooth(q_drop_lo, GRIPPER_OPEN, steps=max(4, int(8 * step_scale)), dt=dt, render=not fast)
            self._move_smooth(q_drop_hi, GRIPPER_OPEN, steps=max(4, int(10 * step_scale)), dt=dt, render=not fast)
            self._move_smooth(HOME_Q, GRIPPER_OPEN, steps=max(5, int(12 * step_scale)), dt=dt, render=not fast)

            p_final = self._run_on_gl(self._body_world_pos, body_name)
            landed_in_tray = bool(
                WOOD_TRAY_X_MIN <= p_final[0] <= WOOD_TRAY_X_MAX
                and WOOD_TRAY_Y_MIN <= p_final[1] <= WOOD_TRAY_Y_MAX
                and p_final[2] < 0.055
            )
            if landed_in_tray:
                self.tray_next_slot += 1

            def _post_scan():
                cams = self._refresh_all_frames(hud=False)
                c, d = self._detect_items_from_rgb(cams["overhead_top"])
                self.last_scan_counts = c
                self.last_detections = d
                self._refresh_all_frames(hud=True)

            self._run_on_gl(_post_scan)
            self.mode = "idle"
            self.progress = 1.0
            self.last_pick_metrics = {
                "item": norm_key,
                "body": body_name,
                "strategy": strat,
                "lift_ok": lift_ok,
                "landed_in_tray": landed_in_tray,
                "ik_pos_err_mm": round(ik_err * 1000.0, 2),
                "servo_trim_mm": [round(servo_dx * 1000.0, 2), round(servo_dy * 1000.0, 2)],
                "final_pos": [round(float(x), 4) for x in p_final],
            }
            self.status_text = (
                f"[{strat}] {gt_spec['label']} 已成功入盘！" if landed_in_tray else f"[{strat}] 物理抓取未入盘（lift={lift_ok}）"
            )
            self._emit()
            return {"ok": landed_in_tray, **self.last_pick_metrics}

    def run_recycle_from_tray(self, item_key: str | None = None, fast: bool = False) -> dict[str, Any]:
        """Recycle an item from the Walnut Delivery Tray back to an available slot in Red Picking Tray."""
        norm_key = SKU_ALIASES.get(item_key, item_key) if item_key else None
        dt = 0.0 if fast else 0.016
        step_scale = 1.0

        with self._motion_lock:
            self._stop_requested = False
            self.mode = "recycling"
            self.progress = 0.05
            self._emit()

            tray_items = self._run_on_gl(self._items_in_delivery_tray)
            if norm_key:
                tray_items = [t for t in tray_items if t["key"] == norm_key]
            if not tray_items:
                self.mode = "idle"
                self.status_text = "木盘中暂无待回收物品"
                self._emit()
                return {"ok": False, "reason": "empty_tray"}

            chosen = None
            for cand in tray_items:
                p_cur = cand["world_pos"]
                tx, ty = float(p_cur[0]), float(p_cur[1])
                ox, oy, _ = cand["pos"]
                hover_z = 0.095
                pick_z = cand["pick_z"] + 0.001
                close_q = cand["close_q"]
                bid = self._body_ids[cand["body"]]
                xmat = self.data.xmat[bid].reshape(3, 3)
                body_yaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
                cur_jaw = body_yaw + math.pi / 2.0

                def _solve_rec():
                    ok1, q_h1, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([tx, ty, hover_z]), cur_jaw, HOME_Q)
                    ok2, q_p1, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([tx, ty, pick_z]), cur_jaw, q_h1)
                    ok3, q_h2, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([ox, oy, hover_z]), 0.0, HOME_Q)
                    ok4, q_p2, _, _ = _solve_ik_symmetric_topdown(self.model, self._ik_data, np.array([ox, oy, pick_z + 0.010]), 0.0, q_h2)
                    return ok1 and ok2 and ok3 and ok4, q_h1, q_p1, q_h2, q_p2

                ok_all, q_h1, q_p1, q_h2, q_p2 = self._run_on_gl(_solve_rec)
                if ok_all:
                    chosen = (cand, close_q, q_h1, q_p1, q_h2, q_p2)
                    break

            if chosen is None:
                self.mode = "idle"
                return {"ok": False, "reason": "ik_unreachable"}

            target, close_q, q_h1, q_p1, q_h2, q_p2 = chosen

            self.status_text = f"回收 {target['label']} 返盘..."
            for q_wp, g_wp, stp, sub in (
                (q_h1, GRIPPER_OPEN, 14, 15),
                (q_p1, GRIPPER_OPEN, 14, 15),
                (q_p1, close_q, 20, 22),
                (q_h1, close_q, 16, 15),
                (q_h2, close_q, 18, 15),
                (q_p2, close_q, 12, 15),
                (q_p2, GRIPPER_OPEN, 8, 12),
                (q_h2, GRIPPER_OPEN, 10, 12),
                (HOME_Q, GRIPPER_OPEN, 12, 12),
            ):
                if not self._move_smooth(q_wp, g_wp, steps=stp, substeps=sub, dt=dt, render=not fast):
                    return {"ok": False, "reason": "stopped"}

            p_end = self._run_on_gl(self._body_world_pos, target["body"])
            recycled = bool(p_end[1] < 0.045)
            self._run_on_gl(lambda: self._refresh_all_frames(hud=True))
            self.mode = "idle"
            self.progress = 1.0
            self.status_text = f"{target['label']} 已完成回收入红盘" if recycled else "回收未到位"
            self._emit()
            return {"ok": recycled, "item": target["key"], "final_pos": p_end.tolist()}

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            q_deg = [round(float(math.degrees(x)), 2) for x in self.data.qpos[:6]]
            tcp_mm = [round(float(x) * 1000.0, 1) for x in self.data.site_xpos[self._tcp_id]]
            grip_val = float(self.data.ctrl[6])
            grip_open_pct = int(np.clip((grip_val - (-0.68)) / (GRIPPER_OPEN - (-0.68)) * 100.0, 0, 100))
            tray_list = [
                {
                    "body": t["body"],
                    "key": t["key"],
                    "label": t["label"],
                    "world_pos": [round(float(v), 4) for v in t["world_pos"]],
                }
                for t in self._items_in_delivery_tray()
            ]
            pct = int(round(float(self.progress) * 100)) if self.progress <= 1.0 else int(self.progress)
            return {
                "state": self.mode,
                "mode": self.mode,
                "strategy": self.strategy,
                "phase": self.status_text,
                "status": self.status_text,
                "progress": pct,
                "joints_deg": q_deg,
                "tcp_mm": tcp_mm,
                "gripper_open_pct": grip_open_pct,
                "scan_counts": dict(self.last_scan_counts),
                "detections": list(self.last_detections),
                "tray_items": tray_list,
                "last_pick_metrics": dict(self.last_pick_metrics),
            }
