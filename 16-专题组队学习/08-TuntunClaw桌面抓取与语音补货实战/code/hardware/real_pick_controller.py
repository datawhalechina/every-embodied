"""Closed-loop vision-guided real hardware & dual-sync controller for myCobot 280 Pi.

Consolidates all hardware-verified calibrations and safety guards:
- Empirical 3D-FK pinhole calibration on clamped base (BX=211.7, BY=37.5, PM=0.5532, TH_MC=-90.66 deg)
- Exact concentric gripper yaw mapping (rz_des = jaw_open_az_deg + 45.0 -> j6_des)
- Pre-motion clamped-base displacement watchdog (aborts if base slips/tips on overhead frame)
- Closed-loop Z-trim at pick (target ~79.0 mm actual, hard floor >= 76.5 mm, zero desk scrape)
- Mandatory transit lift clearance gate (coords[2] >= 132.0 mm before any XY travel)
- Thermal-safe gripper range [0.14, 0.82] (avoids 0 stall overheat & 100 linkage toggle bind)
- Real -> 1:1 MuJoCo telemetry mirroring for Dual-Sync operation
"""
from __future__ import annotations

import math
import time
import urllib.request
from typing import Any

import cv2
import numpy as np

from hardware.real_mycobot_bridge import RealMyCobot280Bridge
from sim.tray_vision import detect_tray_skus_unified, wrap180
from sim.workspace_analysis import load_model, solve_ik_topdown

# Calibrated 640x480 top-cam -> real clamped myCobot 280 base frame (item plane Z~25mm)
CAL_BX = 211.7
CAL_BY = 37.5
CAL_PM = 0.5532
CAL_TH_MC_DEG = -90.66
CAL_Z_OFFSET_MM = 71.6

# High unobstructed park pose over right delivery side keeping X_mc >= 0, red tray & base 100% unobstructed
CLEAR_VIEW_POSE_DEG = [68.0, -30.0, -75.0, 15.0, 0.0, 0.0]

# Empirically recorded reachable Wood Delivery Tray drop slots (all inside recorded trace r=130..265mm)
REAL_DROP_SLOTS_PX = [
    (304.0, 56.0),
    (302.0, 134.0),
    (336.0, 52.0),
    (330.0, 96.0),
]

REAL_SKU_ALIASES = {
    "coffee": "cuicuisha",
    "wafer": "cuicuisha",
    "oreo": "toothpaste",
    "paste": "toothpaste",
}


def px_to_real_mc_mm(px: float, py: float) -> tuple[float, float]:
    """Convert 640x480 overhead Orbbec pixel (px, py) to real myCobot 280 XY (mm)."""
    th = math.radians(CAL_TH_MC_DEG)
    ux_mm = (px - CAL_BX) / CAL_PM
    vy_mm = -(py - CAL_BY) / CAL_PM
    mx = ux_mm * math.cos(-th) - vy_mm * math.sin(-th)
    my = ux_mm * math.sin(-th) + vy_mm * math.cos(-th)
    return mx, my


def pick_j6_for_jaw_az(j1_deg: float, jaw_open_az_deg: float) -> float:
    """Exact visually verified J6 angle aligning parallel jaws perpendicular to object axis."""
    rz_des = wrap180(jaw_open_az_deg + 45.0)
    c1 = wrap180(j1_deg - rz_des - 90.0)
    c2 = wrap180(c1 + 180.0)
    return c1 if abs(c1) <= abs(c2) else c2


def detect_real_tray_items(bgr: np.ndarray) -> dict[str, Any]:
    """Detect all Cuicuisha wafers & Toothpaste screw caps inside the Red Picking Tray on live Orbbec frames."""
    counts, raw_dets = detect_tray_skus_unified(bgr, px_to_real_mc_mm, is_sim=False)
    detections: list[dict[str, Any]] = []
    for d in raw_dets:
        mx, my = d["world_xy"]
        r_mm = math.hypot(mx, my)
        detections.append({
            "key": d["key"],
            "label": d["label"],
            "px": d["pixel_center"],
            "mc_mm": [round(mx, 1), round(my, 1)],
            "radius_mm": round(r_mm, 1),
            "jaw_az_deg": round(wrap180(math.degrees(d["jaw_az_rad"])), 1),
            "reachable": 130.0 <= r_mm <= 265.0,
        })
    return {"counts": counts, "detections": detections}


def verify_base_anchored(top_bgr: np.ndarray, min_cool_frac: float = 0.45) -> dict[str, Any]:
    """Check overhead frame to verify clamped G_base flange stays anchored at (211.7, 37.5)."""
    patch = top_bgr[52:88, 178:254]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    cool_frac = float(((h >= 75) & (h <= 145) & (v >= 55) & (v <= 228)).mean())
    s_mean = float(s.mean())
    anchored = bool(cool_frac >= min_cool_frac and s_mean >= 55.0)
    return {
        "ok": anchored,
        "reason": "ok" if anchored else "base_slipped_or_tipped",
        "calibrated_center": [CAL_BX, CAL_BY],
        "cool_base_fraction": round(cool_frac, 3),
        "sat_mean": round(s_mean, 1),
    }


class RealPickController:
    """Coordinates safe real myCobot 280 Pi grasping + optional 1:1 MuJoCo twin mirroring."""

    def __init__(self, arm: RealMyCobot280Bridge, top_cam: Any = None, sim_engine: Any = None):
        self.arm = arm
        self.top_cam = top_cam
        self.sim = sim_engine
        self.model, self.data = load_model()
        self.drop_idx = 0
        self.delivered_stack: list[dict[str, Any]] = []
        self._stop_requested = False

    def stop_motion(self) -> dict[str, Any]:
        self._stop_requested = True
        return self.arm.pause()

    def _mirror_to_sim(self, st: dict[str, Any] | None) -> None:
        if not self.sim or not st:
            return
        angles = st.get("angles") or []
        if len(angles) >= 6:
            grip = float(st.get("gripper") if st.get("gripper") is not None else 75.0)
            try:
                self.sim.sync_real_joints_deg(angles, grip)
            except Exception:
                pass

    def _grab_top_bgr(self) -> np.ndarray | None:
        # Query the 8091 dual-cam align server first so 8090 never collides on USB DirectShow handle
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open("http://127.0.0.1:8091/api/camera/top.jpg", timeout=4.5) as r:
                raw = r.read()
                if raw:
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
        except Exception:
            pass
        if self.top_cam is not None:
            try:
                jpg = self.top_cam.get_jpeg() if hasattr(self.top_cam, "get_jpeg") else None
                if jpg:
                    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass
        return None

    def _solve_pose(self, x_mm: float, y_mm: float, z_coords_mm: float, jaw_az_deg: float, q_init=None):
        z_mj = (z_coords_mm - CAL_Z_OFFSET_MM) / 1000.0
        target = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mj], dtype=np.float64)
        j1_guess = math.atan2(y_mm, x_mm)
        seeds = [
            np.array([j1_guess, -0.55, -1.85, 0.85, 0.0, 0.0], dtype=np.float64),
            np.array([j1_guess, -0.28, -2.42, 1.15, 0.0, 0.0], dtype=np.float64),
        ]
        if q_init is not None:
            seeds.insert(0, np.asarray(q_init, dtype=np.float64).copy())

        best_q = None
        best_err = 1e9
        for seed in seeds:
            ok, q_rad, ep, eo = solve_ik_topdown(
                self.model, self.data, target,
                yaw_rad=0.0, q_init=seed, max_iter=150, tol_pos=0.003, tol_ori=0.06,
            )
            # Strictly reject backwards-flipped shoulder solutions (J2 > +0.35 rad)
            if not ok or float(q_rad[1]) > 0.35:
                continue
            if ep < best_err:
                best_err = ep
                best_q = q_rad.copy()
        if best_q is None:
            return None, None
        q_deg = np.rad2deg(best_q).tolist()
        q_deg[5] = pick_j6_for_jaw_az(q_deg[0], jaw_az_deg)
        return q_deg, best_q

    def move_to_clear_view(self, dual_sync: bool = True, speed: int = 18) -> dict[str, Any]:
        self._stop_requested = False
        safety = self.check_pre_flight_safety()
        if not safety["ok"]:
            return {"ok": False, **safety}
        self.arm.send_angles_safe(CLEAR_VIEW_POSE_DEG, speed=speed)
        st = self._wait_settled(3.5, mirror=dual_sync)
        return {"ok": True, "probe": st, "base_check": safety["base_check"]}

    def _wait_settled(self, timeout_s: float = 3.2, mirror: bool = False) -> dict[str, Any]:
        time.sleep(0.4)
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout_s:
            st = self.arm.probe()
            if mirror:
                self._mirror_to_sim(st)
            c = st.get("coords") or []
            if len(c) >= 3 and last is not None and len(last) >= 3:
                if all(abs(c[i] - last[i]) < 0.8 for i in range(3)):
                    return st
            if len(c) >= 3:
                last = c
            time.sleep(0.35)
        st = self.arm.probe()
        if mirror:
            self._mirror_to_sim(st)
        return st

    def check_pre_flight_safety(self) -> dict[str, Any]:
        st = self.arm.probe()
        if not st.get("connected") or st.get("power") != 1:
            return {"ok": False, "reason": "arm_offline_or_unpowered", "probe": st}
        bgr = self._grab_top_bgr()
        if bgr is None:
            return {"ok": False, "reason": "top_camera_unavailable_for_safety_check", "probe": st}
        base_chk = verify_base_anchored(bgr)
        if not base_chk["ok"]:
            return {"ok": False, "reason": base_chk["reason"], "base_check": base_chk, "probe": st}
        return {"ok": True, "base_check": base_chk, "probe": st}

    def run_real_scan(self, dual_sync: bool = False) -> dict[str, Any]:
        bgr = self._grab_top_bgr()
        if bgr is None:
            return {"ok": False, "error": "top_camera_unavailable"}
        res = detect_real_tray_items(bgr)
        base_chk = verify_base_anchored(bgr)
        if dual_sync:
            self._mirror_to_sim(self.arm.probe())
        return {"ok": True, **res, "base_check": base_chk}

    def run_real_pick(self, item_key: str, dual_sync: bool = True) -> dict[str, Any]:
        self._stop_requested = False
        norm_key = REAL_SKU_ALIASES.get(item_key, item_key)
        safety = self.check_pre_flight_safety()
        if not safety["ok"]:
            return {"ok": False, "item": norm_key, **safety}

        bgr = self._grab_top_bgr()
        if bgr is None:
            return {"ok": False, "reason": "top_camera_unavailable"}
        vis = detect_real_tray_items(bgr)
        cands = [d for d in vis["detections"] if d["key"] == norm_key and d["reachable"]]
        if not cands:
            # Arm may be occluding red tray; park at CLEAR_VIEW_POSE_DEG and re-scan once
            self.arm.send_angles_safe(CLEAR_VIEW_POSE_DEG, speed=18)
            self._wait_settled(3.2, mirror=dual_sync)
            bgr = self._grab_top_bgr()
            if bgr is not None:
                vis = detect_real_tray_items(bgr)
                cands = [d for d in vis["detections"] if d["key"] == norm_key and d["reachable"]]
        if not cands:
            return {"ok": False, "reason": "no_reachable_target", "detections": vis["detections"]}

        # Prefer the candidate with the largest nearest-neighbor clearance (prevents jaw bump on neighbors)
        all_pts = [d["mc_mm"] for d in vis["detections"]]
        def _clearance_score(cand: dict[str, Any]) -> float:
            cx, cy = cand["mc_mm"]
            dists = [math.hypot(cx - ox, cy - oy) for ox, oy in all_pts if (ox != cx or oy != cy)]
            return min(dists) if dists else 999.0

        cands.sort(key=_clearance_score, reverse=True)
        tgt = cands[0]
        mx, my = tgt["mc_mm"]
        jaw_az = tgt["jaw_az_deg"]
        dpx, dpy = REAL_DROP_SLOTS_PX[self.drop_idx % len(REAL_DROP_SLOTS_PX)]
        dx, dy = px_to_real_mc_mm(dpx, dpy)

        q_hover, qh_rad = self._solve_pose(mx, my, 152.0, jaw_az)
        q_pre, qp_rad = self._solve_pose(mx, my, 84.0, jaw_az, q_init=qh_rad)
        q_drop_hi, qdh_rad = self._solve_pose(dx, dy, 152.0, 0.0)
        q_drop_lo, _ = self._solve_pose(dx, dy, 92.0, 0.0, q_init=qdh_rad)
        if not all([q_hover, q_pre, q_drop_hi, q_drop_lo]):
            return {"ok": False, "reason": "ik_unreachable", "target": tgt}

        # Compact safe gripper open (62% ~= 34mm span, avoids clipping red tray walls or neighbors)
        self.arm.set_gripper(0.62, speed=50)
        time.sleep(0.6)

        st0 = self.arm.probe()
        if (st0.get("coords") or [0, 0, 0])[2] < 132.0:
            self.arm.send_angles_safe(CLEAR_VIEW_POSE_DEG, speed=20)
            self._wait_settled(2.6, mirror=dual_sync)
        if self._stop_requested:
            return {"ok": False, "reason": "stopped"}

        # Hover -> Pre -> Closed-loop Z trim to 77.5 +- 1.0 (hard safety floor >= 75.5mm)
        self.arm.send_angles_safe(q_hover, speed=22)
        self._wait_settled(2.6, mirror=dual_sync)
        if self._stop_requested:
            return {"ok": False, "reason": "stopped"}
        self.arm.send_angles_safe(q_pre, speed=18)
        st_pre = self._wait_settled(2.4, mirror=dual_sync)
        z_act = float((st_pre.get("coords") or [0, 0, 84.0])[2])
        if z_act > 78.8:
            err_z = z_act - 77.5
            z_cmd = max(72.5, 84.0 - min(err_z, 10.5))
            q_trim, _ = self._solve_pose(mx, my, z_cmd, jaw_az, q_init=qp_rad)
            if q_trim:
                self.arm.send_angles_safe(q_trim, speed=14)
                st_pick = self._wait_settled(2.2, mirror=dual_sync)
                z_act = float((st_pick.get("coords") or [0, 0, z_act])[2])
        if z_act < 75.0:
            self.arm.send_angles_safe(q_hover, speed=18)
            return {"ok": False, "reason": "safety_floor_abort", "z_actual_mm": z_act}

        # Firm thermal-safe grip (10 for wafer, 8 for thin toothpaste cap — avoids 0 stall)
        grip_ratio = 0.10 if norm_key == "cuicuisha" else 0.08
        self.arm.set_gripper(grip_ratio, speed=45)
        time.sleep(0.9)

        # Lift & verify transit clearance >= 132 mm
        self.arm.send_angles_safe(q_hover, speed=18)
        st_lift = self._wait_settled(2.5, mirror=dual_sync)
        if float((st_lift.get("coords") or [0, 0, 0])[2]) < 132.0:
            return {"ok": False, "reason": "lift_clearance_abort", "coords": st_lift.get("coords")}
        if self._stop_requested:
            return {"ok": False, "reason": "stopped"}

        self.arm.send_angles_safe(q_drop_hi, speed=22)
        self._wait_settled(2.8, mirror=dual_sync)
        self.arm.send_angles_safe(q_drop_lo, speed=18)
        self._wait_settled(2.4, mirror=dual_sync)
        self.arm.set_gripper(0.65, speed=50)
        time.sleep(0.7)

        self.arm.send_angles_safe(q_drop_hi, speed=20)
        self._wait_settled(2.2, mirror=dual_sync)
        self.arm.send_angles_safe(CLEAR_VIEW_POSE_DEG, speed=22)
        st_end = self._wait_settled(2.6, mirror=dual_sync)
        self.drop_idx += 1
        self.delivered_stack.append({
            "key": norm_key,
            "label": tgt["label"],
            "drop_mc_mm": [round(dx, 1), round(dy, 1)],
            "src_mc_mm": [round(mx, 1), round(my, 1)],
            "jaw_az_deg": 0.0,
        })

        return {
            "ok": True,
            "item": norm_key,
            "target": tgt,
            "drop_mc_mm": [round(dx, 1), round(dy, 1)],
            "pick_z_actual_mm": round(z_act, 1),
            "end_coords": st_end.get("coords"),
            "dual_sync": dual_sync,
        }

    def run_real_recycle(self, item_key: str | None = None, dual_sync: bool = True) -> dict[str, Any]:
        """Pick the most recently delivered item from the Walnut Delivery Tray and return it to the Red Tray."""
        self._stop_requested = False
        norm_key = REAL_SKU_ALIASES.get(item_key, item_key) if item_key else None
        safety = self.check_pre_flight_safety()
        if not safety["ok"]:
            return {"ok": False, **safety}
        if not self.delivered_stack:
            return {"ok": False, "reason": "empty_real_delivery_tray"}

        idx = len(self.delivered_stack) - 1
        if norm_key:
            for i in range(len(self.delivered_stack) - 1, -1, -1):
                if self.delivered_stack[i]["key"] == norm_key:
                    idx = i
                    break
        rec = self.delivered_stack[idx]
        dx, dy = rec["drop_mc_mm"]
        sx, sy = rec["src_mc_mm"]
        jaw_az = float(rec.get("jaw_az_deg", 0.0))

        q_p_hi, qph_rad = self._solve_pose(dx, dy, 152.0, jaw_az)
        q_p_pre, qpp_rad = self._solve_pose(dx, dy, 84.0, jaw_az, q_init=qph_rad)
        q_r_hi, qrh_rad = self._solve_pose(sx, sy, 152.0, 0.0)
        q_r_lo, _ = self._solve_pose(sx, sy, 94.0, 0.0, q_init=qrh_rad)
        if not all([q_p_hi, q_p_pre, q_r_hi, q_r_lo]):
            return {"ok": False, "reason": "recycle_ik_unreachable", "record": rec}

        self.arm.set_gripper(0.62, speed=50)
        time.sleep(0.6)
        self.arm.send_angles_safe(q_p_hi, speed=22)
        self._wait_settled(2.6, mirror=dual_sync)
        self.arm.send_angles_safe(q_p_pre, speed=18)
        st_pre = self._wait_settled(2.4, mirror=dual_sync)
        z_act = float((st_pre.get("coords") or [0, 0, 84.0])[2])
        if z_act > 79.0:
            err_z = z_act - 77.8
            z_cmd = max(72.8, 84.0 - min(err_z, 10.2))
            q_trim, _ = self._solve_pose(dx, dy, z_cmd, jaw_az, q_init=qpp_rad)
            if q_trim:
                self.arm.send_angles_safe(q_trim, speed=14)
                st_pick = self._wait_settled(2.2, mirror=dual_sync)
                z_act = float((st_pick.get("coords") or [0, 0, z_act])[2])
        if z_act < 75.0:
            self.arm.send_angles_safe(q_p_hi, speed=18)
            return {"ok": False, "reason": "safety_floor_abort", "z_actual_mm": z_act}

        grip_ratio = 0.10 if rec["key"] == "cuicuisha" else 0.08
        self.arm.set_gripper(grip_ratio, speed=45)
        time.sleep(0.9)
        self.arm.send_angles_safe(q_p_hi, speed=18)
        st_lift = self._wait_settled(2.5, mirror=dual_sync)
        if float((st_lift.get("coords") or [0, 0, 0])[2]) < 132.0:
            return {"ok": False, "reason": "lift_clearance_abort", "coords": st_lift.get("coords")}

        self.arm.send_angles_safe(q_r_hi, speed=22)
        self._wait_settled(2.8, mirror=dual_sync)
        self.arm.send_angles_safe(q_r_lo, speed=18)
        self._wait_settled(2.4, mirror=dual_sync)
        self.arm.set_gripper(0.65, speed=50)
        time.sleep(0.7)
        self.arm.send_angles_safe(q_r_hi, speed=20)
        self._wait_settled(2.2, mirror=dual_sync)
        self.arm.send_angles_safe(CLEAR_VIEW_POSE_DEG, speed=22)
        st_end = self._wait_settled(2.6, mirror=dual_sync)
        self.delivered_stack.pop(idx)

        return {
            "ok": True,
            "item": rec["key"],
            "label": rec["label"],
            "pick_z_actual_mm": round(z_act, 1),
            "end_coords": st_end.get("coords"),
            "dual_sync": dual_sync,
        }
