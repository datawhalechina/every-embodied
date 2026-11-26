"""Shared overhead Red-Tray foreground segmentation & SKU extraction for Real & 1:1 Sim."""
from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np


def wrap180(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


def extract_red_tray_foreground(
    bgr: np.ndarray,
    roi_bounds: tuple[int, int, int, int] = (100, 360, 75, 295),
    bare_red_min_r: int = 120,
    bare_red_diff: int = 82,
    bare_red_min_v: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (fg_mask, h_ch, s_ch, v_ch) inside the Red Picking Tray interior."""
    H, W = bgr.shape[:2]
    b_c, g_c, r_c = cv2.split(bgr.astype(np.int16))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    y1, y2, x1, x2 = roi_bounds
    roi = np.zeros((H, W), dtype=np.uint8)
    roi[y1:y2, x1:x2] = 255

    tray_red = ((r_c >= 115) & ((r_c - np.maximum(g_c, b_c)) >= 70) & (roi > 0)).astype(np.uint8) * 255
    tray_red = cv2.morphologyEx(tray_red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(tray_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((H, W), dtype=np.uint8), h_ch, s_ch, v_ch

    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    tray_fill = np.zeros_like(roi)
    cv2.drawContours(tray_fill, [hull], -1, 255, -1)
    tray_in = cv2.erode(tray_fill, np.ones((13, 13), np.uint8))
    bare_red = (
        (r_c >= bare_red_min_r)
        & ((r_c - np.maximum(g_c, b_c)) >= bare_red_diff)
        & (b_c <= 85)
        & (v_ch >= bare_red_min_v)
        & (tray_in > 0)
    ).astype(np.uint8) * 255
    fg = ((tray_in > 0) & (bare_red == 0)).astype(np.uint8) * 255
    return fg, h_ch, s_ch, v_ch


def _end_core_width(tube_core: np.ndarray, pt: tuple[float, float], axis: np.ndarray) -> float:
    """Measure average cross-sectional tube width near `pt` along inward unit `axis`."""
    h_img, w_img = tube_core.shape
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    widths: list[float] = []
    for along in (2.0, 5.0, 8.0):
        px = pt[0] + along * float(axis[0])
        py = pt[1] + along * float(axis[1])
        span = 0.0
        for sgn in (-1.0, 1.0):
            for d in range(1, 14):
                u = int(round(px + sgn * d * float(perp[0])))
                y = int(round(py + sgn * d * float(perp[1])))
                if u < 0 or u >= w_img or y < 0 or y >= h_img or tube_core[y, u] == 0:
                    span += d
                    break
            else:
                span += 14.0
        widths.append(span)
    return float(np.mean(widths))


def _suppress_toothpaste_tails(
    cands: list[dict[str, Any]], tube_core: np.ndarray
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    """Drop wide flat crimped tube tails when paired along `tube_core` with a narrow screw cap."""
    drop: set[int] = set()
    tail_dirs: dict[int, tuple[float, float]] = {}
    h_img, w_img = tube_core.shape
    ts = np.linspace(0.0, 1.0, 25)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            p1, p2 = cands[i]["center"], cands[j]["center"]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if not (25.0 <= dist <= 54.0):
                continue
            us = np.clip(np.round(p1[0] + ts * (p2[0] - p1[0])).astype(int), 0, w_img - 1)
            vs = np.clip(np.round(p1[1] + ts * (p2[1] - p1[1])).astype(int), 0, h_img - 1)
            if float((tube_core[vs, us] > 0).mean()) < 0.88:
                continue
            ax = np.array([(p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist], dtype=np.float32)
            w1 = _end_core_width(tube_core, p1, ax)
            w2 = _end_core_width(tube_core, p2, -ax)
            if w1 > w2 + 1.5:
                drop.add(i)
                tail_dirs[j] = (p1[0] - p2[0], p1[1] - p2[1])
            elif w2 > w1 + 1.5:
                drop.add(j)
                tail_dirs[i] = (p2[0] - p1[0], p2[1] - p1[1])
    kept = [c for idx, c in enumerate(cands) if idx not in drop]
    idx_map = {id(c): idx for idx, c in enumerate(cands)}
    kept_dirs = {id(c): tail_dirs[idx_map[id(c)]] for c in kept if idx_map[id(c)] in tail_dirs}
    return kept, kept_dirs


def detect_tray_skus_unified(
    bgr: np.ndarray,
    px_to_xy_fn: Callable[[float, float], tuple[float, float]],
    is_sim: bool = False,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Detect Cuicuisha wafer bars & Toothpaste caps cleanly on either real Orbbec or 1:1 MuJoCo frames."""
    if roi_bounds is None:
        roi_bounds = (12, 178, 56, 178)
    fg, h_ch, s_ch, v_ch = extract_red_tray_foreground(
        bgr,
        roi_bounds=roi_bounds,
        bare_red_min_r=85 if is_sim else 120,
        bare_red_diff=55 if is_sim else 82,
        bare_red_min_v=65 if is_sim else 90,
    )
    counts = {"cuicuisha": 0, "toothpaste": 0}
    detections: list[dict[str, Any]] = []
    if not np.any(fg):
        return counts, detections

    # Toothpaste mask (white body + teal brand band)
    teal_mask = (v_ch >= 115) & (h_ch >= 75) & (h_ch <= 135) & (s_ch >= 35)
    white_mask = (v_ch >= (150 if is_sim else 130)) & (s_ch <= 90)
    white_core = (v_ch >= 165) & (s_ch <= 78)
    tube_bin = ((fg > 0) & (teal_mask | white_mask if is_sim else white_mask)).astype(np.uint8) * 255
    tube_bin = cv2.morphologyEx(tube_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    tube_core = ((fg > 0) & (teal_mask | white_core)).astype(np.uint8) * 255

    nt, lt, st, _ = cv2.connectedComponentsWithStats(tube_bin)
    large_tubes = np.zeros_like(tube_bin)
    for i in range(1, nt):
        if st[i, cv2.CC_STAT_AREA] >= (160 if is_sim else 250):
            large_tubes[lt == i] = 255

    if is_sim:
        for i in range(1, nt):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < 160:
                continue
            pts = np.column_stack(np.where(lt == i))[:, ::-1].astype(np.float32)
            mean, eig = cv2.PCACompute(pts, mean=None)[:2]
            axis, perp = eig[0], eig[1]
            proj = (pts - mean[0]) @ axis
            p_min, p_max = float(np.min(proj)), float(np.max(proj))
            length = max(p_max - p_min, 1.0)
            neg_pts = pts[proj < p_min + 0.24 * length]
            pos_pts = pts[proj > p_max - 0.24 * length]
            w_neg = float(np.ptp(neg_pts @ perp)) if len(neg_pts) else 99.0
            w_pos = float(np.ptp(pos_pts @ perp)) if len(pos_pts) else 99.0
            v_neg = float(np.mean(v_ch[neg_pts[:, 1].astype(int), neg_pts[:, 0].astype(int)])) if len(neg_pts) else 0.0
            v_pos = float(np.mean(v_ch[pos_pts[:, 1].astype(int), pos_pts[:, 0].astype(int)])) if len(pos_pts) else 0.0
            cap_pts = pos_pts if ((w_neg - w_pos) + 0.08 * (v_pos - v_neg)) >= 0 else neg_pts
            cu, cv = float(cap_pts[:, 0].mean()), float(cap_pts[:, 1].mean())
            long_world_az = math.atan2(float(axis[0]), float(axis[1]))
            jaw_az_rad = long_world_az + math.pi / 2.0
            wx, wy = px_to_xy_fn(cu, cv)
            mx_c, my_c = px_to_xy_fn(float(mean[0, 0]), float(mean[0, 1]))
            bx, by, bw, bh = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
            detections.append({
                "key": "toothpaste",
                "label": "舒客+小牙膏",
                "world_xy": [round(wx, 4), round(wy, 4)],
                "centroid_world_xy": [round(mx_c, 4), round(my_c, 4)],
                "pixel_center": [round(cu, 1), round(cv, 1)],
                "bbox": [bx, by, bx + bw, by + bh],
                "jaw_az_rad": jaw_az_rad,
                "confidence": 0.96,
                "zone": "picking",
            })
            counts["toothpaste"] += 1
    else:
        # On real Orbbec frames, isolate bright screw caps and suppress wide crimped tube tails
        caps_bin = ((fg > 0) & (v_ch >= 190) & (s_ch <= 58)).astype(np.uint8) * 255
        caps_bin = cv2.morphologyEx(caps_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nc, lc, sc, _ = cv2.connectedComponentsWithStats(caps_bin)
        raw_cands: list[dict[str, Any]] = []
        for i in range(1, nc):
            area = int(sc[i, cv2.CC_STAT_AREA])
            if not (35 <= area <= 260):
                continue
            pts = np.column_stack(np.where(lc == i))[:, ::-1].astype(np.float32)
            (cu, cv), (rw, rh), _ = cv2.minAreaRect(pts)
            long_l, short_l = (rh, rw) if rw < rh else (rw, rh)
            if short_l < 4.5 or (long_l / max(short_l, 1.0)) > 2.35:
                continue
            raw_cands.append({
                "center": (float(cu), float(cv)),
                "bbox": [int(sc[i, 0]), int(sc[i, 1]), int(sc[i, 0]) + int(sc[i, 2]), int(sc[i, 1]) + int(sc[i, 3])],
            })

        kept_cands, pair_dirs = _suppress_toothpaste_tails(raw_cands, tube_core)
        yy, xx = np.where(tube_core > 0)
        for cand in kept_cands:
            cu, cv = cand["center"]
            if id(cand) in pair_dirs:
                dx, dy = pair_dirs[id(cand)]
                long_img_deg = math.degrees(math.atan2(dy, dx))
                jaw_az_deg = wrap180(wrap180(-long_img_deg) + 90.0)
            else:
                d_r = np.hypot(xx - cu, yy - cv)
                ring = (d_r >= 12.0) & (d_r <= 38.0)
                if int(ring.sum()) > 15:
                    ang_pts = np.degrees(np.arctan2(yy[ring] - cv, xx[ring] - cu))
                    best_a, best_c = 0.0, -1
                    for a_deg in range(0, 360, 5):
                        cnt = int((np.abs((ang_pts - a_deg + 180.0) % 360.0 - 180.0) <= 14.0).sum())
                        if cnt > best_c:
                            best_c = cnt
                            best_a = float(a_deg)
                    jaw_az_deg = wrap180(wrap180(-best_a) + 90.0)
                else:
                    jaw_az_deg = 0.0
            wx, wy = px_to_xy_fn(cu, cv)
            detections.append({
                "key": "toothpaste",
                "label": "舒客+小牙膏",
                "world_xy": [round(wx, 4), round(wy, 4)],
                "centroid_world_xy": [round(wx, 4), round(wy, 4)],
                "pixel_center": [round(cu, 1), round(cv, 1)],
                "bbox": cand["bbox"],
                "jaw_az_rad": math.radians(jaw_az_deg),
                "confidence": 0.95,
                "zone": "picking",
            })
            counts["toothpaste"] += 1

    # Wafer bars (non-tube foreground inside Red Tray)
    dil_k = np.ones((5, 5), np.uint8)
    wafer_mask = ((fg > 0) & (cv2.dilate(large_tubes, dil_k) == 0)).astype(np.uint8) * 255
    wafer_mask = cv2.morphologyEx(wafer_mask, cv2.MORPH_OPEN, np.ones((3 if is_sim else 5, 3 if is_sim else 5), np.uint8))
    wafer_mask = cv2.morphologyEx(wafer_mask, cv2.MORPH_CLOSE, np.ones((5 if is_sim else 7, 5 if is_sim else 7), np.uint8))
    nw, lw, sw, _ = cv2.connectedComponentsWithStats(wafer_mask)
    for i in range(1, nw):
        area = int(sw[i, cv2.CC_STAT_AREA])
        if area < (120 if is_sim else 420):
            continue
        pts_all = np.column_stack(np.where(lw == i))[:, ::-1].astype(np.float32)
        (_, _), (rw0, rh0), _ = cv2.minAreaRect(pts_all)
        short0 = min(rw0, rh0)
        # Split two touching parallel Cuicuisha wafers cleanly along their short axis
        if not is_sim and area >= 1500 and short0 >= 26.0:
            mean_p, eig_p = cv2.PCACompute(pts_all, mean=None)[:2]
            perp_proj = (pts_all - mean_p[0]) @ eig_p[1]
            part_list = [pts_all[perp_proj < 0.0], pts_all[perp_proj >= 0.0]]
        else:
            part_list = [pts_all]

        for pts in part_list:
            if len(pts) < 120:
                continue
            (cu, cv), (rw, rh), ang = cv2.minAreaRect(pts)
            long_img_deg = ang + 90.0 if rw < rh else ang
            if is_sim:
                long_rad = math.radians(long_img_deg)
                jaw_az_rad = math.atan2(math.cos(long_rad), math.sin(long_rad)) + math.pi / 2.0
            else:
                jaw_az_rad = math.radians(wrap180(wrap180(-long_img_deg) + 90.0))
            wx, wy = px_to_xy_fn(cu, cv)
            bx, by = int(pts[:, 0].min()), int(pts[:, 1].min())
            bx2, by2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            detections.append({
                "key": "cuicuisha",
                "label": "雀巢脆脆鲨",
                "world_xy": [round(wx, 4), round(wy, 4)],
                "centroid_world_xy": [round(wx, 4), round(wy, 4)],
                "pixel_center": [round(float(cu), 1), round(float(cv), 1)],
                "bbox": [bx, by, bx2, by2],
                "jaw_az_rad": jaw_az_rad,
                "confidence": 0.95,
                "zone": "picking",
            })
            counts["cuicuisha"] += 1

    return counts, detections
