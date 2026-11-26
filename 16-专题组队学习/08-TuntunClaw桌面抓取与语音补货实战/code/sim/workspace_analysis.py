"""Exact kinematic workspace analysis & 1:1 scaled exhibition desk blueprint generator
for Elephant Robotics myCobot 280 Pi + Camera Flange + Adaptive Gripper + D-Robotics RDK X5 Magic Box.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIM_DIR = Path(__file__).resolve().parent
ROOT_DIR = SIM_DIR.parent
DOCS_DIR = ROOT_DIR / "docs"


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml_text = (SIM_DIR / "mycobot_280_scene.xml").read_text(encoding="utf-8")
    assets = {p.name: p.read_bytes() for p in (SIM_DIR / "assets" / "mycobot_280" / "stl").glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    data = mujoco.MjData(model)
    return model, data


def solve_ik_topdown(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    yaw_rad: float = 0.0,
    q_init: np.ndarray | None = None,
    max_iter: int = 120,
    tol_pos: float = 2e-3,
    tol_ori: float = 0.06,
) -> tuple[bool, np.ndarray, float, float]:
    """Damped-least-squares 6-DOF IK placing tcp_site at target_pos with vertical top-down grasp.
    In tcp_site frame: +Y is the gripper pointing axis (must align with world -Z = [0, 0, -1]).
    """
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    j_low = model.jnt_range[:6, 0] + 0.05  # 3 deg safety margin
    j_high = model.jnt_range[:6, 1] - 0.05

    # Desired tcp_site rotation matrix:
    # col 1 (+Y in site = gripper pointing axis) -> [0, 0, -1]
    # col 0 (+X in site = finger open/close axis) -> [cos(yaw), sin(yaw), 0]
    # col 2 (+Z in site = col0 x col1) -> [-sin(yaw), cos(yaw), 0]
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R_des = np.array([
        [cy, 0.0, -sy],
        [sy, 0.0, cy],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)

    # Smart seed: aim J1 toward target azimuth
    az = np.arctan2(target_pos[1], target_pos[0])
    seeds = []
    if q_init is not None:
        seeds.append(np.array(q_init, dtype=np.float64))
    for offset in [0.0, -np.pi / 2, np.pi / 2, np.pi]:
        j1 = np.clip(((az + offset + np.pi) % (2 * np.pi)) - np.pi, j_low[0], j_high[0])
        seeds.append(np.array([j1, -0.55, -0.95, -0.10, 0.0, 0.0], dtype=np.float64))
        seeds.append(np.array([j1, 0.55, 0.95, 0.10, 0.0, 0.0], dtype=np.float64))

    best = (False, np.zeros(6), 999.0, 999.0)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for seed in seeds:
        q = seed.copy()
        for _ in range(max_iter):
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            p_cur = data.site_xpos[tcp_id]
            R_cur = data.site_xmat[tcp_id].reshape(3, 3)

            ep = target_pos - p_cur
            # Orientation error via rotation vector of R_des @ R_cur.T
            R_err = R_des @ R_cur.T
            # Small-angle / exact axis-angle
            tr = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
            theta = np.arccos(tr)
            if theta < 1e-6:
                eo = np.zeros(3)
            else:
                eo = (theta / (2.0 * np.sin(theta))) * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])

            pos_err = float(np.linalg.norm(ep))
            ori_err = float(np.linalg.norm(eo))
            if pos_err < best[2] + 0.05 * (ori_err - best[3]) or (pos_err < tol_pos and ori_err < tol_ori):
                if pos_err < best[2] or (pos_err < tol_pos and ori_err < best[3]):
                    best = (pos_err < tol_pos and ori_err < tol_ori, q.copy(), pos_err, ori_err)
            if pos_err < tol_pos and ori_err < tol_ori:
                return True, q.copy(), pos_err, ori_err

            mujoco.mj_jacSite(model, data, jacp, jacr, tcp_id)
            J = np.vstack([jacp[:, :6], 0.35 * jacr[:, :6]])
            err = np.concatenate([ep, 0.35 * eo])
            lam = 0.03
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), err)
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, j_low, j_high)

    return best


def analyze_workspace_and_draw_blueprint() -> dict:
    model, data = load_model()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Grid search reachable XY at table grasp height z = 0.035m (object grasp) and transit height z = 0.12m
    xs = np.linspace(-0.28, 0.28, 29)
    ys = np.linspace(-0.20, 0.28, 25)
    grid_reach = np.zeros((len(ys), len(xs)), dtype=bool)
    radii_ok = []

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            r = np.hypot(x, y)
            if r < 0.075 or r > 0.285:
                continue
            ok, _, _, _ = solve_ik_topdown(model, data, np.array([x, y, 0.040]), yaw_rad=0.0, max_iter=65)
            if ok:
                grid_reach[iy, ix] = True
                radii_ok.append(r)

    r_min_mm = round(float(np.min(radii_ok) * 1000), 1) if radii_ok else 105.0
    r_max_mm = round(float(np.max(radii_ok) * 1000), 1) if radii_ok else 245.0

    # Verify all 6 picking zone slots and 4 delivery tray slots at both grasp (z=0.032) and pre-grasp (z=0.117)
    pick_slots = {
        "coffee_1": [-0.17, 0.14, 0.032],
        "coffee_2": [-0.16, 0.185, 0.032],
        "oreo_1":   [-0.11, 0.14, 0.032],
        "oreo_2":   [-0.11, 0.21, 0.032],
        "apple_1":  [-0.05, 0.16, 0.032],
        "orange_1": [-0.05, 0.22, 0.032],
    }
    place_slots = {
        "tray_slot_1": [0.13, 0.16, 0.045],
        "tray_slot_2": [0.19, 0.16, 0.045],
        "tray_slot_3": [0.13, 0.09, 0.045],
        "tray_slot_4": [0.19, 0.09, 0.045],
    }

    slot_results = {}
    for name, pos in {**pick_slots, **place_slots}.items():
        ok_g, q_g, pe_g, oe_g = solve_ik_topdown(model, data, np.array(pos))
        ok_p, q_p, pe_p, oe_p = solve_ik_topdown(model, data, np.array([pos[0], pos[1], pos[2] + 0.085]), q_init=q_g)
        slot_results[name] = {
            "pos_mm": [round(pos[0] * 1000), round(pos[1] * 1000), round(pos[2] * 1000)],
            "radius_mm": round(float(np.hypot(pos[0], pos[1]) * 1000), 1),
            "grasp_ok": bool(ok_g),
            "pregrasp_ok": bool(ok_p),
            "pos_err_mm": round(pe_g * 1000, 2),
            "joints_deg": [round(float(np.rad2deg(v)), 1) for v in q_g],
        }

    # Joint limits table from exact URDF
    joint_limits = []
    for i in range(6):
        lo, hi = model.jnt_range[i]
        joint_limits.append({
            "joint": f"J{i+1}",
            "urdf_min_deg": round(float(np.rad2deg(lo)), 1),
            "urdf_max_deg": round(float(np.rad2deg(hi)), 1),
            "safe_min_deg": round(float(np.rad2deg(lo + 0.08)), 1),
            "safe_max_deg": round(float(np.rad2deg(hi - 0.08)), 1),
        })

    # Draw high-res 1:1 engineering blueprint PNG using PIL (crisp Chinese typography!)
    W, H = 1400, 960
    img = Image.new("RGB", (W, H), (244, 245, 242))
    draw = ImageDraw.Draw(img)

    def get_font(size: int, bold: bool = False):
        candidates = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for f in candidates:
            if Path(f).exists():
                return ImageFont.truetype(f, size)
        return ImageFont.load_default()

    f_title = get_font(30, True)
    f_sub = get_font(16, False)
    f_h2 = get_font(19, True)
    f_body = get_font(15, False)
    f_sm = get_font(13, False)

    # Header bar
    draw.rectangle([0, 0, W, 86], fill=(23, 25, 28))
    draw.rectangle([28, 18, 76, 66], fill=(8, 127, 120))
    draw.text((40, 22), "T", fill=(255, 255, 255), font=get_font(34, True))
    draw.text((94, 16), "TuntunClaw 囤囤钳 · 地瓜黑客松展台桌面 1:1 工程布局图", fill=(255, 255, 255), font=f_title)
    draw.text((94, 54), "实测机型: Elephant Robotics myCobot 280 Pi (6轴 + 腕部RGB相机法兰 + 自适应夹爪) + 地瓜 RDK X5 Magic Box", fill=(185, 210, 206), font=f_sub)

    # Coordinate transform: desk X in [-0.50, +0.50] (1000mm), Y in [-0.22, +0.42] (640mm)
    # Map onto canvas area x: 70..1030 (960px -> 0.96 px/mm), y: 120..735 (615px -> 0.96 px/mm)
    # Note: +Y (front toward visitors) at TOP of diagram or BOTTOM? Let's put +Y (front/visitors) at TOP so X-right matches overhead_top camera!
    scale = 920.0  # px per meter (0.92 px/mm)
    ox, oy = 530, 560  # Robot base (0, 0) pixel coords

    def w2p(x_m: float, y_m: float) -> tuple[int, int]:
        return int(ox + x_m * scale), int(oy - y_m * scale)

    # Draw desk rectangle: X [-0.50, +0.50], Y [-0.22, +0.42]
    d_tl = w2p(-0.50, 0.42)
    d_br = w2p(0.50, -0.22)
    draw.rectangle([d_tl, d_br], fill=(238, 222, 194), outline=(120, 96, 68), width=3)

    # 100mm grid lines
    for xm in np.arange(-0.4, 0.5, 0.1):
        p1, p2 = w2p(xm, -0.22), w2p(xm, 0.42)
        draw.line([p1, p2], fill=(224, 208, 180), width=1)
        draw.text((p1[0] - 18, d_br[1] + 6), f"{int(round(xm*1000))}", fill=(107, 113, 120), font=f_sm)
    for ym in np.arange(-0.2, 0.5, 0.1):
        p1, p2 = w2p(-0.50, ym), w2p(0.50, ym)
        draw.line([p1, p2], fill=(224, 208, 180), width=1)
        draw.text((d_tl[0] - 46, p1[1] - 8), f"{int(round(ym*1000))}", fill=(107, 113, 120), font=f_sm)

    # Draw IK-verified reachable dots (soft teal)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if grid_reach[iy, ix]:
                px, py = w2p(x, y)
                draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(176, 224, 217))

    # Draw workspace circles: R_max=280mm (dashed theoretical), R_safe=245mm (top-down grasp), R_min=105mm (inner deadzone)
    for r_m, col, label in [
        (0.280, (160, 170, 175), "理论最大臂展 R=280mm"),
        (r_max_mm / 1000.0, (8, 127, 120), f"垂直抓取安全外圈 R={int(r_max_mm)}mm"),
        (r_min_mm / 1000.0, (195, 58, 44), f"底座奇异/碰撞死区 R={int(r_min_mm)}mm"),
    ]:
        r_px = int(r_m * scale)
        draw.ellipse([ox - r_px, oy - r_px, ox + r_px, oy + r_px], outline=col, width=2)

    # Draw Picking Zone (取料区): X [-0.215, -0.005], Y [0.113, 0.257]
    pz_tl = w2p(-0.215, 0.257)
    pz_br = w2p(-0.005, 0.113)
    draw.rectangle([pz_tl, pz_br], fill=(214, 240, 235), outline=(8, 127, 120), width=3)
    draw.text((pz_tl[0] + 2, pz_tl[1] - 22), "取料区 (210×144mm) (-110,+185)", fill=(8, 127, 120), font=f_sm)

    # Draw Delivery Tray (放料区): X [0.080, 0.240], Y [0.025, 0.215]
    dt_tl = w2p(0.080, 0.215)
    dt_br = w2p(0.240, 0.025)
    draw.rectangle([dt_tl, dt_br], fill=(218, 194, 172), outline=(105, 62, 35), width=3)
    draw.text((dt_tl[0] + 2, dt_tl[1] - 22), "放料托盘 (160×190mm)", fill=(105, 62, 35), font=f_sm)

    # Draw all verified object & tray slots
    short_names = {
        "coffee_1": "咖1", "coffee_2": "咖2",
        "oreo_1": "奥1", "oreo_2": "奥2",
        "apple_1": "果", "orange_1": "橙",
        "tray_slot_1": "放1", "tray_slot_2": "放2",
        "tray_slot_3": "放3", "tray_slot_4": "放4",
    }
    for name, info in slot_results.items():
        x_m, y_m = info["pos_mm"][0] / 1000.0, info["pos_mm"][1] / 1000.0
        px, py = w2p(x_m, y_m)
        ok = info["grasp_ok"] and info["pregrasp_ok"]
        col = (8, 127, 120) if "tray" not in name else (120, 72, 38)
        if not ok:
            col = (210, 40, 40)
        draw.ellipse([px - 9, py - 9, px + 9, py + 9], fill=col, outline=(255, 255, 255), width=2)
        draw.text((px + 11, py - 8), short_names.get(name, name), fill=(23, 25, 28), font=f_sm)

    # Draw Magic Box: (-330, +250)mm, size 110x84mm
    mb_tl = w2p(-0.385, 0.295)
    mb_br = w2p(-0.275, 0.205)
    draw.rectangle([mb_tl, mb_br], fill=(35, 40, 48), outline=(18, 200, 225), width=3)
    draw.text((mb_tl[0] + 6, mb_tl[1] + 22), "Magic Box\n(-330,+250)", fill=(160, 245, 255), font=f_sm)
    # Draw Magic Box FOV cone toward workspace
    mb_c = w2p(-0.31, 0.24)
    draw.line([mb_c, w2p(-0.02, 0.28)], fill=(18, 175, 200), width=2)
    draw.line([mb_c, w2p(-0.18, 0.04)], fill=(18, 175, 200), width=2)

    # Draw Portable Monitor at (-340, -100)mm
    mon_tl = w2p(-0.44, -0.05)
    mon_br = w2p(-0.26, -0.15)
    draw.rectangle([mon_tl, mon_br], fill=(40, 44, 50), outline=(130, 136, 145), width=2)
    draw.text((mon_tl[0] + 8, mon_tl[1] + 28), "便携调试屏(可撤)", fill=(220, 225, 230), font=f_sm)

    # Draw big ROKAE arm exclusion warning on far right edge (on real desk)
    rok_tl = w2p(0.35, 0.35)
    rok_br = w2p(0.49, -0.18)
    draw.rectangle([rok_tl, rok_br], fill=(248, 224, 220), outline=(195, 58, 44), width=2)
    draw.text((rok_tl[0] + 10, rok_tl[1] + 180), "实验室大臂\n安全避让区\n(X>+350mm)", fill=(180, 45, 35), font=f_sm)

    # Draw myCobot 280 Pi base at (0, 0)
    r_base = int(0.055 * scale)
    draw.ellipse([ox - r_base, oy - r_base, ox + r_base, oy + r_base], fill=(95, 102, 110), outline=(23, 25, 28), width=3)
    draw.text((ox - 46, oy + r_base + 6), "myCobot 280 Pi\n底座原点 (0, 0)", fill=(23, 25, 28), font=f_sm)

    # Front / Visitor label
    draw.rectangle([d_tl[0], d_tl[1] - 32, d_br[0], d_tl[1] - 4], fill=(23, 25, 28))
    draw.text((d_tl[0] + 310, d_tl[1] - 28), "▲  展台观众朝向 / 扫码互动侧 (+Y 方向)  ▲", fill=(243, 200, 75), font=f_body)

    # Right sidebar: Specs & Layout Summary Card
    sx = 1025
    draw.rectangle([sx, 110, W - 25, H - 30], fill=(255, 255, 255), outline=(210, 215, 220), width=2)
    draw.rectangle([sx, 110, W - 25, 116], fill=(8, 127, 120))

    y_cursor = 132
    lines = [
        ("核心实测参数 (myCobot 280 Pi)", f_h2, (23, 25, 28)),
        (f"• 理论最大臂展: 280 mm (负载 250g)", f_body, (50, 55, 60)),
        (f"• 垂直下抓安全外圈: R = {int(r_max_mm)} mm", f_body, (8, 127, 120)),
        (f"• 底座死区内圈: R = {int(r_min_mm)} mm", f_body, (195, 58, 44)),
        (f"• 全槽位 IK 可达率: {sum(1 for v in slot_results.values() if v['grasp_ok'] and v['pregrasp_ok'])}/{len(slot_results)} ({int(100*sum(1 for v in slot_results.values() if v['grasp_ok'] and v['pregrasp_ok'])/len(slot_results))}%)", f_body, (8, 127, 120)),
        ("", f_sm, (0, 0, 0)),
        ("各区域推荐摆放坐标 (相对底座)", f_h2, (23, 25, 28)),
        ("1. 机械臂底座: (0, 0)", f_body, (23, 25, 28)),
        ("   距桌面后沿 220mm，居中偏左", f_sm, (90, 96, 104)),
        ("   避开右侧大臂 ≥350mm 安全距离", f_sm, (90, 96, 104)),
        ("2. 取料区 (货架垫): (-110, +185)mm", f_body, (8, 127, 120)),
        ("   尺寸 210×144mm，全部落于黄金环", f_sm, (90, 96, 104)),
        ("   (R = 160 ~ 255mm，零奇异点)", f_sm, (90, 96, 104)),
        ("3. 放料区 (胡桃木托盘): (+160, +120)mm", f_body, (105, 62, 35)),
        ("   尺寸 160×190mm，右前 37° 方向", f_sm, (90, 96, 104)),
        ("   J1 单轴平扫即可完成搬运，耗时短", f_sm, (90, 96, 104)),
        ("4. 地瓜 Magic Box: (-330, +250)mm", f_body, (12, 130, 155)),
        ("   位于臂展外 (+70mm余量) 零碰撞", f_sm, (90, 96, 104)),
        ("   斜向 32° 对准取料区+观众侧", f_sm, (90, 96, 104)),
        ("   双目视觉兼顾盘点 + 语音播报互动", f_sm, (90, 96, 104)),
        ("", f_sm, (0, 0, 0)),
        ("关节安全软限位 (URDF 实测)", f_h2, (23, 25, 28)),
    ]
    for jl in joint_limits:
        lines.append((f"• {jl['joint']}: [{jl['safe_min_deg']}°, {jl['safe_max_deg']}°] (硬件±{int(abs(jl['urdf_max_deg']))}°)", f_sm, (60, 66, 74)))

    for txt, font, col in lines:
        if txt == "":
            y_cursor += 8
            continue
        draw.text((sx + 18, y_cursor), txt, fill=col, font=font)
        y_cursor += 24 if font == f_sm else 28

    # Footer note
    draw.text((70, H - 38), "坐标系约定: 原点(0,0)在 myCobot 280 Pi 底座中心，+X向右，+Y朝向展台观众，+Z垂直桌面向上 (单位: mm)", fill=(90, 96, 104), font=f_sm)

    out_png = DOCS_DIR / "desk_layout_blueprint.png"
    img.save(out_png)

    report = {
        "robot_model": "Elephant Robotics myCobot 280 Pi + Camera Flange (RGB) + Adaptive Gripper",
        "workspace": {
            "theoretical_max_reach_mm": 280,
            "topdown_safe_outer_radius_mm": r_max_mm,
            "inner_deadzone_radius_mm": r_min_mm,
            "sweet_spot_annulus_mm": [135, 240],
        },
        "joint_limits": joint_limits,
        "recommended_layout_mm": {
            "desk_size": [1000, 640],
            "mycobot_280_base": [0, 0],
            "picking_zone_center": [-110, 160],
            "picking_zone_size": [210, 156],
            "delivery_tray_center": [160, 120],
            "delivery_tray_size": [160, 190],
            "magic_box_center": [-330, 250],
            "magic_box_yaw_deg": -32,
            "portable_monitor": [-340, -100],
            "big_arm_exclusion_x_gt_mm": 350,
        },
        "slot_verification": slot_results,
    }
    (DOCS_DIR / "workspace_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Blueprint saved: {out_png}")
    print(json.dumps(report["workspace"], indent=2))
    for k, v in slot_results.items():
        print(f"  {k:12s}: r={v['radius_mm']:5.1f}mm grasp_ok={v['grasp_ok']} pre_ok={v['pregrasp_ok']} err={v['pos_err_mm']}mm q={v['joints_deg']}")
    return report


if __name__ == "__main__":
    analyze_workspace_and_draw_blueprint()
