"""One-command hardware-free verification for TuntunClaw 1:1 MuJoCo twin & real IK guards."""
from __future__ import annotations

import math
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hardware.real_mycobot_bridge import RealMyCobot280Bridge
from hardware.real_pick_controller import REAL_DROP_SLOTS_PX, RealPickController, px_to_real_mc_mm
from sim.sim_engine import MyCobot280SimEngine


def main() -> None:
    print("[1/4] Initializing 1:1 MuJoCo digital twin...")
    sim = MyCobot280SimEngine()

    print("[2/4] Running overhead tray vision scan...")
    scan = sim.run_scene_scan()
    print(f"      Scan counts: {scan['counts']}")
    assert scan["ok"], f"Scan failed: {scan}"
    assert scan["counts"].get("cuicuisha") == 2 and scan["counts"].get("toothpaste") == 3

    print("[3/4] Running closed-loop pick-and-place + recycle in 1:1 MuJoCo...")
    r_wafer = sim.run_pick_and_place("cuicuisha", strategy="servo_closed_loop", fast=True)
    print(f"      Pick cuicuisha : ok={r_wafer['ok']} ik_err={r_wafer['ik_pos_err_mm']}mm trim={r_wafer['servo_trim_mm']}")
    assert r_wafer["ok"], f"Wafer pick failed: {r_wafer}"

    r_paste = sim.run_pick_and_place("toothpaste", strategy="servo_closed_loop", fast=True)
    print(f"      Pick toothpaste: ok={r_paste['ok']} ik_err={r_paste['ik_pos_err_mm']}mm trim={r_paste['servo_trim_mm']}")
    assert r_paste["ok"], f"Toothpaste pick failed: {r_paste}"

    rc_paste = sim.run_recycle_from_tray("toothpaste", fast=True)
    rc_wafer = sim.run_recycle_from_tray("cuicuisha", fast=True)
    print(f"      Recycle results: toothpaste={rc_paste['ok']}, cuicuisha={rc_wafer['ok']}")
    assert rc_paste["ok"] and rc_wafer["ok"]

    print("[4/4] Verifying real-controller forward-shoulder IK (J2 <= 20 deg) on all walnut drop slots...")
    ctrl = RealPickController(RealMyCobot280Bridge(), top_cam=None, sim_engine=sim)
    for idx, (px, py) in enumerate(REAL_DROP_SLOTS_PX, start=1):
        dx, dy = px_to_real_mc_mm(px, py)
        qh, qhr = ctrl._solve_pose(dx, dy, 152.0, 0.0)
        ql, _ = ctrl._solve_pose(dx, dy, 92.0, 0.0, q_init=qhr)
        assert qh is not None and ql is not None, f"Drop slot {idx} IK failed"
        assert qh[1] <= 20.5 and ql[1] <= 20.5, f"Backwards shoulder flip detected at slot {idx}"
        print(f"      Slot D{idx}: mc=({dx:.1f}, {dy:.1f})mm r={math.hypot(dx, dy):.1f}mm J2_hover={qh[1]:.1f}deg J2_low={ql[1]:.1f}deg")

    print("ALL_TUNTUNCLAW_VERIFICATION_CHECKS_PASSED")


if __name__ == "__main__":
    main()
