"""Real Elephant Robotics myCobot 280 Pi SSH/Socket hardware bridge.
Shares the exact same safety joint limits and command interface as the MuJoCo digital twin.
"""
from __future__ import annotations

import json
import shlex
import threading
import time
from typing import Any

try:
    import paramiko
except ImportError:
    paramiko = None


# Safe software joint limits (deg) with 5 deg buffer inside official URDF hardware limits
SAFE_JOINT_LIMITS_DEG = [
    (-163.0, 163.0),  # J1 (hardware +-168)
    (-135.0, 135.0),  # J2 (hardware +-140)
    (-145.0, 145.0),  # J3 (hardware +-150)
    (-145.0, 145.0),  # J4 (hardware +-150)
    (-150.0, 155.0),  # J5 (hardware -155..+160)
    (-175.0, 175.0),  # J6 (hardware +-180)
]


class RealMyCobot280Bridge:
    """Connects to myCobot 280 Pi over Ethernet/LAN SSH using non-interactive paramiko."""

    def __init__(
        self,
        host: str = "192.168.1.100",
        user: str = "er",
        password: str = "",
        port: int = 22,
        serial_port: str = "/dev/ttyAMA0",
        baud: int = 1000000,
    ) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.serial_port = serial_port
        self.baud = baud
        self._lock = threading.RLock()
        self.connected = False
        self.last_error = ""
        self.last_angles_deg: list[float] = [0.0] * 6
        self.last_coords_mm: list[float] = [0.0] * 6

    def _run_remote(self, py_code: str, timeout: float = 8.0) -> tuple[bool, str]:
        if paramiko is None:
            return False, "paramiko 未安装"
        with self._lock:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password or None,
                    timeout=4.0,
                    look_for_keys=True,
                    allow_agent=False,
                )
                cmd = f"python3 -c {shlex.quote(py_code)}"
                _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
                out = stdout.read().decode("utf-8", "ignore").strip()
                err = stderr.read().decode("utf-8", "ignore").strip()
                if err and not out:
                    self.last_error = err
                    return False, err
                self.connected = True
                self.last_error = ""
                return True, out
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                return False, str(exc)
            finally:
                client.close()

    def probe(self) -> dict[str, Any]:
        code = (
            "import json, glob, time; "
            "from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.4); "
            "print(json.dumps({"
            "'angles': mc.get_angles(), "
            "'coords': mc.get_coords(), "
            "'gripper': mc.get_gripper_value(), "
            "'power': mc.is_power_on(), "
            "'videos': sorted(glob.glob('/dev/video*'))"
            "}))"
        )
        ok, out = self._run_remote(code)
        if not ok:
            return {
                "connected": False,
                "host": self.host,
                "user": self.user,
                "error": out,
                "hint": "请先以管理员身份运行 hardware/connect_mycobot_eth9.cmd 或将机械臂网线接入 192.168.1.1 路由器",
            }
        try:
            info = json.loads(out.splitlines()[-1])
            if isinstance(info.get("angles"), list) and len(info["angles"]) == 6:
                self.last_angles_deg = [float(a) for a in info["angles"]]
            if isinstance(info.get("coords"), list) and len(info["coords"]) == 6:
                self.last_coords_mm = [float(c) for c in info["coords"]]
            return {"connected": True, "host": self.host, **info}
        except Exception as exc:
            return {"connected": False, "host": self.host, "error": f"解析返回失败: {exc} ({out})"}

    def send_angles_safe(self, angles_deg: list[float], speed: int = 30) -> dict[str, Any]:
        clamped = [
            round(max(lo, min(hi, float(a))), 2)
            for a, (lo, hi) in zip(angles_deg, SAFE_JOINT_LIMITS_DEG)
        ]
        safe_speed = max(10, min(50, int(speed)))
        code = (
            "import time; from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.15); "
            f"mc.send_angles({clamped}, {safe_speed}); "
            "print('OK')"
        )
        ok, out = self._run_remote(code)
        if ok:
            self.last_angles_deg = clamped
        return {"ok": ok, "angles_deg": clamped, "speed": safe_speed, "output": out}

    def set_gripper(self, open_ratio: float, speed: int = 50) -> dict[str, Any]:
        # myCobot 280 adaptive gripper value: 0 (closed) .. 100 (fully open)
        val = int(max(0, min(100, round(open_ratio * 100))))
        code = (
            "import time; from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.15); "
            f"mc.set_gripper_value({val}, {int(speed)}); "
            "print('OK')"
        )
        ok, out = self._run_remote(code)
        return {"ok": ok, "gripper_value": val, "output": out}

    def power_on(self) -> dict[str, Any]:
        code = (
            "import time; from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.2); mc.power_on(); time.sleep(0.2); "
            "print(mc.is_power_on())"
        )
        ok, out = self._run_remote(code)
        return {"ok": ok, "output": out}

    def pause(self) -> dict[str, Any]:
        code = (
            "import time; from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.1); mc.pause(); print('PAUSED')"
        )
        ok, out = self._run_remote(code)
        return {"ok": ok, "output": out}

    def release_all_servos(self) -> dict[str, Any]:
        code = (
            "import time; from pymycobot.mycobot import MyCobot; "
            f"mc = MyCobot('{self.serial_port}', {self.baud}); "
            "time.sleep(0.15); mc.release_all_servos(); print('RELEASED')"
        )
        ok, out = self._run_remote(code)
        return {"ok": ok, "output": out}

    def capture_wrist_jpeg(self) -> bytes | None:
        """Fetch a single JPEG snapshot from /dev/video0 on myCobot 280 Pi."""
        code = (
            "import cv2, base64; "
            "cap = cv2.VideoCapture(0); "
            "cap.set(3, 640); cap.set(4, 480); "
            "ret, f = cap.read(); cap.release(); "
            "ok, buf = cv2.imencode('.jpg', f) if ret else (False, b''); "
            "print(base64.b64encode(buf).decode() if ok else '')"
        )
        ok, out = self._run_remote(code, timeout=6.0)
        if ok and out:
            import base64
            try:
                return base64.b64decode(out.splitlines()[-1])
            except Exception:
                return None
        return None
