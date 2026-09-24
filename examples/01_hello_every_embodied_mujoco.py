"""
Every-Embodied quick promo demo (UR-style arm + gripper).

Highlights:
- Hello Every-Embodied banner.
- UR-style 6-DOF arm with 2-finger gripper in MuJoCo.
- Interactive commands: 1/2/3 (wave / dance / random block grasp).
- Grasp pipeline uses trajectory planning + IK (no VLA/GraspNet).
- If ruckig is available, use jerk-limited planning; otherwise fallback to quintic interpolation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("未检测到 mujoco，请先安装：pip install mujoco") from exc

try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore

    HAS_RUCKIG = True
except Exception:
    HAS_RUCKIG = False


MODEL_XML = r"""
<mujoco model="every_embodied_ur_demo">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81" iterations="80" ls_iterations="20"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.96 1.0" rgb2="0.90 0.93 0.97" width="512" height="512"/>
    <material name="mat_floor" texture="grid" texrepeat="5 5"/>
    <material name="mat_ur" rgba="0.93 0.93 0.95 1"/>
    <material name="mat_gripper" rgba="0.14 0.14 0.18 1"/>
    <material name="mat_cube" rgba="0.97 0.43 0.18 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.0 0.0 2.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor" contype="1" conaffinity="2"/>

    <body name="robot_base" pos="0 0 0.05">
      <geom name="robot_base_geom" type="cylinder" size="0.09 0.05" rgba="0.16 0.16 0.18 1"/>

      <body name="shoulder_link" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-180 180" damping="2"/>
        <geom name="robot_shoulder_geom" type="capsule" fromto="0 0 0 0 0 0.16" size="0.03" material="mat_ur"/>

        <body name="upper_arm_link" pos="0 0 0.16">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-170 170" damping="2"/>
          <geom name="robot_upper_arm_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.026" material="mat_ur"/>

          <body name="forearm_link" pos="0.25 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-170 170" damping="1.8"/>
            <geom name="robot_forearm_geom" type="capsule" fromto="0 0 0 0.24 0 0" size="0.022" material="mat_ur"/>

            <body name="wrist1_link" pos="0.24 0 0">
              <joint name="wrist_1" type="hinge" axis="0 1 0" range="-180 180" damping="1.2"/>
              <geom name="robot_wrist1_geom" type="capsule" fromto="0 0 0 0.12 0 0" size="0.018" material="mat_ur"/>

              <body name="wrist2_link" pos="0.12 0 0">
                <joint name="wrist_2" type="hinge" axis="0 0 1" range="-180 180" damping="1.0"/>
                <geom name="robot_wrist2_geom" type="capsule" fromto="0 0 0 0.08 0 0" size="0.016" material="mat_ur"/>

                <body name="wrist3_link" pos="0.08 0 0">
                  <joint name="wrist_3" type="hinge" axis="0 1 0" range="-180 180" damping="0.8"/>
                  <geom name="robot_wrist3_geom" type="box" pos="0.04 0 0" size="0.035 0.025 0.02" material="mat_gripper"/>

                  <body name="tool" pos="0.08 0 0">
                    <site name="ee_site" pos="0 0 0" size="0.008" rgba="0 1 0 1"/>
                    <body name="left_finger" pos="0 0.018 0">
                      <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.03" damping="0.4"/>
                      <geom name="robot_left_finger_geom" type="box" pos="0.02 0 0" size="0.02 0.004 0.012" material="mat_gripper"/>
                    </body>
                    <body name="right_finger" pos="0 -0.018 0">
                      <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.03" damping="0.4"/>
                      <geom name="robot_right_finger_geom" type="box" pos="0.02 0 0" size="0.02 0.004 0.012" material="mat_gripper"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- The scripted grasp attaches the cube kinematically, so the loose cube only collides with the floor. -->
    <body name="object" pos="0.42 0.0 0.021">
      <freejoint name="object_free"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" material="mat_cube" friction="1.5 0.08 0.03" solref="0.005 1" contype="2" conaffinity="0"/>
    </body>

    <body name="drop_zone" pos="0.30 -0.22 0.002">
      <geom name="drop_zone_geom" type="box" size="0.06 0.06 0.002" rgba="0.2 0.82 0.32 0.45"/>
    </body>

    <body name="banner_anchor" pos="0.12 -0.34 0.22">
      <site name="hello_every_embodied" type="sphere" size="0.001" rgba="0 0 0 0"/>
    </body>
  </worldbody>

  <actuator>
    <position joint="shoulder_pan" kp="120"/>
    <position joint="shoulder_lift" kp="120"/>
    <position joint="elbow" kp="110"/>
    <position joint="wrist_1" kp="90"/>
    <position joint="wrist_2" kp="80"/>
    <position joint="wrist_3" kp="70"/>
    <position joint="left_finger_joint" kp="150"/>
    <position joint="right_finger_joint" kp="150"/>
  </actuator>
</mujoco>
"""


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]

GRIPPER_OPEN = np.array([0.028, 0.028], dtype=np.float64)
GRIPPER_CLOSE = np.array([0.005, 0.005], dtype=np.float64)


@dataclass
class KinematicsIndex:
    arm_qpos_ids: np.ndarray
    arm_dof_ids: np.ndarray
    arm_jnt_ids: np.ndarray
    gripper_qpos_ids: np.ndarray
    object_qpos_adr: int
    object_dof_adr: int
    object_body_id: int
    object_geom_id: int
    ee_site_id: int


class URTrajectoryDemo:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.idx = self._build_kinematics_index()
        self.drop_pos = np.array([0.30, -0.22, 0.03], dtype=np.float64)
        self.rng = np.random.default_rng()
        self.carrying = False
        self.attach_offset = np.zeros(3, dtype=np.float64)
        self.attach_rotmat = np.eye(3, dtype=np.float64)
        self._object_collision = (
            int(model.geom_contype[self.idx.object_geom_id]),
            int(model.geom_conaffinity[self.idx.object_geom_id]),
        )
        self._object_gravcomp = float(model.body_gravcomp[self.idx.object_body_id])
        self.pick_success = 0
        self.pick_total = 0

        self.home = np.array([0.0, -0.7, 1.2, -1.1, 1.57, 0.0], dtype=np.float64)
        self.wave_a = np.array([0.3, -0.6, 1.1, -1.3, 1.2, 0.8], dtype=np.float64)
        self.wave_b = np.array([0.3, -0.6, 1.1, -1.3, 1.2, -0.8], dtype=np.float64)
        self.dance_a = np.array([-0.9, -0.8, 1.5, -1.2, 1.2, 0.5], dtype=np.float64)
        self.dance_b = np.array([1.0, -0.9, 1.4, -1.0, 1.6, -0.4], dtype=np.float64)
        self.base_xy = np.array([0.0, 0.0], dtype=np.float64)

    def _build_kinematics_index(self) -> KinematicsIndex:
        arm_jnt_ids = []
        arm_qpos_ids = []
        arm_dof_ids = []
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            arm_jnt_ids.append(jid)
            arm_qpos_ids.append(self.model.jnt_qposadr[jid])
            arm_dof_ids.append(self.model.jnt_dofadr[jid])

        grip_qpos_ids = []
        for name in GRIPPER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            grip_qpos_ids.append(self.model.jnt_qposadr[jid])

        object_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return KinematicsIndex(
            arm_qpos_ids=np.array(arm_qpos_ids, dtype=np.int32),
            arm_dof_ids=np.array(arm_dof_ids, dtype=np.int32),
            arm_jnt_ids=np.array(arm_jnt_ids, dtype=np.int32),
            gripper_qpos_ids=np.array(grip_qpos_ids, dtype=np.int32),
            object_qpos_adr=int(self.model.jnt_qposadr[object_jid]),
            object_dof_adr=int(self.model.jnt_dofadr[object_jid]),
            object_body_id=int(object_body_id),
            object_geom_id=int(object_geom_id),
            ee_site_id=ee_site_id,
        )

    def print_banner(self) -> None:
        print("\n" + "=" * 72)
        print("Hello Every-Embodied! | UR + Gripper + Random Block Grasp Demo")
        print("=" * 72)
        print(f"轨迹规划后端: {'Ruckig (jerk-limited)' if HAS_RUCKIG else 'Quintic fallback'}")
        print("输入说明：")
        print("  1 -> 机械臂打招呼")
        print("  2 -> 机械臂趣味动作")
        print("  3 -> 随机方块抓取与放置（轨迹规划 + IK）")
        print("  4 -> 自动连播模式（适合录屏宣传）")
        print("  q -> 退出")
        print("-" * 72)

    def set_random_block(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        self._set_object_attached(False)
        if pos is None:
            x = self.rng.uniform(0.33, 0.50)
            y = self.rng.uniform(-0.16, 0.16)
            z = 0.021
            pos = np.array([x, y, z], dtype=np.float64)
        yaw = float(self.rng.uniform(-0.35, 0.35))
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        quat = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
        qadr = self.idx.object_qpos_adr
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return pos

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.idx.ee_site_id].copy()

    def get_ee_rotmat(self) -> np.ndarray:
        return self.data.site_xmat[self.idx.ee_site_id].reshape(3, 3).copy()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v.copy()
        return v / n

    def build_grasp_orientation(self, target_pos: np.ndarray) -> np.ndarray:
        # tool-z points down for natural top-down grasp; tool-x points from base to target
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        x_hint = np.array([target_pos[0] - self.base_xy[0], target_pos[1] - self.base_xy[1], 0.0], dtype=np.float64)
        if np.linalg.norm(x_hint) < 1e-6:
            x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = self._normalize(x_hint)
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        return np.column_stack([x_axis, y_axis, z_axis])

    @staticmethod
    def orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        # small-angle orientation error in world frame
        e = 0.5 * (
            np.cross(R_cur[:, 0], R_des[:, 0])
            + np.cross(R_cur[:, 1], R_des[:, 1])
            + np.cross(R_cur[:, 2], R_des[:, 2])
        )
        return e

    @staticmethod
    def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rotmat.reshape(-1))
        return quat

    @staticmethod
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        return t * t * (3.0 - 2.0 * t)

    def _plan_quintic(self, q0: np.ndarray, q1: np.ndarray, steps: int = 120) -> np.ndarray:
        t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        s = self._smoothstep(t)[:, None]
        return q0[None, :] + (q1 - q0)[None, :] * s

    def _plan_ruckig(self, q0: np.ndarray, q1: np.ndarray) -> Optional[np.ndarray]:
        if not HAS_RUCKIG:
            return None
        dof = len(q0)
        otg = Ruckig(dof, self.model.opt.timestep)
        inp = InputParameter(dof)
        out = OutputParameter(dof)
        inp.current_position = q0.tolist()
        inp.current_velocity = [0.0] * dof
        inp.current_acceleration = [0.0] * dof
        inp.target_position = q1.tolist()
        inp.target_velocity = [0.0] * dof
        inp.target_acceleration = [0.0] * dof
        inp.max_velocity = [1.2] * dof
        inp.max_acceleration = [2.0] * dof
        inp.max_jerk = [8.0] * dof

        traj = []
        result = Result.Working
        safe_max_steps = 3000
        for _ in range(safe_max_steps):
            result = otg.update(inp, out)
            traj.append(np.array(out.new_position, dtype=np.float64))
            out.pass_to_input(inp)
            if result == Result.Finished:
                break
            if result != Result.Working:
                return None
        if not traj:
            return None
        return np.vstack(traj)

    def plan_trajectory(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        traj = self._plan_ruckig(q0, q1)
        if traj is not None:
            return traj
        return self._plan_quintic(q0, q1, steps=140)

    def solve_ik_pose(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        max_iters: int = 260,
        w_pos: float = 1.0,
        w_ori: float = 0.35,
    ) -> Optional[np.ndarray]:
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        arm_q = self.data.qpos[self.idx.arm_qpos_ids].copy()

        for _ in range(max_iters):
            self.data.qpos[self.idx.arm_qpos_ids] = arm_q
            mujoco.mj_forward(self.model, self.data)
            err_pos = target_pos - self.get_ee_pos()
            err_ori = np.zeros(3, dtype=np.float64)
            if target_rot is not None:
                err_ori = self.orientation_error(self.get_ee_rotmat(), target_rot)
            if np.linalg.norm(err_pos) < 0.003 and (target_rot is None or np.linalg.norm(err_ori) < 0.05):
                self.data.qpos[:] = q_backup
                self.data.qvel[:] = qvel_backup
                mujoco.mj_forward(self.model, self.data)
                return arm_q.copy()

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.idx.ee_site_id)
            Jp = jacp[:, self.idx.arm_dof_ids]
            Jr = jacr[:, self.idx.arm_dof_ids]

            if target_rot is None:
                J = Jp
                err = err_pos
            else:
                J = np.vstack([w_pos * Jp, w_ori * Jr])
                err = np.hstack([w_pos * err_pos, w_ori * err_ori])

            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
            dq = np.clip(dq, -0.08, 0.08)
            arm_q = arm_q + dq

            # joint range clamp
            for i, jid in enumerate(self.idx.arm_jnt_ids):
                jmin, jmax = self.model.jnt_range[jid]
                arm_q[i] = np.clip(arm_q[i], jmin, jmax)

        self.data.qpos[:] = q_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        return None

    def is_robot_collision_low_risk(self, arm_q: np.ndarray) -> bool:
        # simple collision screening: avoid robot-floor and deep robot-cube intersections
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        self.data.qpos[self.idx.arm_qpos_ids] = arm_q
        self.data.qpos[self.idx.gripper_qpos_ids] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

        ok = True
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            pair = {n1, n2}
            if "floor" in pair and any(name.startswith("robot_") for name in pair):
                ok = False
                break
            if "cube" in pair and any(name.startswith("robot_") for name in pair) and c.dist < -0.002:
                ok = False
                break

        self.data.qpos[:] = q_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        return ok

    def _set_ctrl(self, q_arm: np.ndarray, q_gripper: np.ndarray) -> None:
        self.data.ctrl[:6] = q_arm
        self.data.ctrl[6:8] = q_gripper

    def _update_attached_object(self) -> None:
        if not self.carrying:
            return
        ee = self.get_ee_pos()
        ee_rot = self.get_ee_rotmat()
        qadr = self.idx.object_qpos_adr
        object_rot = ee_rot @ self.attach_rotmat
        object_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(object_quat, object_rot.reshape(-1))
        self.data.qpos[qadr : qadr + 3] = ee + ee_rot @ self.attach_offset
        self.data.qpos[qadr + 3 : qadr + 7] = object_quat
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_object_attached(self, attached: bool) -> None:
        self.carrying = attached
        geom_id = self.idx.object_geom_id
        body_id = self.idx.object_body_id
        if attached:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.body_gravcomp[body_id] = 1.0
        else:
            self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = self._object_collision
            self.model.body_gravcomp[body_id] = self._object_gravcomp
        mujoco.mj_forward(self.model, self.data)

    def attach_object(self) -> None:
        ee_pos = self.get_ee_pos()
        ee_rot = self.get_ee_rotmat()
        qadr = self.idx.object_qpos_adr
        dof_adr = self.idx.object_dof_adr
        object_rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(object_rot, self.data.qpos[qadr + 3 : qadr + 7])
        self.attach_offset = ee_rot.T @ (self.data.qpos[qadr : qadr + 3] - ee_pos)
        self.attach_rotmat = ee_rot.T @ object_rot.reshape(3, 3)
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        self._set_object_attached(True)

    def pin_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        qadr = self.idx.object_qpos_adr
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = quat
        self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def settle_object(self, steps: int = 40, viewer=None, realtime: bool = True) -> None:
        # Small settle loop to avoid contact impulse "catapult" at release.
        q_hold = self.data.qpos[self.idx.arm_qpos_ids].copy()
        qadr = self.idx.object_qpos_adr
        for _ in range(steps):
            self._set_ctrl(q_hold, GRIPPER_OPEN)
            mujoco.mj_step(self.model, self.data)
            # Keep object velocities small during initial release.
            self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 6] *= 0.2
            # Safety clamp: if object sinks slightly, pin back to table top.
            if self.data.qpos[qadr + 2] < 0.021:
                self.data.qpos[qadr + 2] = 0.021
                self.data.qvel[self.idx.object_dof_adr : self.idx.object_dof_adr + 3] = 0.0
            if viewer is not None:
                viewer.sync()
            if realtime:
                time.sleep(self.model.opt.timestep)

    def execute_trajectory(self, q_traj: np.ndarray, gripper: np.ndarray, viewer=None, realtime: bool = True) -> None:
        for q in q_traj:
            self._set_ctrl(q, gripper)
            mujoco.mj_step(self.model, self.data)
            self._update_attached_object()
            if viewer is not None:
                viewer.sync()
            if realtime:
                time.sleep(self.model.opt.timestep)

    def move_arm_to(self, target_q: np.ndarray, gripper: np.ndarray, viewer=None, realtime: bool = True) -> None:
        q0 = self.data.qpos[self.idx.arm_qpos_ids].copy()
        traj = self.plan_trajectory(q0, target_q)
        self.execute_trajectory(traj, gripper, viewer=viewer, realtime=realtime)

    def move_to_xyz(
        self,
        target_xyz: np.ndarray,
        gripper: np.ndarray,
        viewer=None,
        realtime: bool = True,
        target_rot: Optional[np.ndarray] = None,
    ) -> bool:
        q_goal = self.solve_ik_pose(target_xyz, target_rot=target_rot)
        if q_goal is None:
            return False
        self.move_arm_to(q_goal, gripper, viewer=viewer, realtime=realtime)
        return True

    def sample_reachable_block(self, trials: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(trials):
            x = self.rng.uniform(0.34, 0.48)
            y = self.rng.uniform(-0.14, 0.14)
            z = 0.021
            pos = np.array([x, y, z], dtype=np.float64)
            target_rot = self.build_grasp_orientation(pos)
            pre = pos + np.array([0.0, 0.0, 0.13], dtype=np.float64)
            grasp = pos + np.array([0.0, 0.0, 0.02], dtype=np.float64)
            q_pre = self.solve_ik_pose(pre, target_rot=target_rot)
            q_grasp = self.solve_ik_pose(grasp, target_rot=target_rot)
            if q_pre is None or q_grasp is None:
                continue
            if not self.is_robot_collision_low_risk(q_pre):
                continue
            if not self.is_robot_collision_low_risk(q_grasp):
                continue
            self.set_random_block(pos)
            return pos, target_rot

        # fallback to central easy pose
        fallback = np.array([0.40, 0.0, 0.021], dtype=np.float64)
        self.set_random_block(fallback)
        return fallback, self.build_grasp_orientation(fallback)

    def routine_wave(self, viewer=None, realtime: bool = True) -> None:
        self.move_arm_to(self.wave_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        for _ in range(2):
            self.move_arm_to(self.wave_b, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            self.move_arm_to(self.wave_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)

    def routine_dance(self, viewer=None, realtime: bool = True) -> None:
        for _ in range(2):
            self.move_arm_to(self.dance_a, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            self.move_arm_to(self.dance_b, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)

    def routine_autoshow(self, viewer=None, realtime: bool = True, rounds: int = 2) -> None:
        for i in range(rounds):
            print(f"[AutoShow] Round {i + 1}/{rounds}")
            self.routine_wave(viewer=viewer, realtime=realtime)
            self.routine_dance(viewer=viewer, realtime=realtime)
            self.routine_random_pick(viewer=viewer, realtime=realtime)

    def routine_random_pick(self, viewer=None, realtime: bool = True) -> None:
        self.pick_total += 1
        block, grasp_rot = self.sample_reachable_block()
        pre = block + np.array([0.0, 0.0, 0.13], dtype=np.float64)
        grasp = block + np.array([0.0, 0.0, 0.02], dtype=np.float64)
        lift = block + np.array([0.0, 0.0, 0.18], dtype=np.float64)
        place_rot = self.build_grasp_orientation(self.drop_pos)

        ok = self.move_to_xyz(pre, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        ok = ok and self.move_to_xyz(grasp, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        if not ok:
            print("[抓取] IK 失败：目标位置不可达，已返回 home。")
            self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            return

        # close gripper and attach block when close enough (for robust tutorial demo)
        self.execute_trajectory(
            self._plan_quintic(
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                steps=30,
            ),
            GRIPPER_CLOSE,
            viewer=viewer,
            realtime=realtime,
        )
        ee_rot = self.get_ee_rotmat()
        object_pos = self.data.qpos[self.idx.object_qpos_adr : self.idx.object_qpos_adr + 3]
        object_in_gripper = ee_rot.T @ (object_pos - self.get_ee_pos())
        grasp_ok = (
            np.linalg.norm(self.get_ee_pos() - grasp) < 0.04
            and np.linalg.norm(object_in_gripper[:2]) < 0.025
            and abs(object_in_gripper[2]) < 0.045
        )
        if grasp_ok:
            self.attach_object()
        if not self.carrying:
            print("[抓取] 方块未进入夹爪范围，本次跳过搬运。")
            self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
            return

        placed_pos = self.drop_pos.copy()
        placed_pos[2] = 0.021  # cube half-height + tiny clearance
        place = placed_pos - place_rot @ self.attach_offset
        place_pre = place + np.array([0.0, 0.0, 0.14], dtype=np.float64)
        self.move_to_xyz(lift, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=grasp_rot)
        self.move_to_xyz(place_pre, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=place_rot)
        self.move_to_xyz(place, GRIPPER_CLOSE, viewer=viewer, realtime=realtime, target_rot=place_rot)

        place_quat = self.rotmat_to_quat(place_rot)
        self.pin_object_pose(placed_pos, place_quat)
        self._set_object_attached(False)
        self.settle_object(steps=35, viewer=viewer, realtime=realtime)

        # Open gripper first, then retreat up to avoid side impulses.
        self.execute_trajectory(
            self._plan_quintic(
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                self.data.qpos[self.idx.arm_qpos_ids].copy(),
                steps=24,
            ),
            GRIPPER_OPEN,
            viewer=viewer,
            realtime=realtime,
        )
        self.move_to_xyz(place_pre, GRIPPER_OPEN, viewer=viewer, realtime=realtime, target_rot=place_rot)

        self.pick_success += 1
        rate = 100.0 * self.pick_success / max(1, self.pick_total)
        print(f"[抓取] 成功放置到目标区域，累计成功率：{self.pick_success}/{self.pick_total} ({rate:.1f}%)")
        self.move_arm_to(self.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR-style MuJoCo random block grasp demo")
    parser.add_argument("--headless", action="store_true", help="无图形模式运行")
    parser.add_argument("--fast", action="store_true", help="关闭实时 sleep，加速执行")
    parser.add_argument("--autoplay", action="store_true", help="启动后先自动连播一次")
    parser.add_argument("--autoplay-rounds", type=int, default=2, help="自动连播轮数，默认 2")
    return parser.parse_args()


def run_loop(
    demo: URTrajectoryDemo,
    viewer=None,
    realtime: bool = True,
    autoplay: bool = False,
    autoplay_rounds: int = 2,
) -> None:
    demo.print_banner()
    demo.move_arm_to(demo.home, GRIPPER_OPEN, viewer=viewer, realtime=realtime)
    if autoplay:
        demo.routine_autoshow(viewer=viewer, realtime=realtime, rounds=max(1, autoplay_rounds))
    while True:
        cmd = input("请输入 [1/2/3/4/q]: ").strip().lower()
        if cmd == "1":
            demo.routine_wave(viewer=viewer, realtime=realtime)
        elif cmd == "2":
            demo.routine_dance(viewer=viewer, realtime=realtime)
        elif cmd == "3":
            demo.routine_random_pick(viewer=viewer, realtime=realtime)
        elif cmd == "4":
            demo.routine_autoshow(viewer=viewer, realtime=realtime, rounds=2)
        elif cmd == "q":
            print("已退出示例，欢迎继续体验 Every-Embodied。")
            break
        else:
            print("无效输入，请输入 1/2/3/4/q。")


def configure_viewer_branding(viewer) -> None:
    # Show project name directly in MuJoCo interface.
    try:
        import glfw  # type: ignore

        glfw.set_window_title(viewer.window, "hello_every_embodied | Every-Embodied MuJoCo Demo")
    except Exception:
        pass

    try:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    realtime = not args.fast
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    demo = URTrajectoryDemo(model, data)

    if args.headless:
        run_loop(
            demo,
            viewer=None,
            realtime=realtime,
            autoplay=args.autoplay,
            autoplay_rounds=args.autoplay_rounds,
        )
        return

    try:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            configure_viewer_branding(viewer)
            run_loop(
                demo,
                viewer=viewer,
                realtime=realtime,
                autoplay=args.autoplay,
                autoplay_rounds=args.autoplay_rounds,
            )
    except Exception as exc:
        print(f"[提示] viewer 启动失败，切换到 headless 模式：{exc}")
        run_loop(
            demo,
            viewer=None,
            realtime=realtime,
            autoplay=args.autoplay,
            autoplay_rounds=args.autoplay_rounds,
        )


if __name__ == "__main__":
    main()
