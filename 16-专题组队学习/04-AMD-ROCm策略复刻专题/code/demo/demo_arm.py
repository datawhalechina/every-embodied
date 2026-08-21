"""前导课 Demo 2: 机械臂场景 (官方 Renderer, 数据集场景 + 关节正弦运动)"""
import mujoco
import numpy as np
import imageio
import os
from pathlib import Path

TOPIC_ROOT = Path(__file__).resolve().parents[2]
xml_path = str(TOPIC_ROOT / "external" / "mujoco_pnp" / "asset" / "example_scene_y2.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
model.opt.timestep = 1/60.0

print(f"✅ 机械臂场景编译成功")
print(f"   关节数={model.nq} | 自由度={model.nv} | body={model.nbody} | geom={model.ngeom}")
print(f"   相机: {[model.camera(i).name for i in range(model.ncam)]}")
print(f"   执行器: {[model.actuator(i).name for i in range(model.nu)]}")

# 定位 joint1 的 qpos 索引
arm_jnt_ids = [model.joint(i).name for i in range(model.njnt)].index("joint1")
arm_qpos_adr = model.jnt_qposadr[arm_jnt_ids]
dof = 6
amp = np.array([0.6, 0.6, 0.8, 0.5, 0.5, 0.3])
freq = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
phase = np.array([0, 1.0, 0.5, 1.5, 2.0, 0.8])
kp = 20.0

# 用场景自带的相机 (agent 视角, 与数据采集一致)
renderer = mujoco.Renderer(model, height=480, width=640)

frames = []
for step in range(240):  # 4 秒
    t = step * model.opt.timestep
    target = amp * np.sin(2 * np.pi * freq * t + phase)
    qpos_now = data.qpos[arm_qpos_adr:arm_qpos_adr+dof]
    data.ctrl[:dof] = kp * (target - qpos_now)
    mujoco.mj_step(model, data)

    if step % 4 == 0:
        renderer.update_scene(data, camera="agentview")  # 数据采集同款视角
        frames.append(renderer.render().copy())

# 输出到专题目录下 (自动创建文件夹)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "demo_videos")
os.makedirs(out_dir, exist_ok=True)
out = os.path.abspath(os.path.join(out_dir, "arm_swing.mp4"))
imageio.mimsave(out, frames, fps=15)
nonblack = sum(np.mean(f) > 5 for f in frames)
print(f"✅ 视频: {out} ({len(frames)} 帧, 非黑 {nonblack}/{len(frames)})")
print(f"   最后关节角: {np.round(data.qpos[arm_qpos_adr:arm_qpos_adr+dof], 2)} rad")
print()
print("=" * 55)
print(f"  📁 视频已保存到: {out}")
print("  👀 查看方法:")
print("     1. 本机拉回: scp cloud-server:" + out + " ./")
print("     2. VSCode 直接打开: 文件浏览器里没有, 用 scp 拉到本机后播放")
print("=" * 55)
