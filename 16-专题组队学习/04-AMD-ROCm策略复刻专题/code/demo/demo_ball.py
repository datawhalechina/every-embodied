"""前导课 Demo 1: 小球自由落体 (慢动作 + 亮球 + 网格地面, 下落全程清晰可见)"""
import mujoco
import numpy as np
import imageio
import os

# ========== 1. 模型定义 ==========
# 关键: 球用 emissive 材质(自发光, 不受光照影响变暗) → 亮红色保证可见
xml = """
<mujoco model="Falling Ball">
  <option gravity="0 0 -9.81"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.55 0.55 0.55" rgb2="0.45 0.45 0.45"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8"/>
    <material name="ball_mat" rgba="1 0.2 0.2 1" emission="0.9"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid_mat"/>
    <body name="ball" pos="0 0 1.5">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.12" material="ball_mat"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
# 慢动作: 物理步长 1/240s, 视频 20fps → 12x 慢放, 下落约 2.2 秒
model.opt.timestep = 1/240.0
print(f"✅ 模型编译成功 | 关节数={model.nq} | 相机={model.ncam}")

renderer = mujoco.Renderer(model, height=480, width=640)

# MjvCamera 实例路径(已验证可渲染) — 关键: 必须 mjv_updateCamera 计算矩阵
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 2.6
cam.lookat = np.array([0.0, 0.0, 0.8])   # 对准下落轨迹中点
cam.elevation = 0.0                        # 水平侧视
cam.azimuth = 90.0                         # 从 x 方向看
mujoco.mjv_updateCamera(model, data, cam, renderer.scene)

frames = []
heights = []
N = 240 * 3  # 3 秒
# 先步进 12 次跳过 mujoco 初始状态黑帧 bug
for _ in range(12):
    mujoco.mj_step(model, data)
for step in range(N):
    mujoco.mj_step(model, data)
    heights.append(data.qpos[2])
    if step % 12 == 0:  # 20fps 输出
        # 关键: 每次渲染前重新计算相机矩阵(mjv_updateScene 会原地修改 cam 导致漂移)
        mujoco.mjv_updateCamera(model, data, cam, renderer.scene)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

# 输出到专题目录下 (自动创建文件夹)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "demo_videos")
os.makedirs(out_dir, exist_ok=True)
out = os.path.abspath(os.path.join(out_dir, "ball_fall.mp4"))
imageio.mimsave(out, frames, fps=20)
nonblack = sum(np.mean(f) > 5 for f in frames)

# 球检测: mujoco render() 输出 RGB 顺序 (index0=R)
def ball_pixels(f):
    R, G, B = f[:,:,0].astype(int), f[:,:,1].astype(int), f[:,:,2].astype(int)
    return (R > 100) & (R - G > 50) & (R - B > 50)

visible = [i for i, f in enumerate(frames) if ball_pixels(f).sum() > 30]
print(f"✅ 视频: {out} ({len(frames)} 帧, 非黑 {nonblack}/{len(frames)})")
print(f"   高度: {heights[0]:.2f}m → {heights[-1]:.3f}m | 球可见帧: {len(visible)}/{len(frames)}")
print()
print("=" * 55)
print(f"  📁 视频已保存到: {out}")
print("  👀 查看方法:")
print("     1. 本机拉回: scp cloud-server:" + out + " ./")
print("     2. VSCode 直接打开: 文件浏览器里没有, 用 scp 拉到本机后播放")
print("=" * 55)
