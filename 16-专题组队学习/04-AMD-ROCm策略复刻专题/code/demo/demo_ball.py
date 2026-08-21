"""前导课 Demo 1：小球自由落体。

该脚本只依赖 MuJoCo，不需要上游 mujoco_pnp。生成的视频使用 H.264/yuv420p，
便于在浏览器和 JupyterLab 中直接预览。
"""
from __future__ import annotations

from pathlib import Path

import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_io import write_video


def main() -> None:
    xml = """
    <mujoco model="Falling Ball">
      <option gravity="0 0 -9.81"/>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.55 0.55 0.55" rgb2="0.45 0.45 0.45"/>
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
    model.opt.timestep = 1 / 240.0
    print(f"模型编译成功：关节数={model.nq}，相机={model.ncam}")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.6
    camera.lookat = np.array([0.0, 0.0, 0.8])
    camera.elevation = 0.0
    camera.azimuth = 90.0
    mujoco.mjv_updateCamera(model, data, camera, renderer.scene)

    frames = []
    heights = []
    for _ in range(12):
        mujoco.mj_step(model, data)
    for step in range(240 * 3):
        mujoco.mj_step(model, data)
        heights.append(data.qpos[2])
        if step % 12 == 0:
            mujoco.mjv_updateCamera(model, data, camera, renderer.scene)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

    output_dir = (
        Path(__file__).resolve().parents[2] / "outputs" / "demo_videos"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "ball_fall.mp4"
    codec = write_video(output, frames, fps=20)

    def ball_pixels(frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(int)
        green = frame[:, :, 1].astype(int)
        blue = frame[:, :, 2].astype(int)
        return (red > 100) & (red - green > 50) & (red - blue > 50)

    visible = [
        index for index, frame in enumerate(frames)
        if ball_pixels(frame).sum() > 30
    ]
    nonblack = sum(np.mean(frame) > 5 for frame in frames)
    print(
        f"视频已保存：{output}；codec={codec}；{len(frames)} 帧，"
        f"非黑帧 {nonblack}/{len(frames)}，球可见帧 {len(visible)}/{len(frames)}"
    )
    print(f"高度：{heights[0]:.2f}m -> {heights[-1]:.3f}m")


if __name__ == "__main__":
    main()
