"""前导课 Demo 2：在上游 MuJoCo 场景中生成机械臂关节运动视频。

该脚本依赖 external/mujoco_pnp/asset/example_scene_y2.xml，适合用来确认
上游工程、场景资产和官方 Renderer 已经准备好。
"""
from __future__ import annotations

from pathlib import Path

import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_io import write_video


def main() -> None:
    topic_root = Path(__file__).resolve().parents[2]
    xml_path = topic_root / "external" / "mujoco_pnp" / "asset" / "example_scene_y2.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"找不到场景文件：{xml_path}。请先按 external/README.md 准备上游工程。"
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = 1 / 60.0

    print("机械臂场景编译成功")
    print(f"关节数={model.nq}，自由度={model.nv}，body={model.nbody}，geom={model.ngeom}")
    print(f"相机={[model.camera(i).name for i in range(model.ncam)]}")
    print(f"执行器={[model.actuator(i).name for i in range(model.nu)]}")

    joint_names = [model.joint(i).name for i in range(model.njnt)]
    if "joint1" not in joint_names:
        raise ValueError(f"场景中找不到 joint1；当前关节为 {joint_names}")
    arm_joint_id = joint_names.index("joint1")
    arm_qpos_address = model.jnt_qposadr[arm_joint_id]

    dof = 6
    amplitude = np.array([0.6, 0.6, 0.8, 0.5, 0.5, 0.3])
    frequency = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    phase = np.array([0.0, 1.0, 0.5, 1.5, 2.0, 0.8])
    kp = 20.0
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    for step in range(240):
        t = step * model.opt.timestep
        target = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        qpos = data.qpos[arm_qpos_address : arm_qpos_address + dof]
        data.ctrl[:dof] = kp * (target - qpos)
        mujoco.mj_step(model, data)
        if step % 4 == 0:
            renderer.update_scene(data, camera="agentview")
            frames.append(renderer.render().copy())

    output_dir = topic_root / "outputs" / "demo_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "arm_swing.mp4"
    codec = write_video(output, frames, fps=15)
    nonblack = sum(np.mean(frame) > 5 for frame in frames)
    print(
        f"视频已保存：{output}；codec={codec}；{len(frames)} 帧，"
        f"非黑帧 {nonblack}/{len(frames)}"
    )
    print(
        "如果运行在远端 Code Server，请把 outputs/demo_videos/ 下的视频拉回本机播放。"
    )


if __name__ == "__main__":
    main()
