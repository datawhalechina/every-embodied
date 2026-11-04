# MuJoCo Roadster 特技仿真：轨迹优化与闭环控制

汽车如何使用喷气推力完成空中翻转、着陆和坡道行驶？本节介绍 [Khai Nguyen 的开源 Roadster 特技项目](https://github.com/xkhainguyen/roadster-stunt)，从预览、环境安装到规划、仿真与结果检查，理解轨迹优化和反馈控制如何配合。

## 视频与项目的关系

[Kevin Zakka 的 MuJoCo Roadster 视频](https://x.com/kevin_zakka/status/2112051258635042823)展示了 Roadster 特技。他介绍使用 MuJoCo 的 site actuator 模拟喷嘴、adhesion actuator 模拟坡道吸附，并用 box QP 分配喷气推力。目前未查到该视频对应的公开源码。

本教程介绍的 [roadster-stunt](https://github.com/xkhainguyen/roadster-stunt) 由 **Khai Nguyen 独立实现**，同样使用 MuJoCo 重现 Roadster 翻转和坡道动作，另有落到行驶卡车上的演示。它的算法和场景属于另一份实现。项目采用 [MIT 许可证](https://github.com/xkhainguyen/roadster-stunt/blob/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f/LICENSE)；以下动图直接引用该项目的演示文件。车辆和控制参数由该项目作者估算。

| 空中翻转 | 坡道行驶 | 落到移动卡车上 |
| --- | --- | --- |
| ![Khai Nguyen 的 MuJoCo 翻转演示](https://raw.githubusercontent.com/xkhainguyen/roadster-stunt/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f/docs/kickflip.gif) | ![Khai Nguyen 的 MuJoCo 坡道演示](https://raw.githubusercontent.com/xkhainguyen/roadster-stunt/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f/docs/ramp.gif) | ![Khai Nguyen 的 MuJoCo 卡车降落演示](https://raw.githubusercontent.com/xkhainguyen/roadster-stunt/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f/docs/semi.gif) |

*来源：Khai Nguyen，roadster-stunt 项目；演示为上游提供的仿真结果。*

## 控制系统如何工作

项目把汽车表示为带悬架、喷气推进和有限气源的 MuJoCo 模型。作者将轨迹和控制分为三个部分：

1. **离线规划**：使用 CasADi 与 IPOPT 优化飞行轨迹和推力时序，产生参考动作。
2. **空中姿态修正**：以 100 Hz 运行 LQR，根据实际姿态与参考姿态的偏差调整控制。
3. **着陆控制**：以 50 Hz 运行 TinyMPC 模型预测控制器，计算接近地面时的着陆控制。

规划给出一条期望轨迹，反馈控制则在模型误差或外部扰动出现时修正实际运动。坡道示例中，作者用恒定吸力模拟车辆附着；上游 README 给出的吸力参数为 20 kN，这是该仿真模型的设定。

作者在上游 README 中报告了一个扰动示例：车身质量增加 4%、喷嘴推力下降 3%，并在翻转途中施加阵风；仅回放预定喷气时序时，触地速度为 8 m/s，加入反馈后为 0.9 m/s。这是上游项目报告的单个设定下的结果，本仓库尚未复测。

## 安装与运行

下面的步骤固定到上游提交 [`a726d1a`](https://github.com/xkhainguyen/roadster-stunt/tree/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f)。需要 Git、Conda 或兼容的环境管理器；项目环境文件指定 Python 3.11、MuJoCo、CasADi、TinyMPC 等依赖。首次安装涉及依赖和模型下载，具体耗时取决于网络。

```bash
git clone https://github.com/xkhainguyen/roadster-stunt.git
cd roadster-stunt
git checkout a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f
conda env create -f environment.yml
conda activate roadster
pip install -e .
```

运行翻转和坡道实验：

```bash
python scripts/run_kickflip.py
python scripts/run_ramp.py
```

两个脚本分别遍历场景和开环、闭环模式，在 `data/` 保存 `.npz` 日志，并输出落地速度、姿态误差等指标。根据上游说明运行测试：

```bash
MUJOCO_GL=egl pytest
```

`MUJOCO_GL=egl` 是上游的无窗口渲染示例，需要环境具备可用的 EGL 支持。视频生成需要额外的视频与语音依赖，命令和输出位置见[上游 README](https://github.com/xkhainguyen/roadster-stunt/tree/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f)；示例动图可以直接在本页查看。本节命令与上游源码已核对，尚未在本仓库实际运行。

## 继续阅读

- [roadster-stunt 源码与演示](https://github.com/xkhainguyen/roadster-stunt/tree/a726d1a23ee1954d7376e6e48cc7c4460fd3ac4f)：模型、轨迹规划、控制器、实验脚本与测试。
- [MuJoCo 执行器文档](https://mujoco.readthedocs.io/en/stable/modeling.html#actuation-model)：了解关节和 site 上的执行器。
- [CasADi 文档](https://web.casadi.org/docs/) 与 [TinyMPC](https://tinympc.org/)：了解轨迹优化和模型预测控制。

