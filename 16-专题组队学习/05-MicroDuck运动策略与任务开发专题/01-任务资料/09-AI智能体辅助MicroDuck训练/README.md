# AI 智能体辅助 MicroDuck 行走训练：从任务描述到策略

NVIDIA 展示了一条从任务描述出发、构建障碍场景、训练并验证 MicroDuck 运动策略的流程。本节拆解每个环节，并用公开的 Isaac Lab 行走任务说明如何开始。以下命令来自公开教程，本仓库尚未在该版本完成训练验证。

## 演示是如何工作的

[NVIDIA 的流程说明](https://developer.nvidia.com/blog/using-ai-agents-to-teach-robots-like-microduck-to-walk-in-simulation)给出的任务描述以“构建并测试带导航与运动挑战的 Microduck 仿真障碍场”为目标。原文展示了带省略号的示例提示词及其研究规划、概念图、资产清单、交接、建模和验证步骤，可在原文第一节查看完整展示。

1. **规划**：Hermes 和 Nemotron 3.5 Bolt 整理机器人资料、路线、尺寸、场景清单与交接材料。
2. **建模与组装**：Astra 调用 FreeCAD 和 Blender 制作资产，再在 Isaac Sim 中组装场景。仿真资产也可来自 [SimReady Central](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)。
3. **训练**：以 Isaac Lab 的 MicroDuck 示例环境及预训练行走策略为起点，进一步训练具有深度相机与机器人状态观测的障碍运动策略。NVIDIA 说明该适配实验使用一块 RTX PRO 6000 Blackwell Workstation Edition 和 4096 个并行环境。
4. **评估与部署准备**：把训练后的策略放回 Isaac Sim，检查越障、跌倒、碰撞等表现；再通过执行器模型和随机化为真机迁移做准备。

智能体组织工作并调用工具；策略依靠仿真中的观测、动作、奖励和强化学习更新获得。NVIDIA 页面给出了流程和基础任务入口；其障碍场景的完整资产、训练配置和该视觉策略权重是否公开，目前尚未核实到可直接复现的完整发布包。

## 开源基线：Isaac Lab 行走

[NVIDIA 与 Hugging Face 的 Microduck 训练教程](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real)提供平地与粗糙地面行走任务。这里以 [Isaac Lab 固定提交 0c10609](https://github.com/isaac-sim/IsaacLab/tree/0c10609fd76534ccee3ae3973625003d65085441) 为代码参考，[任务注册](https://github.com/isaac-sim/IsaacLab/blob/0c10609fd76534ccee3ae3973625003d65085441/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/__init__.py)包含 Flat 和 Rough 两个任务；上游仓库采用 BSD-3-Clause 许可证。这里链接上游代码，不复制其实现。

Rough 基线包含坡道、台阶和随机地形；当前代码注明其为无视觉的速度行走任务，足部高度射线不进入 actor 观测。因此它适合复现行走训练，不能直接当作 NVIDIA 演示中的深度视觉避障策略。平地和粗糙地面任务使用 61 维策略观测与 14 维关节目标动作；发布策略时还需对齐观测顺序、归一化与动作缩放。

### 环境和训练

需要 Linux、受支持的 NVIDIA GPU 与驱动、Git 和 uv，具体版本与硬件要求以 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)为准。以下是在独立工作目录中使用固定源码版本的步骤：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 0c10609fd76534ccee3ae3973625003d65085441
uv sync
uv run isaaclab list_envs --keyword MicroDuck --show_presets
```

列表中应能看到 Flat、Rough 任务；找不到任务时先核对代码版本与依赖安装。按照上游教程启动粗糙地面训练：

```bash
uv run isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-MicroDuck \
  --num_envs 4096 \
  presets=backlash
```

`presets=backlash` 启用包含机械间隙的执行器配置；训练期间还使用动力学参数随机化。4096 是上游教程的并行环境数，应根据显存调整；减少并行环境后的耗时、训练效果需要重新测量。训练日志和 checkpoint 的具体路径以运行输出为准。

### 回放与导出

训练产生 checkpoint 后，在有图形显示的环境中使用上游回放命令：

```bash
uv run isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-MicroDuck \
  --checkpoint latest \
  --num_envs 16 \
  --viz newton_gl \
  presets=backlash
```

上游教程说明回放时会在该次运行的 `exported/` 中生成 `policy.pt` 与 `policy.onnx`。检查输出并观察是否能保持平衡、跟随速度命令；可记录跌倒次数、速度误差、不同地形表现及视频。ONNX 真机部署须使用匹配的机载接口并参照 [Microduck 部署指南](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real#6-deploy-the-onnx-locomotion-policy-on-a-real-microduck)。

## 与本专题已有内容的衔接

本专题已有 [MuJoCo 行走 HelloWorld](../../03-Notebook/00_MuJoCo策略控制_HelloWorld.ipynb)、[篮球平衡](../01-MicroDuck篮球平衡强化学习/README.md)与多项任务训练。先用 HelloWorld 理解“观测 → 策略 → 关节目标 → 物理更新”，再读 Isaac Lab 的任务配置，可以对比两个仿真训练栈的观测、动作、奖励与执行器模型。视觉越障需要额外的相机观测、任务定义、训练配置和可验证的评估，不能只替换一条文本指令。

参考资料：[NVIDIA 原始流程和示范](https://developer.nvidia.com/blog/using-ai-agents-to-teach-robots-like-microduck-to-walk-in-simulation) · [Isaac Lab 训练与部署教程](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real) · [Microduck 训练源码](https://github.com/pollen-robotics/microduck_rl)

