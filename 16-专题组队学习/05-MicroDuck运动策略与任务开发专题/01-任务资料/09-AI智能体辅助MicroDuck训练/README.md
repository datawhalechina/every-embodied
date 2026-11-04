# AI 智能体辅助 MicroDuck 行走训练：从任务描述到策略

NVIDIA 展示了怎样把任务描述转化为机器人仿真实验：智能体规划障碍场、制作资产、组装场景，再使用强化学习训练 MicroDuck 的运动策略。本教程按阶段解释这条链路，并提供公开行走基线的训练入口。本文图片和动图来自 [NVIDIA 技术文章](https://developer.nvidia.com/blog/using-ai-agents-to-teach-robots-like-microduck-to-walk-in-simulation)，视频来自 NVIDIA，展示的是 NVIDIA 的演示。

[![观看 NVIDIA 完整视频：从任务描述到 MicroDuck 行走](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-labels-1024x576-jpg.webp)](https://www.youtube.com/watch?v=VvJai6HGfyE)

**[观看 NVIDIA 完整视频：From Prompt to Walking Robot With NVIDIA RTX Spark](https://www.youtube.com/watch?v=VvJai6HGfyE)**

## 从一句话到训练结果，中间发生了什么

![NVIDIA 的 MicroDuck 障碍场与障碍标注](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-labels-1024x576-jpg.webp)

*NVIDIA 概念图：坡道、平台及其他障碍构成机器人的训练场景。*

![从规划、建模、场景组装到训练与验证的流程](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-Diagram.webp)

*NVIDIA 的流程图：本地规划与资产制作、云端训练、本地验证，以及真机部署准备。*

任务描述提出要做什么；智能体把需求展开成计划并调用建模与仿真工具。真正产生运动策略的是强化学习：仿真器根据动作推进物理世界，训练算法利用多次尝试和奖励更新策略。NVIDIA 的示范以现成 MicroDuck 行走环境和预训练策略为起点，再增加视觉和障碍运动训练。

### 1. 把任务拆成场景方案

NVIDIA 使用 Hermes 和 Nemotron 3.5 Bolt 在 RTX Spark 上整理项目。规划阶段先查阅 MicroDuck 的模型、能力与约束，再确定行进路线、障碍尺寸、通行空间和所需输出。原文示例要求创建并测试包含运动与导航挑战的仿真障碍场；提示词还描述了概念图、资产识别、建模交接和验证等步骤，[示例全文见原文第一节](https://developer.nvidia.com/blog/using-ai-agents-to-teach-robots-like-microduck-to-walk-in-simulation)。

![障碍场概念图与资产清单](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Robot-Obstacles.webp)

*NVIDIA 概念图和资产清单：从坡道、平台等物体中识别可复用的资产。*

生成概念图以后，规划智能体识别图中的不同物体、合并重复实例，并为每类资产保留名称、位置和参考裁剪图。图片中看不到的背面、尺寸和遮挡部分需要作为假设记录。规划、机器人资料、概念图、清单和执行要求随后打包，交给负责资产制作的 Astra。

![规划智能体生成资产清单并交接](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Plan-inventory-generation-1.webp)

*NVIDIA 演示：将任务描述、概念图与资产清单整理成后续建模所需的材料。*

### 2. 制作可用于仿真的资产

裁剪出来的参考图可能很小，NVIDIA 的流程先使用云端图像编辑整理物体参考，再与原图对比轮廓、比例和颜色。Astra 编排本地运行的 FreeCAD 创建可编辑的几何结构，使用 Blender 添加细节与材质，最终保存模型并导出 USD 资产。

![FreeCAD 建模、Blender 材质和生成的工作灯](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/FreeCADBlender.webp)

*NVIDIA 演示：从 FreeCAD 几何建模，到 Blender 材质，再到完成的工作灯。*

除了生成新物体，也可以检索 [SimReady Central](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories) 中带仿真属性的 OpenUSD 资产，将现成资产与新建模型组合使用。几何外观之外，质量、碰撞体和摩擦等属性决定了物体在仿真中的表现。

![MicroDuck 和生成的坡道及障碍资产](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Walking-Robot.webp)

*NVIDIA 演示：MicroDuck 与坡道、台阶、箱体和垫子等场景资产。*

### 3. 组装场景和机器人

Astra 将资产放入 Isaac Sim，组织机器人、坡道和障碍物。机器人的连杆、关节和执行器共同决定控制信号如何驱动运动；Newton 模拟接触、运动和平衡，Omniverse 的渲染与传感器功能提供场景观测。

![Isaac Sim 中组装的障碍训练场](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Overview-User-Interface.webp)

*NVIDIA 演示：在 Isaac Sim 中检查 MicroDuck、坡道和其他障碍物的布局。*

这个场景同时提供机器人可执行的动作、障碍物与地面产生的物理反馈，以及用于判断前方情况的模拟传感器数据。训练时，碰撞、跌倒和前进都成为环境反馈的一部分。

### 4. 用并行仿真训练感知与运动策略

![机器人策略学习闭环](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-Policy-Learning-Loop.webp)

*NVIDIA 的学习闭环：深度和机器人状态进入策略，动作驱动仿真，再产生下一次观测。*

实验从 Isaac Lab 的 [MicroDuck 行走示例](https://github.com/isaac-sim/IsaacLab/tree/0c10609fd76534ccee3ae3973625003d65085441/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)和预训练运动策略出发，进一步训练跨越坡道和障碍的策略。适配任务使用头部相机深度与机器人状态作为观测；策略输出动作后，物理引擎计算下一时刻的运动和接触，强化学习算法据此更新策略。

![NVIDIA MicroDuck 并行坡道训练动图](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-parallel-training.gif)

*NVIDIA 原始动图：多只仿真小鸭在不同坡道和平台上并行练习。*

NVIDIA 报告该视觉障碍适配实验使用 **4096 个并行环境、1 块 NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU**。大量环境能同时产生训练经验，单次失败也会成为策略更新的反馈。

### 5. 把训练策略放回场景验证

训练阶段策略持续更新；评估阶段选定一个 checkpoint，让策略在场景里运行并观察其表现。NVIDIA 建议关注机器人是否识别障碍、调整动作并完成路线，同时记录跌倒、碰撞和卡住的位置，结合视频决定下一轮怎样修改环境或训练。

![NVIDIA MicroDuck 障碍场景评估动图](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Simulation-Policy-Evaluation.gif)

*NVIDIA 原始动图：在 Isaac Sim 障碍场中评估训练后的策略。*

### 6. 准备从仿真迁移到真机

真机舵机存在延迟、齿轮间隙、摩擦与电池电压变化。Better Actuator Models（BAM）用于在训练中模拟这些执行器响应；领域随机化则在训练时改变物理参数和观测噪声，让策略接触到更多可能的工作条件。NVIDIA 的文章介绍了部署准备，并展示另一段 Isaac Lab 行走策略的真机表现。

![NVIDIA MicroDuck 真机行走动图](https://developer-blogs.nvidia.com/wp-content/uploads/2026/10/Microduck-Physical-Walking.gif)

*NVIDIA 原始动图：物理 MicroDuck 运行 Isaac Lab 训练的行走策略。*

[Microduck 仿真训练与真机部署教程](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real)介绍了模型导出与机载运行方法。NVIDIA 障碍演示的完整场景资产、视觉策略权重和训练配置尚未核实到可直接复现的完整发布包；下面先从公开的行走基线入手。

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


