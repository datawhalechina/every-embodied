# Isaac Lab 3.0：多后端仿真与柔体操作示例

Isaac Lab 3.0 将机器人学习任务、物理求解、渲染和训练框架的接口进一步解耦。对于具身操作研究，它提供了从轻量仿真验证到大规模训练的共同工作流；网线插拔和抽桌布则让我们直观看到灵巧手、接触与柔性物体如何放进同一场景。

> 本文是官方资料导读和示例运行指南，资料核对日期为 2026-11-07；视频来自官方演示。运行命令及硬件需求以对应版本的官方文档为准，本文未声称在本机完成仿真运行。

## 1. 为什么值得学习？

[Isaac Lab 3.0 正式版](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)于 2026-10-20 发布，面向机器人学习提供多种物理、渲染和可视化后端。关键变化是**仿真与训练基础设施**：任务可以在支持的配置之间切换，同时沿用统一的任务接口和训练入口。

| 组成 | 作用 | 本教程中的例子 |
| --- | --- | --- |
| Isaac Lab | 组织机器人、场景、观测、动作和任务，连接训练与评估 | 创建可交互机器人操作环境 |
| Newton | 可独立于完整 Isaac Sim 运行的物理后端，提供多种求解器 | 线缆和桌布中的 VBD［基于顶点的动力学］ |
| PhysX［物理引擎］ | 刚体等物理仿真后端，提供独立运行库和 Isaac Sim 两种路径 | 使用现有 PhysX［物理引擎］任务与传感器 |
| 渲染与可视化 | 生成相机图像或展示仿真状态，可按任务选择 | Newton GL［图形查看器］、Kit［图形运行环境］ |
| 强化学习接口 | 将环境连接到 RSL-RL、RLinf 等训练库 | 定义观测、动作、奖励和终止条件后训练策略 |

Newton 还包括 MuJoCo-Warp［加速物理求解器］、MPM［物质点法］等求解路径，可用于不同的刚体、柔体和颗粒场景。物理、渲染和可视化是不同的选择；**具体能否组合由任务和版本决定**。详见[官方后端和预设说明](https://isaac-sim.github.io/IsaacLab/main/source/concepts/backends_and_presets.html)及[正式版说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)。

## 2. 官方视频：网线插拔与抽桌布

| 插入与拔出网线 | 五工位抽桌布对比 |
| --- | --- |
| [![RJ45 网线操作视频预览](assets/rj45-preview.gif)](assets/rj45-official.mp4) | [![五工位抽桌布视频预览](assets/tablecloth-preview.gif)](assets/tablecloth-official.mp4) |
| [观看官方完整视频](assets/rj45-official.mp4) | [观看官方五速度对比视频](assets/tablecloth-official.mp4) · [另一视角](https://github.com/user-attachments/assets/27ac0365-e862-4d60-bd85-51f585a580d5) |

视频来源：NVIDIA [Isaac Lab 生态页面](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/ecosystem.html)和[官方示例页面](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/demos.html)。网线完整视频为官方原片的压缩版本，预览从官方视频抽取；两段均为官方演示。

### 2.1 Sharpa RJ45：灵巧手与接触

Rizon 4s 机械臂配合 SHARPA 灵巧手，在一次连续仿真中完成插入、松手、重新抓取、按压卡扣和拔出。其主要结构如下：

1. 脚本状态机读取仿真状态，依次给出手掌和手指目标；Newton IK［逆运动学］及动作管理器驱动机器人。
2. 手部与接头外壳及卡扣之间使用 hydroelastic［静水弹性］接触；接头带有弹簧卡扣，线缆使用 VBD［基于顶点的动力学］。
3. 线缆一端固定，另一端跟随测得的插头位置；官方说明该线缆不会将力反馈给插头。官方给出的物理步进为 480 Hz［赫兹］、每步四次 VBD［基于顶点的动力学］迭代。

这是脚本控制示例：官方说明没有奖励、训练预设、策略检查点或自动回合重置。如果希望训练插拔策略，可以在理解场景、接触和动作接口后，另行定义观测、动作、奖励、成功条件、重置与随机化，再选择强化学习库。训练结果需要单独评估。

[示例环境与资产源码（固定版本）](https://github.com/isaac-sim/IsaacLab/tree/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion) · [官方资产和录制说明（固定版本）](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion/README.md)

### 2.2 Tablecloth Trick：柔性布料与多机器人

三个固定基座的 H2 Plus 机器人使用 SHARPA 灵巧手，以不同速度抽拉桌布；Newton VBD［基于顶点的动力学］处理布料、餐具和接触驱动的抓握，机器人通过脚本化 IK［逆运动学］控制。官方将该例标为实验性，并优先设置接触精度。官方展示图为三个工位，本文视频展示五个速度工位；网线插拔是另一示例。

[官方示例与命令](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/demos.html#focused-examples)

## 3. 如何体验

### 3.1 先在浏览器体验

访问[官方交互示例](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/interactive_demos.html)，可先体验材料变化、刚体接触或预训练策略回放。浏览器示例使用简化场景在本地 CPU［中央处理器］中运行，适合熟悉物理现象；网线和桌布的原生示例见下一节。

### 3.2 从源码运行官方示例

**版本区别：** `v3.0.0` 是正式版基线。网线示例目录在该标签中不存在，在核对时的 `main` 分支提交 `e3a79743e3119b82ebb72e8e3e2410c9275cc168` 中存在。下面固定该提交，避免后续分支变化影响命令；示例及依赖以对应源码为准。需要兼容的 NVIDIA GPU［图形处理器］、驱动和网络，先阅读[官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout e3a79743e3119b82ebb72e8e3e2410c9275cc168

# 本机环境与缓存保存在数据盘
export UV_PROJECT_ENVIRONMENT=/data/Data14TB/envs/isaaclab3-main
export UV_CACHE_DIR=/data/Data14TB/cache/uv
export HF_HOME=/data/Data14TB/cache/huggingface

# 查看当前版本的示例列表
uv run isaaclab example list

# 网线插拔：默认 Newton GL 查看器
uv run isaaclab example sharpa-rj45

# 抽桌布：使用第一块 CUDA 设备
uv run isaaclab example tablecloth --device cuda:0
```

`uv run` 会按项目配置准备依赖。网线场景还需要外部机器人与视觉资产，官方说明列出了资产服务器、本地资产目录覆盖选项及接头资源；机器人资产的许可条件以资产提供方为准。首次运行资源准备时间取决于网络和环境。指定项目环境和缓存位置时，请依照本机环境管理约定，避免将大型环境与资源写入系统盘。

如需官方的 Kit［图形运行环境］录制路线，参照[网线示例说明](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion/README.md)和[桌布示例文档](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/demos.html#focused-examples)。运行成功后，先观察模型与资产是否加载、抓取及插拔是否连续，再调整场景；遇到资产缺失或显卡兼容问题，先检查终端输出和官方说明。

### 3.3 正式版强化学习入门

如果目标是了解训练而非播放脚本，官方正式版提供倒立摆入门：

```bash
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
export UV_PROJECT_ENVIRONMENT=/data/Data14TB/envs/isaaclab3-v300
export UV_CACHE_DIR=/data/Data14TB/cache/uv
export HF_HOME=/data/Data14TB/cache/huggingface
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz newton_gl physics=newton_mjwarp
```

该命令训练倒立摆策略，不会训练网线示例。正式版的 Python［编程语言］、PyTorch［深度学习框架］、CUDA［并行计算平台］和驱动要求请以[正式版发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)及[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)为准。

## 4. 学习路径与资料

建议先浏览视频理解任务，再运行网线或桌布示例，阅读环境、接触和控制代码；有训练需求时转到倒立摆，之后再为自己的操作任务设计环境。已有教程中的 [Isaac 部署导览](../01Isaac部署与GR00T实践/00Isaac部署导览.md)介绍工作站与云端部署，[UniLab 与 MotrixSim 训练](../11UniLab-MotrixSim异构RL训练/README.md)提供另一条强化学习基础设施学习路线。

- [Isaac Lab 3.0 正式版发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [官方示例库和命令](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/demos.html)
- [Sharpa RJ45 环境与资源说明](https://github.com/isaac-sim/IsaacLab/blob/e3a79743e3119b82ebb72e8e3e2410c9275cc168/source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion/README.md)
- [交互式浏览器示例](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/interactive_demos.html)
