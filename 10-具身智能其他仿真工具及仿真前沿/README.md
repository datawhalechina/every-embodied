# 机器人仿真教程

本目录包含机器人仿真平台的选型说明、环境配置教程、平台实践文档和挑战赛资料。

## 入口文档

- [Isaac Lab 3.0 多后端与柔体操作](16-IsaacLab3多后端与柔体操作/README.md) - 了解多后端仿真、网线插拔与抽桌布官方示例、视频和运行入口。

- [MuJoCo Roadster 特技仿真](15-MuJoCo-Roadster特技仿真/README.md) - 理解喷气飞行轨迹优化、LQR 姿态修正与 MPC 着陆控制。

- [PhysX-Omni 彩色资产生成实践](14-PhysX-Omni彩色资产生成实践/README.md) - 单图部件生成、Blackwell 环境兼容、权重校验、Isaac Sim 渲染及机器人碰撞验收边界。

- [仿真平台选型与目录导览](01仿真配置文档.md) - 对比 MuJoCo、Isaac Sim、ManiSkill、Habitat、GenieSim、Genesis、AirSim 等平台的定位和适用任务。
- [ManiSkill 环境仿真配置](02Maniskill环境仿真配置.md) - ManiSkill 仿真环境搭建教程。
- [Isaac Sim、Isaac Lab 与 GR00T 部署导览](01Isaac部署与GR00T实践/00Isaac部署导览.md) - 统一说明本地 Windows、Linux 工作站、云服务器、Docker、micromamba/venv、Isaac Lab 和 GR00T 路线。
- [家务机器人环境配置](04家务机器人环境配置.md) - 家务机器人仿真环境配置。
- [SIM1 柔体仿真与数据生成](09SIM1柔体仿真与数据生成/01SIM1环境配置与运行.md) - 双臂布料操作的遥操作、replay、扩散轨迹生成与过滤流程。
- [UniLab + MotrixSim 异构机器人 RL 训练复现](11UniLab-MotrixSim异构RL训练/README.md) - 在 6GB 显卡上跑通 UniLab 的 MotrixSim 后端、PPO 训练、资源监控和视频回放。
- [HumanoidMimicGen 全身规划数据生成导读](12HumanoidMimicGen全身规划数据生成导读/README.md) - 拆解 NVIDIA 等提出的少量人形遥操作示教到千级 loco-manipulation 数据生成方法，覆盖技能标注、全身 IK、运动规划、G1 benchmark 和 sim-and-real co-training。
- [PhysicsNeMo 物理 AI 求解器导读](13-PhysicsNeMo物理AI求解器导读/README.md) - 了解 NVIDIA PhysicsNeMo / Modulus 迁移后的 Physics-ML 框架，区分 neural operator / PINN surrogate 与机器人交互仿真器的边界。

## 平台专题

- [ManiSkill 详细文档](Maniskill详细文档/) - ManiSkill 仿真平台的详细使用文档。
- [Isaac Sim 本地与云端配置教程](01Isaac部署与GR00T实践/01Isaac-Sim本地与云端配置.md) - 本地工作站、云服务器、Docker、pip 和 micromamba 配置。
- [Isaac Lab + GR00T 云部署教程](01Isaac部署与GR00T实践/02阿里云部署Isaac-Lab-GR00T完整教程.md) - 阿里云环境下的历史版本 Isaac Lab 和 GR00T 部署流程。
- [Genesis 环境配置](Genesis仿真环境配置/01环境配置和测试.md) - Genesis 仿真平台环境配置。
- [Genesis 可视化和渲染](Genesis仿真环境配置/02可视化和渲染.md) - Genesis 可视化和渲染实践。
- [Genesis World 1.0 完整体验](Genesis仿真环境配置/03Genesis%20World%201.0完整体验与机器人仿真流水线.md) - 从官方架构、Blackwell/CUDA 环境、Franka 仿真、Nyx 高保真渲染到室内资产导入边界的完整学习章节。
- [GenieSim 配置](07GenieSim配置.md) - GenieSim 环境配置。
- [GenieSim3 配置](08GenieSim3配置.md) - GenieSim3 环境配置。
- [UniLab + MotrixSim 异构训练](11UniLab-MotrixSim异构RL训练/README.md) - CPU 仿真与 GPU learner 分离的机器人 RL 训练实践。
- [HumanoidMimicGen 全身规划数据生成](12HumanoidMimicGen全身规划数据生成导读/README.md) - 人形机器人全身操作数据生成、MuJoCo/robosuite benchmark 与 GR00T/策略学习连接。
- [PhysicsNeMo 物理 AI 求解器](13-PhysicsNeMo物理AI求解器导读/README.md) - 面向 CFD、结构、天气、PDE surrogate 和 Physics-ML 的扩展工具，不作为机器人仿真主线。

## 仿真资源

- [资源文件](assets/) - 仿真相关图片和资源文件。
- [仿真挑战赛](challenge竞赛/) - 机器人仿真相关竞赛资料。

## 推荐学习路径

1. 先阅读 [仿真平台选型与目录导览](01仿真配置文档.md)，确定自己的任务属于操作、导航、系统部署还是前沿平台调研。
2. 操作与强化学习方向：学习 [ManiSkill 环境仿真配置](02Maniskill环境仿真配置.md)，再进入 [ManiSkill 详细文档](Maniskill详细文档/)。
3. 高保真渲染与系统集成方向：学习 [Isaac Sim、Isaac Lab 与 GR00T 部署导览](01Isaac部署与GR00T实践/00Isaac部署导览.md)，再根据本地 Windows、Linux 工作站、云服务器或 GR00T 复现目标选择对应路线。
4. 家务任务与复杂交互方向：学习 [家务机器人环境配置](04家务机器人环境配置.md)。
5. 柔体操作方向：学习 [SIM1 柔体仿真与数据生成](09SIM1柔体仿真与数据生成/01SIM1环境配置与运行.md)。
6. 强化学习训练基础设施方向：学习 [UniLab + MotrixSim 异构机器人 RL 训练复现](11UniLab-MotrixSim异构RL训练/README.md)，理解 CPU 物理仿真与 GPU 策略学习如何拆分。
7. 人形全身操作数据方向：阅读 [HumanoidMimicGen 全身规划数据生成导读](12HumanoidMimicGen全身规划数据生成导读/README.md)，理解少量遥操作示教如何通过技能约束、全身规划和随机化扩成策略训练数据。
8. 物理 AI 扩展方向：阅读 [PhysicsNeMo 物理 AI 求解器导读](13-PhysicsNeMo物理AI求解器导读/README.md)，了解 neural operator、PINN 和 surrogate model 与机器人仿真器的区别。

## 环境要求

不同平台的要求差异很大，建议以各配置文档为准。一般需要：

- Python 3.8+ 或平台指定版本。
- CUDA 和 NVIDIA 驱动，用于 GPU 仿真或高质量渲染。
- MuJoCo、Isaac Sim、ManiSkill、Habitat 等平台对应依赖。
- NVIDIA GPU，复杂渲染和大规模并行仿真建议使用更高显存配置。

