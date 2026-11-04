# UniSuperMimic：用可微世界模型学习人形机器人动作跟踪

本章介绍 [Motphys 的 Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)：它是在 UniLab 上对 [SuperMimic](https://supermimic.github.io/) 的独立复现，目标是让 Unitree G1 跟踪多种参考动作。项目提供 LAFAN1 与 BONES-SEED 数据处理、训练、评测和视频导出，支持 mjwarp、MuJoCo 和 Motrix 后端。

![UniSuperMimic 策略与参考动作对比](https://raw.githubusercontent.com/Motphys/Uni-SuperMimic/main/docs/assets/supermimic_comparison.gif)

*图 1：上游仿真演示，实体机器人是策略输出，蓝色虚影是参考动作。来源：[Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)。*

## 方法：仿真、世界模型与策略

SuperMimic 先从物理仿真采集状态和动作转换，训练可微的动力学模型；策略再通过世界模型中的短时想象轨迹和时间反向传播进行更新。仿真器提供训练数据，不需要对物理仿真器本身求导。世界模型与动作跟踪策略在训练过程中共同更新。

```text
参考动作 + 机器人状态 → 策略动作 → 物理仿真 → 转移数据
                                    ↓
                           学习可微世界模型
                                    ↓
                       想象轨迹 → 梯度更新策略
```

论文的研究结果、Motphys 的独立实现及本章教程应分别理解。截图显示的是社区演示；仓库将其实现标注为独立复现，不能直接视为论文作者原始代码。本章尚未运行训练或评测。

## 从 LAFAN1 路线开始

上游列出的环境为 Linux、Python 3.10–3.13 和 `uv`；mjwarp 训练需要 CUDA GPU。下面的命令来自[上游说明](https://github.com/Motphys/Uni-SuperMimic#installation)，在仓库根目录运行：

```bash
git clone https://github.com/Motphys/Uni-SuperMimic.git
cd Uni-SuperMimic
uv run unisupermimic-train --help
uv run tools/prepare_supermimic_assets.py
uv run tools/prepare_supermimic_assets.py --validate-only
```

资源准备会获取 G1 模型和 LAFAN1 动作数据；上游说明完整 LAFAN1 集合包含 40 个动作文件。检查点是资源验证命令完成且未报告缺失。模型、数据与上游依赖的许可分别见各自来源。

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.log_dir=runs/supermimic_mjwarp_001
```

这条命令采用上游默认 LAFAN1 配置，训练结果写入指定目录；训练需要充足 GPU 资源。选择 Motrix 后端时，按上游方式添加 `task=g1_super_mimic/motrix`，对应的恢复和评测应保持同一后端配置。

已有匹配训练运行目录和检查点时，可执行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt
```

评测从动作首帧开始，关闭观测和动作噪声并运行完整片段；评测结果保存在运行目录。上游也发布了检查点，使用前需按其说明准备匹配的动作库、模型和配置。视频导出支持指定 `jumps1_subject1` 等片段，具体参数见[上游视频导出说明](https://github.com/Motphys/Uni-SuperMimic#video-export)。

## 如何理解作者报告的结果

上游针对 LAFAN1 的表格报告 40 个动作、`ckpt_012000.pt` 的归一化回报 0.827227、失败动作 0/40。其 BONES-SEED 结果使用另一套大规模动作库及配置，不能把不同数据集上的指标直接当作同一条件下的提升。视频文字中的“约 40 分钟”是作者对特定运行的描述，实际训练时间应同时记录硬件、后端、数据集、迭代数和评测方式。

## 继续阅读

- 原论文：[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)
- 项目与安装、训练、评测：[Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)
- 仓库已有内容：[UniLab + MotrixSim 异构机器人强化学习](../../../10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/README.md)
