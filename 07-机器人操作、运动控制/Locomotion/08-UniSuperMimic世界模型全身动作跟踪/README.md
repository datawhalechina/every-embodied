# UniSuperMimic：用可微世界模型学习人形机器人动作跟踪

本章介绍 [Motphys 的 Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic)：它是在 UniLab 上对 [SuperMimic](https://supermimic.github.io/) 的独立复现，目标是让 Unitree G1 跟踪多种参考动作。项目提供 LAFAN1 与 BONES-SEED 数据处理、训练、评测和视频导出，支持 mjwarp、MuJoCo 和 Motrix 后端。

**学习路线：**本页介绍方法与基础训练；[视频动作与社区复现实战](./02-视频动作与社区复现.md)介绍视频到参考动作、训练加速和灵巧手任务；[SuperMimic 社区项目完整合集](./03-SuperMimic社区项目完整合集.md)保留 41 个项目的中文分类、演示与原始来源。

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

## 从羽毛球比赛视频到全身动作跟踪

这条路线把比赛视频转换为 G1 的参考动作，再训练机器人在物理仿真中跟踪。可先打开 [SuperMimic 原论文的交互演示](https://supermimic.github.io/demo/)，观察不同参考动作与机器人跟踪的关系；上方动图展示的是 Uni-SuperMimic 社区复现。具体的羽毛球比赛片段、动作数据和训练检查点，需要来自相应实验的发布者。

```text
比赛视频 + 回合标注 → 选取球员和片段 → GVHMR 人体动作重建
    → GMR 人体到 G1 动作重定向 → 格式与关节顺序适配
    → Uni-SuperMimic 世界模型与跟踪策略训练 → 仿真评测与视频
```

| 环节 | 可用的开源资料 | 作用 |
| --- | --- | --- |
| 比赛与标注 | [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) | 提供比赛、回合、击球时间与球员位置等标注，并提供球场单应性信息；使用比赛影像需核对其获取方式和授权。 |
| 人体动作重建 | [GVHMR](https://github.com/zju3dv/GVHMR) | 从单目视频估计世界坐标系下的人体动作，输出参数化人体模型 SMPL-X 的运动参数。 |
| 动作重定向 | [GMR](https://github.com/YanjieZe/GMR#retargeting-from-gvhmr-video-to-robot) | 将人体动作映射到 G1，生成机器人根部位姿和关节轨迹，并可录制可视化视频。 |
| 跟踪训练 | [Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | 以适配后的机器人动作作为参考，联合训练世界模型和全身跟踪策略。 |

ShuttleSet 的回合与击球标注可用于选取片段；它本身并不提供可直接用于 G1 训练的关节轨迹。比赛画面中可能同时出现两名球员、遮挡和镜头切换，需要先确认目标球员及片段连续性。重建的人体坐标与球场坐标还需要结合画面、球场标定和尺度进行对齐。

### 先跑通人体重建和机器人重定向

下面使用上游提供的网球视频检查前处理链路；替换成已获授权的羽毛球单人片段后，可按相同入口处理。先分别依照 [GVHMR 安装说明](https://github.com/zju3dv/GVHMR)和 [GMR 安装说明](https://github.com/YanjieZe/GMR#installation)准备环境、模型及依赖；其中 GMR 需要按其文档准备 SMPL-X 人体模型。以下命令展示各仓库中的入口，路径应替换为实际位置。运行环境、模型资源和显示环境需要满足各上游项目要求。

在 GVHMR 仓库根目录运行：

```bash
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

`-s` 用于固定相机；移动镜头应查看上游的相机运动估计选项。默认结果位于 `outputs/demo/tennis/hmr4d_results.pt`。对于自有片段，将 `--video` 改为视频路径，并按视频文件名查找输出。

在 GMR 仓库根目录运行，将输入路径替换为上一步的结果：

```bash
uv run scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file /path/to/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
  --robot unitree_g1 \
  --save_path output/tennis_g1.pkl \
  --record_video
```

这会显示重定向过程，在 `output/tennis_g1.pkl` 保存运动数据，并在 `videos/` 下录制视频。脚本保存帧率、根部位置、根部旋转和关节位置；它展示的是参考动作的重定向，后续还需进行物理仿真中的策略训练。本文命令依据 [GVHMR 源码](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/tools/demo/demo.py)及 [GMR 脚本](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/gvhmr_to_robot.py)整理，实际运行前请检查对应版本的依赖及参数。

### 接入跟踪训练需要什么

GMR 保存的 `.pkl` 文件不能直接作为本章训练命令的动作库。以核对时的 [Uni-SuperMimic 动作加载器](https://github.com/Motphys/Uni-SuperMimic/blob/d78a5e2bc26ee26c3a94ae10418b7fb9018836ee/src/unisupermimic/tasks/motion_tracking/g1/supermimic/motion.py)为例，其普通动作输入为每帧 36 列的 `.csv`：根部位置 3 列、`xyzw` 顺序的四元数 4 列、29 个关节位置；源动作默认按每秒 30 帧读取。GMR 脚本同样保存根部旋转为 `xyzw`；[NJU 视频转动作工具](./02-视频动作与社区复现.md#22-生成-g1-动作及对照视频)进一步提供关节限位、脚底接地和 `.csv` 导出。固定版本的两个项目关节顺序一致，仍需检查动作坐标、尺度和质量；具体接入步骤见[视频动作实战](./02-视频动作与社区复现.md#24-接入-uni-supermimic)。项目也支持另一种压缩动作库格式；具体配置应以所用版本为准。

建议先用少量片段完成转换，回放并检查朝向、脚底接触、关节限位和帧率，再训练与评测。记录使用的片段划分、硬件、训练步数、完整回合跟踪成功的判定条件及对应视频。这样可以判断是参考动作重建、重定向还是策略跟踪影响了结果。若要实现对真实来球的反应与击球，还需要球、球拍、观测和交互任务设计。

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
