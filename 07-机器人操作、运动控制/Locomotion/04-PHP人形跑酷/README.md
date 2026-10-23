# PHP 人形跑酷：动作匹配、视觉策略与 MuJoCo 体验

PHP（Perceptive Humanoid Parkour）让 Unitree G1 根据机载深度感知和方向指令，连续完成跨越、攀爬、撑越与翻滚等动作。它适合用来学习一个完整问题：**如何把多个动态技能连接起来，再让机器人根据眼前障碍选择合适的动作。**

本章从浏览器体验入手，再介绍官方预训练模型的本地运行与训练路线。内容依据 2026-10-23 核对的官方论文和代码整理；本章提供方法导读与官方复现步骤，尚未在本项目中运行 PHP 训练或本地策略评测。

## 1. 项目入口与演示

| 入口 | 内容 |
| --- | --- |
| [项目主页](https://php-parkour.github.io/) | 方法介绍、真机视频与交互演示 |
| [浏览器试玩](https://php-parkour.github.io/demo.html) | 在网页 MuJoCo 仿真中体验策略 |
| [官方代码](https://github.com/amazon-far/php_parkour) | 动作匹配、训练、评估、ONNX 导出与 MuJoCo 启动入口 |
| [论文 arXiv:2602.15827v2](https://arxiv.org/abs/2602.15827v2) | *Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching*，RSS 2026 |
| [预训练学生模型](https://github.com/amazon-far/php_parkour/releases/tag/student-assets-v1) | 配套的深度编码器和学生策略 ONNX 文件 |
| [视频号分享](https://weixin.qq.com/sph/AE6abAtYVX) | 罗氏索尔分享的 PHP 开源介绍，本章选题来源 |

先看官方的连续越障演示，留意机器人如何接近障碍、改变支撑方式，再回到行走状态：

<video controls muted playsinline preload="none" width="800">
  <source src="https://php-parkour.github.io/static/videos/step_climb_3.mp4" type="video/mp4">
</video>

视频由[官方项目页](https://php-parkour.github.io/)托管；若当前阅读器无法播放，可打开项目页观看。论文还报告了 G1 攀上最高 1.25 米障碍，以及在障碍位置被实时改变时继续调整动作的真机结果。这些是作者的实验结果。

## 2. 先在浏览器体验

打开[官方交互页面](https://php-parkour.github.io/demo.html)，等待模型和场景加载。网页在浏览器内运行 MuJoCo 仿真，适合先观察策略如何响应方向和速度命令。

| 网页按键 | 作用 |
| --- | --- |
| `W` | 向前移动，越障时持续按住直到落地 |
| `A` / `D` | 向左 / 向右转向 |
| `Y` | 切换高、低速模式 |
| `Space` | 暂停 |
| `Backspace` 或页面 Reset 按钮 | 重置回合 |

建议先对准一个障碍，保持 `W`，观察接近、接触、翻越和落地这几个阶段；熟悉后再用短促的转向操作对准下一个障碍。方向由人控制，具体越障动作由策略结合观测执行。[操作说明来源](https://php-parkour.github.io/)。

## 3. 从动作片段到感知策略

PHP 的训练流程分为三步：先生成连续参考动作，再训练能在物理仿真中跟踪动作的专家，最后将专家蒸馏成使用深度图的统一策略。

![PHP 官方方法图：动作匹配、教师训练与感知学生策略](https://php-parkour.github.io/static/images/fig2_method.png)

图来自[官方项目页](https://php-parkour.github.io/)，保留原图标注。

| 阶段 | 主要输入 | 得到什么 | 学习重点 |
| --- | --- | --- | --- |
| 动作匹配（Motion Matching） | 已重定向到机器人的原子动作、地形和运动目标 | 长时程运动学参考轨迹 | 让技能之间的姿态和运动趋势顺畅衔接 |
| 教师训练（Teacher） | 参考轨迹、特权状态和地形观测 | 各技能的强化学习专家 | 在接触动力学中跟踪参考动作 |
| 学生训练（Student） | 深度图、本体状态和方向命令；教师提供监督 | 一个感知驱动的多技能策略 | 在自己的访问状态上模仿教师，并结合 RL 改进表现 |

**动作匹配负责“怎么连接”。** 论文将其表述为特征空间的最近邻检索：从动作库中寻找与当前状态和运动目标相衔接的片段，组成连续轨迹。参考轨迹随后交给强化学习专家学习物理执行。

**DAgger 负责“走偏后还会不会做”。** 学生策略实际访问的状态会与教师示教状态不同，DAgger 在学生访问的状态上获取教师监督；再结合 RL 优化，形成能够根据深度感知执行多种技能的策略。方法依据见[论文](https://arxiv.org/abs/2602.15827v2)与[官方代码总览](https://github.com/amazon-far/php_parkour)。

部署时，两份模型相互配合：`depth_backbone.onnx` 编码深度图，`student.onnx` 将视觉特征、本体状态和方向命令转换成关节目标。因此运行时需要同一版本配套导出的两份文件。[模型接口说明](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/DEPLOY.md)。

## 4. 选择学习路线

| 路线 | 环境要求 | 适合的目标 |
| --- | --- | --- |
| 浏览器体验 | 能打开官方 demo 的浏览器 | 理解命令、感知与越障行为的关系 |
| 本地 MuJoCo 推理 | 官方 Holosoma MuJoCo / inference 环境与图形桌面 | 运行已发布学生策略，观察深度输入和物理闭环 |
| 动作匹配 | Conda；该步骤不需要 Isaac Sim | 生成、检查和组合参考动作 |
| 教师训练与学生蒸馏 | Linux、NVIDIA GPU、Isaac Sim | 训练技能专家并导出自己的策略 |

本章本地命令采用官方 Linux 路线。完整训练使用的硬件栈见[训练指南](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/README.md)；AMD 学习者可先使用浏览器入口体验。

## 5. 本地运行官方预训练策略

### 5.1 固定代码与 Holosoma 版本

官方代码依赖其固定的 Holosoma 子模块。以下提交是本章核对的版本；源码与模型放在教程仓库之外的独立目录：

```bash
git clone https://github.com/amazon-far/php_parkour.git
cd php_parkour
git checkout c7c9ce3fc75a8754a7e403c58de3b6933254d01b
git submodule update --init --recursive
```

后续命令均从 `php_parkour` 根目录运行，动作匹配小节会明确切换目录。

### 5.2 准备运行环境与模型

首次安装按官方脚本创建 MuJoCo 和推理环境；若对应环境已经按该版本安装，可跳过这两步：

```bash
bash thirdparty/holosoma/scripts/setup_mujoco.sh
bash thirdparty/holosoma/scripts/setup_inference.sh
```

启动脚本默认使用 `~/.holosoma_deps/miniconda3` 下的 `hsmujoco` 与 `hsinference` 环境。若复用其他 Holosoma checkout 的环境，按[官方部署指南](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/DEPLOY.md)重新绑定可编辑安装。

使用官方资源脚本获取学生模型：

```bash
python scripts/download_assets.py student
python scripts/download_assets.py student --verify
```

默认目录为 `~/.cache/php-parkour/student`，可用 `--destination DIR` 改到自己的模型目录。脚本按仓库的 `release-assets.json` 验证压缩包与模型文件的大小、SHA-256；`--verify` 只检查已有文件。公开学生包提供两份 ONNX 模型。[下载入口与资源说明](https://github.com/amazon-far/php_parkour)。

### 5.3 先仿真，再推理

在同一台机器的两个终端中启动。需要图形桌面；通过 SSH 操作时，将两个终端的 `DISPLAY` 配到同一个可用桌面。

终端一：

```bash
bash ./run_php_sim.sh
```

看到深度共享内存 `depth_img_shm` 已创建、形状为 `(1, 1, 58, 87)` 后，启动终端二：

```bash
PHP_STUDENT_DIR="$HOME/.cache/php-parkour/student"
BACKBONE="$PHP_STUDENT_DIR/depth_backbone.onnx" \
STUDENT="$PHP_STUDENT_DIR/student.onnx" \
bash ./run_php_inference.sh
```

推理端应报告已连接同名共享内存。接着在 MuJoCo 窗口按 `8` 降低吊架，使双脚着地，再按 `9` 移除吊架；在策略终端按 `]` 启动策略。用 `w` / `s` 前后移动，`a` / `d` 给出左右 45° 方向，`q` / `e` 给出左右 90° 方向，`=` 切换速度。`o` 进入阻尼模式，`i` 回到启动姿态，MuJoCo 窗口中的 `Backspace` 重置环境。

**本地版与网页按键不同**：网页切速度用 `Y`，本地推理端用 `=`。完整说明见[官方 sim2sim 指南](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/DEPLOY.md)。

### 5.4 常见问题

| 现象 | 检查位置 |
| --- | --- |
| 找不到 Holosoma 或启动脚本 | 确认已经初始化固定版本的子模块，并从 PHP 根目录启动 |
| 提示缺少模型 | 同时传入 `BACKBONE` 和 `STUDENT`，检查两份文件的实际路径 |
| 推理端等不到深度图 | 先启动仿真，确认 `depth_img_shm` 已创建、两端形状一致 |
| 机器人仍被吊着 | 在仿真窗口完成 `8`、`9` 操作，再在策略终端按 `]` |
| 方向键松开后仍持续或按键无效 | 检查图形桌面、窗口焦点和 SSH 的 `DISPLAY` 设置 |

重启时先退出自己上一轮的 PHP 仿真与推理进程；仿真启动器会重建深度共享内存。

## 6. 从参考动作到自己的策略

这一节给出学习顺序和最小入口。完成本地推理后，可先检查动作与地形数据，再投入教师和学生训练。

### 6.1 生成与查看参考动作

在 PHP 根目录准备动作匹配环境和数据库：

```bash
bash motion_matching/setup_env.sh
conda activate php-motion-matching
python scripts/download_assets.py databases
cd motion_matching
python -m motion_matching.run --mode generate --scenario high_speed_climb_76
python -m motion_matching.motion_visualizer \
  --motion-file motion_matching_results/high_speed_climb_76
```

打开可视化程序给出的地址，检查动作和障碍是否对齐、技能衔接是否顺畅。生成结果以 50 FPS 导出，包含参考动作 `*_motion.npz`、障碍描述 `*_terrain.npy` 和可视化网格 `*_terrain.obj`。生成与扩展场景的方法见[动作匹配指南](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/motion_matching/README.md)。

### 6.2 使用官方示例数据训练教师

从动作匹配目录返回 PHP 根目录。官方提供 locomotion、低速/高速 step、低速/高速 climb-76 五组示例动作与地形数据，可以直接从教师训练开始：

```bash
cd ..
bash wbt_training/training_runs/setup_env.sh
source scripts/source_isaacsim_setup.sh
python scripts/download_assets.py motions-locomotion
wandb login

REGISTRY="file://$HOME/.cache/php-parkour/motions-locomotion" \
NGPUS=1 \
bash wbt_training/training_runs/run_terrain_teacher.sh \
  --training.headless True --training.num-envs 4096 \
  --training.name example-locomotion-teacher
```

这里训练的是行走教师；其他技能需要切换对应数据并分别训练。示例包是训练数据，已发布的学生 ONNX 则用于直接推理。

### 6.3 蒸馏、评估和导出

教师训练完成后，把下方引用替换为自己 W&B 项目的教师 run：

```bash
TEACHER_CHECKPOINT='wandb://your-team/php-experiments/TEACHER_RUN_ID' \
REGISTRY=auto NGPUS=1 FINETUNE=1 \
bash wbt_training/training_runs/run_terrain_warp_distill.sh \
  --training.headless True --training.num-envs 4096 \
  --training.name my-student
```

`FINETUNE=1` 为默认的 DAgger＋PPO 路线，`FINETUNE=0` 使用纯 DAgger。多教师时，用逗号分隔教师引用；`REGISTRY=auto` 按顺序恢复各教师的数据集。只训练行走教师时，学生学习范围相应是行走；要学习跨越与攀爬，需要加入对应教师。

```bash
CHECKPOINT='wandb://your-team/php-experiments/STUDENT_RUN_ID' \
REGISTRY=auto bash wbt_training/training_runs/eval_student.sh
```

学生训练保存 checkpoint 时会自动导出配套 ONNX。评估自己的学生后，再把导出的两份模型接回第 5 节的 MuJoCo 运行流程。并行环境数量是跨 GPU 的总数，按显存调整；官方训练与评估默认使用 W&B，明确需要纯本地日志时可设置 `LOGGER=disabled`。完整配置与手动导出方法见[训练指南](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/README.md)，蒸馏开关见[启动脚本](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/wbt_training/training_runs/run_terrain_warp_distill.sh)。

## 7. 学习记录与关联阅读

初次学习可以记录三件事：给出了什么方向命令、机器人在什么距离开始改变动作、越障后是否能稳定继续行走。本地实验再补上代码版本、模型来源、障碍设置、完整回合视频和成功回合数，便于比较不同模型与场景。

接着阅读同章的 [Booster K1 网页 MuJoCo 与 AMP 训练](../02-BoosterK1-mjlab-AMP/README.md)，比较运动先验与动作跟踪的训练思路；也可以阅读 [BeyondRetarget 单目视频动作复现](../03-BeyondRetarget单目视频动作复现/README.md)，理解从人类视频生成机器人动作与训练物理控制策略之间的连接。

本章代码核对版本为 [`c7c9ce3`](https://github.com/amazon-far/php_parkour/tree/c7c9ce3fc75a8754a7e403c58de3b6933254d01b)。上游代码采用 [Apache-2.0](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/LICENSE)；图、视频通过官方地址引用，转载或再分发素材时按各自授权处理。
