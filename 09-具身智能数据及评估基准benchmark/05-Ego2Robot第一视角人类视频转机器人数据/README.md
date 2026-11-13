# Ego2Robot：把第一视角人类视频转成机器人训练数据

> 从人手示范到机器人画面：先看官方视频，再理解动作、视觉和数据质量如何对齐。

[论文](https://arxiv.org/abs/2608.02580) · [官方项目页与视频](https://www-ye.github.io/ego2robot_blog/) · [官方代码](https://github.com/www-Ye/ego2robot_blog) · [中文使用说明](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/README_zh.md)

[![Ego2Robot 官方展示：人类操作转成机器人训练视频](https://www-ye.github.io/ego2robot_blog/assets/fig_ego2r_realplay_gallery.png)](https://www-ye.github.io/ego2robot_blog/assets/ego2robot_video.mp4)

*点击图片观看作者的约 2 分 47 秒完整视频。封面是人类操作转换为机器人画面的合成样例；后半段另有真实机器人执行。图片与视频来源：[Ego2Robot 官方项目页](https://www-ye.github.io/ego2robot_blog/)。*

## 1. 看什么：三个值得留意的画面

下面三个入口指向**同一段官方视频的不同时间位置**；具体播放位置由浏览器对时间片段的支持决定。也可在[项目页](https://www-ye.github.io/ego2robot_blog/#video)播放完整视频后手动拖动进度条。

| 视频位置 | 内容 | 应该怎么看 |
| --- | --- | --- |
| [约 00:24：人手与多形态机械臂对照](https://www-ye.github.io/ego2robot_blog/assets/ego2robot_video.mp4#t=24) | 原始人手视频和不同机器人外观的合成画面并排出现 | 观察同一类手部动作如何转换为不同机械臂画面；机器人是渲染叠加的 |
| [约 01:36：日常操作转换](https://www-ye.github.io/ego2robot_blog/assets/ego2robot_video.mp4#t=96) | 人类操作与对应的 ACone 合成画面 | 观察夹爪与物体的重叠、遮挡、开合和轨迹 |
| [约 01:43：真实机器人实验](https://www-ye.github.io/ego2robot_blog/assets/ego2robot_video.mp4#t=103) | 水果放篮、积木进抽屉、折毛巾、清扫等真实机器人片段 | 这部分是作者的真机执行演示；视频中标注部分片段以 2 倍速播放 |

这里展示的是论文作者发布的素材，不是本教程生成的运行结果。视频前半展示**数据合成过程**，后半展示**使用数据训练后的真实机器人实验**。

## 2. 这项工作解决什么问题

机器人示教数据需要真实机器、场地和遥操作，采集成本高；而人类第一视角的操作视频更容易积累。Ego2Robot 尝试把视频中的人手轨迹转成机械臂轨迹，将人手从画面中去掉，合成对应的机器人外观，形成可用于机器人策略预训练的数据。

论文使用四种来源约 **1,940 小时**的人类操作数据，针对 **15 种机器人形态**合成并筛选出 **18,561 小时**机器人格式数据。后一个数字包含同一来源动作在不同机器人形态上的扩展，不是额外采集了 18,561 小时独立的人类视频。数据混合预训练改善了若干仿真和真机任务表现。[论文第 3、5 节](https://arxiv.org/html/2608.02580)。

## 3. 完整流程

![Ego2Robot 官方流程图](https://www-ye.github.io/ego2robot_blog/assets/fig1_ego2r_pipeline.png)

*来源：[作者项目页](https://www-ye.github.io/ego2robot_blog/)。图中展示两条输入路径和共同处理阶段。*

```text
已有人手姿态标注的数据 ────────────────┐
                                     ├─→ 动作对齐：人手关键点 → 夹爪位姿 / 开合 → 平滑
原始视频 → 子任务标注 + 深度 / 手部重建 ┘                 │
                                                         ▼
    背景修复 ← 人手分割       机器人基座搜索 → 逆运动学求解
          └──────────── 深度遮挡合成机器人画面 ────────────┘
                                  │
                     论文中的多层质量筛选
                                  │
                  机器人视频 + 动作 + 指令 → 策略训练
```

1. **人手变动作。** 已带手部姿态的数据直接读取；原始视频则先估计深度、相机运动和手部轨迹，必要时分段并生成任务描述。以拇指和食指、中指融合出的关键点构造虚拟夹爪，得到中心、朝向和开合宽度，再进行轨迹平滑。
2. **画面变机器人。** 分割人手与手臂、修复背景；搜索目标机械臂的基座位置，利用逆运动学求解关节，再根据场景深度将机械臂叠回视频。
3. **筛选数据。** 论文描述三层筛选：处理过程中的求解和碰撞检查、统计异常检查，以及视觉语言模型对视频与任务描述的一致性检查。
4. **训练和评测。** 将合成数据和真实机器人数据混合预训练，再用目标任务示范微调；论文报告 RoboTwin 2.0、EBench 以及 ARX ACone 真机实验。可结合本站 [EBench 教程](../04-EBench.md)了解评测场景。

核心贡献在于把现有的手部重建、机器人动作重定向、视频修复和深度合成组合成可跨数据源、跨机器人形态扩展的数据管线，并通过分维度评测研究合成数据的训练收益。

## 4. 训练收益和实际边界

| 论文评测 | 仅机器人数据预训练 | 与合成数据混合预训练 | 说明 |
| --- | ---: | ---: | --- |
| RoboTwin Clean | 62.2% | 68.1% | 合成数据与机器人数据 1:1 |
| RoboTwin Randomized | 50.9% | 53.5% | 同一 1:1 设置，提升 2.6 个百分点 |
| EBench | 39.6% | 51.7% | 这里是 3:1 设置，不能和上面 1:1 当作同一次实验 |
| Franka 跨本体迁移 | 7.0% | 5.3% | 1:1 设置，收益并非每个机器人都相同 |

上表为[论文项目页](https://www-ye.github.io/ego2robot_blog/)报告的成功率。论文的几何有效帧条件包括逆运动学位置误差小于 0.05 米、机器人可见以及自碰撞检查，并检查双臂接触等异常；这些条件用于筛选合成轨迹，**不包含完整的物体接触、抓取稳定性和摩擦动力学验证**。夹爪与物体穿插、手部被不自然替换及重遮挡时的修复瑕疵，都值得在看视频时关注。人手映射到平行夹爪还会丢失精细手指动作。[论文局限与附录 A.5](https://arxiv.org/html/2608.02580)。

视频观感与接触质量需要分别检查：机器人画面叠加得流畅，可以帮助学习外观和运动对应关系；判断动作能否稳定抓取，还需结合物体几何、接触和真实执行来评估。

## 5. 代码在哪里，怎样开始

截至 2026-11-13，作者仓库包含合成管线、动作和渲染处理器、机器人配置、LeRobot 格式导出和 RoboTwin 评测代码。下面固定到本文检查的源码版本；完整运行需要准备视频数据、模型依赖和权重，本教程没有执行完整合成或训练。

```bash
git clone https://github.com/www-Ye/ego2robot_blog.git
cd ego2robot_blog
git checkout --detach 244a4ee02f02b2e93e94a81f27963597e0815ce6

# 官方要求 Python 3.12 及以上；使用独立环境
conda create -n ego2robot python=3.12 -y
conda activate ego2robot
pip install -e .
cp .env.example .env
```

按照[官方中文说明](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/README_zh.md)填写数据目录和模型路径，准备 Depth Anything 3、SAM 3、ProPainter；处理未标注视频还需要手部重建相关依赖。作者要求 `ffmpeg`、`ffprobe` 以及满足模型依赖的计算环境。Dyn-HaMR 使用独立环境。需要模型权重的项目应分别核对其下载方式、许可和使用条件。

| 输入类型 | 参考配置 | 阶段 |
| --- | --- | --- |
| 已标注 EgoDex | `configs/egodex.yaml` | 数据读取 → 动作 → 平滑 → 分割 → 修复 → 机器人合成 |
| 已标注 ViTRA | `configs/vitra.yaml` | 同上，数据读取阶段不同 |
| 已标注 EgoVerse | `configs/egoverse_mecka.yaml` 或 `configs/egoverse_scale.yaml` | 同上，按数据版本选择配置 |
| 原始视频 | `configs/arx.yaml` | 子任务、深度和手部重建后接入共同阶段；需要匹配标定 |

准备好数据及配置后，官方提供以下入口：

```bash
# 在上游仓库根目录，先按实际数据位置编辑配置和 .env
set -a; source .env; set +a
python scripts/run_pipeline.py --config configs/egoverse_mecka.yaml

# 对已有单个示范，仅处理动作和平滑阶段
python scripts/run_pipeline.py --config configs/egoverse_mecka.yaml \
  --demo-dir /path/to/demo --mode action,smoothing
```

程序可输出 `training_data_<Robot>_<setup>.npz` 和 `video_overlay_<Robot>_<setup>.mp4`。默认每个示范从配置中的机器人池选择一个形态；`--all-robots` 才会为所有形态渲染。导出数据可按[官方说明](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/README_zh.md#使用)安装 `.[lerobot]` 并调用 `scripts/convert_to_lerobot.py`，注意导出位姿为世界坐标，需按下游模型约定转换。

原始视频任务如使用云端视频标注，官方说明要求显式启用并提供服务密钥；使用本地标注可跳过 `subtask` 阶段。密钥仅保存在个人环境，不写入仓库或公开教程。

### 论文与代码如何对应

| 论文步骤 | 已检查源码入口 |
| --- | --- |
| 管线阶段与调度 | [`ego2robot/pipeline.py`](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/ego2robot/pipeline.py) |
| 手部动作与平滑 | [`action_processor.py`](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/ego2robot/processors/action_processor.py)、[`smoothing_processor.py`](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/ego2robot/processors/smoothing_processor.py) |
| 人手分割、背景修复、机器人合成 | [`processors/`](https://github.com/www-Ye/ego2robot_blog/tree/244a4ee02f02b2e93e94a81f27963597e0815ce6/ego2robot/processors) |
| 数据配置及评测 | [`configs/`](https://github.com/www-Ye/ego2robot_blog/tree/244a4ee02f02b2e93e94a81f27963597e0815ce6/configs) · [`evaluation/robotwin/`](https://github.com/www-Ye/ego2robot_blog/tree/244a4ee02f02b2e93e94a81f27963597e0815ce6/evaluation/robotwin) |

论文描述的统计筛选和视觉语言模型复核，在本次检查的阶段注册表中没有作为独立阶段列出；仓库使用说明也没有给出完整预训练入口。运行公开管线与复现论文的全部数据筛选、训练结果，需要分别核对。仓库当前未标明统一的项目许可证，机器人资源及第三方模型还需查看各自条款；本教程直接链接官方图片和视频。

## 6. 延伸阅读

- [Ego2Robot 论文](https://arxiv.org/abs/2608.02580)与[项目页](https://www-ye.github.io/ego2robot_blog/)
- [官方源代码及中文运行说明](https://github.com/www-Ye/ego2robot_blog/blob/244a4ee02f02b2e93e94a81f27963597e0815ce6/README_zh.md)
- [本站 EBench 数据与评测教程](../04-EBench.md)
- [本站第一视角手部到机器人动作重定向](../../07-机器人操作、运动控制/Manipulation/02-MANO第一视角到机器人动作重定向/README.md)
