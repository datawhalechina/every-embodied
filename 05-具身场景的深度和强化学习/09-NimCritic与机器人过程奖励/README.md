# 机器人过程奖励：NimCritic 与视频进度模型

**抓住、滑落、重新抓取、放好：怎样给每一步提供有意义的反馈？** 本章从过程奖励的基本概念出发，介绍 NimCritic 的标注、训练与评测，并对照 Robo-Dopamine、Robometer、RoboReward 等相关工作。文末有官方演示、代码和学习入口。

[打开互动学习网页](./index.html) · [NimCritic 论文](https://arxiv.org/abs/2610.32776) · [NimCritic 模型与推理代码](https://huggingface.co/RimbotAI/NimCritic) · [作者项目页](https://rimbot.com/research/nimcritic/) · [在线体验](https://critic.rimbot.com/)

## 1. 先看视频：机器人动作与进度曲线

下面的视频由各自论文作者发布。观看时注意：曲线是否随完成上升，在物体掉落、动作失败时能否下降。

| 官方演示 | 画面与观察重点 | 来源 |
| --- | --- | --- |
| [观看插座放入盒中](https://rimbot.com/image/research/nimcritic/power-strip-in-box.mp4) | NimCritic 单任务原始视频；到作者页面可同步查看进度曲线 | [NimCritic 交互演示](https://rimbot.com/research/nimcritic/) |
| [观看堆叠三色方块](https://rimbot.com/image/research/nimcritic/stack-cubes_source.mp4) | NimCritic 多任务原始视频；作者页面还可切换袜子收纳、路由器接线等任务 | [NimCritic 交互演示](https://rimbot.com/research/nimcritic/) |
| [观看折裤子进度预测](https://robo-dopamine.github.io/assets/videos/grm_demo_1.mp4) | Robo-Dopamine 在动作过程中估计进度，观察视频与曲线的对应关系 | [Robo-Dopamine 项目页](https://robo-dopamine.github.io/) |
| [观看倾倒失败检测](https://robometer.github.io/assets/failure_detection/failure_detection_1.mp4) | Robometer 检测倾倒过程中的颗粒洒落，画面右侧给出进度曲线 | [Robometer 项目页](https://robometer.github.io/) |
| [观看勺子掉落检测](https://robometer.github.io/assets/failure_detection/failure_detection_2.mp4) | 观察操作失败时曲线的响应 | [Robometer 项目页](https://robometer.github.io/) |
| [观看真机在线训练](https://robometer.github.io/assets/online_rl/robometer_auto_online_rl_small.mp4) | 作者的 Robometer 在线强化学习演示，原视频标注 **100 倍速** | [Robometer 训练实验](https://robometer.github.io/) |
| [查看 NimCritic 官方演示与在线体验](https://rimbot.com/research/nimcritic/) | 任务进度与完成预测；[在线体验入口](https://critic.rimbot.com/) | [日冕开物](https://rimbot.com/research/) |

上面不同项目的视频展示的是各自的方法，不能当作 NimCritic 在同一任务上的对照实验。NimCritic 原视频与进度曲线由作者网页同步展示：点击[官方项目页](https://rimbot.com/research/nimcritic/)的单任务或多任务卡片即可切换；直接打开的视频文件只显示画面。

<p align="center"><img src="https://robometer.github.io/assets/example_trajs/example_1.gif" width="220" alt="Robometer 官方机器人操作轨迹示例一"> <img src="https://robometer.github.io/assets/example_trajs/example_4.gif" width="220" alt="Robometer 官方机器人操作轨迹示例二"> <img src="https://robometer.github.io/assets/example_trajs/example_6.gif" width="220" alt="Robometer 官方机器人操作轨迹示例三"></p>

*Robometer 作者展示的不同操作轨迹样例；来源：[官方项目页](https://robometer.github.io/)。*

## 2. 什么是过程奖励？

终局奖励只问「最后有没有完成」，例如放进盒子才给 1 分。过程反馈还关注「是否靠近目标」：完成抓取、靠近盒子、滑落、重抓、放稳，都可以在过程中留下不同信号。这能帮助分析失败片段、筛选示范，或给训练提供更密集的信息。

```text
任务指令 + 机器人画面 → 过程模型 → 进度 / 完成预测
                                         ├─ 轨迹筛选与故障分析
                                         ├─ 候选动作或轨迹比较
                                         └─ 设计奖励接口 → 强化学习更新策略
```

| 输出或用途 | 它回答的问题 | 示例 |
| --- | --- | --- |
| 终局成功 | 最后是否达到目标？ | 积木是否最终在盒中 |
| 进度 `p(t)` | 当前任务大概完成多少？ | `0 → 0.4 → 0 → 0.4 → 1` |
| 进度变化 `Δp` | 这一步让进度增加还是减少？ | `+0.4、−0.4、+0.4、+0.6` |
| 训练奖励 `r` | 策略优化实际使用什么反馈？ | 终局奖励与经过设计的过程反馈共同构成 |
| 动作价值 `Q` / 状态价值 `V` | 对未来累计回报的估计 | 与当前进度相关，但定义和训练目标不同 |

上面数字只用于说明，不是任何模型的实测预测。**进度、进度差和强化学习奖励是不同的量**：高进度反复计为每一步奖励，可能鼓励策略停留在高分状态；把进度变化转换为奖励时，还需要考虑折扣、成功判定、终止处理与模型误判。

一种常见的势函数奖励塑形是 `r′(s,a,s′) = r(s,a,s′) + γΦ(s′) − Φ(s)`：`Φ` 表示对状态的势函数估计，`γ` 为折扣因子。在理论条件满足、终止状态处理一致时，该形式可保持最优策略；真实系统里的学习模型会有误差、观测和历史依赖，需要按实际任务验证。[Robo-Dopamine 方法说明](https://robo-dopamine.github.io/)讨论了如何将过程评估用于奖励塑形。

## 3. NimCritic：从动作效果计算进度

![NimCritic 论文架构图](https://arxiv.org/html/2610.32776v1/fig2_revision4_20261023.png)

*架构与进度标注以[论文原文](https://arxiv.org/html/2610.32776)为准；图像来自作者在论文中的公开图示。*

### 3.1 标签怎样得到

作者把同一物体的连续操作看作一个子任务。视觉语言模型根据任务与视频识别子任务、动作边界、结果、撤销关系，以及未完成时还需哪些动作；人工核验时间边界和部分尝试、撤销的时点，再由规则计算进度。训练集抽样核验，LOOM 测试集的每条轨迹均进行时间校验。[论文第 3.2 节与附录 A](https://arxiv.org/html/2610.32776)。

| 动作效果 | 标注规则的直观含义 |
| --- | --- |
| 有效推进 | 获得该动作类别的完整权重 |
| 有接触但未完成 | 暂记一半权重，后续可撤销 |
| 无效动作 | 不增加进度 |
| 破坏先前成果 | 撤销对应的历史贡献；重新完成后再取得贡献 |

动作类别权重来自数据集内该类别的归一化持续时间。子任务预算综合已观察动作的净贡献及任务尚缺的必要动作，单个动作在其起止区间内线性分配贡献，局部进度由累计保留贡献除以该固定预算；跨子任务的撤销会更新原来所属子任务的贡献。

例如仅为理解规则，设抓取与放置权重各为 1：抓取完成约 50%，物体滑落撤销抓取贡献后回到 0%，重新抓住回到 50%，放好达到 100%。真实标签取决于动作类别、分段及完整预算；这组数值不是论文给出的动作权重。可在[互动网页](./index.html#progress-lab)逐步点击观察。

### 3.2 模型怎样学习和推理

训练时，NimCritic 将 Wan2.2 视频专家与进度专家组成双流模型，通过联合视频预测与进度 / 完成预测学习视觉变化和任务进展；历史进度扰动帮助模型处理上一窗口的预测偏差，视频与进度的对比对齐让中间视觉特征关联进度轨迹。进度和是否完成分别有输出。该架构借鉴 WVM；动作感知标注和上述训练设计共同组成这项工作。

论文的长程评测先用视觉语言模型将**完整视频**切分为物体级子任务，再对子任务评分并按时长及子任务出口进度合成全局曲线。模型以 2 Hz 采样、五帧窗口、20 步去噪进行评分，推理时重用视觉特征，无须生成未来视频。模型仓库提供的是**单视频评分脚本**，不等于论文中完整的长视频切分、合成与评测程序。[论文第 3 节](https://arxiv.org/html/2610.32776) · [模型仓库说明](https://huggingface.co/RimbotAI/NimCritic)。

## 4. 数字怎么读

| 结果 | 作者报告 | 正确理解 |
| --- | --- | --- |
| LOOM 整体 / 多子任务 `PRC-F1` | `0.86 / 0.95` | 正向和反向进度排序相关性的组合指标，不是二分类的精确率 / 召回率 |
| VPB 整体 `PRC-F1` / 终局成功准确率 | `0.77 / 0.82` | 分别衡量过程排序与终局完成判定 |
| RoboArena 成功轨迹召回率 | `96.72%` | 人工选出的成功轨迹被找回的比例，不是机器人成功率 |
| 移除子任务切分 | 多子任务 `PRC-F1` 从 `0.95` 降到 `0.55` | 同一模型的消融，长任务分解对结果很重要 |
| 相同子任务切分 | NimCritic `0.95`，GPT-6-Sol `0.96` | 两者在该进度指标接近；完成准确率分别为 `0.93 / 0.65` |

RoboArena 包含 3,872 个评测场次和 10,750 条轨迹，论文给出 1,220 条人工选出的成功轨迹；根据分组召回率可还原约 1,180 条被找回，`1,180 / 1,220 ≈ 96.72%`。这项结果单独没有提供成功候选的精确率，不能由召回率推出误选了多少失败轨迹。论文原始结果和统计方法见[第 4 节与附录 B](https://arxiv.org/html/2610.32776)。

## 5. 相关工作：同样评分，侧重点不同

| 方法 | 输入与输出 | 独到做法 | 演示 / 资源 |
| --- | --- | --- | --- |
| **NimCritic** | 视频、任务 → 进度曲线和完成预测 | 动作效果积分、历史撤销与恢复；视频预测和进度联合学习；长程任务按物体分解 | [论文](https://arxiv.org/abs/2610.32776) · [代码和权重](https://huggingface.co/RimbotAI/NimCritic) |
| **Robo-Dopamine** | 多视角的初始、目标、前后状态 → 相对推进 / 倒退 | 离散化相对进度并融合多种参考；结合奖励塑形用于在线强化学习 | [项目与视频](https://robo-dopamine.github.io/) · [代码](https://github.com/FlagOpen/Robo-Dopamine) |
| **Robometer** | 视频、任务 → 逐帧进度 / 成功；还能比较两条轨迹 | 同时用专家进度监督和轨迹偏好监督，利用次优和失败数据；探索筛选、规划与强化学习 | [项目与视频](https://robometer.github.io/) · [代码](https://github.com/robometer/robometer) |
| **RoboReward** | 视频、任务 → 1–5 档轨迹评价 | 基于机器人视频构造不同质量的样本；用语言目标重标和截断生成失败或未完成样本 | [论文](https://arxiv.org/abs/2601.00675) · [模型](https://huggingface.co/teetone/RoboReward-8B) |
| **RynnValue** | 视频、任务 → 距离任务完成的剩余时间等信息 | 利用时间距离构造监督，可用于进度估计、轨迹排序和离线数据筛选 | [代码和模型入口](https://github.com/alibaba-damo-academy/RynnValue) |

这些方法输出的数值、输入画面、训练数据和评测协议不同，跨论文的指标不能直接拼成统一排行榜。Robo-Dopamine 作者报告其单示范适配后的一个真机训练例子在约 150 条在线轨迹后达到 95% 成功率；Robometer 作者展示了在线与离线训练、失败检测等应用。它们说明过程模型可以进入策略训练，而 NimCritic 论文的主要直接证据集中在进度预测与轨迹筛选评测。[Robo-Dopamine 项目页](https://robo-dopamine.github.io/) · [Robometer 项目页](https://robometer.github.io/)。

## 6. 上手 NimCritic：单视频评分

[官方模型仓库](https://huggingface.co/RimbotAI/NimCritic)提供推理代码、权重、视觉自编码器及文本编码器；模型仓库文件合计约 **24.84 GB**。该版本的模型卡标注 Apache-2.0，建议至少 24 GB 显存；实际资源消耗随环境变化。以下以官方仓库内容为准，先准备足够磁盘空间和隔离环境：

```bash
# 需要已安装 Hugging Face 命令行工具 hf 和 uv
export NIM_DIR=/path/to/data/nimcritic
export UV_PROJECT_ENVIRONMENT=/path/to/envs/nimcritic
hf download RimbotAI/NimCritic \
  --revision e2bcfdaef1826a2bbdf4be467d8420b8e241c789 \
  --local-dir "$NIM_DIR"
cd "$NIM_DIR"
uv sync

# 无需加载权重的抽帧、缩放与参数自检
uv run python infer_nimcritic.py --self-check

# 对一个视频计算逐帧进度和完成输出
uv run python infer_nimcritic.py \
  --video /path/to/clip.mp4 \
  --task "pick up the cup" \
  --output result.json \
  --rollout-stride 2
```

`--video` 默认按 2 Hz 抽帧，`--frames` 则接收已抽样帧；默认 `--rollout-stride` 是 1，上面的 2 对应论文短任务表设置。输出 JSON 中 `progress`、`success` 是 0–1 的逐帧值，`sampling` 记录抽帧；具体完成阈值应在目标验证集校准，不应把任意阈值直接当成通用成功标准。

**本机核验范围：**已获取固定版本的论文与 `infer_nimcritic.py`、`nimcritic_model.py`、`pyproject.toml` 和模型卡，在现有环境运行 `--self-check`，结果为 `self-check ok`。没有下载模型权重，没有做模型推理或实时速度测试。模型仓库未提供本论文的完整训练和子任务分解流水线；截至本次核查，[LOOM 数据集仓库](https://huggingface.co/datasets/RimbotAI/LOOM)仅列出说明文件和仓库配置，未列出评测数据文件。

## 7. 怎样在自己的任务中使用

1. **先定义终局标准。** 例如「物体稳定位于盒内」；记录成功、失败、滑落、重试和接近但未完成的例子。
2. **用已有视频离线查看曲线。** 对照事件时点检查进度是否上升、失败时是否下降、任务未完成时是否误报完成；保留与人工标签的逐条对比。
3. **选阈值并同时看查全与查准。** 召回率检查成功轨迹漏掉多少；精确率检查选出的轨迹里混入多少失败。失败少时要特别关注误报。
4. **再接入下游应用。** 可先用于数据排序和失败检索，再考虑候选选择或训练奖励；接入在线训练前测量实际延迟、掉帧和任务成功率，并设定超时行为。

相关教程：[HIL-SERL 人机协同强化学习](../08-HIL-SERL人机协同强化学习与MuJoCo插孔/README.md) · [奖励对齐与人类纠正示范](../04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md) · [Ego2Robot 人类视频数据合成](../../09-具身智能数据及评估基准benchmark/05-Ego2Robot第一视角人类视频转机器人数据/README.md)。

## 8. 论文与官方材料

- [NimCritic 论文、附录及实验表](https://arxiv.org/abs/2610.32776) · [模型及推理代码](https://huggingface.co/RimbotAI/NimCritic)
- [Robo-Dopamine 论文](https://arxiv.org/abs/2512.23703) · [官方项目和视频](https://robo-dopamine.github.io/)
- [Robometer 论文](https://arxiv.org/abs/2603.02115) · [官方项目和视频](https://robometer.github.io/)
- [RoboReward 论文](https://arxiv.org/abs/2601.00675) · [RoboRewardBench](https://crfm.stanford.edu/helm/robo-reward-bench/)
