# MotionBricks 与 CrowdBricks：从关键姿态到实时群体动作

如何用同一套动作生成模型，让角色走路、跳跃、坐下，再让大量角色并行运动？本章梳理 NVIDIA 的 **MotionBricks** 和社区项目 **CrowdBricks**：前者负责根据姿态约束生成连续动作，后者把人体动画的生成、缓存和调度组织成批处理。

本文根据论文、项目页面及公开代码整理，**没有在本机下载模型或运行演示**。视频和性能数字均来自相应项目作者。

## 1. 先看效果

### 1.1 人形机器人动作

[![官方 Unitree G1 关键姿态插值演示](https://media.githubusercontent.com/media/NVlabs/motionbricks/main/docs/media/inbetweener_g1.gif)](https://research.nvidia.com/labs/gear/motionbricks/videos/motionbricks_teaser_demo_robotics_1080p.mp4)

上图是官方关键姿态插值的动态示例；点击可播放[官方机器人综合视频](https://research.nvidia.com/labs/gear/motionbricks/videos/motionbricks_teaser_demo_robotics_1080p.mp4)，观察仿真及真机的人形动作。视频来自 MotionBricks 项目，不是本机录制。图片与视频来源：[NVlabs 源码库](https://github.com/NVlabs/motionbricks)、[官方项目页](https://nvlabs.github.io/motionbricks/)；官方文档及媒体采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。

### 1.2 动画与场景交互

[![官方角色场景预览](https://media.githubusercontent.com/media/NVlabs/motionbricks/main/docs/media/scene_preview_soma.gif)](https://research.nvidia.com/labs/gear/motionbricks/videos/aibm_runtime_main_demo_latest_art_v3_1080p.mp4)

上图是官方场景预览；点击可播放[官方完整动画演示](https://research.nvidia.com/labs/gear/motionbricks/videos/aibm_runtime_main_demo_latest_art_v3_1080p.mp4)，其中展示不同移动风格和环境交互。[官方项目页](https://nvlabs.github.io/motionbricks/)还提供跨越长椅、坐下、拾取物体等分项视频。

### 1.3 多角色动作分叉

![CrowdBricks 的多角色分叉与动作规划](https://raw.githubusercontent.com/patch-gg/crowdbricks/main/docs/futures.webp)

图源：[CrowdBricks 项目](https://github.com/patch-gg/crowdbricks)。其中一个角色分出 48 条共享过去的未来轨迹，粉色线条展示尚未播放的动作计划。项目另有 `examples/horde.py`：192 个角色追逐移动目标；示例说明和播放方式见[项目说明](https://github.com/patch-gg/crowdbricks#quick-start)。

## 2. 论文、项目与各自职责

| 资料 | 内容 |
| --- | --- |
| [MotionBricks 论文](https://arxiv.org/abs/2604.24833) · [官方论文页面](https://doi.org/10.1145/3811334) | 2026 年发表于 SIGGRAPH 与 ACM Transactions on Graphics；研究动作生成骨干及可组合的交互接口 |
| [MotionBricks 官方代码](https://github.com/NVlabs/motionbricks) · [官方项目页与视频](https://nvlabs.github.io/motionbricks/) | 人体与 Unitree G1 动作生成模型、运行时、场景演示和编辑器 |
| [NVIDIA 模型权重](https://huggingface.co/nvidia/MotionBricks) · [G1 模型卡](https://huggingface.co/nvidia/MotionBricks/blob/main/g1/README.md) | 人体和 G1 的预训练模型、输入输出和使用条件 |
| [CrowdBricks 代码](https://github.com/patch-gg/crowdbricks) | 社区构建的批量生成、动作缓存和多角色调度系统 |
| [早期 G1 预览与训练代码](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks) | 官方机器人预览、合成数据训练代码和数据格式说明 |

独立的 MotionBricks 发布仓库提供推理、导出和演示，模型卡说明不包含完整训练流水线。官方较早的机器人预览仓库另有合成数据训练代码；[项目说明](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks#training)列出了训练与数据入口。

## 3. 完整流程：从控制意图到连续动作

```text
应用侧：移动方向、速度、风格或物体交互目标
        ↓
智能动作原语：组织对象与角色的相对位置，给出 1–4 个目标姿态
        ↓
最近 4 帧已播放姿态 + 目标姿态 + 时长等约束
        ↓
根运动模块：预测动作时长和骨盆根轨迹
        ↓
姿态模块：结合根轨迹与姿态约束，预测离散动作编码
        ↓
解码器：还原连续骨架运动及接触信息，处理姿态衔接
        ↓
播放动作；把实际播放的最近 4 帧作为下一段上下文
```

模型每次生成一段短动作，运行时通过重复生成把短片段衔接起来。以官方运行时为例，每个离散动作编码代表 4 帧，模型选择 6–16 个编码；输出帧率为 30 赫兹，一次生成的新动作约为 0.6–1.9 秒。应用程序决定目标从哪里来，因此相同生成接口既能做姿态插值，也能连接移动与物体交互。详见[官方运行时说明](https://github.com/NVlabs/motionbricks/blob/main/docs/runtime_api.md)。

### 3.1 分模块动作生成

1. **姿态编码与解码**：向量量化变分自编码器将运动压缩成离散表示；多个码本头共同表示动作，解码时恢复连续姿态和运动特征。
2. **根运动预测**：先处理动作持续时间和角色根部路径，给全身运动提供位置与朝向条件。
3. **全身姿态预测**：掩码式姿态 Transformer［注意力序列模型］结合根轨迹和关键姿态约束，预测各时刻的离散动作编码，再由解码器还原全身动作。
4. **衔接和播放**：运行时按约束处理生成结果的首尾，在播放过程中保留历史姿态，为下一段生成提供上下文。

论文的关键设计是**多码本动作表示、由粗到细的模块化生成，以及统一的关键姿态约束接口**。这使底层模型可供多类行为复用；具体要走向哪里、何时与物体交互，仍由上层应用和动作原语确定。[论文](https://arxiv.org/abs/2604.24833)第 3–4 节介绍模型与原语。

### 3.2 两类智能动作原语

| 原语 | 输入与功能 | 例子 |
| --- | --- | --- |
| 智能移动 | 根据目标速度、朝向和风格构造代理关键姿态，交给生成模型补出自然过渡 | 站立、行走、奔跑、侧移和风格切换 |
| 智能物体 | 将目标姿态、交互区域与物体局部坐标绑定；角色接近后按序生成交互动作 | 跨越障碍、坐下、拾取物体 |

例如，跨越障碍时先定义障碍的交互区域及跨越所需的目标姿态；角色接近后，上层选择合适的原语和姿态序列，模型再生成接近、跨越与后续动作。原语提供的是可编辑的条件和时序，路径规划及机器人动力学执行需要由应用层另行组织。

## 4. CrowdBricks：让多个角色共享生成调用

CrowdBricks 是基于 MotionBricks 的社区工程，不是论文的新模型。它复用原有的人体模型，将整条生成路径与多角色运行时扩展为批量处理：

```text
每个角色：当前动作 / 下一目标 / 播放进度
                  ↓
Crowd［群体调度器］收集接近衔接点的请求
                  ↓
BatchGenerator［批量生成器］整理多角色姿态及目标
                  ↓
Warp［并行计算框架］批量编码 → TensorRT［推理引擎］批量推理 → 批量解码与端点修正
                  ↓
显存动作库保存结果 → 各角色按自己的时间播放 → 到衔接点继续请求
```

| 组件 | 具体作用 | 阅读入口 |
| --- | --- | --- |
| 批量生成器 | 同一批请求可有不同目标、风格和时长；不足一批时补齐，丢弃补齐项 | [`batch.py`](https://github.com/patch-gg/crowdbricks/blob/main/crowdbricks/batch.py) |
| 并行算子 | 批量计算坐标变换、姿态条件编码、解码和端点修正 | [`kernels.py`](https://github.com/patch-gg/crowdbricks/blob/main/crowdbricks/kernels.py) |
| 群体调度 | 移动类动作允许最多约 0.15 秒的提前请求窗口；批次已满或某角色将耗尽动作时执行。对交互这类时序敏感的动作，在指定衔接点生成 | [`crowd.py`](https://github.com/patch-gg/crowdbricks/blob/main/crowdbricks/crowd.py) |
| 动作库与播放 | 每个角色保留自身生成片段，播放时采样；分叉角色可以复制已有运动历史 | [`batch.py`](https://github.com/patch-gg/crowdbricks/blob/main/crowdbricks/batch.py) · [`futures.py`](https://github.com/patch-gg/crowdbricks/blob/main/examples/futures.py) |

**批处理的关键收益**是摊薄模型调用和前后处理的开销。较小批次需要填充到固定容量，等候凑批也有时延和动作衔接上的权衡；这些因素在实际群体规模、动作长度与交互密度变化时都值得测量。

## 5. 性能数字怎么读

以下是 CrowdBricks 作者在 RTX 4090［图形处理器］、FP16［半精度浮点］下，运行 50 次后的中位数。每次请求都是从站立开始生成跑步动作；表内吞吐量由每批角色数除以每批墙钟时间得到。来源：[项目性能表](https://github.com/patch-gg/crowdbricks#numbers)、[基准脚本](https://github.com/patch-gg/crowdbricks/blob/main/benchmarks/throughput.py)。

| 测试 | 批大小 | 每批墙钟时间 | 每秒完成的动作生成请求 |
| --- | ---: | ---: | ---: |
| 原版单角色接口 | 1 | 2.57 毫秒 | 389 |
| CrowdBricks | 16 | 3.38 毫秒 | 4,727 |
| CrowdBricks | 48 | 4.66 毫秒 | 10,307 |

10,307 ÷ 389 ≈ **26.5 倍**，比较对象是上述单角色接口和批大小为 48 的完整批次。这里统计的是**短动作生成请求数**，不包含同时渲染一万名角色；每个请求会产生一个包含多帧的动作片段。

群体演示是另一组指标：作者报告 192 个角色以 60 赫兹更新，约每秒 215 次生成，使用大小为 16 的批次；整个群体单步更新的中位数约 2.4 毫秒。它与上面的固定批次吞吐量测试不是同一个测量场景。

MotionBricks 原始模型及其导出接口也支持批维度；CrowdBricks 的贡献是将编码、解码、动作缓存和请求调度组织为多角色可用的整体。比较不同项目时，应分别核对批大小、输入、预热、精度、计时范围和是否包含渲染。

## 6. 人体动画与机器人控制

| 项目 | 角色表示 | 输出和适用范围 |
| --- | --- | --- |
| MotionBricks 人体模型 | SOMA 人体骨架 | 人体动画姿态与动作片段 |
| MotionBricks G1 模型 | 34 个模型关节；可转换为浮动根位姿和 29 个机器人关节角 | G1 参考动作；真实机器人执行还需要单独控制器 |
| CrowdBricks 当前批量生成器 | 代码固定使用 SOMA 人体模型 | 多角色人体动画；迁移到 G1 需适配模型、骨架和相关算子 |

G1 预训练组件约 1.95 亿参数：姿态编码器及解码器约 2,380 万、姿态预测器约 1.366 亿、根运动预测器约 3,410 万，见 [G1 模型卡](https://huggingface.co/nvidia/MotionBricks/blob/main/g1/README.md)。论文报告基础数据包含超过 35 万个动作片段；不同模型版本的筛选与拆分请以对应模型卡为准。

从机器人应用出发，可以借鉴**批量参考动作生成、显存动作库和按需调度**。若要在仿真或真机上执行，还要接入目标机器人运动学、碰撞检查、动力学控制和安全约束。论文展示了官方 G1 真机系统；这里引用的 CrowdBricks 群体实现本身使用人体模型，且项目说明其简化了上游手脚逆运动学和输入平滑。

### 数值一致性

[项目测试](https://github.com/patch-gg/crowdbricks/blob/main/tests/test_matches_upstream.py)将批量结果与上游单角色接口比较：单样本 FP32［单精度浮点］测试限制根位置最大分量误差小于 0.1 毫米、最大关节旋转误差小于 0.2 度；较大批次 FP16［半精度浮点］测试限制根位置最大分量误差小于 5 厘米、平均关节旋转误差小于 1 度，并输出但未断言最大单关节误差。这些是代码中的测试阈值，迁移到具体控制任务时应按任务自行评估。

## 7. 继续阅读与使用许可

1. 看[官方项目页](https://nvlabs.github.io/motionbricks/)的机器人视频、移动风格与物体交互，再看[论文](https://arxiv.org/abs/2604.24833)的方法和实验。
2. 阅读[官方入门教程](https://github.com/NVlabs/motionbricks/blob/main/docs/guided_tutorial.md)，了解姿态插值、场景与原语编辑器；需要执行时按照[官方安装说明](https://github.com/NVlabs/motionbricks#setup)准备环境。
3. 从 [CrowdBricks 项目说明](https://github.com/patch-gg/crowdbricks#quick-start)进入 `examples/futures.py` 和 `examples/horde.py`，再阅读批量、调度和性能测试代码。
4. 软件代码、模型、示例动作和媒体各有许可：代码为 Apache 2.0；权重受 NVIDIA Open Model Agreement 约束；部分动作样例受 NVIDIA Sample Data Evaluation License 约束。使用前查看[官方许可说明](https://github.com/NVlabs/motionbricks#license)和 [CrowdBricks 许可说明](https://github.com/patch-gg/crowdbricks#licenses)。
