# APXInf：具身模型端侧推理引擎

## 这一章完成什么

这一章从系统角度认识 APXInf（ApxInf）：它为什么针对具身模型单独做 Runtime、它和 RLinf、OpenPI、EVA-Client、Jetson-PI 以及 TensorRT 的边界在哪里，以及如何按照官方流程完成一次最小化的 π0.5 推理验证。

这里关注的是“模型如何在机器人本体上稳定地产生动作”，不是训练一个新的 VLA 模型。APXInf 的官方开源载体是 [RLinf/APXinf-robo](https://github.com/RLinf/APXinf-robo)，其中通过 Git submodule 引入 ApxInf 引擎。

## 1. 为什么 VLA 需要专门的端侧 Runtime

VLA 的训练负载和机器人部署负载差异很大。训练阶段通常使用大显存 GPU 和较大的 batch；部署阶段则常见于 Jetson Orin、Jetson Thor 或单张 RTX 4090，batch 通常为 1，而且必须不断重复下面的闭环：

```text
相机图像 + 语言指令 + 机器人状态
                ↓
          VLA / WAM 推理
                ↓
          action chunk
                ↓
          控制器与机器人
```

对于 π0.5 这类使用 flow matching 的策略，生成一个 action chunk 还包含多步采样。模型前向、图像预处理、采样、显存搬运、Python 调度和动作后处理都会进入控制闭环。端侧部署的目标因此不是单一算子的峰值 FLOPS，而是固定输入形状、小 batch、连续运行条件下的端到端延迟和稳定性。

## 2. APXInf 的系统位置

APXInf 应当理解为“具身模型推理 Runtime”，而不是完整的机器人操作系统：

```text
OpenPI / 自定义策略
        ↓  checkpoint 与观测
APXInf Runtime
  Rust 调度 + CUDA Kernel + CUDA Graph + 量化
        ↓  normalized actions
APXinf-robo policy / OpenPI-compatible server
        ↓
机器人客户端或评测环境
```

![APXInf 从 Python 接口、Rust Runtime 到高性能算子库和端侧硬件的分层结构](./assets/official_images/apxinf-figure-1.png)

图 1：APXInf 的官方分层示意。上层保留 Python 模型加载、前后处理和推理调用方式，中间由 Rust Runtime 负责调度与资源生命周期，底层再连接面向 Jetson/RTX 和具身模型定制的高性能算子库。

来源：[无问芯穹 APXInf 官方介绍](https://www.infinigence-ai.com/news-updates/233.html)。

官方仓库首发重点是 π0.5，并提供 BF16、FP8 和 INT8 路径。当前公开支持的硬件包括 Jetson AGX Thor、Jetson AGX Orin 和 RTX 4090；FP8 需要校准数据，INT8 主要面向 Orin 和 Ada 架构。具体支持范围应以目标版本的 README 为准。

### 表 1 相关项目的边界

| 项目 | 主要位置 | 是否等同于 APXInf |
| --- | --- | --- |
| RLinf | 具身 RL 训练、rollout、仿真和评测基础设施 | 否，属于训练/评测上层 |
| OpenPI | π0/π0.5 模型、预处理和策略服务参考实现 | 否，属于模型与策略栈 |
| APXInf | 小 batch、低延迟的具身模型端侧推理 Runtime | 本章主体 |
| EVA-Client | 连接策略服务和真实机器人的部署、采集、评测客户端 | 否，属于真机执行层 |
| Jetson-PI | 通过异步推理和前瞻校正隐藏 VLA 延迟的研究方案 | 否，属于控制/调度方法 |
| TensorRT / TensorRT-LLM | 通用 NVIDIA 推理后端 | 可作为底层路线，但不是具身专用闭环 |

EVA-Client 的论文和代码见 [项目仓库](https://github.com/Noietch/EVA-CLIENT) 与 [论文](https://arxiv.org/abs/2607.02646)；Jetson-PI 的论文和 Edge Runtime 见 [论文](https://arxiv.org/abs/2607.12659)；Embodied.cpp 见 [项目仓库](https://github.com/SEU-PAISys/Embodied.cpp) 和 [论文](https://arxiv.org/abs/2607.02501)。这些项目的定位不同，不能仅凭“都提到 inference”就当作同一个框架。

## 3. APXInf 值得看的地方

### 3.1 面向端侧闭环，而不是只优化裸模型

APXInf 的公开 benchmark 把测量分成 L1、L2、L3：L1 是从 resized RGB 到模型输出，L2 包括 policy 的预处理和后处理，L3 还包括服务端往返。这种拆分有助于判断瓶颈究竟在 CUDA 计算、Python/Rust 调用、策略封装，还是通信链路。

官方 README 给出的 π0.5、batch=1、两路视角、10 个 flow steps、action horizon 为 10 的示例数据如下。它们是官方测量结果，不代表任意模型和任意输入形状都能达到相同延迟：

| 硬件 | 精度 | 基础延迟 | onestep pruning 后 |
| --- | --- | ---: | ---: |
| Jetson AGX Thor | BF16 | 72.45 ms | 44.05 ms |
| Jetson AGX Thor | FP8 | 41.16 ms | 26.32 ms |
| Jetson AGX Orin | BF16 | 165.67 ms | 119.05 ms |
| RTX 4090 | BF16 | 31.38 ms | 20.36 ms |

![PI 0.5 在 Jetson AGX Thor 上从基线到端到端优化的延迟拆解](./assets/official_images/apxinf-figure-2.png)

图 2：官方给出的优化链路。延迟逐步下降来自融合算子、静态图、CUDA Graph、量化、Kernel 优化和动作生成裁剪的组合；不能把最后的 26 ms 简单归因于 Rust 或某一个 CUDA Kernel。

来源：[无问芯穹 APXInf 官方介绍](https://www.infinigence-ai.com/news-updates/233.html)。

Thor FP8 的 26.32 ms 对应约 38 Hz，说明它的优化目标确实是机器人实时控制，而不是只追求离线吞吐。[官方性能表](https://github.com/RLinf/APXinf-robo#performance)

### 3.2 Rust Runtime 与 CUDA Kernel 分工清楚

Python API 负责模型调用和上层集成，Rust 负责更贴近系统的运行时组织，CUDA Kernel 负责 GPU 计算。Rust 并不是具身智能算法本身，它的价值在于减少运行时资源管理风险、收敛接口和调度开销；真正的性能收益仍然来自图捕获、融合算子、精度路径、内存复用和模型结构特化。

### 3.3 OpenPI 兼容降低了替换成本

APXinf-robo 提供 OpenPI-compatible WebSocket server。已经使用 `openpi-client` 的客户端可以把服务地址切换到 APXInf，而不必重写整个观察和动作接口。这是它从“一个 benchmark 优化实现”走向部署组件的重要工程设计。

## 4. 最小验证流程

### 4.1 环境要求

官方构建流程需要 Linux、NVIDIA 驱动、CUDA Toolkit、Rust 和 CMake。编译会读取当前可见 GPU 的 compute capability，并为目标架构构建 CUDA Kernel；因此最好直接在部署机器上构建。跨机器构建时需要显式设置 `APXINF_CUDA_ARCH`，例如 Orin 为 `sm_87`。

下面命令在 APXinf-robo 仓库根目录执行。它们是官方构建流程的整理版，不包含模型权重：

```bash
git clone --recursive https://github.com/RLinf/APXinf-robo.git
cd APXinf-robo

python3 -m venv .venv
source .venv/bin/activate
pip install maturin

CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda \
  --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
pip install -e "./apxinf/python/apxinf[serving]" --config-settings editable_mode=strict
pip install -e ".[serve]"
```

### Checkpoint 1：绑定和包可以导入

```bash
python -c 'import apxinf_py; print(apxinf_py.__version__)'
python -c 'import apxinf_robo; print(apxinf_robo.__version__)'
```

这一步只证明 Python 扩展和上层包安装成功，不证明模型输出正确，也不证明端到端延迟达到官方结果。如果导入失败，优先检查 Rust、CUDA、CMake 版本以及当前 GPU 架构是否和构建目标一致。

### Checkpoint 2：用随机权重验证运行时链路

```bash
python scripts/bench_pi05.py \
  --random-weights \
  --precision bf16 \
  --layer l1 \
  --views 2 \
  --token-count 10 \
  --action-horizon 10 \
  --num-flow-steps 10 \
  --warmup 10 \
  --samples 30 \
  --autotune
```

随机权重 benchmark 适合检查 CUDA Kernel、输入形状、图捕获和计时链路。它不验证模型效果，也不能替代真实 checkpoint 的 LIBERO 或真机评测。

### Checkpoint 3：使用真实 checkpoint

真实评测需要兼容的 π0.5 checkpoint，以及对应的 tokenizer、归一化统计和模型配置。官方 README 给出的 LIBERO 流程要求额外准备 `norm_stats.json`，并使用 `apxinf-robo eval-libero` 运行 10 个任务的评测。建议先跑单任务单回合 smoke test，再进行完整评测：

```bash
apxinf-robo eval-libero \
  --backend in-process \
  --model-dir /path/to/pi05_libero_base \
  --norm-stats /path/to/pi05_libero_base/norm_stats.json \
  --precision bf16 \
  --action-horizon 10 \
  --suite libero_10 \
  --tasks 0 \
  --trials-per-task 1 \
  --results-jsonl /path/to/results/smoke.jsonl \
  --summary-json /path/to/results/smoke.json
```

如果 smoke test 通过，再把 `--tasks 0 --trials-per-task 1` 改为完整任务和回合数。完整成功率需要和相同 checkpoint、相同 normalization、相同 seed 的参考结果比较，不能只比较一条演示视频。

## 5. 量化和部署时的注意事项

- BF16 是最容易复现的基线，不需要校准文件。
- FP8 需要代表性观测数据生成 activation scale；校准数据应尽量覆盖实际相机、状态和指令分布。
- INT8 不是“打开开关就一定更快”。需要同时检查模型精度、输入形状、GPU 架构和实际控制成功率。
- CUDA Graph 通常要求输入形状和执行路径稳定。动态视角数、动态 action horizon 或临时改变 flow steps，可能导致重新捕获或失去优化收益。
- 端到端 latency 应拆成模型计算、预处理、后处理、传输和机器人执行几个部分。只报告 L1 latency 容易高估真实闭环频率。

## 6. 怎样理解它的创新性

APXInf 的主要贡献是系统工程，不是新的 VLA 训练算法。它没有改变 π0.5 的任务定义，也没有提出新的视觉编码器或动作学习目标；它做的是把已经训练好的策略压到机器人端，并在固定形状、小 batch 和实时闭环条件下把整个执行路径做得更紧凑。

因此可以这样评价：

- 算法创新：有限；
- Runtime 和 Kernel 工程价值：较高；
- 对 NVIDIA 端侧具身部署的实用价值：较高；
- 通用性：仍受模型和硬件支持范围限制；
- 当前成熟度：早期版本，需自行验证版本锁定、长时间运行和真实机器人安全性。

## 7. 和相关工作的组合方式

实际系统不一定要在这些项目中“二选一”：

```text
RLinf：训练 / rollout / 评测
        ↓
APXInf：模型端侧推理
        ↓
EVA-Client 或自研控制客户端：异步动作执行、IK、记录和真机评测
        ↓
机器人控制器
```

如果单次推理已经很快但机器人仍然有停顿，应优先检查异步调度、action chunk、控制周期和通信延迟；这时 Jetson-PI 或 EVA-Client 的策略可能比继续改 Kernel 更有效。如果目标是跨模型、跨硬件的统一 Runtime，可以继续跟踪 [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp)，但应把它与 APXInf 的公开 benchmark 分开比较。

## 参考资料

1. [APXinf-robo 官方仓库](https://github.com/RLinf/APXinf-robo)
2. [RLinf 官方仓库](https://github.com/RLinf/RLinf)
3. [RLinf ApxInf Rollout Backend 文档](https://rlinf.readthedocs.io/en/latest/rst_source/guides/apxinf.html)
4. [无问芯穹 APXInf 项目介绍](https://www.infinigence-ai.com/news-updates/233.html)
5. [EVA-Client 官方仓库](https://github.com/Noietch/EVA-CLIENT)
6. [Jetson-PI 论文](https://arxiv.org/abs/2607.12659)
7. [Embodied.cpp 项目与论文](https://arxiv.org/abs/2607.02501)

