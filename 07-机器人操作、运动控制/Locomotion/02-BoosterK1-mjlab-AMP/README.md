# Booster K1：浏览器推理与 AMP 训练体验

> 更新日期：2026-09-28

本教程带你在 Windows 浏览器中体验 Booster K1 的 AMP 速度控制策略，并提供 Linux + NVIDIA GPU 上的可选训练流程。浏览器体验不需要安装 Python、MuJoCo 或模型权重；训练则需要配置独立的 GPU 环境。

## 1. 体验内容与复现边界

上游 `booster_mjlab` 项目为 Booster K1 提供 MuJoCo `mjlab` 模型、速度控制与动作跟踪任务，以及基于动作先验（Adversarial Motion Priors，AMP）的训练流程。官方网页把导出的速度策略和 MuJoCo WebAssembly 放进浏览器，策略以 50 Hz 运行，可用键盘、摇杆或手柄控制。

这里有两条不同的体验路径：

| 路径 | 运行位置 | 是否训练 | 硬件要求 |
| :--- | :--- | :--- | :--- |
| 浏览器推理 | Windows 上的 Edge / Chrome | 否，使用网页提供的导出策略 | 支持 WebAssembly 与 WebGL 的现代浏览器 |
| 可选训练与评估 | Linux 环境中的上游项目 | 是，训练自己的策略并在仿真中评估 | 上游要求 NVIDIA GPU；未公布最低显存 |

浏览器 demo 复现的是“加载现成策略并在本地浏览器仿真中推理”，不是在这台 Windows 笔记本上训练模型，也不等于 Booster K1 真机部署。上游项目页展示了真机运行策略的结果，但上游仓库目前说明真机部署代码尚未提供；本教程不声称复现了真机实验或其训练过程。

## 2. Windows 浏览器推理

### 2.1 打开交互页面

使用较新的 Microsoft Edge 或 Google Chrome 打开 [Booster K1 官方项目页](https://intelligentroboticslab.github.io/booster_mjlab/)，滚动到 **In your browser**，点击 **Load interactive demo ≈14 MB**。首次加载需要联网下载网页运行所需的 WebAssembly 与策略文件；页面提示推理在浏览器中运行，之后可直接控制仿真中的机器人。

### 2.2 控制机器人

1. 等待仿真画布和左上角的速度读数出现。
2. 点击仿真场景让它获得键盘焦点。
3. 按 `W` 前进、`S` 后退、`A` / `D` 横向移动，按 `Q` / `E` 转向。也可以拖动左下角摇杆，或使用游戏手柄。
4. 点击画面中的 **Reset** 按钮，恢复初始状态。鼠标拖动场景可旋转观察视角。

### 2.3 检查是否运行成功

- 页面能显示 K1 仿真模型、地面和控制摇杆。
- 按下方向键后，速度读数会变化，机器人随之运动；松开按键后，对应速度命令回到零。
- 点击 **Reset** 后，仿真回到初始状态。

本教程编写时在 Windows 的 Microsoft Edge 中实际打开了该页面并加载交互 demo：画布正常渲染，按 `W` 后 `vx` 从 `0.00` 变为非零，松开后回到 `0.00 m/s`，浏览器未报告页面脚本异常。具体读数会随按键时长和页面运行状态变化。这验证了该 Windows 浏览器上的网页推理链路，不代表所有显卡驱动、浏览器版本和设备都已覆盖测试。

### 2.4 常见问题

- **按钮点击后没有画面**：确认网络可访问官方页面，等待首次资源加载完成；刷新页面后再试。必要时更新浏览器和显卡驱动，并在浏览器设置中启用图形加速。
- **按键没有反应**：先点击仿真画布，再按方向键；确认焦点没有落在地址栏或页面其他控件上。
- **画面卡顿或加载时间较长**：首次需要下载约 14 MB 资源；关闭占用较高的程序后重新加载，并确认浏览器没有禁用 WebAssembly / WebGL。
- **想在 Windows 本地运行 Python 版仿真**：本教程验证的是官方网页 demo，不是上游 Python 项目的 Windows 安装。上游当前给出的训练环境要求为 NVIDIA GPU，也没有提供 Windows 原生训练步骤；请参考下一节，在受支持的 Linux + NVIDIA 环境中训练。

## 3. 可选：训练 AMP 速度策略

本节面向希望体验强化学习训练的读者。**不需要为了体验浏览器 demo 执行本节。**命令基于上游仓库 README；上游项目更新后，任务名和参数可能变化，运行前请以其最新说明为准。

### 3.1 环境要求

- Linux 环境和可正常工作的 NVIDIA GPU。上游说明训练需要 NVIDIA GPU，但没有给出最低显存或推荐显卡型号；不要据此推断任意笔记本 GPU 都能承担默认配置。
- Git、可联网环境，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器。项目会通过 `uv` 同步 Python 及项目依赖。
- Weights & Biases（W&B）账号，用于记录训练运行并按 run 路径加载 checkpoint。首次使用时按 [W&B 官方说明](https://docs.wandb.ai/models/quickstart)完成登录；API key 不要写进代码或提交到仓库。
- 稳定的网络和数 GB 可用磁盘空间。首次 `uv run` 会同步 PyTorch / MuJoCo 等依赖；实际下载量因平台和依赖解析结果而异，上游没有公布精确的磁盘最低要求。

上游没有给出 Windows 原生训练支持说明。Windows 用户如需训练，请使用有 NVIDIA GPU 的 Linux 工作站或云端 Linux 环境；不要把浏览器推理验证理解成 Windows Python 训练已验证。默认的 4096 个并行环境可能需要较多显存；若显存不足，可先将并行环境数调小。具体速度和显存占用取决于 GPU、驱动和依赖版本。

### 3.2 安装并检查任务

在 Linux 终端执行：

```bash
git clone https://github.com/IntelligentRoboticsLab/booster-mjlab.git
cd booster-mjlab
uv run list_envs
```

`uv run list_envs` 会按项目配置准备运行环境，并列出已注册任务。确认命令能结束且列表中包含 `Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1` 后，再开始训练。若尚未安装 `uv`，请先按其官方安装文档操作，重新打开终端并确认 `uv --version` 可用。

### 3.3 启动速度控制训练

上游 README 给出的默认命令是 4096 个并行环境：

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \
  --env.scene.num-envs 4096
```

如果显存不足，可先尝试较小的并行数，例如：

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \
  --env.scene.num-envs 512
```

训练期间观察终端和 W&B run 页面，确认仿真步数及训练指标持续更新，并记录此次 run 的路径。`512` 只是降低并行规模的起始尝试，不是上游保证的最低配置或性能承诺；如果初始化时显存不足，继续降低环境数或换用更大显存的 GPU。

默认速度任务使用上游提供的少量 LAFAN1 行走动作片段作为 AMP 动作先验。若要替换动作数据，上游支持 Hugging Face 数据集 ID 或本地文件 / 目录，可通过 `--agent.dataset-root` 传入；请先按上游数据格式准备并验证数据。

### 3.4 在仿真中评估 checkpoint

从 W&B run 页面取得训练运行路径，格式为 `组织或用户名/mjlab/run-id`，然后执行：

```bash
uv run play Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \
  --wandb-run-path 组织或用户名/mjlab/run-id
```

将命令中的示例路径替换为自己的 W&B run 路径。上游说明该命令会从 W&B 获取最新 checkpoint，并启动策略仿真评估。检查策略是否能响应速度命令、机器人能否在仿真中持续行走；这仍是仿真评估，不是向真机下发控制命令。

## 4. 结果解读与注意事项

- 官方项目页描述了真机 Booster K1 在训练 30k steps 后运行 AMP 速度策略的演示。这是上游展示的结果，不是本教程在读者硬件上的复现保证；训练步数也不能单独作为相同效果的保证。
- 上游 README 当前说明真机策略部署代码“即将提供”。本教程只覆盖浏览器端推理，以及上游训练命令和仿真评估入口，不提供真机部署或安全操作步骤。
- 浏览器版使用网页公开的导出策略；训练版则是在本机 / Linux GPU 环境中重新训练。两者依赖、运行位置和产物不同。
- 本教程只链接上游仓库、网页和数据集，没有复制上游代码、模型权重或网页资源。截至更新日期，上游仓库页面未见 LICENSE 文件；如需二次分发其代码、权重或数据，请先向上游维护者确认授权与数据使用条件。

## 5. 参考资料

- [Booster K1 官方项目页与浏览器 demo](https://intelligentroboticslab.github.io/booster_mjlab/)
- [booster_mjlab 上游仓库与训练说明](https://github.com/IntelligentRoboticsLab/booster_mjlab)
- [上游 LAFAN1 K1 动作数据集](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
- [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)
- [Weights & Biases 快速开始](https://docs.wandb.ai/models/quickstart)
