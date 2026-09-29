# Booster K1 网页 MuJoCo：本机部署与 AMP 训练体验

> 更新日期：2026-09-29

本教程先体验上游在线 demo，再把上游静态网页和随仓库提供的模型资产部署到本机 Windows 浏览器；另说明获授权后如何发布到自己的 GitHub Pages。网页推理不需要安装 Python 项目的 MuJoCo 环境或训练模型；本机静态托管需要 Python 运行 HTTP 服务，重新训练则需要 Linux + NVIDIA GPU。

## 1. 网页 demo 的运行方式

上游 `booster_mjlab` 项目提供 Booster K1 的 MuJoCo 模型、速度控制与动作跟踪任务，以及基于动作先验（Adversarial Motion Priors，AMP）的训练流程。它的网页是静态站点：浏览器从网站目录加载页面、JavaScript 和 `static/demo/` 中的策略与模型，再从 jsDelivr 加载 MuJoCo WebAssembly、Three.js 和 Draco 解码器。仿真和策略推理在浏览器中运行，官方页面标注策略频率为 50 Hz；这不是远程桌面或视频流。

因此要区分三件事：

| 内容 | 运行位置 | 需要什么 |
| :--- | :--- | :--- |
| 官方在线 demo | 上游托管网页；策略在本机浏览器计算 | 浏览器、网络和 WebGL |
| 本机静态托管 | 本机 HTTP 服务提供上游 `website/` 文件 | 上游仓库副本、Python 3、浏览器和网络 |
| 可选训练 | Linux 上游项目 | Linux、NVIDIA GPU、W&B；最低显存未公布 |

本机部署这一步使用上游网页源码和预导出的策略资产，不是本教程重新实现了 MuJoCo 前端，也不是本机训练出的模型。官方项目页展示了真机运行策略的结果，但上游仓库目前说明真机部署代码尚未提供；本教程不声称复现真机实验。

## 2. 先体验上游在线 demo

### 2.1 打开交互页面

使用较新的 Microsoft Edge 或 Google Chrome 打开 [Booster K1 官方项目页](https://intelligentroboticslab.github.io/booster_mjlab/)，滚动到 **In your browser**，点击 **Load interactive demo ≈14 MB**。这是上游托管的网页；首次加载需要联网下载 WebAssembly、渲染库和策略等资源。

### 2.2 控制机器人

1. 等待仿真画布和左上角的速度读数出现。
2. 点击仿真场景让它获得键盘焦点。
3. 按 `W` 前进、`S` 后退、`A` / `D` 横向移动，按 `Q` / `E` 转向。也可以拖动左下角摇杆，或使用游戏手柄。
4. 点击画面中的 **Reset** 按钮，恢复初始状态。鼠标拖动场景可旋转观察视角。

### 2.3 检查是否运行成功

- 页面能显示 K1 仿真模型、地面和控制摇杆。
- 按下方向键后，速度读数会变化，机器人随之运动；松开按键后，对应速度命令回到零。
- 点击 **Reset** 后，仿真回到初始状态。

本教程编写时在 Windows 的 Microsoft Edge 中实际打开上游在线页面：画布正常渲染，按 `W` 后 `vx` 从 `0.00` 变为非零，松开后回到 `0.00 m/s`。这只验证上游在线 demo 在该浏览器可交互，不是本机托管验证。

### 2.4 常见问题

- **按钮点击后没有画面**：确认网络可访问官方页面，等待首次资源加载完成；刷新页面后再试。必要时更新浏览器和显卡驱动，并在浏览器设置中启用图形加速。
- **按键没有反应**：先点击仿真画布，再按方向键；确认焦点没有落在地址栏或页面其他控件上。
- **画面卡顿或加载时间较长**：首次需要下载约 14 MB 资源；关闭占用较高的程序后重新加载，并确认浏览器没有禁用 WebAssembly / WebGL。
- **想在自己的电脑运行网页文件**：继续看下一节。本机静态托管网页与安装上游 Python 仿真 / 训练环境是两件事。
- **想在 Windows 运行 Python 版训练**：上游当前要求 NVIDIA GPU，但没有提供 Windows 原生训练步骤；训练请使用 Linux + NVIDIA 环境。

## 3. 在 Windows 本机托管网页

网页本身是静态文件，不需要在 Windows 上安装 MuJoCo Python 包，也不需要 GPU 训练。浏览器仍会从 jsDelivr 下载 MuJoCo WebAssembly、Three.js 和 Draco，因此首次加载需要联网。不要双击 `index.html` 以 `file://` 打开：浏览器模块和 `fetch()` 需要 HTTP 来源。

### 3.1 获取上游网页与模型文件

在教程仓库根目录打开 PowerShell，克隆上游仓库：

```powershell
$BoosterRoot = Join-Path $env:USERPROFILE 'src\booster_mjlab'
git clone --depth 1 https://github.com/IntelligentRoboticsLab/booster_mjlab.git $BoosterRoot
```

如果 GitHub 克隆连接失败，可从 [上游仓库](https://github.com/IntelligentRoboticsLab/booster_mjlab)选择 **Code → Download ZIP**，解压后把 `$BoosterRoot` 改为解压目录。确认网页和策略文件已在磁盘：

```powershell
Test-Path "$BoosterRoot\website\index.html"
Test-Path "$BoosterRoot\website\static\demo\scene.json"
Test-Path "$BoosterRoot\website\static\demo\policy.bin"
```

三个结果都应为 `True`。演示依赖一组相互引用的文件，下载时保留整个 `website/` 目录结构。

### 3.2 启动本机 HTTP 服务

本教程附带的 `serve_demo.py` 是一个标准库静态服务器。Windows Python 3.11 的默认 MIME 表会把 `.js` 当成 `text/plain`，浏览器会拒绝执行网页的 ES module；这个脚本显式设置 `.js`、`.mjs` 和 `.wasm` 的 MIME 类型。

仍在教程仓库根目录执行：

```powershell
python ".\07-机器人操作、运动控制\Locomotion\02-BoosterK1-mjlab-AMP\serve_demo.py" `
  --directory "$BoosterRoot\website" `
  --host 127.0.0.1 `
  --port 8765
```

终端显示 `Serving ... at http://127.0.0.1:8765` 后，保持该窗口运行，在 Edge 或 Chrome 打开 `http://127.0.0.1:8765/`。点击 **Load interactive demo**，等待模型出现，再按 `W`；速度读数应变为非零，松开后会平滑回落到零。按 `Ctrl+C` 关闭服务。

`127.0.0.1` 仅本机可访问。该 Python 服务适合本机检查，不是生产公网服务器；对局域网临时开放可将 `--host` 改为 `0.0.0.0`，但需要自行处理防火墙和访问范围。

本教程在 Windows 11、Python 3.11 和 Microsoft Edge 上实测了这条本机托管路径：网页模块以 `text/javascript` 返回，交互画布和策略均成功加载；按住 `W` 时速度读数升至非零，松开后回落至 `0.00 m/s`。页面中的上游介绍视频请求在自动化检查中被浏览器取消，不影响仿真 demo。

## 4. 网页模型和策略文件从哪里来

上游仓库已提交当前 demo 使用的预导出资产，不需要再从 Hugging Face 下载网页推理权重。它们位于 [`website/static/demo/`](https://github.com/IntelligentRoboticsLab/booster_mjlab/tree/main/website/static/demo/)：

| 文件 | 用途 |
| :--- | :--- |
| `policy.bin` | 浏览器执行的 AMP 速度策略权重；不是 PyTorch checkpoint |
| `k1_web.xml` | 浏览器仿真使用的 MuJoCo XML 模型 |
| `meshes/*.stl` | XML 引用的碰撞网格，需保留相对路径 |
| `robot.glb` | Three.js 渲染用的机器人外观网格 |
| `scene.json` | 策略、关节映射、控制参数和上述文件的索引 |

要复用官方 demo，最简单可靠的方式是下载 / 克隆上游仓库，并完整保留 `website/static/demo/`。`lafan_locomotion_k1` [Hugging Face 数据集](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)是 AMP 训练使用的动作参考数据，不是网页 demo 的 `policy.bin`。

上游导出脚本 `export-web-demo` 接受带 mjlab 元数据的 ONNX 速度策略，并生成 `policy.bin`、XML、网格和场景元数据；其使用示例为：

```bash
uv run --with fast-simplification export-web-demo path/to/policy.onnx
```

默认输出目录是仓库的 `website/static/demo/`。上游目前没有说明如何把训练生成的 `.pt` checkpoint 转成这个 ONNX 输入，因此不要把 `.pt` 文件直接改名成 `policy.bin`；要替换为自己训练的策略，需要先按上游提供的 ONNX 导出流程准备模型。

## 5. 获授权后发布到自己的 GitHub Pages

截至本教程更新日期，上游仓库没有项目级 `LICENSE` 文件；仓库中的 `BULMA-LICENSE.txt` 只说明 Bulma 样式库的许可，不能代替网页、机器人模型和策略资产的许可。**在公开再托管或分发网页代码、模型、策略前，先向上游维护者确认授权或等待仓库补充许可证。**以下步骤仅适用于已确认有权公开托管这些文件的情况；GitHub Pages 上的 `policy.bin`、模型和网页文件都可被访客访问。

上游自带的 [`.github/workflows/docs.yml`](https://github.com/IntelligentRoboticsLab/booster_mjlab/blob/main/.github/workflows/docs.yml) 已展示发布方式：网站是静态文件，无构建步骤，GitHub Actions 直接把 `website/` 目录上传到 Pages。最省事的做法是在授权明确后 fork 上游仓库；若使用自己的仓库，则需按许可复制网页文件和该工作流，并保留原有目录结构。

1. Fork [`IntelligentRoboticsLab/booster_mjlab`](https://github.com/IntelligentRoboticsLab/booster_mjlab)，或在获得授权后把上游 `website/` 和 `.github/workflows/docs.yml` 放入自己的仓库。工作流默认监听 `main` 分支；若你的默认分支名称不同，先相应修改工作流中的分支名。
2. 打开仓库 **Settings → Pages → Build and deployment**，将 **Source** 设为 **GitHub Actions**。确认 **Actions** 已启用；若工作流页面提示需要启用 `Docs`，先点击启用。
3. 打开 **Actions → Docs → Run workflow** 手动发布；以后推送对 `main` 分支 `website/` 或该工作流文件的修改时也会自动发布。工作流成功后，在 **Actions** 的部署详情或 **Settings → Pages** 查看站点地址。项目仓库通常使用 `https://<用户名>.github.io/booster_mjlab/` 这样的路径。
4. 从公网地址打开页面并点击 **Load interactive demo**。确认机器人画布能加载、键盘控制有响应，并检查 `scene.json`、`policy.bin`、XML、网格和 GLB 等资源均可访问。

网页的 MuJoCo 与渲染依赖仍从 jsDelivr 加载，所以访问者需要能连接 CDN。若部署环境不能访问 CDN，还需要在许可允许的前提下自行托管这些依赖，并修改网页中的 import map 和 WASM 路径；当前教程没有验证离线打包方案。GitHub Pages 支持通过 GitHub Actions 发布静态站点，详见 [GitHub Pages 自定义工作流文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)和[配置发布源说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。

## 6. 可选：训练 AMP 速度策略

本节面向希望体验强化学习训练的读者。**不需要为了体验浏览器 demo 执行本节。**命令基于上游仓库 README；上游项目更新后，任务名和参数可能变化，运行前请以其最新说明为准。

### 6.1 环境要求

- Linux 环境和可正常工作的 NVIDIA GPU。上游说明训练需要 NVIDIA GPU，但没有给出最低显存或推荐显卡型号；不要据此推断任意笔记本 GPU 都能承担默认配置。
- Git、可联网环境，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器。项目会通过 `uv` 同步 Python 及项目依赖。
- Weights & Biases（W&B）账号，用于记录训练运行并按 run 路径加载 checkpoint。首次使用时按 [W&B 官方说明](https://docs.wandb.ai/models/quickstart)完成登录；API key 不要写进代码或提交到仓库。
- 稳定的网络和数 GB 可用磁盘空间。首次 `uv run` 会同步 PyTorch / MuJoCo 等依赖；实际下载量因平台和依赖解析结果而异，上游没有公布精确的磁盘最低要求。

上游没有给出 Windows 原生训练支持说明。Windows 用户如需训练，请使用有 NVIDIA GPU 的 Linux 工作站或云端 Linux 环境；不要把浏览器推理验证理解成 Windows Python 训练已验证。默认的 4096 个并行环境可能需要较多显存；若显存不足，可先将并行环境数调小。具体速度和显存占用取决于 GPU、驱动和依赖版本。

### 6.2 安装并检查任务

在 Linux 终端执行：

```bash
git clone https://github.com/IntelligentRoboticsLab/booster-mjlab.git
cd booster-mjlab
uv run list_envs
```

`uv run list_envs` 会按项目配置准备运行环境，并列出已注册任务。确认命令能结束且列表中包含 `Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1` 后，再开始训练。若尚未安装 `uv`，请先按其官方安装文档操作，重新打开终端并确认 `uv --version` 可用。

### 6.3 启动速度控制训练

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

### 6.4 在仿真中评估 checkpoint

从 W&B run 页面取得训练运行路径，格式为 `组织或用户名/mjlab/run-id`，然后执行：

```bash
uv run play Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \
  --wandb-run-path 组织或用户名/mjlab/run-id
```

将命令中的示例路径替换为自己的 W&B run 路径。上游说明该命令会从 W&B 获取最新 checkpoint，并启动策略仿真评估。检查策略是否能响应速度命令、机器人能否在仿真中持续行走；这仍是仿真评估，不是向真机下发控制命令。

## 7. 结果解读与注意事项

- 官方项目页描述了真机 Booster K1 在训练 30k steps 后运行 AMP 速度策略的演示。这是上游展示的结果，不是本教程在读者硬件上的复现保证；训练步数也不能单独作为相同效果的保证。
- 上游 README 当前说明真机策略部署代码“即将提供”。本教程只覆盖浏览器端推理，以及上游训练命令和仿真评估入口，不提供真机部署或安全操作步骤。
- 浏览器版使用网页公开的导出策略；训练版则是在本机 / Linux GPU 环境中重新训练。两者依赖、运行位置和产物不同。
- 本教程只链接上游仓库、网页和数据集，没有复制上游代码、模型权重或网页资源。截至更新日期，上游仓库页面未见 LICENSE 文件；如需二次分发其代码、权重或数据，请先向上游维护者确认授权与数据使用条件。

## 8. 参考资料

- [Booster K1 官方项目页与浏览器 demo](https://intelligentroboticslab.github.io/booster_mjlab/)
- [booster_mjlab 上游仓库与训练说明](https://github.com/IntelligentRoboticsLab/booster_mjlab)
- [上游 LAFAN1 K1 动作数据集](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
- [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)
- [Weights & Biases 快速开始](https://docs.wandb.ai/models/quickstart)
