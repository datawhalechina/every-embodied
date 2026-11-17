# 代码目录与复现入口

本目录包括已用于项目的 Aura 客户端、视觉/交互模块、参数化数字孪生和同场景 Agent 适配代码。公开副本把实验室绝对路径改成读者配置，不携带设备口令、模型权重、授权场景和登录会话。

## 1. 先运行零依赖离线检查

Python 3.11+，在此目录执行：

```bash
python tools/course_lab.py audit --root ..
python -m unittest discover -s tests -v
python tools/course_lab.py twin --root .
```

`audit` 核对真实保存的原生证明和媒体元数据；`twin` 检查 MJCF 资源完整。它们不会启动模型、仿真或真机。Notebook 的教学状态机和数值示例与实际执行记录分开。

## 2. Aura 客户端

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m aura_robot.debug --host YOUR_ROBOT_IP probe --output artifacts/probe.json
python -m aura_robot.debug --host YOUR_ROBOT_IP --timeout 30 camera --output artifacts/camera.jpg
python -m aura_robot.cli demo
```

最后一条是明确的 Mock 示范，不访问真机。`src/aura_robot` 中的 `debug` 才走实际 daemon 接口；`vision`、`wave_follow`、`person_follow`、`lightnav` 分别提供算法与适配命令。电池端点、自建相机桥取决于设备安装情况，正文说明了这些条件。

## 3. 同场景原生环境

沿用已验证的隔离环境和有授权的数据。历史组合为 OmniGibson 3.9.1 / Isaac Sim 5.1.0、R1 Pro、`house_double_floor_lower`、`picking_up_trash` instance 301。它加入了人物与 Aura，属于集成演示。

上游 RoboHarness 固定版本：`d58e2defb25b9167a1ef14ab778995bf9e9de42f`。准备源码时放在 `external/RoboHarness`，按上游安装文档准备仿真、数据与工具 workspace；外部资源下载使用自己环境适用的下载入口。

```bash
git clone https://github.com/BinceQu/RoboHarness.git external/RoboHarness
git -C external/RoboHarness checkout d58e2defb25b9167a1ef14ab778995bf9e9de42f
cp config/runtime.example.json config/runtime.json
```

填写 `config/runtime.json` 的 workspace、仿真解释器、数据与上游目录，再复制/编辑 `integrations/aura_household_world/world.json`。该文件是**位置配置示例**，不是最终三罐录像所用的已验收位置快照；根据已加载房间设置人/Aura 的客厅位置与 R1 厨房起点，执行 clearance 和相机可见性检查。

CPU USD 作者工具需要 `ISAACSIM_ROOT`、`AURA_HOME_USD` 和 `ISAAC_PEOPLE_DIR` 指向已有安装/资产；视频导出 `AURA_VIDEO_FONT` 指向已授权的手写字体；`CODEX_EXEC` 可指定 Codex 可执行程序。仅对当前项目进程设置这些变量，不改共享环境。

准备入口只写请求和配方，不启动原生场景：

```bash
python -m integrations.aura_household_world.prepare --output artifacts/new-preparation --text '帮我把垃圾收拾一下'
python -m integrations.aura_household_world.build_cpu --config integrations/aura_household_world/world.json --output artifacts/new-usd
```

完成上游任务环境的原生启动、独立工作目录绑定与 `qualification.json` 后，先阅读 `run_live --help` 和宿主协议，再接入控制器。资格文件与 `.session.json` 由自己运行的宿主生成，不能下载别人的会话来代替。

```bash
python -m integrations.aura_household_exec.run_live \
  --workspace /path/to/your/bound-workspace \
  --evidence /path/to/your/new-agent-evidence \
  --qualification /path/to/your/qualification.json \
  --codex /path/to/your/codex \
  --context-mode compact --goal-macros --canonical-placement \
  --durable-checkpoints --archive-public-context \
  --completion-scope three_cans --native-completion /path/to/your/three-can-proof.json \
  --task '先根据当前相机收集三个饮料罐，逐个投入原垃圾桶；按原生状态验收。'
```

上面任务是课程简化示例；实际返回阶段完整原提示保存在第九章。主控制器使用当时已验证的 Sol / xhigh CLI 通道。若修改模型，记录模型与 CLI 版本并重新核验 schema；运行器不因模型文字 `done` 自动跳过原生验收。

现有工作站的设备资源、仿真数据和已绑定 workspace 不随课程打包。代码结构、准备命令与 CPU 测试可在开源副本运行；完整原生重新执行需先完成自己的资产和宿主配置。

## 4. 更多执行层回归

`integrations/aura_household_exec/test_*.py` 保留宿主、恢复和工具合同的 CPU 测试。一部分使用 Pillow 和 Linux 的 `fcntl`，建议在已有工作站 Python 环境运行：

```bash
python -m unittest discover -s integrations/aura_household_exec -t . -p 'test_*.py' -v
```

`integrations/aura_household_world/test_world.py` 包含原实验准备材料的环境级测试，需外部数据/profile 准备好后运行，不属于零依赖验收。

## 5. 查看完整视频与同步公开过程

在本专题目录运行静态服务，然后打开浏览器显示的 `demo/index.html`：

```bash
cd ..
python -m http.server 8080 --bind 127.0.0.1
```

在本机打开 `http://127.0.0.1:8080/demo/index.html`。这是读者自己启动的服务，不依赖作者电脑；视频地址是公开 GitHub Release。给另一台电脑访问时使用本机局域网地址，并按需要调整绑定与防火墙，仅共享课程静态目录。

完整 MP4 共 11 路、约 170 MB，不自动下载；可点击正文或 Release 逐个观看。动图均为节选，完整片保留恢复提示和音轨。
