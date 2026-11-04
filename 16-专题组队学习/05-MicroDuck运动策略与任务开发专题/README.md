# MicroDuck 运动策略与任务开发专题

本专题集中整理 MicroDuck / OpenDuck Mini V2 相关的强化学习、运动控制、物理仿真、任务编排和端侧部署材料。目录按照“先跑通基线，再理解代码，最后做任务开发”的顺序组织，适合跟着学习逐步补充实验记录。

第一次运行，先打开 [MuJoCo 策略控制 HelloWorld](./03-Notebook/00_MuJoCo策略控制_HelloWorld.ipynb)。它自动下载行走模型，在同一个场景中用 Python 执行前进、转向和停止，直接展示“实时观测 → ONNX 推理 → 关节目标 → MuJoCo 物理更新”。[最小环境安装步骤](./02-可运行代码/hello-world/README.md)只需要 CPU，不要求显卡或开发板。

按 [8 天学习计划](./学习计划.md)每天学习一本 Notebook，查看[任务演示总览](./01-任务资料/精选视频/README.md)，或阅读[领学员招募文案](./招募文案.md)。

另有 [21 天月度组队学习招募文案](./月度组队学习招募文案.md)，每个 Task 安排 2–3 天。两套方案共用教程和 Notebook，原有 8 天短期方案保留。

## 学习路线

1. 先看 [篮球平衡](./01-任务资料/01-MicroDuck篮球平衡强化学习/README.md)，理解速度命令、61 维观测、14 维动作、PPO 和接触奖励。
2. 再看 [浏览器物理扰动](./01-任务资料/02-mjswan-MicroDuck浏览器物理扰动/README.md)，理解 MuJoCo 状态、扰动、录制和可视化接口。
3. 进入 [高跷行走](./01-任务资料/03-MicroDuck高跷行走强化学习复现/README.md)，学习执行器参数、形态课程、动作平滑和 sim-to-real 约束。
4. 学习 [摆动旋转](./01-任务资料/04-MicroDuck摆动旋转强化学习复现/README.md) 和 [球平衡与 FastSAC](./01-任务资料/05-MotrixLab-MicroDuck球平衡与FastSAC/README.md)，对比不同任务和算法接口。
5. 最后进入 [梯面攀爬](./01-任务资料/06-MicroDuck梯面攀爬强化学习导读/README.md)，先跑通攀爬、登桌与起身，再学习双策略切换、横档接触和课程训练。
6. 需要运行或修改代码时，进入 [可运行代码包](./02-可运行代码/microduck-playground-stilts/README.md)。
7. 需要查看 RDK X5、网页演示和多策略调度时，进入 [端侧与网页部署](./01-任务资料/07-RDK端侧与网页部署/README.md)。
8. 需要按直播顺序运行检查时，打开 [Notebook 学习入口](./03-Notebook/README.md)。
9. 第 8 天让小鸭子理解语言指令并在不同房间中导航，运行 [LightNav-0 多场景导航](./03-Notebook/08_LightNav0_视觉语言导航_GPU_多场景.ipynb)。使用 GPU［图形处理器］接入已有模型，不训练导航模型、不连接开发板。
10. 拓展体验[跨缝跳跃](./01-任务资料/08-MicroDuck跨缝跳跃/README.md)：高平台起跳、低平台落地，比较原始模型、10 轮 PPO 续训与 ONNX。第 9 本 Notebook 不改动原有学习日程。
11. 了解 [AI 智能体辅助 MicroDuck 训练](./01-任务资料/09-AI智能体辅助MicroDuck训练/README.md)，区分任务规划、开源行走训练与视觉越障适配。

## 目录结构

```text
05-MicroDuck运动策略与任务开发专题/
├── README.md
├── 学习计划.md
├── 招募文案.md
├── 月度组队学习招募文案.md
├── 01-任务资料/
│   ├── 01-MicroDuck篮球平衡强化学习/
│   ├── 02-mjswan-MicroDuck浏览器物理扰动/
│   ├── 03-MicroDuck高跷行走强化学习复现/
│   ├── 04-MicroDuck摆动旋转强化学习复现/
│   ├── 05-MotrixLab-MicroDuck球平衡与FastSAC/
│   ├── 06-MicroDuck梯面攀爬强化学习导读/
│   ├── 07-RDK端侧与网页部署/
│   ├── 08-MicroDuck跨缝跳跃/
│   ├── 09-AI智能体辅助MicroDuck训练/
│   └── 精选视频/
├── 02-可运行代码/
│   ├── microduck-playground-stilts/
│   │   ├── src/          # 机器人、执行器、任务和奖励
│   │   ├── scripts/      # 训练、评测、导出和推理脚本
│   │   ├── tests/        # 配置和物理不变量测试
│   │   ├── hardware/     # 高跷参数化生成器
│   │   └── docs/         # 训练计划和策略说明
│   ├── hello-world/     # 自动下载模型，Python 速度命令与 MuJoCo 闭环
│   ├── lightnav-learning/ # 导航服务、场景构建与闭环视频
│   └── long-jump-learning/ # 跨缝跳跃、训练恢复与视频
└── 03-Notebook/          # HelloWorld、8 本主线教程和 1 本跳跃拓展
```

## Notebook 学习线

下载仓库后，先使用 HelloWorld 的最小环境运行 `00_MuJoCo策略控制_HelloWorld.ipynb`，按顺序执行所有单元。运行完成后，修改速度命令，再观察动作、轨迹和新生成的视频。其余任务使用对应的模型、场景和训练环境，准备步骤见各任务教程；代码仓库不包含大型训练权重。

Notebook 推荐运行在 Ubuntu 工作站上，Windows 本机只通过浏览器访问 Jupyter。这样 MuJoCo、训练代码和 Python 依赖都在同一环境中；运动任务可再连接 RDK X5，把开发板作为独立的 BPU 推理端。第 8 本 LightNav-0 导航教程只使用工作站 GPU，不连接开发板。Notebook 内置任务契约、模型检查和视频输出，不需要额外维护一层“Windows Notebook -> Ubuntu -> RDK”的跳转。

ONNX 使用与 RDK X5 SDK 匹配的工具链编译为 HBM，编译在 Ubuntu Docker 中完成。RDK 端运行对应任务 HBM；准备好板端服务和仿真工作站后，可重新生成任务视频。

## 任务与学习内容

| 任务 | 学习内容 | 演示 |
| --- | --- | --- |
| 篮球平衡 | 观测与动作、PPO、循环网络和接触奖励 | 篮球上保持平衡 |
| 浏览器物理扰动 | MuJoCo 状态、外力交互、录制与可视化 | 拖拽与外力交互 |
| 高跷行走 | BAM 执行器、形态课程和动作平滑 | 25 cm / 200 cm 高跷行走 |
| 摆动旋转 | 任务奖励、ONNX 导出和部署接口 | 荡秋千与旋转 |
| 球平衡与 FastSAC | FastSAC 训练、模型导出与评测 | 球面平衡训练 |
| 梯面攀爬 | 攀爬与起身双策略、横档接触和课程训练 | GPU 推理完成爬梯、登桌与站稳 |
| RDK X5 / 网页部署 | HBM 编译、BPU 推理、TCP 和多策略调度 | 板端推理与仿真行走 |
| LightNav-0 视觉语言导航 | 图像与语言、轨迹预测、MPC 和多场景实验 | 住宅、客厅与走廊导航 |
| 跨缝跳跃（拓展） | 起跳与落地奖励、PPO 续训、ONNX 对齐 | 跨过间隙，在低平台落地站稳 |

## 配套代码

`02-可运行代码/microduck-playground-stilts/` 是从本地
`microduck-playground-stilts` 工作目录整理出的可复现轻量快照，包含当前的任务代码和我们增加的梯面、执行器、评测及测试材料。没有复制 `.venv/`、日志、训练缓存、checkpoint 和生成结果；这些文件体积大、绑定本机环境，也不应该直接提交到主资料仓库。

各任务教程分别说明模型来源、训练方法和对应实现；代码包保留上游许可证与本地实验说明。

## 视频素材

每个任务在[演示总览](./01-任务资料/精选视频/README.md)中单独展示，包含 GPU 推理、BPU 推理、高跷尺寸对照、完整爬梯和多任务总览。页面使用压缩 GIF；[本轮完整 MP4 与运行报告](./03-Notebook/recordings/README.md)随资料收录，重新运行 Notebook 会生成新的任务视频和评测报告。

## 后续记录方式

每次实验记录任务名、代码版本、checkpoint、环境数量、训练轮数、随机种子、成功判定、视频路径和评测结果，方便复现与对照。

