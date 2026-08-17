<div align="center">
    <img src="assets/main.png" width="100%" alt="Banner" />


  <pre style="font-family: 'Courier New', monospace; font-size: 16px; color: #000000; margin: 0; padding: 0; line-height: 1.2; transform: skew(-1deg, 0deg); display: block;">
  ███████╗███╗   ███╗██████╗  ██████╗ ██████╗ ██╗███████╗██████╗ 
  ██╔════╝████╗ ████║██╔══██╗██╔═══██╗██╔══██╗██║██╔════╝██╔══██╗
  █████╗  ██╔████╔██║██████╔╝██║   ██║██║  ██║██║█████╗  ██║  ██║
  ██╔══╝  ██║╚██╔╝██║██╔══██╗██║   ██║██║  ██║██║██╔══╝  ██║  ██║
  ███████╗██║ ╚═╝ ██║██████╔╝╚██████╔╝██████╔╝██║███████╗██████╔╝
  ╚══════╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚═════╝ ╚═╝╚══════╝╚═════╝ 
  </pre>

  # Every-Embodied : Zero to Hero in Embodied AI

  <p align="center">
    📌 <a href="https://datawhale-eai.feishu.cn/wiki/QvxQwV2Qyij1NakV9IdcjTl1nJd">组队学习文档 (Team Learning)</a> · ✨ <a href="#learning-map">学习地图 (Learning Map)</a> · 🤖 <a href="#sota">前沿复现 (SOTA)</a> · 📖 <a href="https://datawhalechina.github.io/every-embodied/zh-cn/" target="_blank">在线阅读</a>
  </p>

  <p align="center">
      <a href="https://github.com/datawhalechina/every-embodied/stargazers" target="_blank">
          <img src="https://img.shields.io/github/stars/datawhalechina/every-embodied?color=0052cc&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e" alt="Stars"></a>
      <a href="https://github.com/datawhalechina/every-embodied/network/members" target="_blank">
          <img src="https://img.shields.io/github/forks/datawhalechina/every-embodied?color=0052cc&style=for-the-badge&logo=git-fork&logoColor=white&labelColor=1a1a2e" alt="Forks"></a>
      <a href="LICENSE" target="_blank">
          <img src="https://img.shields.io/badge/License-CC--BY%204.0-4ecdc4?style=for-the-badge&logo=creative-commons&logoColor=white&labelColor=1a1a2e" alt="License"></a>
  </p>
  <p align="center">
    <a href="./README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
    <a href="./README.en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  </p>
  <div align="center">
    <p><strong>Supported By</strong></p>
    <img src="./assets/logo.drobo.svg" height="30" style="margin: 0 10px;"> 
    <img src="./assets/zhiyuan.png" height="40" style="margin: 0 10px;"> 
    <img src="./assets/hugging-face.png" height="40" style="margin: 0 10px;">
    <img src="./assets/shanghaiailab.png" height="45" style="margin: 0 10px;"> 
    <img src="./assets/damo1.png" height="52" style="margin: 0 10px;"> 
    <img src="./assets/damo2.png" height="50" style="margin: 0 10px;"> 
    <img src="./assets/aup-logo.svg" height="34" style="margin: 0 12px;" alt="AUP — AMD University Program">
    <img src="./assets/rocm-logo.png" height="34" style="margin: 0 12px;" alt="ROCm">
  </div>

</div>

  

  

  ## 🚀 快速开始 (Quick Start) ：一分钟体验 Hello Every-Embodied

  想要立刻在本地跑通第一个具身智能仿真 Demo？只需三步：

  ```bash
  # 1. 克隆仓库
  git clone --depth 1 https://github.com/datawhalechina/every-embodied.git
  cd every-embodied
  
  # 2. 创建并激活 Conda 环境
  conda create -n embodied python=3.8
  conda activate embodied
  
  # 3. 安装依赖并运行基础机械臂抓取 Demo
  pip install mujoco
  # 可选：更平滑的 jerk 限制轨迹规划
  pip install ruckig
  python examples/01_hello_every_embodied_mujoco.py
  ```
  详细说明请见：`examples/README.md`



  <br>

<table align="center">
  <tr>
    <td width="33%" valign="top" align="center">
      <img src="assets/quick_start.gif" width="100%">
      <br>
      <strong><a href="./examples/README.md">项目快速入门</a></strong>
      <br>
      <sub>基于mujoco一键了解项目基础</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/zhiyuan.gif" width="100%">
      <br>
      <strong><a href="./10-具身智能其他仿真工具及仿真前沿/08GenieSim3配置.md">Genie Sim的Pi0部署</a></strong>
      <br>
      <sub>基于Pi0和Isaac Sim实现高保真仿真</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/2025-07-02-20-50-54-image.png" width="100%">
      <br>
        <strong><a href="./03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/03RDK-X5连接lerobot机械臂进行遥操作.md">LeRobot 遥操作</a></strong>
      <br>
      <sub>支持地瓜 RDK-X5 连接 SO101 机械臂实操</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top" align="center">
      <img src="assets/2025-06-17-12-11-28-image.png" width="100%">
      <br>
        <strong><a href="./04-具身场景的计算机视觉、3D重建/01-sam和深度估计.md">视觉语义感知</a></strong>
      <br>
      <sub>场景分割与目标检测，让机器人“看懂”环境</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/plane.gif" width="100%">
      <br>
        <strong><a href="./13-其他前沿项目复现/无人机大模型+Groundingdino实践/无人机多模态大模型.md">LLM控制无人机导航VLN</a></strong>
      <br>
      <sub>通过WebUI快速上手无人机大模型VLN导航</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/libero.gif" width="100%">
      <br>
      <strong><a href="./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md">基于SmolVLA微调LIBERO基准</a></strong>
      <br>
      <sub>小型VLA测试机器人终身学习基准</sub>
    </td>
  </tr>
    <tr>
    <td width="33%" valign="top" align="center">
      <img src="assets/chunwan_robot.gif" width="100%">
      <br>
        <strong><a href="./07-机器人操作、运动控制/Locomotion/01春晚舞蹈机器人复刻.md">春晚机器人复刻</a></strong>
      <br>
      <sub>输入豆包生成功夫视频，输出机器人mujoco仿真</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/ETPNav.gif" width="100%">
      <br>
        <strong><a href="./08-具身导航及VLN/03前沿VLN复现/01VLNCE/02ETPNav代码复现.md">ETPNav-VLN导航复现</a></strong>
      <br>
      <sub>连续环境视觉语言导航 (VLN-CE) 领域强力 Baseline</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="assets/lingbot_map_university.gif" width="100%">
      <br>
        <strong><a href="./04-具身场景的计算机视觉、3D重建/03-LingBot-Map视频流式三维重建.md">LingBot-Map 视频建图</a></strong>
      <br>
      <sub>从连续校园视频流式估计轨迹、深度与点云</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top" align="center">
      <img src="./10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/assets/screenshots/teaser_pbr_overview.png" width="100%">
      <br>
        <strong><a href="./10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/README.md">UniLab + MotrixSim 异构 RL</a></strong>
      <br>
      <sub>6GB 显卡跑通 state-based RL，并体验 uv run --no-sync demo teaser 的 PBR 渲染器</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="./10-具身智能其他仿真工具及仿真前沿/09SIM1柔体仿真与数据生成/assets/sim1/sim1-replay-smoke-frame.png" width="100%">
      <br>
        <strong><a href="./10-具身智能其他仿真工具及仿真前沿/09SIM1柔体仿真与数据生成/01SIM1环境配置与运行.md">SIM1 柔体仿真与数据生成</a></strong>
      <br>
      <sub>双臂布料任务的 replay、渲染与扩散轨迹生成流程</sub>
    </td>
    <td width="33%" valign="top" align="center">
      <img src="./21-机械臂和机器人设计/03ForgeCAD视觉逆向工程入门/assets/official_3dprinter_gpt52codex_frame.png" width="100%">
      <br>
        <strong><a href="./21-机械臂和机器人设计/03ForgeCAD视觉逆向工程入门/README.md">ForgeCAD 3D 打印机、键盘与灵巧手</a></strong>
      <br>
      <sub>复刻 public kit 的 3D 打印机、视频键盘和可动灵巧手案例，跑通参数渲染 GIF 与 STEP/STL/3MF 导出</sub>
    </td>
  </tr>



</table>



  <div align="center">
    <h3>⭐ 欢迎点点 Star 共同构建具身智能开源生态 ❤️</h3>
  </div>
  具身智能（Embodied AI）是通往通用人工智能（AGI）的关键钥匙。

  **Every-Embodied 致力于打造一个“基础友好、工程可复现、前沿可拓展”的学习库。**
  我们希望降低门槛：即使你目前只有 Python 基础，也可以从仿真和最小可运行 Demo 出发，逐步完成从“看懂”到“动手复现”的跨越。

  - **入门友好**：优先提供可运行示例与环境脚本，不要求你一开始就掌握复杂机器人学推导
  - **实践导向**：强调“跑起来-看结果-再理解原理”的学习路径，减少抽象概念带来的挫败感
  - **前沿连接**：从 MuJoCo / Isaac Sim 到 VLA / VLN，让学习路径与产业热点保持同步

## 👥 项目受众

这个项目不仅面向机器人专业背景同学，也欢迎更多跨领域学习者加入：

- **Python 初学者 / 转型开发者**：有基础编程能力，希望系统进入 AI+机器人、具身智能方向
- **高校学生 / 研究生**：希望通过课程项目、毕业设计或论文复现快速建立具身智能实践能力
- **算法工程师**：希望把视觉、强化学习、大模型能力迁移到真实物理交互场景
- **硬件与机器人爱好者**：希望理解从硬件控制、仿真验证到 Sim2Real 部署的完整链路
- **教育工作者与社区组织者**：希望基于开源教程组织课程、读书会、训练营和项目共创

---


  本教程分为三个阶段，带你逐层深入具身智能的世界：

| 阶段         | 核心技能                            | 产出                                                         |
| :----------- | :---------------------------------- | :----------------------------------------------------------- |
| **第一阶段** | 机器人学基础、硬件选型、ROS 入门    | 搭建自己的第一台仿真机器人 / 配置仿真环境                    |
| **第二阶段** | 计算机视觉、运动规划、强化学习      | 完成“识别-规划-抓取”闭环任务                                 |
| **第三阶段** | 模仿学习、大模型(VLA/VLN)、Sim2Real | 复现大模型导航VLN、OpenVLA、SmolVLA 等前沿项目，实现仿真或真实部署 |

  ## 🔥 News & Highlights
  - **[2026-08-01]** 新增 [ATEC2026 赛后复盘与开源方案教程](./15-Challenge竞赛/ATEC2026/README.md)：将 Task A/B/D/E 的任务边界、观测契约、环境视角调试、Task B 公开方案对照、评估与复现边界，以及中文 Workspace Memory 统一归入第 15 章，并配套 [Task B 公开复现归档](https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility) 与 Task E 开源资源。
  - **[2026-07-18]** 新增 [Dexbotic-RLinf 工程化 VLA 后训练导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/17-Dexbotic-RLinf工程化VLA后训练导读/README.md)：归入现有 VLA 章节，系统讲解 Dexbotic 如何把 RLinf 作为分布式强化学习后端，用 PPO 对 Dexbotic π0 / DM0 在 LIBERO 上做在线 fine-tuning，覆盖 Dexbotic 三层工具箱架构、RLinf frontend 与 Dexbotic backend 两种启动方式、`dexbotic_pi0` / `dexbotic_dm0` 动态模型注册、actor / rollout / env worker 分工、checkpoint 下载、TensorBoard 指标、standalone evaluation，以及该链路更适合作为工程化后训练导读而非轻量一键复现的边界。
  - **[2026-08-05]** 新增 [DM0.5 高性能推理与 OpenDM 开源导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/18-DM0.5高性能推理与OpenDM导读/README.md)：归入现有 VLA 章节，系统拆解 DM0.5 的 Gemma3 VLM prefix、Action Expert suffix、KV cache、Flow Matching action chunk 和公开 OpenDM 代码入口；进一步分析官网公告中的 Vision TensorRT、Tuned FlexAttention、FP8 E4M3 Tensor Core MLP、Triton fused kernels 与启动期 CUDA Graph 如何组成 9.29× 推理优化链路，并明确 OpenDM 当前可复现的模型/训练/推理/评测范围，以及 DM0.5 专用 9.29× 部署代码尚未在公开仓库完整出现的边界。
  - **[2026-07-18]** 新增 [Robots That Know What to Ask 奖励对齐导读](./05-具身场景的深度和强化学习/04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md)：归入 `05-具身场景的深度和强化学习`，系统拆解 RSS 2026 论文 *Robots That Know What to Ask: Recovering Misaligned Rewards through Targeted Explanations*，重点讲清 ASQ 如何从示教特征方差中识别欠指定奖励维度，用自然语言解释引导人类补充 corrective demonstrations，并覆盖 JacoRobot 仿真、Franka 真实用户实验、LLM filtering、demo-specific rationality / weighting、与 VLA / 世界模型后训练的接口，以及当前未检索到官方代码仓库的复现边界。
  - **[2026-07-18]** 新增 [GE-Sim 2.0 闭环视频世界模拟器导读](./17-具身世界模型/GE-Sim-V2闭环视频世界模拟器导读/README.md)：归入 `17-具身世界模型`，系统拆解 AgiBot Genie Envisioner World Simulator 2.0，重点讲清它如何从动作条件视频生成推进到可闭环调用的 learned world simulator，覆盖 Pixel-aligned Action Condition、Proprioceptive State Expert、World Judge、DMD2 加速、WorldArena 公共榜第一、2B G01 + OmniPicker 开源权重、`WorldModelEnv.step(actions)` 接口、pi05 policy demo rollout、成功/失败示例和当前 World Judge 尚需自接 RewardClient 或等待进一步发布的复现边界。
  - **[2026-07-18]** 新增 [Agentic-VLA 在线适应导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/16-Agentic-VLA在线适应导读/README.md)：归入现有 VLA 章节，系统拆解 *Efficient Online Adaptation for Vision-Language-Action Models*，重点讲清它不是新的 VLA 基座，而是围绕 OpenVLA-OFT 构建的 agentic online adaptation 外环，覆盖 Adaptive Reward Synthesis、Language-Guided Exploration、Experience Memory、GRPO、LIBERO / RoboTwin 2.0 评测、1-shot 与跨任务迁移结果、reward hacking / memory interference 风险，以及当前未检索到官方代码仓库的复现边界。
  - **[2026-07-18]** 新增 [Shape Your Body 价值梯度机器人设计导读](./21-机械臂和机器人设计/04-Shape-Your-Body价值梯度机器人设计导读/README.md)：归入 `21-机械臂和机器人设计`，系统讲解 TU Darmstadt / RIG / DFKI 的 *Value Gradients for Multi-Embodiment Robot Design*，重点拆解如何先训练多具身 policy 和 value function，再冻结 critic 并通过 Value-Gradient Design Search 优化新的机器人身体参数，覆盖 URMA、direct-design critic、soft trust region、50 机器人训练集、190-1177 维连续设计空间、单机器人与 held-out robot 设计结果、设计分析热力图，以及当前项目页显示 `Code (soon)` 的开源边界。
  - **[2026-07-18]** 新增 [VisualThink-VLA 视觉证据推理导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/15-VisualThink-VLA视觉证据推理导读/README.md)：归入现有 VLA 章节，系统拆解 *Visual Intermediate Reasoning for Effective and Low-Latency VLA Policies*，重点讲清为什么文本 CoT 不适合实时机器人闭环控制，VisualThink-VLA 如何用 `bbox / edge / motion / relation` 四类 routed visual evidence 替代长文本推理，覆盖 frozen VLA backbone、Visual State Composer、FullSoft teacher、soft-hard collaborative routing、VisualEvidence-Kit、754.7k VisualEvidence-Set、BridgeData V2 22.8x 延迟加速和当前 MIT 代码骨架复现入口。
  - **[2026-07-18]** 新增 [LWD 真机机群强化学习导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/14-LWD真机机群强化学习导读/README.md)：归入现有 VLA 章节，系统拆解智元 AGIBOT Finch / Shanghai Innovation Institute 的 *Learning while Deploying*，重点讲清真机部署如何变成数据飞轮，覆盖 16 台 AgiBot G1 双臂机器人、8 个真实操作任务、离线到在线 replay、Distributional Implicit Value Learning、Q-learning with Adjoint Matching、flow-based VLA action head、长时程任务收益，以及当前项目页公开但尚未提供一键复现代码入口的复现边界。
  - **[2026-07-18]** 新增 [Dexora 高自由度双臂灵巧 VLA 导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/README.md)：归入现有 VLA 章节，系统拆解清华、智源、港大、北大等团队的 ICRA 2026 Manipulation and Locomotion 最佳论文候选工作，重点讲清 Dexora 为什么是 36-DoF 双臂双灵巧手 VLA，覆盖外骨骼 + Apple Vision Pro 混合遥操作、MuJoCo 数字孪生、100K 级仿真轨迹、10K+ 真实遥操作 episodes、discriminator-guided quality-aware diffusion-transformer 训练、双手灵巧任务、跨本体泛化和当前代码/数据公开入口。
  - **[2026-07-18]** 新增 [Galaxea G0.5 自回归 VLA 导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/12-Galaxea-G0.5自回归VLA导读/README.md)：归入现有 VLA 章节，围绕 “记忆 + CoT + 动作表征” 这条技术线，系统拆解 G0.5 如何用过去 5 秒 6 帧多视角 RGB、Embodiment ID、任务指令和 proprioception 作为输入，在同一条自回归流中生成 `Subtask / BBox / Trace / ActionHint` 与 action tokens，覆盖 ActionCodec、27 维共享动作空间、Qwen3.5 2B VLM-as-Actor、DROID / Bridge-SimplerEnv / RoboTwin / LIBERO / BEHAVIOR-1K 结果和 G0.5 Community License 开源边界。
  - **[2026-07-18]** 新增 [PRTS 强化学习原生 VLA 导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/11-PRTS强化学习原生VLA导读/README.md)：归入现有 VLA 章节，围绕 TeleAI PRTS-Droid 跻身 MolmoSpaces 全球 TOP3 的榜单事件，系统讲解 PRTS 如何把 reward-label-free contrastive RL 放进 VLA 预训练，覆盖 Qwen3-VL backbone、single-forward attention mask、Flow Matching Action Expert、goal-reachability value、LIBERO / RealMan / Flexiv / MolmoSpaces 评测和 CC BY-NC 4.0 开源边界。
  - **[2026-07-18]** 新增 [τ0-WM 统一视频-动作世界模型导读](./17-具身世界模型/tau0-WM统一视频动作世界模型导读/README.md)：归入 `17-具身世界模型`，系统拆解 Shanghai Innovation Institute 与 AGIBOT Finch 的 5B 级开源 video-action world model，覆盖 Wan2.2-TI2V-5B / Action DiT 架构、VAM 与 ACVS 双接口、27.3K 小时异构数据、modality-specific supervision masks、re-denoising consistency、test-time proposal-evaluation-revision，以及 GitHub / Hugging Face 当前可复现和暂未放出的 Simulator / TTC 边界。
  - **[2026-07-18]** 新增 [NBS 主动感知路径规划导读](./08-具身导航及VLN/03前沿VLN复现/03-主动感知NBS导读/README.md)：归入 `08-具身导航及VLN` 的导航规划前沿线，系统讲解 IJRR OnlineFirst 论文 *An Efficient Beam Search Algorithm for Active Perception in Mobile Robotics*，覆盖 active perception 问题定义、Node-wise Beam Search、Expected Gain、RRAG 在线构图、Habitat / HSSD 三类主动感知任务、ANYmal 真机验证，以及 OpenApero 当前代码尚未公开的复现边界。
  - **[2026-07-18]** 新增 [BWM 动作条件世界模型导读](./17-具身世界模型/BWM动作条件世界模型导读/README.md)：归入 `17-具身世界模型`，系统讲解同济 Boundless 团队在 WorldArena 上开源第 1、overall 第 2 的 action-conditioned video world model，覆盖 Wan2.2-TI2V-5B 基座、Action Encoder、AdaLN 动作注入、自回归 rollout、WorldArena 榜单图、官方 GIF 示例、HF 权重和当前训练代码/技术报告未发布边界。
  - **[2026-07-18]** 新增 [3DVLA 三维空间实例增强 VLA 导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/09-3DVLA三维空间实例增强VLA导读/README.md)、[PhysBrain 1.0 物理常识增强 VLA 导读](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/10-PhysBrain物理常识增强VLA导读/README.md)、[HumanoidMimicGen 全身规划数据生成导读](./10-具身智能其他仿真工具及仿真前沿/12HumanoidMimicGen全身规划数据生成导读/README.md) 与 [Gamma-World 多智能体世界模型导读](./17-具身世界模型/Gamma-World多智能体世界模型导读/README.md)：3DVLA 与 PhysBrain 归入现有 VLA 章节，分别拆解 3D 空间实例 token 注入 action expert，以及人类第一人称视频如何转成物理常识监督、再通过 TwinBrainVLA 和 LangForce 迁移到机器人控制；HumanoidMimicGen 归入仿真与数据生成章节，系统讲解少量人形示教、技能约束、全身 IK/运动规划、G1 benchmark 和 sim-and-real co-training；Gamma-World 归入世界模型章节，补充 NVIDIA x 清华多智能体共享世界模型的项目图、代码/权重入口、Simplex RoPE、Sparse Hub Attention、24 FPS 流式 rollout 和两人训练到四人泛化。
  - **[2026-07-07]** 更新 [Datawhale 具身智能零基础入门组队学习路径 v6](./19-Datawhale每月组队学习/06组队学习路径v6.md)：基于最近新增的世界模型、VLA 前沿、Locate Anything 视觉语言定位、LingBot-Map 三维重建、Build123d / Text-to-CAD / ForgeCAD 代码建模、SIM1、UniLab + MotrixSim 与 AMD ROCm 专题，重新组织 20 天组队学习路线，按“操作控制”“感知建图”“导航规划”“世界模型”四条主线拆分，并明确 Task01 截止时间为 07月17日03:00。
  - **[2026-07-02]** 新增 [RoboDream 可组合世界模型数据合成导读](./17-具身世界模型/RoboDream可组合世界模型数据合成导读/README.md)，放入 `17-具身世界模型` 前沿导读线：系统拆解 USC PSI Lab 与 TRI 提出的 *Compositional World Models for Scalable Robot Data Synthesis*，重点讲清它不是可实时 `step(action)` 的神经仿真器，而是基于 robot-only 轨迹、scene prior、object prior 的机器人示教视频合成数据引擎；教程补充官方架构图、prior extraction、retrieval and rebirth、prop-free teleoperation、真实机器人实验表和当前代码开源边界。
  - **[2026-06-29]** 新增 [EventVLA 视觉证据记忆](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/06EventVLA视觉证据记忆导读/README.md)、[WALL-OSS 开源 VLA 模型](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/07WALL-OSS开源VLA模型导读/README.md)、[WALL-X 工程框架导航](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/08WALL-X开源工程框架导航/README.md) 与 [WALL-WM 事件级世界动作模型](./17-具身世界模型/WALL-WM事件级世界动作模型导读/README.md)：前三篇归入现有 VLA 章节，WALL-WM 归入世界模型章节，分别补充官方图解、开源地址、权重/数据入口和当前适合的复现边界。
  - **[2026-06-29]** 新增 [AMD ROCm 策略复刻专题](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README.md)，基于 AMD Ryzen AI MAX+ / ROCm 设备复刻 MuJoCo 抓杯任务，整理 ACT DAgger 诊断、SmolVLA 红/蓝杯加权采样、pi_0 权限 smoke test 与训练门控，并配套 Notebook、成功率图表和成功/失败关键帧，方便大家从排障过程学会在 AMD 平台上迁移具身策略。
  - **[2026-06-27]** 新增 [Locate Anything 视觉语言定位复现教程](./04-具身场景的计算机视觉、3D重建/04-Locate-Anything视觉语言定位.md)，覆盖 `nvidia/LocateAnything-3B` 本地环境、RTX PRO 6000 Blackwell smoke test、Parallel Box Decoding 原理，以及与 YOLO26n 的同图速度和任务边界对比。
  - **[2026-06-27]** 新增 [WoG 条件空间世界模型导读](./17-具身世界模型/WoG条件空间世界模型导读/README.md)，系统拆解 ByteDance Seed 与港大提出的 *World Guidance*：从总览图、两阶段训练、Future Encoder、Q-Former query、condition alignment 到 SIMPLER/真机/人类视频实验，重点说明它不是可 rollout 的传统世界模型，而是把未来观测蒸馏成动作相关 latent condition 来增强 VLA 动作生成。
  - **[2026-06-27]** 新增 [RAW-Dream 任务无关世界模型强化 VLA 导读](./17-具身世界模型/RAW-Dream任务无关世界模型强化VLA/README.md)，梳理微软等作者的 *Reinforcing VLAs in Task-Agnostic World Models*：明确不是 pi0.5 基座，而是 OpenVLA-OFT + Wan 2.1 world model + Qwen3-VL reward + GRPO 的后训练范式，并补充 DNV 技巧、LIBERO/真机评测边界和可继续追踪的开源仓库。
  - **[2026-06-02]** 新增 [RISE 自我改进机器人策略复现教程](./17-具身世界模型/RISE自我改进机器人策略复现/README.md)，覆盖论文方法、组合世界模型流水线、OpenPI policy/value、LTX-Video dynamics model、RLinf imagination RL、Blackwell cu128 环境适配、公开模型资产下载校验，以及官方图与视频素材本地归档。
  - **[2026-06-02]** 新增 [扩散数理基础及问题解析入门](./17-具身世界模型/1、扩散数理基础及问题解析入门/扩散数理基础及问题解析入门.md)，讲解扩散模型需要的数理基础，用简单的代码讲解扩散模型，让大家在不推导复杂公式的前提下，熟悉vae和ddpm，为后续的世界模型学习打下基础。
  - **[2026-06-02]** 新增 [AGILE 人形机器人 Loco-Manipulation Isaac Lab 复现教程](./05-具身场景的深度和强化学习/03AGILE人形机器人Loco-Manipulation复现/README.md)，覆盖官方任务边界、Isaac Sim 5.1 / Isaac Lab 2.3.2 复刻、T1/G1 本地渲染视频、pick-place checkpoint 未随仓库开源说明、评估报告与 Sim2MuJoCo 链路。
  - **[2026-06-02]** 新增 [HIMLoco 四足机器人运动控制 Isaac Lab 复现教程](./05-具身场景的深度和强化学习/02HIMLoco-IsaacLab复现/README.md)，覆盖论文方法、原仓库流水线、Blackwell GPU 旧栈排障、Isaac Lab 迁移、smoke test 曲线与本地渲染视频。
  - **[2026-06-02]** 新增 [Genesis World 1.0 完整体验与机器人仿真流水线](./10-具身智能其他仿真工具及仿真前沿/Genesis仿真环境配置/03Genesis%20World%201.0完整体验与机器人仿真流水线.md)，补充 Genesis World 1.0 环境配置、机器人仿真流程和功能体验记录。
  - **[2026-05-31]** 新增 [ForgeCAD 官方 3D 打印机、键盘与灵巧手案例复现](./21-机械臂和机器人设计/03ForgeCAD视觉逆向工程入门/README.md)，对齐 `KoStard/forgecad-public-kit` 的 `3dprinter-gpt52codex` benchmark，补充视频键盘和官方可动灵巧手，跑通参数渲染 GIF、多视角截图、STEP/STL/3MF 导出和打印检查边界说明。
  - **[2026-05-31]** 新增 [UniLab + MotrixSim 异构机器人 RL 训练复现](./10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/README.md)，覆盖 6GB 显卡上的 state-based RL 训练、资源占用记录、MotrixSim checkpoint 回放，以及 `uv run --no-sync demo teaser` 的 PBR renderer 多视角截图体验。
  - **[2026-05-14]** 新增 [LingBot-Map 视频流式三维重建教程](./04-具身场景的计算机视觉、3D重建/03-LingBot-Map视频流式三维重建.md)，覆盖官方连续校园场景 smoke test、点云渲染预览、环境配置、代码与论文数据流讲解，以及轻量本地 Web demo。
  - **[2026-05-12]** 新增 [Build123d 代码建模](./21-机械臂和机器人设计/01Build123d代码建模入门/README.md) 与 [Text-to-CAD 6DOF 教学机械臂](./21-机械臂和机器人设计/02Text-to-CAD工程化建模入门/README.md) 教程，补充 CAD-as-code、Codex CAD skill、STEP/STL/GLB 生成与预览。
  - **[2026-05-11]** 新增 [DiT4DiT-LIBERO 训练与评估复现教程](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/05DiT4DiT-LIBERO/01DiT4DiT-LIBERO训练与评估.md)，覆盖官方方法图解、LIBERO 评估 smoke test、`libero_spatial` 训练 smoke test、数据元信息修复、日志与多视角视频结果展示。
  - **[2026-04-29]** 新增 [SIM1 柔体仿真与数据生成](./10-具身智能其他仿真工具及仿真前沿/09SIM1柔体仿真与数据生成/01SIM1环境配置与运行.md)、[InternVerse / InternDataEngine 小空间体验教程](./10-具身智能其他仿真工具及仿真前沿/10Internverse教程/InternDataEngine_小空间功能体验教程.md)、[EBench / GenManip 最小复现记录](./09-具身智能数据及评估基准benchmark/04-EBench.md)，补充仿真数据生成、合成数据引擎与评测基准介绍。
  - **[2026-04-11]** 新增了无人机相关教程：系统的讲解无人机从控制到轨迹生成再到轨迹优化的完整流程，包括比较难以理解的微分平坦性，SE3控制器，minimumsnap轨迹优化等内容，包含12个可运行的简单易懂案例，不用复杂的环境，不用复杂的代码，助你从零入门无人机。
  - **[2026-04-07]** 新增LeWorldModel世界模型教程和复现指导！用最通俗的话、最清晰的结构，把LeWM从头到尾讲明白，一步到位带你吃透最新的LeWM世界模型算法，不管是入门世界模型还是深入科研都能用
  - **[2026-03-17]** 新增具身导航综合入门实践视频教程！仅需“半天”即可走通从感知决策再到规划控制的所有的算法：从入门具身导航到深入研究一站式
  - **[2026-03-17]** 新增视频教程：LeHome柔性衣物折叠ICRA2026比赛视频教程 http://xhslink.com/o/2oxCz0RTXcA
  - **[2026-03-16]** 新增达摩院 x Datawhale 组队学习 https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/01-%E8%BE%BE%E6%91%A9%E9%99%A2%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0
  - **[2026-03-16]** 新增LeHome ICRA2026 比赛教程 https://github.com/datawhalechina/every-embodied/blob/main/15-Challenge%E7%AB%9E%E8%B5%9B/LeHome/README.md
  - **[2026-03-05]** 算力自由平台复刻ETPNav连续环境导航算法视频教程 http://xhslink.com/o/8t08X6dROt5 
  - **[2026-03-02]** 新增[具身智能面试教程](./12-具身智能面试问题汇总)
  - **[2026-02-28]** 复刻ACT机械臂抓取水杯算法 https://www.bilibili.com/video/BV1YFAbzoECf
  - **[2026-02-25]** 复刻mujoco机械臂数据采集 https://www.bilibili.com/video/BV1DNAbzpE6Z
  - **[2026-02-23]** 新增春晚武术机器人复刻视频教程 https://www.datawhale.cn/learn/content/258/6228
  - **[2026-02-22]** 新增 ETPNav(IEEE TPAMI 2024)复现教程指导
  - **[2026-02-21]** 新增春晚武术机器人复刻
  - **[2026-02-16]** 新增GenieSim一键启动镜像和视频教程
  - **[2026-01-15]** 新增Habitat导航实战一键启动镜像和视频教程及大模型导航实战VLN教程

  <details>
  <summary>Past News</summary>

  - **[2025-07-02]** 新增 **地瓜 RDK-X5** 连接 LeRobot SO101 遥操作实战教程。
  - **[2025-06-25]** 发布 **OpenVLA** 及 OmniGibson 仿真环境深度指南。
  - **[2025-03-31]** 推出 **具身场景的计算机视觉** 模块，支持物体识别与复杂抓取。
  - **[2025-01-01]** 项目初始化，发布具身智能基础路线图。
    </details>


  <span id="sota"></span>
  ## 📽️视频教程

  Habitat导航基础复现： https://www.datawhale.cn/learn/content/258/6154

  GenieSim一键启动教程：https://www.datawhale.cn/learn/content/258/6172

  Joycon服务器本地异步遥控机械臂教程：https://www.datawhale.cn/learn/content/258/5728

  Mujoco仿真UR机械臂抓取实验：https://www.bilibili.com/video/BV1WhxeznE61/

  春晚武术机器人复刻视频教程：http://xhslink.com/o/1lB9dX0Vt2t

  ACT机械臂抓取水杯算法训练 https://www.bilibili.com/video/BV1YFAbzoECf
    
  Mujoco机械臂数据采集 https://www.bilibili.com/video/BV1DNAbzpE6Z

  算力自由平台复刻ETPNav连续环境导航算法视频教程 http://xhslink.com/o/8t08X6dROt5

  具身导航入门到实践！仅需“半天”即可走通从感知决策再到规划控制的所有的算法：从入门具身导航到深入研究一站式
  - Part 1 环境基础与安装: http://xhslink.com/o/9KuYgWrRpaE
  - Part 2 代码演示讲解与分析: http://xhslink.com/o/2xY3fit6cmj

  


  <span id="learning-map"></span>
  ## 🗺️ 学习地图

  ### 一、基础入门 - 走进机器人的世界

| 章节                | 关键内容                                                     | 状态 |
| :------------------ | :----------------------------------------------------------- | :--- |
| **1. 具身智能概述** | [定义背景、发展历程、未来趋势](./01-具身智能概述/01具身智能概述.md) | ✅    |
| **2. 机器人学基础** | [运动学(DH参数)](<./02-机器人基础和控制、手眼协调/02机器人运动学与 DH 参数.md>)、[动力学](02-机器人基础和控制、手眼协调/03机器人动力学.md)、[坐标变换](02-机器人基础和控制、手眼协调/01机器人空间描述与坐标变换.md) | ✅    |
| **3. 硬件与电子**   | [地瓜开发板实战](03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/01RDKX5超新手入门教程.md)、[LeRobot遥操作](03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/03RDK-X5连接lerobot机械臂进行遥操作.md)、[传感器选型](03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/06传感器选型与数据采集.md)、[执行器原理](03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/05执行器原理与选型.md)、阿加犀开发板实战 | ✅    |
| **4. 软件基础设施** | ROS/ROS2 通信机制、[Build123d / Text-to-CAD / ForgeCAD / Shape Your Body 机器人设计](./21-机械臂和机器人设计/README.md)、[FlashLib 高速机器学习库](./11-其他辅助工具/FlashLib高速机器学习库.md)、[其他辅助工具](./11-其他辅助工具/README.md)、AutoCAD/SolidWorks 基础                   | 🚧    |

  ### 二、核心技术 - 给机器人装上大脑

  | 章节                   | 关键内容                                       | 状态 |
  | :--------------------- | :--------------------------------------------- | :--- |
  | **5. 计算机视觉 (CV)** | [目标检测 (YOLO)](./04-具身场景的计算机视觉、3D重建/02-抓取注意力热图.md)、[视觉语言定位 (Locate Anything)](./04-具身场景的计算机视觉、3D重建/04-Locate-Anything视觉语言定位.md)、[图像分割 (SAM)](./04-具身场景的计算机视觉、3D重建/01-sam和深度估计.md)、[视频流式三维重建 (LingBot-Map)](./04-具身场景的计算机视觉、3D重建/03-LingBot-Map视频流式三维重建.md)、6D 位姿估计   | ✅    |
  | **6. 运动与控制**      | 路径规划 (A*/RRT)、轨迹优化、PID 与 MPC 控制、模仿学习 (IL)、[ACT复现](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/3.train.ipynb)、[Hand-Eye 标定](./02-机器人基础和控制、手眼协调/补充01手眼协调.md)、AnyGrasp 抓取算法、灵巧手操作   | 🚧    |
  | **7. 强化学习 (RL)**   | [多机器人PPO/SAC 算法详解](./05-具身场景的深度和强化学习/01多机器人搬运家具强化学习.md)、[Robots That Know What to Ask 奖励对齐](./05-具身场景的深度和强化学习/04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md)、[UniLab + MotrixSim state-based RL 训练](./10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/README.md)、Isaac Gym 并行训练实战       | ✅    |
  | **8. 仿真环境**        | Isaac Sim 高级渲染、[MuJoCo 物理引擎下OMY/Nova5/Franka机械臂和ACT/Pi0/SmolVLA算法复现](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA)、[Genie-Sim3教程](./10-具身智能其他仿真工具及仿真前沿/08GenieSim3配置.md)、[MotrixSim PBR teaser 渲染体验](./10-具身智能其他仿真工具及仿真前沿/11UniLab-MotrixSim异构RL训练/README.md#7-扩展体验motrixsim-pbr-teaser-渲染器)、[HumanoidMimicGen 全身规划数据生成](./10-具身智能其他仿真工具及仿真前沿/12HumanoidMimicGen全身规划数据生成导读/README.md)、[PhysicsNeMo 物理 AI 求解器](./10-具身智能其他仿真工具及仿真前沿/13-PhysicsNeMo物理AI求解器导读/README.md) | ✅    |

  ### 三、具身大脑 - 前沿论文复现与真机部署

| 章节                  | 关键内容                                    | 状态 |
| :-------------------- | :------------------------------------------ | :--- |
| **9. 具身智能benchmark和数据讲解** | [LIBERO](./09-具身智能数据及评估基准benchmark/01-libero.md)、[SimplerENV](./09-具身智能数据及评估基准benchmark/02-simplerenv.md)       | ✅    |
| **10. VLA 大模型**    | [SmolVLA 训练和部署](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md)、[OpenVLA部署](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/02OpenVLA复现/02openvla复现.md)、[DiT4DiT-LIBERO 训练与评估](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/05DiT4DiT-LIBERO/01DiT4DiT-LIBERO训练与评估.md)、[RT-1 / RT-2 / RT-X 论文解读与代码分析](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/03RT系列论文解读与代码分析/01RT系列论文解读与代码分析.md)、[EventVLA 视觉证据记忆](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/06EventVLA视觉证据记忆导读/README.md)、[WALL-OSS 开源 VLA 模型](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/07WALL-OSS开源VLA模型导读/README.md)、[WALL-X 工程框架导航](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/08WALL-X开源工程框架导航/README.md)、[3DVLA 三维空间实例增强 VLA](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/09-3DVLA三维空间实例增强VLA导读/README.md)、[PhysBrain 1.0 物理常识增强 VLA](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/10-PhysBrain物理常识增强VLA导读/README.md)、[PRTS 强化学习原生 VLA](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/11-PRTS强化学习原生VLA导读/README.md)、[Galaxea G0.5 自回归 VLA](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/12-Galaxea-G0.5自回归VLA导读/README.md)、[Dexora 高自由度双臂灵巧 VLA](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/README.md)、[LWD 真机机群强化学习](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/14-LWD真机机群强化学习导读/README.md)、[VisualThink-VLA 视觉证据推理](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/15-VisualThink-VLA视觉证据推理导读/README.md)、[Agentic-VLA 在线适应](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/16-Agentic-VLA在线适应导读/README.md)、[Dexbotic-RLinf 工程化 VLA 后训练](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/17-Dexbotic-RLinf工程化VLA后训练导读/README.md)、[DM0.5 高性能推理与 OpenDM](./06-策略抓取或抓取VLA/大模型控制、VLA、VLM/18-DM0.5高性能推理与OpenDM导读/README.md)  | ✅    |
| **11. 具身世界模型** | [LeWM 世界模型分析解读与实验复现](./17-具身世界模型/Leworldmodel分析解读与实验复现/Leworldmodel分析解读与实验复现.md)、[RAW-Dream 任务无关世界模型强化 VLA 导读](./17-具身世界模型/RAW-Dream任务无关世界模型强化VLA/README.md)、[WoG 条件空间世界模型导读](./17-具身世界模型/WoG条件空间世界模型导读/README.md)、[WALL-WM 事件级世界动作模型](./17-具身世界模型/WALL-WM事件级世界动作模型导读/README.md)、[RoboDream 可组合世界模型数据合成导读](./17-具身世界模型/RoboDream可组合世界模型数据合成导读/README.md)、[Gamma-World 多智能体世界模型](./17-具身世界模型/Gamma-World多智能体世界模型导读/README.md)、[BWM 动作条件世界模型](./17-具身世界模型/BWM动作条件世界模型导读/README.md)、[τ0-WM 统一视频-动作世界模型](./17-具身世界模型/tau0-WM统一视频动作世界模型导读/README.md)、[GE-Sim 2.0 闭环视频世界模拟器](./17-具身世界模型/GE-Sim-V2闭环视频世界模拟器导读/README.md) | ✅    |
| **12. 导航与主动感知**    | [VLN概念基础](./08-具身导航及VLN/03前沿VLN复现/01VLNCE/02ETPNav代码复现.md)、[ETPNav](./08-具身导航及VLN/03前沿VLN复现/01VLNCE/02ETPNav代码复现.md)、[NBS 主动感知路径规划](./08-具身导航及VLN/03前沿VLN复现/03-主动感知NBS导读/README.md)  | ✅    |
| **13. 综合项目复现** | [无人机多模态LLM导航](./13-其他前沿项目复现/无人机大模型+Groundingdino实践/无人机多模态大模型.md)| ✅    |

  ### 四、专题组队学习 - 面向活动组织与专题共学

| 专题 | 关键内容 | 状态 |
| :--- | :--- | :--- |
| **19. Datawhale 每月组队学习路径** | [具身智能零基础入门 v6](./19-Datawhale每月组队学习/06组队学习路径v6.md)、[v5 旧版路径](./19-Datawhale每月组队学习/05组队学习路径v5.md) | ✅ |
| **16. 达摩院组队学习专题** | [专题导航](./16-专题组队学习/01-达摩院组队学习/README.md)、[Task01 理论与趋势](./16-专题组队学习/01-达摩院组队学习/Task%2001_理论与趋势.md)、[Task02 技术透视](./16-专题组队学习/01-达摩院组队学习/Task%2002_技术透视.md)、[Task03 演示实操](./16-专题组队学习/01-达摩院组队学习/Task%2003_详细演示实操%20-%20从想法到实现.md)、[Task04 总结分享](./16-专题组队学习/01-达摩院组队学习/Task%2004_总结分享.md) | ✅ |

  ## 🛠️ 环境要求

  在开始之前，请确保你的开发环境满足以下基础要求（不同子项目复现可能略有不同，请参考各自项目的readme，我们会通过conda或mamba环境实现）：

  - **Python**: 3.8+
  - **GPU**: 推荐 NVIDIA RTX 3060+ (用于 Isaac Sim 渲染与 RL 训练)
  - **OS**: Ubuntu 20.04 / 22.04 (推荐)
  - **Core Libs**:
    - `MuJoCo` (物理引擎)
    - `Isaac Sim` (Nvidia 仿真平台，需要IsaacSim配置时我们会为大家提供预安装的一键启动镜像)

  ## 🤝 参与贡献

  我们非常欢迎社区的贡献！无论是修复 Bug、补充文档，还是提交新的复现代码！

  

## 贡献者名单（教程部分）

| 姓名            | 职责                     | 简介           | GitHub                                                 |
| --------------- | ------------------------ | -------------- | ------------------------------------------------------ |
| Ethan-Chen-plus | 项目负责人               | 中国科学院大学 | [@Ethan-Chen-plus](https://github.com/Ethan-Chen-plus) |
| YYPro           | 第1、2、3、4章贡献者     | 中国科学院大学 | [@YYpro](https://github.com/Suibian-YY-pro)            |
| 李昀迪          | 第2、8、13章贡献者       | 北京科技大学   | [@muzilyd](https://github.com/muzilyd)                 |
| 张天一          | 第8章贡献者              | 北京工业大学   | [@GodoneZ](https://github.com/GodoneZ)                 |
| 陈可为          | 第5、6、7、9、10章贡献者 | 中国科学院大学 | [@Ethan-Chen-plus](https://github.com/Ethan-Chen-plus) |
| 霍海杰          | 第1章贡献者             | 佛山大学      | [@howe12](https://github.com/howe12) |

  **其他贡献者风采（补充相关资源、单独push readme子教程、文档挑错）：**

  感谢以下小伙伴的参与和贡献：howe, Miles, 麦芒, HAO, [zhangningboo](https://github.com/zhangningboo)，[机智流硬件冷小莫](1412195676@qq.com)，[icarried](https://github.com/icarried)[修复雅可比矩阵文档](https://github.com/datawhalechina/every-embodied/issues/29) , [PeterH0323-贡献RDKx5入门教程](https://github.com/PeterH0323) 

  **英文教程翻译者贡献：**

我们同时准备了[英文部分](./en)的教程，感谢如下小伙伴的贡献：Lune、刘远洋、苏家煜、梁坚斌

  <div align=center style="margin-top: 30px;">
    <a href="https://github.com/datawhalechina/every-embodied/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=datawhalechina/every-embodied" />
    </a>
  </div>

  ## Star History

  [![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/every-embodied&type=date&legend=top-left)](https://www.star-history.com/#datawhalechina/every-embodied&type=date&legend=top-left)

  ## 📬 联系我们

  <div align=center>
  <p>如果你发现教程有 Bug，或者有任何想要交流、学习、讨论、吐槽的具身智能内容，或者有更 High-level 的 Idea，别让它只停留在你的神经元里！扫码直连微信交流群（Every-Embodied读者交流2群），一起为具身智能‘注入灵魂’。</p>
  <img src="./assets/wechat.png" width="180" height="180">
  <p>扫描下方二维码关注公众号：Datawhale</p>
  <img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180">
  <p>
    <a href="https://datawhale-eai.feishu.cn/wiki/QvxQwV2Qyij1NakV9IdcjTl1nJd">📚 飞书知识库</a> | 
    <a href="https://www.datawhale.cn/learn/summary/258">🌐 官方网站</a>
  </p>


  </div>

  ## 📄 LICENSE

  <div align="center">
  <a rel="license" href="https://creativecommons.org/licenses/by/4.0/">
    <img alt="CC BY 4.0 License" style="border-width:0" src="https://img.shields.io/badge/license-CC--BY%204.0-lightgrey" />
  </a>
  <br />
  本项目文档与教程内容采用 <strong>Creative Commons Attribution 4.0 International (CC BY 4.0)</strong> 许可协议。
  <br />
  你可以自由分享与改编本项目内容，但需保留来源署名。详细条款见 <a href="LICENSE">LICENSE</a>。
  </div>
