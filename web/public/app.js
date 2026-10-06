const state = {
  catalog: null,
  flatDocs: [],
  currentDoc: null,
  search: "",
  openGroups: new Set(),
  homeTypingTimer: null,
};

const els = {
  topMeta: document.getElementById("topMeta"),
  sidebar: document.getElementById("sidebar"),
  openNav: document.getElementById("openNav"),
  closeNav: document.getElementById("closeNav"),
  scrim: document.getElementById("scrim"),
  searchInput: document.getElementById("searchInput"),
  navTree: document.getElementById("navTree"),
  catalogStatus: document.getElementById("catalogStatus"),
  breadcrumb: document.getElementById("breadcrumb"),
  docTitle: document.getElementById("docTitle"),
  article: document.getElementById("article"),
  outlineNav: document.getElementById("outlineNav"),
  prevLink: document.getElementById("prevLink"),
  nextLink: document.getElementById("nextLink"),
  toggleCatalog: document.getElementById("toggleCatalog"),
  refreshCatalog: document.getElementById("refreshCatalog"),
  catalogOverview: document.getElementById("catalogOverview"),
  overviewGrid: document.getElementById("overviewGrid"),
  catalogLink: document.getElementById("catalogLink"),
  brandLogo: document.getElementById("brandLogo"),
  homePage: document.getElementById("homePage"),
  showcasePage: document.getElementById("showcasePage"),
  homeHeroImage: document.getElementById("homeHeroImage"),
  homeTitle: document.getElementById("homeTitle"),
  homeDemoGrid: document.getElementById("homeDemoGrid"),
  homeModuleList: document.getElementById("homeModuleList"),
  starCount: document.getElementById("starCount"),
};

const homeTitleLines = ["从 0 到 1，", "走进具身智能。"];
const GITHUB_REPO_URL = "https://github.com/datawhalechina/every-embodied";

const homeDemos = [
  {
    title: "G0.5 × 机器人弹琴",
    subtitle: "《我爱你，中国》双手多指演奏",
    type: "video",
    media: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/assets/piano-xiaohongshu.mp4",
    poster: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/assets/piano-xiaohongshu-poster.jpg",
    doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/README.md",
  },
  {
    title: "钢琴移位后继续演奏",
    subtitle: "G0.5 辅助双臂重新就位，接回下一个乐句",
    type: "video",
    media: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/assets/piano-relocation-performance.mp4",
    poster: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/assets/piano-relocation-poster.jpg",
    doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/20-GPT6-Astra机器人弹琴与上下文工程/G05.md",
  },
  {
    title: "AMD Physical AI Evidence Lab",
    subtitle: "家庭操作、灵巧手、仿真迁移与安全控制的完整 AMD 实践",
    mediaUrl:
      "https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/posters/dexjoco/recovery/bimanual-assembly.jpg",
    href: "#/amd-physical-ai",
  },
  {
    title: "项目快速入门",
    subtitle: "基于 Mujoco 一键了解项目基础",
    media: "assets/quick_start.gif",
    doc: "examples/README.md",
  },
  {
    title: "Genie Sim 的 Pi0 部署",
    subtitle: "基于 Pi0 和 Isaac Sim 实现高保真仿真",
    media: "assets/zhiyuan.gif",
    doc: "10-具身智能其他仿真工具及仿真前沿/08GenieSim3配置.md",
  },
  {
    title: "LeRobot 遥操作",
    subtitle: "支持地瓜 RDK-X5 连接 SO101 机械臂实操",
    media: "assets/2025-07-02-20-50-54-image.png",
    doc: "03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/03RDK-X5连接lerobot机械臂进行遥操作.md",
  },
  {
    title: "视觉语义感知",
    subtitle: "场景分割与目标检测，让机器人“看懂”环境",
    media: "assets/2025-06-17-12-11-28-image.png",
    doc: "04-具身场景的计算机视觉、3D重建/01-sam和深度估计.md",
  },
  {
    title: "LLM 控制无人机导航 VLN",
    subtitle: "通过 WebUI 快速上手无人机大模型 VLN 导航",
    media: "assets/plane.gif",
    doc: "13-其他前沿项目复现/无人机大模型+Groundingdino实践/无人机多模态大模型.md",
  },
  {
    title: "基于 SmolVLA 微调 LIBERO 基准",
    subtitle: "小型 VLA 测试机器人终身学习基准",
    media: "assets/libero.gif",
    doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md",
  },
  {
    title: "春晚机器人舞蹈复刻",
    subtitle: "输入视频，生成机器人 Mujoco 仿真动作",
    media: "assets/chunwan_robot.gif",
    doc: "07-机器人操作、运动控制/Locomotion/01春晚舞蹈机器人复刻.md",
  },
  {
    title: "ETPNav-VLN 导航复现",
    subtitle: "连续环境视觉语言导航强基线复现",
    media: "assets/ETPNav.gif",
    doc: "08-具身导航及VLN/03前沿VLN复现/01VLNCE/02ETPNav代码复现.md",
  },
];

const homeModules = [
  {
    id: "module-navigation",
    label: "Navigation Model",
    title: "导航模型",
    headline: "从地图、定位到语言导航。",
    copy: "先理解导航算法和仿真环境，再进入 VLN-CE、ETPNav 等视觉语言导航复现。",
    cards: [
      {
        title: "VLN-CE 方法概述",
        copy: "从连续环境理解视觉语言导航的任务定义与方法。",
        image: "08-具身导航及VLN/03前沿VLN复现/01VLNCE/images/image-28.png",
        doc: "08-具身导航及VLN/03前沿VLN复现/01VLNCE/01连续环境下视觉语言导航（VLN-CE）方法概述.md",
      },
      {
        title: "Habitat 仿真环境基础",
        copy: "搭建 Habitat Lab / Sim，准备导航实验环境。",
        image: "08-具身导航及VLN/02仿真环境基础/habitat导航环境/assets/observation_0.png",
        doc: "08-具身导航及VLN/02仿真环境基础/habitat导航环境/habitat_lab基础.md",
      },
      {
        title: "ETPNav-VLN 导航复现",
        copy: "进入连续环境下视觉语言导航方法复现。",
        image: "assets/ETPNav.gif",
        doc: "08-具身导航及VLN/03前沿VLN复现/01VLNCE/02ETPNav代码复现.md",
      },
      {
        title: "NBS 主动感知路径规划",
        copy: "理解节点级 Beam Search、Expected Gain 和 RRAG 如何服务移动机器人主动建图。",
        image: "08-具身导航及VLN/03前沿VLN复现/03-主动感知NBS导读/assets/hardware_surface_reconstruction.png",
        doc: "08-具身导航及VLN/03前沿VLN复现/03-主动感知NBS导读/README.md",
      },
    ],
  },
  {
    id: "module-operation",
    label: "Manipulation Model",
    title: "操作模型",
    headline: "让机器人理解动作，并完成操作。",
    copy: "把机器人运动学、动力学、抓取策略、VLA 和运动控制串成一条操作学习路线。",
    cards: [
      {
        title: "机器人运动学与 DH 参数",
        copy: "从空间描述和关节建模进入机械臂控制基础。",
        image: "01-具身智能概述/module1_2_机器人系统组成与分类/images/six_dof_manipulator.png",
        doc: "02-机器人基础和控制、手眼协调/02机器人运动学与 DH 参数.md",
      },
      {
        title: "VLA 综述",
        copy: "理解视觉语言动作模型如何连接感知与控制。",
        image: "assets/libero.gif",
        doc: "06-策略抓取或抓取VLA/01VLA相关总结综述.md",
      },
      {
        title: "EventVLA 视觉证据记忆",
        copy: "理解长时域 VLA 如何保存关键视觉证据。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/06EventVLA视觉证据记忆导读/assets/eventvla_framework.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/06EventVLA视觉证据记忆导读/README.md",
      },
      {
        title: "WALL-OSS 与 WALL-X",
        copy: "梳理开源 VLA 模型、权重和工程复现入口。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/07WALL-OSS开源VLA模型导读/assets/walloss_lerobot_overview.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/07WALL-OSS开源VLA模型导读/README.md",
      },
      {
        title: "3DVLA 三维空间实例增强",
        copy: "理解 3D 空间、实例和遮挡补全 token 如何增强 VLA 动作预测。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/09-3DVLA三维空间实例增强VLA导读/assets/3dvla_architecture.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/09-3DVLA三维空间实例增强VLA导读/README.md",
      },
      {
        title: "PhysBrain 1.0 物理常识增强 VLA",
        copy: "理解人类第一人称视频如何转成物理常识监督并迁移到机器人控制。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/10-PhysBrain物理常识增强VLA导读/assets/physbrain_system_overview.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/10-PhysBrain物理常识增强VLA导读/README.md",
      },
      {
        title: "PRTS 强化学习原生 VLA",
        copy: "理解 reward-label-free contrastive RL 如何让 VLA 具备目标可达性意识。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/11-PRTS强化学习原生VLA导读/assets/prts-overview-v1.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/11-PRTS强化学习原生VLA导读/README.md",
      },
      {
        title: "Galaxea G0.5 自回归 VLA",
        copy: "拆解视觉记忆、结构化 CoT 与 ActionCodec 如何统一成动作 token 流。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/12-Galaxea-G0.5自回归VLA导读/assets/images_teaser.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/12-Galaxea-G0.5自回归VLA导读/README.md",
      },
      {
        title: "Dexora 高自由度双臂灵巧 VLA",
        copy: "理解 36-DoF 双臂双手、混合遥操作和质量感知训练如何推进灵巧操作。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/assets/img_teaser.JPG",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/13-Dexora高自由度双臂灵巧VLA导读/README.md",
      },
      {
        title: "LWD 真机机群强化学习",
        copy: "拆解智元 LWD 如何用 DIVL、QAM 和部署数据飞轮持续强化通用 VLA。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/14-LWD真机机群强化学习导读/assets/teaser.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/14-LWD真机机群强化学习导读/README.md",
      },
      {
        title: "VisualThink-VLA 视觉证据推理",
        copy: "理解 routed visual evidence 如何替代长文本 CoT，实现低延迟思考再行动。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/15-VisualThink-VLA视觉证据推理导读/assets/overview.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/15-VisualThink-VLA视觉证据推理导读/README.md",
      },
      {
        title: "Agentic-VLA 在线适应",
        copy: "拆解奖励合成、语言引导探索和经验记忆如何提升 VLA 在线后训练效率。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/16-Agentic-VLA在线适应导读/assets/agentic_vla_framework.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/16-Agentic-VLA在线适应导读/README.md",
      },
      {
        title: "Dexbotic-RLinf 后训练工程",
        copy: "理解 Dexbotic VLA 工具箱如何接入 RLinf 分布式 PPO 后端。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/17-Dexbotic-RLinf工程化VLA后训练导读/assets/dexbotic_overview.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/17-Dexbotic-RLinf工程化VLA后训练导读/README.md",
      },
      {
        title: "DM0.5 高性能推理",
        copy: "拆解 OpenDM 的 VLM + Action Expert、Flow Matching 与 9.29× 推理优化边界。",
        image: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/18-DM0.5高性能推理与OpenDM导读/assets/dm05_header_zh.png",
        doc: "06-策略抓取或抓取VLA/大模型控制、VLA、VLM/18-DM0.5高性能推理与OpenDM导读/README.md",
      },
      {
        title: "Robots That Know What to Ask 奖励对齐",
        copy: "理解机器人如何识别欠指定奖励特征，并用解释引导人类补充示教。",
        image: "05-具身场景的深度和强化学习/04-Robots-That-Know-What-to-Ask奖励对齐导读/assets/method_overview.png",
        doc: "05-具身场景的深度和强化学习/04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md",
      },
      {
        title: "春晚舞蹈机器人复刻",
        copy: "把动作迁移到机器人仿真，理解运动控制实践。",
        image: "assets/chunwan_robot.gif",
        doc: "07-机器人操作、运动控制/Locomotion/01春晚舞蹈机器人复刻.md",
      },
    ],
  },
  {
    id: "module-world",
    label: "World Model",
    title: "世界模型",
    headline: "在仿真世界里学习与预测。",
    copy: "围绕世界模型、强化学习和仿真平台，建立可复现、可评估的具身智能实验环境。",
    cards: [
      {
        title: "Leworldmodel 分析解读与实验复现",
        copy: "从论文解读到复现实验，理解具身世界模型。",
        image: "17-具身世界模型/Leworldmodel分析解读与实验复现/assets/lewm.png",
        doc: "17-具身世界模型/Leworldmodel分析解读与实验复现/Leworldmodel分析解读与实验复现.md",
      },
      {
        title: "RAW-Dream 任务无关世界模型强化 VLA",
        copy: "读懂 OpenVLA-OFT、Wan 2.1 和 Qwen3-VL 组成的 VLA 后训练范式。",
        image: "17-具身世界模型/RAW-Dream任务无关世界模型强化VLA/assets/motivation.png",
        doc: "17-具身世界模型/RAW-Dream任务无关世界模型强化VLA/README.md",
      },
      {
        title: "WoG 条件空间世界模型导读",
        copy: "拆解未来观测如何被蒸馏成动作相关 latent condition。",
        image: "17-具身世界模型/WoG条件空间世界模型导读/assets/pipe.png",
        doc: "17-具身世界模型/WoG条件空间世界模型导读/README.md",
      },
      {
        title: "WALL-WM 事件级世界动作模型",
        copy: "理解如何按语义事件边界预演未来视频与动作。",
        image: "17-具身世界模型/WALL-WM事件级世界动作模型导读/assets/wallwm_framework.png",
        doc: "17-具身世界模型/WALL-WM事件级世界动作模型导读/README.md",
      },
      {
        title: "RoboDream 可组合世界模型",
        copy: "理解 robot-only 轨迹、场景先验和物体先验如何合成机器人示教数据。",
        image: "17-具身世界模型/RoboDream可组合世界模型数据合成导读/assets/robodream_architecture.png",
        doc: "17-具身世界模型/RoboDream可组合世界模型数据合成导读/README.md",
      },
      {
        title: "Gamma-World 多智能体世界模型",
        copy: "拆解共享世界、多 agent 独立控制和 24 FPS 流式 rollout。",
        image: "17-具身世界模型/Gamma-World多智能体世界模型导读/assets/gamma_method.png",
        doc: "17-具身世界模型/Gamma-World多智能体世界模型导读/README.md",
      },
      {
        title: "BWM 动作条件世界模型",
        copy: "理解 Wan2.2 基座如何接入机器人动作轨迹并自回归生成未来操作视频。",
        image: "17-具身世界模型/BWM动作条件世界模型导读/assets/track1_open_source.png",
        doc: "17-具身世界模型/BWM动作条件世界模型导读/README.md",
      },
      {
        title: "τ0-WM 统一视频动作世界模型",
        copy: "拆解 VAM、ACVS 和测试时动作评估修正如何组成 5B 级机器人世界模型。",
        image: "17-具身世界模型/tau0-WM统一视频动作世界模型导读/assets/tau0wm_teaser.jpg",
        doc: "17-具身世界模型/tau0-WM统一视频动作世界模型导读/README.md",
      },
      {
        title: "GE-Sim 2.0 闭环世界模拟器",
        copy: "理解动作条件多视角视频、状态专家和 RewardClient 如何组成可闭环调用的 learned simulator。",
        image: "17-具身世界模型/GE-Sim-V2闭环视频世界模拟器导读/assets/gesim2_overview.png",
        doc: "17-具身世界模型/GE-Sim-V2闭环视频世界模拟器导读/README.md",
      },
      {
        title: "HumanoidMimicGen 全身规划数据生成",
        copy: "学习少量人形遥操作示教如何扩成千级 loco-manipulation 数据。",
        image: "10-具身智能其他仿真工具及仿真前沿/12HumanoidMimicGen全身规划数据生成导读/assets/hmgen_benchmark.png",
        doc: "10-具身智能其他仿真工具及仿真前沿/12HumanoidMimicGen全身规划数据生成导读/README.md",
      },
      {
        title: "PhysicsNeMo 物理 AI 求解器",
        copy: "理解 neural operator、PINN 和物理 surrogate model 与机器人仿真的边界。",
        image: "10-具身智能其他仿真工具及仿真前沿/13-PhysicsNeMo物理AI求解器导读/assets/physicsnemo-card.svg",
        doc: "10-具身智能其他仿真工具及仿真前沿/13-PhysicsNeMo物理AI求解器导读/README.md",
      },
      {
        title: "ManiSkill 强化学习",
        copy: "学习任务、数据集和强化学习实验如何连接仿真世界。",
        image: "10-具身智能其他仿真工具及仿真前沿/assets/2025-03-07-16-20-07-e32f0eaa26b84a2f2dfa61d5d594f7e.png",
        doc: "10-具身智能其他仿真工具及仿真前沿/Maniskill详细文档/05强化学习.md",
      },
      {
        title: "GenieSim3 配置",
        copy: "使用前沿仿真工具连接高保真场景和策略部署。",
        image: "assets/zhiyuan.gif",
        doc: "10-具身智能其他仿真工具及仿真前沿/08GenieSim3配置.md",
      },
    ],
  },
  {
    id: "module-practice",
    label: "Application Practice",
    title: "应用实践",
    headline: "从硬件、数据到完整项目复现。",
    copy: "围绕开发板、LeRobot、数据基准、无人机和比赛项目，把教程落到真实工作流。",
    cards: [
      {
        title: "RDK-X5 超新手入门教程",
        copy: "从开发板准备开始，进入具身硬件实践。",
        image: "assets/2025-07-02-20-50-54-image.png",
        doc: "03-机器人硬件、lerobot及地瓜RDK-X5开发板控制教程/01RDKX5超新手入门教程.md",
      },
      {
        title: "LIBERO 基准",
        copy: "用数据与基准评估机器人学习任务。",
        image: "09-具身智能数据及评估基准benchmark/assets/fig1.png",
        doc: "09-具身智能数据及评估基准benchmark/01-libero.md",
      },
      {
        title: "无人机多模态大模型",
        copy: "把多模态大模型用于无人机导航与交互实践。",
        image: "assets/plane.gif",
        doc: "13-其他前沿项目复现/无人机大模型+Groundingdino实践/无人机多模态大模型.md",
      },
      {
        title: "Shape Your Body 价值梯度机器人设计",
        copy: "理解多具身价值函数如何通过梯度优化机器人身体参数。",
        image: "21-机械臂和机器人设计/04-Shape-Your-Body价值梯度机器人设计导读/assets/hero.png",
        doc: "21-机械臂和机器人设计/04-Shape-Your-Body价值梯度机器人设计导读/README.md",
      },
      {
        title: "FlashLib 高速机器学习库",
        copy: "把 KMeans、KNN、PCA、SVD、ANN 等经典 ML 算子迁移到现代 GPU pipeline。",
        image: "11-其他辅助工具/assets/flashlib-card.svg",
        doc: "11-其他辅助工具/FlashLib高速机器学习库.md",
      },
    ],
  },
];

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function idFromRelPath(relPath) {
  return btoa(unescape(encodeURIComponent(relPath)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function encodePathForUrl(relPath) {
  return relPath
    .replace(/\\/g, "/")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function fileHref(relPath) {
  return `files/${encodePathForUrl(relPath)}`;
}

function docHash(id, anchor = "") {
  return `#/doc/${id}${anchor ? `?anchor=${encodeURIComponent(anchor)}` : ""}`;
}

function setIconHref(selector, href) {
  const link = document.querySelector(selector);
  if (link && href) {
    link.href = href;
  }
}

function formatCompactNumber(value) {
  if (!Number.isFinite(value)) return "--";
  if (value >= 1000000) return `${(value / 1000000).toFixed(1).replace(/\.0$/, "")}m`;
  if (value >= 1000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(value);
}

function parseQuery(query = "") {
  return Object.fromEntries(new URLSearchParams(query.replace(/^\?/, "")));
}

function getRoute() {
  const hash = window.location.hash || "#/";
  if (hash === "#/" || hash === "#") {
    return { type: "home" };
  }

  const homeMatch = hash.match(/^#\/home(?:\/([^?]+))?$/);
  if (homeMatch) {
    return { type: "home", anchor: homeMatch[1] || "" };
  }

  if (hash === "#homeDemos" || hash === "#homePaths" || /^#module-/.test(hash)) {
    return { type: "home", anchor: hash.slice(1) };
  }

  const showcaseMatch = hash.match(/^#\/amd-physical-ai(?:\?(.*))?$/);
  if (showcaseMatch) {
    const query = parseQuery(showcaseMatch[1]);
    return { type: "showcase", anchor: query.anchor || "" };
  }

  const bookMatch = hash.match(/^#\/book(?:\?(.*))?$/);
  if (bookMatch) {
    return {
      type: "doc",
      id: state.flatDocs[0]?.id || "",
      anchor: "",
      openCatalog: parseQuery(bookMatch[1]).catalog === "1",
    };
  }

  const docMatch = hash.match(/^#\/doc\/([^?]+)(?:\?(.*))?$/);
  if (docMatch) {
    const query = parseQuery(docMatch[2]);
    return {
      type: "doc",
      id: docMatch[1],
      anchor: query.anchor ? decodeURIComponent(query.anchor) : "",
      openCatalog: query.catalog === "1",
    };
  }

  return { type: "home" };
}

function setMode(mode) {
  if (mode !== "home") {
    els.homeDemoGrid.querySelectorAll("video").forEach((video) => video.pause());
  }
  document.body.classList.toggle("home-mode", mode === "home");
  document.body.classList.toggle("book-mode", mode === "book");
  document.body.classList.toggle("showcase-mode", mode === "showcase");
  els.homePage.hidden = mode !== "home";
  els.showcasePage.hidden = mode !== "showcase";
}

function setCatalogOverview(open) {
  els.catalogOverview.hidden = !open;
  els.toggleCatalog.textContent = open ? "收起章节总览" : "展开章节总览";
}

function closeSidebar() {
  els.sidebar.classList.remove("open");
  els.scrim.classList.remove("open");
}

function openSidebar() {
  els.sidebar.classList.add("open");
  els.scrim.classList.add("open");
}

async function fetchJson(relPath) {
  const response = await fetch(relPath, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`读取失败: ${relPath}`);
  }
  return response.json();
}

async function fetchCatalog() {
  els.catalogStatus.textContent = "扫描 Markdown 中";
  state.catalog = await fetchJson("data/catalog.json");
  state.flatDocs = state.catalog.docs;

  if (state.catalog.project?.logo) {
    els.brandLogo.src = state.catalog.project.logo;
  }
  if (state.catalog.project?.favicon) {
    setIconHref('link[rel="icon"]', state.catalog.project.favicon);
    setIconHref('link[rel="apple-touch-icon"]', state.catalog.project.favicon);
  }

  els.topMeta.textContent = `${state.catalog.stats.groups} 组 · ${state.catalog.stats.docs} 篇文档`;
  els.catalogStatus.textContent = `已自动识别 ${state.catalog.stats.docs} 篇`;
  renderNav();
  renderOverview();
  renderHome();
}

async function fetchGithubStats() {
  try {
    const stats = await fetchJson("data/github.json");
    els.starCount.textContent = formatCompactNumber(Number(stats.stars));
  } catch {
    els.starCount.textContent = "--";
  }
}

function renderNav() {
  const keyword = state.search.trim().toLowerCase();
  const groups = state.catalog.groups
    .map((group) => ({
      ...group,
      docs: group.docs.filter((doc) => {
        if (!keyword) return true;
        return `${doc.title} ${doc.relPath} ${doc.groupTitle}`.toLowerCase().includes(keyword);
      }),
    }))
    .filter((group) => group.docs.length);

  els.navTree.innerHTML =
    groups
      .map((group) => {
        const isOpen = Boolean(keyword) || state.openGroups.has(group.key);
        return `
          <section class="nav-group">
            <button class="nav-group-toggle" type="button" data-group="${escapeHtml(group.key)}" aria-expanded="${isOpen ? "true" : "false"}">
              <span>${escapeHtml(group.title)}</span>
              <span>
                <span class="nav-group-count">${group.docs.length}</span>
                <span class="nav-group-chevron">›</span>
              </span>
            </button>
            <div class="nav-doc-list" ${isOpen ? "" : "hidden"}>
              ${group.docs
                .map(
                  (doc) => `
                    <button class="nav-doc${state.currentDoc?.id === doc.id ? " active" : ""}" type="button" data-id="${doc.id}">
                      <span class="doc-index">${doc.order}</span>
                      <span>
                        <span class="doc-title">${escapeHtml(doc.title)}</span>
                        <span class="doc-path">${escapeHtml(doc.dirname || doc.relPath)}</span>
                      </span>
                    </button>
                  `,
                )
                .join("")}
            </div>
          </section>
        `;
      })
      .join("") || '<div class="loading">没有匹配的 Markdown。</div>';
}

function renderOverview() {
  els.overviewGrid.innerHTML = state.catalog.groups
    .filter((group) => group.docs.length)
    .map((group) => {
      const first = group.docs[0];
      const excerpt = first.excerpt || "从当前文件夹读取章节，按路径顺序展示。";
      return `
        <a class="overview-card" href="${docHash(first.id)}">
          <span class="overview-badge">${group.docs.length} 篇</span>
          <strong>${escapeHtml(group.title)}</strong>
          <p>${escapeHtml(excerpt)}</p>
        </a>
      `;
    })
    .join("");
}

function stopHomeTitleTyping() {
  if (state.homeTypingTimer) {
    window.clearTimeout(state.homeTypingTimer);
    state.homeTypingTimer = null;
  }
}

function setHomeTitleText() {
  els.homeTitle.classList.remove("is-typing");
  els.homeTitle.innerHTML = homeTitleLines.map((line) => `<span>${escapeHtml(line)}</span>`).join("");
}

function typeHomeTitle() {
  stopHomeTitleTyping();
  if (!els.homeTitle) return;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    setHomeTitleText();
    return;
  }

  els.homeTitle.innerHTML = homeTitleLines.map(() => "<span></span>").join("");
  els.homeTitle.classList.add("is-typing");

  const spans = Array.from(els.homeTitle.querySelectorAll("span"));
  let lineIndex = 0;
  let charIndex = 0;

  const tick = () => {
    const currentLine = homeTitleLines[lineIndex];
    const currentSpan = spans[lineIndex];

    if (!currentLine || !currentSpan) {
      els.homeTitle.classList.remove("is-typing");
      spans.forEach((span) => span.classList.remove("is-active"));
      state.homeTypingTimer = null;
      return;
    }

    spans.forEach((span) => span.classList.remove("is-active"));
    currentSpan.classList.add("is-active");
    currentSpan.textContent = currentLine.slice(0, charIndex + 1);
    charIndex += 1;

    if (charIndex >= currentLine.length) {
      lineIndex += 1;
      charIndex = 0;
      state.homeTypingTimer = window.setTimeout(tick, 260);
      return;
    }

    state.homeTypingTimer = window.setTimeout(tick, 72);
  };

  state.homeTypingTimer = window.setTimeout(tick, 180);
}

function renderHome() {
  const project = state.catalog.project || {};
  const heroImage = project.homepageHero || project.hero || project.cover;
  if (heroImage) {
    els.homeHeroImage.src = heroImage;
    els.homePage.style.setProperty("--hero-image", `url("${heroImage}")`);
  }

  els.homeDemoGrid.innerHTML = homeDemos
    .map((demo) => {
      const doc = state.flatDocs.find((item) => item.relPath === demo.doc);
      const href = demo.href || (doc ? docHash(doc.id) : "#/book");
      const media = demo.mediaUrl || fileHref(demo.media);
      if (demo.type === "video") {
        return `
          <article class="home-demo-card">
            <span class="home-demo-media">
              <video controls playsinline preload="none" poster="${fileHref(demo.poster)}" aria-label="${escapeHtml(demo.title)}">
                <source src="${media}" type="video/mp4" />
                <a href="${media}">下载视频</a>
              </video>
            </span>
            <a class="home-demo-copy" href="${href}">
              <strong>${escapeHtml(demo.title)}</strong>
              <span>${escapeHtml(demo.subtitle)}</span>
            </a>
          </article>
        `;
      }
      return `
        <a class="home-demo-card" href="${href}">
          <span class="home-demo-media">
            <img src="${media}" alt="${escapeHtml(demo.title)}" loading="lazy" />
          </span>
          <span class="home-demo-copy">
            <strong>${escapeHtml(demo.title)}</strong>
            <span>${escapeHtml(demo.subtitle)}</span>
          </span>
        </a>
      `;
    })
    .join("");

  const demoVideos = els.homeDemoGrid.querySelectorAll("video");
  demoVideos.forEach((video) => {
    video.addEventListener("play", () => {
      demoVideos.forEach((other) => {
        if (other !== video) other.pause();
      });
    });
  });

  els.homeModuleList.innerHTML = homeModules
    .map(
      (module) => `
        <section class="home-module" id="${module.id}">
          <div class="home-module-copy">
            <p class="home-kicker">${escapeHtml(module.label)}</p>
            <h2>${escapeHtml(module.title)}</h2>
            <h3>${escapeHtml(module.headline)}</h3>
            <p>${escapeHtml(module.copy)}</p>
          </div>
          <div class="home-module-cards">
            ${module.cards
              .map((card, index) => {
                const doc = state.flatDocs.find((item) => item.relPath === card.doc);
                return `
                  <a class="home-module-card" href="${doc ? docHash(doc.id) : "#/book?catalog=1"}">
                    <span class="home-module-media">
                      <img src="${fileHref(card.image)}" alt="${escapeHtml(card.title)}" loading="lazy" />
                    </span>
                    <span class="home-module-card-body">
                      <small>${String(index + 1).padStart(2, "0")}</small>
                      <strong>${escapeHtml(card.title)}</strong>
                      <span>${escapeHtml(card.copy)}</span>
                      <em>进一步了解 ›</em>
                    </span>
                  </a>
                `;
              })
              .join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

function renderOutline(headings) {
  const useful = headings.filter((heading) => heading.level <= 4).slice(0, 80);
  els.outlineNav.innerHTML =
    useful
      .map(
        (heading) => `
          <a href="#${encodeURIComponent(heading.id)}" class="level-${heading.level}" data-anchor="${escapeHtml(heading.id)}">
            ${escapeHtml(heading.text)}
          </a>
        `,
      )
      .join("") || '<span class="doc-path">本文没有可用标题。</span>';
}

function updatePager(doc) {
  const previous = state.flatDocs.find((item) => item.id === doc.previousId);
  const next = state.flatDocs.find((item) => item.id === doc.nextId);

  els.prevLink.href = previous ? docHash(previous.id) : "#";
  els.prevLink.innerHTML = previous
    ? `<span>上一篇</span><strong>${escapeHtml(previous.title)}</strong>`
    : "";

  els.nextLink.href = next ? docHash(next.id) : "#";
  els.nextLink.innerHTML = next ? `<span>下一篇</span><strong>${escapeHtml(next.title)}</strong>` : "";
}

function scrollToAnchor(anchor) {
  if (!anchor) {
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }

  const target = document.getElementById(anchor);
  if (target) {
    target.scrollIntoView({ block: "start", behavior: "smooth" });
  }
}

function showHome(anchor = "") {
  state.currentDoc = null;
  setMode("home");
  closeSidebar();
  setCatalogOverview(false);
  renderNav();
  document.title = "Every-Embodied 电子书";
  typeHomeTitle();
  requestAnimationFrame(() => {
    if (anchor) {
      document.getElementById(anchor)?.scrollIntoView({ block: "start", behavior: "auto" });
      return;
    }
    window.scrollTo({ top: 0, behavior: "auto" });
  });
}

function showShowcase(anchor = "") {
  state.currentDoc = null;
  stopHomeTitleTyping();
  setMode("showcase");
  closeSidebar();
  setCatalogOverview(false);
  renderNav();
  document.title = "AMD Physical AI Evidence Lab · Every-Embodied";
  requestAnimationFrame(() => {
    if (anchor) {
      document.getElementById(anchor)?.scrollIntoView({ block: "start", behavior: "smooth" });
      return;
    }
    window.scrollTo({ top: 0, behavior: "auto" });
  });
}

async function loadDoc(id, anchor = "", options = {}) {
  if (!id) return;
  stopHomeTitleTyping();
  setHomeTitleText();
  setMode("book");
  els.article.innerHTML = '<div class="loading">正在加载 Markdown 内容。</div>';

  const response = await fetch(`data/docs/${id}.json`, { cache: "no-cache" });
  if (!response.ok) {
    els.article.innerHTML =
      '<div class="error-box">这篇文档不在目录中，可能是英文文档、资源说明或已被移动。</div>';
    return;
  }

  const doc = await response.json();
  state.currentDoc = doc;
  els.breadcrumb.textContent = `${doc.groupTitle} / ${doc.relPath}`;
  els.docTitle.textContent = doc.title;
  els.article.innerHTML = doc.html;
  renderOutline(doc.headings || []);
  updatePager(doc);
  renderNav();
  setCatalogOverview(Boolean(options.openCatalog));

  document.title = `${doc.title} · Every-Embodied 电子书`;
  requestAnimationFrame(() => scrollToAnchor(anchor));
}

async function route() {
  const routeInfo = getRoute();
  if (routeInfo.type === "home") {
    showHome(routeInfo.anchor);
    return;
  }
  if (routeInfo.type === "showcase") {
    showShowcase(routeInfo.anchor);
    return;
  }
  await loadDoc(routeInfo.id, routeInfo.anchor, { openCatalog: routeInfo.openCatalog });
  closeSidebar();
}

function bindEvents() {
  els.openNav.addEventListener("click", openSidebar);
  els.closeNav.addEventListener("click", closeSidebar);
  els.scrim.addEventListener("click", closeSidebar);

  els.searchInput.addEventListener("input", (event) => {
    state.search = event.target.value;
    renderNav();
  });

  els.navTree.addEventListener("click", (event) => {
    const groupButton = event.target.closest(".nav-group-toggle");
    if (groupButton) {
      const key = groupButton.dataset.group;
      if (state.openGroups.has(key)) {
        state.openGroups.delete(key);
      } else {
        state.openGroups.add(key);
      }
      renderNav();
      return;
    }

    const button = event.target.closest(".nav-doc");
    if (!button) return;
    window.location.hash = docHash(button.dataset.id);
  });

  els.toggleCatalog.addEventListener("click", () => {
    const nextOpen = els.catalogOverview.hidden;
    setCatalogOverview(nextOpen);
    if (nextOpen) {
      els.catalogOverview.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  });

  els.refreshCatalog.addEventListener("click", async () => {
    await fetchCatalog();
    await fetchGithubStats();
    await route();
  });

  els.catalogLink.addEventListener("click", (event) => {
    if (document.body.classList.contains("book-mode")) {
      event.preventDefault();
      setCatalogOverview(true);
      els.catalogOverview.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  });

  els.outlineNav.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-anchor]");
    if (!link) return;
    event.preventDefault();
    scrollToAnchor(link.dataset.anchor);
  });

  els.article.addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (!link) return;
    const href = link.getAttribute("href");
    if (!href || !href.startsWith("#/doc/")) return;
    event.preventDefault();
    window.location.hash = href;
  });

  window.addEventListener("hashchange", route);
}

async function init() {
  bindEvents();
  setCatalogOverview(false);
  try {
    await fetchCatalog();
    await fetchGithubStats();
    await route();
    document.body.classList.remove("app-booting");
  } catch (error) {
    document.body.classList.remove("app-booting");
    els.catalogStatus.textContent = "目录读取失败";
    els.article.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

init();
