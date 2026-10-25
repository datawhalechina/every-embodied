# 具身场景的深度和强化学习

本目录收集强化学习、模仿学习和深度学习在具身场景中的复现教程。建议大家先从任务边界和仿真平台开始理解，再进入具体代码和训练命令。

## 章节入口

- [多机器人搬运家具强化学习](01多机器人搬运家具强化学习.md)：CooHOI 协作式人形机器人搬运任务入门。
- [HIMLoco 四足机器人运动控制 Isaac Lab 复现](02HIMLoco-IsaacLab复现/README.md)：从论文理解、旧 Isaac Gym 栈排障到 Isaac Lab 新栈 smoke test。
- [AGILE 人形机器人 Loco-Manipulation Isaac Lab 复现](03AGILE人形机器人Loco-Manipulation复现/README.md)：理解 AGILE 的人形机器人 RL 工作流、官方任务边界、本地 Isaac Sim 5.1 复刻视频和评估导出链路。
- [Robots That Know What to Ask 奖励对齐导读](04-Robots-That-Know-What-to-Ask奖励对齐导读/README.md)：拆解 ASQ 如何识别示教中的欠指定奖励特征，并用自然语言解释引导人类补充 corrective demonstrations。
- [Open Duck Mini 与 Microduck 双足强化学习](05-OpenDuckMini与Microduck双足强化学习/README.md)：区分两代小鸭子机器人，跑通 URDF/STL 资产下载、走路与踢球 PPO 训练、ONNX 导出、RDK 策略推理和 MuJoCo WebAssembly 浏览器部署链路。
- [SMP 可复用分数匹配运动先验](06-SMP可复用分数匹配运动先验/README.md)：拆解扩散运动先验如何通过 SDS、ESM、Adaptive Normalization 和 GSI 变成 PPO 奖励，并给出 MimicKit 原始实现与 Unitree G1 开源复现路线。
- [UniCross 统一跨技能灵巧操作导读](07-UniCross统一跨技能灵巧操作导读/README.md)：从手-物关系运动出发，拆解抓取、搬移、手内旋转和平移的统一观测、动作和奖励，以及十个 PPO 专家如何通过 DAgger 蒸馏成单一跨技能策略。
- [HIL-SERL 人机协同强化学习与 MuJoCo 插孔](08-HIL-SERL人机协同强化学习与MuJoCo插孔/README.md)：配图理解干预动作、双回放缓冲区和 actor/learner，并通过官方 Panda 仿真学习采集、训练与插孔任务移植流程。

## 学习建议

如果大家刚接触机器人强化学习，建议先看 HIMLoco 章节，理解 proprioception、privileged critic 和 sim-to-real 的基本思路；然后通过 Open Duck Mini/Microduck 做一次体量更小、资产和命令更清晰的双足训练练习；接着看 SMP，理解如何把动作扩散模型冻结成可复用的运动先验奖励；再看 UniCross，理解统一任务表述为什么能让多个接触技能共享一个策略；最后看 AGILE 章节，学习更完整的人形机器人任务配置、teacher-student 蒸馏、评估报告和 Sim2MuJoCo 复核流程。对 VLA、世界模型后训练或人类示教数据感兴趣的同学，可以继续看 Robots That Know What to Ask，重点理解 reward misalignment、欠指定特征和 human-in-the-loop 奖励修正。
