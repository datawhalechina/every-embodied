<div align="center">

# RDK Challenge 参赛分享

## TuntunClaw：从家庭库存记忆到真实机器人操作

<img src="./assets/rdk-challenge-poster.png" width="100%" alt="Robotics Dream Keeper Challenge 官方海报">

<p><strong>Power on. Build. Launch your intelligent robot on RDK X5.</strong></p>

<p>
  <a href="https://github.com/D-Robotics/Robotics-Dream-Keeper-Challenge"><img src="https://img.shields.io/badge/Official-RDK_Challenge-ff5a1f?style=for-the-badge&logo=github&logoColor=white" alt="RDK Challenge 官方仓库"></a>
  <a href="https://github.com/Ethan-Chen-plus/rdk-x5-smart-inventory-robot"><img src="https://img.shields.io/badge/Project-TuntunClaw-1f6feb?style=for-the-badge&logo=github&logoColor=white" alt="TuntunClaw 项目仓库"></a>
  <a href="https://youtu.be/mVvQPtZMKm4"><img src="https://img.shields.io/badge/Watch-Final_Demo-ff0000?style=for-the-badge&logo=youtube&logoColor=white" alt="TuntunClaw 演示视频"></a>
</p>

</div>

## 关于 RDK Challenge

[Robotics Dream Keeper Challenge](https://github.com/D-Robotics/Robotics-Dream-Keeper-Challenge) 是 D-Robotics 围绕 RDK X5 发起的全球机器人实践活动。参赛者通过 **Ignite、Build、Launch** 三个阶段，从开发板启动和板端 AI Demo 出发，逐步完成机器人系统设计、软硬件集成与真实场景展示。

我们以 **TuntunClaw 家庭物资助手** 参加本次挑战，尝试把具身智能仿真、边缘感知、真实机械臂操作、库存记忆和语音提醒串联起来，让机器人真正参与家庭物资管理。

## 从仿真走向真实世界

| 阶段 | 实现内容 | 展示结果 |
| --- | --- | --- |
| 仿真验证 | OpenClaw 自然语言任务调度、VLM + SAM 目标理解、GraspNet 抓取位姿推理 | MuJoCo 中连续抓取、移动和放置，场景状态与库存记忆持续保留 |
| RDK X5 感知 | Magic Box 相机、麦克风、扬声器与 BPU 推理 | 完成板端实时感知、语音交互和设备能力验证 |
| 真实机器人 | 机械臂取放、库存更新、阈值判断 | 物品送达后更新库存，并在达到阈值时触发补货提醒 |
| 人机交互 | 平板库存界面与 Magic Box 语音播报 | 用户可查看实时库存，系统主动提示低库存物品 |

## 参赛感受

这次挑战让我们把仿真中的具身智能流程真正延伸到边缘设备和真实机器人。RDK X5 带来了实用而有趣的开发体验，也让感知、交互和机器人执行之间的连接更容易落地。感谢 D-Robotics 团队和全球开发者社区提供这次交流、学习与实践机会。

## 开源资料

- [TuntunClaw 完整项目仓库](https://github.com/Ethan-Chen-plus/rdk-x5-smart-inventory-robot)
- [TuntunClaw 英文演示视频](https://youtu.be/mVvQPtZMKm4)
- [Every Embodied：OpenClaw 家庭物资助手教程](../../16-专题组队学习/02-OpenClaw家庭物资助手/README.md)
- [RDK Challenge 官方规则与优秀项目](https://github.com/D-Robotics/Robotics-Dream-Keeper-Challenge)
- [D-Robotics RDK X5 开发者文档](https://d-robotics.github.io/rdk_doc/)

欢迎关注 RDK X5、RDK Challenge 和更多具身智能开源实践。

`#RDKChallenge` `#RDKX5` `#DRobotics` `#EmbodiedAI` `#Robotics`

> 海报来源：D-Robotics Robotics Dream Keeper Challenge 官方开源仓库。
