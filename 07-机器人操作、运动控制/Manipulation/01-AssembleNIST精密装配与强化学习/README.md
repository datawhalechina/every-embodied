# AssembleNIST：用强化学习完成精密装配

AssembleNIST 在 MuJoCo 中使用一套 PPO 策略控制 Franka 机械臂，学习 NIST Assembly Task Board 1 上的 20 个单零件装配任务，包括插销、圆杆、齿轮、螺母和连接器。本章介绍任务、训练机制与上游提供的策略运行方法，并说明如何理解作者发布的评测结果。

**项目与方法来源：**[AssembleNIST 开源仓库](https://github.com/suniique/assembly-nist)（Tao Sun）是对 [AlphaNIST](https://alpha-nist.github.io/) 和 [Success Guided Sampling（SGS）](https://sgs-rl.github.io/) 的独立实现，采用 MuJoCo 及作者自行设计的接触模型、动作空间和初始状态。本文根据上游提交 [`eb5d15e`](https://github.com/suniique/assembly-nist/tree/eb5d15e004591d380c0d521da67e025ebee37703) 整理；下列结果均来自项目作者，环境搭建、推理和训练命令本章尚未实测。

## 1. 先看装配演示

https://github.com/user-attachments/assets/81129b64-3911-485c-b9f9-cc1282c2d831

*视频 1：AssembleNIST 作者发布的 MuJoCo 多零件装配展示。来源：[项目主页](https://github.com/suniique/assembly-nist)。*

展示视频中，策略逐个处理零件，切换零件时由脚本执行抬手和移动。根据上游 [`demo.py`](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/assembly_policy/viz/demo.py)，每个阶段模拟当前零件和相关夹具，其余零件主要作为渲染对象；展示使用的每零件默认时间预算为 30 秒。阅读视频时应把它与每回合一个零件、15 秒时限的标准评测区分开。

## 2. 任务与学习流程

```text
当前零件、夹具、夹爪点云 + 本体状态历史
                    ↓
          SimBa 风格策略网络（PPO）
                    ↓
        关节相对位置目标 + 夹爪动作（25 Hz）
                    ↓
        MuJoCo 接触仿真与任务成功判断
                    ↓
    成功反馈 + 控制惩罚 → 策略更新与初始状态采样
```

**图 1：**从观测到动作、仿真反馈及策略更新的关系，依据上游 README 整理。

这个任务的难点是接触丰富且公差小：齿轮要与轴对准，螺母必须旋入螺纹，连接器需要匹配插座。上游提供不同的初始状态，包括零件散放、抓取、运输、部分装配和接近配合。成功引导采样根据不同初始状态上的表现调整采样权重，让策略持续练习当前有学习价值的状态。

策略接收零件、夹具和夹爪点云以及本体状态历史；这套仿真观测不能直接等同于真实相机部署。作者使用二值装配成功信号并加入小幅控制惩罚，成功判定使用 1 毫米关键点容差。训练中零件类型对应不同接触建模：圆杆和插销采用基础几何，齿轮使用凸形齿面，螺纹及连接器采用 SDF 接触模型。上游报告齿轮和螺母使用 0.5 ms 仿真步长，其他零件使用 2.5 ms。

## 3. 环境与预训练策略运行

上游构建脚本针对 **Linux x86-64**，使用 `g++`、MuJoCo Python 包中的头文件和 `libmujoco.so` 编译 SDF 插件，并启用 `-mf16c`。Windows 可通过满足这些条件的 Linux 或 WSL 环境尝试；图形后端还需要可用的 EGL，或按上游说明改用 OSMesa。项目声明 Python ≥ 3.10、MuJoCo ≥ 3.3 和 PyTorch ≥ 2.1。

下面的命令在终端执行。创建并进入自己的工作目录后，克隆上游并切换到本章核对过的提交，再创建 Python 虚拟环境：

```bash
git clone https://github.com/suniique/assembly-nist.git
cd assembly-nist
git checkout eb5d15e004591d380c0d521da67e025ebee37703
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash plugin/build.sh
```

**检查点 1：**构建成功时，脚本会输出 `built plugin/build/libassemblenist_sdf.so`。若提示找不到 `libmujoco`，先确认虚拟环境已激活且 `python -c "import mujoco; print(mujoco.__file__)"` 可以运行；若编译器不支持 `-mf16c`，检查运行平台与 CPU 指令集。

在同一项目目录和环境中，运行上游的预训练演示：

```bash
MUJOCO_GL=egl python -m assembly_policy.viz.demo \
  --ckpt checkpoints/e8192_it12500.pt \
  --out demo.mp4
```

**检查点 2：**预期在当前目录生成 `demo.mp4`，终端逐项输出零件装配结果。若缺少初始状态库，演示会按上游说明生成稳定初始姿态；时间取决于机器。`MUJOCO_GL=egl` 初始化失败时，可在已安装 OSMesa 的环境中尝试将其改为 `MUJOCO_GL=osmesa`。如果权重路径不存在，先检查上游仓库的 `checkpoints/` 是否完整。

生成视频可以检查模型加载、场景渲染和示例执行；模型成功率应使用独立评测流程计算。

## 4. 评测与训练

上游要求先构建初始状态库。以下命令来自上游，工作目录仍为 `assembly-nist`；请按机器的 CPU 资源调整 worker 数量：

```bash
python scripts/build_banks.py --workers 16
python -m assembly_policy.evaluate \
  --policy checkpoints/e8192_it12500.pt \
  --out eval.json
```

评测输出为 `eval.json`。上游称默认评测运行 3,200 个回合，在 16 个 CPU 核上约需 1 小时；实际耗时取决于机器。作者说明其统计采用每任务 5 个随机种子、每个种子 32 次试验，并对策略动作采样；需要在 15 秒内完成装配且没有安全违规，才算成功。

先验证较小规模的训练配置，可以参考上游的单机命令：

```bash
python -m assembly_policy.train --run-dir runs/small --envs 512 --nthread 32
```

训练日志和检查点写入 `runs/small`。512 个并行环境仍会消耗较多 CPU 和内存，数字是上游示例，不是普通笔记本已验证配置。完整参考训练使用 8,192 个环境；作者报告使用约 30 台节点、960 个 CPU worker 和一个 GPU learner，10,000 次迭代约 26 小时。分布式启动及共享目录要求见[上游训练说明](https://github.com/suniique/assembly-nist/tree/eb5d15e004591d380c0d521da67e025ebee37703#training-and-evaluation)。

## 5. 如何看待结果与接触设计

上游报告：12,500 次迭代检查点在其评测中整体成功率为 **98.7%**，10,000 次迭代为 **98.4%**。上游同时列出了 AlphaNIST 的数值，但也明确说明两者使用不同的仿真器、资产和实现，不能据此作为严格对照实验。

作者记录了几个值得研究的工程细节：调整策略探索强度、按关节调整动作尺度以满足时间限制，以及修改螺纹接触间隙和仿真步长以避免卡死。这些设置相互影响；比较算法时应记录动作频率、奖励、初始状态分布、几何公差和物理参数，再分析失败发生在抓取、对准还是装配阶段。

## 6. 资料与许可

- [AssembleNIST 源码、评测与检查点](https://github.com/suniique/assembly-nist/tree/eb5d15e004591d380c0d521da67e025ebee37703)
- [AlphaNIST 项目与论文](https://alpha-nist.github.io/)；[SGS 项目与论文](https://sgs-rl.github.io/)
- [上游第三方资产及许可说明](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/THIRD_PARTY.md)：项目代码采用 MIT 许可，但上游标注 NIST 零件与夹具资产的再分发许可仍待确认。本教程链接上游资源，不复制这些资产。
- 关联阅读：[HIL-SERL 人机协同强化学习与 MuJoCo 插孔](../../../05-具身场景的深度和强化学习/08-HIL-SERL人机协同强化学习与MuJoCo插孔/README.md)。
