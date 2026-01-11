# 任务定义：

![](images/image-12.png)

​	视觉语言导航( Visual Language Navigation，VLN )是具身AI的一个关键研究问题，旨在使智能体能够按照语言指令在不可见的环境中导航。VLN要求机器人能够理解复杂多样的视觉观察，同时以不同的粒度解释指令。

​	VLN的输入通常由两部分组成：视觉信息和自然语言指令。视觉信息既可以是过去轨迹的视频，也可以是历史-当前观测图像的集合。自然语言指令包括具身智能体需要达到的目标或者具身智能体期望完成的任务。具身agent必须利用上述信息从候选动作列表中选择一个或一系列动作来完成自然语言指令的要求。

## 任务设定

  视觉语言导航 (VLN) 任务在数学形式上均可被统一建模为一个部分可观测马尔可夫决策过程 (POMDP)，由元组 $\langle S, A, T, O, R, J \rangle$ 描述。其中，$\mathcal{S}$ 代表状态空间，$\mathcal{A}$ 为智能体的动作空间，$\mathcal{T}$ 为状态转移函数，$\mathcal{O}$ 为视觉观测空间，$\mathcal{R}$ 为奖励函数，$\mathcal{J}$ 为自然语言指令。

  在这一框架下，导航问题的核心在于学习一个策略映射函数 $a_t = \pi(o_t, I, s_t)$ （通常参数化为深度神经网络模型 $F$）。即在时间步 $t$，智能体根据当前的第一人称视觉观测 $o_t$、全局自然语言指令 $I$ 以及智能体状态 $s_t$，推导出最优动作 $a_t$，以最大化累积奖励并逼近目标位置。

  根据状态空间、动作空间及状态转移函数上的不同，VLN 任务被划分为离散 VLN 与连续 VLN。

### 离散环境视觉语言导航（VLN）任务设定

![](images/image-13.png)

​									**图 1 离散环境 VLN。蓝色连线图表示预定义的连通图；绿色线条表示专家轨迹。**

离散VLN 研究主要依托 Matterport3D 模拟器展开。如图1所示，环境被抽象为一个预定义的连通图 $\mathcal{G}=\langle V,\mathcal{E}\rangle$ ，图中的蓝色节点图所示，其中$V$代表一组固定的全景视点，$\mathcal{E}$表示视点间的连通关系。

**状态空间 (**$\boldsymbol{S}_{\boldsymbol{disc}}$**)：**&#x667A;能体的状态$S_t$被严格限制为图$g$中的某个节点，即$s_t\in V$。

**动作空间 (**$\mathcal{A}_{disc}$**)：**&#x5047;定*t*时刻智能体的状态$s_t$在图上$g$的所有邻域连通节点表示为$N(s_t)$，$N(s_t)$中的每个节点称为候选节点，$N(s_t)$是智能体在*t*时刻的动作空间。智能体第*t*时刻的动作决策表现为“选择候选节点”，策略函数$F$输出一个关于$N(s_t)$的概率分布，用于从候选节点中选择一个节点做为下一个“跳跃点”。

**状态转移 (**$\mathcal{T}_{disc}$**)：** 状态转移表现为确定性的“隐形传送”机制。一旦下一个“跳跃点”生成，系统会将智能体位置瞬间切换至选中的候选节点坐标。该过程忽略中间路径，且不考虑物理碰撞或摩擦，这种“Oracle Navigation”设定虽然简化了控制难度，但严重脱离了真实机器人在现实环境的运动规律。

### 连续环境视觉语言导航（VLNCE）任务设定

![](images/image-14.png)

​                                                                                                    **图 2 连续环境 VLN。绿色线条表示专家轨迹。**

​	为了弥合仿真环境与现实环境的差距，VLN-CE在Habitat模拟器中进行训练。如图2所示，该环境加载了来自 Matterport3D 数据集的高度逼真室内 3D 扫描场景，能够提供与真实世界一致的第一人称视觉观测。与此同时，Habitat 引入了导航网格来定义物理运动空间。该网格将环境的可行走区域构建为一系列紧密连通的多边形表面，允许智能体驻留在网格范围内的任意连续坐标点上。正是这种空间上的连续性，使得智能体能够在物理引擎的支持下执行“前进固定距离”与“旋转固定角度”等低级动作。当智能体执行一步前进指令（如 0.25米）时，导航网格确保了这不仅是坐标数值的跳变，而是一段在连续几何表面上真实的滑动位移过程，从而像在真实世界一样处理连续的位姿变化。与离散VLN中智能体从一个节点“跳跃式”移动到与之连通的节点相比，连续VLN中智能体基于导航网格的平滑运动更符合真实环境中的移动规律。

​	**状态空间 (**$\boldsymbol{S}_{\boldsymbol{disc}}$**)：**&#x667A;能体的状态由世界坐标系中的精确三维坐标${P}=(x,y,z)$和姿态朝向$\mathrm{Φ~=(\theta,\psi)}$表示，其中分别表示俯仰角和航向角。

​	**动作空间 (**$\mathcal{A}_{disc}$**)：**&#x667A;能体在任意时刻的动作空间由一组低级动作原语组成，如图2中“Low-Level Actions”标签所示，包括固定距离的前进（如前进0.25米），左转固定角度的旋转（如左转15°）, 右转固定角度的旋转（如右转15 °）和停止（stop）。智能体第*t*时刻的动作决策表现为“选择一个动作原语”，策略函数输出一个关于所有动作原语的概率分布，用于从中选择一个做为具体的运动控制指令。

​	**状态转移 (**$\mathcal{T}_{disc}$**)：** 状态转移$\mathcal{T}(s_t,a_t)$表现为确定性的“隐形传送”机制。受到仿真器物理引擎的动力学规则的严格约束。当智能体执行动作与环境物体发生接触时，会触发 “滑动（Sliding）”或阻滞机制。

## VLNCE任务数据集

### 常用数据集：

#### R2RCE

##### 模拟器

* 仿真环境：**Habitat-Sim v0.1.7**&#x20;

* 环境场景：基于 **Matterport3D (MP3D)** 数据集的 90 个真实室内场景（房屋、办公室等）。

##### 输入：

**每一时刻的观测空间 (**$O_t$**)：**

* **视觉输入：** **RGB-D 图像**（RGB + 深度图）。

* **分辨率：** 通常RGB图片分辨率为（224×224）， Depth图片分辨率为（256×256）。

* **视场角 (FOV)：** **90度**。

* **注意：** 这里默认是单目视角。许多模型如ETPNav会通过从左到右旋转12次(每30°一个间隔)来构建全景特征。

##### **文本指令 (Text Instruction)：**

* **指令语言：仅英语**。

* **指令类型： 分步导航（Step-by-Step）**。指令并不仅仅告诉智能体“去哪里（目的地）”，而是提供了一个**动作序列**的描述。它会告诉智能体如何从当前位置一步步导航到终点。

* **实例：**"Exit the room through the doorway nearest you, and continue into the adjacent room, exiting the room via the exit to your left." (先出门，进入隔壁房间，再从左边出口离开。)

##### 输出：

**动作空间 (Action Space)：**

* 智能体输出**离散的低级控制指令**：

  1. `MoveForward` (向前移动 **0.25米**)

  2. `TurnLeft` (向左旋转 **15度**)

  3. `TurnRight` (向右旋转 **15度**)

  4. `Stop` (停止，判断是否到达终点)

* **判定标准：** 智能体执行 `Stop` 动作时，距离目标点 **3.0米** 以内即视为成功。

##### 数据集规模：

* **路径数量：** 约 **4,475 条** 连续轨迹（从离散 R2R 路径重建而来）。

* **指令数量：** 约 **13,425 条**（平均每条路径 3 条指令）。

* **平均路径长度：** 约 **9.89 米**。

* **划分：** 训练集 (10.8k)、验证集 (Seen: 778 / Unseen: 1.8k)、测试集 (3.4k)。

#### RXRCE

##### **模拟器**

* **仿真环境：** **Habitat-Sim v0.1.7**&#x20;

* **环境场景：** 同样基于 **Matterport3D (MP3D)** 数据集的真实室内场景。但相比 R2R-CE，RxR-CE 在这些场景中采样的路径覆盖率更高，探索了更多复杂的区域和回环。

##### 输入

**每一时刻的观测空间 (**$O_t$**)：**

* **视觉输入：** RGB-D 图像（RGB + 深度图）。

* **分辨率：** 通常 RGB 图片分辨率为（224×224），Depth 图片分辨率为（256×256）。

* **视场角 (FOV)：** 79°。

* **注意：** 同样默认为单目视角。



**文本指令 (Text Instruction)：**

* **指令语言：** **多语言 (Multilingual)**。包含 **英语 (English)、印地语 (Hindi)、泰卢固语 (Telugu)**。

* **指令类型：** **过程导向 (Process-Oriented)**。指令非常详尽，不仅描述动作序列，还像“导游”一样描述沿途的所有视觉细节。最重要的是，指令中的词语与智能体在路径上的位姿（Pose）是时间对齐的。

* **示例：** "Standing on the rug, turn right and face the wooden cabinet. Walk past the cabinet, keeping it on your left, and enter the hallway. You will see a painting of a flower on the wall. Stop just after passing the painting." (站在地毯上，右转面向木柜。走过柜子，保持它在你的左手边，然后进入走廊。你会看到墙上有一幅花的画。刚走过画就停下。)



**输出**

**动作空间 (Action Space)：**

* 智能体输出离散的低级控制指令（与 R2R-CE 一致）：

  1. `MoveForward` (向前移动 **0.25米**)

  2. `TurnLeft` (向左旋转 **15度**)

  3. `TurnRight` (向右旋转 **15度**)

  4. `Stop` (停止，判断是否到达终点)

* **判定标准：**

  * **成功标准：** 智能体执行 Stop 动作时，距离目标点 **3.0米** 以内即视为成功 (Success)。

  * **路径一致性 (NDTW)：** 由于 RxR 的路径通常不是最短路，单纯到达终点不够，该任务更看重智能体是否**忠实地**按照指令描述的路线行走（即预测路径与真实路径的几何重合度）。



**数据集规模**

* **路径数量：** 约 **42,000+ 条** 连续轨迹（规模比 R2R 大一个数量级）。

* **指令数量：** 约 **126,069 条**（包含三种语言的总和）。

* **平均路径长度：** 约 **15 米**（比 R2R 更长，且多为弯曲的非最短路径）。

* **划分：** 训练集 (Train)、验证集 (Val-Seen / Val-Unseen)、测试集 (Test)。

#### 数据集获取：https://github.com/jacobkrantz/VLN-CE



### 评价指标:

(1)**导航误差 (Navigation Error, NE)**

​	定义为智能体最终停止位置与目标位置之间的平均测地距离（米）。

$NE=\frac{1}{N}\sum_{i=1}^Ndist(p_{final}^{(i)},p_{target}^{(i)})$

(2)**成功率 (Success Rate, SR):**

​	定义为智能体停止位置与目标位置之间的测地距离小于特定阈值（标准设定为3m）的比例。

$SR=\frac{1}{N}\sum_{i=1}^N\mathbb{I}(\mathrm{dlist}(p_{final}^{(i)},p_{target}^{(i)})<3m)$

(3)**路径长度加权成功率 (Success weighted by Path Length, SPL)**

​	SPL 是 VLN-CE 的核心指标，它旨在平衡“成功率”与“路径长度”。

$SPL=\frac{1}{N}\sum_{i=1}^NS_i\frac{l_i}{\max{(l_i,p_i)}}$

(4)**Oracle 成功率 (Oracle Success Rate, OSR)**

​	智能体的轨迹中任意一点与目标的测地距离小于阈值，视为成功。OSR 衡量了智能体“经过”目标的潜力。OSR 与 SR 之间的差距通常反映了智能体的停止策略（Stop Policy）缺陷，即智能体虽然找到了目标，但未能正确触发停止动作。

$OSR=\frac{1}{N}\sum_{i=1}^N\mathbb{I}(\min_tdist(p_t^{(i)},p_{target}^{(i)})<3m)$

(5)**归一化动态时间规整 (Normalized Dynamic Time Warping, nDTW)**

​	nDTW 利用动态规划算法计算预测轨迹P与参考轨迹Q（即人工演示轨迹）之间的最小累积距离，并进行指数归一化处理。该指标惩罚偏离指令描述的捷径（Shortcuts）或绕路行为，要求智能体在空间几何上严格遵循指令描述的路线。

$nDTW=\exp{(-\frac{DTW(P,Q)}{\sqrt{|P|\cdot|Q|}})}$

（6）**SDTW (Success weighted by normalized Dynamic Time Warping)**

​	SDTW 统计成功到达终点的任务的nDTW指标，是衡量“指令依从性”与“任务完成度”的综合指标。

$SDTW=\frac{1}{N}\sum_{i=1}^NS_i\cdot\mathrm{nDTW}_i$



# 连续环境视觉语言导航方法：

## **传统单阶段方法**

### 基础端到端基准模型（2020ECCV）

**论文题目：**Beyond the Nav-Graph: Vision-and-Language  Navigation in Continuous Environments

**作者:**  Jacob Krantz1, Erik Wijmans2,3, Arjun Majumdar2, Dhruv Batra2,3, and Stefan Lee1

**单位:**  Oregon State University、Georgia Institute of Technology、Facebook AI Research

**论文地址：**https://www.ecva.net/papers/eccv\_2020/papers\_ECCV/papers/123730103.pdf

**项目地址：**https://jacobkrantz.github.io/vlnce/

​	主要包括序列到序列（Sequence-to-Sequence, Seq2Seq）模型和交模态注意力（Cross-Modal Attention, CMA）模型 ，采用端到端架构，模型直接输出低级动作的概率。

![](images/image-6.png)

​	Seq2Seq 架构如图(a)所示。该模型由语言编码器和动作解码器组成。语言编码器（LSTM）将导航指令压缩为一个固定长度的语义上下文向量；动作解码器则基于该语义向量和当前时间步的第一人称视觉特征（由 ResNet 等提取），通过循环神经网络（RNN）预测动作空间 {MOVE\_FORWARD, TURN\_LEFT, TURN\_RIGHT, STOP} 上的概率分布。在 Seq2Seq 的架构基础上，CMA 模型（如图(b)所示）旨在解决长指令中的信息遗忘与对齐问题。不同于 Seq2Seq 仅利用指令的最终状态向量，CMA 进一步引入了注意力机制。它利用当前时刻的视觉特征作为查询（Query），与双向 GRU 输出的完整语言指令特征序列进行交叉注意力计算，从而动态聚焦于与当前视觉场景最相关的指令片段（例如，看到“沙发”时聚焦于指令中的“走到沙发旁”），实现了视觉与语言的细粒度对齐。

## 传统两阶段导航方法

采用分层结构，将导航分解为“决策”和“执行”两个阶段。**第一阶段（决策）**&#x6839;据全景图和指令，预测下一个高级目标点（Local Goal / Waypoint），通常是 $r, \theta$ 坐标。**第二阶段（执行）**：**底层控制器**接收目标点坐标，通过算法（如 PID、A\*、Fast Marching Method 或 DWB）自动规划路径并控制智能体到达该点。

### （1）航点预测器的提出，两阶段导航的开端：

论文题目：Bridging the Gap Between Learning in Discrete and Continuous  Environments for Vision-and-Language Navigation（2022CVPR）

作者：Yicong Hong Zun Wang Qi Wu Stephen Gould

单位：1The Australian National University, 2University of Adelaide

论文地址：https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Bridging_the_Gap_Between_Learning_in_Discrete_and_Continuous_Environments_CVPR_2022_paper.pdf

项目地址：https://github.com/YicongHong/Discrete-Continuous-VLN

方法：

![](images/image-7.png)

这篇论文（Hong et al., CVPR 2022）**旨在填补“离散”与“连续”视觉语言导航（VLN）环境之间的巨大鸿沟**。主要有以下创新点：

* &#x20;提出了候选航点预测器 (Candidate Waypoint Predictor)，在连续环境（Habitat）中基于深度图实时生成可通行的“虚拟节点”，让连续空间看起来像是一个离散的导航图。

* 构建了一个“分层导航”系统： 将导航拆解为两步——先由高层规划器选目标点，再由底层控制器执行低级动作。

* 将离散环境中的预训练范式迁移到连续环境中： 证明了只需将在离散环境（Nav-Graph）中训练好的模型拿过来，稍作微调，就能在连续环境（R2R-CE）中取得当时最先进的效果（SOTA），极大地降低了训练成本。

![](images/image-8.png)

### （2）基于地图的方法

#### 拓扑地图

论文题目：ETPNav: Evolving Topological Planning for  Vision-Language Navigation in  Continuous Environments(2024TPAMI)

作者：Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang

单位：模式识别国家重点实验室（NLPR）智能感知与计算研究中心（CRIPAC）；中国科学院大学未来技术学院及人工智能学院

论文地址：https://arxiv.org/abs/2304.03047

项目地址：https://github.com/MarSaKi/ETPNav

![ETP导航框架](images/image-9.png)

![ETPNav生成第t时刻拓扑图的过程](images/image-10.png)

​	早期的VLN-CE方法主要使用循环神经网络(RNN)，该类方法的不足之处在于它们将历史信息表示为一个固定大小的状态向量，而这种隐式记忆机制不足以存储和利用具有丰富时空结构的历史经验。

​	ETPNav 是一种针对连续环境视觉语言导航（VLN-CE）的分层导航框架，其核心在于通过在线演化拓扑建图（Online Evolving Topological Mapping），将未知的连续物理空间实时抽象为一张动态更新的拓扑图，通过拓扑地图来存储历史信息，把复杂的连续控制问题转化为高效的图上决策问题。在这一架构下，跨模态规划器负责基于拓扑图和语言指令进行长程语义推理以选定最佳导航航点，而试错法控制器（Trial-and-Error Controller）则负责底层的动作执行与鲁棒避障，两者结合有效解决了传统端到端模型在长距离导航中容易语义迷失和物理碰撞的难题。

#### BEV地图：

**论文题目：**BEVBert: Multimodal Map Pre-training for Language-guided Navigation（2023ICCV）

**作者：**Dong An Yuankai Qi Yangguang Li Yan Huang Liang Wangieniu Tan Jing Shao

**单位：**1中国科学院自动化研究所 2中国科学院大学未来技术学院 3阿德莱德大学澳大利亚机器学习研究院 4商汤科技研究院 5南京大学 6上海人工智能实验室

**论文地址：**https://arxiv.org/pdf/2212.04385

**项目地址：**https://github.com/MarSaKi/VLN-BEVBert

![](images/image-11.png)

​	BEVBert 提出了一种空间感知的多模态地图预训练范式，旨在解决传统全景图输入导致的空间感知缺失与观测冗余问题。该方法核心在于构建混合地图（Hybrid Map）机制，即结合用于消除视觉重叠、明确短程几何结构的局部鸟瞰图（BEV）度量地图，以及用于记忆长程路径的全局拓扑地图。在此基础上，通过专门设计的空间感知预训练任务（如掩码地图建模），模型学会了直接在地图表征上进行跨模态推理，从而大幅提升了智能体在复杂导航任务中的空间理解力与规划能力。

### （3）基于未来想象的方法

#### HNR

**论文题目：**Lookahead Exploration with Neural Radiance Representation for  Continuous Vision-Language Navigation(2024CVPR)

**作者：**Zihan Wang Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu Ming Jiang Shuqiang Jiang

**单位：**中国科学院计算技术研究所，中国科学院大学

**论文地址：**https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf

**项目地址：**https://github.com/MrZihan/HNR-VLN

![](images/image-15.png)

![](images/image-16.png)

​	针对连续环境（VLN-CE）中智能体因缺乏预定义导航图而易陷入视觉盲区，以及现有方法（如 ETPNav）在多路口决策时仅依赖局部可视特征的局限性，本文提出了**分层神经辐射表示（HNR）框架。该方法引入前瞻性探索策略（Lookahead Exploration）**，利用 CLIP 语义嵌入而非传统像素重建来合成环境特征，有效应对视觉遮挡并提升鲁棒性；同时，通过构建延伸至当前视界之外的**未来路径树**，智能体能够并行评估多个潜在分支的语义匹配度，从而在做决策时超越局部最优，实现兼顾长远规划的全局最优路径选择。

#### NavMorph(世界模型)

**论文题目：**NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation  in Continuous Environments(2025ICCV)

**作者：**Xuan Yao Junyu Gao and Changsheng Xu

**单位：**中国科学院自动化研究所 (CASIA) 多模态人工智能系统国家重点实验室 (MAIS) 中国科学院大学 (UCAS) 人工智能学院  鹏城实验室，中国深圳

**论文地址：**https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.pdf

项目地址：https://github.com/Feliciaxyao/NavMorph

![](images/image-17.png)

![](images/image-18.png)

关键问题：

* **泛化与适应能力不足：** 现有的 VLN 方法在面对新奇环境或导航过程中的动态变化时，往往难以快速适应

* **现有世界模型的局限性：**  之前的导航世界模型多依赖离散的状态动力学，难以捕捉 VLN-CE 任务中连续的时空动力学特征。

* **在线适应缺失：** 现有的模型通常是静态预训练的，缺乏在部署阶段针对新环境进行持续更新和在线适应的机制。

* **计算开销高：** 许多模型采用像素级的未来图像预测，导致巨大的计算量和训练复杂性

创新:

**提出了 NavMorph 自演化世界模型框架：** 这是一个专门为 VLN-CE 设计的自演化世界模型，通过在线交互构建一个不断进化的潜空间（Latent Space)。

**定制化的循环状态空间模型 (RSSM)：** 引入了能够显式建模“动作-状态”转换的潜动力学学习机制，从而更有效地捕获环境的连续导航特征。

**上下文演化记忆 (Contextual Evolution Memory, CEM)：**

* **积累洞察：** 该机制允许智能体积累导航见解和场景上下文信息 。

* **高效更新：** 不同于依赖梯度下降的传统方法，CEM 采用前向迭代更新机制，能以极低的计算成本在在线测试阶段快速学习和适应新场景。

**分层任务协作架构：**

* **世界感知导航器 (World-aware Navigator)：** 负责从历史上下文和当前观察中推断环境动力学，构建稳健的潜状态表示。

* **前瞻动作规划器 (Foresight Action Planner)：** 通过在潜空间（特征层级）进行“预想”，模拟未来可能发生的视觉嵌入和动作序列，从而提供更具预见性的规划决策。

**特征级预测与对齐：** 放弃了昂贵的像素级图像生成，改为在特征层级进行未来预测和重构，显著提升了效率并强化了语义对齐能力。

### （4）基于Sim2real提出的方法：

#### Sim2Real-VLN-3DFF

**论文题目：**Sim-to-Real Transfer via 3D Feature Fields for Vision-and-Language Navigation（2024CORL）

**作者：**Zihan Wang Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu Ming Jiang Shuqiang Jiang

**单位：**1中国科学院计算技术研究所，北京 2中国科学院大学，北京 3中国科学院智能计算技术研究所，苏州 4鹏程实验室，深圳

**论文地址：**https://proceedings.mlr.press/v155/anderson21a/anderson21a.pdf

**项目地址：**https://github.com/MrZihan/Sim2Real-VLN-3DFF

方法：

![](images/image-19.png)

![](images/image-20.png)

关键问题：

**仿真与现实的差距 (Sim-to-Real Gap)：** VLN 智能体通常在仿真器中训练，但在现实部署时，物理环境的复杂性、传感器的噪声以及未见过的场景会导致性能大幅下降。

**全景观察与单目硬件的矛盾：** 目前主流的高性能 VLN 模型大多依赖于全景（Panoramic）视觉输入，而大多数廉价、常见的移动机器人仅配备单目（Monocular）摄像头，其视野（FOV）有限，导致难以直接部署高性能模型。

**可导航路径预测的难度：** 在现实连续环境中，机器人需要自主识别障碍物并预测可通行的航路点（Waypoints），单目视角下的遮挡问题使得这一任务非常困难。



创新:

**引入 3D 特征场 (3D Feature Fields)：**

* 利用神经辐射场（NeRF）的思想，将单目序列观察到的视觉特征映射到 3D 空间，构建 3D 特征场 。

* 通过体积渲染（Volume Rendering）技术，智能体可以从当前有限的单目视角“渲染”出周围未见区域的虚拟全景表示，从而弥补单目相机的视野缺陷。

**语义可通行地图 (Semantic Traversable Map)：**

* 提出了一种代理中心（Agent-centric）的语义可通行地图，用于同时进行环境的语义对齐和可通行性感知。该地图能够预测周围 360 度的导航航路点，并识别障碍物，实现了高效的路径规划和避障。



#### monoVLN

**论文题目：**monoVLN: Bridging the Observation Gap between Monocular and Panoramic  Vision and Language Navigation（2025ICCV）

**作者：**Renjie Lu, Yu Zhou, Hao Cheng, Jingke Meng, Wei-Shi Zheng

**单位：**1中山大学计算机科学与工程学院、2湖南大学、3彭城实验室、4机器智能与先进计算教育部重点实验室

**论文地址：**https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_monoVLN_Bridging_the_Observation_Gap_between_Monocular_and_Panoramic_Vision_ICCV_2025_paper.pdf

方法:

![](images/image-21.png)

**关键问题：**

* **配置差异 (Configuration Disparity)：** 大多数高性能 VLN 模型假设智能体拥有 360° 全景视野，但实际机器人通常只配备单目 RGB-D 摄像头，视野极小（约 79°\~90°）。

* **信息不完整性 (Information Incompleteness)：** 单目智能体在导航时无法实时感知侧方和后方的环境，导致难以在多个候选路径中做出正确选择，极易偏离航线。

* **渲染开销与质量权衡：** 现有的视图合成方法（如神经辐射场 NeRF）在预测未来或未知视角时，往往面临计算开销大（难以实时）或渲染出的特征图质量不稳定的问题。



**创新：**

论文提出了一个名为 **monoVLN** 的框架，利用 **3D 高斯泼溅 (3DGS)** 技术来桥接单目与全景视觉的差距：

* **引入 3DGS 特征渲染：**

  * 不同于传统的 NeRF 渲染，monoVLN 使用 **3D Gaussian Splatting (3DGS)** 来构建环境表示 。

  * 这种方法支持从单目序列中快速重建 3D 特征场，并渲染出当前视角之外的“虚拟全景图” 。

* **隐式局部补全模块 (Implicit Partial Completion Module)：**

  * 针对 3DGS 渲染出的全景特征图中由于遮挡或未探索区域产生的空洞（Missing Regions），提出了补全算法。

  * 该模块通过学习环境的先验知识，自动“预测”并补全缺失区域的特征，从而生成高质量、完整的全景特征表示 。

* **不确定性感知的有源感知策略 (Uncertainty-aware Active Perception Strategy)：**

  * 模型能够评估渲染特征的置信度。

  * 当环境极其陌生、特征预测不确定性较高时，智能体会主动执行“转身”等探索动作来采集真实数据，从而优化 3D 环境表示，减少盲目决策。



## 零样本视觉语言导航（Zero-Shot Vision-and-Language Navigation）

​	零样本VLNCE任务要求智能体在没有任何环境先验、拓扑图支持或专家轨迹参考的前提下，仅依靠实时的第一视角视觉输入和自然语言指令，在Habitat等连续仿真平台或现实世界中寻找目标。

![](images/image-22.png)

### Open-Nav(2025 ICRA)

**论文名称：**Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs

**作者：**Yanyuan Qiao1, Wenqi Lyu1, Hui Wang1, Zixu Wang2, Zerui Li1, Yuan Zhang1, Mingkui Tan2, Qi Wu1∗

**单位：**阿德莱德大学澳大利亚机器学习研究所，华南理工大学软件工程学院

**论文地址：**https://arxiv.org/abs/2409.18794

**项目地址：**https://github.com/YanyuanQiao/Open-Nav

![](images/image-23.png)

### CA-Nav(2025 TPAMI)

**论文名称：**Constraint-Aware Zero-Shot Vision-Language  Navigation in Continuous Environments

**作者：**Kehan Chen, Dong An, Yan Huang, Rongtao Xu, Yifei Su, Yonggen Ling, Ian Reid, Liang Wang

**论文地址：**https://arxiv.org/pdf/2412.10137

**项目地址：**https://chenkehan21.github.io/CA-Nav-project/

![](images/image-24.png)

![](images/image-25.png)

### LaViRA

**论文名称：**LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot  Vision Language Navigation in Continuous Environments

**作者：**Hongyu Ding, Ziming Xu, Yudong Fang, You Wu, Zixuan Chen, Jieqi Shi, Jing Huo, Yifan Zhang, Yang Gao

**论文：**https://arxiv.org/abs/2510.19655

**项目地址：**https://robo-lavira.github.io/lavira-zs-vln/&#x20;

### SmartWay(2025 IROS)

**论文名称：**SmartWay: Enhanced Waypoint Prediction and Backtracking for  Zero-Shot Vision-and-Language Navigation

**作者：**Xiangyu Shi, Zerui Li, Wenqi Lyu, Jiatong Xia, Feras Dayoub, Yanyuan Qiao, Qi Wu

**论文：**https://arxiv.org/abs/2503.10069

**项目地址：**https://sxyxs.github.io/smartway/

### Fast-SmartWay

**论文名称：**Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation

**作者：**Xiangyu Shi, Zerui Li, Yanyuan Qiao2, Qi Wu1

**论文：**https://arxiv.org/pdf/2511.00933

**开源代码：**



## 基于VLA的方法

​	在VLNCE领域，基于视觉-语言-动作（Vision-Language-Action, VLA）的模型正逐渐取代传统的两阶段（路标预测+导航执行）模块化架构。VLA模型的核心特征是将感知、语言理解和物理动作预测统一在单个端到端框架中，直接从原始图像流生成连续控制参数。

### NaVILA

**论文题目：**NaVILA: Legged Robot Vision-Language-Action Model for Navigation

**作者：**An-Chieh Cheng1,∗ Yandong Ji1,∗ Zhaojing Yang2,∗ Zaitian Gongye1 Xueyan Zou1 Jan Kautz3 Erdem Bıyık2 Hongxu Yin3,† Sifei Liu3,† Xiaolong Wang1,3,†

**单位：**1UC San Diego 2USC 3NVIDIA

**论文地址：**https://navila-bot.github.io/static/navila\_paper.pdf

**开源代码：**https://github.com/AnjieCheng/NaVILA

方法：

![](images/image-26.png)

NaVILA 提出了一个精巧的二级架构：

1. **高级VLA层**：该层基于预训练的视觉语言模型（如VILA），输入单帧RGB图像和历史背景信息。其创新点在于输出的是“中级动作（Mid-level actions）”，这些动作以自然语言的形式表达，例如“左转30度”或“向前移动75厘米” 。这种设计不仅降低了动作空间的维度，更重要的是，它使得模型能够利用互联网上丰富的自然语言文本和人类视角视频（如YouTube旅游视频）进行语义增强，而无需昂贵的机器人关节示踪数据。  &#x20;

2. **低级执行层**：该层是一个基于强化学习（RL）训练的视觉运动策略，负责将高级层下达的语言指令实时转化为精确的电机控制命令。由于低级策略在物理仿真器中经过了大量避障和地形适应训练，它能够处理高级模型无法感知的局部障碍物，并确保在湿滑或崎岖地面上的稳定性。

VLA框架：

![](images/image-27.png)

sim2real:

![](images/image-28.png)



### StreamVLN

**论文题目：**StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling

**作者：**Meng Wei∗,1,2 Chenyang Wan∗,1,3 Xiqian Yu∗,1 Tai Wang∗†,1 Yuqiang Yang1 Xiaohan Mao1,4 Chenming Zhu1,2 Wenzhe Cai1 Hanqing Wang1 Yilun Chen1

**单位：**1上海人工智能实验室 2香港大学 3浙江大学 4上海交通大学

**论文地址：**https://arxiv.org/abs/2507.05240

**项目地址：**https://github.com/InternRobotics/StreamVLN

方法：

![](images/image-29.png)

​	这篇论文提出了 StreamVLN，这是一个专为解决基于视频的大模型（Video-LLMs）在长程导航中面临的“计算效率低”和“显存占用高”问题而设计的流式导航框架。其核心方法论是 SlowFast（快慢）上下文建模策略，模仿了人类处理短期记忆和长期记忆的机制：

​	快流上下文 (Fast-Streaming Context) 作为智能体的“短期工作记忆”，利用滑动窗口（Sliding Window）机制，仅保留最近几帧的视觉特征和当前的对话历史，专注于处理即时的视觉输入和动作生成，确保低延迟的实时响应，让智能体能快速看清眼前的路并做出反应。

​	慢更新记忆上下文 (Slow-Updating Memory Context)则作为智能体的“长期压缩记忆”，维护整个导航过程的长程上下文，防止“忘记”走过的路或之前的指令；为了避免显存爆炸，它不存储所有历史帧，而是采用 3D感知 Token 剪枝策略 (3D-aware Token Pruning)，定期识别并丢弃历史缓存中在 3D 空间结构上冗余的视觉 Token，只保留关键的地标和空间特征，从而用最小的内存成本维持对长程历史的有效记忆。这种快慢结合的设计使得 StreamVLN 能够在有界显存成本下实现高效的长视频流式导航。

​	通过这种设计，StreamVLN 实现了高效的 KV Cache 复用。无论导航路径多长，模型的显存占用和推理成本都保持在有界范围内，从而使得在消费级显卡上运行长程流式导航成为可能。

### DualVLN

论文题目：GROUND SLOW, MOVE FAST: A DUAL-SYSTEM FOUNDATION MODEL FOR GENERALIZABLE VISIONAND-LANGUAGE NAVIGATION

作者： Meng Wei1,2 Chenyang Wan1,3 Jiaqi Peng1,4 Xiqian Yu1 Yuqiang Yang1 Delin Feng1 Wenzhe Cai1 Chenming Zhu1,2 Tai Wang1,† Jiangmiao Pang1,‡ Xihui Liu2,‡

单位： 1上海人工智能实验室 2香港大学 3浙江大学 4清华大学

论文地址：https://arxiv.org/pdf/2512.08186

项目地址：https://github.com/InternRobotics/InternNav?tab=readme-ov-file

方法：

![](images/image-30.png)

​	这篇论文提出了一个名为 **DualVLN** 的双系统基础模型架构，其设计灵感来源于人类认知的“双过程理论”（Dual-Process Theory），即将导航过程解耦为“慢思考”的高层规划和“快反应”的底层执行。

​	慢系统 (System 2: "Ground Slow") —— 全局规划器，作为智能体的“大脑”，由强大的多模态大模型 InternVLA 驱动。它负责深思熟虑的高层决策，通过结合全景视觉和自然语言指令，理解复杂的跨模态语义（例如识别“红色的沙发”在哪里），并低频地运行以预测并输出中期航点（Mid-term Waypoints），其输出形式包括全景图上的像素级目标（Pixel Goal）以及语义潜变量特征，专注于解决“去哪里（Where to go）”的问题。

​	快系统 (System 1: "Move Fast") —— 局部执行器，则作为智能体的“小脑”，由一个轻量级的多模态扩散 Transformer (Diffusion Transformer)构成。它负责快速反应的底层控制，接收慢系统传递的“像素目标”和“语义特征”，结合当前的以自我为中心的实时观测（RGB-D + 里程计），高频地输出连续的线速度 ($v$) 和角速度 ($\omega$)，专注于处理具体的运动控制、动态避障和路径平滑，即解决“怎么走（How to go）”的问题。

​	两者协同工作，慢系统负责看懂地图指方向（解决语义迷茫），快系统负责盯着路况踩油门（解决控制颠簸），既实现了大模型的强推理，又保证了实机导航的流畅与安全。
