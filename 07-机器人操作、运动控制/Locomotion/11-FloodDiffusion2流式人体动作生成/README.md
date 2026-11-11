# FloodDiffusion 2：从文字和路径实时生成连续人体动作

如何让虚拟人物先走路、再挥手，并沿着指定路线持续运动？[FloodDiffusion 2](https://alayalab.github.io/FloodDiffusion2/)（FD2［第二代流式动作生成模型］）接收随时间变化的文字指令和可选的平面路径，逐帧生成连续的人体动作。本章从官方演示出发，解释共享历史的多窗口训练、运动学几何损失，以及模型下载、评测和训练入口。

**推荐位置：**本章属于“机器人操作与运动控制 → 人形运动生成”。FD2［第二代流式动作生成模型］生成的是人体动作特征；若要驱动机器人，还需要动作重定向、物理跟踪和部署环节，可继续阅读[UniSuperMimic 全身动作跟踪](../08-UniSuperMimic世界模型全身动作跟踪/README.md)。

| 资源 | 内容 |
| --- | --- |
| [官方项目主页](https://alayalab.github.io/FloodDiffusion2/) | 文字切换、路径控制和长时间动作视频 |
| [论文与附录](https://arxiv.org/abs/2609.33167) | 方法、公式、实验设置和算力表 |
| [官方源码，固定版本](https://github.com/AlayaLab/FloodDiffusion2/tree/12c38faa67d25ffb49722f064d9a49950a27b87d) | 模型、训练、评测、渲染及测试 |
| [预训练权重](https://huggingface.co/AlayaLab/FloodDiffusion2) · [预处理数据](https://huggingface.co/datasets/AlayaLab/FloodDiffusion2-Data) | 文字与路径模型、统计量和数据包 |
| [在线交互演示](https://huggingface.co/spaces/AlayaLab/FloodDiffusion2-Live) | 在浏览器体验提示词切换及连续生成，实际可用性取决于服务状态 |

## 1. 先看官方演示

点击画面进入原始视频。以下均为论文作者展示的**虚拟人体动作**：

| 慢跑、跳跃、继续前进 | 停下、挥手、继续行走 |
| --- | --- |
| [![慢跑、跳跃、继续前进](https://alayalab.github.io/FloodDiffusion2/assets/posters/jog_jump_continue.jpg)](https://alayalab.github.io/FloodDiffusion2/assets/videos/jog_jump_continue.mp4) | [![停下、挥手、继续行走](https://alayalab.github.io/FloodDiffusion2/assets/posters/stop_wave_resume.jpg)](https://alayalab.github.io/FloodDiffusion2/assets/videos/stop_wave_resume.mp4) |
| 不同文字指令在同一条动作流中衔接。 | 观察停止、挥手与再次行走之间的过渡。 |

| 沿曲线路径前进 | 同一提示词下调整速度 |
| --- | --- |
| [![沿曲线路径前进](https://alayalab.github.io/FloodDiffusion2/assets/posters/curved_path.jpg)](https://alayalab.github.io/FloodDiffusion2/assets/videos/curved_path.mp4) | [![同一提示词下调整速度](https://alayalab.github.io/FloodDiffusion2/assets/posters/variable_speed.jpg)](https://alayalab.github.io/FloodDiffusion2/assets/videos/variable_speed.mp4) |
| 路径约束角色的平面移动，同时生成身体动作。 | 同样的动作类型可配合不同的根部速度。 |

来源：[FloodDiffusion 2 官方项目页](https://alayalab.github.io/FloodDiffusion2/)。封面和视频由原站提供；[完整演示](https://alayalab.github.io/FloodDiffusion2/assets/videos/demo-v45-1080p.mp4)还包含动作与路径切换。

## 2. 整体流程：文字、路径如何变成动作

![论文中根据文字和路径生成动作的示意图](https://arxiv.org/html/2609.33167v2/fd2_task_teaser.png)

**图 1：**文字指令和目标路径共同控制连续人体运动；图中轨迹对比了输入路径与生成动作对应的身体轨迹。来源：[原论文图 1](https://arxiv.org/html/2609.33167v2)。

```text
文字指令 ─→ 冻结的 UMT5-XXL 文本编码器 ─────────┐
                                              ↓
可选路径 → 根部位置、朝向的相对运动编码 → 动作生成主干
                                              ↓
                          活动窗口逐步去噪 + 已完成历史缓存
                                              ↓
                        人体动作特征 → 骨架/人体网格 → 视频
```

FD2［第二代流式动作生成模型］的去噪主干约 1.14 亿参数，使用 8 层、宽度 1024、8 个注意力头的 Transformer［注意力网络］；冻结的 UMT5-XXL［多语言文本编码器］单独提供文本条件，并不包含在主干参数量里。主干借鉴 Wan［视频生成模型］的模块设计，针对动作特征和流式注意力进行了适配。

HumanML3D［人体动作与文字数据集］配置使用每帧 263 维特征；SEED［人体动作数据集］配置使用每帧 138 维特征。路径版本将 3 个根部运动通道作为已知条件，分别生成其余 260 或 135 个通道。SEED［人体动作数据集］的路径根节点采用身体质量加权中心的地面投影，与骨盆的相对姿态分开表示。

## 3. 核心一：历史只编码一次，多窗口共同训练

![官方部分注意力与多窗口共享历史示意图](https://arxiv.org/html/2609.33167v2/fd2_partial_attention_packing.svg)

**图 2：**已完成的历史使用因果可见范围，当前窗口内部双向交互；训练时多个彼此隔离的窗口共享一条干净参考历史。来源：[原论文图 3](https://arxiv.org/html/2609.33167v2#S3.F3)。

假设前两帧已经完成，第 3、4 帧正在一起去噪。下面每行是发出查询的帧，每列是可读取的帧；`1` 表示可见，`0` 表示遮蔽：

| 查询帧 | 历史 1 | 历史 2 | 当前 3 | 当前 4 |
| --- | ---: | ---: | ---: | ---: |
| 历史 1 | 1 | 0 | 0 | 0 |
| 历史 2 | 1 | 1 | 0 | 0 |
| 当前 3 | 1 | 1 | 1 | 1 |
| 当前 4 | 1 | 1 | 1 | 1 |

历史不读取当前窗口，因此历史特征不会随当前窗口反复去噪而改变；当前窗口中的帧又可以互相参考，协调动作过渡。这里的“未来帧”是同一窗口里正在共同生成的候选帧。

**推理阶段：**每层缓存已完成历史的键和值；新完成的帧经过一次因果重编码后加入缓存。更新当前窗口时直接读取历史缓存。文字切换后，已完成帧保留生成时的条件。

**训练阶段：**对同一条干净动作，从不同扩散时间区间采样多个噪声窗口，拼接到同一次前向计算中。每个窗口只读取自己起点之前的干净历史和本窗口；窗口之间互相不可见。即使两个窗口来自有重叠的原始帧，也不会互相读取。它们由此共享同一次历史计算；这不是跨训练步骤复用缓存。

例如，参考序列为第 1～6 帧，窗口 A 使用第 3、4 帧，窗口 B 使用第 5、6 帧：A 只能读取干净参考的第 1、2 帧及 A 自身；B 读取干净参考的第 1～4 帧及 B 自身。损失在有效窗口帧上汇总。源码入口为 [`_build_multi_window_attn_mask`](https://github.com/AlayaLab/FloodDiffusion2/blob/12c38faa67d25ffb49722f064d9a49950a27b87d/models/diffusion_forcing_wan.py#L911) 和 [`forward`](https://github.com/AlayaLab/FloodDiffusion2/blob/12c38faa67d25ffb49722f064d9a49950a27b87d/models/diffusion_forcing_wan.py#L1005)。

## 4. 核心二：把运动学影响做成固定几何损失

![官方几何监督与固定二次损失示意图](https://arxiv.org/html/2609.33167v2/fd2_loss_characterization.svg)

**图 3：**离线估计几何响应并构造固定权重矩阵，训练时使用二次损失。来源：[原论文图 4](https://arxiv.org/html/2609.33167v2#S3.F4)。

普通均方误差把各维特征的相同数值误差一视同仁，但不同关节的误差可能引起不同的身体位置变化。为理解几何权重，先看一个简化的双关节例子。假设某姿态下末端位置对两个关节的局部响应为：

$$
J=\begin{bmatrix}0&0\\2&1\end{bmatrix},\qquad
J^TJ=\begin{bmatrix}4&2\\2&1\end{bmatrix}.
$$

若误差为 $e=[e_1,e_2]^T$，则局部末端位置误差平方近似为 $e^T J^T J e=4e_1^2+4e_1e_2+e_2^2$。两个关节各自出现 0.1 弧度误差时，单独作用的近似末端误差平方分别为 0.04 和 0.01；交叉项则表达误差的耦合。这个二维例子用于理解矩阵含义，实际人体表示和权重以论文及代码为准。

论文在训练数据上构造并平均几何响应矩阵，再归一化，并与单位矩阵组合：

$$
\overline G=\mathbb E[J^T M J],\qquad
W_{\rm FK}=\frac{d\overline G}{\operatorname{tr}(\overline G)},\qquad
W_\gamma=\frac{I+\gamma W_{\rm FK}}{1+\gamma},\qquad
L=\frac{1}{d}e^T W_\gamma e.
$$

这里 $e$ 是预测扩散速度与训练目标的误差；$M$ 衡量几何输出的权重，$d$ 是预测通道数。固定正定的 $W_\gamma$ 会调整误差方向的权重，而其最优预测仍为条件均值；论文进一步在其假设下给出了与 Bregman 散度［一类衡量预测误差的函数］相关的判别条件。

两种数据表示的构造并不完全相同：SEED［人体动作数据集］使用人体网格解码器的局部响应；HumanML3D［人体动作与文字数据集］使用设计的关节运动学响应。路径模型只对需要预测的通道构造对应矩阵。详见[论文第 3.3 节](https://arxiv.org/html/2609.33167v2#S3.SS3)和[矩阵准备脚本](https://github.com/AlayaLab/FloodDiffusion2/blob/12c38faa67d25ffb49722f064d9a49950a27b87d/prepare_fk_matrix.py)。

## 5. 实验结果和资源需求

论文使用冻结的 UMT5-XXL［多语言文本编码器］、每个源序列 8 个窗口、每窗口 30 帧，以及学习率 `2e-4`。下面是作者在 300 帧、每步窗口数固定时报告的 H200［数据中心显卡］显存基准；显存数字来自论文的测量，不是本机实测：

| 每条序列窗口数 | 单卡序列批量 | 每步窗口数 | 参与显卡数 | 峰值显存合计 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 512 | 4 | 164.27 GiB［二进制吉字节］ |
| 4 | 128 | 512 | 1 | 54.27 GiB［二进制吉字节］ |
| 8 | 64 | 512 | 1 | 36.61 GiB［二进制吉字节］ |
| 16 | 32 | 512 | 1 | 27.97 GiB［二进制吉字节］ |
| 8 | 128 | 1024 | 1 | 72.37 GiB［二进制吉字节］ |

在 512 窗口的比较下，8 窗口共享训练的去噪主干前后向计算量约降为原来的 1/4.6；它是特定条件下的计算量比较。论文以 RTX 4090［显卡型号］测得 4500 帧历史时，缓存方案单次主干更新为 2.303 毫秒；该数字包含新历史帧重编码和缓存写入，使用预先算好的文本特征，测量不包含文本编码和渲染。[论文表 2、表 3](https://arxiv.org/html/2609.33167v2#S4)

| 数据与模型 | 论文选用的检查点 | 配置中的训练步数上限 |
| --- | ---: | ---: |
| HumanML3D［人体动作与文字数据集］：文字 | 60,000 | 100,000 |
| HumanML3D［人体动作与文字数据集］：文字 + 路径 | 55,000 | 100,000 |
| SEED［人体动作数据集］：文字 | 300,000 | 300,000 |
| SEED［人体动作数据集］：文字 + 路径 | 300,000 | 300,000 |

作者验证了单张 H200［数据中心显卡］训练，论文未给完整训练的实际耗时。示例配置的单卡源序列批量为 128、窗口数为 8、梯度累积默认 1；表中 36.61 GiB［二进制吉字节］对应批量 64，不应直接当作默认批量的实测值。96 GB［十进制吉字节］显存的 RTX PRO 6000［显卡型号］在容量上值得尝试，实际峰值和每步耗时需在空闲显卡上测量；训练过程中还需考虑数据加载、验证和保存时间。

可用稳定训练后的实测每步耗时估算基础训练时间。下表只是乘法示例，并非作者或本机测量；它尚未计入启动、验证和检查点保存：

| 稳定每步耗时 | 60,000 步 | 300,000 步 |
| ---: | ---: | ---: |
| 0.5 秒 | 约 8.3 小时 | 约 41.7 小时 |
| 1 秒 | 约 16.7 小时 | 约 83.3 小时 |
| 2 秒 | 约 33.3 小时 | 约 166.7 小时 |

## 6. 按官方项目复现

先在[在线演示](https://huggingface.co/spaces/AlayaLab/FloodDiffusion2-Live)观察文字和路径控制，再按照本节准备官方权重、数据和环境。以下命令依据固定版本的官方说明整理；本教程尚未在本机运行完整推理或训练。

### 6.1 硬件、磁盘与环境

官方测试环境为 Linux［操作系统］、Python 3.10［编程语言］、PyTorch 2.9.1［深度学习框架］及 CUDA 12.8［显卡计算平台］。论文推理测量使用 RTX 4090［显卡型号］，训练测量使用 H200［数据中心显卡］。若渲染人体网格，需要自行从官方渠道申请 SMPL-H［人体参数模型］及其许可，并准备可用的图形渲染环境。

官方资源清单显示：共享依赖压缩包约 13 GB［十进制吉字节］，四个主要检查点各约 2.28 GB［十进制吉字节］；`--with-data` 还会获取 HumanML3D［人体动作与文字数据集］、BABEL［人体动作标注数据集］和 SEED［人体动作数据集］三个数据包。压缩包、解压数据与训练检查点会同时占用空间，建议提前预留至少 80 GB［十进制吉字节］，长期训练另留保存空间。数据、人体模型和权重分别遵守各自许可。

下面将源码、环境、资源和输出放在同一个自选数据盘路径；将 `/path/to/robot-data` 替换为实际可写目录：

```bash
export FD2_ROOT=/path/to/robot-data/flooddiffusion2
mkdir -p "$FD2_ROOT"
git clone https://github.com/AlayaLab/FloodDiffusion2.git "$FD2_ROOT/source"
cd "$FD2_ROOT/source"
git checkout 12c38faa67d25ffb49722f064d9a49950a27b87d
conda create -p "$FD2_ROOT/env" python=3.10
conda activate "$FD2_ROOT/env"
```

确认驱动与显卡可用，再安装依赖并获取官方预训练资产和数据：

```bash
nvidia-smi
python setup_project.py --with-data
python setup_project.py --check --with-data
```

`setup_project.py` 会安装依赖并验证清单中的文件；`--with-data` 下载所有清单数据包。运行时需要能访问资源服务器并有足够磁盘；中断后可按上游说明续传。命令在源码目录内放置 `deps/`、`checkpoints/` 和 `data/`，默认结果写入 `outputs/`。如已自行安装依赖，可按官方说明使用 `--skip-install`。来源：[安装说明](https://github.com/AlayaLab/FloodDiffusion2/tree/12c38faa67d25ffb49722f064d9a49950a27b87d#installation)、[资源清单](https://github.com/AlayaLab/FloodDiffusion2/blob/12c38faa67d25ffb49722f064d9a49950a27b87d/assets_manifest.json)。

如果需要网格视频，先取得经许可的中性人体模型文件，然后在相同环境运行：

```bash
python setup_project.py --smplh /path/to/neutral/model.npz
```

### 6.2 用预训练模型评测与生成示例

在源码根目录运行文字条件模型：

```bash
python evaluate.py --config configs/df_humanml3d_263.yaml
```

这是官方配置选择的检查点和数据集评测入口：`val_meta_paths` 指定指标评测数据，`test_meta_paths` 指定生成和渲染示例，默认包括 `test_min.txt`；输出位于 `outputs/` 下。仅生成与渲染、跳过文字到动作指标时，可按上游提供的覆盖项运行：

```bash
python evaluate.py --config configs/df_humanml3d_263.yaml --override metrics.t2m=null
```

路径条件模型使用以下配置；官方评测入口从输入动作数据读取根部条件：

```bash
python evaluate.py --config configs/df_humanml3d_263_path.yaml
```

检查日志中的资源加载、生成与评测结果，以及输出目录里的特征与视频。若缺少 `Mean.npy`、`Std.npy` 或 `W.npy`，先重新运行资源检查；若网格渲染报人体模型或图形环境错误，核对模型许可、文件路径和上游渲染说明。自由输入提示词、交互式切换与路径绘制可先使用[官方在线演示](https://huggingface.co/spaces/AlayaLab/FloodDiffusion2-Live)，其结果不等同于本机运行记录。

### 6.3 训练和重新构造几何矩阵

确认数据、统计量和显卡空闲后，官方文字模型的训练入口为：

```bash
python train_df.py --config configs/df_humanml3d_263.yaml
```

路径或 SEED［人体动作数据集］模型改用表中对应配置。训练从配置指定的初始化方式开始，`test_ckpt` 是评测使用的检查点；恢复训练应根据配置中的 `resume_ckpt` 和上游说明选择匹配的完整状态。检查 `outputs/` 中的配置、日志和定期保存的检查点。配置默认采用单卡、混合精度、批量 128；更换批量或窗口数会改变训练设置，应在报告中记录。

资源包已提供对应的 `W.npy`。如需理解矩阵构造，可在数据和依赖准备完成后将结果写到新文件：

```bash
python prepare_fk_matrix.py --config configs/df_humanml3d_263.yaml --output outputs/recomputed_W.npy
```

SEED［人体动作数据集］矩阵重建还需要合法取得的 SMPL-H［人体参数模型］。检查输出维度、数值和配置，并与匹配版本的官方矩阵比较；预计算结果的误差与随机采样设置有关。

## 7. 从虚拟人体走向机器人

FD2［第二代流式动作生成模型］可以作为机器人运动参考的上游：

```text
文字与路径 → 人体参考动作 → 关节映射与动作重定向
           → 物理仿真中训练/评测跟踪策略 → 真机部署验证
```

这一接口需要统一坐标系、帧率、骨架和关节限制，并处理接触与平衡。FD2［第二代流式动作生成模型］的官方成果验证了人体动作生成和路径控制；要回答机器人能否稳定执行，需要后续使用机器人模型、仿真控制器和真实平台分别评测。

建议先对一条生成的行走或蹲行动作可视化并检查脚底与根部轨迹；再按[UniSuperMimic 视频动作与社区复现](../08-UniSuperMimic世界模型全身动作跟踪/02-视频动作与社区复现.md)了解 G1 动作格式和跟踪训练。将 FD2［第二代流式动作生成模型］输出接到该流程时，还需要针对其人体动作表示开发并验证转换器。

## 8. 结果核对

完成复现时，至少记录源码版本、模型配置与检查点、数据版本、输入文字及路径、生成的动作长度、显卡型号、训练或推理耗时，以及生成视频和评测命令。对比论文数字时保持数据切分、条件、评测工具及是否启用路径一致。
