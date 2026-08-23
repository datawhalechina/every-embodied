# LeHome Challenge 比赛镜像使用教程

本文档面向已经拿到官方 LeHome 比赛镜像、希望尽快开始训练与评测的同学。

重点回答四个问题：

1. 当前镜像环境能不能直接用。
2. 训练应该下载 `merged` 还是非 `merged` 数据。
3. 如何用最小代价启动训练、评测、复现实验。
4. 训练配置具体怎么写，输出放到哪里。
5. 哪些大文件不要提交进教程仓库。

视频教程：LeHome柔性衣物折叠ICRA2026比赛视频教程 http://xhslink.com/o/2oxCz0RTXcA


<video src="https://assets.datawhale.cn/1090/dashboard/1773720966495/LeHome%E6%95%99%E7%A8%8B.mp4" width="100%" controls>
  您的浏览器不支持 HTML5 视频。
</video>



---

## 1. 当前镜像是否可用

结论：可用。

已在当前镜像中完成以下验证：

- `LeHome` 仓库路径：`/root/lehome-challenge`
- `isaacsim` 可用：`5.1.0.0`
- `lerobot` 可用：`0.4.3`
- GPU：`NVIDIA L40`
- 本地 HEAD：`2953e2f9c1376a1e49ac10b3ad690efb886f4c6c`
- 当前远端 `origin/main`：`32b53595da504880592a79ed5e362ad0ba0fac6b`
- 已成功跑通一轮 ACT 评测
- 已成功跑通一轮 DP 评测，并且 `--policy_num_inference_steps 1` 生效
- 旧的失败评测视频已清理，后续统一输出到：
  - `/root/gpufree-data/lehome-outputs/eval/`

说明：

- 环境是能启动、能加载模型、能建场景、能完成评测的。
- 当前本地代码不是远端最新。
- 如果大家想严格对齐当前镜像，请先切到：

```bash
git checkout 2953e2f9c1376a1e49ac10b3ad690efb886f4c6c
```

- 如果大家想跟远端最新主分支对齐，再切到：

```bash
git fetch origin
git checkout 32b53595da504880592a79ed5e362ad0ba0fac6b
```

- 当前复现实验结果与官方复现包描述一致，成功率依旧偏低，这更像是任务本身/物理效果问题，不是镜像起不来。

---

## 2. 训练数据该下载哪个

### 2.1 结论

如果目标是尽快开始基线训练，优先下载：

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

也就是优先使用 `merged` 版本。

### 2.2 原因

官方训练配置默认就是 `merged` 数据：

- `configs/train_act.yaml` 使用 `Datasets/example/top_long_merged`
- `configs/train_dp.yaml` 使用 `Datasets/example/top_long_merged`
- `configs/train_smolvla.yaml` 使用 `Datasets/example/top_long_merged`

这说明官方预期的训练入口就是“合并后的数据集”。

### 2.3 `merged` 和非 `merged` 的区别

可以这样理解：

- `dataset_challenge`
  - 更偏原始/分散的数据组织。
  - 适合你自己做数据处理、拆分、增广、补深度、补点云。
  - 如果你想研究数据生产流程，或者自己重构训练集，可以用它。
- `dataset_challenge_merged`
  - 更偏训练直接可用的整理结果。
  - 已经按类别合并成 `top_long_merged`、`pant_long_merged` 这类目录。
  - 最适合直接喂给 `lerobot-train`。

### 2.4 深度数据要不要下载

如果你只是先跑通 baseline：

- 先用 `merged`
- 先不用深度
- 先用官方 RGB + joint state 基线

原因：

- 官方文档明确支持 `state + RGB` 作为已验证组合
- 深度虽然可用，但会增加存储、I/O 和配置复杂度
- 官方数据处理文档里还专门给了“去掉 depth 减小存储”的流程

### 2.5 推荐选择

推荐顺序：

1. 入门/比赛 baseline：`dataset_challenge_merged`
2. 需要深度实验：在 `merged` 基础上再加 `observation.top_depth`
3. 需要自己做数据重构：再研究非 `merged` 的 `dataset_challenge`

### 2.6 教学起步该选哪一类任务

如果目的是做教学，而不是一上来就冲完整比赛，我建议优先从：

- `top_short`

开始。

这里要说清楚：这是一个工程上的教学选择，不是因为当前复现包里它已经成功率最高。当前复现包里四类历史统计都是 `0.0%` 成功率，所以不能把“最容易成功”说成已经被现有结果证明。

之所以仍然建议先讲 `top_short`，是因为它更适合教学起步：

- 相比 `top_long`，短袖上衣通常少了长袖带来的远端布料耦合
- 相比裤子类，短袖上衣的视觉目标更直观，学习者更容易理解“折叠成功”是什么
- 同样是上衣任务，`top_short` 通常比 `top_long` 更适合作为第一课

因此，本教程采用两层路线：

1. 教学路线：先跑 `top_short`
2. 比赛路线：再把四类任务全部跑完

---

## 3. 最小启动流程

## 3.1 克隆官方仓库

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

如果镜像已经预装好了，直接进入现成目录即可：

```bash
cd /root/lehome-challenge
```

## 3.2 下载资产

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

## 3.3 下载训练样例数据

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

如果只想快速验证一个类别，也可以只下载需要的子目录，例如：

```bash
hf download lehome/dataset_challenge_merged \
  --repo-type dataset \
  --local-dir Datasets/example \
  --include 'top_long_merged/**'
```

如果你准备按本教程完整学习，建议确认四类目录都在：

- `Datasets/example/top_short_merged`
- `Datasets/example/top_long_merged`
- `Datasets/example/pant_short_merged`
- `Datasets/example/pant_long_merged`

这里要明确区分：

- 教学起步：先用 `top_short_merged`
- 完整比赛：四类都要跑

### 3.3.1 教学流程和比赛流程的区别

教学流程不是为了第一天就把比赛所有任务跑完，而是为了先让学习者掌握：

- 数据目录怎么选
- 训练怎么起
- 日志怎么看
- 模型保存在哪里
- 评测视频在哪里

所以教学流程先只跑一个类别，推荐 `top_short`。

比赛流程则不同。比赛流程要求你最终覆盖四类任务：

- `top_short`
- `top_long`
- `pant_short`
- `pant_long`

## 3.4 教学第一课：先训练 `top_short` 的 ACT

建议使用教程专用配置，统一把输出保存到数据盘：

- 配置文件：
  - `/root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml`

```bash
mkdir -p /root/gpufree-data/lehome-outputs/train/act_top_short

lerobot-train \
  --config_path /root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_act_every_embodied.yaml \
  2>&1 | tee /root/gpufree-data/lehome-outputs/train/act_top_short/train.log
```

这条命令做了两件事：

1. 用教程专用 YAML 启动训练。
2. 用 `tee` 把终端日志同时保存到 `train.log`，方便后面分析 loss 和画图。

对于初学者，建议第一次不要改命令行，而是先读懂 YAML 里每个字段的作用，再只改一两个参数做实验。

这份 YAML 已经是 `top_short` 教学版，不需要你再手改。

## 3.5 教学第二课：在同一个 `top_short` 任务上训练 DP

同样建议使用教程专用配置：

- 配置文件：
  - `/root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml`

```bash
mkdir -p /root/gpufree-data/lehome-outputs/train/dp_top_short

lerobot-train \
  --config_path /root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_dp_every_embodied.yaml \
  2>&1 | tee /root/gpufree-data/lehome-outputs/train/dp_top_short/train.log
```

DP 相比 ACT 通常更慢、更吃资源，所以教程配置里把：

- `batch_size` 设得比 ACT 小
- `steps` 拉长到 `90000`
- `log_freq` 调细到 `100`

这样更符合 Diffusion Policy 的常见训练习惯，也更方便观察收敛过程。

这份 YAML 同样已经是 `top_short` 教学版，不需要手改。

## 3.6 教学评测：评测 `top_short` 的 ACT

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /root/gpufree-data/lehome-outputs/train/act_top_short/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir /root/gpufree-data/lehome-outputs/eval/act_top_short \
  --device cpu
```

## 3.7 教学评测：评测 `top_short` 的 DP

CPU 评测时建议显式限制 diffusion 推理步数：

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /root/gpufree-data/lehome-outputs/train/dp_top_short/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/top_short_merged \
  --garment_type top_short \
  --num_episodes 2 \
  --enable_cameras \
  --save_video \
  --video_dir /root/gpufree-data/lehome-outputs/eval/dp_top_short \
  --device cpu \
  --policy_device cpu \
  --policy_num_inference_steps 1
```

### 3.7.1 为什么教学不直接从四类一起开始

因为四类一起做，对第一次上手的人来说会同时叠加：

- 四套数据目录
- 四套模型输出目录
- 四套评测命令
- 更长训练时间
- 更复杂的排障路径

教学最重要的是先把一个完整闭环讲清楚，所以这里先拿 `top_short` 做第一课。

### 3.7.2 完整比赛流程怎么做

完整比赛不等于只跑 `top_short`。完整比赛建议按类别分别完成：

1. `top_short`
2. `top_long`
3. `pant_short`
4. `pant_long`

也就是说，你最终应该把“训练 + 评测 + 结果汇总”这套流程复制到四个类别上。

### 3.7.3 四类任务的完整训练组织方式

最稳妥、最容易教学的方案是：

- 四类分别训练
- 四类分别评测
- 最后统一汇总

推荐的目录组织：

```text
/root/gpufree-data/lehome-outputs/
├── train/
│   ├── act_top_short/
│   ├── act_top_long/
│   ├── act_pant_short/
│   ├── act_pant_long/
│   ├── dp_top_short/
│   ├── dp_top_long/
│   ├── dp_pant_short/
│   └── dp_pant_long/
├── eval/
└── plots/
```

### 3.7.4 四类任务怎么依次跑

你可以把 `top_short` 的流程看作模板，然后把下面三个量替换掉：

- `dataset.root`
- `output_dir`
- `garment_type`

例如：

- `top_short`
  - `dataset.root: Datasets/example/top_short_merged`
  - `garment_type: top_short`
- `top_long`
  - `dataset.root: Datasets/example/top_long_merged`
  - `garment_type: top_long`
- `pant_short`
  - `dataset.root: Datasets/example/pant_short_merged`
  - `garment_type: pant_short`
- `pant_long`
  - `dataset.root: Datasets/example/pant_long_merged`
  - `garment_type: pant_long`

所以教程里最关键的学习点不是死记一个命令，而是学会：

- 哪三个地方一换，就能迁移到另一类任务

### 3.7.5 比赛提交前要整理什么

当前官方仓库 README 里并没有给出固定的提交通道命令，它写的是：

- 提交说明将在官网提供

所以这里不能伪造一个“官方提交脚本”。但你完全可以把“提交前要准备好的材料”先整理好。

建议至少准备：

- 四类任务各自最终模型
- 四类任务各自评测日志
- 四类任务各自 success rate
- 四类任务各自代表性视频
- 训练配置 YAML
- 对应代码 commit

从教程角度，这一步可以理解成：

- 先把你的“比赛实验包”整理完整
- 等官方提交格式明确后再上传

## 3.8 训练配置详解

教程专用配置里，最值得解释的是这些字段：

- `dataset.root`
  - 训练数据目录。
  - 教学第一课默认使用 `Datasets/example/top_short_merged`。
  - 这样设计的原因是官方基线配置本身就是指向 `*_merged`，初学者不需要先理解多数据集合并逻辑，也能直接开练。
- `policy.type`
  - `act` 或 `diffusion`。
  - ACT 是更推荐的第一站，因为训练逻辑更直观，速度通常也更快。
  - Diffusion Policy 更适合在你已经跑通 ACT baseline 后再继续比较。
- `policy.device`
  - 训练建议 `cuda`。
  - 因为图像编码器和策略网络都比较重，CPU 训练几乎没有效率。
- `input_features`
  - baseline 推荐 `observation.state + top/left/right RGB`。
  - 这是一个工程上很稳妥的起点。
  - `state` 提供机械臂关节信息，RGB 提供布料形态和位姿信息。
  - 对学习者来说，这个组合也最容易理解：一部分是“机器人自己当前姿态”，一部分是“摄像头看到的场景”。
- `output_features`
  - baseline 推荐输出 `action`。
  - 也就是直接预测关节动作。
  - 这种 joint-space control 比 `ee_pose` 更适合作为教学 baseline，因为它少了一层 IK 误差。
- `output_dir`
  - 已改到数据盘，例如 `/root/gpufree-data/lehome-outputs/train/act_top_short`。
  - 这样做是为了避免把系统盘写满。
- `batch_size`
  - 和显存直接相关。
  - ACT 教学版当前默认 `64`，DP 教学版当前默认 `48`。
  - 对初学者可以简单理解为：一次喂给模型多少条训练样本。
  - 越大通常训练越快，但显存占用也越高。
- `steps`
  - 总训练步数。
  - ACT 可先跑 `30000`，DP 建议更长，教程配置里给到 `90000`。
  - 训练步数不是“越大越好”，而是“足够看到收敛趋势”。
  - 教学上先用一个可跑完、可观察的步数更重要。
- `save_freq`
  - checkpoint 保存间隔。
  - 教程里调高了保存频率，便于中间结果回看。
  - 这对教学尤其重要，因为你可以比较不同阶段 checkpoint 的效果，而不是只看最后一个模型。
- `log_freq`
  - 日志打印频率。
  - 教程里调高到更细，方便后续画图和排障。
  - 对学习者来说，训练日志就是最直观的“训练过程观察窗口”。
- `eval_freq`
  - 训练中的阶段性评估间隔。
  - 便于观察并不是只有 loss 在变化。
  - 如果一个模型 loss 在下降，但 eval success rate 不涨，这通常意味着“学到了训练分布内拟合”，但未必真正学会了任务。

## 3.8.1 为什么 ACT 和 DP 的参数不一样

这是很多学习者第一次会问的问题。

### ACT

ACT 可以先理解成“基于视觉和状态做动作序列预测”的 transformer 类策略。

在本教程里，我们给 ACT 这样一组更激进、接近 L40 利用上限的参数：

- `batch_size: 64`
- `steps: 30000`
- `save_freq: 5000`
- `log_freq: 200`
- `eval_freq: 5000`

这样设计的目的：

- 比较容易在单卡上跑起来
- 日志足够细，便于教学观察
- 每隔 `5000` step 就能拿到一个 checkpoint
- 总步数 `30000` 足够看到一个 baseline 的基本收敛趋势

### Diffusion Policy

Diffusion Policy 的推理和训练都更重，尤其在多相机输入下更明显。

教程配置里我们给了：

- `batch_size: 48`
- `steps: 90000`
- `save_freq: 10000`
- `log_freq: 100`
- `eval_freq: 10000`

这样设计的目的：

- 降低显存压力
- 延长训练步数，给 diffusion 模型足够的收敛空间
- 用更细的日志频率观察 loss 和 gradient norm

所以不要机械地认为“ACT 和 DP 的参数应该一样”。不同策略结构，本来就应该有不同的训练节奏。

## 3.8.2 训练要跑多久，显存大概多少

这里给的是经验预估，不是绝对值。

在当前这类 `top_short_merged + state + 3路RGB` 配置下，可以先这样估算：

### ACT

- 推荐显存：至少 `16GB`，更稳妥是 `24GB+`
- 在 L40 这类卡上通常可以单卡训练
- `30000 step` 一般是“小时级”任务，不是分钟级
- 如果日志里 `data_s` 和 `updt_s` 都不大，通常几个小时内能看到完整 baseline 结果

### DP

- 推荐显存：`24GB+` 更稳
- `90000 step` 通常明显比 ACT 更久
- 更适合作为过夜训练任务

学习者应该关心的是：

- 有没有 OOM
- 每 step 花多久
- loss 是否稳定下降
- 中间 checkpoint 的 eval 是否真的变好

不要一上来纠结“到底 3 小时还是 5 小时”，先看自己机器日志里的：

- `updt_s`
- `data_s`
- `step_per_sec`

这些值才是真正决定训练时间的依据。

## 3.8.3 第一次训练建议怎么做

推荐按下面顺序学：

1. 先训练 ACT，不要先上 DP。
2. 不要先加 depth。
3. 不要先改成 `ee_pose`。
4. 先跑通一个类别，比如 `top_short_merged`。
5. 先观察 loss、grad norm、lr 和中间 eval。
6. 确认 pipeline 稳定以后，再扩大到更多类别。

## 3.9 建议保留的训练中间结果

建议至少保留这些：

- `train.log`
- 所有 checkpoint
- `config.yaml` 或训练配置副本
- 训练曲线图
- 训练指标 CSV / JSON 摘要
- 若有阶段性 eval，也保留对应日志和视频

对于教学用途，还建议加上：

- 一张训练曲线截图
- 一份“最佳 checkpoint”和“最后 checkpoint”的对比结论
- 一份训练耗时记录
- 一份显存占用记录

建议统一放到：

```text
/root/gpufree-data/lehome-outputs/
├── train/
├── eval/
└── plots/
```

## 3.10 训练日志解析与绘图

教程已附带一个脚本：

- `/root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py`

它会从 `train.log` 中解析并生成：

- `train_metrics.csv`
- `train_metrics_summary.json`
- `train_metrics.png`

示例：

```bash
python /root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/scripts/plot_train_metrics.py \
  --log_file /root/gpufree-data/lehome-outputs/train/act_top_short/train.log \
  --out_dir /root/gpufree-data/lehome-outputs/plots/act_top_short \
  --title "ACT Top-Short Training Metrics"
```

当前脚本默认记录这些指标：

- `loss`
- `grdn`，也就是 gradient norm
- `lr`
- `updt_s`
- `data_s`
- `step_per_sec`

这些指标的教学意义分别是：

- `loss`
  - 最基础的优化目标，先看是否整体下降。
- `grdn`
  - 用来看梯度是否爆炸或异常不稳定。
- `lr`
  - 用来看学习率调度是否正常。
- `updt_s`
  - 单步参数更新耗时。
- `data_s`
  - 单步数据加载耗时。
- `step_per_sec`
  - 训练吞吐量的直观指标。

## 3.11 除了 loss，还建议看哪些指标

如果站在 ACT / Diffusion Policy 论文复现和工程训练角度，除了 loss，还建议尽量记录：

- 训练 loss
- 验证 loss
- gradient norm
- learning rate
- data loading time
- update time
- steps/sec 或 samples/sec
- checkpoint 对应的阶段性 eval 成功率
- 平均回报 `return`
- episode length
- 不同 garment/category 的 success rate

真正和比赛结果最相关的，不是单独看 loss，而是：

1. `loss` 是否稳定下降
2. `gradient norm` 是否异常爆炸
3. `lr` 调度是否正常
4. 中间 checkpoint 的 `eval success rate` 是否同步提升

## 3.12 训练结束后会保存什么

训练完成后，通常最值得关注的是这些内容：

- `train.log`
  - 原始训练日志
- `checkpoints/`
  - 每个阶段保存下来的模型
- `last/`
  - 最后一次训练状态
- `pretrained_model/`
  - 用于评测和部署的模型目录
- 你自己生成的：
  - `train_metrics.csv`
  - `train_metrics_summary.json`
  - `train_metrics.png`

从教学角度，建议把这些文件分成两类理解：

### 第一类：训练产物

- checkpoint
- last
- pretrained_model

这些是“模型本身”。

### 第二类：训练证据

- train.log
- 指标 CSV/JSON
- 曲线图
- 中间 eval 结果

这些是“为什么我们认为这个模型训练正常”的证据。

教学里两类都要保留，不能只保留模型文件。

如果后续你要再往上加，我建议优先补：

- 每个 checkpoint 自动 eval
- 每个类别的 success rate 曲线
- best checkpoint 自动挑选

## 3.13 已经真实跑过的 smoke 结果

为了避免教程只停留在“命令应该能跑”，这里保留本机已经真实跑过的最小训练证据。

### 3.13.1 ACT 教学版 smoke

- 训练日志：
  - `/root/gpufree-data/lehome-outputs/train/act_top_short_probe64.log`
- checkpoint 输出：
  - `/root/gpufree-data/lehome-outputs/train/act_top_short_probe64/checkpoints/000008/pretrained_model`
- 曲线图：
  - `/root/gpufree-data/lehome-outputs/plots/act_top_short_probe64/train_metrics.png`
- 指标摘要：
  - `/root/gpufree-data/lehome-outputs/plots/act_top_short_probe64/train_metrics_summary.json`

本次 smoke 的关键信息：

- 数据：`top_short_merged`
- 有效 batch size：`64`
- 参数量：约 `52M`
- 已跑到 step `8`
- loss 从 `57.147` 降到 `22.134`
- 平均 `updt_s` 约 `1.77s`
- 平均 `data_s` 约 `0.60s`

这说明当前教程默认的 ACT 教学配置至少能在本机 L40 上稳定起跑，不需要用户再手工改 YAML 才能启动。

### 3.13.2 DP 教学版 smoke

- 训练日志：
  - `/root/gpufree-data/lehome-outputs/train/dp_top_short_probe48.log`
- checkpoint 输出：
  - `/root/gpufree-data/lehome-outputs/train/dp_top_short_probe48/checkpoints/000008/pretrained_model`
- 曲线图：
  - `/root/gpufree-data/lehome-outputs/plots/dp_top_short_probe48/train_metrics.png`
- 指标摘要：
  - `/root/gpufree-data/lehome-outputs/plots/dp_top_short_probe48/train_metrics_summary.json`

本次 smoke 的关键信息：

- 数据：`top_short_merged`
- 有效 batch size：`48`
- 参数量：约 `271M`
- 已跑到 step `8`
- loss 从 `1.119` 降到 `1.097`
- 平均 `updt_s` 约 `0.94s`
- 平均 `data_s` 约 `0.79s`

这说明 DP 教学版配置也已经在本机验证过能稳定启动，并且能正常写出 checkpoint 和训练曲线。

### 3.13.3 如何理解这些 smoke 结果

这些结果不是正式比赛成绩，也不是完整收敛实验。它们的作用是：

- 证明命令、配置、输出路径已经打通
- 证明当前镜像里 `top_short` 教学版配置可以直接启动
- 证明教程附带的画图脚本可以直接消费真实日志

正式比赛训练时，仍然建议按完整步数继续跑：

- ACT：`30000` steps
- DP：`90000` steps

如果你想进一步吃满 L40 的 40G 显存，最稳妥的方法不是盲目加步数，而是优先继续试探：

- ACT 再增大 `batch_size`
- DP 再增大 `batch_size`
- 或者开启更多周期性的中间 eval

但教学版文档默认保留当前已经验证过的安全起点，这样读者第一次运行更稳。

### 3.13.4 已经导出的 smoke 评测视频

除了训练 smoke 之外，本机还实际把 `top_short` 的首个 smoke 评测 episode 视频导出来了，位置如下：

- ACT:
  - `/root/gpufree-data/lehome-outputs/eval/act_top_short_smoke/failure/episode0_observation_images_top_rgb.mp4`
  - `/root/gpufree-data/lehome-outputs/eval/act_top_short_smoke/failure/episode0_observation_images_left_rgb.mp4`
  - `/root/gpufree-data/lehome-outputs/eval/act_top_short_smoke/failure/episode0_observation_images_right_rgb.mp4`
- DP:
  - `/root/gpufree-data/lehome-outputs/eval/dp_top_short_smoke/failure/episode0_observation_images_top_rgb.mp4`
  - `/root/gpufree-data/lehome-outputs/eval/dp_top_short_smoke/failure/episode0_observation_images_left_rgb.mp4`
  - `/root/gpufree-data/lehome-outputs/eval/dp_top_short_smoke/failure/episode0_observation_images_right_rgb.mp4`

当前 smoke 评测里，DP 已确认至少完成了首个 episode 并写出日志：

- `Return=102.25`
- `Length=600`
- `Success=False`

这里保留的是“教学样例输出”，不是完整比赛统计。因为官方 `category eval` 会继续遍历该类别下的 garment 列表，做教程时通常先保留首个 episode 的真实样例就够用了。

## 3.14 比赛强化版：L40 单卡长训配置

如果目标从“先学会跑通”切到“做一个更像比赛提交的基线”，建议不要继续只盯 `top_short`。更合理的做法是直接切到：

- 数据：`four_types_merged`
- 模型：ACT
- 设备：单卡 `L40`

这里先把边界讲清楚：这不等于“保证前三名”。比赛排名最终取决于数据、策略结构、评测细节和训练时间，不能在教程里伪造承诺。但它确实比教学 smoke 更接近一个可拿去继续打榜的起点。

### 3.14.1 为什么这里先推 ACT，而不是先推 DP

原因很实际：

- ACT 在这套数据上更容易先把长训稳定跑起来
- 单卡 L40 下，ACT 更容易控制 batch size 和训练节奏
- 如果要在一小时内把显卡稳定吃满并启动一个长期实验，ACT 的工程风险更低

所以本教程把“比赛强化版第一枪”定为：

- `four_types_merged + ACT + batch_size 64 + 50000 steps`

对应配置文件：

- `/root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml`

### 3.14.2 为什么不是一直把 batch size 往上堆

我在本机对四类联合训练做了三档上探：

- `batch_size=96`：OOM
- `batch_size=88`：OOM
- `batch_size=80`：OOM

最后落回 `batch_size=64`，这是为了保证：

- 能稳定启动长训
- 不在前几步直接炸掉
- 还能维持很高的 GPU 占用

也就是说，这里追求的是“稳定跑完整个长训”的最强可用点，而不是只追第一步把显存堆满。

### 3.14.3 当前正在跑的长训

当前本机已经启动：

- 训练日志：
  - `/root/gpufree-data/lehome-outputs/train/act_four_types_l40.log`
- GPU 采样：
  - `/root/gpufree-data/lehome-outputs/monitor/act_four_types_l40_gpu.csv`
- 训练曲线图：
  - `/root/gpufree-data/lehome-outputs/plots/act_four_types_l40_live/train_metrics.png`
- GPU 曲线图：
  - `/root/gpufree-data/lehome-outputs/plots/act_four_types_l40_live/gpu_metrics.png`

训练命令等价于：

```bash
cd /root/lehome-challenge
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --config_path /root/gpufree-data/every-embodied/15-Challenge竞赛/LeHome/resources/configs/train_act_competition_l40.yaml \
  2>&1 | tee /root/gpufree-data/lehome-outputs/train/act_four_types_l40.log
```

### 3.14.4 当前已经看到的训练信号

长训刚起步时，本机已经出现下面这些变化：

- step `50`：loss `12.334`
- step `100`：loss `3.554`
- step `150`：loss `2.940`

这说明至少在训练早期，loss 下降是明显的，不是卡死或者纯噪声震荡。

GPU 采样也已经表明这轮长训确实把 L40 吃得很满：

- 显存占用：约 `45.1 GiB / 46.1 GiB`
- GPU Utilization：长期接近 `100%`
- 功耗：大约 `280W` 到 `300W`

从工程角度，这说明当前这份配置已经达到了“稳定且接近吃满单卡”的目标。

### 3.14.5 教程里建议展示的图

下面两张图就是更适合放进教程正文的版本，标题全部用英文，方便读者直接看图学判断。

训练曲线：

> **训练曲线图待补充**：原图片托管于作者本地服务器（`/root/gpufree-data/...`），对读者不可见。

GPU 曲线：

> **GPU 监控图待补充**：原图片托管于作者本地服务器（`/root/gpufree-data/...`），对读者不可见。

### 3.14.6 这个长训为什么定在 50000 steps

这里故意没有写成无限拉长，原因是：

- 比赛训练不是“越久越一定更好”
- 在很多视觉策略里，前中期下降最快，后期会进入边际收益变小的区间
- 对教程和服务器资源来说，`50000` 已经属于“足够长、能看到明显趋势、但还算折中”的范围

所以这份配置更适合作为：

- 第一轮正式 baseline
- 第一轮打榜前的主实验
- 后续再决定是否延长到 `70000` 或 `90000` 的参考起点

如果长训到中后期发现：

- loss 已明显平台化
- 中间 checkpoint 的 eval success rate 不再涨

那就不一定值得无限继续烧卡。

---

## 3.15 公开高分方案参考与复盘

本节整理的是 2026-05-13 可以公开查到的 LeHome Challenge 相关方案。这里先把结论说清楚：**官方榜单前 1 到 3 名没有看到明确开源的完整方案**。已经公开的内容里，最值得大家参考的是接近前列的队伍复盘、VLA 改进仓库，以及若干可对照的参赛仓库。

这些内容不建议直接当成“复制就能进前三”的配方。更合理的用法是：先用前面章节跑通 `ACT/DP + state + RGB` baseline，再按本节的方法逐项增强。

### 3.15.1 官方榜单当前前列情况

官网 Leaderboard 由接口动态加载，页面源码里对应接口是：

- `https://lightwheel.ai/lwapi/open/lehome/ranking`
- 请求方式：`POST`
- 请求体：`{}`

截至 2026-05-13，该接口返回的仿真赛前 8 名如下：

| 排名 | Registration ID | Team | Long-Sleeved Top | Short-Sleeved Top | Long Pant | Short Pant | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `r84` | `ilya` | 74.5% | 70.0% | 80.5% | 93.5% | 79.63% |
| 2 | `r196` | `Shubham @ Vorwerk` | 73.0% | 62.5% | 71.5% | 87.0% | 73.50% |
| 3 | `r55` | `Dum-E` | 76.5% | 62.0% | 75.5% | 79.5% | 73.38% |
| 4 | `r161` | `SCUT-Unlimited` | 65.5% | 66.0% | 70.0% | 91.0% | 73.13% |
| 5 | `r201` | `GraspYesAI` | 73.5% | 61.0% | 69.0% | 79.0% | 70.63% |
| 6 | `r162` | `sZs` | 70.5% | 64.0% | 68.5% | 75.5% | 69.63% |
| 7 | `r218` | `ClothFolder50k` | 77.0% | 56.0% | 58.5% | 82.5% | 68.50% |
| 8 | `r13` | `sisigakgak` | 68.0% | 52.5% | 64.0% | 77.5% | 65.50% |

大家需要注意两点：

- 前 8 名并不等于全部开源。排行榜只给了队伍名和成功率，没有给方法、代码或 checkpoint。
- 公开仓库和榜单队伍名不一定一一对应。除非仓库 README 或队伍说明明确写了对应关系，否则本教程只把它们当作“公开参考方案”。

### 3.15.2 方案一：UCAS 复盘，分类器 + 专家 ACT + PI0.5

参考链接：

- 参赛仓库：[wangerforcs/lehome-challenge-ucas](https://github.com/wangerforcs/lehome-challenge-ucas)
- 中文复盘：[wangerforcs/EILearn/competition/lehome.md](https://github.com/wangerforcs/EILearn/blob/master/competition/lehome.md)

这个复盘是目前最值得大家细读的公开资料之一。它不是一个只给命令的 README，而是比较完整地记录了从 baseline 到比赛后期的尝试路径。

第一阶段是 `classifier + ACT`。核心想法是：评测时官方不会直接告诉你当前 garment 属于哪一类，因此如果想为四类衣物分别训练专家策略，就需要先判断类别。做法大致是：

1. 用 ResNet 之类的视觉分类器识别当前衣物类别。
2. 对 `top_long`、`top_short`、`pant_long`、`pant_short` 分别训练 ACT。
3. 推理时先分类，再路由到对应的 ACT 专家模型。

这个路线的优点是简单、清晰，容易在官方 LeRobot 框架里实现。缺点也很明显：分类错误会直接把样本送到错误专家；如果某个类别数据质量差，单独专家也会被带偏。复盘里提到，这个阶段整体提交后大约在 42% 左右，说明它能跑通，但距离 60% 以上还有差距。

第二阶段主要切到 `PI0.5`。他们保留了和 ACT 类似的输入输出形式：

- 输入：`observation.state`
- 输入：`observation.images.top_rgb`
- 输入：`observation.images.left_rgb`
- 输入：`observation.images.right_rgb`
- 输出：12 维 `action`

这里没有把 depth 当成第一优先级，原因和本教程前面的建议一致：如果预训练模型本身不是围绕 depth 训练的，直接加 depth 会增加工程复杂度，但不一定带来稳定收益。

第三阶段开始做更细的数据和类别处理。复盘里特别提到 `top_short` 和 `pant_short` 是主要瓶颈：有些数据 replay 出来的动作本身不稳定，有些成功/失败样本保存逻辑也会引入脏数据。这个经验非常重要：LeHome 不是只靠换模型就能解决的任务，数据质量经常比模型名字更关键。

对大家复刻的启发：

- 如果只想快速超过普通 baseline，可以先做“四类专家模型 + 类别识别器”。
- 如果发现某一类明显拖后腿，不要只看总分，要单独看四个类别的成功率。
- 对 `top_short`、`pant_short` 这类难点，优先检查数据 replay 和失败样本，而不是盲目加大模型。
- 如果训练资源有限，先把一个类别训到可复现的成功率，再扩到四类。

### 3.15.3 方案二：SPGVLA，给 VLA 增加进度引导和世界模型监督

参考链接：

- 代码仓库：[blackcat0615/spgvla](https://github.com/blackcat0615/spgvla)
- 模型链接 1：[spgvla](https://huggingface.co/blackcat0615/spgvla)
- 模型链接 2：[spgvla0.7](https://huggingface.co/blackcat0615/spgvla0.7)

`SPGVLA` 的全称是 `Simple Progress Guidance For Vision Language Action Model`。它的出发点是：衣物折叠是长时序任务，模型很容易不知道自己已经处在“抓取、拉平、折叠、收尾”中的哪一步，于是出现状态混淆。

这个仓库做了两个增强：

1. `SPG`：Simple Progress Guidance。给模型额外提供任务进度相关的信息，帮助 VLA 判断当前处于任务的哪个阶段。
2. `WM`：world model module。用世界模型提供更密集的监督信号，缓解 VLA 训练时只有稀疏行为克隆信号的问题。

仓库 README 里给出的公开实验结果如下：

| 实验设置 | top long | top short | pants long | pants short | mean SR |
| --- | --- | --- | --- | --- | --- |
| baseline SmolVLA | 61.67% | 10.00% | 31.67% | 76.67% | 45.00% |
| baseline + SPG | 55.00% | 21.67% | 45.00% | 80.00% | 50.40% |
| baseline + SPG + bs64 | 63.33% | 25.00% | 33.33% | 88.33% | 52.50% |
| baseline + SPG + bs64 + WM | 70.00% | 25.00% | 45.00% | 86.67% | 56.67% |
| baseline + SPG + bs96 + WM + data aug retrain | 73.30% | 45.00% | 58.33% | 85.00% | 65.40% |

这个结果对大家最有用的地方不是“照抄模块名”，而是它展示了一个清晰趋势：单纯换 VLA baseline 不一定够，进度信号、世界模型辅助监督、batch size 和数据增强叠加后，才从 45% 拉到 65.4%。

复刻时建议这样理解：

- `SPG` 适合解决长时序阶段混淆，例如袖子还没拉平就开始折。
- `WM` 适合补充中间状态监督，让模型不只学习最终动作标签。
- `data_aug_retrain` 对 LeHome 很关键，因为官方每类数据并不大，视觉分布稍微变化就会影响成功率。
- `top_short` 是明显短板，即使增强后也只有 45%，说明公开方案里它依然难。

如果大家已经跑通本教程的 `four_types_merged + ACT`，下一步可以参考 SPGVLA 的思想，不一定马上切完整 VLA，而是先做两个轻量实验：

1. 给 ACT/SmolVLA 加类别或阶段条件，例如 `garment_type`、`stage_id`、进度比例。
2. 对每类数据做保守图像增强，观察 `top_short` 是否提升。

### 3.15.4 方案三：LaundryNauts，VLA 微调式参赛仓库

参考链接：

- [cwoodhayes/lehome-laundrynauts](https://github.com/cwoodhayes/lehome-laundrynauts)

这个仓库描述为 `fine-tuned VLA for bimanual garment folding`，目录结构基本是在官方 LeHome 仓库上扩展，包含 `configs`、`docker_policy`、`scripts`、`source/lehome` 等内容。它和官网榜单里的 `LaundryNauts` 名字能对应上，但榜单平均分是 40.00%，所以它更适合作为“VLA 工程组织方式”的参考，而不是高分策略模板。

大家可以重点看三类内容：

- 它如何在官方仓库结构里组织自定义 policy。
- 它如何准备 Docker/submission 相关文件。
- 它如何把 VLA 训练和官方评测脚本接起来。

这类仓库的价值在于工程参考：当大家自己的方法已经从 `lerobot-train` 走向自定义模型、Docker 提交、远端评测时，可以对照它检查目录和脚本是否完整。

### 3.15.5 方案四：S.N.N Neural Lab，公开但非前列的对照仓库

参考链接：

- [alifestone/lehome-challenge_S.N.N](https://github.com/alifestone/lehome-challenge_S.N.N)

该仓库和官网榜单里的 `S.N.N Neural Lab` 基本能对上，榜单平均分是 40.38%。它不是高分方案，但适合作为对照材料：同样是基于官方环境做参赛工程，最后分数可能仍然停留在 40% 左右。

对教程来说，这类仓库提醒大家：LeHome 的主要难点不是“能不能启动训练”，而是：

- 数据质量是否足够干净。
- 四类衣物是否分别优化。
- 推理时是否能处理随机 garment 类别。
- 模型是否理解长时序进度。
- 评测脚本、checkpoint 路径、Docker 提交是否严格一致。

### 3.15.6 从公开方案提炼出的实用路线

如果大家已经完成前面章节的 smoke test，本教程建议按下面顺序继续：

1. **先做稳定 baseline**
   使用 `dataset_challenge_merged`，输入保持 `state + top/left/right RGB`，输出保持 12 维 joint action。先跑通 `ACT`，再考虑 `DP` 或 `SmolVLA`。

2. **看四类单独分数，不只看平均分**
   官方榜单里很多队伍都是 `short pant` 很高、`top_short` 偏低。平均分掩盖了短板，大家要单独记录四类成功率。

3. **尝试专家模型和类别分类器**
   如果四类差异明显，可以为四类衣物分别训练专家策略，再用分类器路由。这个方法工程简单，但要防止分类错误。

4. **清洗和补强困难类别数据**
   公开复盘里反复提到数据问题。大家在训练前最好 replay 一部分数据，尤其是 `top_short` 和 `pant_short`，把明显失败或动作偏移严重的 episode 单独标出来。

5. **再上 VLA 或进度增强**
   如果 baseline 已经稳定，再参考 `SPGVLA` 增加进度引导、世界模型辅助监督或数据增强。不要在环境还没跑通时直接堆复杂模型。

6. **提交前固定评测和提交链路**
   高分方案最终也要能在官方评测里跑起来。自定义 policy、Docker、checkpoint 路径、依赖版本要单独验证，不能只看本地训练 loss。

### 3.15.7 参考链接汇总

- LeHome 官网：[https://lehome-challenge.com/](https://lehome-challenge.com/)
- 官方仓库：[lehome-official/lehome-challenge](https://github.com/lehome-official/lehome-challenge)
- 官方资产数据：[lehome/asset_challenge](https://huggingface.co/datasets/lehome/asset_challenge)
- 官方合并训练数据：[lehome/dataset_challenge_merged](https://huggingface.co/datasets/lehome/dataset_challenge_merged)
- UCAS 参赛仓库：[wangerforcs/lehome-challenge-ucas](https://github.com/wangerforcs/lehome-challenge-ucas)
- UCAS 中文复盘：[wangerforcs/EILearn/competition/lehome.md](https://github.com/wangerforcs/EILearn/blob/master/competition/lehome.md)
- SPGVLA 仓库：[blackcat0615/spgvla](https://github.com/blackcat0615/spgvla)
- SPGVLA 模型：[blackcat0615/spgvla](https://huggingface.co/blackcat0615/spgvla)
- SPGVLA 0.7 模型：[blackcat0615/spgvla0.7](https://huggingface.co/blackcat0615/spgvla0.7)
- LaundryNauts 仓库：[cwoodhayes/lehome-laundrynauts](https://github.com/cwoodhayes/lehome-laundrynauts)
- S.N.N Neural Lab 仓库：[alifestone/lehome-challenge_S.N.N](https://github.com/alifestone/lehome-challenge_S.N.N)

---

## 4. 推荐训练策略

### 4.1 baseline 配置

建议先从下面这套开始：

- policy: `ACT`
- 输入：`observation.state + top/left/right RGB`
- 输出：`action`
- device: 训练时用 `cuda`
- 评测时 env 用 `cpu`

### 4.2 为什么不建议一上来就上 depth

- 深度会增大数据量和训练复杂度
- baseline 先跑通更重要
- 官方文档明确说 `state + RGB` 已验证可用
- 如果要加 depth，建议在 baseline 收敛后做对照实验

### 4.3 是否建议用 EE pose

不建议作为首选。

官方文档也明确提醒：

- `observation.ee_pose`
- `action.ee_pose`

由于 IK 和硬件限制，稳定性不如 joint-space control。比赛 baseline 更建议用：

- `observation.state`
- `action`

---

## 5. 清理与仓库控制

如果你要把这个教程整理进自己的大仓库，只建议保留这些轻量内容：

- 教程文档
- 训练配置样例
- 评测命令
- 少量日志片段
- 少量截图或短视频链接
- 必要的 patch / overlay 说明

不建议提交这些大资源：

- `Assets/`
- `Datasets/`
- `outputs/`
- `logs/`
- `models/`
- `videos/`
- `plots/`
- `.cache/`
- 任何大于几十 MB 的权重和素材

原因很简单：这些内容都应该通过下载命令获取，不应该进教程子仓库。

### 5.1 之前失败的评测结果

如果只是教程清理，之前那批 `0%` 成功率的失败视频可以删除。

当前需要删除的目录是：

```text
/root/gpufree-data/lehome-eval-outputs/
```

后续统一改用：

```text
/root/gpufree-data/lehome-outputs/eval/
```

---

## 6. 建议的仓库组织方式

建议在 `every-embodied` 中新增如下结构：

```text
15-Challenge竞赛/
└── LeHome/
    ├── README.md
    ├── .gitignore
    └── resources/
        ├── commands.md
        ├── configs/
        └── scripts/
```

如果后续你还会增加其他比赛，也可以沿用同样的结构。

---

## 7. 可直接复用的检查命令

## 7.1 检查 GPU

```bash
nvidia-smi
```

## 7.2 检查核心包

```bash
python - <<'PY'
import isaacsim, lerobot
print("isaacsim ok")
print("lerobot", getattr(lerobot, "__version__", "unknown"))
PY
```

## 7.3 检查官方资产是否已下载

```bash
ls Assets
ls Datasets/example
```

## 7.4 检查单件服装评测列表

```bash
cat Assets/objects/Challenge_Garment/Release/Release_test_list.txt
```

---

## 8. 镜像发布建议

如果你要宣传这个镜像，建议在说明里强调：

- 已预装官方 LeHome 环境
- 已验证可跑通训练与评测
- 建议首次使用时下载 `asset_challenge` + `dataset_challenge_merged`
- 已适配 `L40` / 服务器场景
- 提供 ACT/DP baseline 启动命令
- 默认不把比赛大资源打进教程仓库

可直接对外描述为：

> 一个面向 LeHome Challenge 的开箱即用镜像，预装 Isaac Sim / LeRobot / LeHome 运行环境，提供训练、评测、数据下载和问题复现的最小闭环。

---

## 9. 本地验证结果

本地已经验证：

- ACT 单 garment 评测可跑通，结果 `Success Rate = 0.00%`
- DP 单 garment 评测可跑通，且 `policy_num_inference_steps=1` 能生效
- 评测视频可以成功导出

这说明：

- 镜像可用
- 训练/评测链路可用
- 教程可以基于这套镜像直接编写和发布


## 备注

安装图片和视频查看器

```bash
sudo apt install celluloid

sudo apt install viewnior

# 将常见图片格式的默认打开程序设为 Viewnior
xdg-mime default viewnior.desktop image/jpeg image/png image/gif image/bmp image/webp

```
