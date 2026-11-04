# MorphMimic：跨人形机器人共享世界模型与动作策略

MorphMimic 使用一套共享策略，让 **Unitree G1、AgiBot X2 Ultra 和 Booster T2** 跟踪各自的动作参考；模型通过机器人形态参数区分不同身体。本章给出浏览器体验、预训练策略仿真、单机器人训练和三机器人联合训练路线。

- [官方项目与演示](https://lzyang2000.github.io/morphmimic/) · [浏览器交互演示](https://lzyang2000.github.io/morphmimic/demo/) · [代码](https://github.com/lzyang2000/morphmimic) · [论文](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf) · [v0.1 预训练策略](https://github.com/lzyang2000/morphmimic/releases/tag/v0.1)
- 本章按官方代码 [`770394b4`](https://github.com/lzyang2000/morphmimic/tree/770394b4a0cad9effbf807a6c03435d1699733d0) 编写。代码采用 Apache-2.0；机器人模型、依赖与 BONES-SEED 数据各自使用其许可证。
- 文中命令根据上游源码和说明整理；本教程尚未在本机执行训练、仿真或真机部署。作者的展示和结果均标明为作者实验。

## 1. 一套策略如何适配三种身体

MorphMimic 基于作者对 [SuperMimic](https://supermimic.github.io/) 的独立重实现：通过学习到的可微动力学模型展开短轨迹，对动作跟踪策略反向传播。它进一步让多个不同机器人共享世界模型、策略或二者。

```text
G1 / X2 Ultra / T2
   → 按运动学功能映射到 G1 的 29 关节顺序和 14 个跟踪刚体
   → 本机器人独立的归一化、参考动作和回放数据
   → 125 维形态描述（驱动、几何、质量）
   → 共享世界模型与策略，输出对应机器人的动作
```

策略的运动编码器将参考动作编码为 **32 维潜变量**。形态描述输入策略与世界模型，并通过 FiLM 调节策略解码器；对应动作帧的潜变量向跨机器人的均值对齐，人体编码器再学习与动作编码器产生一致的表示。`motion` 接收完整机器人参考动作；`human` 接收未来 10 帧人体骨架，可选 24 关节 `full` 或骨盆、双腕和双踝 `five`。PICO 身体跟踪可直接送入人体接口。

默认目标采用与全局位置无关的表示，策略依靠 IMU 和关节编码器运行。若需要修正相对参考路径的漂移，可研究 `--goal-anchor robot`；该模式需要机器人世界位置与朝向，当前上游 G1 硬件节点尚未接入位置来源。[方法与实现](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/README.md#anchor-mode)

## 2. 先看作者演示和结果

打开[浏览器交互演示](https://lzyang2000.github.io/morphmimic/demo/)，等待约 100 MB 资源加载；页面使用 WebGL，可切换 `motion` / `human` 输入并拖动机器人施加扰动。项目页还提供共享策略在三机器人上行走、舞蹈和跳跃，以及 G1 真机的[视频](https://lzyang2000.github.io/morphmimic/)。

论文对 **128,786 个三机器人共有动作片段**逐条从首帧运行确定性策略，统计完整播放比例。作者报告的共享策略结果如下：

| 接口 | G1 | X2 Ultra | T2 |
| --- | ---: | ---: | ---: |
| 完整动作 `motion`，片段完成率 | 99.747% | 99.891% | 99.886% |
| 五点输入，片段完成率 | 99.729% | 99.880% | 99.852% |

这些是论文的 **40,000 次迭代**结果；v0.1 发布权重来自后来采用新配方训练的 **60,000 次迭代**版本。单 G1 训练策略直接迁移到其他机器人时效果明显下降，说明统一坐标和形态描述需要配合多本体训练。[作者结果表](https://lzyang2000.github.io/morphmimic/#results)

## 3. 环境与数据准备

训练与首次导出仿真资产使用 Linux、NVIDIA GPU、驱动和 `uv`；上游锁定 Python 3.11、mjlab 1.6.0 和 CUDA 12.8 的 PyTorch，并说明 575 或更新的驱动可用。仿真资产导出完成后，ONNX 策略与 MuJoCo 仿真节点可在 CPU 上运行。三机器人联合训练使用同机三张 GPU，每机器人一个进程。完整数据约需每进程 120–130 GB 主机内存；4096 环境时，1 小时子集约需每 GPU 28 GB 显存，完整集约需 51–58 GB。

以下示例使用 Linux shell。将数据、源码和结果放在仓库外的大容量磁盘；`$DATA`、`$WORK` 是示例目录。依赖与数据下载按机器现有下载专线执行；实验室 Ubuntu 的 `lab-download` 入口只包住资源下载和依赖准备，训练与评测直接使用预先准备的环境。

```bash
export WORK=/path/to/morphmimic-work
export DATA=/path/to/robot-data
mkdir -p "$WORK" "$DATA"
cd "$WORK"
git clone https://github.com/lzyang2000/morphmimic.git
cd morphmimic
git checkout 770394b4a0cad9effbf807a6c03435d1699733d0
git submodule update --init --recursive

# 实验室 Ubuntu：使用现有专线准备依赖
lab-download uv sync --group export --group teleop
```

训练使用 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)。先在数据页登录并接受许可；获取 G1 CSV、SOMA Uniform BVH，以及所需的 `seed_metadata_v003.csv` / `seed_metadata_v004.csv`。数据页说明动作资源可能以 tar 包提供，下载并解压后检查以下路径；文件实际位于 `metadata/` 时，在命令中使用该实际路径：

```text
$DATA/seed/g1/csv/<session>/*.csv
$DATA/seed/soma_uniform/bvh/<session>/*.bvh
$DATA/seed/metadata/seed_metadata_v003.csv
$DATA/seed/metadata/seed_metadata_v004.csv
```

上游 G1 转换需要 v003，X2/T2 重定向与转换示例需要 v004；如果所获数据版本没有相应文件，应核对[数据页](https://huggingface.co/datasets/bones-studio/seed/tree/main)与上游要求。下面设置变量集中管理路径：

```bash
export META3="$DATA/seed/metadata/seed_metadata_v003.csv"
export META4="$DATA/seed/metadata/seed_metadata_v004.csv"
test -f "$META3" && test -d "$DATA/seed/g1/csv"
```

## 4. 预训练策略：先跑 G1 仿真

### 4.1 准备 50 Hz 动作与仿真资产

```bash
uv run --no-sync python -m morphmimic.scripts.convert_seed \
  --csv-dir "$DATA/seed/g1/csv" --metadata "$META3" \
  --out-dir "$DATA/seed_g1_npz50" --workers 16
uv run --no-sync python -m morphmimic.scripts.build_subsets \
  --manifest "$DATA/seed_g1_npz50/manifest.csv" \
  --out-dir "$DATA/seed_g1_npz50/subsets"
```

输出包括 `manifest.csv` 与 `subsets/tiny.yaml`、`1h.yaml`、`7h.yaml`、`40h.yaml`、`all.yaml`。作者以 96 workers 转换完整 G1 数据，输出约 91 GB；上例的 16 workers 可按机器 CPU 调整。

官方示例的动作名为 `walk_ff_loop_315_R_001__A257`。先确认它在 `1h.yaml` 中，再导出。资产导出需要 GPU：

```bash
grep -F 'walk_ff_loop_315_R_001__A257' "$DATA/seed_g1_npz50/subsets/1h.yaml"
uv run --no-sync python -m deploy.sim2sim.export_assets --robot g1_amber \
  --subset-yaml "$DATA/seed_g1_npz50/subsets/1h.yaml" \
  --clips walk_ff_loop_315_R_001__A257 --out assets
```

若动作名不在子集中，从 YAML 选一个实际存在的片段，并将后续两个命令中的名称同步替换。导出目录包含机器人模型与所选参考动作；其他机器人要用各自重定向后的动作子集和相应 `--robot` 导出。

### 4.2 获取权重并运行

```bash
lab-download curl --fail --location \
  --output morphmimic_policy.zip \
  --url 'https://github.com/lzyang2000/morphmimic/releases/download/v0.1/morphmimic_policy.zip'
unzip morphmimic_policy.zip
(cd morphmimic_policy && sha256sum -c SHA256SUMS)
export W="$PWD/morphmimic_policy"

uv run --no-sync --group export python -m deploy.run \
  --deploy "$W" --assets assets --robot g1_amber \
  --clip walk_ff_loop_315_R_001__A257 --interface motion --viewer native
```

运行结束查看片段是否完成、关节和身体跟踪误差及漂移。`policy.onnx` 是共享策略；每个机器人的 JSON 还保存各自归一化、形态向量及常量。发布包中的 G1 配置名为 `g1_amber`，是作者针对腰部电机较弱的 G1 调整阻尼的配置；因此资产导出和运行命令都使用 `g1_amber`，并复用 G1 动作子集。无显示器的机器可依上游用 `MUJOCO_GL=egl` 和 `--video out.mp4` 输出视频。[官方部署步骤](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/README.md#deploy)

## 5. 单机器人训练：先验证训练链路

先只训练 G1 的 `motion` 接口，用 1 小时子集检查日志与 checkpoint。以下 100 次迭代是教程自定义的链路检查，不能与论文训练结果直接比较：

```bash
G1="$DATA/seed_g1_npz50/subsets/1h.yaml"
uv run --no-sync python -m morphmimic.scripts.train \
  --robot g1 --subset-yaml "$G1" --gpu 0 \
  --iterations 100 --ckpt-every 100 --log-dir logs --run-name g1_smoke
```

默认仍为 4096 个并行环境；上游 1 小时子集的约 28 GB 显存数字可用于选卡。训练首次编译 CUDA 图可能需要几分钟；`--compile none` 使用普通执行。检查 `logs/g1_smoke/` 的训练记录和 `ckpt_100.pt` / `latest.pt`；续训用相同配置加 `--resume logs/g1_smoke/latest.pt`。若机器显存不足，先通过 `uv run --no-sync python -m morphmimic.scripts.train --help` 查看可调整的环境数，记录修改后的配置。

人体接口还需将 SOMA Uniform BVH 转为 24 关节的 `human50` sidecar：

```bash
uv run --no-sync python -m morphmimic.scripts.convert_human \
  --npz-root "$DATA/seed_g1_npz50" \
  --bvh-root "$DATA/seed/soma_uniform/bvh" --workers 16
uv run --no-sync python -m morphmimic.scripts.check_human --subset-yaml "$G1"
```

检查输出中的缺失或不匹配片段。三个机器人的同名动作共享 G1 生成的人体 sidecar。

## 6. 三机器人联合训练与消融

SEED 提供 G1 动作；X2 与 T2 需要先从 SOMA Uniform BVH 重定向。上游使用 [NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)，固定版本 `cbdb2b4`。先按其仓库说明安装和准备网格，然后在 MorphMimic 目录中运行：

```bash
export RETARGETER=/path/to/soma-retargeter
uv run --no-sync python tools/soma_retargeter/stage_subset.py \
  --subset-yaml "$DATA/seed_g1_npz50/subsets/all.yaml" \
  --metadata "$META4" --bvh-root "$DATA/seed/soma_uniform/bvh" \
  --out "$WORK/seed_stage/all"
uv run --no-sync python tools/soma_retargeter/assemble_robots.py \
  --retargeter "$RETARGETER" --out "$WORK/soma_robots"
```

将 `tools/soma_retargeter/batch_x2.json` 与 `batch_t2.json` 中 `__IMPORT__`、`__EXPORT__`、`__ROBOTS__` 替换为上面 staging 目录、各机器人 CSV 输出目录、机器人配置目录的绝对路径；分别运行 Retargeter 的 `app/bvh_to_csv_converter.py --config <配置文件>`。完整数据可用上游 `shard_stage.py` / `launch_shards.sh` 分片，以 `validate_csv.py` 检查输出。具体 Retargeter 环境、配置字段及其模型资产见[上游工具](https://github.com/lzyang2000/morphmimic/tree/770394b4a0cad9effbf807a6c03435d1699733d0/tools/soma_retargeter)和 [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)。

重定向完成后，X2 与 T2 的子集采用 G1 的相同片段列表：

```bash
for r in x2 t2; do
  uv run --no-sync python -m morphmimic.scripts.convert_seed --robot "$r" \
    --csv-dir "$DATA/seed_${r}_csv" --metadata "$META4" \
    --out-dir "$DATA/seed_${r}_npz50" --workers 16
  uv run --no-sync python -m morphmimic.scripts.build_subsets \
    --manifest "$DATA/seed_${r}_npz50/manifest.csv" \
    --out-dir "$DATA/seed_${r}_npz50/subsets" \
    --clips-from "$DATA/seed_g1_npz50/subsets/all.yaml"
done
```

采用上游当前 60,000 次迭代的联合训练配方：

```bash
G1="$DATA/seed_g1_npz50/subsets/all.yaml"
X2="$DATA/seed_x2_npz50/subsets/all.yaml"
T2="$DATA/seed_t2_npz50/subsets/all.yaml"
uv run --no-sync python -m morphmimic.scripts.train \
  --robots g1 x2 t2 --subset-yamls "$G1" "$X2" "$T2" \
  --share all --gpu-ids '[0,1,2]' \
  --kp-interface --kp-student copy --no-kp-preview-mix \
  --human-root "$DATA/seed_g1_npz50/human50" \
  --library-dtype float16 --iterations 60000 --ckpt-every 4000 \
  --log-dir logs --run-name unified
```

`--share all` 同时共享世界模型和策略；保持其他参数一致，改成 `--share wm` 可共享世界模型、策略分开，改成 `--share policy` 则共享策略、世界模型分开。各机器人的检查点在 `logs/unified/<robot>/`，续训使用 `--resume logs/unified`。上游默认每个机器人 4096 环境、32 步想象 rollout；前 500 次迭代只训练世界模型。当前训练默认每回合采样并保持 0–20 ms 驱动延迟。

## 7. 评估、导出和扩展

对自己训练的 checkpoint 做确定性评估，输出逐片段完成情况、跟踪误差和漂移：

```bash
uv run --no-sync python -m morphmimic.scripts.eval \
  --ckpt logs/unified/g1/ckpt_60000.pt \
  --subset-yaml "$G1" --num-envs 1024
uv run --no-sync python -m morphmimic.scripts.eval \
  --ckpt logs/unified/g1/ckpt_60000.pt \
  --subset-yaml "$G1" --interface human --human-points five

uv run --no-sync --group export python -m deploy.export_onnx \
  logs/unified out/unified --iteration 60000
```

导出器先比较 ONNX 与 PyTorch 动作，再写 `policy.onnx` 和机器人配置。第 4 节下载的 v0.1 发布包是已经导出的 ONNX 策略；上述 `eval --ckpt` 需要训练 checkpoint，不能把 ONNX 文件当成 checkpoint 传入。

PICO 4 / 4 Pro 遥操作需要头显、双手控制器、两个踝部追踪器及同网 Linux 电脑；按照[上游遥操作步骤](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/README.md#4-teleop)安装 PICO SDK 与 XRoboToolkit，先运行 streamer，再在仿真端选择 `--teleop --interface human --human-points five`。作者也提供无需头显的录制与回放工具。

作者已在真实 G1 上展示动作跟踪和遥操作；X2 Ultra、T2 的真机结果在项目页仍列为待验证。准备 G1 真机时，先完成仿真，按照[上游真机步骤](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/README.md#5-the-real-g1)确认网络接口、吊挂或扶持和遥控器阻尼操作，再从站立姿态开始。

## 参考资料

- [MorphMimic 项目页与作者演示](https://lzyang2000.github.io/morphmimic/)
- [MorphMimic 固定版本代码与复现说明](https://github.com/lzyang2000/morphmimic/tree/770394b4a0cad9effbf807a6c03435d1699733d0)
- [MorphMimic 论文](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf)
- [SuperMimic 原方法](https://supermimic.github.io/)
- [BONES-SEED 数据集与许可](https://huggingface.co/datasets/bones-studio/seed)
