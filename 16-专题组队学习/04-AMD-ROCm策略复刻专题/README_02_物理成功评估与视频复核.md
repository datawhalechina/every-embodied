# 02 物理成功评估与视频复核

本任务解决一个核心问题：日志显示成功时，策略是否真的完成了抓取。这里把环境原始成功率和物理成功率分开统计，并用视频复核典型成功和典型失败。

配套实操 Notebook：[02_physical_success_review.ipynb](./notebooks/02_physical_success_review.ipynb)。

## 为什么不能只看环境 success

在抓杯子放盘子的任务中，原始 `check_success()` 往往更接近几何条件。策略可能没有稳定夹起杯子，只是把杯子推到盘子附近；也可能杯子已经倒下，但位置满足了终止条件。

因此，本专题推荐报告两个指标：

| 指标 | 含义 |
| --- | --- |
| `legacy_success` | 环境原始成功条件 |
| `physical_success` | 目标杯被抬起、放到盘上且最终姿态基本直立 |

当两者不一致时，以视频和物体状态为准。

## 推荐物理口径

建议的最小物理口径：

1. legacy success 为真；
2. 目标杯相对初始高度至少抬起 `0.03 m`；
3. 抬起状态持续至少若干控制 tick；
4. 终态杯子没有明显倒下。

这套指标不要求阈值永远不变。更重要的是：同一组模型对比时必须使用同一口径。

## 批量评估脚本形态

建议把批量评估做成 Python 脚本，而不是手动在 Notebook 中重复运行。脚本入口可以设计成下面这种形态：

```bash
python tools/audit_language_policy_physical.py \
  --policy-type smolvla \
  --policy-path "$MODEL_ROOT/checkpoints/000500/pretrained_model" \
  --instruction "Place the red mug on the plate." \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-action-steps 600 \
  --output-jsonl outputs/eval_red.jsonl \
  --summary-json outputs/summary_red.json
```

脚本输出建议包含：

- seed；
- instruction；
- legacy success；
- physical success；
- first success step；
- 最大抬升高度；
- 终态杯子到盘子的 xy 距离；
- 终态 upright 指标；
- 失败原因桶。

## 视频复核

每个 checkpoint 至少录两类视频：

1. 一个真实成功视频；
2. 一个典型失败视频。

视频旁边要写清楚它属于哪一种证据：

```html
<video controls muted preload="metadata" width="100%">
  <source src="assets/smolvla_weighted500_blue_success_seed0.mp4" type="video/mp4">
</video>

[直接打开或下载蓝杯成功视频](./assets/smolvla_weighted500_blue_success_seed0.mp4)
```

图注示例：

> 图 1：SmolVLA 在蓝杯任务上的真实成功 rollout。复核重点是杯子是否被夹起、是否被放到盘上，以及终态是否保持直立。

不要只放成功视频。失败视频更适合教学，因为它能解释为什么需要更严格的评估口径。

## 分支内可直接查看的真实视频

下面的媒体文件已经随轻量分支提交，不依赖 `/path/to/outputs` 或 Notebook 当前工作目录。复核时不要只看最后一帧，而要沿着时间轴观察是否出现了稳定夹取、抬升、搬运和释放。

### SmolVLA：真实失败参考

<video controls muted preload="metadata" width="100%">
  <source src="assets/smolvla_weighted500_red_failure_seed8.mp4" type="video/mp4">
</video>

[直接打开或下载 SmolVLA 红杯失败视频](./assets/smolvla_weighted500_red_failure_seed8.mp4)

这里使用的是分支中实际提交的红杯失败 rollout，作为失败阶段的可播放参考；它不是缺失的蓝杯 baseline 原始视频。原始蓝杯 baseline 和 ACT rollout 视频依赖完整实验输出，没有被放进轻量分支。

### SmolVLA：蓝杯成功

<video controls muted preload="metadata" width="100%">
  <source src="assets/smolvla_weighted500_blue_success_seed0.mp4" type="video/mp4">
</video>

[直接打开或下载 SmolVLA 蓝杯成功视频](./assets/smolvla_weighted500_blue_success_seed0.mp4)

这条视频可用于检查蓝杯指令下的接触、夹取、搬运和释放是否完整。`smolvla_blue_success_sequence.jpg` 由这条已经提交的 MP4 生成，不再使用不可用视频占位图。

### 四视角严格成功回放

<video controls muted preload="metadata" width="100%">
  <source src="assets/pnp_four_view_strict_success.mp4" type="video/mp4">
</video>

[直接打开或下载四视角严格成功视频](./assets/pnp_four_view_strict_success.mp4)

四视角回放同时显示 Agent、Egocentric、Top、Side 画面，适合检查单一相机视角中容易漏掉的接触、抬升和终态姿态证据。

### ACT：当前分支提供曲线，不伪造 rollout

![ACT DAgger 进展曲线](./assets/act_dagger_progress_curve.png)

ACT 的原始失败/成功 rollout 视频没有随本轻量分支提交，因此不再放置“Source video is unavailable”关键帧占位图。拿到自己的 `OUTPUT_ROOT` 后，按下面命令从真实 MP4 重新生成关键帧：

```bash
python code/generate_tutorial_assets.py --source-root "$OUTPUT_ROOT"
```

只有当 `OUTPUT_ROOT` 中包含 `act_reset_oracle_v1/dagger_best025_representative_videos/seed1088.mp4` 和 `seed1089.mp4` 等原始文件时，脚本才会生成对应 ACT 预览。

## Checkpoint

完成本任务后，保留这些证据：

- 红杯和蓝杯各一份 JSONL 评估文件；
- 一份 summary JSON 或 Markdown 表；
- 至少 1 个成功视频和 1 个失败视频；
- 对失败原因的简短分类。
