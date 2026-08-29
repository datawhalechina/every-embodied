# 统一评估、视频与结果归档

本章把不同 benchmark［评测基准］的训练、评估和视频整理成统一验收方法。目标是让读者能从一个结果目录确认：运行的是哪个模型、哪个任务、哪些 seed［随机种子］、成功分母是多少，以及视频是否属于同一回合。

## 1. 评估前检查

```bash
python - <<'PY'
import json
import os
import torch

info = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "mujoco_gl": os.environ.get("MUJOCO_GL"),
}
print(json.dumps(info, indent=2))
PY
```

同时记录上游 Git commit［代码提交］、模型目录、归一化统计、任务配置和相机列表。

## 2. 单回合结果格式

```json
{
  "benchmark": "DexJoCo",
  "model": "pi05_dexjoco_multi_task",
  "task": "bimanual_hanoi",
  "seed": 0,
  "episode": 0,
  "success": true,
  "steps": 476,
  "elapsed_s": 47.6,
  "video": "videos/bimanual_hanoi_seed0.mp4"
}
```

不同项目可以增加阶段字段，但 `task`、`seed`、`success`、`steps` 和 `video` 保持统一。

## 3. 聚合成功率

```python
from collections import defaultdict
import json
from pathlib import Path

rows = [json.loads(path.read_text()) for path in Path("eval").glob("**/episode.json")]
by_task = defaultdict(list)
for row in rows:
    by_task[row["task"]].append(bool(row["success"]))

summary = {
    task: {
        "successes": sum(values),
        "episodes": len(values),
        "success_rate": sum(values) / len(values),
    }
    for task, values in sorted(by_task.items())
}
Path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```

不要从文件名中的 `success` 文本反推结果；优先读取环境成功条件或官方 `eval_info.json`。

## 4. 多视角同步采集

```python
camera_names = ["center", "left", "right", "wrist"]
frames = {name: [] for name in camera_names}

observation = env.reset()
for _ in range(max_steps):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    for name in camera_names:
        frames[name].append(env.render(camera_name=name))
    if terminated or truncated:
        break
```

所有相机必须在同一仿真 step［时间步］采集。单相机原片用于诊断，多视角合成片用于浏览。

## 5. 视频编码

```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p -movflags +faststart output.mp4
```

检查完整解码：

```bash
ffmpeg -v error -i output.mp4 -f null -
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json output.mp4
```

## 6. 训练输出门禁

一个训练目录至少包含：

```text
train.log
metrics.jsonl
run_config.json
checkpoints/{step}/pretrained_model/
```

检查最近训练记录与 checkpoint［检查点］：

```bash
tail -n 20 "$RUN_DIR/train.log"
tail -n 5 "$RUN_DIR/metrics.jsonl"
find "$RUN_DIR/checkpoints" -maxdepth 3 -type f | head
```

训练完成后使用刚生成的 checkpoint［检查点］评估，不在评估脚本中偷偷回退到历史模型。

## 7. benchmark 结果清单

| benchmark［评测基准］ | 正式协议 | 教程入口 |
| --- | --- | --- |
| Every Embodied | 红/蓝任务各 30 seed［随机种子］ | Task 02–13 |
| RoboCasa365 | 16 任务 × 50 回合 | [RoboCasa365 教程](./README_11_RoboCasa365_ROCm下载训练评估.md) |
| DexJoCo | 11 任务，官方固定 seed［随机种子］ | [DexJoCo 教程](./README_12_DexJoCo_Pi05_ROCm_JAX迁移训练评估.md) |
| DISCOVERSE | 任务门禁、专家数据、策略闭环分开统计 | [DISCOVERSE 教程](./README_13_DISCOVERSE_ROCm数据生成训练与多视角视频.md) |
| RoboWits | 10 任务 × 50 seed/mutation［原始条件/突变条件］ | [RoboWits 教程](./README_14_RoboWits_ROCm下载训练与创意任务评估.md) |
| Unitree G1 | 固定抛球 seed［随机种子］与安全间距 | [Unitree G1 教程](./README_15_Unitree_G1预测CBF_ROCm复现.md) |

## 8. 最终归档目录

```text
release/{benchmark}/{model}/
├── README.md
├── environment.txt
├── run_config.json
├── train_metrics.jsonl
├── checkpoint_link.txt
├── eval_summary.json
├── per_episode/
└── videos/
```

`README.md` 说明下载、训练与评估命令；`checkpoint_link.txt` 保存模型下载地址；大模型、数据和缓存不直接提交到 Git 仓库。

## 9. 完成判定

一套 benchmark［评测基准］教程完成，需要同时满足：

1. 新机器能按文档下载源码、资产、数据和模型；
2. 环境能完成 `reset → step → render → MP4`；
3. 训练入口能产生真实梯度更新与 checkpoint［检查点］；
4. 评估读取该 checkpoint［检查点］并输出完整分母；
5. 成功率可由逐回合文件重新计算；
6. 成功和典型失败视频都能定位到对应回合。
