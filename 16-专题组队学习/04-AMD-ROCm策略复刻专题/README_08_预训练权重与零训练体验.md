# 预训练权重与零训练体验

本章提供三类可直接加载的课程模型。学习者可以先运行推理和闭环评估，熟悉正确行为、输入输出和结果目录，再决定是否重新训练。

## 模型仓库

| 模型 | Hugging Face 仓库 | 课程用途 |
| --- | --- | --- |
| SmolVLA | [Datawhale/every-embodied-smolvla-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp) | 红杯/蓝杯抓取与采样平衡 |
| Pi0 | [Datawhale/every-embodied-pi0-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-pi0-mujoco-pnp) | 视觉语言动作模型加载、继续训练和诊断 |
| ACT | [Datawhale/every-embodied-act-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-act-mujoco-pnp) | 轻量训练、闭环基线和 DAgger 实验 |

模型仓库包含推理权重、模型配置和训练配置。优化器状态、缓存和批量视频不属于零训练体验所需材料。

## 下载

```bash
export MODEL_ROOT=${MODEL_ROOT:-$HOME/physical-ai/checkpoints}
mkdir -p "$MODEL_ROOT"

hf download Datawhale/every-embodied-smolvla-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-smolvla"

hf download Datawhale/every-embodied-pi0-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-pi0"

hf download Datawhale/every-embodied-act-mujoco-pnp \
  weights/model.safetensors weights/config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-act"
```

下载后确认文件可由 `safetensors` 打开：

```bash
export POLICY_PATH="$MODEL_ROOT/every-embodied-smolvla/weights"
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import os

path = Path(os.environ["POLICY_PATH"]) / "model.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("tensor_count:", len(keys))
print("config_exists:", (path.parent / "config.json").is_file())
PY
```

## 四视角回放

打开 [11_mujoco_closed_loop_deploy.ipynb](./notebooks/11_mujoco_closed_loop_deploy.ipynb)，运行“零训练成功预览”单元格。该单元格播放：

```text
assets/pnp_four_view_strict_success.mp4
```

回放用于学习接近、夹取、抬升、搬运、释放和稳定放置的动作阶段。随后加载模型运行新的闭环回合，结果写入自己的 `$RUN_ROOT`。

## 加载与评估

以 SmolVLA 为例：

```bash
export POLICY_TYPE=smolvla
export POLICY_PATH="$MODEL_ROOT/every-embodied-smolvla/weights"
export EVAL_SEEDS=1000,1001,1002,1003
export OUTPUT_ROOT="$RUN_ROOT/zero_train/smolvla"
python code/run_closed_loop.py
```

Pi0 和 ACT 分别将 `POLICY_TYPE` 改为 `pi0` 或 `act`，并把 `POLICY_PATH` 指向对应的 `weights/` 目录。

运行完成后检查：

```text
$OUTPUT_ROOT/
├── results.jsonl
├── summary.json
└── videos/
```

模型加载、环境版本、随机种子和成功条件写入同一运行目录，便于与重新训练后的模型比较。

## 课程结果索引

| 模型 | 固定协议结果 | 对应材料 |
| --- | ---: | --- |
| SmolVLA weighted500 | 红杯 `27/30`，蓝杯 `30/30`，合计 `57/60` | 评估摘要、成功/失败视频、Notebook 14 |
| Pi0 protected clean40 | strict `12/14`；unseen `9/10`；hard `6/8` | 评估摘要、诊断视频、Notebook 15 |
| ACT protected repair15 | strict `15/30` | 三组固定随机种子结果、Notebook 16 |

表中数值对应课程保存的固定协议。学习者重新运行时，以新生成的逐回合文件和当前环境配置为准。

## 继续训练

继续训练时将下载目录作为 `pretrained_path`，并使用新的输出目录：

```text
base model: $MODEL_ROOT/every-embodied-<model>/weights
run output: $RUN_ROOT/<model>/<run_name>
```

训练配置记录：

- 数据集版本、回合数和任务分布；
- 观测键、状态维度和动作维度；
- 控制频率、动作块长度和归一化方式；
- batch size、学习率、训练步数和保存间隔；
- GPU、ROCm 和框架版本；
- 评估任务、随机种子和物理成功条件。

## 存储建议

保留模型权重、配置、训练指标和正式评估结果。下载缓存可以重新生成，空间不足时优先清理缓存和重复的中间导出。大型文件不写入教程 Git 仓库。
