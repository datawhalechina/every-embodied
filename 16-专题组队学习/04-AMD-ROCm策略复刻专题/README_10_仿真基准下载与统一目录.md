# 仿真基准下载与统一目录

本章先完成所有 benchmark［评测基准］共用的存储、鉴权、断点下载和运行目录。后续 RoboCasa365、DexJoCo、DISCOVERSE、RoboWits 与 Unitree G1 教程都沿用这里的变量，避免把数据、checkpoint［检查点］和缓存写进源码仓库。

## 1. 推荐目录

先进入 Every Embodied 仓库根目录，再定义教程与大文件目录：

```bash
export EVERY_EMBODIED_ROOT="$(git rev-parse --show-toplevel)"
export WORK_ROOT=${WORK_ROOT:-$HOME/physical-ai}

# AMD 开发者云的持久化 PVC 可使用：
# export WORK_ROOT=/workspace/physical-ai

# AUP 镜像若约定 /home/jovyan 持久化，可使用：
# export WORK_ROOT=/home/jovyan/physical-ai
export SRC_ROOT=$WORK_ROOT/src
export DATA_ROOT=$WORK_ROOT/datasets
export MODEL_ROOT=$WORK_ROOT/checkpoints
export RUN_ROOT=$WORK_ROOT/runs
export CACHE_ROOT=$WORK_ROOT/cache
export ENV_ROOT=$WORK_ROOT/envs
export HF_HOME=$CACHE_ROOT/huggingface
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv

test -f "$EVERY_EMBODIED_ROOT/README.md"
mkdir -p "$SRC_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT" "$ENV_ROOT" \
  "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

源码只保存代码和小配置。数据、模型、视频与训练输出分别放到独立目录，便于删除缓存而不误删训练结果。

## 2. AMD 设备预检

PyTorch［张量计算框架］在 ROCm［AMD 开放计算平台］上仍使用 `torch.cuda` 设备接口；判断后端时要同时检查 `torch.version.hip`。

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
assert torch.version.hip is not None
PY
```

JAX［高性能数组计算框架］项目另做矩阵预检：

```bash
JAX_PLATFORMS=rocm python - <<'PY'
import jax
import jax.numpy as jnp

print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
x = jnp.ones((64, 64), dtype=jnp.float32)
print("sum:", float((x @ x).sum().block_until_ready()))
assert jax.default_backend() == "rocm"
PY
```

## 3. Hugging Face 鉴权与下载

公开模型可直接下载；受限数据集需要先在网页申请访问，再在终端登录。token［访问令牌］只写入本机凭据，不写进脚本或 Notebook［交互式笔记本］。

```bash
hf auth login
hf auth whoami
```

推荐使用 `hf download` 的 `--local-dir`，中断后重新执行同一命令即可续传。下面先定义仓库和本地目录，替换变量值后可直接运行：

```bash
export HF_MODEL_REPO=robocasa/robocasa365_checkpoints
export HF_DATASET_REPO=DexJoCo/DexJoCo-Datasets-LeRobot
export LOCAL_MODEL_DIR="$MODEL_ROOT/robocasa365"
export LOCAL_DATASET_DIR="$DATA_ROOT/dexjoco-lerobot"

hf download "$HF_MODEL_REPO" \
  --repo-type model \
  --local-dir "$LOCAL_MODEL_DIR"

hf download "$HF_DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$LOCAL_DATASET_DIR"
```

只取仓库中的一个目录时使用 `--include`：

```bash
export HF_SUBDIR='pi05_pretrain_human300_multitask_75000/**'
hf download "$HF_MODEL_REPO" \
  --include "$HF_SUBDIR" \
  --local-dir "$LOCAL_MODEL_DIR"
```

## 4. 无头渲染

远端设备没有桌面窗口时使用 EGL［无窗口图形渲染接口］：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
```

最小验证不是“程序可导入”，而是完成一次 `reset → step → render → MP4`：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 "$RUN_ROOT/robocasa365/smoke/episode_000.mp4"
```

## 5. 两层运行协议

每套 benchmark［评测基准］都分成两层：

| 层级 | 目的 | 最小输出 |
| --- | --- | --- |
| smoke［冒烟测试］ | 验证环境、资产、动作维度和视频链路 | 1 个任务、1 个短回合、MP4、环境信息 |
| formal evaluation［正式评估］ | 统计模型能力 | 固定任务、固定 seed［随机种子］、完整分母、逐回合结果 |

训练前先通过 smoke［冒烟测试］，但正式结果只从完整评估目录汇总。

## 6. 统一运行目录

```text
runs/{benchmark}/{model}/{run_name}/
├── run_config.json
├── environment.txt
├── train.log
├── metrics.jsonl
├── checkpoints/
├── eval/
│   └── {task}/
│       ├── eval_info.json
│       └── videos/
└── summary.json
```

`run_config.json` 至少记录上游版本、任务、模型、动作/观测维度、相机、控制频率、训练步数和评估 seed［随机种子］。`summary.json` 必须从逐回合文件聚合，不手填成功率。

## 7. 下载完成检查

```bash
export PROJECT_DIR="$SRC_ROOT/robocasa"
export DATASET_DIR="$DATA_ROOT/dexjoco-lerobot"
export CHECKPOINT_DIR="$MODEL_ROOT/robocasa365"

test -d "$PROJECT_DIR/.git"
test -n "$(find "$DATASET_DIR" -type f -print -quit)"
test -n "$(find "$CHECKPOINT_DIR" -type f -print -quit)"
df -h "$WORK_ROOT"
```

确认源码、数据和模型目录都非空，并预留训练 checkpoint［检查点］与视频空间后，再进入对应教程。

## 8. 后续入口

- [RoboCasa365：资产、策略训练与 16 任务评估](./README_11_RoboCasa365_ROCm下载训练评估.md)
- [DexJoCo：Pi0.5、原生 ROCm JAX 与 11 任务评估](./README_12_DexJoCo_Pi05_ROCm_JAX迁移训练评估.md)
- [DISCOVERSE：专家数据、策略训练与多视角视频](./README_13_DISCOVERSE_ROCm数据生成训练与多视角视频.md)
- [RoboWits：受限数据、ACT/Pi0/Pi0.5 与突变评估](./README_14_RoboWits_ROCm下载训练与创意任务评估.md)
- [Unitree G1：预测 CBF 安全控制复现](./README_15_Unitree_G1预测CBF_ROCm复现.md)
- [统一评估、视频与结果归档](./README_16_统一评估视频与结果归档.md)
