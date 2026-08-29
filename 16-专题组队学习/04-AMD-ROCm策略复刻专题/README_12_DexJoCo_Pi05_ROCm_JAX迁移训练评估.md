# DexJoCo：Pi0.5、ROCm JAX 与 11 任务复现

DexJoCo 是基于 MuJoCo 的灵巧操作 benchmark［评测基准］，覆盖 11 个单臂、双臂、工具使用和长时序任务。本章把模拟器与 OpenPI/Pi0.5 分成两个隔离环境，通过 WebSocket［网络通信协议］连接，完成原生 AMD JAX［AMD 平台高性能数组计算框架］推理、数据转换、训练和闭环视频评估。

## 本章产出与任务

| 任务 | 类型 |
| --- | --- |
| `click_mouse`、`fold_glasses`、`hammer_nail`、`pick_bucket`、`pinch_tongs`、`water_plant` | 单臂或工具操作 |
| `bimanual_assembly`、`bimanual_hanoi`、`bimanual_microwave_cook`、`bimanual_photograph`、`bimanual_unlock_ipad` | 双臂协同与长时序操作 |

单臂策略输出 22 维动作，双臂策略输出 44 维动作。每个任务由官方环境的完成条件产生 `success`，包括目标物体关系、工具状态或任务末态。逐回合结果保存任务名、随机种子、步数、成功字段和视频。

安装后记录源码版本：

```bash
git -C "$SRC_ROOT/dexjoco" rev-parse HEAD
```

## 1. 下载源码、数据和模型

```bash
cd "$SRC_ROOT"
git clone https://github.com/brave-eai/dexjoco.git
cd dexjoco

conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

LeRobot［机器人学习数据框架］数据与 Pi0.5 模型：

```bash
hf download DexJoCo/DexJoCo-Datasets-LeRobot \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/dexjoco-lerobot"

hf download DexJoCo/DexJoCo-Pi05 \
  --include 'pi05_dexjoco_multi_task/**' \
  --local-dir "$MODEL_ROOT/dexjoco-pi05"
```

只下载多任务模型可节省空间。单任务和 `rand_full` 视觉随机化权重按需下载。

## 2. 模拟器环境与 11 任务检查

```bash
export MUJOCO_GL=egl
python scripts/amd_rocm_smoke.py \
  --output "$RUN_ROOT/dexjoco/smoke_11_tasks"
```

输出应包含 `smoke_report.json` 和 11 个任务的 MP4。这里确认的是环境、动作接口和渲染链路；策略评估在后续步骤完成。

## 3. 无 Docker 的原生 ROCm JAX 环境

Python 3.12 环境与模拟器的 Python 3.11 环境分开：

```bash
export ROCM_JAX010_ENV="$ENV_ROOT/openpi-amd-jax010"
python3.12 -m venv "$ROCM_JAX010_ENV"
export PIP_NO_CACHE_DIR=1

"$ROCM_JAX010_ENV/bin/python" -m pip install --upgrade pip
"$ROCM_JAX010_ENV/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[libraries,device-gfx1151]==7.14.0'
"$ROCM_JAX010_ENV/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'jax_rocm7_plugin==0.10.0+rocm7.14.0' \
  'jax_rocm7_pjrt==0.10.0+rocm7.14.0'
"$ROCM_JAX010_ENV/bin/python" -m pip install 'jax==0.10.0' 'jaxlib==0.10.0'
```

然后执行 README_10 的 JAX［高性能数组计算框架］矩阵预检。不同 ROCm JAX［AMD 平台 JAX 运行栈］版本不要装进同一个环境。

## 4. 数据采集和回放

交互采集成功示教：

```bash
conda activate dexjoco
python scripts/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True \
  --out_dir="$DATA_ROOT/dexjoco-raw/water_plant"
```

回放并复核初始状态、动作和视频：

```bash
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir="$DATA_ROOT/dexjoco-raw/water_plant" \
  --out_dir="$RUN_ROOT/dexjoco/replay/water_plant" \
  --randomize=True \
  --restore_state=True
```

## 5. 转换为 LeRobot 数据

```bash
bash dexjoco-data-converter/install.bash
conda activate dexjoco-dc

dexjoco-dc-single-lerobot \
  --input "$DATA_ROOT/dexjoco-raw/water_plant" \
  --output "$DATA_ROOT/dexjoco-lerobot-local/water_plant" \
  --language-instruction 'Grasp the watering can and apply water to the plant.' \
  --selected-data-yaml '{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}' \
  --slice-yaml '{state: [null, 23]}'
```

单臂策略动作为 22 维，双臂为 44 维旋转向量动作；模拟器封装负责转换为内部四元数动作。转换后检查 episode［回合］边界、相机帧数、状态维度和动作维度。

## 6. Pi0.5 训练

```bash
cd "$SRC_ROOT/dexjoco/openpi"
bash install.bash
```

训练前依次完成：

1. 在 `openpi/config.yaml` 中把 `dataset_root` 指向 `$DATA_ROOT/dexjoco-lerobot`，并把 `pretrained_model_path`、`pretrained_model_action_dim_44_path` 与 `ckpts_root` 指向实际 checkpoint［检查点］目录；
2. 双臂任务运行 `convert_to_action_dim_44_model.py`；
3. 运行 `compute_norm_stats.py` 生成归一化统计；
4. 用 `scripts/train.py` 启动单任务或多任务训练。

```bash
export DEXJOCO_CONFIG=multi_task

JAX_PLATFORMS=rocm \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$ROCM_JAX010_ENV/bin/python" scripts/compute_norm_stats.py "$DEXJOCO_CONFIG"

JAX_PLATFORMS=rocm \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$ROCM_JAX010_ENV/bin/python" scripts/train.py "$DEXJOCO_CONFIG" \
  --exp-name=dexjoco_pi05_amd
```

`multi_task` 读取三相机、多任务数据并采用 44 维双臂动作。单任务配置名与任务名相同，例如 `water_plant`、`hammer_nail`、`fold_glasses` 和 `bimanual_hanoi`；完整配置可用下面的命令列出：

```bash
rg 'name="' src/openpi/training/dexjoco_configs.py
```

官方说明中单臂 LoRA［低秩适配微调］通常训练 30k steps［训练步数］，双臂/多任务训练 60k steps［训练步数］。本地步数应由验证集与闭环评估共同选择。

## 7. 11 任务正式评估

公开适配脚本：

```bash
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git \
  "$SRC_ROOT/radeon-physical-ai-evidence-suite"

DEXJOCO_ROOT="$SRC_ROOT/dexjoco" \
CHECKPOINT="$MODEL_ROOT/dexjoco-pi05/pi05_dexjoco_multi_task" \
OPENPI_PYTHON="$ROCM_JAX010_ENV/bin/python" \
DEXJOCO_EVAL="$CONDA_PREFIX/bin/dexjoco-openpi-eval" \
OUT="$RUN_ROOT/dexjoco/pi05_multitask_seed0" \
  bash "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/run_dexjoco_pi05_multitask_eval.sh"
```

脚本为每个任务启动匹配的策略服务，固定 seed［随机种子］ 0、每任务 1 回合，并保存结果与视频。我们的同协议结果为 `5/11`。

对失败任务寻找可复现成功动作过程时，可固定搜索 seed［随机种子］ 1–10，并在首个成功后停止：

```bash
DEXJOCO_ROOT="$SRC_ROOT/dexjoco" \
CHECKPOINT="$MODEL_ROOT/dexjoco-pi05/pi05_dexjoco_multi_task" \
OPENPI_PYTHON="$ROCM_JAX010_ENV/bin/python" \
BASE_OUT="$RUN_ROOT/dexjoco/pi05_multitask_seed0" \
OUT="$RUN_ROOT/dexjoco/pi05_multitask_recovery" \
  python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/run_dexjoco_pi05_multitask_recovery.py"
```

该成功案例档案覆盖 10/11 个任务，用于学习动作过程；正式 seed［随机种子］ 0 成绩仍单独保留。

## 8. 迁移中最关键的接口

| 层 | AMD 处理 |
| --- | --- |
| MuJoCo 模拟器 | Python 3.11、NumPy 1.26、EGL［无窗口图形渲染接口］ |
| OpenPI | Python 3.12、ROCm JAX［AMD 平台 JAX 运行栈］ 0.10 |
| 进程连接 | WebSocket［网络通信协议］策略服务 |
| checkpoint［检查点］ | Orbax［JAX 检查点系统］恢复兼容新旧 metadata［元数据］对象 |
| 状态输入 | 按模型要求补齐到 46 维，再由配置切片 |
| 动作 | 22/44 维旋转向量转模拟器四元数动作 |

三环境隔离是复现稳定性的核心：模拟器、数据转换器、OpenPI 不共享一个依赖集合。

## 9. 视频案例

[双臂河内塔完整视频](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/dexjoco/recovery/bimanual-hanoi.mp4)为 47.6 秒、外部视角加左右腕部视角。所有相机在同一仿真循环采集，视频结果与该回合成功标记绑定。

## 10. 官方资料

- [DexJoCo 源码](https://github.com/brave-eai/dexjoco)
- [DexJoCo LeRobot 数据](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot)
- [DexJoCo Pi0.5 模型](https://huggingface.co/DexJoCo/DexJoCo-Pi05)
- [AMD ROCm JAX 安装指南](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
