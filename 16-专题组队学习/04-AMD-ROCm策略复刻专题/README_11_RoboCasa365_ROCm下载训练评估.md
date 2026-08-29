# RoboCasa365：ROCm 下载、训练与评估

RoboCasa365 是家庭场景机器人训练与 benchmark［评测基准］框架，包含 365 个任务、厨房资产、示教数据和 GR00T/Pi0.5 等策略入口。本章复现我们在 AMD Ryzen AI MAX+ 395 上完成的资产加载、无头渲染、官方 checkpoint［检查点］推理、16 任务正式评估和四视角视频。

## 1. 本章产出

- RoboCasa、RoboSuite 与 MuJoCo 环境；
- 约 10 GB 厨房资产和可选 AI 生成物体资产；
- GR00T N1.5 与 Pi0.5 多任务 checkpoint［检查点］；
- 16 个任务 × 50 回合的同协议结果；
- 中央、左侧、右侧和手眼相机四视角视频；
- GR00T 或 Pi0.5 的继续训练入口。

## 2. 下载源码和资产

```bash
# 先在当前终端执行 README_10 中的 WORK_ROOT、SRC_ROOT 等目录变量。
test -n "$SRC_ROOT" && test -n "$DATA_ROOT" && test -n "$MODEL_ROOT"
cd "$SRC_ROOT"

git clone https://github.com/ARISE-Initiative/robosuite.git
git clone https://github.com/robocasa/robocasa.git

python -m pip install -e "$SRC_ROOT/robosuite"
python -m pip install -e "$SRC_ROOT/robocasa"
python -m pip install imageio imageio-ffmpeg

cd "$SRC_ROOT/robocasa"
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

资产下载较大，中断后重新执行下载命令。浏览厨房、物体和任务：

```bash
python -m robocasa.demos.demo_kitchen_scenes
python -m robocasa.demos.demo_objects
python -m robocasa.demos.demo_tasks
```

训练 `pretrain_human300` 需要预训练划分中的人工示教。RoboCasa 会根据数据注册表下载全部对应任务：

```bash
cd "$SRC_ROOT/robocasa"
python -m robocasa.scripts.download_datasets \
  --split pretrain \
  --source human
```

这组数据体积很大。只学习一两个任务时，可先按任务名下载：

```bash
python -m robocasa.scripts.download_datasets \
  --tasks PickPlaceCounterToCabinet CloseFridge
```

下载完成后，使用注册表确认 `pretrain_human300` 已解析出数据目录：

```bash
python - <<'PY'
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

rows = DATASET_SOUP_REGISTRY["pretrain_human300"]
print("datasets:", len(rows))
print("first:", rows[0])
assert len(rows) > 0
PY
```

## 3. 下载官方策略

```bash
hf download robocasa/robocasa365_checkpoints \
  --include 'gr00t_n1-5_multitask_120000/**' \
  --local-dir "$MODEL_ROOT/robocasa365"

hf download robocasa/robocasa365_checkpoints \
  --include 'pi05_pretrain_human300_multitask_75000/**' \
  --local-dir "$MODEL_ROOT/robocasa365"
```

模型目录应包含：

```text
checkpoints/robocasa365/
├── gr00t_n1-5_multitask_120000/
└── pi05_pretrain_human300_multitask_75000/
```

## 4. 先跑环境门禁

公开配套脚本位于 [Radeon Physical AI Evidence Suite](https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite)：

```bash
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git \
  "$SRC_ROOT/radeon-physical-ai-evidence-suite"

export MUJOCO_GL=egl
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa_amd_smoke.py" \
  --task PickPlaceCounterToCabinet \
  --episodes 3 \
  --steps 40 \
  --out-dir "$RUN_ROOT/robocasa365/smoke"
```

PandaOmron 移动操作接口为 12 维：右臂 6、夹爪 1、平面底盘 3、升降躯干 1、底盘模式 1。使用配套动作门禁：

```bash
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa_mobile_mvp.py" \
  --env-name PickPlaceCounterToMicrowave \
  --dataset-base-path "$DATA_ROOT/robocasa365" \
  --output-dir "$RUN_ROOT/robocasa365/mobile_gate" \
  --episodes 1 \
  --steps 40
```

## 5. GR00T N1.5 训练

```bash
cd "$SRC_ROOT"
git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git
python -m pip install -e "$SRC_ROOT/Isaac-GR00T"

cd "$SRC_ROOT/Isaac-GR00T"
python scripts/gr00t_finetune.py \
  --output-dir "$RUN_ROOT/robocasa365/gr00t_train" \
  --dataset_soup pretrain_human300 \
  --max_steps 120000 \
  --save_steps 5000 \
  --batch_size 32 \
  --report_to tensorboard
```

`pretrain_human300` 对应 RoboCasa365 的 300 任务人工示教数据组合。正式长训前先把 `--max_steps` 设为 10，确认 loss［损失值］、GPU［图形处理器］占用和 checkpoint［检查点］写入，再恢复 120000 步。官方建议训练显存 80 GB 以上；较小 AMD 设备适合推理、评估或降低 batch size［批大小］的实验。降低 batch size［批大小］后，要在运行配置中记录实际值。

## 6. Pi0.5 训练

Pi0.5 使用独立 JAX［高性能数组计算框架］环境：

```bash
cd "$SRC_ROOT"
git clone https://github.com/robocasa-benchmark/openpi.git robocasa-openpi
python -m pip install -e "$SRC_ROOT/robocasa-openpi"
python -m pip install -e "$SRC_ROOT/robocasa-openpi/packages/openpi-client"

cd "$SRC_ROOT/robocasa-openpi"
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
  pi05_pretrain_human300 \
  --exp-name=robocasa_pi05_amd
```

AMD JAX［AMD 平台高性能数组计算框架］必须先通过 README_10 的矩阵预检。模型、归一化统计和训练配置要保存在同一输出目录。

## 7. 正式评估

GR00T：

```bash
cd "$SRC_ROOT/Isaac-GR00T"
export MATCH16_SITE="$EVERY_EMBODIED_ROOT/16-专题组队学习/04-AMD-ROCm策略复刻专题/code/robocasa_match16"
export PYTHONPATH="$MATCH16_SITE${PYTHONPATH:+:$PYTHONPATH}"

python scripts/run_eval.py \
  --model_path "$MODEL_ROOT/robocasa365/gr00t_n1-5_multitask_120000" \
  --task_set amd_match16 \
  --split pretrain \
  --n_episodes 50 \
  --n_envs 1 \
  --video_dir "$RUN_ROOT/robocasa365/gr00t_eval"

python gr00t/eval/get_eval_stats.py \
  --dir "$RUN_ROOT/robocasa365/gr00t_eval" \
  --task_set amd_match16
```

`sitecustomize.py` 在 Python［编程语言］启动时注册 `amd_match16`，不修改 RoboCasa 上游源码。`MATCH16_SITE` 直接使用 README_10 中定义的 `$EVERY_EMBODIED_ROOT`，因此 Every Embodied 与仿真源码可以放在不同磁盘。

Pi0.5 分为策略服务与仿真客户端。策略服务会持续占用当前终端，因此需要分别打开两个终端，并在两个终端中激活同一个 ROCm JAX［AMD 平台高性能数组计算环境］。

终端 A 启动策略服务：

```bash
cd "$SRC_ROOT/robocasa-openpi"
python scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=pi05_pretrain_human300 \
  --policy.dir="$MODEL_ROOT/robocasa365/pi05_pretrain_human300_multitask_75000"
```

看到端口 `8000` 开始监听后，保持终端 A 不动。终端 B 运行仿真客户端和结果汇总：

```bash
cd "$SRC_ROOT/robocasa-openpi"
export MATCH16_SITE="$EVERY_EMBODIED_ROOT/16-专题组队学习/04-AMD-ROCm策略复刻专题/code/robocasa_match16"
export PYTHONPATH="$MATCH16_SITE${PYTHONPATH:+:$PYTHONPATH}"
python examples/robocasa/main.py \
  --args.port 8000 \
  --args.task_soup amd_match16 \
  --args.split pretrain \
  --args.num_trials 50 \
  --args.log_dir "$RUN_ROOT/robocasa365/pi05_eval"

python examples/robocasa/get_eval_stats.py \
  --dir "$RUN_ROOT/robocasa365/pi05_eval"
```

我们的正式协议固定 16 个任务，每任务 50 回合，两个模型共享相同任务、分母和环境成功条件：

```text
ArrangeTea, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug,
DeliverStraw, OpenCabinet, OpenDrawer, PackIdenticalLunches,
PickPlaceDrawerToCounter, PortionHotDogs, PrepareCoffee,
RecycleBottlesByType, RinseSinkBasin, ScrubCuttingBoard,
SeparateFreezerRack, SlideDishwasherRack
```

| 模型 | 成功数 | 总回合 | 成功率 |
| --- | ---: | ---: | ---: |
| GR00T N1.5 | 230 | 800 | 28.75% |
| Pi0.5 | 142 | 800 | 17.75% |

## 8. 四视角长程视频

```bash
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa365_showcase.py" \
  --help
```

展示脚本保持策略输入分辨率不变，只提高录像相机分辨率。推荐中央、左侧、右侧和手眼相机各 960×540，再拼成 1920×1080@20fps。完整示例见 [PackIdenticalLunches 长程成功视频](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/robocasa-recovery/pack-success.mp4)。

## 9. AMD 迁移重点

| 模块 | 上游假设 | AMD 处理 | 验收方式 |
| --- | --- | --- | --- |
| GR00T N1.5 | CUDA PyTorch | 安装 ROCm PyTorch，保留 `cuda` 设备字符串 | 前向、反向、优化器与模型保存 |
| Pi0.5 | JAX CUDA | 隔离安装 ROCm JAX | 64×64 矩阵、模型加载与策略服务 |
| MuJoCo | 本地图形窗口 | `MUJOCO_GL=egl` | `reset → step → render → MP4` |
| 数据 | 注册表中的相对路径 | 统一挂载到数据盘，并让两个策略仓库读取同一注册表 | 数据条目数、视频解码与动作维度 |
| 动作接口 | PandaOmron 12 维动作 | 保持右臂、夹爪、底盘、躯干和模式位顺序 | 固定动作逐维门禁 |
| 评估 | 不同策略各自脚本 | 注册同一 `amd_match16` 任务集合 | 16×50 同分母汇总 |

PyTorch［张量计算框架］、JAX［高性能数组计算框架］和 MuJoCo［机器人动力学仿真器］最好分环境安装。策略加载成功后仍要先做动作接口门禁；动作归一化、机器人类型或控制频率错误时，模型可以正常前向，但闭环行为会完全失效。

## 10. 常见问题

| 现象 | 检查 |
| --- | --- |
| 创建环境时报资产缺失 | 重新运行 `download_kitchen_assets`，检查 `macros_private.py` |
| 远端渲染失败 | 设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl` |
| 模型加载成功但动作异常 | 核对机器人类型、动作维度、归一化统计和控制频率 |
| 成功视频与统计不一致 | 视频文件必须来自同一回合目录，并读取环境 success 字段 |
| Pi0.5 找不到 AMD 设备 | 在隔离环境重装 ROCm JAX［AMD 平台 JAX 运行栈］并重跑矩阵预检 |

## 11. 官方资料

- [RoboCasa 源码](https://github.com/robocasa/robocasa)
- [RoboCasa 文档](https://robocasa.ai/docs/introduction/overview.html)
- [RoboCasa GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T)
- [RoboCasa OpenPI](https://github.com/robocasa-benchmark/openpi)
- [RoboCasa365 checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints)
