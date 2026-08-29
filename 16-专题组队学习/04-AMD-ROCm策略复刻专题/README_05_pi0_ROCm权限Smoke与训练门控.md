# 05 Pi0/Pi0.5 权限、短训练与闭环评估

本章建立 Pi0/Pi0.5 的标准运行顺序：访问权限、模型加载、单步反向、短训练、模型保存和固定协议闭环评估。动作表示、数据行序和恢复数据实验见 [Pi0/Pi0.5 实验记录](./appendices/PI0_PI05_EXPERIMENT_NOTES.md)。

## 本章产出

- Hugging Face 访问检查结果；
- 一次前向、反向和优化器更新；
- 短训练模型和 `metrics.jsonl`；
- 固定随机种子的 `results.jsonl`、成功率和视频；
- 动作维度、控制频率和归一化统计记录。

## 1. 访问权限

在模型页面申请访问后执行：

```bash
hf auth login
hf auth whoami

hf download lerobot/pi0 \
  --include 'config.json' \
  --local-dir "$MODEL_ROOT/pi0-base"
```

下载成功后再启动训练，避免在模型初始化中途才发现权限或网络问题。

## 2. 环境和数据预检

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("gpu:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert torch.version.hip is not None
PY
```

数据预检至少覆盖：

| 检查项 | 期望 |
| --- | --- |
| `episode_index` | 每个回合连续、边界清楚 |
| `frame_index` | 回合内单调递增 |
| `observation.state` | 维度与模型配置一致 |
| `action` | 维度、单位和绝对/增量语义明确 |
| 相机 | 键名、分辨率、颜色通道一致 |
| 语言 | 每个回合具有与任务匹配的指令 |

Pi0.5 使用动作块时，还要检查跨回合窗口。边界处的动作块只能读取当前回合帧，缺失部分按训练实现的约定填充。

## 3. 单步训练

先将正式配置复制为短训练配置，仅修改运行规模和输出目录：

```yaml
steps: 1
batch_size: 1
save_freq: 1
output_dir: ${RUN_ROOT}/pi0/smoke
```

执行后检查：

1. 模型由 CPU 加载后移动到 GPU；
2. 损失值为有限数；
3. 反向传播产生非零梯度；
4. 优化器完成一次更新；
5. 保存目录包含模型配置和可重新加载的权重。

大模型直接加载到 GPU 时出现运行时崩溃，可先在 CPU 读取 `safetensors`，再调用模型的设备迁移接口。该顺序能降低权重反序列化阶段的显存峰值。

## 4. 短训练与正式训练

推荐分三段运行：

| 阶段 | 步数 | 用途 |
| --- | ---: | --- |
| 单步 | 1 | 数据、前向、反向和保存 |
| 短训练 | 20–100 | 观察损失、显存和耗时 |
| 正式训练 | 按任务配置 | 生成可评估模型 |

每段使用独立目录：

```text
$RUN_ROOT/pi0/<run_name>/
├── run_config.json
├── train.log
├── metrics.jsonl
└── checkpoints/
```

恢复训练时，从模型目录的 `train_config.json` 和优化器状态加载，并确认日志中的步数从上次保存点继续。输出目录使用新的运行名，保留上一个可用模型。

## 5. 动作表示

训练前固定以下契约：

```text
observation = images + robot_state + language
action = arm_or_eef + gripper
frequency = 20 Hz
chunk = N consecutive actions from the same episode
```

关节增量适合与原始控制器一致的数据。末端增量适合显式控制平移和姿态的任务。末端增量必须在统一坐标系中计算：

```text
delta_position = target_tcp_position - current_tcp_position
delta_rotation = relative_rotation(current_tcp, target_tcp)
```

由关节轨迹转换末端动作时，使用 MuJoCo 正向运动学重新计算每一帧末端位姿，并验证回放轨迹与原始示教一致。

## 6. 闭环评估

模型保存后使用 [批量闭环脚本](./code/README.md)：

```bash
export POLICY_TYPE=pi0
export POLICY_PATH="$RUN_ROOT/pi0/<run_name>/checkpoints/<step>/pretrained_model"
export EVAL_SEEDS=1000,1001,1002,1003
export OUTPUT_ROOT="$RUN_ROOT/pi0/<run_name>/eval"
python code/run_closed_loop.py
```

短评估用于检查动作方向和接触行为。正式结果应扩大随机种子集合，并记录：

| 指标 | 含义 |
| --- | --- |
| approach | 末端进入目标邻域 |
| contact | 夹爪与物体形成接触 |
| grasp | 物体受控于夹爪 |
| lift | 物体离开支撑面并持续若干帧 |
| transport | 物体进入目标区域 |
| place | 放置位置、姿态和稳定性满足任务条件 |
| physical success | 完成任务的最终物理判定 |

视频和逐回合结果使用同一运行目录，便于从失败阶段定位对应帧。

## 7. 视觉与语言参数检查

微调配置需要明确哪些模块参与训练。启动后统计各模块的可训练参数和梯度：

```python
groups = {
    "vision": [],
    "language": [],
    "action": [],
}

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    key = "vision" if "vision" in name else "language" if "language" in name else "action"
    groups[key].append(param)

for key, params in groups.items():
    trainable = sum(p.numel() for p in params)
    with_grad = sum(p.numel() for p in params if p.grad is not None)
    print(key, trainable, with_grad)
```

训练日志同时保存模块梯度范数，可验证视觉塔、语言塔和动作头是否符合配置。

## 8. 结果记录

实验摘要至少包含：

```text
base model:
dataset:
episodes / frames:
observation keys:
state dimension:
action representation and dimension:
control frequency:
chunk size:
train steps / batch size / learning rate:
GPU / ROCm / PyTorch:
evaluation tasks and seeds:
physical success:
artifact directory:
```

更细的 EEF-delta、动作块行序、恢复数据和长训练对照见 [Pi0/Pi0.5 实验记录](./appendices/PI0_PI05_EXPERIMENT_NOTES.md)。
