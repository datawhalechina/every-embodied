# Unitree G1：预测 CBF 的 ROCm 复现

本章复现 PAC-MAN 中的 predictive CBF［预测控制障碍函数］控制路径，并使用 Unitree G1 模型生成全身体态避障视频。上游完整训练由 mjlab、MuJoCo Warp、AMP［对抗运动先验］与 PPO［近端策略优化］组成；本教程先把可独立验证的安全控制与 ONNX［开放神经网络交换格式］策略部署到 AMD PyTorch［AMD 平台张量计算框架］。

## 1. 下载源码和上游资产

```bash
cd "$SRC_ROOT"
git clone https://github.com/lzyang2000/perceptive_cbf_rl.git
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git

python -m pip install mujoco imageio imageio-ffmpeg onnxruntime
```

上游仓库包含 G1 MJCF、网格、动作数据和 `deploy/ckpts/dodge_link_cbf.onnx`。

## 2. CBF 控制路径

预测控制依次完成：

1. 根据球的位置与速度估计水平轨迹；
2. 判断球是否接近、是否在空中、是否进入感知半径；
3. 选择并锁存垂直于球轨迹的躲避方向；
4. 把名义速度投影到带时间约束的安全半空间；
5. 将安全动作送入 G1 策略或回放控制器。

这段张量计算可以批量运行在 ROCm［AMD 开放计算平台］上。

## 3. AMD 控制门禁

```bash
cd "$SRC_ROOT/radeon-physical-ai-evidence-suite"
python code/perceptive_cbf_rl_amd/amd_pacman_cbf_smoke.py \
  --output-dir "$RUN_ROOT/pacman/cbf_gate" \
  --episodes 12 \
  --device rocm
```

输出包含固定 seed［随机种子］安全指标、最小间距和双视角视频。

## 4. G1 全身体态回放

```bash
export MUJOCO_GL=egl
python code/perceptive_cbf_rl_amd/g1_amd_dodge_replay.py \
  --upstream-xml "$SRC_ROOT/perceptive_cbf_rl/src/assets/robots/unitree_g1/xmls/scene_g1.xml" \
  --output-dir "$RUN_ROOT/pacman/g1_replay" \
  --episodes 8
```

输出：

```text
g1_replay/
├── eval_info.json
├── run_manifest.json
└── unitree-g1-predictive-cbf-amd-replay.mp4
```

固定 8 个 seed［随机种子］的回放结果为 `8/8` 保持安全间距，最小间距约 0.42 m。

## 5. ONNX 策略推理

```bash
python - <<'PY'
import onnxruntime as ort

path = "src/perceptive_cbf_rl/deploy/ckpts/dodge_link_cbf.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
for value in session.get_inputs():
    print(value.name, value.shape, value.type)
for value in session.get_outputs():
    print(value.name, value.shape, value.type)
PY
```

若使用 PyTorch［张量计算框架］版本策略，输入为 960 维特征，输出覆盖 29 个关节。执行前检查输入帧堆叠、归一化、关节顺序和控制频率。

## 6. 完整训练配置

上游训练由 mjlab、MuJoCo Warp、AMP［对抗运动先验］和 rsl_rl［机器人强化学习库］组成。训练前先同步依赖并列出已注册环境：

```bash
cd "$SRC_ROOT/perceptive_cbf_rl"
uv sync
uv run python scripts/list_envs.py
```

训练矩阵由三种感知输入和三种安全约束组成：

| 感知输入 | 无屏障约束 | Link-CBF［连杆控制障碍函数］ | Joint-CBF［关节控制障碍函数］ |
| --- | --- | --- | --- |
| state oracle［真值状态］ | `state_none.sh` | `state_link.sh` | `state_joint.sh` |
| fixed camera［固定相机］ | `vision_none.sh` | `vision_link.sh` | `vision_joint.sh` |
| gimbal camera［云台相机］ | `gimbal_none.sh` | `gimbal_link.sh` | `gimbal_joint.sh` |

每个脚本默认使用 8192 个并行环境、25000 次迭代，论文结果固定读取第 20000 次迭代的 checkpoint［检查点］。`vision_link.sh` 是部署到真机并导出 `deploy/ckpts/dodge_link_cbf.onnx` 的配置。

### 6.1 小规模训练门禁

先把并行环境和迭代数缩小，验证资产加载、深度渲染、策略前向、反向传播和 checkpoint［检查点］保存：

```bash
cd "$SRC_ROOT/perceptive_cbf_rl"
NUM_ENVS=64 ./train_runs/vision_link.sh \
  --agent.max-iterations=10 \
  --video False
```

训练日志应显示任务 `Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat`、实验名 `vision_link_64` 和持续更新的 loss［损失值］。输出位于：

```text
logs/rsl_rl/vision_link_64/{run}/
```

这一步只验证训练链路，不用于报告避障成功率。

### 6.2 正式视觉策略训练

显存允许时，直接运行论文配置：

```bash
cd "$SRC_ROOT/perceptive_cbf_rl"
NUM_ENVS=8192 ./train_runs/vision_link.sh
```

资源较小的单卡可以降低并行环境数。脚本会把环境数附加到实验名，避免覆盖论文配置：

```bash
NUM_ENVS=1024 ./train_runs/vision_link.sh
```

视觉策略读取 BallOnly masked depth［仅保留球体的掩码深度图］，默认相机上仰 20 度，使用 `(0,3,8,18)` 四帧偏移。训练与部署必须保持相机安装角、关节顺序、控制频率和深度预处理一致。

如果只想先验证安全奖励与 PPO［近端策略优化］收敛，可运行不需要深度渲染的状态策略：

```bash
NUM_ENVS=8192 ./train_runs/state_link.sh
```

### 6.3 固定 checkpoint 评估

上游论文统一比较第 20000 次迭代的 checkpoint［检查点］。先确定运行目录，再显式传入模型文件：

```bash
cd "$SRC_ROOT/perceptive_cbf_rl"
export DODGE_CKPT="$(find logs/rsl_rl/vision_link -name model_20000.pt -print -quit)"
test -f "$DODGE_CKPT"
echo "$DODGE_CKPT"

uv run python scripts/dodge_benchmark.py \
  --only vision_link \
  --ckpt "$DODGE_CKPT"
```

`DODGE_CKPT` 会读取首个匹配的模型文件。若同一实验有多次训练，应先删除旧输出或按训练时间显式选择目标文件。部署模式还会在两次投球之间调用行走策略，把 G1 恢复到站立位置：

```bash
uv run python scripts/dodge_benchmark.py \
  --only vision_link \
  --walk-recover \
  --ckpt "$DODGE_CKPT"
```

完整九组矩阵评估直接运行：

```bash
uv run python scripts/dodge_benchmark.py
```

评估需要记录碰撞数、跌倒数、最小安全间距、下蹲/左移/右移模式和单步推理延迟。公开视频可从成功回合中选取，但统计表仍以固定 checkpoint［检查点］和固定 seed［随机种子］的完整分母为准。

## 7. AMD 迁移路线

完整训练不能只替换 PyTorch［张量计算框架］安装包。需要同时迁移张量计算、批量物理仿真和计算图三个层次：

| 层次 | 上游路径 | AMD 路径 | 验证门禁 |
| --- | --- | --- | --- |
| PPO/AMP | CUDA PyTorch | ROCm PyTorch | 前向、反向、优化器更新 |
| 批量物理 | MuJoCo Warp CUDA | AMD-Ecosystem MuJoCo Warp | 相同初态下状态误差和吞吐 |
| 条件图 | CUDA Graph 条件节点 | HIP Graph［AMD 计算图］条件节点 | 分支命中、物理等价性和加速比 |
| 策略部署 | ONNX Runtime | ONNX Runtime 或 ROCm PyTorch | 输入形状、关节顺序和延迟 |

先下载 AMD 维护的 Warp 与 MuJoCo Warp：

```bash
cd "$SRC_ROOT"
git clone https://github.com/AMD-Ecosystem/warp.git warp-amd
git clone https://github.com/AMD-Ecosystem/mujoco_warp.git mujoco-warp-amd
```

AMD 训练环境应与回放环境隔离，避免不同 MuJoCo API［应用程序接口］版本互相覆盖。建议在新环境中依次完成：

1. ROCm PyTorch 的矩阵计算、反向传播和优化器门禁；
2. G1 MJCF 资产加载与单环境 MuJoCo 步进；
3. 64 个并行环境的静态图训练门禁；
4. `capture_while` 条件求解器与 `capture_if` 休眠岛分支的等价性验证；
5. 逐步执行、静态图和条件图三组吞吐对比；
6. 再把并行环境扩展到 1024 或设备能够稳定承载的规模。

条件图的加速结果必须同时满足物理状态误差阈值。只看到吞吐上升、但机器人位置或接触状态持续漂移，不能作为等价训练结果。

## 8. 评估指标

| 指标 | 含义 |
| --- | --- |
| safe episode［安全回合］ | 全程未与球碰撞且未跌倒 |
| minimum clearance［最小间距］ | 球轨迹与身体安全包络的最近距离 |
| inference latency［推理延迟］ | 每次 CBF/策略调用耗时 |
| moving joints［运动关节数］ | 策略是否产生全身体态而非平移替代 |
| dodge mode［避障模式］ | 下蹲、左移、右移或复合动作 |

## 9. 官方资料

- [PAC-MAN 源码](https://github.com/lzyang2000/perceptive_cbf_rl)
- [PAC-MAN 项目主页](https://lzyang2000.github.io/perceptive_cbf_rl/)
- [AMD Warp](https://github.com/AMD-Ecosystem/warp)
- [AMD MuJoCo Warp](https://github.com/AMD-Ecosystem/mujoco_warp)
- [AMD 配套脚本](https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite/tree/main/code/perceptive_cbf_rl_amd)
