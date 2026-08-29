# 06 ROCm 排障手册

本章按运行层、数据层、模型层和闭环层组织排障。每次只改变一个变量，并把现象、诊断命令、修复和验证结果写入运行目录。完整案例见 [ROCm 排障案例索引](./appendices/ROCM_DEBUG_CASES.md)。

## 1. 排障顺序

```text
设备可见
→ 基础张量计算
→ 数据读取
→ 模型加载
→ 单步前向与反向
→ 模型保存与恢复
→ 单回合闭环
→ 批量评估
```

上一步通过后再进入下一步。这样可以将运行时问题、数据问题和策略问题分开定位。

## 2. 设备与运行时

```bash
rocminfo | sed -n '1,100p'
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print("result:", float(y.square().mean()))
PY
```

同时记录：

```bash
rocm-smi --showproductname --showuse --showmemuse --showtemp --showpower
df -h "$WORK_ROOT"
```

设备可见但张量计算失败时，优先核对 PyTorch 构建、ROCm 用户态库和设备权限。环境变量只在运行命令前设置，避免在 Python 导入运行时之后切换缓存或库路径。

## 3. 权重加载

大模型加载失败通常发生在三个位置：文件不完整、CPU 反序列化或 GPU 搬运。按顺序检查：

```bash
test -s "$POLICY_PATH/model.safetensors"
test -f "$POLICY_PATH/config.json"
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import os

path = Path(os.environ["POLICY_PATH"]) / "model.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("tensor_count:", len(keys))
print("first_keys:", keys[:5])
PY
```

先在 CPU 打开权重，再由模型接口迁移到 GPU。加载后运行一次有限值检查：

```python
for name, value in model.state_dict().items():
    if value.is_floating_point() and not value.isfinite().all():
        raise RuntimeError(f"non-finite tensor: {name}")
```

## 4. 数据与动作块

LeRobot 数据重点检查回合边界和物理行序：

```python
required = ["episode_index", "frame_index", "index", "action"]
for key in required:
    assert key in table.column_names, key

for episode in sorted(set(table["episode_index"])):
    rows = table.filter(table["episode_index"] == episode)
    frames = rows["frame_index"]
    assert frames == list(range(len(frames)))
```

合并多个数据集后，重新生成全局 `index`，并验证每个动作块的第一项等于当前物理行的动作。只检查索引集合是否齐全，无法发现行顺序错配。

## 5. 单步训练

单步训练需要输出以下指标：

```text
loss
learning_rate
grad_norm
param_norm
GPU memory allocated / reserved
step time
checkpoint path
```

判定规则：

| 现象 | 优先检查 |
| --- | --- |
| 损失为 NaN/Inf | 输入范围、归一化、混合精度和学习率 |
| 梯度全为零 | 冻结配置、损失连接和参数分组 |
| 显存持续增长 | 计算图引用、视频帧缓存和日志对象 |
| 保存失败 | 磁盘、临时目录和输出目录权限 |
| 恢复后步数归零 | 加载的训练配置、优化器状态和模型路径 |

恢复训练后，第一条日志应延续上次保存步数。模型、优化器、学习率调度器和随机状态使用同一个保存点。

## 6. 无头渲染与视频

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python code/demo/demo_ball.py
ffprobe -v error -show_streams outputs/demo_videos/demo_ball.mp4
```

如果 EGL 不可用，可在独立显示会话中使用 Xvfb：

```bash
xvfb-run -a -s '-screen 0 1920x1080x24' python code/demo/demo_arm.py
```

视频采用 H.264 和 `yuv420p`，浏览器兼容性更稳定。渲染测试只确认环境、相机和编码链路；策略结果由闭环评估生成。

## 7. 闭环行为

将每个回合拆为接近、接触、夹取、抬升、搬运、放置和释放。逐阶段记录状态比只看最终布尔值更容易定位问题。

| 失败阶段 | 常见来源 | 需要查看的字段 |
| --- | --- | --- |
| 接近 | 图像定位、动作方向、坐标系 | TCP 与目标距离、动作符号 |
| 接触 | 控制频率、动作幅值、时间对齐 | 接触数、末端速度、帧时间 |
| 夹取 | 夹爪标度和训练分布 | 夹爪命令、物体相对位姿 |
| 抬升 | 动作块后半段、接触稳定性 | 最大高度、连续抬升帧 |
| 放置 | 目标定位、释放时机 | 目标距离、姿态、释放帧 |

固定随机种子用于版本对比，未参与训练的随机种子用于估计泛化。逐回合 JSON 和视频必须来自同一次运行。

## 8. 性能记录

```bash
rocm-smi --showuse --showmemuse --showtemp --showpower --json \
  > "$RUN_ROOT/device_snapshot.json"
```

训练或评估报告记录：

- GPU 型号、ROCm 和框架版本；
- batch size、精度、分辨率和相机数量；
- 每步耗时、吞吐和峰值显存；
- 视频开启与关闭时的评估耗时；
- 实际修改过的配置项。

性能对比使用相同任务、模型、精度、批大小和评估分母。

## 9. 问题记录模板

```markdown
### 问题名称

- 环境：GPU、ROCm、框架和上游版本
- 输入：数据集、模型和运行命令
- 现象：首个错误和发生阶段
- 诊断：最小复现与关键指标
- 根因：代码、数据、运行时或资源
- 修复：修改文件和配置
- 验证：重新执行的命令与输出
- 产物：日志、模型、逐回合结果和视频
```

同一问题的后续尝试追加在一条记录中，避免将终端流水账散落在多个章节。

## 10. 深入案例

[ROCm 排障案例索引](./appendices/ROCM_DEBUG_CASES.md)保留 ACT、SmolVLA、Pi0/Pi0.5 的数据行序、模型恢复、视觉参数、动作表示和闭环诊断。阅读时先用本章确定问题所属层级，再跳到对应案例。
