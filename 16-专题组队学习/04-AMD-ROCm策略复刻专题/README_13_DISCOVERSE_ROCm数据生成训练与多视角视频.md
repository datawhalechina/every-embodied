# DISCOVERSE：ROCm 数据生成、训练与多视角视频

DISCOVERSE 是面向 Real2Sim2Real［真实到仿真再到真实］机器人学习的仿真框架，覆盖 AIRBOT、MMK2、专家状态机、模仿学习、LiDAR［激光雷达］与 3DGS［三维高斯泼溅］。本章从源码和模型下载开始，完成专家数据生成、LeRobot［机器人学习数据框架］转换、ACT/Diffusion Policy［扩散策略］训练、闭环评估和三路高清视频。

## 1. 下载和安装

```bash
cd "$SRC_ROOT"
git lfs install
git clone https://github.com/discoverse-dev/DISCOVERSE.git
cd DISCOVERSE

export DISCOVERSE_ENV="$ENV_ROOT/discoverse-rocm"
python3.10 -m venv "$DISCOVERSE_ENV"
source "$DISCOVERSE_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/setup_submodules.py
python scripts/check_installation.py
```

按用途增加依赖：

```bash
python -m pip install -e '.[act_full]'
python -m pip install -e '.[lidar,visualization]'
python -m pip install -e '.[gs]'
```

3DGS［三维高斯泼溅］模型由 `tatp/DISCOVERSE-models` 提供，首次运行会自动下载，也可提前登录：

```bash
hf auth login
```

## 2. 基础运行门禁

```bash
export MUJOCO_GL=egl
python discoverse/robots_env/airbot_play_base.py
python discoverse/robots_env/mmk2_base.py

python examples/mocap_ik/mocap_ik_manipulator.py \
  -r airbot_play \
  -t place_block \
  --record \
  --camera-names global_camera wrist_camera
```

迁移验收覆盖 18 个运行门禁、AIRBOT 12 个任务和 MMK2 8 个固定场景。基础门禁通过后，再生成训练数据。

## 3. 专家轨迹生成

官方任务脚本已经包含专家状态机。以 AIRBOT 放置积木为例：

```bash
python examples/tasks_airbot_play/place_block.py --help
```

不同版本的参数名可能变化，先以 `--help` 为准。批量生成时固定数据根目录、seed［随机种子］范围和目标成功数：

```bash
export DISCOVERSE_DATA_ROOT="$DATA_ROOT/discoverse"
mkdir -p "$DISCOVERSE_DATA_ROOT/place_block"

for seed in $(seq 0 499); do
  python examples/tasks_airbot_play/place_block.py \
    --data_idx "$seed" \
    --data_set_size 1 \
    --auto
done

rsync -a data/place_block/ "$DISCOVERSE_DATA_ROOT/place_block/"
```

官方脚本先写入 DISCOVERSE 源码目录下的 `data/place_block`；最后一行把通过成功条件后实际保存的回合增量同步到大容量数据盘。再次执行相同 seed［随机种子］前，先检查目标回合是否已经存在，避免重复生成。

每条轨迹至少检查：

- 仿真成功条件为真；
- 状态、动作和相机帧数量一致；
- 时间戳严格递增；
- 末尾动作 padding［填充］不跨 episode［回合］；
- 回放仍能完成抓取、运输和放置。

我们归档的 `place_block` 专家语料为 500 个通过验收的 episode［回合］、44,439 个对齐帧。

## 4. 转换为训练数据

DISCOVERSE 原始数据通常包含 `obs_action.json`、相机图像或视频和任务元数据。转换为 LeRobot［机器人学习数据框架］时建立统一字段：

```text
observation.state
observation.images.{camera}
action
task
episode_index
frame_index
timestamp
```

先对单条轨迹转换，再批量执行：

```bash
python "$EVERY_EMBODIED_ROOT/16-专题组队学习/04-AMD-ROCm策略复刻专题/code/convert_discoverse_to_lerobot.py" \
  --input-root "$DISCOVERSE_DATA_ROOT/place_block" \
  --output-root "$DATA_ROOT/discoverse-lerobot/place_block" \
  --repo-id local/discoverse-place-block \
  --task 'Place the green block in the pink bowl' \
  --fps 20 \
  --image-size 256
```

转换器读取每个成功回合的 `obs_action.json`、`cam_0.mp4` 和 `cam_1.mp4`，校验两路图像、7 维关节状态与 7 维动作逐帧等长。DISCOVERSE 在执行动作后记录观测，因此默认 `--action-bias=-1`，把观测时刻与下一条动作对齐。转换结果中的 `discoverse_conversion_manifest.json` 记录源回合、帧数和动作索引范围。不要仅比较数组 shape［形状］，还要回放动作验证时序。

## 5. ACT 训练

```bash
lerobot-train \
  --dataset.repo_id=local/discoverse-place-block \
  --dataset.root="$DATA_ROOT/discoverse-lerobot/place_block" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --steps=100000 \
  --batch_size=32 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/discoverse/act_place_block"
```

在 ROCm［AMD 开放计算平台］上 `--policy.device=cuda` 仍是正确写法。先用 10–50 steps［训练步数］验证数据与反向传播，再启动正式长训。

## 6. Diffusion Policy 训练

```bash
lerobot-train \
  --dataset.repo_id=local/discoverse-place-block \
  --dataset.root="$DATA_ROOT/discoverse-lerobot/place_block" \
  --policy.type=diffusion \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/discoverse/diffusion_place_block"
```

具体策略名称以所安装 LeRobot［机器人学习数据框架］版本的 `lerobot-train --help` 为准。训练配置必须保存观测映射、动作归一化和时间窗口。

## 7. 闭环评估

评估循环必须让策略动作真实推进模拟器：

```python
observation = env.reset()
for step in range(max_steps):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    recorder.append(observation, action, info)
    if terminated or truncated:
        break
```

固定 train-like［训练分布相似］与 unseen［未见分布］两组 seed［随机种子］，分别统计接近、抓取、抬升、运输、对齐和最终放置。评估结果与视频写入同一回合目录。

## 8. 多视角高清输出

MMK2 的 `kiwi_pick.py` 默认已经记录 3 个相机，但分辨率为 640×480。复制任务脚本并只调整录像分辨率，可以保留官方专家状态机：

```bash
cd "$SRC_ROOT/DISCOVERSE"
mkdir -p "$RUN_ROOT/discoverse/mmk2_kiwi_hd"
cp examples/tasks_mmk2/kiwi_pick.py \
  "$RUN_ROOT/discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py"

python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / "discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py"
text = path.read_text()
text = text.replace('"width"  : 640', '"width"  : 1920')
text = text.replace('"height" : 480', '"height" : 1080')
path.write_text(text)
PY

python "$RUN_ROOT/discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py" \
  --data_idx 0 \
  --data_set_size 1 \
  --auto
```

任务成功后，三路原片位于 `data/pick_kiwi/000/cam_0.mp4`、`cam_1.mp4` 和 `cam_2.mp4`。脚本只在成功条件成立后写盘，因此这些视频与对应的专家动作属于同一回合。

保持单相机原片，同时生成三视角合成版：

```bash
ffmpeg \
  -i data/pick_kiwi/000/cam_0.mp4 \
  -i data/pick_kiwi/000/cam_1.mp4 \
  -i data/pick_kiwi/000/cam_2.mp4 \
  -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[v]' \
  -map '[v]' -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$RUN_ROOT/discoverse/mmk2_kiwi_hd/composite.mp4"
```

## 9. 3DGS 渲染

```bash
python -m pip install -e '.[gs]'
hf auth login
```

在支持 Gaussian renderer［高斯渲染器］的场景中设置 `cfg.use_gaussian_renderer = True`，交互窗口用 `Ctrl+g` 切换。验证时保存：

- 实际加载的 `.ply` 点云路径；
- 点数量与渲染分辨率；
- 至少一段动态回放；
- 同任务 MuJoCo 网格渲染作为对照。

## 10. AMD 迁移重点

| 模块 | 迁移内容 |
| --- | --- |
| PyTorch［张量计算框架］策略 | 保留 `torch.cuda` 接口，安装 ROCm wheel［AMD 软件包］ |
| MuJoCo | 使用 EGL［无窗口图形渲染接口］输出视频 |
| 3DGS［三维高斯泼溅］ | 替换 CUDA 专用扩展或使用 ROCm 可编译实现 |
| 数据 | 修正 episode［回合］边界、动作 chunk［动作块］和相机时序 |
| ACT | 保持训练与执行的 chunk［动作块］长度一致 |
| 多机器人 | 分别验证机器人 MJCF、控制器和成功条件 |

## 11. 结果口径

运行门禁、专家成功率和学习策略成功率是三个独立指标。DISCOVERSE 的公开迁移证据包括运行 `18/18`、AIRBOT `12/12`、MMK2 `8/8`、高清任务 `4/4` 和 500 条专家轨迹；训练模型需要使用自己的闭环分母另行汇总。

## 12. 官方资料

- [DISCOVERSE 源码](https://github.com/discoverse-dev/DISCOVERSE)
- [DISCOVERSE 项目主页](https://air-discoverse.github.io/)
- [DISCOVERSE 模型](https://huggingface.co/tatp/DISCOVERSE-models)
