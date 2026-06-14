# Examples：快速体验 Every-Embodied

本目录提供面向新手的快速体验示例，目标是 **低门槛、跨平台、可交互**。

## 1. 推荐示例

### `01_hello_every_embodied_mujoco.py`

一个项目宣传风格的 MuJoCo 交互 Demo，包含：

- 启动后显示 `Hello Every-Embodied`
- 输入 `1/2/3/4` 触发不同动作：
  - `1`：机械臂打招呼（Wave）
  - `2`：趣味动作（Dance）
  - `3`：随机方块抓取与放置（Random Pick-and-Place）
  - `4`：自动连播模式（录屏更方便）
- 采用轨迹规划 + IK：
  - 优先使用 `ruckig`（jerk 限制轨迹）
  - 未安装时自动回退到 quintic 插值
- 抓取稳定性增强：
  - 末端姿态约束（抓取时保持更自然的朝向）
  - 随机点可达性与简单碰撞筛选（先筛后抓）
  - 放置防弹飞：落桌稳定阶段 + 先张爪再抬升撤离
- 不依赖大模型或 GraspNet，适合教学与快速演示
- MuJoCo 界面 Branding：
  - 窗口标题显示 `hello_every_embodied`
  - 场景中启用 site label，可见 `hello_every_embodied` 标签

## 2. 运行方式

推荐使用 Python 3.10 或更高版本创建独立环境。当前未指定版本的 `mujoco` 最新包要求 Python 3.10+；如果必须使用 Python 3.8，请安装 `mujoco==3.1.6`，并注意可选的 `ruckig` 当前要求 Python 3.9+。

在项目根目录执行：

```bash
# 必需依赖
pip install mujoco

# 可选（推荐）：更平滑、更工程化的轨迹规划
pip install ruckig

python examples/01_hello_every_embodied_mujoco.py
```

如果图形界面无法启动（例如远程无显示环境），可以使用：

```bash
python examples/01_hello_every_embodied_mujoco.py --headless
```

加速执行（关闭实时 sleep）：

```bash
python examples/01_hello_every_embodied_mujoco.py --fast
```

自动连播（用于生成视频）：

```bash
python examples/01_hello_every_embodied_mujoco.py --autoplay --autoplay-rounds 3
```

## 3. 为什么先用 MuJoCo？

对“快速体验 + 跨平台”场景，MuJoCo 的优势是：

- 安装轻：`pip install mujoco` 即可
- 跨平台：Windows / Linux / macOS 均可用
- 物理仿真稳定，适合做轨迹规划与控制演示

后续如需更复杂任务（视觉感知、并行训练、大规模场景），可再扩展到 Isaac Sim / ManiSkill / Habitat 等环境。
