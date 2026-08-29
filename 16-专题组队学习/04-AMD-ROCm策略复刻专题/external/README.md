# 上游 MuJoCo 运行工程

采集、训练和闭环 Notebook 使用 `PROJECT_ROOT` 连接上游 MuJoCo 工程。专题仓库保存教程、轻量脚本和短示例媒体，不复制数据集、模型缓存和完整训练输出。

## 推荐方式

直接指向 Every Embodied 中的上游课程目录：

```bash
export EVERY_EMBODIED_ROOT=/path/to/every-embodied
export PROJECT_ROOT="$EVERY_EMBODIED_ROOT/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA"
```

上游入口：[LeRobot MuJoCo 训练 ACT、SmolVLA、Pi0](https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA)。

## 最小检查

不同上游版本的文件布局可能变化。当前工作流至少需要：

```text
$PROJECT_ROOT/
├── asset/example_scene_y2.xml
├── mujoco_env/
├── train_model.py
└── eval_policy_success.py
```

运行：

```bash
test -f "$PROJECT_ROOT/asset/example_scene_y2.xml"
test -d "$PROJECT_ROOT/mujoco_env"
test -f "$PROJECT_ROOT/train_model.py"
test -f "$PROJECT_ROOT/eval_policy_success.py"
```

专题自带的批量入口位于 `code/run_closed_loop.py`，无需复制到上游目录。

## 数据与模型

```bash
export DATA_ROOT=/path/to/datasets/every_embodied
export MODEL_ROOT=/path/to/checkpoints/every_embodied
export RUN_ROOT=/path/to/runs/every_embodied
```

| 目录 | 内容 |
| --- | --- |
| `PROJECT_ROOT` | 上游源码、场景和小型配置 |
| `DATA_ROOT` | LeRobot 数据和元数据 |
| `MODEL_ROOT` | 下载或训练得到的模型 |
| `RUN_ROOT` | 日志、逐回合评估和视频 |

这些大文件目录不放入教程 Git 仓库。更换机器时重新设置环境变量即可。

## 独立运行包

需要制作独立课程包时，只复制当前任务依赖的源码、场景 XML、网格和配置到 `external/mujoco_pnp/`。Notebook 中仍通过 `PROJECT_ROOT` 指向该目录：

```bash
export PROJECT_ROOT=/path/to/04-AMD-ROCm策略复刻专题/external/mujoco_pnp
```

复制后重新执行设备检查、环境导入和一次 `reset → step → render`，确认依赖完整。
