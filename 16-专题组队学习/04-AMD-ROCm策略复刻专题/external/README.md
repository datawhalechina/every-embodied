# MuJoCo 上游运行工程

本目录是 AMD ROCm 专题的可选运行时入口。课程中的采集、ACT、SmolVLA、pi_0 和 MuJoCo 闭环 Notebook 使用 `PROJECT_ROOT` 指向上游 `mujoco_pnp` 工程；指标复核、视频预览和大多数诊断单元不依赖本目录中的完整副本。

## 期望目录

如果要让训练、采集和闭环单元在专题目录内直接运行，可以把经过筛选的上游工程放到：

```text
16-专题组队学习/04-AMD-ROCm策略复刻专题/external/mujoco_pnp/
```

至少需要确认这些入口存在：

```text
asset/example_scene_y2.xml
mujoco_env/
train_model.py
eval_policy_success.py
code/run_closed_loop.py
```

不同 Notebook 还会读取场景 XML、机器人模型、LeRobot 适配代码和少量第三方模块。最终以 Notebook 的路径审计结果为准；缺少工程时，它会明确打印缺失路径，不会回退到作者机器上的绝对目录。

## 源码、资产与实验数据分层

教程目录不重复复制完整上游工程。完整工程除了运行源码，还包含示例数据、评测视频、媒体文件和大量网格/纹理资产，工作目录还可能包含 checkpoint、缓存和实验输出。源码与小型配置保存在 Git 中；数据集、checkpoint、缓存、日志和批量视频通过外部目录或模型仓库连接。

这里需要区分两个数字：本机当前工作目录约 3.69 GiB，但这不是 GitHub 上游代码仓库的体积。当前工作区的实测构成为：

| 目录 | 约占空间 | 是否应随课程源码发布 | 内容 |
| --- | ---: | --- | --- |
| `third_party/` | 2.79 GiB | 否 | `mujoco_menagerie`、`teleop_xr`、`xarm_ros` 三个本地嵌套依赖仓库，且被 `.gitignore` 忽略 |
| `demo_data*` | 0.62 GiB | 否 | 本地采集的示教数据和实验数据集 |
| `ckpt/` | 0.19 GiB | 否 | 本地模型 checkpoint，通常来自训练或模型下载 |
| `asset/`、`mujoco_env/`、脚本 | 约 0.07 GiB | 视任务筛选 | MuJoCo 场景、机器人/物体网格、环境代码和运行脚本 |

本仓库已经包含上游课程快照，主要由场景资产、网格/纹理、少量示例数据/视频、Notebook 和 Python 代码组成。若只阅读 Markdown，可直接使用电子书；若要执行训练和闭环评估，再按下面的方式设置 `PROJECT_ROOT`。

本地已有主仓库时，可以直接把 `PROJECT_ROOT` 指向上游目录：

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
```

制作“克隆后即可运行”的独立课程包时，只需要整理运行时源码、当前场景所需资产和最小配置到 `external/mujoco_pnp`；数据集、checkpoint、缓存、日志和生成视频放到外部磁盘或公开模型仓库。

`mujoco_pnp` 本身来自本仓库的上游课程目录，不是 Hugging Face 上的独立代码仓库。Hugging Face 只可能涉及另外下载的模型权重或数据集，例如 SmolVLA、pi_0 的 gated 权重；这些不应和 `mujoco_pnp` 源码混在一起。

上游课程入口：[LeRobot MuJoCo 训练 ACT、SmolVLA、pi_0](https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA)。
