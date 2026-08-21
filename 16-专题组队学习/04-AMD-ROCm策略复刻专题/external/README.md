# MuJoCo 上游运行工程

本目录是 AMD ROCm 专题的可选运行时入口。课程中的采集、ACT、SmolVLA、pi_0 和 MuJoCo 闭环 Notebook 使用 `PROJECT_ROOT` 指向上游 `mujoco_pnp` 工程；分支内的指标复核、视频预览和大多数诊断单元不依赖它。

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

## 当前分支的取舍

`amd-rocm` 是轻量教学分支，目前不直接复制完整上游目录。完整目录除了运行源码，还包含示例数据、评测视频、媒体文件和大量网格/纹理资产，原始 Git 跟踪内容约 194 MiB，工作目录还可能包含 checkpoint、缓存和实验输出。把整个工作目录压缩后提交到普通 Git，会重新引入大仓库 clone 和二进制版本管理问题。

本地已有主仓库时，可以直接把 `PROJECT_ROOT` 指向上游目录：

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
```

如果要发布“克隆后即可运行”的 AMD 课程包，建议只整理运行时源码、当前场景所需资产和最小配置到 `external/mujoco_pnp`；数据集、checkpoint、缓存、日志和生成视频放到外部磁盘。超过 GitHub 普通文件/仓库体量边界的完整包，应使用 Git LFS 或 GitHub Release 资产，并在这里记录固定版本和校验值。

上游课程入口：[LeRobot MuJoCo 训练 ACT、SmolVLA、pi_0](https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA)。
