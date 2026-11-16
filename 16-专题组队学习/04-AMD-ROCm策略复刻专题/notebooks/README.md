# Notebook 实操入口

本目录包含三类 Notebook：运行模板、已保存输出的教学 Notebook，以及依赖上游工程和本地模型的端到端工作流。类型不同，执行状态也不同。

## 运行前设置

```bash
export PROJECT_ROOT=/path/to/mujoco_pnp
export DATA_ROOT=/path/to/datasets/every_embodied
export MODEL_ROOT=/path/to/checkpoints/every_embodied
export RUN_ROOT=/path/to/runs/every_embodied
export OUTPUT_ROOT=$RUN_ROOT/notebooks

cd /path/to/04-AMD-ROCm策略复刻专题
jupyter lab
```

`PROJECT_ROOT` 指向上游 MuJoCo 工程。数据、模型和输出目录放在专题仓库之外。

## Notebook 分类

### 01–06：运行模板

01–06 保存检查代码、图表逻辑和命令模板，提交版本没有预先执行。读者在自己的设备上运行后得到当前环境和当前模型的结果。

| Notebook | 用途 |
| --- | --- |
| [01_device_env_check.ipynb](./01_device_env_check.ipynb) | ROCm、PyTorch、显存、温度和目录 |
| [02_physical_success_review.ipynb](./02_physical_success_review.ipynb) | 物理成功条件和关键帧复核 |
| [03_act_dagger_diagnostics.ipynb](./03_act_dagger_diagnostics.ipynb) | ACT 曲线、动作和 DAgger 诊断 |
| [04_smolvla_weighted_sampling.ipynb](./04_smolvla_weighted_sampling.ipynb) | 红杯/蓝杯分组和采样权重 |
| [05_pi0_smoke_gate.ipynb](./05_pi0_smoke_gate.ipynb) | 访问权限、模型加载和短训练配置 |
| [06_rocm_debug_playbook.ipynb](./06_rocm_debug_playbook.ipynb) | 分层排障记录 |

### 07–13：带教学输出的拆分流程

07–13 已保存教学执行输出，可直接阅读表格和图像。交互采集、长训练和批量闭环由 `RUN_*` 环境变量控制，打开文件不会自动启动长任务。

| Notebook | 用途 | 保存状态 |
| --- | --- | --- |
| [07_data_collection_and_audit.ipynb](./07_data_collection_and_audit.ipynb) | 键盘采集、四视角录像和数据审计 | 11/11 代码单元已保存输出 |
| [08_act_training_rocm.ipynb](./08_act_training_rocm.ipynb) | ACT 短训练和正式配置 | 11/11 |
| [09_smolvla_training_rocm.ipynb](./09_smolvla_training_rocm.ipynb) | SmolVLA 训练与任务分组 | 11/11 |
| [10_pi0_training_rocm.ipynb](./10_pi0_training_rocm.ipynb) | Pi0 权限和训练配置 | 12/12 |
| [11_mujoco_closed_loop_deploy.ipynb](./11_mujoco_closed_loop_deploy.ipynb) | 模型加载、逐回合结果和视频 | 10/10 |
| [12_pi0_strict_input_end_to_end.ipynb](./12_pi0_strict_input_end_to_end.ipynb) | Pi0 输入与环境随机化检查 | 10/10 |
| [13_pi05_random_position_eef_delta.ipynb](./13_pi05_random_position_eef_delta.ipynb) | EEF-delta、动作块和恢复数据 | 13/13 |

### 14–16：端到端工作流

14–16 将模型配置、训练、评估和视频放在同一个文件中。仓库保存参数和历史教学输出；真正执行训练与评估时，必须提供上游工程、数据和模型目录。

| Notebook | 模型 | 运行说明 |
| --- | --- | --- |
| [14_smolvla_end_to_end.ipynb](./14_smolvla_end_to_end.ipynb) | SmolVLA | 设置 `RUN_LONG_TRAIN=1` 启动训练，设置 `RUN_EVAL=1` 评估本次模型 |
| [15_pi0_end_to_end.ipynb](./15_pi0_end_to_end.ipynb) | Pi0 | 需要已获授权的基座与匹配数据 |
| [16_act_end_to_end.ipynb](./16_act_end_to_end.ipynb) | ACT | 普通训练与 repair 配方由 `ACT_RECIPE` 选择 |

端到端 Notebook 的默认开关为关闭状态。未提供上游路径时，配置和说明单元可以运行，训练和评估单元会显示所需变量。

## 普通训练与保护训练

[workflows/README.md](./workflows/README.md)提供普通训练和保护训练的独立副本。两类实验使用不同的 `$RUN_ROOT` 子目录，评估读取当前 Notebook 本次训练产生的模型。

## 重新执行

安装 Jupyter 内核后优先使用 Jupyter 原生执行。教学镜像缺少 `nbclient` 时，可使用轻量执行器执行不含交互窗口的单元：

```bash
python code/execute_tutorial_notebooks.py \
  notebooks/07_data_collection_and_audit.ipynb \
  notebooks/08_act_training_rocm.ipynb \
  notebooks/09_smolvla_training_rocm.ipynb \
  notebooks/10_pi0_training_rocm.ipynb \
  notebooks/11_mujoco_closed_loop_deploy.ipynb \
  notebooks/12_pi0_strict_input_end_to_end.ipynb \
  notebooks/13_pi05_random_position_eef_delta.ipynb
```

长训练期间，进度条在同一输出区域刷新；逐步指标写入 `metrics.jsonl`，避免为每一步增加一个 Notebook 输出块。

## 验收

完成一个模型的端到端流程后检查：

1. 训练配置中的数据、观测和动作与评估环境一致；
2. 评估模型路径位于本次运行的 `checkpoints/`；
3. 逐回合结果数量等于配置中的评估分母；
4. 成功/失败视频可播放，并能定位对应结果行；
5. Notebook 中没有异常输出和机器私有绝对路径。


## 完整实训与保存输出的新入口

| 笔记本 | 内容 |
| --- | --- |
| [00：完整实训与结果导航](./00_amd_完整实训与结果导航.ipynb) | 数据流、正式配方、两项结果、环境分工、磁盘与内存、恢复和验收 |
| [17：厨房完整实训](./17_robocasa365_amd_train_eval.ipynb) | 全部数据、3000 步、基座与最终权重各 20 回合、曲线、逐种子对照及嵌入视频 |
| [18：灵巧手完整实训](./18_dexjoco_amd_train_eval.ipynb) | 全数据统计、30000 步、恢复、基座与最终权重各 20 回合、曲线和视频 |

三本都保存真实证据的阅读输出。打开即可查看；重新执行默认只读取 `results/`，不启动设备负载。单任务完整复现时设置 `AMD_RUN_FULL=1`，配置新运行目录及各设备路径，再从头执行；资源下载单元需单独显式启用。两项串行完整复现也可用入口笔记本的清单单元，禁止与单任务入口重复运行同一目录。

灵巧手必须根据检查点元数据保留任务适配分支，并使用 `policy_loader_contract=strict-adapters-v2` 禁止静默丢弃参数。每次评估保存实际加载审计、预热及随机数恢复记录。`results/dexjoco/` 使用官方异步时序；`results/dexjoco_clock/` 是独立部署条件，只在等待新动作时暂停仿真，视频不展示等待墙钟时间。两种条件不得合并分母或相互替代。

复现独立时钟条件时，在第 18 本完成正式训练后设置 `AMD_RUN_CLOCK_COMPARISON=1`，以 `AMD_CLOCK_RUN_ROOT` 指定新运行目录。该分支引用已有最终权重，分别运行起始和训练后权重各 20 个固定种子，不重复训练。逐回合等待和墙钟时间保存在 `clock_audit.json`，不代表设备推理速度提升。

`results/` 保存来源配置、指标、逐回合结果、内容指纹和代表视频；原始数据、大模型及完整检查点不打入课程仓库。灵巧手训练尚未结束时只展示实际进度，独立的 20 步整本执行视频标记为执行验收，不填入正式成功率。正式结束后刷新结果快照并执行阅读模式，保存最终输出。

三本阅读模式依赖交互内核中的固定笔记本组件及图表库；完整设备执行额外需要相应训练与模拟器环境。详细环境约定已嵌入两本训练笔记本，[配套说明](../README_17_AMD两类基准训练评估实操.md)保留为补充。

发布方式参照 MicroDuck 专题：章节短说明与代码、结果交替；发布副本保留动态图预览、完整视频相对链接、来源与权重记录。完整视频和预览位于 `results/`，清单见 `recordings/manifest.json`。执行版保留原始内核输出，发布版不重复嵌入视频数据。动态图降低显示帧率但使用完整原视频；它不会启动新的推理。
