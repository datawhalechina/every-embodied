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
