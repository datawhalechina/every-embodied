# Notebook 实操入口

本目录保存 AMD ROCm 策略复刻专题的配套 Notebook。01–06 负责环境、指标和诊断；07–13 补齐从键盘采集、正式训练、MuJoCo closed-loop、pi0 strict-input 到 Pi0.5 EEF-delta/chunk 对齐诊断的完整执行链；14–16 按“一个模型一个 Notebook”的方式，把训练、长训日志、严格评估、视频和结论放到同一个文件里。Markdown 章节负责讲清楚概念、判断口径和实验结论，Notebook 负责逐格运行代码、生成配置、启动任务和整理结果表。

07–16 已逐格执行并保存输出。07–12 使用 AMD 教学环境，13 使用同一批 AMD 实验摘要在隔离 Jupyter 环境中复算表格和图像，14–16 使用已经完成的模型重建证据生成端到端学习入口。交互式键盘采集、正式长训练和批量 closed-loop 仍由 `RUN_*` 开关控制，仓库里展示的是路径审计、真实数据摘要、已完成训练进度、严格成功率、关键帧和四视角视频，不会在读者打开 Notebook 时自动启动数小时任务。

建议从仓库根目录启动 Jupyter。这样 Notebook 的路径发现不会依赖当前文件浏览器所在目录：

```bash
cd /path/to/every-embodied
jupyter lab --notebook-dir .
```

启动后从文件浏览器进入 `16-专题组队学习/04-AMD-ROCm策略复刻专题/notebooks/`，不要把服务器根目录设成专题的上一级临时目录。

如果在自己的 AMD 设备或远端服务器上运行，请先按实际情况设置：

```bash
export AMD_TOPIC_ROOT=/path/to/every-embodied/16-专题组队学习/04-AMD-ROCm策略复刻专题
export PROJECT_ROOT=/path/to/04mujoco复现ACT、Pi0、SmolVLA
export DATA_ROOT=/path/to/datasets/every_embodied
export OUTPUT_ROOT=/path/to/outputs
export MODEL_ROOT="$PROJECT_ROOT/ckpt"
```

只有在训练、闭环评估或采集单元中才需要 `PROJECT_ROOT`、`DATA_ROOT` 和模型目录；01–06 的路径审计、指标读取和分支内视频预览可以直接运行。分支中没有复制完整 MuJoCo 工程，未设置 `PROJECT_ROOT` 时训练单元会明确提示缺失，而不会访问 `/path/to/...` 这种模板目录。上游工程的目录要求和自包含发布建议见 [`../external/README.md`](../external/README.md)。

| Notebook | 对应章节 | 主要用途 |
| --- | --- | --- |
| [01_device_env_check.ipynb](./01_device_env_check.ipynb) | 01 设备与环境确认 | 检查 ROCm、PyTorch、显存、温度和目录规划 |
| [02_physical_success_review.ipynb](./02_physical_success_review.ipynb) | 02 物理成功评估 | 理解 `physical_success`，复核成功/失败关键帧 |
| [03_act_dagger_diagnostics.ipynb](./03_act_dagger_diagnostics.ipynb) | 03 ACT 诊断 | 查看 ACT 进展曲线，整理 DAgger 评估命令 |
| [04_smolvla_weighted_sampling.ipynb](./04_smolvla_weighted_sampling.ipynb) | 04 SmolVLA 加权采样 | 比较红/蓝杯成功率，重新生成图表 |
| [05_pi0_smoke_gate.ipynb](./05_pi0_smoke_gate.ipynb) | 05 pi_0 训练门控 | 检查 gated 权限、1-step smoke 和训练命令模板 |
| [06_rocm_debug_playbook.ipynb](./06_rocm_debug_playbook.ipynb) | 06 排障复盘 | 按“现象、证据、根因、修复、验证”整理问题 |
| [07_data_collection_and_audit.ipynb](./07_data_collection_and_audit.ipynb) | 07 数据采集 | 键盘采集、严格保存门槛、四视角录像、20 条已有数据实测审计 |
| [08_act_training_rocm.ipynb](./08_act_training_rocm.ipynb) | 08 ACT 训练 | 生成 smoke/full 配置、启动训练、查看 5000-step 历史进度和日志 |
| [09_smolvla_training_rocm.ipynb](./09_smolvla_training_rocm.ipynb) | 09 SmolVLA 训练 | 完成基础权重加载、训练配置、历史进度和分指令检查 |
| [10_pi0_training_rocm.ipynb](./10_pi0_training_rocm.ipynb) | 10 pi_0 训练 | 检查 gated 权限，完成 smoke/full 配置和历史训练结果复盘 |
| [11_mujoco_closed_loop_deploy.ipynb](./11_mujoco_closed_loop_deploy.ipynb) | 11 闭环部署 | 运行 checkpoint，保存 JSONL、严格成功率、关键帧和四视角视频 |
| [12_pi0_strict_input_end_to_end.ipynb](./12_pi0_strict_input_end_to_end.ipynb) | 12 pi0 strict-input | 对照 raw/learned head、固定/随机环境，复现最终严格判定 |
| [13_pi05_random_position_eef_delta.ipynb](./13_pi05_random_position_eef_delta.ipynb) | 13 Pi0.5 EEF-delta | 审计阶段方向、coherent recovery、parquet 行序、prefix DAgger、action chunk 和全新 seed strict 结果 |
| [14_smolvla_end_to_end.ipynb](./14_smolvla_end_to_end.ipynb) | SmolVLA 端到端 | 一个 Notebook 内完成 SmolVLA 零训练预览、长训日志、严格评估、57/60 结果和成功/失败视频 |
| [15_pi0_end_to_end.ipynb](./15_pi0_end_to_end.ipynb) | Pi0 端到端 | 一个 Notebook 内完成 Pi0 权限检查、保护式训练、长训日志、12/14 评估、失败分析和诊断视频 |
| [16_act_end_to_end.ipynb](./16_act_end_to_end.ipynb) | ACT 端到端 | 一个 Notebook 内完成 ACT smoke、保护式训练配方、长训日志和严格评估；当前可审计保护候选为 15/30，历史 17/30 作为参考 |

第一次从零学习时，推荐顺序是 `01 → 14`。14 先让学习者看到 SmolVLA 正确行为、长训日志和严格评估口径，再进入 `07 → 09 → 11` 自己采集、训练和评估。想理解失败诊断时，再读 `15 → 16 → 02/03/05/06`。如果已经有数据和 checkpoint，可以直接从 14–16 的端到端 Notebook 进入对应模型，再回到 07–13 查看拆分章节。

预训练权重、严格评测结果和下载状态见 [预训练权重与零训练体验](../README_08_预训练权重与零训练体验.md)。没有公开下载链接的历史 checkpoint 不应写进运行命令；先使用仓库内置视频，或训练自己的可复核权重。

07 的交互采集和 11 的可视化 rollout 需要可用的 `DISPLAY`；08–10 的长训练需要足够的模型缓存、checkpoint 空间和稳定电源。所有 `RUN_*` 开关默认关闭，先检查命令和路径，再显式打开。

## 重新执行与验收

AMD 教学镜像如果没有安装 `jupyter`、`nbclient` 或 `ipykernel`，可以使用仓库里的轻量执行器。它按 Notebook 顺序共享命名空间，执行普通 Python code cell，并把 stdout/stderr 写回标准 `.ipynb` outputs；交互采集和长训练仍由 Notebook 内的开关决定。

```bash
python code/execute_tutorial_notebooks.py \
  notebooks/07_data_collection_and_audit.ipynb \
  notebooks/08_act_training_rocm.ipynb \
  notebooks/09_smolvla_training_rocm.ipynb \
  notebooks/10_pi0_training_rocm.ipynb \
  notebooks/11_mujoco_closed_loop_deploy.ipynb \
  notebooks/12_pi0_strict_input_end_to_end.ipynb \
  notebooks/13_pi05_random_position_eef_delta.ipynb \
  notebooks/14_smolvla_end_to_end.ipynb \
  notebooks/15_pi0_end_to_end.ipynb \
  notebooks/16_act_end_to_end.ipynb
```

当前提交的执行结果是：07 为 `11/11` code cells、08 为 `11/11`、09 为 `11/11`、10 为 `12/12`、11 为 `10/10`、12 为 `10/10`、13 为 `13/13`；新增端到端 Notebook 中，14 为 `11/11`、15 为 `9/9`、16 为 `9/9`。公开 Notebook 使用 `$PROJECT_ROOT` 等变量表达机器相关路径，不包含远端用户名、私有 IP 或本机绝对目录。
