# 两类基准的真实训练与评估

本节把厨房操作和灵巧手操作分别连接到可执行的数据准备、训练、检查点恢复和官方仿真评估。每个运行目录只对应一套数据、基座和训练配方；结果由本次权重产生，不使用历史成绩填充。

## 入口

| 笔记本 | 训练方式 | 评估接口 |
| --- | --- | --- |
| [厨房操作](notebooks/17_robocasa365_amd_train_eval.ipynb) | GR00T［通用机器人策略模型］视觉语言投影层适配；冻结其余参数 | 官方厨房仿真客户端、环境成功条件、逐回合视频 |
| [灵巧手](notebooks/18_dexjoco_amd_train_eval.ipynb) | Pi0.5［机器人策略模型］原生低秩适配；模拟器与模型环境隔离 | 官方灵巧手客户端、固定随机种子、逐回合成功标记 |

辅助模块在 `code/amd_benchmarks/`。笔记本保留配置、执行入口、训练曲线、模型身份与视频；底层库的训练和仿真实现不复制进笔记本。

## 路径和解释器

```bash
export AMD_WORK_ROOT=/path/to/persistent/amd-course
export SRC_ROOT=$AMD_WORK_ROOT/src
export DATA_ROOT=$AMD_WORK_ROOT/datasets
export MODEL_ROOT=$AMD_WORK_ROOT/models
export RUN_ROOT=$AMD_WORK_ROOT/runs
export ENV_ROOT=/path/to/existing/environments
export TRAIN_PYTHON=$ENV_ROOT/robocasa365-rocm-py312/bin/python
```

灵巧手改用对应的数组计算环境解释器。模拟器可执行文件通过配置中的 `simulator_eval` 单独指定。不要将处理器版张量数据加载库安装到厨房策略的设备张量环境：灵巧手模型由 JAX［高性能数组计算框架］在设备上训练，处理器版张量库仅负责数据加载与视频解码。

已有设备可以用一个配置文件覆盖路径。厨房笔记本读取 `AMD_ROBOCASA_CONFIG`，灵巧手笔记本读取 `AMD_DEXJOCO_CONFIG`。这些文件只保存路径与训练参数，不含访问令牌。

## 源码与环境契约

厨房路径需要官方 `robocasa`、`robosuite` 和 RoboCasa365 配套的 `Isaac-GR00T`。训练器读取 `panda_omron` 数据配置，使用官方机器人元数据切分 16 维状态和 12 维动作。训练数据必须是相应任务的真实三相机数据。

灵巧手路径需要包含 `openpi/`、`configs/rand_obj/` 和模拟器客户端的 DexJoCo 工程。低秩适配配置来自该工程的 `openpi.training.config`，不得用其他机器人配置替代。`lerobot_source` 可以显式指向支持第三版数据格式的机器人学习框架源码；已有环境中的旧版框架不能读取此数据格式。

依赖安装复用各项目解释器。先运行预检，禁止用处理器回退通过设备验收。资源下载单元要求 `DOWNLOAD_ENTRY` 指向已配置的下载入口；源码、模型及数据来源分别见第 11、12 章。大文件放在持久化磁盘，不放在课程仓库。

## 固定版本与设备配置

`reproducibility/source-manifest.json` 记录源码提交及关键文件内容指纹；两个环境清单记录实机安装版本。这些清单用于比对，不是可直接整表安装的依赖锁；厨房环境含系统继承包，设备专用轮包也需要对应官方索引。新建环境须先按第 10–12 章安装设备计算栈，再按清单核对核心库，禁止直接安装上游默认的另一设备计算依赖。

| 组件 | 固定版本 |
| --- | --- |
| 厨房策略源码 | `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |
| 厨房仿真源码 | `b4684e6ee37d377cc392e98302a6b916d588b415` |
| 机器人仿真组件 | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| 灵巧手兼容分支源码 | `46fcd70db460b3be5487387adab2aaf3e2e7b74f` |
| 第三版数据读取源码 | `00b5f65752e1e5882f96c99a34e1d8b6d5d67f11` |

灵巧手参考提交包含新版检查点兼容修改，来自公开兼容分支，不能仅克隆官方主线后假设源码相同：

```bash
git clone https://github.com/Ethan-Chen-plus/dexjoco.git "$SRC_ROOT/dexjoco"
git -C "$SRC_ROOT/dexjoco" checkout 46fcd70db460b3be5487387adab2aaf3e2e7b74f
```

厨房核心版本为设备张量库 `2.12.1+rocm7.2`、视觉库 `0.27.1+rocm7.2`、模型库 `4.51.3`、扩散库 `0.30.2`。灵巧手采用数组库 `0.10.0`、设备插件 `0.10.0+rocm7.14.0`、神经网络库 `0.12.8`、检查点库 `0.12.1`，配套处理器版张量库 `2.9.0+cpu` 和视觉库 `0.24.0+cpu`。

灵巧手已有数组计算环境补齐数据加载依赖时，可通过专用下载入口安装：

```bash
"$DOWNLOAD_ENTRY" "$TRAIN_PYTHON" -m pip install \
  'torch==2.9.0+cpu' 'torchvision==0.24.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$DOWNLOAD_ENTRY" "$TRAIN_PYTHON" -m pip install \
  'datasets==4.0.0' 'pandas==2.3.3' 'pyarrow==21.0.0' \
  'av==15.1.0' 'fsspec==2025.3.0' 'dill==0.3.8' \
  'multiprocess==0.70.16' 'accelerate==1.10.1' \
  'wandb==0.19.11' 'protobuf==6.33.6'
```

课程辅助模块在子进程内适配神经网络库的参数遍历、数组库的自动分片模式，并关闭不适配该设备共享内存上限的编译优先融合阶段；不覆盖上游源码。厨房辅助模块使用数学注意力内核，并在模型库内适配注意力映射。更换核心版本须重新检查反向梯度是否有限。

两个 `*-device-profile.json` 配置指向已有设备路径，采用新的正式实验目录；它们是待运行配置，不代表已完成正式训练。启动厨房笔记本时设置：

```bash
export AMD_ROBOCASA_CONFIG="$PWD/reproducibility/robocasa-device-profile.json"
export TRAIN_PYTHON=/home/aup/envs/robocasa365-rocm-py312/bin/python
```

灵巧手改用 `AMD_DEXJOCO_CONFIG` 和对应解释器。启动目录应为本专题根目录；两本笔记本也支持从其 `notebooks/` 目录打开。交互内核与设备训练进程分开。`TRAIN_PYTHON` 选择训练解释器，交互内核可以复用厨房环境。给已有交互解释器安装固定组件：

```bash
"$DOWNLOAD_ENTRY" "$NOTEBOOK_PYTHON" -m pip install \
  'ipykernel==6.29.5' 'ipython==8.37.0' \
  'nbclient==0.10.2' 'nbformat==5.10.4' \
  'matplotlib==3.10.3' 'contourpy==1.3.2' 'kiwisolver==1.4.9'
```

旧系统图表轮包可能按旧数组库编译；不能仅检查模块是否存在，必须实际生成图表。

灵巧手还需预取语言分词资源。笔记本下载单元包含对应的公开云存储命令；`asset_cache` 指定其持久缓存目录，已有设备配置复用现存缓存。模型与数据阶段拒绝临时发起资源下载，缺缓存会明确报错。

灵巧手训练默认要求至少 24 GiB 可用系统内存。这是启动门槛，不是实测峰值或不会耗尽内存的保证；统一内存同时承载主机加载、设备参数、编译与优化器。按运行时峰值调整门槛，并保留系统余量。磁盘空闲检查同样只检查启动门槛。灵巧手正式实验默认要求 24 GiB 空闲磁盘，为原生检查点和原子保存时的新旧副本留空间；一份短程原生检查点约为 8.7 GiB。

可以使用配套的整本执行器验证单元顺序：

```bash
export TRAIN_PYTHON=/home/aup/envs/openpi-amd-jax010/bin/python
export JUPYTER_PATH=/home/aup/amd-notebook-work/jupyter/share/jupyter
/home/aup/envs/robocasa365-rocm-py312/bin/python code/execute_course_notebook.py \
  notebooks/18_dexjoco_amd_train_eval.ipynb \
  --profile reproducibility/dexjoco-device-profile.json \
  --kernel amd-course --enable-workloads --output /path/to/new/executed_18.ipynb
```

使用前把配置中的运行目录改为新目录，需要检查环境时，可在独立目录将目标总步数改为 2–20、评估选 1 个种子；该结果只用于执行验收。正式配置使用全部数据与完整步数，不能将短程结果作为正式实验。其他机器用自己的交互解释器及内核名称；输出路径必须是新文件。执行器默认不限制单元时长，支持长训练；可用 `--timeout` 显式设置秒数。执行失败仍保存当时输出并返回异常。此执行器不会自动启用资源下载。

## 执行阶段

1. `preflight`：检查必需文件、真实设备和矩阵乘法，不产生任务成功率。
2. `prepare`：记录基座文件与数据元数据身份；厨房数据保留视频时间原点，灵巧手重新计算归一化统计。
3. `train`：执行真实梯度更新并保存本次产物。配置中记录批量、学习率、步数及冻结范围。
4. `baseline`：用未修改的基座权重和原归一化资产评估；与最终权重使用相同种子及任务时长。
5. `evaluate`：读取 `trained_model.json` 指定的检查点；权重身份变化或逐回合分母不完整时失败。
6. 查看结果：从本次运行目录生成曲线，展示真实成功或失败视频。

同样的阶段可以在终端执行：

```bash
export PYTHONPATH=/path/to/04-AMD-ROCm策略复刻专题/code
"$TRAIN_PYTHON" -m amd_benchmarks.common --config /path/to/run.json --stage preflight
"$TRAIN_PYTHON" -m amd_benchmarks.common --config /path/to/run.json --stage prepare
"$TRAIN_PYTHON" -m amd_benchmarks.common --config /path/to/run.json --stage train
"$TRAIN_PYTHON" -m amd_benchmarks.common --config /path/to/run.json --stage baseline
"$TRAIN_PYTHON" -m amd_benchmarks.common --config /path/to/run.json --stage evaluate
```

长训练和评估默认关闭。启用后的单元会等待真实进程完成；异常会保留日志并让单元失败。首次环境检查可在独立目录短程运行；正式实验必须完成配置中的全步数和逐回合评估。笔记本默认执行最终权重评估；基座对照可用上述终端入口或串行执行器完成。

## 检查点与恢复

厨房模型使用组合检查点：冻结基座由文件哈希固定，增量文件保存可训练投影层、持久缓冲区、优化器和随机状态。重建策略时同时需要基座和增量，不能把增量当作完整模型。此方式避免为每次保存重复复制数吉字节冻结权重。训练仅更新 `action_head.vlln` 与 `action_head.vl_self_attention`，不等同于官方完整策略训练。

灵巧手使用上游原生检查点，包含模型参数、优化器状态与归一化资产。加载模型权重重新开一个实验，与完整中断恢复是两种操作。课程恢复器会按固定种子重放已消费批次数以恢复数据顺序，因此长训练恢复需要额外读取时间。实际恢复步骤必须与上游保存索引一致。

恢复时保持同一数据、基座、批量和学习率，设置 `resume=True`，将 `steps` 改成目标总步数。更改训练配方或数据时使用新目录。每次评估创建新的尝试目录，防止失败重试或换权重时串用旧结果。

## 产物

```text
run_root/
  identity.json
  preflight_status.json
  prepared/ 或 prepared.json
  metrics.jsonl
  checkpoint.pt 或 checkpoints/
  trained_model.json
  train_status.json
  eval_<weight-identity>_<attempt>/
    server.log
    results.jsonl
    summary.json
    视频文件
  evaluate_status.json
```

数据、权重、逐回合结果与视频必须来自同一运行。配置更改与失败尝试均保留记录。

## 方法边界

- 两本笔记本以单任务完整链路为主；多任务与双臂任务需要相应数据和动作维度，不能自动套用单臂检查点。
- 少量步数只验收真实梯度、保存和闭环接口，不证明策略收敛。
- 厨房任务默认采用官方完整时长；显式缩短时长会标记为链路验证。
- 不同任务、时长、随机种子或训练配方的成功率不可直接合并比较。
- 历史厨房比赛面板和灵巧手多任务预训练模型成绩，不是本次新训练模型的结果。

## 正式实验配方

设备配置使用以下配方，成功率必须由完整回合重新生成：

| 项目 | 厨房 | 灵巧手 |
| --- | --- | --- |
| 训练数据 | 全部 502 回合、131904 帧 | 全部 100 回合、27745 帧 |
| 目标更新步数 | 3000 | 30000 |
| 实际批量／累积／有效批量 | 1／1／1 | 1／1／1 |
| 学习率 | 0.00001 | 峰值 0.00001，预热 100 步 |
| 可训练范围 | 视觉语言投影相关层，201437184 个参数 | 上游冻结过滤器之外的 466957072 个参数，含视觉与低秩适配参数 |
| 保存间隔 | 500 步 | 1000 步，保留最新完整原生检查点 |
| 评估 | 基座、最终权重各 20 回合 | 基座、最终权重各 20 回合 |

两套均显式使用种子 100–119。厨房采用官方目标划分（独立厨房场景和物体，配置值 `target`）与完整 750 步上限，每个种子重新创建环境；灵巧手使用官方任务终止条件。厨房训练是投影层适配，不能宣称完整策略全参数复现。灵巧手冻结过滤器保留的可训练部分不仅包含低秩增量，不应称为仅低秩参数训练。

灵巧手归一化遍历全部 27745 帧，统计批量 5、读取进程 2；统计批量必须整除帧数，防止上游末批丢弃。训练批量仍为 1。统计资产固定后重复准备只校验并复用，不重复改变统计。

长任务可用 `code/amd_benchmarks/pipeline.py` 串行执行两套。已有设备示例为 `reproducibility/pipeline-device-manifest.json`。清单包含 `job_root`、`code_root` 与 `runs`；每个运行条目指定 `name`、`config`、`python`。各自配置必须用独立持久化目录。可选 `prerequisite_status` 用于要求整本笔记本通过；常规用户无需设置该字段。

```bash
python -m amd_benchmarks.pipeline --manifest /path/to/pipeline_manifest.json
```

执行器对同一任务目录加锁，逐阶段保存状态，失败即停止；修复后重新启动会复用已完成阶段并从已保存训练状态恢复。恢复时将检查点之后的未提交指标归档，只保留与检查点一致的指标序列。最终验收要求完整连续步数、有限损失、全数据统计、同种子逐回合对照，产出 `formal_comparison.json`。

串行执行器会在阶段启动前核对可用系统内存。低于配置门槛时保持排队并每 30 秒复查，不启动负载、不停止其他进程；内存恢复后继续同一阶段。厨房设备配置门槛为 18 GiB，灵巧手为 24 GiB。端口探测允许复用已退出服务留下的暂态连接，但仍拒绝活动监听端口。
