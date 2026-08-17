# Notebook 工作流

本目录把每个模型拆成两条可以独立执行的 Notebook：

- **普通训练**：从普通数据和基础模型开始训练，再运行闭环评估。
- **保护训练**：复现我们用于得到保护结果的训练配方，再运行同一套闭环评估。

两种模式使用不同的输出目录，不会覆盖彼此的 checkpoint、日志、视频或评估结果。Notebook 内的训练循环是真实执行的 Python 单元格，长训会在当前单元格中显示一条每 100 步刷新的紧凑进度条；逐步指标写入 `notebook_train_metrics.jsonl`，每次训练的开始时间、结束时间、总耗时和 checkpoint 写入 `training_run_summary.json`，不会为每一步生成一行 Notebook 输出。

## 普通训练

| 模型 | Notebook | 主要内容 |
| --- | --- | --- |
| SmolVLA | [`14_smolvla_end_to_end.ipynb`](./ordinary/14_smolvla_end_to_end.ipynb) | 基础训练、checkpoint、闭环评估、视频 |
| Pi0 | [`15_pi0_end_to_end.ipynb`](./ordinary/15_pi0_end_to_end.ipynb) | 基础训练、checkpoint、闭环评估、视频 |
| ACT | [`16_act_end_to_end.ipynb`](./ordinary/16_act_end_to_end.ipynb) | 基础训练、checkpoint、闭环评估、视频 |

## 保护训练

| 模型 | Notebook | 主要内容 |
| --- | --- | --- |
| SmolVLA | [`14_smolvla_end_to_end.ipynb`](./protected/14_smolvla_end_to_end.ipynb) | 保护配方、保护 checkpoint、正式闭环评估、视频 |
| Pi0 | [`15_pi0_end_to_end.ipynb`](./protected/15_pi0_end_to_end.ipynb) | clean/protected 配方、保护 checkpoint、正式闭环评估、视频 |
| ACT | [`16_act_end_to_end.ipynb`](./protected/16_act_end_to_end.ipynb) | repair15 配方、保护 checkpoint、正式闭环评估、视频 |

## 执行建议

1. 先执行 `code/rocm/activate_lerobot_rocm.sh`，再运行 `code/rocm/verify_lerobot_rocm_env.py`；只有输出 `ENVIRONMENT_OK` 才继续。
2. 首次学习先运行普通 Notebook 的环境检查和 smoke 单元格。
3. 确认数据、模型和设备正常后，再执行长训单元格。
4. 训练完成后运行 Notebook 后面的 Full 14 评估单元格，查看本次训练的成功率、阶段统计、视频和 action trace。
5. 想复现保护结果时，单独打开 protected Notebook；不要在普通 Notebook 中修改路径来代替保护配方。

两台 AMD 设备应统一使用同一套 Python 3.10、ROCm 7.13、PyTorch 2.11 和 legacy LeRobot commit。395 工作站已经通过校验；W7900 云端恢复 SSH 后必须先运行同一校验脚本。详细版本锁定见 [`code/rocm/ROCM_LEROBOT_ENV.md`](../../code/rocm/ROCM_LEROBOT_ENV.md)，依赖清单见 [`requirements-rocm713.txt`](../../code/rocm/requirements-rocm713.txt)。

每个 Notebook 的结果都保存在专题目录下对应的 `outputs/ordinary/<model>` 或 `outputs/protected/<model>` 中。评估单元格只接受当前 Notebook 训练后生成的 `TRAINED_POLICY_PATH`，没有 checkpoint 会直接报错，不会回退到历史或预训练权重。长训耗时取决于模型、步数和 GPU；进度条只做阶段性刷新，详细指标以 JSONL 和 `training_run_summary.json` 为准。

## Full 14 评估与可视化

每个模型的普通版和 Protected 版都包含一个独立的 `Full 14-episode closed-loop evaluation` 单元格。执行后会：

- 实际闭环运行 14 个固定 seed，并写出完整 `result.jsonl`；
- 保留首个物理成功回合和首个物理失败回合的视频；
- 保存这两个回合的动作 trace；
- 在后续单元格内播放成功/失败视频，并绘制动作序列图；
- 以 `physical_success=x/14` 汇总结果。

这两个案例视频和序列图来自本次评估，不是 Notebook 中预置的静态结果。评估使用 `RENDER_EVAL` 或 Full 评估单元格中的渲染设置；无桌面环境时 Notebook 会尝试启动 Xvfb。
