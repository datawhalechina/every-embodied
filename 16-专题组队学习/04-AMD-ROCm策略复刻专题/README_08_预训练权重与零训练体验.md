# 预训练权重与零训练体验

这一页用于区分三类材料：可以直接观看的成功回放、可以加载运行的预训练权重，以及只保留了实验记录但暂时不能下载的历史 checkpoint。学习者不需要先训练数小时，先确认正确行为和评估口径，再决定是否使用云额度训练自己的模型。

正式权重统一发布到 Datawhale Hugging Face 组织。课程只提供经过复核的模型文件、配置、SHA256、评估 JSON 和加载说明；不上传优化器状态、完整训练缓存或个人机器路径。学习者可以先下载权重观看成功回放，再选择从零训练。

当前模型仓库已创建并完成权重上传：

- [SmolVLA：every-embodied-smolvla-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp)
- [Pi0：every-embodied-pi0-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-pi0-mujoco-pnp)
- [ACT：every-embodied-act-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-act-mujoco-pnp)

三个仓库均包含模型卡、配置文件、评估摘要和 `weights/model.safetensors`。本地 `huggingface/` 目录保留同一份发布清单与模型卡，便于后续维护；不上传优化器状态、完整训练缓存或个人机器路径。

## 重新下载与复现

本地清理历史 checkpoint 后，学习者可以直接从 Datawhale Hugging Face 仓库恢复课程权重。下面只下载推理和评估所需文件，不下载优化器状态：

```bash
export HF_HOME=/data/cache/huggingface

# SmolVLA
hf download Datawhale/every-embodied-smolvla-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$HF_HOME/every-embodied-smolvla"

# Pi0
hf download Datawhale/every-embodied-pi0-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$HF_HOME/every-embodied-pi0"

# ACT
hf download Datawhale/every-embodied-act-mujoco-pnp \
  weights/model.safetensors weights/config.json \
  --repo-type model --local-dir "$HF_HOME/every-embodied-act"
```

下载后，将对应的 `weights/` 目录通过 Notebook 的 `SMOLVLA_EVAL_POLICY_PATH` / `SMOLVLA_POLICY_PATH`、`PI0_EVAL_POLICY_PATH` / `PI0_POLICY_PATH` 或 `ACT_EVAL_POLICY_PATH` / `ACT_POLICY_PATH` 传入，即可进行零训练加载和闭环评估；要从保护 recipe 继续训练时，再使用 `PI0_PRETRAINED_PATH_OVERRIDE` 或 `POLICY_PRETRAINED_PATH_OVERRIDE`。正式发布的权重 SHA256 已记录在 `huggingface/upload_manifest.json`。

## 零训练成功预览

打开 [11_mujoco_closed_loop_deploy.ipynb](./notebooks/11_mujoco_closed_loop_deploy.ipynb)，依次运行环境定位和“零训练成功预览”单元格。Notebook 会用 `IPython.display.Video` 显示：

```text
assets/pnp_four_view_strict_success.mp4
```

这段视频对应固定环境 seed 0、策略采样 seed 3 的严格成功回放。策略为 `pi0 + visual/history learned head`，不是 raw pi0；输入只有双相机图像、语言、robot proprio 和历史执行动作，没有读取杯子坐标、盘子坐标、GT phase 或 oracle action。它用于展示接近、夹取、抬升、搬运、释放和稳定放置的正确顺序，不代表随机位置泛化已经解决。

## 权重发布门槛

公开 checkpoint 前必须同时保留以下信息：

| 字段 | 要求 |
| --- | --- |
| 模型 | 模型类型、基座版本、训练步数和关键超参数 |
| 数据 | 数据集版本、episode 数量、语言任务和采样方式 |
| 动作 | state/action 维度、绝对或增量动作、控制频率和 chunk 设置 |
| 评估 | 环境版本、seed 列表、`physical_success` 成功率和代表视频 |
| 完整性 | 权重文件 SHA256、配置文件和加载命令 |
| 边界 | 明确是 raw policy、learned head 还是带规则的 scaffold |

缺少这些字段的权重即使能加载，也不作为课程基线发布。

## 当前可下载清单

| 候选 | 已验证结果 | 当前发布状态 | 说明 |
| --- | --- | --- | --- |
| SmolVLA weighted500 | 红杯 `27/30`、蓝杯 `30/30`，总计 `57/60` | 推荐发布 | 当前端到端 Notebook 重建结果；发布时同时提供模型 SHA、评估 JSON 和代表视频 |
| Pi0 protected clean40 | strict `12/14`；未见位置 `9/10`；hard 组 `6/8` | 推荐作为进阶权重 | 适合学习保护式续训、动作对齐和失败分析；不把它描述为 raw Pi0 零样本成功 |
| ACT stable61 fallback | strict `7/30` | 历史基线权重 | AMD395 上的旧重建分支，保留用于对照 |
| ACT protected repair15 candidate | strict `15/30` | 教学/诊断候选 | 从 stable61 protected checkpoint 低学习率续训 2500 steps；三组固定种子各 10 条，结果为 `3/10 + 4/10 + 8/10`，模型 SHA 为 `b9b178377995a674a06bc5d1500c8e7e7fc5d02649268855f892b3987bf5bfeb4` |
| ACT protected DAgger artifact | strict `2/30` | 负向对照权重 | exact checkpoint 已找回并完成 SHA 校验；保留用于解释为什么数据与 recipe 需要逐项复核 |
| pi0 + visual/history head | 固定环境 `6/12`，随机环境 `1/4` | 暂不作为入门权重 | 固定场景确有提升，但不是 raw pi0，且随机位置泛化不足；本章只公开成功回放和方法边界 |
| Pi0.5 canonical/recovery | 当前 strict 仍未稳定成功 | 不发布为成功基线 | 仅作为动作表示、chunk 对齐和 recovery 诊断案例 |

这里采用“宁可晚一点发布，也不发布无法复核的 checkpoint”的原则。SmolVLA 和 Pi0 的保护权重可以作为学习者的下载入口；ACT repair15 已有完整 strict30 JSON、分组结果和 SHA，可以作为当前教学保护候选，但它是 `15/30`，与历史 `17/30` 仍差 2 条，因此不能写成完全复现。旧 DAgger `2/30` 继续作为负向诊断分支保留。

## 自己训练时的额度策略

先执行 1–2 step smoke，确认数据读取、前向、反向、优化器和保存链路。随后运行短训 checkpoint，并尽早做 4 个固定 seed 的 closed-loop 小面板。只有模型出现真实接近、夹取或抬升，再扩大训练和评估规模。

不要把免费时长全部用于一次无法观察的长训练。每个阶段至少保留：

- 一个可恢复 checkpoint；
- 一份训练日志；
- 一组固定 seed 的 closed-loop JSONL；
- 一条成功或典型失败视频。

最终是否“训练够了”，由 held-out strict success 和视频决定，而不是由 `5000 steps` 这个数字决定。
