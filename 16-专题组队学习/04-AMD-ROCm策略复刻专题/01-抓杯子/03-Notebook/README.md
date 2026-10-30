# 抓杯子：采集、训练与物理成功

按目标选择路线：先查看成熟配方及已有结果，再逐格学习采集、数据回放、训练、闭环评估和视频。共享程序位于专题 `code/`，真实结果和完整媒体位于专题 `notebooks/results/` 与 `assets/`。

## 推荐入口与完整训练配方

- [任务入口、已发布权重和结果总览](00_任务入口与已记录结果.ipynb)
- [成熟轻量视觉语言动作策略：完整训练、保护配方与评估](14_smolvla_end_to_end.ipynb)
- [视觉语言动作策略：权限、训练谱系与严格评估](15_pi0_end_to_end.ipynb)
- [动作分块策略：训练与闭环诊断](16_act_end_to_end.ipynb)

## 逐格展开全过程代码的独立实训

- [19号：40条新采集、物理回放、5000次训练更新和30回合评估（2/30）](19_cup_collect_train_evaluate.ipynb)
- [20号：复用19号同批40条数据，确定性模型20000次更新和30回合评估（3/30）](20_cup_deterministic_train_evaluate.ipynb)

两篇保存各自训练权重、曲线及成功／失败完整视频；20号复用19号的数据，训练配方与产物独立。

## 数据、基础训练与诊断

- [采集与数据审计](07_data_collection_and_audit.ipynb)
- [动作分块策略基础训练](08_act_training_rocm.ipynb)
- [轻量视觉语言动作策略基础训练](09_smolvla_training_rocm.ipynb)
- [视觉语言动作策略基础训练](10_pi0_training_rocm.ipynb)
- [闭环部署与视频](11_mujoco_closed_loop_deploy.ipynb)
- [严格物理成功判定](02_physical_success_review.ipynb)
- [动作分块策略与聚合数据诊断](03_act_dagger_diagnostics.ipynb)
- [红蓝任务采样加权](04_smolvla_weighted_sampling.ipynb)
- [严格输入评估](12_pi0_strict_input_end_to_end.ipynb)
- [随机位置与末端增量动作](13_pi05_random_position_eef_delta.ipynb)

## 普通与保护训练变体

以下变体与上方完整训练教程分开列出，按需选择，不必重复执行六本。

- [普通训练：14号轻量视觉语言动作策略](workflows/ordinary/14_smolvla_end_to_end.ipynb)
- [普通训练：15号视觉语言动作策略](workflows/ordinary/15_pi0_end_to_end.ipynb)
- [普通训练：16号动作分块策略](workflows/ordinary/16_act_end_to_end.ipynb)
- [保护配方训练：14号轻量视觉语言动作策略](workflows/protected/14_smolvla_end_to_end.ipynb)
- [保护配方训练：15号视觉语言动作策略](workflows/protected/15_pi0_end_to_end.ipynb)
- [保护配方训练：16号动作分块策略](workflows/protected/16_act_end_to_end.ipynb)

阅读时保留数据来源、评估分母与运行条件。复现实训前显式启用对应计算开关。
