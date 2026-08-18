# 资产说明

本目录保存 AMD ROCm 策略复刻专题中使用的小体积教学图和少量真实回放视频。图表和关键帧序列由 `../code/generate_tutorial_assets.py` 生成，用来核对教程中的成功率和 rollout 行为；`aup_cloud_guide/` 中的截图来自 AMD / AUP 云平台使用指南压缩包。

重新生成图表时，先准备自己的实验输出目录，再运行：

```bash
python code/generate_tutorial_assets.py --source-root "$OUTPUT_ROOT"
```

其中 `$OUTPUT_ROOT` 应包含批量评估 JSONL/TSV 和代表性 rollout 视频。若不传 `--source-root`，脚本会使用内置的示例指标生成图表，并只用分支内已有 MP4 生成可播放预览；缺失源视频不会再生成误导性的占位图。

| 文件 | 用途 |
| --- | --- |
| `model_status_summary.png` | 专题当前复刻状态总览 |
| `smolvla_red_blue_success.png` | SmolVLA 红杯/蓝杯固定指令成功率对比 |
| `act_dagger_progress_curve.png` | ACT 从基线到 DAgger 纠偏的成功率变化 |
| `smolvla_failure_reference_sequence.jpg` | 分支内红杯失败参考视频的关键帧；不冒充缺失的蓝杯 baseline |
| `smolvla_blue_success_sequence.jpg` | 分支内蓝杯成功视频的关键帧 |
| `smolvla_weighted500_red_failure_seed8.mp4` | SmolVLA 红杯失败参考回放 |
| `smolvla_weighted500_blue_success_seed0.mp4` | SmolVLA 蓝杯成功回放 |
| `pnp_four_view_strict_success.mp4` | Agent/Egocentric/Top/Side 四视角严格成功实测视频 |
| `pi0_raw_vs_finisher_diagnostic.png` | pi_0 raw 与脚本收尾器的尾段诊断指标图 |
| `pi0_ep2_raw_vs_finisher_side_by_side.mp4` | pi_0 episode2 raw-vs-hybrid 对比视频 |
| `pi0_ep2_raw_vs_finisher_frame.png` | pi_0 episode2 对比视频关键帧 |
| `pi0_ep2_raw_vs_finisher_metrics.md` | pi_0 episode2 raw-vs-hybrid 指标小表 |
| `metrics_snapshot.json` | 小体积指标快照，包含 ACT、SmolVLA、pi_0 小集诊断、pi_0 full20 open-loop 和 closed-loop strict 数字 |
| `collection_dataset_snapshot.json` | AMD 设备上 20 episodes、2621 frames、20 Hz 数据集实测摘要 |
| `training_progress_snapshot.json` | ACT、SmolVLA、pi0 和 learned head 的历史训练节点摘要 |
| `training_progress_overview.png` | 历史训练步数与闭环结果总览，避免把训练完成度误当成功率 |
| `pi0_strict_input_results.json` | raw/head、固定/随机环境和 seed bug 的最终严格结果 |
| `pi0_strict_input_progress.png` | pi0 strict-input 固定场景与随机环境对照图 |
| `pnp_four_view_strict_success_sequence.jpg` | 四视角视频 5 个时刻的关键帧序列 |
| `aup_cloud_guide/` | AUP Learning Cloud JupyterHub / Code Server 使用指南截图（优先入口） |
| `amd_radeon_cloud/` | AMD Radeon Cloud 开发者云官方教程图片与 AMD ROCm Embodied AI Policy Replication 工作区截图（备用入口） |
