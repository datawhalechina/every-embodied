# 诊断课源码快照

这些只读源码快照用于展示已完成排障对应的实现，不自动导入仿真或模型运行库。笔记本逐项展开相关函数，并以内容指纹核对来源。

- `eval_dexjoco_openpi.py` 与原始严格计时档案记录的客户端指纹完全相同。
- `dexjoco_openpi_env.py` 是原动作接口诊断使用的适配器。
- `openpi_*.py` 来自此前保留的 OpenPI 源码；许可保留为 `OPENPI_LICENSE`。
- `strict_latency_instrumentation.py` 展示当时如何插入计时事件；包含原设备路径，阅读时保留原样，重跑时须改为独立新目录。

每个文件的来源与指纹见 `../../notebooks/results/diagnostics/source-manifest.json`。这些快照不替换项目运行环境中的上游源码。
