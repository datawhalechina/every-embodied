# 推理与仿真加速

`03-Notebook/` 按顺序提供入口、推理计时、固定图重放、仿真多步调用和原生条件图实验。每篇包含输入、可修改代码、计时边界和结果检查；不启动训练或完整任务评估。

依赖使用既有环境中的 PyTorch［张量计算框架］、MuJoCo［多关节接触仿真器］、NumPy［数组计算库］、Matplotlib［绘图库］和 Jupyter［交互式笔记本环境］。固定图重放需要受支持的图形计算设备。没有设备时明确保存未执行状态。

`../notebooks/results/acceleration/` 保存来源与历史记录。小型随机动作头用于学习工程优化，原生设备历史吞吐与状态配对检查分开列明。源码来自 [原项目](https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph)，随附原许可；构建补丁需在隔离环境完成。
