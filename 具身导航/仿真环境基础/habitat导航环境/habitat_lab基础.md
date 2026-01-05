# Habitat-lab仿真基础

### Habitat-Lab 是基于 Habitat-Sim 的算法层，封装了标准化的任务、评估指标和 API，让研究者无需关注底层仿真细节，专注于具身智能算法的设计、训练与评估。

## 一、Habitat-lab特性介绍

### 1. 预定义具身任务

* 导航类：PointGoalNav（点目标导航）、ObjectNav（物体目标导航）、VLN（视觉语言导航）。
* 交互类：Rearrange（物体重排）、PickPlace（拾取放置）、OpenDoor（开门）。
* 支持自定义任务（如多智能体协作、长程规划）。

### 2.标准化评估体系

* 内置核心指标（成功率 SR、路径长度效率 SPL、DTW 对齐分数等），可直接对比算法性能。
* 支持与公开基准（如 Habitat Challenge）对齐，便于论文复现和成果对比。

### 3.模块化智能体架构

* 解耦 “传感器 → 编码器 → 策略 → 控制器”，可快速替换组件（如用 CNN 或者 Transformer 做视觉编码，用 RL/IL 做决策）；
* 深度集成 PyTorch，支持强化学习（RL）、模仿学习（IL）、端到端深度学习等主流算法。

### 4.易用的 API

* 提供简洁的 Python 接口，一键加载场景、初始化智能体、运行仿真循环。

## 二、Habitat-lab环境搭建

这里可以直接使用habitat-sim创建的conda环境

```python
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

在下载habitat-lab的时候注意需要与安装的habitat-sim版本一致，例如本教程使用的habitat-sim版本是0.2.5，则需要下载0.2.5版本的habitat-lab，复现其他论文的时候也要注意habitat-sim和habitat-lab版本一致。

```python
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

同时安装habitat-baselines

```python
pip install -e habitat-baselines
```

下载3D场景数据和点导航数据

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```


