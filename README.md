# Every Embodied：AMD ROCm 轻量教程分支

`amd-rocm` 分支面向只学习 AMD ROCm 具身智能实验的读者。它保留专题 Markdown、Notebook、小型 Python 脚本、必要截图和少量教学回放，不包含完整数据集、模型 checkpoint、训练缓存或批量视频。

完整项目首页和电子书由 `main` 主干维护：

- [Every Embodied 在线电子书](https://datawhalechina.github.io/every-embodied/zh-cn/)
- [主干仓库](https://github.com/datawhalechina/every-embodied/tree/main)
- [AMD ROCm 专题电子书源码](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98)

## 轻量下载

只下载当前分支的最新快照：

```bash
git clone --depth 1 --single-branch --branch amd-rocm \
  https://github.com/datawhalechina/every-embodied.git every-embodied-amd-rocm
cd every-embodied-amd-rocm
```

`--depth 1` 只获取最新提交，`--single-branch` 不下载其他分支历史。只阅读 Markdown 时，不需要安装 Python 环境，也不需要下载模型与数据集。

如果只需要 Markdown，不需要截图、Notebook 和教学回放，可以使用按需克隆与稀疏检出：

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
  --single-branch --branch amd-rocm \
  https://github.com/datawhalechina/every-embodied.git every-embodied-amd-markdown
cd every-embodied-amd-markdown
git sparse-checkout init --no-cone
git sparse-checkout set \
  '/README.md' \
  '/README.en.md' \
  '/LICENSE' \
  '/16-专题组队学习/04-AMD-ROCm策略复刻专题/*.md' \
  '/16-专题组队学习/04-AMD-ROCm策略复刻专题/**/*.md'
git checkout amd-rocm
```

这种方式只在阅读到对应文件时下载 Git blob，不会检出图片、视频、Notebook 或 Python 源码。

## 教程入口

- [AMD ROCm 策略复刻专题首页](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README.md)
- [仿真基准总览与长程视频](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_09_AMD_Physical_AI仿真基准与长程视频复现.md)
- [仿真基准下载与统一目录](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_10_仿真基准下载与统一目录.md)
- [RoboCasa365 下载、训练与评估](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_11_RoboCasa365_ROCm下载训练评估.md)
- [DexJoCo Pi0.5 与 ROCm JAX](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_12_DexJoCo_Pi05_ROCm_JAX迁移训练评估.md)
- [DISCOVERSE 数据生成、训练与多视角视频](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_13_DISCOVERSE_ROCm数据生成训练与多视角视频.md)
- [RoboWits 下载、训练与创意任务评估](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_14_RoboWits_ROCm下载训练与创意任务评估.md)
- [Unitree G1 预测 CBF 安全控制](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_15_Unitree_G1预测CBF_ROCm复现.md)
- [统一评估、视频与结果归档](./16-专题组队学习/04-AMD-ROCm策略复刻专题/README_16_统一评估视频与结果归档.md)

## 大文件放置方式

教程使用四个外部目录连接源码、数据、模型和实验结果：

```bash
export SRC_ROOT=/path/to/sources
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/models
export RUN_ROOT=/path/to/runs
```

每章会继续定义具体子目录、下载命令和验收方法。长程案例使用在线 MP4 链接；数据集和模型按章节从 Hugging Face 或上游项目下载，不写入教程仓库。

## 分支分工

| 分支 | 用途 | 内容 |
| --- | --- | --- |
| `main` | 项目主干与电子书构建 | 全部课程 Markdown、网站构建程序和项目资源 |
| `amd-rocm` | AMD 专题轻量学习 | AMD 专题 Markdown、Notebook、小脚本、必要截图和少量回放 |

教程内容在两个分支之间保持同步；新增章节先完成链接、脚本和电子书渲染检查，再发布到对应分支。
