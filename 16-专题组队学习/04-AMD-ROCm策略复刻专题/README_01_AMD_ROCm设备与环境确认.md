# 01 AMD ROCm 设备与环境确认

本任务先确认设备真的适合跑后续实验。很多复刻失败看起来像模型问题，实际可能是 ROCm 版本、显存分配、缓存路径、网络或 Hugging Face 权限没有准备好。

配套实操 Notebook：[01_device_env_check.ipynb](./notebooks/01_device_env_check.ipynb)。

## 学习目标

完成本任务后，可以直接判断：

- 查看 AMD GPU / APU 是否被 ROCm 正确识别；
- 判断训练时显存、系统内存和温度是否还有余量；
- 把数据集、checkpoint 和 Hugging Face cache 放到合适磁盘；
- 准备远端 SSH、代理和 token 的基本链路；
- 写出一份便于对照实验结果的设备资源表。

## 设备资源检查

在 AMD 设备上运行：

```bash
rocm-smi --showuse --showtemp --showmemuse
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
```

实验记录里至少写清楚：

| 项目 | 示例填写 |
| --- | --- |
| 设备型号 | AMD Ryzen AI MAX+ 395 / Radeon GPU |
| 系统版本 | Ubuntu 24.04 / WSL / 其他 |
| ROCm 版本 | 例如 7.x |
| PyTorch 版本 | 例如 ROCm build |
| 空闲温度 | 例如 30 到 45 摄氏度 |
| 训练温度 | 例如 75 到 85 摄氏度 |
| 空闲 VRAM | 例如 9% |
| 训练 VRAM | 记录 ACT / SmolVLA / pi_0 各自占用 |

## 统一内存和显存的理解

一些 AMD APU 设备采用统一内存架构。学习时要注意，操作系统、GPU、数据加载进程和浏览器都可能共享同一块物理内存。显存调得很大时，不代表训练一定更稳；如果系统可用内存被挤压，数据加载或模型初始化也可能失败。

用实验记录而不是猜测来判断资源是否够用：

```bash
watch -n 2 'rocm-smi --showuse --showtemp --showmemuse; free -h'
```

如果训练还没有开始，GPU 利用率为 0%，但 Python 进程内存持续增长，通常是在加载模型或下载权重；如果 GPU 利用率长期 100%，温度升高但 loss 正常下降，通常是正常训练。

## 大文件目录规划

为了让命令在不同机器上都能复用，不要在笔记和脚本里写死个人机器路径，而是统一使用变量：

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
export DATA_ROOT=/path/to/datasets/every_embodied
export HF_HOME=/path/to/cache/huggingface
export CHECKPOINT_ROOT=$PROJECT_ROOT/ckpt
```

大文件包括：

- Hugging Face 模型缓存；
- LeRobot 数据集；
- checkpoint；
- rollout 视频；
- 批量评估 JSONL；
- 中间日志。

这些文件通常体积很大，只需要在实验记录中写清楚保存位置和用途。教程和报告里保留路径变量、目录规划和清理方法即可。

## 网络与 Hugging Face 权限

pi_0 会依赖 gated model，例如 `google/paligemma-3b-pt-224`。先确认：

1. Hugging Face 账号已接受 gated model 条款；
2. token 有读取 public gated repository 的权限；
3. 远端机器可以访问 Hugging Face；
4. token 不出现在命令行、日志和 Notebook 输出中。

推荐用 smoke 脚本先验证权限，而不是直接开始长训练。

## Checkpoint

完成本任务时，保存一份可复现实验资源表：

```markdown
| 项目 | 结果 |
| --- | --- |
| ROCm 能否识别 GPU | 是 / 否 |
| torch.cuda.is_available | True / False |
| 空闲温度 |  |
| 空闲 VRAM |  |
| 数据目录 | 使用变量，说明数据放在哪类磁盘 |
| HF gated 权限 | 已通过 / 未通过 |
```
