# AMD ROCm 云平台与远程开发

本章用于选择运行平台、确认持久化目录并建立远程开发入口。网页按钮、JupyterHub 和 Code Server 的基础操作见 [平台操作附录](./appendices/AUP_JUPYTER_CODE_SERVER.md)。

## 平台选择

| 平台 | 适合任务 | 存储与网络 | 使用建议 |
| --- | --- | --- | --- |
| AUP Learning Cloud | 下载公开数据、安装依赖、Notebook 学习、训练与评估 | 可访问 GitHub 和 Hugging Face；以平台显示的持久化目录为准 | 本专题优先入口 |
| AMD 开发者云 | ROCm 模板预检、GPU 训练和性能测试 | 仅 `/workspace` 的 PVC 在实例重启后保留；其他目录按临时空间处理 | 将源码、模型和结果放到 `/workspace` |
| 本地 AMD 工作站 | 长训练、批量评估和视频生成 | 自行规划数据盘与缓存 | 适合持续实验 |

额度、GPU 型号和实例生命周期可能调整，以平台页面的当前信息为准。

## 首次检查

进入终端后运行：

```bash
id
pwd
df -h
python --version
which python
```

随后确认 AMD GPU 和 ROCm：

```bash
rocminfo | sed -n '1,80p'
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
assert torch.version.hip is not None
PY
```

PyTorch 在 ROCm 上沿用 `torch.cuda` 接口，因此设备字符串仍写作 `cuda`。`torch.version.hip` 用于确认当前安装来自 ROCm 构建。

## 持久化目录

推荐用 `WORK_ROOT` 统一管理大文件：

```bash
export WORK_ROOT=${WORK_ROOT:-$HOME/physical-ai}

# AMD 开发者云使用持久化 PVC：
# export WORK_ROOT=/workspace/physical-ai

# AUP 镜像若约定 /home/jovyan 持久化：
# export WORK_ROOT=/home/jovyan/physical-ai

export SRC_ROOT=$WORK_ROOT/src
export DATA_ROOT=$WORK_ROOT/datasets
export MODEL_ROOT=$WORK_ROOT/checkpoints
export RUN_ROOT=$WORK_ROOT/runs
export CACHE_ROOT=$WORK_ROOT/cache
export HF_HOME=$CACHE_ROOT/huggingface
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv

mkdir -p "$SRC_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT" \
  "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

实例重启前，把模型、数据和结果复制到持久化目录。临时虚拟环境可以重建，不应占用 Git 仓库空间。

## 源码与模型下载

```bash
cd "$SRC_ROOT"
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied

hf auth login
hf auth whoami
```

下载受限模型前，先在 Hugging Face 网页完成访问申请。登录凭据保存在用户配置目录，不写入 Notebook、脚本或仓库。

## SSH 服务

部分镜像没有预装 SSH 服务。拥有管理员权限时可执行：

```bash
sudo apt-get update
sudo apt-get install -y openssh-server iproute2
sudo mkdir -p /run/sshd
sudo ssh-keygen -A
sudo service ssh restart
ss -lntp | grep ':22'
```

平台若采用端口映射，以页面显示的主机和端口为准：

```bash
ssh -p <PORT> <USER>@<HOST>
```

SSH 连接只负责远程终端。实例开机、关机和销毁仍通过平台控制台完成。

## 运行专题

```bash
cd "$SRC_ROOT/every-embodied/16-专题组队学习/04-AMD-ROCm策略复刻专题"
jupyter lab --ip=0.0.0.0 --no-browser
```

先运行 [设备与环境确认](./README_01_AMD_ROCm设备与环境确认.md)，再按 [统一目录](./README_10_仿真基准下载与统一目录.md) 下载基准数据和模型。

## 平台切换

在另一台设备恢复实验时，只需重新设置目录变量并确认四类材料：

1. 源码仓库与上游版本；
2. 数据集目录和元数据；
3. 模型目录和训练配置；
4. 评估结果、视频和运行配置。

训练脚本和 Notebook 使用 `$SRC_ROOT`、`$DATA_ROOT`、`$MODEL_ROOT`、`$RUN_ROOT`，避免绑定某台服务器的绝对路径。
