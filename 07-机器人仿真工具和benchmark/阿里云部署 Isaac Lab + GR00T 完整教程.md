# 阿里云部署 Isaac Lab + GR00T 完整教程

> 面向有 Linux/GPU 基础的工程师和研究者的具身智能开发环境部署指南
> 
> 作者实测环境：阿里云 A10 GPU 抢占式实例

⚠️ **版本声明**：本教程基于 **2026年1月** 的软件版本编写，强依赖以下版本组合：
- Isaac Sim 4.2.0 + Isaac Lab v1.4.1
- GR00T N1.6 + transformers==4.51.3
- PyTorch 2.5.1 + CUDA 12.1

**仅保证在上述版本下可复现**。如使用更新版本，可能需要自行调整依赖和配置。

⚠️ **安全声明**：本教程为快速实验设计，使用了 root 用户、`--network host`、`xhost +` 等简化配置。**不建议用于生产环境或长期运行的服务器**。

## 目录

1. [环境概述](#1-环境概述)
2. [阿里云实例创建](#2-阿里云实例创建)
3. [基础环境配置](#3-基础环境配置)
4. [VNC远程桌面配置](#4-vnc远程桌面配置)
5. [Isaac Sim + Isaac Lab 部署](#5-isaac-sim--isaac-lab-部署)
6. [GR00T 环境配置](#6-groot-环境配置)
7. [GR00T + MuJoCo 评估](#7-groot--mujoco-评估)
8. [常见问题与踩坑记录](#8-常见问题与踩坑记录)

---

## 1. 环境概述

### 最终架构

```
宿主机 (Ubuntu 22.04 + A10 GPU)
├── Docker: isaac-sim-gui
│   ├── Isaac Sim 4.2.0
│   ├── Isaac Lab v1.4.1
│   └── VNC显示 (端口6080)
└── Conda: groot环境
    ├── Isaac-GR00T代码
    ├── GR00T-N1.6-3B模型 (通用)
    └── GR00T-N1.6-G1-PnPAppleToPlate模型 (G1专用)
```

### 为什么这样设计？

- **Isaac Lab 在 Docker 内**：Isaac Sim 依赖复杂，官方 Docker 镜像最省心
- **GR00T 在宿主机 Conda**：GR00T 需要 PyTorch 2.5+，与 Isaac Sim 的 PyTorch 2.4 冲突
- **VNC 远程桌面**：WebRTC Livestream 在云服务器上有兼容性问题，VNC 更稳定

#### 关键设计决策

| 决策 | 为什么这么做 | 不这么做会怎样 |
|------|-------------|---------------|
| GR00T 在 Conda 而非 Docker | PyTorch 2.5+ 与 Isaac Sim 冲突 | 依赖地狱，环境崩溃 |
| 跳过 flash-attn | 编译耗时且非必须 | OOM/卡死，服务器重启 |
| 锁定 transformers==4.51.3 | 新版 API 变化 | 模型加载报错 |
| 用 --no-deps 装 GR00T | 避免重复编译 flash-attn | pip 卡住不动 |
| 手动克隆子模块 | curl 下载没有 .git | setup 脚本报错 |

### 部署顺序（必须按顺序执行）

```
基础配置 (3) → VNC (4) → Isaac Docker (5) → GR00T Conda (6) → MuJoCo 评估 (7)
     │              │              │                │
     └── NVIDIA 驱动 → Docker → Container Toolkit ──┘
```

> ⚠️ **不要跳步或乱序**，每一步都依赖前面的配置。

### 硬件要求

- GPU: NVIDIA A10 24GB 或更高（RTX 3090/4090 也可）
- 内存: 32GB+
- 硬盘: **150GB+ SSD**（实际占用约 80-100GB）
  - Ubuntu 系统：~5GB
  - Isaac Sim Docker 镜像：~25GB
  - Docker 运行时缓存：~10GB
  - Conda 环境 (PyTorch + CUDA)：~10GB
  - GR00T 模型 (3B + G1)：~12GB
  - 日志和临时文件：~10GB
  - 预留空间：~30GB

---

## 2. 阿里云实例创建

### 2.0 事前准备

#### 注册 NGC 账户（拉取 Isaac Sim 镜像需要）

1. 访问 https://ngc.nvidia.com
2. 点击 "Sign Up" 用邮箱注册（或用 Google/GitHub 登录）
3. 登录后点右上角头像 → "Setup" → "Generate API Key"
4. 点 "Generate API Key"，复制保存

> ⚠️ API Key 只显示一次，务必保存好。后续 `docker login nvcr.io` 时需要用到。

### 2.1 费用说明

- **实测成本**：完成整个教程约 **30 元人民币**（抢占式实例 + 流量费）
- **账户要求**：阿里云账户需充值 **100 元以上**才能购买抢占式实例
- 抢占式实例价格波动，实际费用可能有差异

### 2.2 选择实例规格

1. 登录阿里云控制台 → 云服务器 ECS → 创建实例
2. 选择地域：**西南1（成都）** 推荐（价格最低）
3. 实例规格：搜索 `ecs.gn7i`，选择 A10 24GB 规格
   - **ecs.gn7i-c16g1.4xlarge**（推荐）：16 vCPU + 60GB 内存
   - ecs.gn7i-c8g1.2xlarge：8 vCPU + 32GB 内存

> 💡 **省钱技巧**：不同地域价格差异很大！
> - 成都 c16g1.4xlarge：约 **2.4 元/小时**（高配低价）
> - 北京/杭州 c8g1.2xlarge：约 3.1 元/小时
> 
> 建议选成都，配置更高、价格更低。
4. **付费模式：抢占式实例**（比按量付费便宜 80-90%）
   - 设置最高价格为按量付费的 50-70%
   - 勾选"实例释放保护"

### 2.2 镜像选择

- 操作系统：**Ubuntu 22.04 64位**
- 或选择 NVIDIA GPU 云加速镜像（预装驱动）

### 2.3 网络配置

- 分配公网 IP（按流量计费）
- 带宽：5-10 Mbps 足够

### 2.4 安全组配置

创建或修改安全组，开放以下端口：

| 端口 | 用途 |
|------|------|
| 22 | SSH |

> ⚠️ **安全建议**：不要直接开放 VNC 端口（5901/6080），使用 SSH 隧道更安全，见下文。

---

## 3. 基础环境配置

### 3.1 SSH 连接

```bash
ssh root@<你的公网IP>
```

> ⚠️ **重要**：使用阿里云 Workbench 连接时，不要选择"免密连接"！
> 
> 免密连接使用的是容器化 Web 终端，权限受限，无法正常启动 Docker。
> 
> 请选择"密码"或"密钥对"方式连接，或直接用本地 SSH 客户端连接。

### 3.2 安装 NVIDIA 驱动（如果镜像没预装）

```bash
# 检查驱动
nvidia-smi

# 如果没有，安装驱动
apt update
apt install -y nvidia-driver-535
reboot
```

### 3.3 安装 Docker

```bash
# 检查是否已安装
docker --version

# 如果没有，安装 Docker
apt update
apt install -y docker.io
systemctl enable docker
systemctl start docker

# 验证安装
docker --version
```

### 3.4 配置 Docker 国内镜像

Docker Hub 在国内访问不稳定，配置镜像加速：

```bash
cat > /etc/docker/daemon.json << 'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
EOF

systemctl restart docker
```

### 3.5 安装 NVIDIA Container Toolkit

直接下载 deb 包安装（国内稳定可复现）：

```bash
cd /tmp

# 下载 4 个必需的包
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container1_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container-tools_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit-base_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit_1.17.4-1_amd64.deb

# 按顺序安装
dpkg -i libnvidia-container1_1.17.4-1_amd64.deb
dpkg -i libnvidia-container-tools_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit-base_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit_1.17.4-1_amd64.deb

# 配置 Docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 验证（应该能看到 nvidia-smi 输出）
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

> 💡 **提示**：下载链接会自动重定向到国内镜像（nvidia.cn），速度较快。

---

## 4. VNC远程桌面配置

### 4.1 安装桌面环境和 VNC

```bash
apt update
apt install -y xfce4 xfce4-goodies tigervnc-standalone-server tigervnc-common novnc websockify
```

> 💡 **安装过程中的提示**：
> - 如果出现 "Daemons using outdated libraries" 对话框，按 Tab 键选中 `<Ok>` 然后回车继续
> - 这是 Ubuntu 系统更新后提示重启服务，正常现象

### 4.2 配置 VNC

```bash
# 设置 VNC 密码
vncpasswd
# 输入密码（至少6位），view-only 选 n

# 创建 VNC 配置
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# 启动 VNC 服务
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

# 验证 VNC 是否正常启动
netstat -tlnp | grep 5901
# 应该看到 Xtigervnc 在监听 5901 端口，类似：
# tcp        0      0 0.0.0.0:5901            0.0.0.0:*               LISTEN      26811/Xtigervnc

# 启动 noVNC（Web 访问）
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### 4.3 访问 VNC

#### 方式一：SSH 隧道（推荐，更安全）

无需在安全组开放 VNC 端口，在本地电脑执行：

```bash
# Windows PowerShell / Mac Terminal / Linux
ssh -L 6080:localhost:6080 root@<你的公网IP>
```

保持 SSH 连接，然后浏览器打开：`http://localhost:6080/vnc.html`

#### 方式二：直接访问（需开放端口）

如果选择直接访问，需要在安全组开放 6080 端口（仅对你的 IP）：

浏览器打开：`http://<你的公网IP>:6080/vnc.html`

输入 VNC 密码即可进入桌面。

> ⚠️ **安全警告**：VNC 协议安全性较弱，直接暴露端口容易被爆破。强烈建议使用 SSH 隧道方式。

### 4.4 VNC 自启动（可选）

```bash
cat > /etc/systemd/system/vncserver.service << 'EOF'
[Unit]
Description=VNC Server
After=network.target

[Service]
Type=forking
User=root
ExecStart=/usr/bin/vncserver :1 -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
EOF

systemctl enable vncserver
```

---

## 5. Isaac Sim + Isaac Lab 部署

### 5.1 登录 NGC 并拉取镜像

Isaac Sim 镜像托管在 NVIDIA NGC（GPU Cloud），需要先登录：

```bash
# 登录 NGC
docker login nvcr.io
```

系统会提示输入用户名和密码：
- **Username**: 输入 `$oauthtoken`（固定值，直接复制）
- **Password**: 输入你在第 2.0 节获取的 NGC API Key

看到 `Login Succeeded` 表示登录成功。

```bash
# 拉取 Isaac Sim 4.2.0 镜像（约 15GB，需要一些时间）
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
```

下载成功后会显示：
```
Status: Downloaded newer image for nvcr.io/nvidia/isaac-sim:4.2.0
```

验证镜像已下载：
```bash
docker images | grep isaac-sim
# 应该看到类似输出：
# nvcr.io/nvidia/isaac-sim   4.2.0   <IMAGE_ID>   <SIZE>
```

### 5.2 启动容器

```bash
# 在宿主机设置 X11 权限
export DISPLAY=:1
xhost +local:docker

# 启动容器
# 注意：--network host 和 xhost + 是为了简化配置，仅建议用于短期实验
docker run -it --name isaac-sim-gui --gpus all --network host \
  -e DISPLAY=:1 \
  -e ACCEPT_EULA=Y \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint bash \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

### 5.3 准备工作：在宿主机下载依赖

由于容器内网络环境限制，建议在宿主机下载 Isaac Lab 和 robomimic：

```bash
# 退出容器，回到宿主机
exit

# 在宿主机执行（不要进入容器）
cd ~

# 安装下载工具（如果没有）
apt update
apt install -y git wget unzip

# 下载 Isaac Lab（使用国内镜像）
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git

# 下载 robomimic（用于模仿学习任务）
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic

# 重新启动容器
docker start isaac-sim-gui

# 复制进容器
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
docker cp robomimic isaac-sim-gui:/tmp/
```

### 5.4 在容器内安装依赖

```bash
# 进入容器
docker exec -it isaac-sim-gui bash

# 安装编译工具和 EGL 库（robomimic 需要）
apt update
apt install -y build-essential cmake pkg-config git
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# 安装 robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### 5.5 修复 Isaac Lab 配置文件

Isaac Lab 的依赖配置有两个问题需要修复：

```bash
cd /isaac-sim/IsaacLab

# 备份原文件
cp source/extensions/omni.isaac.lab_tasks/setup.py source/extensions/omni.isaac.lab_tasks/setup.py.bak

# 修复 1：rsl-rl 名称不匹配
# 原因：setup.py 中引用的是 GitHub 仓库名 "rsl-rl"，但 PyPI 上的包名是 "rsl-rl-lib"
# 这是上游的命名不一致问题，不影响功能
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py

# 修复 2：robomimic 已手动安装，删除 git 克隆配置
# 原因：setup.py 试图从 GitHub 克隆 robomimic，但容器内网络不稳定
# 我们已在宿主机下载并复制进来，所以删除这个配置
sed -i '/robomimic@git/d' source/extensions/omni.isaac.lab_tasks/setup.py

# 验证修复
echo "=== 检查第 46 行（rsl-rl）==="
sed -n '46p' source/extensions/omni.isaac.lab_tasks/setup.py
echo "=== 检查第 50-56 行（robomimic 应该已删除）==="
sed -n '50,56p' source/extensions/omni.isaac.lab_tasks/setup.py
```

预期输出：
```
=== 检查第 46 行（rsl-rl）===
"rsl-rl": ["rsl-rl-lib==2.3.0"],
=== 检查第 52-56 行（robomimic）===
# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))
```

如果第 46 行还是 `rsl-rl@git+https://...`，说明 sed 命令没生效，检查行号是否正确。

### 5.6 安装 Isaac Lab

```bash
cd /isaac-sim/IsaacLab

# 创建符号链接
ln -s /isaac-sim _isaac_sim

# 修复 pip（Isaac Sim 容器内 pip 可能损坏）
/isaac-sim/python.sh -m ensurepip --upgrade
/isaac-sim/python.sh -m pip install --upgrade pip setuptools

# 安装 Isaac Lab（约 5-10 分钟）
./isaaclab.sh --install
```

安装过程中会看到很多依赖包的下载和安装，最后应该显示：
```
Successfully installed omni-isaac-lab_tasks-0.10.18 ...
```

### 5.7 测试 Isaac Lab

```bash
# 在容器内运行 demo
export DISPLAY=:1
cd /isaac-sim/IsaacLab

# 机械臂 demo
./isaaclab.sh -p source/standalone/demos/arms.py

# 双足机器人 demo
./isaaclab.sh -p source/standalone/demos/bipeds.py

# 人形机器人强化学习训练
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Humanoid-Direct-v0 \
  --num_envs 64 \
  --max_iterations 100
```

在 VNC 桌面上应该能看到仿真画面。按 `Ctrl+C` 可以停止程序。

---

## 6. GR00T 环境配置

### 6.1 安装 Miniconda

```bash
# 退出 Docker 容器，回到宿主机
exit

# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc

# 接受 Conda 服务条款（新版 Conda 要求）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 6.2 创建 GR00T 环境

```bash
# 创建 Python 3.10 环境
conda create -n groot python=3.10 -y
conda activate groot

# 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### flash-attn 安装（可选，推荐跳过）

> ⚠️ **重要提示**：flash-attn 编译非常耗时（15-60分钟），且编译期间会占用大量系统资源，可能导致 SSH/VNC 断连。
> GR00T 在没有 flash-attn 的情况下也能正常运行，只是推理速度稍慢。
> **建议跳过此步骤**，直接进入 6.3 节。

如果你确实需要 flash-attn（追求极致性能）：

```bash
# 安装编译依赖
pip install numpy psutil packaging ninja

# 设置 CUDA 环境变量（必须）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 限制并行编译数，防止内存不足
# 32GB 内存用 MAX_JOBS=4
# 58GB+ 内存用 MAX_JOBS=6
MAX_JOBS=6 pip install flash-attn --no-build-isolation -v
```

> ⏳ **编译过程说明**：
> - `running bdist_wheel` 阶段会卡 1-3 分钟，这是正常的（在配置 cmake）
> - 之后会看到 `[1/73] ...` 表示开始真正编译（共 73 个文件）
> - 整个编译过程 15-60 分钟，取决于 CPU 核心数和 MAX_JOBS 设置
>
> ⚠️ **编译期间的异常现象（正常！）**：
> - **SSH 新连接可能连不上**：编译占用大量资源，sshd 响应变慢
> - **VNC 可能断线**：同上原因
> - **已有终端可能卡住无响应**：系统负载过高
> - **只要 CPU 占用率还在 50%+ 就说明在编译**，不要强制重启！
> - 等 CPU 降到 5% 以下就是编译完成了
>
> 💡 可以在阿里云控制台的"实例监控"页面观察 CPU 使用率，无需 SSH 连接。
>
> ⚠️ **编译完成后可能需要重启**：
> - 编译完成后 SSH 可能仍然连不上
> - 在阿里云控制台重启实例
> - 重启后再次运行 `MAX_JOBS=6 pip install flash-attn --no-build-isolation -v`
> - 这次会直接使用缓存的 wheel，几秒钟就能装完

**验证安装**：

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
# 应该输出 2.8.3 或类似版本
```

### 6.3 安装 Isaac-GR00T

```bash
cd ~

# 下载 Isaac-GR00T（注意：仓库在 NVIDIA 组织下，不是 NVIDIA-Omniverse）
# 方法1：git clone（国内可能较慢）
git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git

# 方法2：使用 gitclone 镜像
git clone --depth 1 https://gitclone.com/github.com/NVIDIA/Isaac-GR00T.git

# 方法3（推荐）：用 ghproxy 加速下载
curl -L -o isaac-groot.tar.gz https://ghproxy.cn/https://github.com/NVIDIA/Isaac-GR00T/archive/refs/heads/main.tar.gz
tar -xzf isaac-groot.tar.gz
mv Isaac-GR00T-main Isaac-GR00T

cd Isaac-GR00T
```

#### 安装 GR00T

> ⚠️ **重要**：不要直接用 `pip install -e .`，它会重复编译 flash-attn 导致卡住。
> 
> **原因**：GR00T 的 setup.py 将 flash-attn 列为依赖，即使已安装，pip 也会尝试重新构建。
> 使用 `--no-deps` 跳过自动依赖解析，手动安装核心包可避免此问题。

```bash
# 使用 --no-deps 跳过依赖自动安装
pip install -e . --no-build-isolation --no-deps

# 手动安装核心依赖
# transformers 必须锁定 4.51.3，更高版本会导致模型加载报错
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec

# 验证安装
python -c 'from gr00t.policy.gr00t_policy import Gr00tPolicy; print("GR00T loaded!")'
```

> 💡 **关于依赖版本警告**：安装完成后可能会看到一堆 `pip's dependency resolver` 警告，提示版本不兼容。
> 这是因为 GR00T 的 setup.py 锁定了非常严格的版本号，但实际上稍新的版本也能正常工作。
> **只要验证命令输出 `GR00T loaded!` 就说明安装成功**，可以忽略这些警告。

> 💡 这种方式避免了 pip 重复编译 flash-attn 的问题。如果之前已安装 flash-attn，GR00T 会自动使用它；如果没装，会 fallback 到标准 attention。

### 6.4 下载 GR00T 模型

**注意**：HuggingFace 在国内需要使用镜像

```bash
conda activate groot
cd ~/Isaac-GR00T

# 下载通用模型 GR00T-N1.6-3B
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-3B', local_dir='/root/groot_n16_model')"

# 下载 G1 机器人专用模型（用于评估）
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-G1-PnPAppleToPlate', local_dir='/root/groot_g1_model')"
```

### 6.5 测试 GR00T 推理

```bash
cd ~/Isaac-GR00T

# 创建测试脚本
cat > test_groot.py << 'EOF'
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

print('Loading GR00T N1.6...')
policy = Gr00tPolicy(
    model_path='/root/groot_n16_model',
    embodiment_tag=EmbodimentTag('gr1'),
    device='cuda',
)

obs = {
    'video': {
        'ego_view_bg_crop_pad_res256_freq20': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
    },
    'state': {
        'left_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'right_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'left_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'right_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'waist': np.random.rand(1, 1, 3).astype(np.float32),
    },
    'language': {
        'task': [['pick up the red apple']],
    },
}

action = policy.get_action(obs)
print('Action output:')
for k, v in action[0].items():
    print(f'  {k}: shape={v.shape}')
print('GR00T Inference Success!')
EOF

# 运行测试
python test_groot.py
```

---

## 7. GR00T + MuJoCo 评估

### 7.1 准备工作

```bash
conda activate groot
cd ~/Isaac-GR00T

# 安装依赖
pip install uv
apt-get update && apt-get install -y libegl1-mesa-dev libglu1-mesa git-lfs
git lfs install

# 配置 git 使用 ghproxy 加速（关键！）
git config --global http.version HTTP/1.1
git config --global url."https://ghproxy.cn/https://github.com/".insteadOf "https://github.com/"

# 克隆 GR00T-WholeBodyControl 子模块（必须用 git clone，不能用 tar.gz）
mkdir -p external_dependencies
cd external_dependencies
rm -rf GR00T-WholeBodyControl
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

cd ~/Isaac-GR00T

# 验证 LFS 文件已下载（应该是 100KB+ 而不是 131 字节）
ls -la ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/robot_model/model_data/g1/meshes/left_hip_pitch_link.STL
```

> ⚠️ **重要**：
> - 必须用 `git clone` 而不是 `curl` 下载 tar.gz，因为仓库包含 Git LFS 大文件（机器人 STL 模型）
> - tar.gz 下载只能拿到 LFS 指针文件（131 字节），不是真正的模型文件
> - 如果 STL 文件只有 131 字节，MuJoCo 仿真会报错 `Failed to determine STL storage representation`

### 7.2 修改 setup 脚本并运行

```bash
cd ~/Isaac-GR00T

# 注释掉 git submodule 和 git lfs pull 命令（我们已经手动处理了）
sed -i 's/^git submodule update/#git submodule update/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i 's/^git -C/#git -C/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# 注释掉 robosuite 的删除和 git clone（我们手动下载）
sed -i '26s/^rm/#rm/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i '27s/^git clone/#git clone/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# 手动下载 robosuite（setup 脚本里的 git clone 经常失败）
cd ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg
curl -L -o robosuite.tar.gz https://ghproxy.cn/https://github.com/xieleo5/robosuite/archive/refs/heads/leo/support_g1_locomanip.tar.gz
tar -xzf robosuite.tar.gz
mv robosuite-leo-support_g1_locomanip gr00trobosuite
rm robosuite.tar.gz

# 运行 setup 脚本
cd ~/Isaac-GR00T
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

脚本会：
1. 创建 uv 虚拟环境
2. 安装 robosuite、robocasa、lerobot 等依赖（下载 torch 等大包需要几分钟）
3. 验证环境（应该看到 `Imports OK` 和 `Env OK`）

> 💡 **网络问题排查**：
> - 如果报错 `HTTP2 framing layer`：确认已执行 `git config --global http.version HTTP/1.1`
> - 如果报错 `Connection timed out`：确认已配置 ghproxy 加速
> - 如果 SSH 断连：重连后重新运行 setup 脚本，已下载的包会被缓存

### 7.3 修改代码启用实时可视化

```bash
# 备份原文件
cp ~/Isaac-GR00T/gr00t/eval/rollout_policy.py ~/Isaac-GR00T/gr00t/eval/rollout_policy.py.bak

# 修改渲染模式
sed -i 's/os.environ\["MUJOCO_GL"\] = "egl"/os.environ["MUJOCO_GL"] = "glx"/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
sed -i 's/onscreen=False/onscreen=True/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
```

### 7.4 运行评估

需要两个终端：

**终端1 - 启动 GR00T Server：**
```bash
cd ~/Isaac-GR00T
conda activate groot
python gr00t/eval/run_gr00t_server.py \
  --model-path /root/groot_g1_model \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

等待显示 `Server is ready and listening on tcp://127.0.0.1:5555`

**终端2 - 启动评估 Client（在 VNC 里执行）：**

> 💡 **VNC 复制粘贴技巧**：
> 1. 在 noVNC 网页左侧点击展开菜单，找到"剪贴板"（Clipboard）
> 2. 把下面的命令粘贴到剪贴板文本框里
> 3. 在 VNC 桌面打开终端（右键桌面 → Terminal）
> 4. 在终端里用 `Ctrl+Shift+V` 粘贴命令
> 5. 回车执行
>
> ⚠️ **注意**：命令已合并成单行，避免复制多行命令时带入隐藏字符导致报错（如 `invalid int value: '5555~'`）

```bash
cd ~/Isaac-GR00T && export DISPLAY=:1 && gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n_episodes 3 --max_episode_steps 500 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc --n_action_steps 20 --n_envs 1 --policy_client_host 127.0.0.1 --policy_client_port 5555
```

在 VNC 桌面上会弹出 MuJoCo 窗口，实时显示 G1 机器人执行"抓苹果放盘子"任务。

---

## 8. 常见问题与踩坑记录

### Q1: 如何安装 NVIDIA Container Toolkit？

**推荐方法**：直接下载 deb 包，见第 3.4 节。这种方式最稳定可复现。

**备选方法**（如果想用 apt 源）：
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
```

注意：nvidia.github.io 在国内可能被墙。

### Q2: WebRTC Livestream 黑屏

**解决**：改用 VNC 远程桌面，更稳定。

### Q3: Isaac Lab 克隆失败

**症状**：`fatal: unable to access 'https://github.com/isaac-sim/IsaacLab.git/'`

**解决**：在宿主机使用国内镜像下载后复制进容器
```bash
# 在宿主机
cd ~
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
```

### Q4: robomimic 安装失败

**症状**：`Failed to build 'robomimic'` 或 `egl_probe` 编译错误

**解决**：
```bash
# 在宿主机下载
cd ~
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic
docker cp robomimic isaac-sim-gui:/tmp/

# 在容器内安装 EGL 库和编译工具
docker exec -it isaac-sim-gui bash
apt update
apt install -y build-essential cmake pkg-config
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# 安装 robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### Q5: setup.py 配置错误

**症状**：`IndentationError` 或 `Could not find a version that satisfies the requirement rsl-rl`

**解决**：按照教程第 5.5 节修复配置文件
```bash
cd /isaac-sim/IsaacLab
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py
sed -i '53,54d' source/extensions/omni.isaac.lab_tasks/setup.py
```

### Q6: Isaac Lab 最新版报错

**原因**：最新版需要 Isaac Sim 4.5+
**解决**：使用 `git clone --depth 1 --branch v1.4.1` 直接克隆指定版本。

### Q7: rsl-rl 版本冲突

**解决**：已在教程第 5.5 节修复，使用 `rsl-rl-lib==2.3.0`

### Q8: HuggingFace 无法访问

**解决**：使用镜像 `HF_ENDPOINT=https://hf-mirror.com`

### Q9: GR00T 模型加载报错 shape mismatch

**原因**：下载了旧版 N1-2B 模型，但代码是 N1.6 版本
**解决**：下载 `nvidia/GR00T-N1.6-3B` 模型

### Q10: flash-attn 编译问题

#### 症状1：编译时 SSH/VNC 断连，但 CPU 占用率高

**这是正常的！** 编译占用大量资源导致系统响应变慢。
- 在阿里云控制台监控 CPU 使用率
- 等 CPU 降到 5% 以下就是编译完成
- **不要强制重启服务器！** 否则编译白费

#### 症状2：CPU 占用率接近 0%，编译卡住

说明编译进程异常退出，需要重试：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
```

#### 症状3：重启后 flash-attn 没了

编译被中断，需要重新编译。如果不想等，可以跳过 flash-attn：
```bash
cd ~/Isaac-GR00T
pip install -e . --no-deps
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec
```

#### 症状4：安装成功但有一堆版本警告

```
gr00t 0.1.0 requires flash-attn==2.7.4.post1, but you have flash-attn 2.8.3 which is incompatible.
...
```

**可以忽略！** 这是因为 GR00T 的 setup.py 锁定了非常严格的版本号，但实际上稍新的版本也能正常工作。只要 `python -c "import flash_attn"` 不报错就行。

> 💡 flash-attn 是性能优化组件，不是必需的。跳过它不影响 GR00T 的功能。

### Q11: MuJoCo 无法显示窗口

**解决**：
1. 确保 VNC 正在运行：`netstat -tlnp | grep 5901`
2. 设置 `export DISPLAY=:1`
3. 修改代码中的 `MUJOCO_GL` 为 `glx`，`onscreen` 为 `True`

### Q12: VNC 启动后无法连接

**症状**：websockify 日志显示 `Connection refused`

**解决**：
```bash
# 杀掉旧进程
vncserver -kill :1
pkill -9 websockify

# 修复 xstartup 配置
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# 重启 VNC（注意 -localhost no 参数）
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### Q13: 抢占式实例被释放

**建议**：
- 设置合理的最高价格
- 重要数据及时备份
- 使用数据盘存储模型和代码

### Q14: 磁盘空间不足

**解决**：
- 清理 Docker 缓存：`docker system prune -a`
- 清理 pip 缓存：`pip cache purge`
- 清理 conda 缓存：`conda clean -a`
- 建议使用 150GB+ 系统盘，或挂载独立数据盘

### Q15: VNC 画面卡顿

**说明**：VNC 传输 3D 仿真画面会有延迟和丢帧，这是正常的。VNC 主要用于"确认运行状态"，而非流畅操作。

**进阶方案**：
- 使用 NoMachine（性能更好）
- 录制视频后下载到本地观看

### Q16: GR00T-WholeBodyControl 克隆失败

**症状**：
- `fatal: repository 'https://github.com/NVIDIA/GR00T-WholeBodyControl.git/' not found`
- `error: RPC failed; curl 16 Error in the HTTP2 framing layer`

**原因**：子模块地址是 `NVlabs` 组织下，不是 `NVIDIA`；或者网络问题

**回查**：第 7.1 节

**解决**：
```bash
cd ~/Isaac-GR00T/external_dependencies

# 方法1：正确地址
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git

# 方法2（推荐）：用 ghproxy 加速
curl -L -o wbc.tar.gz https://ghproxy.cn/https://github.com/NVlabs/GR00T-WholeBodyControl/archive/refs/heads/main.tar.gz
tar -xzf wbc.tar.gz
mv GR00T-WholeBodyControl-main GR00T-WholeBodyControl
rm wbc.tar.gz
```

### Q17: transformers 版本不兼容

**症状**：`AttributeError: 'Eagle3_VLConfig' object has no attribute '_attn_implementation_autoset'`

**原因**：transformers 版本太新

**回查**：第 6.3 节，确认安装时用了 `transformers==4.51.3`

**解决**：
```bash
pip install transformers==4.51.3
```

---

## 附录：常用命令速查

```bash
# 启动 VNC
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &

# 进入 Isaac Sim 容器
docker start isaac-sim-gui
docker exec -it isaac-sim-gui bash

# 在容器内运行 Isaac Lab demo
export DISPLAY=:1
cd /isaac-sim/IsaacLab
./isaaclab.sh -p source/standalone/demos/arms.py

# 激活 GR00T 环境
conda activate groot

# 启动 GR00T Server
python gr00t/eval/run_gr00t_server.py \
  --model-path /root/groot_g1_model \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

---

## 参考链接

- [Isaac Sim 官方文档](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T-WholeBodyControl GitHub](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [GR00T 模型 HuggingFace](https://huggingface.co/nvidia/GR00T-N1.6-3B)

---

*教程完成时间：2026年1月*
*实测环境：阿里云 ecs.gn7i-c16g1.4xlarge (A10 24GB)*
