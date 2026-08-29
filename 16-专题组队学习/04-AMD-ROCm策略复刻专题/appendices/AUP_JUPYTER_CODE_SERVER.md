# AUP 浏览器开发环境操作附录

本附录说明 AUP Learning Cloud 中 JupyterHub、Jupyter Notebook 和 Code Server 的界面操作。持久化目录、环境变量和课程运行命令见[云平台与远程开发主章](../README_00_AMD_AUP免费云平台使用指南.md)。

## 1. 工具选择

| 工具 | 适用工作 | 入口 |
| --- | --- | --- |
| Jupyter Notebook | 分段执行代码、查看图表、播放评估视频 | Notebook 镜像或 JupyterHub 启动器 |
| Code Server | 编辑多文件工程、运行终端命令、调试脚本、使用版本控制 | Code Server GPU Environment |
| 浏览器终端 | 安装依赖、检查设备、启动训练和评估 | JupyterLab Terminal 或 Code Server Terminal |

训练与批量评估使用脚本；Notebook 用于讲解、配置检查和结果回放。两种入口应共享同一个持久化工作目录。

## 2. 登录并选择环境

1. 打开 `https://tpe.aupcloud.io`。
2. 选择 **Use GitHub Login**，完成 GitHub 授权。
3. 在启动页选择课程镜像或开发镜像。
4. 设置本次运行时长并启动工作区。

![GitHub 授权登录](../assets/aup_cloud_guide/screenshot_07.jpg)

平台常用入口分为四类：

- **Course**：预装课程依赖的教学镜像；
- **Development**：Code Server 的 CPU 或 GPU 开发环境；
- **Test**：HIP 与 ROCm 运行检查；
- **Custom Repo**：从基础镜像创建自定义环境。

![AUP 环境选择页面](../assets/aup_cloud_guide/screenshot_04.jpg)

启动前先确认镜像的 ROCm、Python 和 PyTorch 版本。需要长期保留的源码、模型、数据和结果统一放到 `/home/jovyan`；`/ryzers/notebooks` 作为本次镜像的临时工作目录。

## 3. Jupyter Notebook 操作

### 3.1 新建与运行

在启动器中选择 **Python 3** 创建 Notebook。常用操作如下：

| 操作 | 快捷键 |
| --- | --- |
| 运行当前单元 | `Shift + Enter` |
| 保存文档 | `Ctrl + S` 或 `Cmd + S` |
| 中断运行 | Kernel 菜单中的 Interrupt |
| 重启内核 | Kernel 菜单中的 Restart |

最小设备检查：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU unavailable")
```

ROCm 版 PyTorch 沿用 `torch.cuda` 接口，因此 `torch.cuda.is_available()` 返回 `True` 时还要结合设备名称和张量计算确认环境。

### 3.2 保存到持久化目录

```bash
mkdir -p /home/jovyan/every-embodied
cp -a /ryzers/notebooks/<project-name> /home/jovyan/every-embodied/
```

建议直接从持久化目录打开工程，避免结束镜像前再次复制：

```bash
cd /home/jovyan/every-embodied
git clone https://github.com/datawhalechina/every-embodied.git
```

### 3.3 Notebook 运行顺序

课程 Notebook 按以下顺序执行：

1. 设备与目录配置；
2. 数据集和模型路径检查；
3. 单批前向与反向；
4. 短训；
5. 长训；
6. 闭环评估；
7. 视频和统计结果展示。

长训单元默认由显式开关控制。启动后保留训练日志和输出目录，不要依赖浏览器页面持续连接。

## 4. Code Server 操作

### 4.1 启动开发环境

选择 **Code Server GPU Environment**，再选择所需的 AMD Radeon GPU 配置并启动。

![Code Server 环境选择](../assets/aup_cloud_guide/screenshot_03.jpg)

![Code Server 主界面](../assets/aup_cloud_guide/screenshot_01.jpg)

界面分为四个区域：

- 左侧活动栏：文件、搜索、版本控制、运行与扩展；
- 中间编辑区：源码和 Markdown 文档；
- 下方终端：命令、日志和进程状态；
- Ports 面板：本地服务的转发端口。

### 4.2 推荐扩展

| 扩展 | 用途 |
| --- | --- |
| Python | Python 语法、调试与环境选择 |
| Jupyter | 在编辑器中运行 Notebook |
| Ruff | Python 格式与静态检查 |
| YAML | 配置文件语法支持 |
| GitLens | 版本历史和提交定位 |
| C/C++ | HIP 或原生扩展开发 |

![Code Server 扩展页面](../assets/aup_cloud_guide/screenshot_05.jpg)

扩展保存在镜像运行环境时，应把常用扩展名称记录在项目文档中，便于重建环境。

### 4.3 终端检查

```bash
rocm-smi
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
df -h /home/jovyan
```

训练前再检查当前目录和输出目录：

```bash
pwd
echo "$WORK_ROOT"
echo "$OUTPUT_ROOT"
```

## 5. 端口转发

Code Server 会检测终端中启动的 HTTP 服务，并在 **Ports** 面板创建转发入口。

```bash
python -m http.server 8080 --bind 0.0.0.0
```

![端口转发通知](../assets/aup_cloud_guide/screenshot_02.jpg)

![Ports 面板](../assets/aup_cloud_guide/screenshot_06.jpg)

转发地址通常采用以下形式：

```text
https://tpe.aupcloud.io/user/<username>/proxy/<port>/
```

服务必须监听 `0.0.0.0` 或平台允许的接口。若页面无法访问，依次检查：

```bash
ss -lntp | grep 8080
curl -I http://127.0.0.1:8080
```

随后在 Ports 面板确认端口号与转发地址一致。

## 6. 模型、数据与鉴权

缓存和输出统一放在持久化目录：

```bash
export WORK_ROOT=/home/jovyan/every-embodied-work
export HF_HOME="$WORK_ROOT/cache/huggingface"
export TORCH_HOME="$WORK_ROOT/cache/torch"
export DATA_ROOT="$WORK_ROOT/datasets"
export MODEL_ROOT="$WORK_ROOT/models"
export OUTPUT_ROOT="$WORK_ROOT/outputs"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$DATA_ROOT" "$MODEL_ROOT" "$OUTPUT_ROOT"
```

Hugging Face 登录信息只保存在用户配置目录或环境变量中。公开 Notebook、Markdown、截图和日志不记录访问令牌。

```bash
huggingface-cli login
huggingface-cli whoami
```

下载后检查模型目录和数据集元数据：

```bash
find "$MODEL_ROOT" -maxdepth 2 -type f | head
find "$DATA_ROOT" -maxdepth 3 -type f | head
du -sh "$MODEL_ROOT" "$DATA_ROOT" "$HF_HOME"
```

## 7. 切换镜像与结束工作区

切换镜像前执行以下步骤：

1. 保存 Notebook 和源码；
2. 确认训练输出位于 `/home/jovyan`；
3. 记录未完成任务的恢复命令；
4. 停止当前服务；
5. 在启动页选择 **Stop my server**；
6. 启动新的镜像。

![停止当前工作区](../assets/aup_cloud_guide/image_00.png)

结束前检查大文件位置：

```bash
du -sh /home/jovyan/* 2>/dev/null | sort -h
find /ryzers/notebooks -type f -size +100M -print 2>/dev/null
```

## 8. 常见问题

### 页面加载缓慢

重新登录并检查浏览器网络；页面恢复后先确认后端进程是否仍在运行。

### Notebook 单元执行顺序混乱

重启内核并从设备、目录和数据检查单元开始顺序执行。长训单元使用独立输出目录，避免覆盖已有结果。

### Code Server 扩展安装失败

刷新扩展索引并重试；网络受限时使用镜像中已有扩展完成核心操作。

### 端口没有自动出现

确认服务已监听，再在 Ports 面板手动添加端口。代理路径访问依赖当前用户登录状态。

### 工作区结束后文件消失

检查文件是否写入 `/ryzers/notebooks`。重新运行时把源码、数据、模型和输出根目录统一设置到 `/home/jovyan`。

## 9. 返回课程主线

平台操作完成后，按以下顺序继续：

1. [设备与环境确认](../README_01_AMD_ROCm设备与环境确认.md)
2. [仿真基准下载与统一目录](../README_10_仿真基准下载与统一目录.md)
3. [Notebook 运行索引](../notebooks/README.md)
4. [统一评估、视频与结果归档](../README_16_统一评估视频与结果归档.md)
