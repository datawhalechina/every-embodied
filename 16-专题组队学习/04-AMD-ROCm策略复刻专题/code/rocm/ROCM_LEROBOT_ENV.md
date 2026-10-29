# AMD ROCm LeRobot 环境锁定

AMD 395 工作站与 Radeon Cloud/W7900 Notebook 使用同一套 LeRobot 兼容基线。Notebook 依赖旧版 `lerobot.common`，因此不能把新版 `lerobot.policies` checkout 混入同一个 Python 进程。

## 锁定基线

| 项目 | 版本/路径 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.11.0+rocm7.13.0a20260425 |
| torchvision | 0.26.0+rocm7.13.0a20260425 |
| HIP | 7.13.26162 |
| LeRobot | `10b7b3532543b4adfb65760f02a49b4c537afde7` |
| LeRobot import root | `/home/aup/jiahang/lerobot_10b7_legacy` |
| 关键 API | `lerobot.common.datasets`, `lerobot.common.policies` |

训练、评估和 Notebook runner 必须先设置：

```bash
export LEROBOT_SRC=/home/aup/jiahang/lerobot_10b7_legacy
export PYTHONPATH=/home/aup/jiahang/lerobot_10b7_legacy:$PYTHONPATH
```

也可以执行：

```bash
source code/rocm/activate_lerobot_rocm.sh
python code/rocm/verify_lerobot_rocm_env.py
```

输出 `ENVIRONMENT_OK` 后再启动 Notebook。输出 `ENVIRONMENT_MISMATCH` 时不要训练，先修复 Python/ROCm/LeRobot 版本。

## 两个平台的分工

- AMD 395：本地调试、短 smoke、数据检查、Notebook 单步复现。
- Radeon Cloud/W7900：相同环境锁定后进行长训、14 条闭环评估和视频生成。

模型权重、数据集、Notebook 代码可以共享；Python `site-packages` 不跨机器复制，云端应按本清单重新安装或使用同一容器/环境模板。

## 当前已知差异

W7900 当前 SSH 端口暂时无法完成握手，因此尚未完成云端在线验收。云端恢复连接后，第一条命令必须是：

```bash
python code/rocm/verify_lerobot_rocm_env.py --json
```

两端 JSON 中的核心版本、LeRobot commit 和 import 路径全部一致，才算环境对齐。
