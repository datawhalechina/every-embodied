# 云端学习、免费额度申请与远程开发

本章帮助学习小组申请AMD［超威半导体］云资源、核对设备与存储，并为策略训练和性能实验选择环境。[21天学习计划](学习计划.md)从这里开始。网页操作与远程开发界面见[平台附录](appendices/AUP_JUPYTER_CODE_SERVER.md)。

## 可申请的官方资源

| 入口 | 面向与申请方式 | 适合本课程的用途 |
| --- | --- | --- |
| AMD［超威半导体］开发者云 | 符合条件的开发者申请免费初始额度；官方当前列出100美元，按审核结果发放。 | 策略训练、模型推理和性能测试。 |
| AUP Learning Cloud［大学计划学习云］ | 学生、教师和研究人员注册后提交远程访问申请，批准后使用分配资源。 | 课程实验与教学；按实际机型选择匹配环境。 |
| 已有本地工作站 | 使用自己的设备和已验证环境。 | 基础测试、持续实验和视频生成。 |

官方依据：[开发者云与额度规则](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)、[额度申请步骤](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html)、[大学计划学习云](https://www.amd.com/en/corporate/university-program/learning-cloud.html)。核对日期为2026年11月13日，客户端洛杉矶日期。

## 免费开发者云额度怎么申请

1. 在[官方开发者计划](https://developer.amd.com/ai-developer-program/)注册，进入会员权益页面的云额度申请入口。
2. 填写身份、公开项目资料及用途，可具体说明本课程的策略训练、灵巧手推理或性能实验。
3. 等待审批并按确认邮件创建／登录云账号、激活资源；组织上建议开营前3–5个工作日开始准备。
4. 核对获批额度、计费方式与到期日。官方列出初始100美元额度和通常30天有效期，具体以确认邮件与控制台为准。

额度覆盖的服务、附加存储费用和后续付款设置需要单独核对。官方说明该开发者云实例关闭电源后仍可能计费，销毁才停止占用；先导出成果，再按控制台规则释放资源。已绑定付款方式的账号还应核对额度用尽后的计费设置。课程不要求为完成基础学习而付费升级。

## 大学计划学习云

从[官方学习云页面](https://www.amd.com/en/corporate/university-program/learning-cloud.html)进入远程申请表。获批后使用平台分配的区域和设备；本入口面向学术学习与教学，不把开发者云的100美元额度规则套用到这里。

## 环境与资源分层

处理器可运行故障夹具、结果复算和部分仿真练习。模型训练、真实策略推理与图重放需要匹配的设备计算环境。课程历史实验涉及不同设备；平台分配的新机型应先做依赖、相机和算子检查。定制原生条件图运行库属于进阶实验，不是所有基础学员的前置条件。

## 首次检查

进入终端后运行：

```bash
id
pwd
df -h
python --version
which python
```

随后确认AMD［超威半导体］的 GPU［图形处理器］与 ROCm［开放计算软件栈］：

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

PyTorch［张量计算框架］在 ROCm［开放计算软件栈］上沿用 `torch.cuda` 接口，因此设备字符串仍写作 `cuda`。`torch.version.hip` 用于确认当前安装来自 ROCm［开放计算软件栈］构建。

## 持久化目录

推荐用 `WORK_ROOT` 统一管理大文件：

```bash
export WORK_ROOT=${WORK_ROOT:-$HOME/physical-ai}

# 仅当当前实例已确认 /workspace 是持久化卷时：
# export WORK_ROOT=/workspace/physical-ai

# 仅当当前学习云镜像已确认 /home/jovyan 会保留时：
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

先按平台规则验证存储保留方式，再保存模型、数据和结果；实例销毁前另行导出备份。临时虚拟环境可以重建，不应占用 Git 仓库空间。


## 存储规则按当前实例确认

上面的 `/workspace` 和 `/home/jovyan` 只是常见目录示例，不能仅凭目录名判断永久保留。以所选平台和实例说明为准，实际测试重启前后是否保留；额度到期或实例销毁前，将所需模型、结果和视频导出到已确认的外部存储。

## 进入课程

先完成[设备与环境确认](README_01_AMD_ROCm设备与环境确认.md)，再按[任务目录](任务目录.md)选择课程。受限权重访问许可、环境与模型缓存应在正式训练前准备好。凭据在终端或平台指定位置管理，不写入公开笔记本或仓库。
