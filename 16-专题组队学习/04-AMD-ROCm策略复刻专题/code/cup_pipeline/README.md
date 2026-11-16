# 抓杯子完整实训接口

入口为 `01-抓杯子/03-Notebook/19_cup_collect_train_evaluate.ipynb`。默认显示保存结果；完整执行设置 `AMD_RUN_FULL=1`。

执行前在笔记本内核环境设置：

- `EXPERIMENT_PYTHON`：已安装匹配旧版 LeRobot、PyTorch［深度学习框架］和 ROCm［AMD异构计算平台］ 的解释器。
- `LEROBOT_SRC`：提供 `lerobot.common.datasets` 和 `lerobot.common.policies.act` 的旧版源码根目录。
- `PNP_ASSET_ROOT`：原抓杯子工程的 `asset` 目录，包含 `example_scene_y.xml` 与其引用资源。
- `CUP_RUN_ROOT`：独立实验输出目录。默认 `/data/Data14TB/runs/every_embodied/cup-full-v1`，不放进源码目录。

无桌面显示时需要系统提供 `xvfb-run`。所有依赖和视觉骨干提前缓存；运行阶段采用离线资源，缺失时明确报错。资源下载应单独通过 `lab-download` 进行。

顺序接口：

```bash
"$EXPERIMENT_PYTHON" course_run.py collect
"$EXPERIMENT_PYTHON" course_run.py audit
"$EXPERIMENT_PYTHON" course_run.py train
"$EXPERIMENT_PYTHON" course_run.py evaluate
```

40个采集种子1000–1039，成功轨迹入库后逐条回放。动作分块策略训练5000次更新，批量8、梯度累积1、学习率0.0001、动作块20。评估为独立种子2000–2029，各最多1200步，20赫兹。物理判定要求抬升、放盘、松爪及最终直立。评估加载本次新权重，不调用专家控制器，不输入物体真值。

`pipeline_state.json` 记录阶段、源码和配方指纹；已完成阶段重用前会复验产物。失败日志保留；修改源码或配方需使用新实验目录。数据、权重、逐回合轨迹及全部评估视频保存在实验目录，课程包仅包含轻量证据与代表媒体。

依赖版本与源码指纹在 `reproducibility/cup-runtime-profile.json`。实际使用 LeRobot 提交 `10b7b3532543b4adfb65760f02a49b4c537afde7`；`reproducibility/cup-lerobot-pretrained.patch` 保存权重先加载至中央处理器、再迁移至图形处理器的兼容补丁。原环境另有只影响其他模型的修改，本篇动作分块策略不调用这些模块。

在另一台已配置同版本加速库的设备上，可将此提交取到源码盘，再应用补丁。下载单独执行，不把整个训练程序放进下载代理：

```bash
lab-download --download-exec git clone https://github.com/huggingface/lerobot.git "$LEROBOT_SRC"
git -C "$LEROBOT_SRC" checkout 10b7b3532543b4adfb65760f02a49b4c537afde7
git -C "$LEROBOT_SRC" apply /absolute/path/to/reproducibility/cup-lerobot-pretrained.patch
```

首次机器部署时先完成设备与环境专题，并核对版本档案；本篇不会自动替换已有的加速库或删除其他环境。
