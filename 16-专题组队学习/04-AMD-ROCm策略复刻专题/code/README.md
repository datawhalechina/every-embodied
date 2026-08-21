# Python 脚本入口

这些脚本补充 Notebook，而不是替代 Notebook。Notebook 用来逐格观察图像、
动作和诊断指标；脚本用来批量评估、回放数据集和生成可复核的视频。脚本不
包含数据集、模型权重或完整 mujoco_pnp 工程。

## 运行前准备

在专题目录中设置路径：

~~~bash
cd /path/to/every-embodied/16-专题组队学习/04-AMD-ROCm策略复刻专题
export PROJECT_ROOT=/path/to/mujoco_pnp
export DATA_ROOT=/path/to/datasets/every_embodied
export MODEL_ROOT=/path/to/checkpoints/every_embodied
export OUTPUT_ROOT=/path/to/outputs/every_embodied
~~~

PROJECT_ROOT 至少需要包含 asset/example_scene_y2.xml 和
mujoco_env/y_env2.py。DATA_ROOT 下默认查找
omy_pnp_language/meta/info.json；也可以直接设置 DATASET_ROOT。上游工程
的来源、体积和外部放置方式见 [../external/README.md](../external/README.md)。

ROCm 的 PyTorch 仍使用 DEVICE=cuda。如果只是排查路径和导入问题，可以显式
设置 DEVICE=cpu，但不能据此判断模型的实际 GPU 性能。

视频写出优先使用 imageio 和 imageio-ffmpeg 的 H.264；当前环境只有
opencv-python 时，脚本会自动回退到 OpenCV 可用的 MP4 编码器。希望浏览器
稳定播放时建议额外安装：

~~~bash
pip install imageio imageio-ffmpeg
~~~

## 批量闭环评估

run_closed_loop.py 直接读取 checkpoint，把策略放回 SimpleEnv2，按 20 Hz
获取双相机图像、6 维关节状态和语言指令，再输出 7 维动作。它会同时记录：

- legacy_success：上游环境的旧几何成功条件；
- physical_success：在旧条件上增加抬升持续时间、放置高度、杯子直立、
  盘子位移、夹爪释放、末端抬升和连续稳定帧的严格条件；
- xy_dist、max_target_lift、upright_cos 等复核字段；
- 每个 seed 的 H.264/yuv420p rollout 视频（编码器不可用时回退到 OpenCV）和 results.jsonl。

示例：

~~~bash
export POLICY_TYPE=smolvla
export MODEL_RUN_DIR="$MODEL_ROOT/smolvla_weighted_000500"
export EVAL_SEEDS=1000,1001,1002,1003
export RENDER=0
python code/run_closed_loop.py
~~~

POLICY_TYPE 支持 act、smolvla 和 pi0。如果模型目录本身就是 pretrained_model，
可以改用：

~~~bash
export POLICY_PATH=/path/to/pretrained_model
python code/run_closed_loop.py
~~~

默认闭环不打开 3D 窗口，适合 Xvfb 或无窗口服务器；需要观察实时窗口时设置
RENDER=1，并先准备 DISPLAY。正式报告不要只跑默认的四个 seed，至少应
扩展到 20–30 个 held-out seed，并按红杯、蓝杯和物理成功分别统计。

## SmolVLA 快捷入口

~~~bash
python code/task11_smolvla_eval.py
~~~

它只是设置 POLICY_TYPE=smolvla 和四个默认 seed，然后调用
run_closed_loop.py。如果模型路径不同，优先显式设置 MODEL_RUN_DIR；不会
再通过固定 Notebook cell 编号执行代码，也不会写入服务器上的
/root/.Xauthority。

## 数据集回放

先列出 episode：

~~~bash
python code/replay_dataset.py --list
~~~

再导出 agent 或 wrist 视角：

~~~bash
python code/replay_dataset.py --episode 0 --view agent
python code/replay_dataset.py --episode 3 --view wrist --fps 20
~~~

默认输出到 $OUTPUT_ROOT/replay_ep{episode}_{view}.mp4。数据集的主视角优先
读取 observation.image，同时兼容旧的 agent_image 命名；腕部视角读取
observation.wrist_image。如果路径不是 DATA_ROOT/omy_pnp_language，设置：

~~~bash
export DATASET_ROOT=/path/to/omy_pnp_language
~~~

## MuJoCo 入门 Demo

小球 Demo 是自包含的：

~~~bash
python code/demo/demo_ball.py
~~~

机械臂 Demo 需要上游场景：

~~~bash
python code/demo/demo_arm.py
~~~

两个视频都会写到 outputs/demo_videos，优先使用浏览器更容易解码的 H.264/yuv420p
格式。Demo 只验证渲染、场景和视频链路，不等同于策略训练或闭环
成功率。

## 常见限制

- 轻量 amd-rocm 分支不提交示教数据、checkpoint、Hugging Face cache 或完整
  third_party/；
- 脚本需要与当前上游 mujoco_pnp、LeRobot 版本配套，版本不一致时先看
  Notebook 01 的路径和环境审计；
- 没有 PROJECT_ROOT、数据集元数据或模型权重时，脚本会明确报缺失路径，不会
  访问作者机器上的绝对目录；
- physical_success 是教程里的严格协议，不要把脚本的四 seed smoke 结果当成
  充分的泛化结论。
