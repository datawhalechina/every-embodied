# RoboWits：ROCm 下载、训练与创意任务评估

RoboWits 是面向机器人意外条件与创造性解题的 benchmark［评测基准］。环境基于 Genesis，数据采用 LeRobot［机器人学习数据框架］格式，官方提供 ACT、Pi0 和 Pi0.5 训练脚本。本章覆盖受限数据申请、资产准备、单卡 AMD 训练、断点续训、10 任务 seed/mutation［原始条件/突变条件］评估与视频输出。

## 1. 申请并下载数据

先在网页申请 [RoboWits LeRobot 数据](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset)，获批后登录：

```bash
hf auth login
hf auth whoami

hf download XHRlyb2001/RoboWits_lerobot_dataset \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/RoboWits_lerobot_dataset"
```

完整数据记录约 1,207 个 episode［回合］、1,194,300 帧、24 个任务。下载后先检查元数据和视频目录：

```bash
find "$DATA_ROOT/RoboWits_lerobot_dataset" -type f | wc -l
find "$DATA_ROOT/RoboWits_lerobot_dataset/meta" -maxdepth 2 -type f | head
find "$DATA_ROOT/RoboWits_lerobot_dataset/videos" -type f | head
```

## 2. 安装源码和资产

```bash
cd "$SRC_ROOT"
git clone https://github.com/UMass-Embodied-AGI/RoboWits.git
cd RoboWits
git submodule update --init --recursive
```

保留已有 ROCm PyTorch［AMD 平台张量计算框架］，不要让上游依赖重新安装 CUDA wheel［英伟达软件包］：

```bash
python -m pip install -e . --no-deps
python -m pip install accelerate datasets pyav wandb
python -m pip install -e third_party/lerobot --no-deps
```

部分完整场景需要 BlenderKit 资产。通过 Blender 插件登录后下载到上游规定目录；密钥不写入仓库。

## 3. 环境与数据形状门禁

```bash
export PYTHONPATH="$SRC_ROOT/RoboWits/third_party/lerobot/src:$SRC_ROOT/RoboWits"
python scripts/robowits/examples/run_env.py
```

RoboWits 有两套主要接口：

| 模式 | 状态/动作维度 | 用途 |
| --- | ---: | --- |
| EE / EE_ABS | 14 | 末端位姿策略 |
| JOINT / JOINT_ABS | 16 | 官方 ACT 数据与评估 |

官方 ACT 训练使用 16 维 JOINT［关节状态］与 JOINT_ABS［绝对关节动作］。训练和评估必须保持同一接口。

## 4. 单卡 AMD ACT 训练

官方 `scripts/robowits/train/train_act.sh` 默认通过 Accelerate［分布式训练启动器］使用 8 卡。多卡设备可直接设置数据、输出目录和进程数：

```bash
cd "$SRC_ROOT/RoboWits"
HF_DATASET=XHRlyb2001/RoboWits_lerobot_dataset \
OUTPUT_DIR="$RUN_ROOT/robowits/act_100k" \
NUM_PROCESSES=8 \
STEPS=100000 \
SAVE_FREQ=5000 \
  bash scripts/robowits/train/train_act.sh
```

单卡 AMD 运行时直接调用 `lerobot-train`，保留与官方脚本相同的模型和验证参数：

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --dataset.video_backend=pyav \
  --dataset.val_frac=0.04 \
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.dim_model=512 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=4 \
  --save_freq=5000 \
  --log_freq=100 \
  --eval_freq=0 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/act_100k"
```

先用 `--steps=10` 做反向传播门禁，再启动 100k 长训。每 5k 保存 checkpoint［检查点］，避免远端实例中断后从头开始。

## 5. ACT 断点续训

```bash
lerobot-train \
  --resume=true \
  --config_path="$RUN_ROOT/robowits/act_100k/checkpoints/last/pretrained_model/train_config.json" \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=4 \
  --wandb.enable=false
```

恢复前确认最后 checkpoint［检查点］包含模型、优化器状态和 `train_config.json`。

## 6. Pi0 与 Pi0.5 训练

Pi0：

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --policy.type=pi0 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.pretrained_revision=26b99b9439acb1e352439e34ee9c67af0d76efa3 \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=5000 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/pi0_100k"
```

Pi0.5：

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.pretrained_revision=9e55186ad36e66b95cda57bc47818d9e6237ae30 \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=5000 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/pi05_100k"
```

Pi0/Pi0.5 需要先取得对应基础权重访问权限。显存不足时降低 batch size［批大小］，但把偏差写入 `run_config.json`。

## 7. 官方 10 任务评估

```bash
cd "$SRC_ROOT/RoboWits"
export CHECKPOINT_PATH="$RUN_ROOT/robowits/act_100k/checkpoints/last/pretrained_model"
export TASK_IDS='01 02 03 04 06 09 13 16 17 25'
export CONFIG=JOINT_ABS
export OBSERVATION_MODE=JOINT
export ENV_DEVICE=cpu
export N_EPISODES=50
export OUTPUT_DIR="$RUN_ROOT/robowits/act_100k_eval"

bash scripts/robowits/eval/eval.sh
```

突变条件评估：

```bash
export OUTPUT_DIR="$RUN_ROOT/robowits/act_100k_mutation_eval"
bash scripts/robowits/eval/eval_mutation.sh
```

每个任务使用官方 `eval_dataset_50/{task}.json` 的 50 个初始状态。评估保存 `eval_info.json`、逐任务统计与视频。

## 8. Genesis 与 ROCm 的分工

策略网络在 AMD GPU［图形处理器］上运行；Genesis 场景可先使用 CPU 后端稳定执行。PyTorch［张量计算框架］参数仍写 `cuda`，因为 ROCm［AMD 开放计算平台］沿用该设备字符串。

```bash
export ROBO_WITS_TORCH_DEVICE=cuda
export ROBO_WITS_GENESIS_BACKEND=cpu
```

如果 GPU Genesis［GPU 物理仿真后端］构建场景失败，不影响先验证 AMD 策略训练；物理后端和策略后端在结果中分别记录。

## 9. 训练记录

长训至少记录：step［训练步数］、loss［损失值］、learning rate［学习率］、grad norm［梯度范数］、param norm［参数范数］、每步耗时、显存、温度、checkpoint［检查点］路径和评估任务。loss［损失值］只用于训练诊断，最终结论来自闭环成功率和阶段进度分数。

## 10. 官方资料

- [RoboWits 项目](https://umass-embodied-agi.github.io/RoboWits/)
- [RoboWits 源码](https://github.com/UMass-Embodied-AGI/RoboWits)
- [RoboWits 数据](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset)
- [RoboWits ACT AMD checkpoint](https://huggingface.co/Datawhale/robowits-act-amd-rocm)
