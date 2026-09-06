# LeRobot MuJoCo 训练ACT、SmolVLA、Pi0教程
本仓库提供了一个最小可运行示例：用于采集示教数据，并在自定义数据集上训练（或微调）视觉-语言-动作（VLA）模型。

## 目录
- [安装](#安装)
- [更新计划](#更新计划)
- [1. 采集示教数据](#1-采集示教数据)
- [2. 回放数据](#2-回放数据)
- [3. 训练 Action-Chunking-Transformer（ACT）](#3-训练-action-chunking-transformeract)
- [4. 部署 ACT 策略](#4-部署-act-策略)
- [5-6. 语言条件环境中的采集与可视化](#5-6-语言条件环境中的采集与可视化)
- [模型与数据集](#模型与数据集)
- [7. 训练与部署 pi_0](#7-训练与部署-pi_0)
- [8. 训练与部署 SmolVLA](#8-训练与部署-smolvla)
- [致谢](#致谢)

## 安装
我们在 **Python 3.10** 上测试通过。

不建议直接使用 `pip install lerobot`，可能会报错。

安装 MuJoCo 相关依赖和 lerobot：
```bash
conda create -n py310 python=3.10
pip install -r requirements.txt
conda install jupyterlab
pip install ipywidgets ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"
jupyter lab .
# 在当前目录启动


# 如果torch的cuda有问题：
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"


```

请确认 MuJoCo 版本为 **3.1.6**。

解压资源文件：
```bash
cd asset/objaverse
unzip plate_11.zip
```

## 更新计划
- [x] Viewer 更新
- [x] 增加多种 mug、plate，对应不同语言指令
- [x] 增加 pi_0 训练与推理
- [x] 增加 SmolVLA

## 1. 采集示教数据
运行 [1.collect_data.ipynb](1.collect_data.ipynb)

在给定环境中采集示教数据。任务是抓起杯子并放到盘子上。当杯子在盘子上、夹爪打开且末端执行器位于杯子上方时，环境判定成功。

<img src="./media/teleop.gif" width="480" height="360">

键位说明：
- `WASD`：x-y 平面移动
- `R/F`：z 轴移动
- `Q/E`：倾斜
- `方向键`：其余旋转
- `空格`：切换夹爪状态
- `Z`：重置环境，并丢弃当前回合缓存数据

叠加图像说明：
- 右上：Agent 视角
- 右下：腕部（第一人称）视角
- 左上：侧视图
- 左下：俯视图

数据集结构：
```python
fps = 20,
features={
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 个关节角 + 1 个夹爪
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # 仅物体初始位置，训练中不使用
    },
},
```

数据默认保存在 `./demo_data` 目录。仓库中已提供示例数据：[demo_data_example](./demo_data_example/)。

## 2. 回放数据
按采集时使用的机器人选择回放入口，数据路径和机器人模型必须匹配：

- [OMY 回放](2.visualize_data-omy.ipynb)
- [Nova5 回放](2.visualize_data-nova5.ipynb)
- [xArm7 回放](2.visualize_data-xarm7.ipynb)

<img src="./media/data.gif" width="480" height="360"></img>

在重建后的仿真场景中可视化你的动作。主窗口会回放动作；右上和右下叠加图像来自数据集。

## 3. 训练 Action-Chunking-Transformer（ACT）
运行 [3.train.ipynb](3.train.ipynb)

**大约需要 30~60 分钟**。

在自定义数据集上训练 ACT。示例中 `chunk_size=10`。

训练好的 checkpoint 会保存在 `./ckpt/act_y`。

可通过与数据集真值动作对比，评估策略误差。

<image src="./media/inference.png"  width="480" height="360">

<details>
    <summary>PicklingError: Can't pickle &lt;function &lt;lambda&gt;...&gt;</summary>
如遇 pickling 错误，请将 `num_workers` 设为 `0`，例如：

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0, # 4
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
```
</details>

## 4. 部署 ACT 策略
运行 [4.deploy.ipynb](4.deploy.ipynb)

如果没有可用于训练的 GPU，可从 Google Drive 下载 checkpoint：
- https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing

<img src="./media/rollout.gif" width="480" height="360" controls></img>

## 5-6. 语言条件环境中的采集与可视化
- [5.language_env.ipynb](5.language_env.ipynb)：键盘遥操作采集数据（键位与第一个环境一致）
- 1中只采集一条数据，5中采集20条数据，两个任务，红杯子和蓝杯子
- [6.visualize_data.ipynb](6.visualize_data.ipynb)：可视化已采集数据

**数据示例**

<img src="./media/data_v2.gif" width="480" height="360" controls></img>

## 模型与数据集
### 获取语言条件数据集

Pi0 和 SmolVLA 使用同一份语言条件数据，选择以下一种方式即可。所有命令均在本章目录运行，数据统一放在 `./demo_data_language`。

**方式一：下载公开样例**

[Datawhale 公开样例](https://huggingface.co/datasets/Datawhale/datawhale_eai_pnp_language) 是 OMY 场景的 LeRobot v2.1 数据集，包含 20 个 episode、2621 帧，以及红杯子和蓝杯子两个任务。

先安装 Git LFS，然后执行：

```bash
git lfs install
git clone https://huggingface.co/datasets/Datawhale/datawhale_eai_pnp_language demo_data_language
git -C demo_data_language lfs pull
```

已有同名目录时不要重复 clone 或删除原数据。如果目录是该数据集的 Git 仓库，可直接执行最后一条命令补齐 LFS 文件。仅下载到 LFS 指针文件不代表数据已完整。

**方式二：自行采集**

运行 [5.language_env.ipynb](5.language_env.ipynb)，保留 `REPO_NAME = 'datawhale_eai_pnp_language'` 和 `ROOT = "./demo_data_language"`，采集并保存红杯子、蓝杯子示教，然后用 [6.visualize_data.ipynb](6.visualize_data.ipynb) 检查轨迹。该 Notebook 在目录已存在时包含删除旧数据的确认步骤；已有下载数据或示教数据时，应先备份或更换目录。

两种方式均可使用本章的 `pi0_datawhale_eai.yaml` 和 `smolvla_datawhale_eai.yaml`，无需改配置：

```yaml
dataset:
  repo_id: datawhale_eai_pnp_language
  root: ./demo_data_language
```

`root` 是相对于运行目录的本地数据路径；完整数据已在该目录时，固定版本 LeRobot 使用本地文件。`repo_id` 是数据集标识，上述不带组织名的值沿用采集 Notebook 的命名，不能当作可搜索的 Hugging Face 完整地址。需要从 Hub 自动获取时应使用完整 ID `Datawhale/datawhale_eai_pnp_language`。更改本地目录时，需同步修改 YAML 和部署 Notebook 中的路径。

### 模型参考
| Model 🤗                                                      | Dataset 🤗                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [pi_0 finetuned](https://huggingface.co/Jeongeun/omy_pnp_pi0) | [dataset](https://huggingface.co/datasets/Jeongeun/omy_pnp_language) |
| [smolvla finetuned](https://huggingface.co/Jeongeun/omy_pnp_smolvla) | 同上                                                         |

## 7. 训练与部署 pi_0

以下命令均在本章目录执行，使用 `requirements.txt` 固定的 LeRobot `10b7b3532543b4adfb65760f02a49b4c537afde7`。
先执行 `python check_training_configs.py` 检查两个配置（不训练、不下载权重）。
YAML 命令中 `--config_path` 与文件名之间保留空格；该版本的等号写法会走 JSON checkpoint 加载分支。

- [train_model.py](train_model.py)：训练脚本
- [pi0_datawhale_eai.yaml](pi0_datawhale_eai.yaml)：训练配置
- [7.pi0.ipynb](7.pi0.ipynb)：部署示例

训练命令：
```bash
python train_model.py --config_path pi0_datawhale_eai.yaml
```

部署效果：

<img src="./media/rollout2.gif" width="480" height="360" controls></img>

训练日志：

<image src="./media/wandb.png"  width="480" height="360">

配置以本节链接的 YAML 为准。默认读取第 5 节的 `./demo_data_language`，`dataset.repo_id` 为 `datawhale_eai_pnp_language`；使用其他数据时需同时修改两项。默认关闭 W&B，启用前需设置自己的项目和账号。训练需要完整数据、模型权重和 CUDA 环境；配置解析通过不代表完整训练已验证。

## 8. 训练与部署 SmolVLA
- [train_model.py](train_model.py)：训练脚本
- [smolvla_datawhale_eai.yaml](smolvla_datawhale_eai.yaml)：训练配置
- [8.smolvla.ipynb](8.smolvla.ipynb)：部署示例

训练命令：
```bash
python train_model.py --config_path smolvla_datawhale_eai.yaml
```

部署效果：

<img src="./media/rollout3.gif" width="480" height="360" controls></img>

训练日志：

<image src="./media/wandb2.png"  width="480" height="360">

配置以本节链接的 YAML 为准。默认读取第 5 节的 `./demo_data_language`，`dataset.repo_id` 为 `datawhale_eai_pnp_language`；使用其他数据时需同时修改两项。默认关闭 W&B，启用前需设置自己的项目和账号。训练需要完整数据、模型权重和 CUDA 环境；配置解析通过不代表完整训练已验证。



后续，我们还会为大家补充展示多条数据训练ACT和diffusion训练抓取实验。敬请期待。



## 致谢

- Robotis-OMY 机械臂资源来自 [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main)
- [MuJoco Parser Class](./mujoco_env/mujoco_parser.py) 改自 [yet-another-mujoco-tutorial-v3](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)
- 教程参考了 [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples)
- plate 与 mug 资源来自 [Objaverse](https://objaverse.allenai.org/)
