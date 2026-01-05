# Habitat-sim环境搭建及数据集介绍

### Habitat 是 Meta AI 开源的具身智能仿真平台，专为室内场景下的智能体导航、交互、决策等研究设计。其核心由两个互补组件构成：Habitat-Sim（仿真引擎）和 Habitat-Lab（算法框架），前者负责底层物理和视觉仿真，后者负责高层任务定义、算法开发与评估。

## 一、Habitat-sim特性介绍
### 1. 高保真渲染

* 支持基于物理的渲染，生成接近真实的 RGB 图像、深度图、语义分割图。

* 支持自定义传感器（如相机、激光雷达、IMU），可配置视角、分辨率、帧率。

### 2. 高效场景加载

* 原生支持 GLB/GLTF 格式，兼容主流场景数据集（Matterport3D、Gibson、Replica、HM3D）。

* 优化了大规模场景的内存占用和加载速度，适合批量训练。

### 3. 轻量级物理模拟

* 支持刚体物理、关节运动（如机械臂操作），满足智能体与场景物体的交互需求。

* 低延迟、GPU 加速，单 GPU 可同时运行数千个并行仿真环境。

### 4. 跨平台与扩展性

* 支持 Linux/Windows，兼容 CUDA。

* 提供 C++ 核心 + Python 绑定，可自定义扩展物理规则、传感器或渲染逻辑。

## 二、Habitat-sim环境搭建

创建conda环境（推荐在ubuntu22.04或者ubuntu24.04）
```python
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

使用conda安装habitat环境，我们可以根据论文的复现要求或者自己的需求，将以下的habitat-sim替换成habitat-sim=0.2.5这种版本格式，推荐使用habitat-sim=0.2.5，本教程均以0.2.5版本进行书写，如采用0.1版本或者其他的版本，可能出现函数调用错误，或者数据集不匹配的问题。

**如果为了学习使用，安装以下命令中的第一条即可，如需复现其他论文，可按第二个命令仿照第一条进行修改**

1）在自己的电脑上运行或者服务器在自己身边，有显示器的情况，**推荐**安装带有物理模拟的habitat-sim

```python
conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

2）在自己的电脑上运行或者服务器在自己身边，有显示器的情况，也可以安装不带物理模拟的habitat-sim

```python
conda install habitat-sim=0.2.5 -c conda-forge -c aihabitat
conda install habitat-sim -c conda-forge -c aihabitat
```

3）在租的服务器上，没有显示器的电脑上运行时，使用headless的habitat-sim（带有物理模拟的安装）

```python
conda install habitat-sim=0.2.5 withbullet headless -c conda-forge -c aihabitat
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

4）在租的服务器上，没有显示器的电脑上运行时，使用headless的habitat-sim（没有物理模拟的安装）

```python
conda install habitat-sim=0.2.5 -c conda-forge -c aihabitat
conda install habitat-sim -c conda-forge -c aihabitat
```

考虑到租用服务器的可能性较高，因此，测试仅需要使用example.py即可，在此之前还需要下载一些3d资产，包括一个3d场景和示例对象，关于数据集场景的具体介绍会在第三部分进行详细介绍。（命令中的path/to/可以直接删除变成data/，或者自己更换路径）

下载 3D 测试场景
```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/
```

下载示例对象

```python
python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /path/to/data/
```

example.py测试

```python
python /path/to/habitat-sim/examples/example.py --scene /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

如果有显示器的，可以运行以下代码进行测试
```python
habitat-viewer /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
#或者使用
python examples/viewer.py --scene /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

## 三、Habitat-sim数据集场景介绍

**教程中使用的数据集可以不需要注册授权即可下载，第三部分仅作介绍和为后续最新算法复现做铺垫**

在后续的最新算法实践中，我们通常会看到一些项目中会指出采用了hm3d的train以及val场景数据集，对于habitat-sim=0.2.5或者0.2.x版本的环境来说，我们通常使用的是HM3D v0.2的数据集，可以参考这个网站下载相关数据集：https://github.com/matterport/habitat-matterport-3dresearch?tab=readme-ov-file。

由于habitat需要授权，需要我们注册获取授权，流程也很简单，进入如下网页：https://my.matterport.com/settings/account/devtools?organization=Msg6zBCkcPg ，注册并发送验证邮箱即可，随后我们可以在这个网页的Developer Tools下的API Token Management中获取我们的API。

<img src="assets\habitat_api.png"/>

此时我们可以使用如下命令，获取我们所需要的数据集，其中`<api-token-id>`和`<api-token-secret>`填写刚刚获取到的API。**（现在先不用下载这个数据集，了解即可，教程中未使用这个数据集，仅用作后续复现论文参考）**

```python
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

对于habitat-sim项目中的datasets_download.py文件，通过API下载hm3d-train-habitat-v0.2或者其他的数据集，可以参考文件中的uids格式。

```python
data_groups.update(
{
        f"hm3d_val_{version}": [
            f"hm3d_val_habitat_{version}",
            f"hm3d_val_configs_{version}",
            f"hm3d_val_semantic_annots_{version}",
            f"hm3d_val_semantic_configs_{version}",
        ],
    }
)
data_groups.update(
    {
        f"hm3d_train_{version}": [
            f"hm3d_train_habitat_{version}",
            f"hm3d_train_configs_{version}",
            f"hm3d_train_semantic_annots_{version}",
            f"hm3d_train_semantic_configs_{version}",
        ],
    }
)
data_groups.update(
    {
        f"hm3d_minival_{version}": [
            f"hm3d_minival_habitat_{version}",
            f"hm3d_minival_configs_{version}",
            f"hm3d_minival_semantic_annots_{version}",
            f"hm3d_minival_semantic_configs_{version}",
        ]
    }
)
data_groups.update(
    {
        f"hm3d_semantics_{version}": [
            f"hm3d_example_semantic_annots_{version}",
            f"hm3d_example_semantic_configs_{version}",
            f"hm3d_val_semantic_annots_{version}",
            f"hm3d_val_semantic_configs_{version}",
            f"hm3d_train_semantic_annots_{version}",
            f"hm3d_train_semantic_configs_{version}",
            f"hm3d_minival_semantic_annots_{version}",
            f"hm3d_minival_semantic_configs_{version}",
        ]
    }
)
```

例如，下载hm3d-train-habitat-v0.2可以采用如下命令进行下载。
```python
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_train_habitat_v0.2
```

包括其他下载的资产也可以在datasets_download.py文件中找到，例如刚刚下载的habitat_test_scenes测试环境以及habitat_example_objects，于datasets_download.py中的代码块如下

```python
"habitat_test_scenes": {
    "source": "https://huggingface.co/datasets/ai-habitat/habitat_test_scenes.git",
    "link": data_path + "scene_datasets/habitat-test-scenes",
    "version": "main",
}
```

```python
"habitat_example_objects": {
    "source": "http://dl.fbaipublicfiles.com/habitat/objects_v0.2.zip",
    "package_name": "objects_v0.2.zip",
    "link": data_path + "objects/example_objects",
    "version": "0.2",
}
```