# Habitat-sim基础实践

### 本章中将完成从habitat-sim入门到上手这个目标，以Habitat-Sim 0.2.5 版本（当前应用最广泛的稳定版）为基准，从核心模块的入门使用，再到基础仿真任务的实现。我们将避开纯理论堆砌，以 “问题驱动” 为导向 —— 每个实践步骤都对应实际开发中最常遇到的需求（如如何快速加载 Matterport3D 场景、如何配置 RGB-D 传感器等），为后续进阶的具身导航算法实现打下坚实的基础。

## 一、Habitat-sim的Hello World
### 1. 下载MP3D的示例场景数据

```python
conda activate habitat
```

```python
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
```

### 2. Habitat-sim基础功能测试

运行habitat_test.py文件
```python
python habitat_test.py
```

<div style="display: flex; justify-content: space-between;">
  <img src="assets\observation_0.png" alt="Image 1" style="width: 48%; margin-right: 2%;">
  <img src="assets\observation_1.png" alt="Image 2" style="width: 48%;">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="assets\observation_2.png" alt="Image 1" style="width: 48%; margin-right: 2%;">
  <img src="assets\observation_3.png" alt="Image 2" style="width: 48%;">
</div>

### 3. 代码详解

首先是环境配置设置，核心函数是make_simple_cfg()，用来配置环境和机器人的传感器，其中test_scene为自己存放的环境数据路径。

```python
test_scene = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
sim_settings = {
    "scene": test_scene,
    "default_agent": 0,
    "sensor_height": 1.5,
    "width": 256,
    "height": 256,
}

def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]


    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
```

随后对智能体进行初始化，设置机器人的初始状态，并发布机器人的状态信息。
```python
agent = sim.initialize_agent(sim_settings["default_agent"])

agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])
agent.set_state(agent_state)

agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)
```

定义机器人的运动函数navigateAndSee(),并执行动作，将display设置为True，即可保存每一步行动后的图像。
```python
def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])


action = "turn_right"
navigateAndSee(action)

action = "turn_right"
navigateAndSee(action)

action = "move_forward"
navigateAndSee(action)

action = "turn_left"
navigateAndSee(action)
```

以下为display_sample()显示函数，可以用来保存机器人每一步执行后的图像。

```python
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    global img_counter

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    
    image_name = f"observation_{img_counter}.png"
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    img_counter +=1
    
    plt.show(block=False)
    plt.pause(3)  # 显示3秒（可修改秒数）
    plt.close()
```

## 二、Habitat-sim进阶配置(见habitat_random.py文件)
### 1. 传感器配置讲解

在Habitat-sim的Hello World中，我们使用了Color Sensor这个RGB颜色传感器传感器，常用的还有Depth Sensor(深度传感器)和Semantic Sensor(语义传感器)。

1） Color Sensor（RGB 颜色传感器）

模拟人类视觉 / 普通 RGB 相机，捕捉场景的色彩、纹理、外观信息，是最基础的视觉感知传感器，对应代码中 habitat_sim.SensorType.COLOR 。

关键配置代码如下：
```python
color_sensor_spec = habitat_sim.CameraSensorSpec()
color_sensor_spec.uuid = "color_sensor"
color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
color_sensor_spec.resolution = [settings["height"], settings["width"]]
color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
sensor_specs.append(color_sensor_spec)
```

输出格式如下：

* 数据类型：uint8（0-255），4 通道数组（RGBA，A 为透明度，通常为 255）；
* 形状：[height, width, 4]（如 256×256×4）；
* 数值含义：每个像素的红、绿、蓝、透明度值（比如 (255,0,0,255) 对应纯红色）。

核心作用：提供场景的直观视觉特征：比如识别物体外观、场景布局、纹理差异。

2） Depth Sensor(深度传感器)

模拟真实深度相机，捕捉每个像素对应的物理距离（以米为单位），反映场景的 3D 空间结构，对应代码中 habitat_sim.SensorType.DEPTH。

关键配置代码如下：
```python
depth_sensor_spec = habitat_sim.CameraSensorSpec()
depth_sensor_spec.uuid = "depth_sensor"
depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor_spec.resolution = [settings["height"], settings["width"]]
depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
sensor_specs.append(depth_sensor_spec)
```

输出格式如下：

* 数据类型：float32（浮点型），单通道数组；
* 形状：[height, width]（如 256×256）；
* 数值含义：每个像素对应的场景点到相机的直线距离（米），比如数值0.5表示该点距离相机 0.5 米，5.0表示距离 5 米。

核心作用：补充 3D 空间信息，弥补 RGB 仅能提供 2D 外观的不足，还原场景的几何结构；数值换算，通过深度值 + 相机内参，将像素坐标转换为真实世界的 3D 坐标。

3） Semantic Sensor(语义传感器)

模拟语义感知相机，捕捉每个像素对应的物体语义 ID（绑定到 3D 场景中的物体），反映每个像素属于哪个物体 / 类别，对应代码中 habitat_sim.SensorType.SEMANTIC。

关键配置代码如下：

```python
semantic_sensor_spec = habitat_sim.CameraSensorSpec()
semantic_sensor_spec.uuid = "semantic_sensor"
semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
sensor_specs.append(semantic_sensor_spec)
```

输出格式如下：

* 数据类型：uint32/int32，单通道数组；
* 形状：[height, width]（如 256×256）；
* 数值含义：每个像素对应的物体语义 ID（如 1 = 墙、5 = 床、8 = 桌子），需通过 JSON 配置文件（mp3d.scene_dataset_config.json）映射为人类可读的类别名。**（这里的mp3d.scene_dataset_config.json会在后续做讲解）**。

核心作用：高层语义理解，从像素数值到物体类别，实现对场景的语义认知（比如识别哪个像素是墙、哪个是床）；统计场景内物体分布（比如卧室里有 1 张床、2 把椅子）。

在habitat_random.py文件中，为了代码的简洁性和易修改性，对以上传感器的代码做了整合，写成如下格式，但是功能是一样的。

```python
sensors = {
    "color_sensor": {
        "sensor_type": habitat_sim.SensorType.COLOR,
        "resolution": [settings["height"], settings["width"]],
        "position": [0.0, settings["sensor_height"], 0.0],
    },
    "depth_sensor": {
        "sensor_type": habitat_sim.SensorType.DEPTH,
        "resolution": [settings["height"], settings["width"]],
        "position": [0.0, settings["sensor_height"], 0.0],
    },
    "semantic_sensor": {
        "sensor_type": habitat_sim.SensorType.SEMANTIC,
        "resolution": [settings["height"], settings["width"]],
        "position": [0.0, settings["sensor_height"], 0.0],
    },
}

sensor_specs = []
for sensor_uuid, sensor_params in sensors.items():
    if settings[sensor_uuid]:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]

        sensor_specs.append(sensor_spec)
```

### 2. 动作配置讲解

在habitat_test.py文件中，我们没有定义机器人的动作空间，但是AgentConfiguration() 会自动加载默认动作空间，如果我们想要自定义向前移动的距离和转弯的角度，我们就需要自定义机器人的动作空间，如下所示：

```python
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = sensor_specs
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
}
```

### 3. 代码执行结果与json文件分析

运行habitat_random.py

```python
python habitat_random.py
```
运行完这个文件后，我们会得到如下的结果：
<img src="assets\random_0.png"/>
<img src="assets\random_1.png"/>
<img src="assets\random_2.png"/>
<img src="assets\random_3.png"/>
<img src="assets\random_4.png"/>

我们可以看到，rgb图像和深度图像都是正常的，但是语义图像并没有正常显示，这主要是因为我们没有加载mp3d.scene_dataset_config.json文件，这个文件可以加载我们需要的像素语义对应关系，可映射到具体的物体类别，同时还可以加载mesh地图文件。（关于mesh地图的内容在habitat_sim的navmesh详解中介绍）。

在habitat_random.py代码中我们将这两行注释取消，即可加载mp3d.scene_dataset_config.json文件。

```python
# mp3d_scene_dataset = "./data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"

# "scene_dataset": mp3d_scene_dataset,

# sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
```

注释取消后，代码变成：

```python
test_scene = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
mp3d_scene_dataset = "./data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "scene_dataset": mp3d_scene_dataset,
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}
```

此时，我们可以得到的输出结果为：

<img src="assets\randomtest_0.png"/>
<img src="assets\randomtest_1.png"/>
<img src="assets\randomtest_2.png"/>
<img src="assets\randomtest_3.png"/>
<img src="assets\randomtest_4.png"/>

可以看出，semantic图片能够正常的显示。

## 三、Habitat-sim的NavMesh详解

### 1. NavMesh 介绍

NavMesh（Navigation Mesh）是 Habitat-Sim中场景导航与运动规划的核心空间表示，本质是覆盖场景中智能体可通行区域的多边形网格，是实现智能体移动、路径查找、碰撞避免的基础。

核心作用:

* 可导航区域判定：标记场景中智能体能行走的区域（如地面、走廊），排除障碍物（墙壁、家具、悬空区域等）。
* 路径规划基础：Habitat 的 PathFinder 基于 NavMesh 快速计算两点间的最优可行路径。
* 运动合法性校验：智能体执行移动转向动作时，通过 NavMesh 校验目标位置是否可通行，避免穿墙或者卡入障碍物。
* 空间抽象降维：将复杂的 3D 场景几何结构抽象为 2D 的导航网格，大幅降低运动规划的计算复杂度。

### 2. habitat_mesh.py代码详解

habitat_mesh.py主要为了完成 NavMesh 生成并可视化场景俯视图。

前面的仿真环境和机器人初始化部分与本章中habitat_random.py部分的代码一致，不同之处主要有两个。

一个是使用Habitat_sim中的API生成的俯视图

```python
sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
```

一个是使用get_topdown_map函数来生成的俯视图

```python
def get_topdown_map(pathfinder, height, meters_per_pixel) -> np.ndarray:
    # 获取场景的导航边界（x, z轴，忽略y轴高度）
    bounds = pathfinder.get_bounds()
    min_x, _, min_z = bounds[0]
    max_x, _, max_z = bounds[1]

    # 计算地图的像素尺寸（x对应宽度，z对应高度）
    map_width = int(np.ceil((max_x - min_x) / meters_per_pixel))
    map_height = int(np.ceil((max_z - min_z) / meters_per_pixel))

    # 初始化地图：0=不可导航，1=可导航
    topdown_map = np.zeros((map_height, map_width), dtype=np.uint8)

    # 遍历每个像素，判断是否可导航（优化：用向量化操作替代双重循环，提升速度）
    x_coords = np.linspace(min_x, max_x, map_width, endpoint=False)
    z_coords = np.linspace(min_z, max_z, map_height, endpoint=False)
    x_grid, z_grid = np.meshgrid(x_coords, z_coords)
    # 构造(x, height, z)的坐标数组
    world_coords = np.stack([x_grid.ravel(), np.full_like(x_grid.ravel(), height), z_grid.ravel()], axis=1)
    # 批量判断是否可导航
    navigable = np.array([pathfinder.is_navigable(coord) for coord in world_coords])
    # 重塑为地图尺寸
    topdown_map = navigable.reshape((map_height, map_width)).astype(np.uint8)

    # 计算边界（可选，匹配旧版本的2值）
    edges = ndimage.laplace(topdown_map) != 0
    topdown_map[edges] = 2

    return topdown_map
```

get_topdown_view()函数返回自定义函数返回带边界标记的 0/1/2 数组，使用颜色重映射：0→白（不可导航）、1→灰（可导航）、2→黑（边界）。


```python
hablab_topdown_map = get_topdown_map(
    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
)
recolor_map = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)
hablab_topdown_map = recolor_map[hablab_topdown_map]
```

对于这里自定义的get_topdown_view()函数，也可以直接使用，但是这里的maps需要下载Habitat_lab，对于想要用habitat_sim来完成整个教程的，可以直接使用上面自定义的函数，所得到的效果是一样的。

```python
from habitat.utils.visualizations import maps
hablab_topdown_map = maps.get_topdown_map(
    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
)
```

最终得到的mesh图结果如下所示：

1） 使用sim.pathfinder.get_topdown_view生成的俯视图

<img src="assets\navmesh_0.png"/>

2） 使用get_topdown_map生成的俯视图

<img src="assets\navmesh_1.png"/>

### 3. habitat_pathfind.py代码详解

habitat_pathfind.py 是在 habitat_mesh.py 场景俯视图生成的基础上，新增了场景内最短路径查找、路径与智能体可视化、路径点传感器观测渲染三大核心功能，实现了从生成场景地图到规划并可视化导航路径的完整流程。

复用了 habitat_mesh.py 中的 make_cfg、get_topdown_map、display_map 函数，新增以下核心函数实现路径相关的坐标转换、可视化绘制和传感器观测渲染：

1） to_grid函数：世界坐标转地图像素网格坐标

该函数替代 Habitat-Lab 的 maps.to_grid 功能，解决 3D 世界坐标到 2D 地图像素坐标的映射问题，为路径绘制提供基础。

参数说明：

* z: 世界坐标的z值（path_point[2]）
* x: 世界坐标的x值（path_point[0]）
* grid_dimensions: 地图的尺寸 (height, width)（对应top_down_map.shape[0], shape[1]）
* pathfinder: sim.pathfinder实例

```python
def to_grid(z, x, grid_dimensions, pathfinder):
    # 获取导航网格的边界（min_x, _, min_z）和（max_x, _, max_z）
    min_bounds = pathfinder.get_bounds()[0]
    max_bounds = pathfinder.get_bounds()[1]
    min_x, _, min_z = min_bounds
    max_x, _, max_z = max_bounds

    # 归一化x到[0, grid_width-1]，z到[0, grid_height-1]
    grid_height, grid_width = grid_dimensions
    px = (x - min_x) / (max_x - min_x) * (grid_width - 1)
    py = (z - min_z) / (max_z - min_z) * (grid_height - 1)

    # 限制坐标在地图范围内，避免越界
    px = np.clip(px, 0, grid_width - 1)
    py = np.clip(py, 0, grid_height - 1)

    # 转换为整数像素坐标
    return int(round(py)), int(round(px))
```

2）draw_path函数：在俯视图上绘制路径线段

该函数替代 Habitat-Lab 的 maps.draw_path 功能，在 RGB 格式的俯视图上绘制红色路径线段，直观展示最短路径。

参数说明：

* top_down_map: RGB格式的地图数组（np.ndarray，shape=(H, W, 3)）
* trajectory: 网格坐标的列表，每个元素是(py, px)（行，列）
* color: 路径颜色，默认红色(255,0,0)
* thickness: 线条粗细，默认2像素

```python
def draw_path(top_down_map, trajectory, color=(255, 0, 0), thickness=2):
    # 转换为PIL Image以便绘制线段（也可用OpenCV）
    img = Image.fromarray(top_down_map)
    draw = ImageDraw.Draw(img)

    # 遍历轨迹点，绘制线段
    for i in range(len(trajectory) - 1):
        # PIL的坐标是（x, y），对应网格坐标的（px, py）
        start = (trajectory[i][1], trajectory[i][0])
        end = (trajectory[i+1][1], trajectory[i+1][0])
        draw.line([start, end], fill=color, width=thickness)

    # 转换回numpy数组
    top_down_map[:] = np.array(img)
```

3）draw_agent函数：在俯视图上绘制智能体

该函数替代 Habitat-Lab 的 maps.draw_agent 功能，在俯视图上标注智能体的位置和朝向，清晰展示导航起点状态。

参数说明：
* top_down_map: RGB格式的地图数组（np.ndarray，shape=(H, W, 3)）
* agent_pos: 智能体的网格坐标(py, px)（行，列）
* angle: 智能体的朝向角度（弧度）
* agent_radius_px: 智能体圆形半径，默认8像素
* agent_color: 智能体圆形颜色，默认绿色(0,255,0)
* arrow_color: 朝向箭头颜色，默认蓝色(0,0,255)

```python
def draw_agent(top_down_map, agent_pos, angle, agent_radius_px=8, agent_color=(0, 255, 0), arrow_color=(0, 0, 255)):
    # 转换为PIL Image以便绘制
    img = Image.fromarray(top_down_map)
    draw = ImageDraw.Draw(img)
    py, px = agent_pos  # 网格坐标（行，列）
    x, y = px, py  # PIL坐标（x=列，y=行）

    # 1. 绘制智能体的圆形身体
    # PIL的椭圆绘制需要左上角和右下角坐标
    bbox = (x - agent_radius_px, y - agent_radius_px, x + agent_radius_px, y + agent_radius_px)
    draw.ellipse(bbox, fill=agent_color, outline=(0, 0, 0), width=1)

    # 2. 绘制朝向箭头（根据角度计算箭头终点）
    arrow_length = agent_radius_px * 1.5
    # 角度转换：math.atan2的角度是从x轴逆时针，这里调整为地图的朝向
    end_x = x + arrow_length * math.cos(angle)
    end_y = y + arrow_length * math.sin(angle)
    draw.line([(x, y), (end_x, end_y)], fill=arrow_color, width=2)

    # 转换回numpy数组
    top_down_map[:] = np.array(img)
```

同时，与habitat_mesh.py中的get_topdown_map函数类似，habitat_pathfind.py中的to_grid、draw_path 和 draw_agent 函数，皆可以用 maps.to_grid、maps.draw_path 和 maps.draw_agent 来替代。

**主逻辑流程详解**

主函数部分主要利用 Habitat-Sim 的路径查找 API 实现，除了教程详解之外，代码中也做了详细的注释。

步骤 1：仿真环境初始化（复用逻辑）

habitat_pathfind.py 与 habitat_mesh.py 一致，通过 make_cfg 配置仿真器的场景、传感器和智能体动作空间，初始化 Simulator 实例。

步骤 2：生成随机可导航点并查找最短路径（核心步骤）

```python
seed = 4
sim.pathfinder.seed(seed)
# 生成两个随机可导航点（场景内可行走的位置）
sample1 = sim.pathfinder.get_random_navigable_point()
sample2 = sim.pathfinder.get_random_navigable_point()

# 初始化最短路径对象
path = habitat_sim.ShortestPath()
path.requested_start = sample1  # 路径起点
path.requested_end = sample2    # 路径终点
# 查找最短路径
found_path = sim.pathfinder.find_path(path)
# 获取路径信息：geodesic距离（地面真实距离）、路径点列表
geodesic_distance = path.geodesic_distance
path_points = path.points
```

关键函数说明：

* get_random_navigable_point()：从场景 NavMesh 中随机采样一个可导航的 3D 坐标；
* ShortestPath：Habitat-Sim 的最短路径类，需指定起点 / 终点；
* find_path()：执行路径查找，返回布尔值（是否找到有效路径）；
* geodesic_distance：路径的地面真实距离（米）；
* path.points：路径的 3D 坐标列表（按顺序存储起点→终点的所有路径点）。

步骤 3：路径可视化（俯视图绘制路径 + 智能体）

```python
if found_path:
    meters_per_pixel = 0.025  # 更高分辨率（0.025米/像素）
    # 获取场景最低高度（匹配NavMesh高度）
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    height = scene_bb.y().min
    # 生成带边界的俯视图（复用get_topdown_map）
    top_down_map = get_topdown_map(sim.pathfinder, height, meters_per_pixel)
    # 颜色重映射（0→白，1→灰，2→黑）
    recolor_map = np.array([[255,255,255],[128,128,128],[0,0,0]], dtype=np.uint8)
    top_down_map = recolor_map[top_down_map]
    
    # 路径坐标转换：世界坐标→像素网格坐标
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    trajectory = [to_grid(p[2], p[0], grid_dimensions, sim.pathfinder) for p in path_points]
    
    # 计算路径起点的朝向角度（基于路径第二个点）
    grid_tangent = mn.Vector2(trajectory[1][1]-trajectory[0][1], trajectory[1][0]-trajectory[0][0])
    path_initial_tangent = grid_tangent / grid_tangent.length()
    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
    
    # 绘制路径（红色）和智能体（绿色圆形+蓝色箭头）
    draw_path(top_down_map, trajectory)
    draw_agent(top_down_map, trajectory[0], initial_angle)
    # 显示并保存带路径的俯视图
    display_map(top_down_map)
```

关键细节：
* 提升地图分辨率（meters_per_pixel=0.025）：让路径绘制更精细；
* 朝向角度计算：通过路径起点和第二个点的像素坐标差，计算智能体的初始朝向；
* 颜色映射：保持与 habitat_mesh.py 一致的视觉风格，路径用红色突出显示。

步骤 4：路径点传感器观测渲染

遍历路径上的每个点，设置智能体位姿（位置 + 朝向），获取传感器观测并可视化：

```python
display_path_agent_renders = True
if display_path_agent_renders:
    print("Rendering observations at path points:")
    tangent = path_points[1] - path_points[0]
    agent_state = habitat_sim.AgentState()
    for ix, point in enumerate(path_points):
        if ix < len(path_points) - 1:
            # 更新朝向：指向当前点到下一个点的方向
            tangent = path_points[ix + 1] - point
            agent_state.position = point  # 设置智能体位置
            # 计算朝向矩阵（look_at：从当前点看向下一个点）
            tangent_orientation_matrix = mn.Matrix4.look_at(point, point + tangent, np.array([0, 1.0, 0]))
            # 转换为四元数（Habitat-Sim智能体旋转格式）
            tangent_orientation_q = mn.Quaternion.from_matrix(tangent_orientation_matrix.rotation())
            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
            agent.set_state(agent_state)  # 应用智能体位姿
            
            # 获取传感器观测
            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"]
            semantic = observations["semantic_sensor"]
            depth = observations["depth_sensor"]
            
            # 渲染并显示观测结果
            if display:
                display_sample(rgb, semantic, depth)
```

核心逻辑：

* 遍历路径点（除最后一个），为每个点设置智能体位置；
* 计算朝向：让智能体始终朝向路径的下一个点（通过 mn.Matrix4.look_at 生成朝向矩阵）；
* 转换朝向矩阵为四元数（Habitat-Sim 智能体旋转的标准格式）；
* 应用位姿后获取 RGB、深度、语义传感器数据，调用 display_sample 渲染显示。

最终得到的俯视图以及第一视角结果如下所示：

俯视图如下：

<img src="assets\pathfind2_0.png"/>

机器人第一视角运行视图如下：

<img src="assets\pathfind2_1.png"/>

<img src="assets\pathfind2_2.png"/>

<img src="assets\pathfind2_3.png"/>

<img src="assets\pathfind2_4.png"/>

<img src="assets\pathfind2_5.png"/>

<img src="assets\pathfind2_6.png"/>

<img src="assets\pathfind2_7.png"/>

<img src="assets\pathfind2_8.png"/>

<img src="assets\pathfind2_9.png"/>

<img src="assets\pathfind2_10.png"/>

<img src="assets\pathfind2_11.png"/>

参考资料：

https://github.com/facebookresearch/habitat-sim

