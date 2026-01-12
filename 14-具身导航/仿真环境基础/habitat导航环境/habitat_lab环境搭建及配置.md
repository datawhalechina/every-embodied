# Habitat-lab仿真基础

### Habitat-Lab 是基于 Habitat-Sim 的算法层，封装了标准化的任务、评估指标和 API，让研究者无需关注底层仿真细节，专注于具身智能算法的设计、训练与评估。

## 一、Habitat-lab特性介绍

### 1. 预定义具身任务

* 导航类：PointGoalNav（点目标导航）、ObjectNav（物体目标导航）、VLN（视觉语言导航）。
* 交互类：Rearrange（物体重排）、PickPlace（拾取放置）、OpenDoor（开门）。
* 支持自定义任务（如多智能体协作、长程规划）。

### 2.标准化评估体系

* 内置核心指标（成功率 SR、路径长度效率 SPL、DTW 对齐分数等），可直接对比算法性能。
* 支持与公开基准（如 Habitat Challenge）对齐，便于论文复现和成果对比。

### 3.模块化智能体架构

* 解耦 “传感器 → 编码器 → 策略 → 控制器”，可快速替换组件（如用 CNN 或者 Transformer 做视觉编码，用 RL/IL 做决策）；
* 深度集成 PyTorch，支持强化学习（RL）、模仿学习（IL）、端到端深度学习等主流算法。

### 4.易用的 API

* 提供简洁的 Python 接口，一键加载场景、初始化智能体、运行仿真循环。

## 二、Habitat-lab环境搭建

这里可以直接使用habitat-sim创建的conda环境

```python
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

在下载habitat-lab的时候注意需要与安装的habitat-sim版本一致，例如本教程使用的habitat-sim版本是0.2.5，则需要下载0.2.5版本的habitat-lab，复现其他论文的时候也要注意habitat-sim和habitat-lab版本一致。

```python
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

同时安装habitat-baselines

```python
pip install -e habitat-baselines
```

下载3D场景数据和点导航数据

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```

## 三、Habitat-lab初探与yaml配置基础讲解

### 本节将以habitat-lab的项目中提供的交互式导航代码habitatlab_test.py为核心，详细讲解 Habitat-Lab 中 yaml 配置文件的编写逻辑与加载流程，对于后续更加复杂的 yaml 文件的阅读提供基本的思路。

### 1. habitatlab_test.py的yaml配置基础及实现

habitatlab_test.py的运行流程：

1） 加载 YAML 配置 

2） 初始化环境

3） 重置环境获取观测 

4） 按键交互控制智能体 

5） 依据配置的最大步数终止循环。

可以看出与Habitat-sim最不同的部分就是YAML的加载和配置，在导入yaml配置后，初始化其实还是用habitat-sim来运行的，下面将对pointnav_habitat_test.yaml文件做详细的讲解。

```python
# @package _global_

defaults:
  - pointnav_base
  - /habitat/dataset/pointnav: habitat_test
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agents:
      main_agent:
        sim_sensors:
          rgb_sensor:
            width: 256
            height: 256
          depth_sensor:
            width: 256
            height: 256

```

yaml配置输入：
1. 加载pointnav_base.yaml（基础 PointNav 配置）
2. 覆盖 /habitat/dataset/pointnav 为 habitat_test
3. 用当前文件的habitat.environment.max_episode_steps:500、simulator.agents.main_agent.sim_sensors覆盖基础配置

**刚开始使用habitat-lab的初学者可能会很疑惑，那这些就全部配置结束了？其实不是，在代码中，有很多的默认配置，最终通过habitat-lab解析，传入habitat-sim完成环境和智能体的创建，下面将给大家进行详细的讲解，来疏通大家的困惑**

具体的yaml可配置项可以参考:https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md

以下关于yaml配置的代码均参考habitat-lab中的default_structured_configs.py文件中的函数，可去源文件中进行察看。

**yaml配置详解：**

**1） yaml中的数据集配置详解**

```python
class DatasetConfig(HabitatBaseConfig):
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: List[str] = field(default_factory=lambda: ["*"])
    data_path: str = (
        "data/datasets/pointnav/"
        "habitat-test-scenes/v1/{split}/{split}.json.gz"
    )
    # TODO: Make this field a structured dataclass.
    metadata: Optional[Any] = None
```

关键配置项的默认值说明：

* habitat.dataset.type：默认值为PointNav-v1，表示点导航任务数据集
* habitat.dataset.split：默认值为train，表示训练集分割
* habitat.dataset.scene_dir：默认值为data/scene_datasets，表示场景数据集的根目录
* habitat.dataset.data_path：默认值为data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz，表示数据集文件路径，其中{split}会被实际的 split 值替换。
* 数据集配置还支持通过content_scenes字段指定要加载的具体场景列表，默认情况下会加载所有可用场景。

**2） yaml中的任务配置详解**

```python
@dataclass
class TaskConfig(HabitatBaseConfig):
    physics_target_sps: float = 60.0
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    # Temporary structure for sensors
    lab_sensors: Dict[str, LabSensorConfig] = field(default_factory=dict)
    measurements: Dict[str, MeasurementConfig] = field(default_factory=dict)
    # Measures to only construct in the first environment of the first rank for
    # vectorized environments.
    rank0_env0_measure_names: List[str] = field(
        default_factory=lambda: ["habitat_perf"]
    )
    # Measures to only record in the first rank for vectorized environments.
    rank0_measure_names: List[str] = field(default_factory=list)
    goal_sensor_uuid: str = "pointgoal"
    # REARRANGE task
    count_obj_collisions: bool = True
    settle_steps: int = 5
    constraint_violation_ends_episode: bool = True
    constraint_violation_drops_object: bool = False
    # Forced to regenerate the starts even if they are already cached
    force_regenerate: bool = False
    # Saves the generated starts to a cache if they are not already generated
    should_save_to_cache: bool = False
    object_in_hand_sample_prob: float = 0.167
    min_start_distance: float = 3.0
    gfx_replay_dir = "data/replays"
    render_target: bool = True
    # Spawn parameters
    filter_colliding_states: bool = True
    num_spawn_attempts: int = 200
    spawn_max_dist_to_obj: float = 2.0
    base_angle_noise: float = 0.523599
    spawn_max_dist_to_obj_delta: float = 0.02
    # Factor to shrink the receptacle sampling volume when predicates place
    # objects on top of receptacles.
    recep_place_shrink_factor: float = 0.8
    # EE sample parameters
    ee_sample_factor: float = 0.2
    ee_exclude_region: float = 0.0
    base_noise: float = 0.05
    spawn_region_scale: float = 0.2
    joint_max_impulse: float = -1.0
    desired_resting_position: List[float] = field(
        default_factory=lambda: [0.5, 0.0, 1.0]
    )
    use_marker_t: bool = True
    cache_robot_init: bool = False
    success_state: float = 0.0
    # Measurements for composite tasks.
    should_enforce_target_within_reach: bool = False
    # COMPOSITE task CONFIG
    task_spec_base_path: str = "habitat/task/rearrange/pddl/"
    task_spec: str = ""
    # PDDL domain params
    pddl_domain_def: str = "replica_cad"
    obj_succ_thresh: float = 0.3
    # Disable drop except for when the object is at its goal.
    enable_safe_drop: bool = False
    art_succ_thresh: float = 0.15
    robot_at_thresh: float = 2.0

    # The minimum distance between the agents at start. If < 0
    # there is no minimal distance
    min_distance_start_agents: float = -1.0
    actions: Dict[str, ActionConfig] = MISSING
```

关键配置项的默认值说明：

* habitat.task.type：默认值为Nav-v0，表示导航任务​
* habitat.task.physics_target_sps：默认值为60.0，表示物理引擎的更新频率为每秒 60 次​
* habitat.task.reward_measure：默认值为None，需要在具体配置中指定奖励测量指标​
* habitat.task.success_measure：默认值为None，需要在具体配置中指定成功测量指标​
* habitat.task.success_reward：默认值为2.5，表示成功完成任务的奖励​
* habitat.task.slack_reward：默认值为-0.01，表示每步的基础奖励（惩罚）​
* habitat.task.end_on_success：默认值为False，表示成功后不会自动结束 episode​
* 任务配置还包含了大量与重排任务相关的默认值，如settle_steps（默认 5 步）、constraint_violation_ends_episode（默认 True）等。

**3） yaml中的环境配置详解**

```python
class EnvironmentConfig(HabitatBaseConfig):
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()
```

关键配置项的默认值说明：

* habitat.environment.max_episode_steps：默认值为1000，表示 episode 的最大步数限制​
* habitat.environment.max_episode_seconds：默认值为10000000，表示 episode 的最大时间限制​
* habitat.environment.iterator_options：包含多个迭代器选项的默认配置，如循环模式（cycle）默认为 True、随机洗牌（shuffle）默认为 True 等

**4） yaml中的模拟器配置详解**

```python
class SimulatorConfig(HabitatBaseConfig):
    type: str = "Sim-v0"
    forward_step_size: float = 0.25  # in metres
    turn_angle: int = 10  # angle to rotate left or right in degrees
    create_renderer: bool = False
    requires_textures: bool = True
    # Sleep options
    auto_sleep: bool = False
    step_physics: bool = True
    concur_render: bool = False
    # If markers should be updated at every step:
    needs_markers: bool = True
    # If the articulated_agent camera positions should be updated at every step:
    update_articulated_agent: bool = True
    scene: str = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    # The scene dataset to load in the metadatamediator,
    # should contain simulator.scene:
    scene_dataset: str = "default"
    # A list of directory or config paths to search in addition to the dataset
    # for object configs. should match the generated episodes for the task:
    additional_object_paths: List[str] = field(default_factory=list)
    # Use config.seed (can't reference Config.seed) or define via code
    # otherwise it leads to circular references:
    #
    seed: int = II("habitat.seed")
    default_agent_id: int = 0
    debug_render: bool = False
    debug_render_articulated_agent: bool = False
    kinematic_mode: bool = False
    # If False, will skip setting the semantic IDs of objects in
    # `rearrange_sim.py` (there is overhead to this operation so skip if not
    # using semantic information).
    should_setup_semantic_ids: bool = True
    # If in render mode a visualization of the rearrangement goal position
    # should also be displayed
    debug_render_goal: bool = True
    robot_joint_start_noise: float = 0.0
    # Rearrange agent setup
    ctrl_freq: float = 120.0
    ac_freq_ratio: int = 4
    load_objs: bool = True
    # Rearrange agent grasping
    hold_thresh: float = 0.15
    grasp_impulse: float = 10000.0
    # we assume agent(s) to be set explicitly
    agents: Dict[str, AgentConfig] = MISSING
    # agents_order specifies the order in which the agents
    # are stored on the habitat-sim side.
    # In other words, the order to return the observations and accept
    # the actions when using the environment API.
    # If the number of agents is greater than one,
    # then agents_order has to be set explicitly.
    agents_order: List[str] = MISSING

    # Simulator should use default navmesh settings from agent config
    default_agent_navmesh: bool = True
    # if default navmesh is used, should it include static objects
    navmesh_include_static_objects: bool = False

    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()
    # ep_info is added to the config in some rearrange tasks inside
    # merge_sim_episode_with_object_config
    ep_info: Optional[Any] = None
    # The offset id values for the object
    object_ids_start: int = 100
    # Configuration for rendering
    renderer: RendererConfig = RendererConfig()
```

关键配置项的默认值说明：

* habitat.simulator.type：默认值为Sim-v0​
* habitat.simulator.forward_step_size：默认值为0.25米，控制前进动作的步长​
* habitat.simulator.turn_angle：默认值为10度，控制旋转动作的角度​
* habitat.simulator.create_renderer：默认值为False​
* habitat.simulator.requires_textures：默认值为True​
* habitat.simulator.lag_observations：默认值为0​
* habitat.simulator.auto_sleep：默认值为False​
* habitat.simulator.step_physics：默认值为True

**5） yaml中的Agent配置详解**

```python
class AgentConfig(HabitatBaseConfig):
    height: float = 1.5
    radius: float = 0.1
    max_climb: float = 0.2
    max_slope: float = 45.0
    grasp_managers: int = 1
    sim_sensors: Dict[str, Any] = field(default_factory=dict)
    is_set_start_state: bool = False
    start_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    start_rotation: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    joint_start_noise: float = 0.1
    joint_that_can_control: Optional[List[int]] = None
    # Hard-code the robot joint start. `joint_start_noise` still applies.
    joint_start_override: Optional[List[float]] = None
    articulated_agent_urdf: Optional[str] = None
    articulated_agent_type: Optional[str] = None
    ik_arm_urdf: Optional[str] = None
    # File to motion data, used to play pre-recorded motions
    motion_data_path: str = ""
    auto_update_sensor_transform: bool = True
```

* habitat.simulator.agents.main_agent.height：默认值为1.5米，定义智能体的基础高度，直接影响传感器（如RGB、深度相机）的观测视角，默认贴合人类视角高度。
* habitat.simulator.agents.main_agent.radius：默认值为0.1米，设置智能体的碰撞半径，用于模拟器中碰撞检测，避免智能体穿透场景物体。
* habitat.simulator.agents.main_agent.auto_update_sensor_transform：默认值为True，每帧自动更新传感器相对于智能体的位姿；关闭后需手动更新，适用于自定义传感器布局场景。


**结合以上的默认配置和pointnav_habitat_test.yaml以及pointnav_base.yaml，可以得到一个示意的yaml完整文件如下。（具体代码运行的时候不会生成这个yaml文件，只是一个示意的yaml供参考使用）**

```python
# @package _global_
defaults:
  # 基础PointNav配置（包含核心默认逻辑，优先级低于后续配置）
  - pointnav_base
  # 引用habitat_test数据集预定义配置
  - /habitat/dataset/pointnav: habitat_test
  # 加载导航任务所需离散动作（前进、转向、停止等）
  - /habitat/task/actions:
    - move_forward
    - turn_left
    - turn_right
    - stop
    - look_up
    - look_down
  # 加载导航任务核心测量指标（步数、距离、成功率、SPL等）
  - /habitat/task/measurements:
    - num_steps
    - distance_to_goal
    - success
    - spl
    - distance_to_goal_reward
  # 加载代码依赖的传感器（GPS+罗盘+目标点传感器）
  - /habitat/task/lab_sensors:
    - pointgoal_with_gps_compass_sensor
    - compass_sensor
    - gps_sensor
  # 自身配置（优先级最高，覆盖前面的默认值）
  - _self_

# 核心配置主体
habitat:
  # 环境配置（用户定义500步上限，保留并补充默认项）
  environment:
    max_episode_steps: 500
    max_episode_seconds: 10000000  # 默认值，防止超时
    iterator_options:
      cycle: true  # 循环迭代episode
      shuffle: true  # 训练时打乱episode顺序（默认开启）

  # 数据集配置（补充habitat_test数据集默认参数，与预定义配置对齐）
  dataset:
    type: PointNav-v1  # 默认数据集类型，适配PointNav任务
    split: train  # 默认分割集（可在代码中动态修改为val/test）
    scenes_dir: data/scene_datasets  # 默认场景文件目录
    data_path: data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz  # 默认数据路径
    content_scenes: []  # 空表示加载所有场景，可指定具体场景名过滤

  # 任务配置（补充导航任务默认参数，适配奖励和成功判定）
  task:
    type: Nav-v0  # 默认导航任务类型
    physics_target_sps: 60.0  # 物理引擎更新频率（默认60Hz）
    reward_measure: distance_to_goal_reward  # 奖励指标（基于距离变化）
    success_measure: success  # 成功判定指标
    success_distance: 0.2  # 成功阈值（距离目标<0.2米视为成功，与代码判定一致）
    end_on_success: false  # 成功后不自动结束episode（需手动按F触发stop动作）
    slack_reward: -0.01  # 每步基础惩罚（防止无意义徘徊，默认值）

  # 模拟器配置（保留用户传感器分辨率，补充默认物理和代理参数）
  simulator:
    type: Sim-v0  # 默认模拟器类型
    forward_step_size: 0.25  # 前进步长（默认0.25米）
    turn_angle: 10.0  # 转向角度（默认10度/次）
    auto_sleep: false  # 禁用自动休眠（保证实时交互）
    step_physics: true  # 启用物理步进（导航任务必需）
    requires_textures: true  # 启用纹理渲染（保证RGB图正常显示）
    agents:
      main_agent:
        height: 1.5  # 代理高度（默认1.5米，符合人类视角）
        radius: 0.1  # 代理碰撞半径（默认0.1米）
        sim_sensors:
          # RGB传感器（用户定义256x256，补充默认视场角等参数）
          rgb_sensor:
            width: 256
            height: 256
            hfov: 90.0  # 水平视场角（默认90度）
            sensor_type: COLOR
            position: [0.0, 1.5, 0.0]  # 传感器位置（与代理高度一致）
          # 深度传感器（用户定义256x256，补充默认参数）
          depth_sensor:
            width: 256
            height: 256
            hfov: 90.0
            sensor_type: DEPTH
            position: [0.0, 1.5, 0.0]
            min_depth: 0.0  # 最小探测深度
            max_depth: 10.0  # 最大探测深度（默认10米）

  # 补充传感器细化配置（与lab_sensors对应，确保观测正常输出）
  task:
    lab_sensors:
      pointgoal_with_gps_compass_sensor:
        type: PointGoalWithGPSCompassSensor  # 目标点+GPS+罗盘集成传感器
      compass_sensor:
        type: EpisodicCompassSensor  # 罗盘传感器（输出与初始朝向的角度差）
      gps_sensor:
        type: EpisodicGPSSensor  # GPS传感器（输出与初始位置的坐标差）

  # 测量指标细化配置（确保奖励和成功判定逻辑正确）
  task:
    measurements:
      distance_to_goal:
        distance_to: POINT  # 测量到目标点的距离（默认）
      success:
        success_distance: 0.2  # 与任务配置一致，确保判定统一
      spl:
        success_distance: 0.2  # SPL计算依赖的成功阈值
```


**成功运行habitatlab_test.py并了解了基本的yaml配置原理后，可以跳过后面的部分，想要了解yaml文件配置导入代码的运行流程的可以参考后续内容，学习yaml文件导入代码的基本思路。**

### 2. yaml文件配置运行流程详解

**1） 获取配置文件并合并基础配置**

配置对象的构建通过get_config函数完成，该函数位于habitat-lab/habitat/config/default.py文件中。构建过程如下：

```python
def get_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    configs_dir: str = _HABITAT_CFG_DIR,
) -> DictConfig:
    r"""Returns habitat config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat.seed=1"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_HABITAT_CFG_DIR`).
    :return: composed config object.
    """
    register_configs()
    config_path = get_full_config_path(config_path, configs_dir)
    # If get_config is called from different threads, Hydra might
    # get initialized twice leading to issues. This lock fixes it.
    with lock, initialize_config_dir(
        version_base=None,
        config_dir=osp.dirname(config_path),
    ):
        cfg = compose(
            config_name=osp.basename(config_path),
            overrides=overrides if overrides is not None else [],
        )

    return patch_config(cfg)
```

1. 路径解析：首先使用get_full_config_path函数解析配置文件的绝对路径，该函数检查配置路径是否存在，如果不存在则在配置目录中查找。
2. 配置组合：使用compose函数组合配置，compose函数的参数包括配置名称（配置文件的基本名称）和覆盖项列表。
3. 配置修补：返回配置对象前，调用patch_config方法对配置进行内部修补，推断缺失的键并确保某些键存在且相互兼容。

**2）config传入Env类进行初始化**

在get_config函数后，在habitat-lab/habitat/core/env.py文件中，完成模拟器、任务和传感器的实例化。habitat.Env类是 Habitat Lab 环境的核心类，它集成了三个主要组件：数据集、模拟器和任务。

Env 类初始化流程详细步骤：

1. 构造函数参数处理：

```python
def __init__(self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None) -> None:
```
* config参数：环境配置，应包含模拟器 ID 和任务名称，这些将传递给make_sim和make_task函数​

* dataset参数：任务实例级信息的数据集引用，可以为None

2. 配置预处理：

```python
if "habitat" in config:
    config = config.habitat
self._config = config
```

如果配置中包含habitat键，则提取habitat子配置作为实际使用的配置。

3. 数据集初始化：

```python
self._dataset = dataset​
if self._dataset is None and config.dataset.type:​
    self._dataset = make_dataset(​
        id_dataset=config.dataset.type,​
        config=config.dataset​
    )
```
如果未提供数据集且配置中指定了数据集类型，则使用make_dataset创建数据集。

4. 当前 episode 和 episode 迭代器初始化：

```python
self._current_episode = None
self._episode_iterator = None
self._episode_from_iter_on_reset = True
self._episode_force_changed = False
```

5. episode 迭代器设置（如果有数据集）：

```python
if self._dataset:
    assert (
        len(self._dataset.episodes) > 0
    ), "dataset should have non-empty episodes list"
    self._setup_episode_iterator()
    self.current_episode = next(self.episode_iterator)
    with read_write(self._config):
        self._config.simulator.scene_dataset = (
            self.current_episode.scene_dataset_config
        )
        self._config.simulator.scene = self.current_episode.scene_id

    self.number_of_episodes = len(self.episodes)
else:
    self.number_of_episodes = None
```

* 调用_setup_episode_iterator方法设置 episode 迭代器
* 获取第一个 episode 并设置为当前 episode
* 更新配置中的场景数据集和场景 ID
* 设置 episode 总数

6. 模拟器初始化：

```python
self._sim = make_sim(
    id_sim=self._config.simulator.type,
    config=self._config.simulator
)
```

使用配置中的模拟器类型和配置初始化模拟器。

7. 任务初始化：

```python
self._task = make_task(
    self._config.task.type,
    config=self._config.task,
    sim=self._sim,
    dataset=self._dataset,
)
```

使用配置中的任务类型、任务配置、模拟器实例和数据集初始化任务。

8. 观察空间和动作空间设置：

```python
self.observation_space = spaces.Dict({
    **self._sim.sensor_suite.observation_spaces.spaces,
    **self._task.sensor_suite.observation_spaces.spaces,
})
self.action_space = self._task.action_space
```

观察空间由模拟器传感器套件和任务传感器套件的观察空间合并而成，动作空间来自任务。

9. 最大步数和时间限制设置：

```python
self._max_episode_seconds = self._config.environment.max_episode_seconds
self._max_episode_steps = self._config.environment.max_episode_steps
self._elapsed_steps = 0
self._episode_start_time: Optional[float] = None
self._episode_over = False
```

从配置中读取最大 episode 步数和时间限制，并初始化相关状态变量。

在上述过程中，配置驱动的组件实例化是 Habitat Lab 的核心机制，通过配置文件可以灵活地创建不同类型的环境。

上述代码中的模拟器实例化make_sim，任务实例化make_task，以及传感器实例化的代码和路径如下：

* 模拟器实例化 

(habitat-lab/habitat/sims/registration.py)

```python
def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(id_sim)
    return _sim(**kwargs)
```

* 任务实例化 

(habitat-lab/habitat/tasks/registration.py)

```python
def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(id_task)
    return _task(**kwargs)
```

* 传感器实例化 

(habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py)

传感器实例化没有函数，直接在habitat_simulator.py中定义，并设置habitat-sim环境。

```python
sim_sensors = []
for agent_config in self.habitat_config.agents.values():
    for sensor_cfg in agent_config.sim_sensors.values():
        sensor_type = registry.get_sensor(sensor_cfg.type)
        assert sensor_type is not None, "invalid sensor type {}".format(sensor_cfg.type)
        sim_sensors.append(sensor_type(sensor_cfg))
```

从配置中读取每个智能体的传感器配置​，然后使用registry.get_sensor获取传感器类​，最后使用传感器配置创建传感器实例。

**3） 依据配置设置habitat-sim环境**

在前面对于yaml文件的处理后，将最终得到的配置用来设置habitat-sim环境。

1. Simulator 配置结构

```python
habitat.simulator
├── type: "Sim-v0"  # 模拟器类型
├── scene_dataset: "data/scene_datasets/habitat-test-scenes"  # 场景数据集路径
├── scene: "van-gogh-room.glb"  # 当前场景
├── agents:  # 智能体配置
│   └── rgbd_agent:
│       ├── height: 1.5
│       ├── radius: 0.1
│       └── sim_sensors:  # 传感器配置
│           ├── rgb_sensor:
│           └── depth_sensor:
├── forward_step_size: 0.25  # 前进步长
└── turn_angle: 10  # 旋转角度
```

在HabitatSim类（habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py）的create_sim_config方法中，配置被转换为 Habitat-Sim 的内部配置格式。

```python
def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
    sim_config = habitat_sim.SimulatorConfiguration()
    
    # 从Habitat Lab配置复制到Habitat-Sim配置
    overwrite_config(
        config_from=self.habitat_config.habitat_sim_v0,
        config_to=sim_config,
        ignore_keys={"gpu_gpu"},
    )
    
    sim_config.scene_dataset_config_file = self.habitat_config.scene_dataset
    sim_config.scene_id = self.habitat_config.scene
    
    # 配置智能体
    agent_config = habitat_sim.AgentConfiguration()
    overwrite_config(
        config_from=lab_agent_config,
        config_to=agent_config,
        # 这些键只在Hab-Lab中使用，忽略
        ignore_keys={
            "is_set_start_state", "sensors", "sim_sensors", "start_position", 
            "start_rotation", "articulated_agent_urdf", "articulated_agent_type",
            ...
        },
    )
    
    # 配置传感器
    sensor_specifications = []
    for sensor in _sensor_suite.sensors.values():
        sim_sensor_cfg = sensor._get_default_spec()
        overwrite_config(
            config_from=sensor.config,
            config_to=sim_sensor_cfg,
            ignore_keys=sensor._config_ignore_keys,
            trans_dict={
                "sensor_model_type": lambda v: getattr(habitat_sim.FisheyeSensorModelType, v),
                "sensor_subtype": lambda v: getattr(habitat_sim.SensorSubType, v),
            },
        )
        sensor_specifications.append(sim_sensor_cfg)
    
    agent_config.sensor_specifications = sensor_specifications
    
    return habitat_sim.Configuration(sim_config, [agent_config])
```

2. Task 配置结构（PointNav 任务）

```python
habitat.task
├── type: "Nav-v0"  # 任务类型
├── reward_measure: "distance_to_goal_reward"  # 奖励度量
├── success_measure: "spl"  # 成功度量
├── success_reward: 2.5  # 成功奖励
├── slack_reward: -0.01  # 失败惩罚
├── end_on_success: true  # 是否在成功时结束
└── lab_sensors:  # 实验室传感器
    └── pointgoal_with_gps_compass_sensor:
        ├── type: "PointGoalWithGPSCompassSensor"
        └── goal_format: "POLAR"
```

在NavigationTask类（habitat-lab/habitat/tasks/nav/nav.py）的初始化过程中，配置被用于设置任务的各种属性：

```python
@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [
                    float(k) for k in episode.start_rotation
                ]
                agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
```

3. Agent 配置结构

```python
habitat.simulator.agents.rgbd_agent
├── height: 1.5  # 智能体高度
├── radius: 0.1  # 智能体半径
├── sim_sensors:  # 传感器配置
│   ├── rgb_sensor:
│   │   ├── type: "HabitatSimRGBSensor"
│   │   ├── width: 256
│   │   └── height: 256
│   └── depth_sensor:
│       ├── type: "HabitatSimDepthSensor"
│       ├── width: 256
│       └── height: 256
├── start_position: [0.0, 0.0, 0.0]  # 初始位置
└── start_rotation: [0.0, 0.0, 0.0, 1.0]  # 初始旋转（四元数）
```

在HabitatSim类（habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py）的create_sim_config方法中，智能体配置被转换为 Habitat-Sim 的内部格式：

```python
def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
    # ... 其他配置处理 ...
    
    # 配置智能体
    agent_config = habitat_sim.AgentConfiguration()
    
    # 从Habitat Lab配置复制到Habitat-Sim配置
    overwrite_config(
        config_from=lab_agent_config,
        config_to=agent_config,
        ignore_keys={
            "is_set_start_state", "sensors", "sim_sensors", "start_position", 
            "start_rotation", "articulated_agent_urdf", "articulated_agent_type",
            ...
        },
    )
    
    agent_config.sensor_specifications = sensor_specifications

    agent_config.action_space = {
        0: habitat_sim.ActionSpec("stop"),
        1: habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.forward_step_size
            ),
        ),
        2: habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.turn_angle
            ),
        ),
        3: habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.turn_angle
            ),
        ),
    }
    
    # ... 传感器配置处理 ...
    
    return habitat_sim.Configuration(sim_config, [agent_config])
```

4. Sensor 配置结构

```python
habitat.simulator.agents.rgbd_agent.sim_sensors.rgb_sensor
├── type: "HabitatSimRGBSensor"  # 传感器类型
├── width: 256  # 图像宽度
├── height: 256  # 图像高度
├── position: [0.0, 1.25, 0.0]  # 传感器位置
├── orientation: [0.0, 0.0, 0.0]  # 传感器方向
└── hfov: 90  # 水平视场角
```

在传感器类的实现中（habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py），配置被用于设置传感器的属性：

```python
@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self._width = config.width
        self._height = config.height
        self._hfov = config.hfov
        self._position = config.position
        self._orientation = config.orientation
```

在HabitatSim类（habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py）的传感器初始化过程中，传感器配置被转换为 Habitat-Sim 的传感器规范：

```python
for sensor in _sensor_suite.sensors.values():
    sim_sensor_cfg = sensor._get_default_spec()
    
    # 从Habitat Lab配置复制到Habitat-Sim传感器配置
    overwrite_config(
        config_from=sensor.config,
        config_to=sim_sensor_cfg,
        ignore_keys=sensor._config_ignore_keys,
        trans_dict={
            "sensor_model_type": lambda v: getattr(habitat_sim.FisheyeSensorModelType, v),
            "sensor_subtype": lambda v: getattr(habitat_sim.SensorSubType, v),
        },
    )
    
    sim_sensor_cfg.uuid = sensor.uuid
    sim_sensor_cfg.resolution = [sensor.observation_space.shape[1], sensor.observation_space.shape[0]]
    sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
    sensor_specifications.append(sim_sensor_cfg)
```

### habitat-lab的基础配置及其在代码中的运行过程介绍完毕，可以看出，habitat-lab的配置系统使得开发者可以灵活地创建各种复杂的具身 AI 环境，实现从简单导航到复杂操作任务的多样化应用。

参考资料：

https://github.com/facebookresearch/habitat-lab
