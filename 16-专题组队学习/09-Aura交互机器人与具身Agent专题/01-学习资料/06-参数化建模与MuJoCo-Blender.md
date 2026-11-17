# 06 参数化模型、MuJoCo 物理与 Blender 展示

目标是让同一台 Aura 的几何、运动和圆屏数据在两个工具中保持一致。MuJoCo 负责可检查的运动状态，Blender 用该轨迹制作展示；高保真家庭中的任务执行另在 OmniGibson / PhysX 中运行。

## 6.1 模型的精度如何描述

产品公开整体尺寸约 21 × 20 × 20 cm。本项目根据照片和整体尺寸建立参数化外形，生成包围盒约 204.03 × 210.00 × 200.45 mm。分件尺寸、真实质量、惯量、轮胎摩擦、屏幕有效直径、倾角与相机标定尚未得到工程图确认，因此称为“照片与尺寸约束的参数化重建”。

参数源在 [`model_parameters.py`](../code/digital_twin/model_parameters.py)：CAD 使用 mm，仿真输出明确换算成 m。修改轮半径、相机位置或前脸倾角，重新生成所有模型；不要分别在两个软件里拖动网格造成比例分叉。

## 6.2 已附带的模型与文件

```text
code/digital_twin/
  model_parameters.py    # 参数与未测项目
  aura_twin.py           # build123d 建模源
  model_manifest.json    # 生成几何的事实记录
  assets/meshes/         # 本项目生成的网格
  aura.xml               # MuJoCo 模型
  build_mujoco.py        # 从网格生成 MJCF
  run_mujoco.py          # 有限物理运行与圆屏纹理
  build_blender.py       # 从轨迹生成 Blender 场景
```

课程分发自建网格和自制圆屏测试状态，不包含第三方家居模型、人物资产和设备原厂屏幕视频。自制屏幕用于检查 UV 与逐帧显示；需要设备原屏时，在获得授权后自行替换纹理源，并记录来源。

## 6.3 运行轻量物理实验

在 `code` 目录，建议单独的数字孪生环境，不改训练环境：

```bash
python -m venv .venv-twin
source .venv-twin/bin/activate
# Windows 改用 .venv-twin/Scripts/Activate.ps1
python -m pip install -r digital_twin/requirements.txt
python digital_twin/build_mujoco.py
python digital_twin/run_mujoco.py --output digital_twin/outputs/my-first-run --no-video
```

运行脚本仍需要可用的渲染上下文；无显示的 Linux 可按 MuJoCo 环境配置 EGL，再运行。输出目录必须不存在，避免覆盖旧结果。检查轨迹是否有限、是否出现数值警告、轮角是否改变、位置与偏航是否响应控制。

包中还提供不启动渲染器的结构检查：

```bash
python tools/course_lab.py twin --root .
```

这条命令检查资源路径和单位，不代替物理动力学验收。历史物理试验执行 2501 步、约 5 秒，前进约 0.07052 m，偏航约 0.3417 rad；这是模型响应记录，不是对真机动力学的拟合精度。

## 6.4 圆屏是真的在更新吗

屏幕采用带 UV 的圆形网格。先在机器人静止时切换 `hello/listening/thinking` 纹理，比较屏幕区域像素；再在运动时按统一时间轴更新纹理。历史原屏试验观察到约 4.3 万变化像素。公开版本使用自制 UI 重新运行了 2501 步，前进约 0.07052 m、无数值警告，固定姿态切换两种自制状态得到 4180 个变化像素；两次输入纹理不同，分别保留结果。

眼睛、任务文本、相机缩略图都必须进入纹理后再渲染，而不是仅在最终视频外贴一块矩形。相机纹理和状态文本应有帧 ID；屏幕显示“完成”来自验收事件，不由剪辑时间预先决定。

## 6.5 Blender 使用同一条实走轨迹

```bash
blender --background --factory-startup --python digital_twin/build_blender.py -- \
  --trajectory digital_twin/outputs/my-first-run/trajectory.json \
  --output digital_twin/outputs/my-blender --preview-animation
```

`bpy` 随 Blender 程序运行，不用普通 Python 安装一个同名包代替。该脚本从 MuJoCo 的实际轨迹读取位置、旋转、轮角和屏幕时间，Blender 动画是该轨迹的展示。要验证碰撞，以物理执行记录为依据；渲染画面本身不证明碰撞求解成功。

## 6.6 高保真家庭与资源

高保真人物和家庭可以接 Isaac 资产、合法授权的家居 USD、BEHAVIOR 场景。资产体量、显存和许可证各自检查，资源不放进教程 Git。完整家庭任务使用 OmniGibson 3.9.1 / Isaac Sim 5.1.0 历史环境，不能随意把一个新版本 Kit 环境覆盖到现有任务环境。

工作站负责重渲染，电脑只查看浏览器与视频。需要新渲染时创建独立输出目录、记录本任务 GPU 占用，不停止其他作业。当前五路完整视频已生成，学习视频与代码审查无需启动 GPU。

作业：修改一个外观参数，比较生成前后的几何记录；在同一运动轨迹上导出 MuJoCo 与 Blender 的前/侧/圆屏对照，并说明哪些参数仍未标定。
