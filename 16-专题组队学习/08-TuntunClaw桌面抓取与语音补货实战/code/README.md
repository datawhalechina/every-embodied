# TuntunClaw（囤囤钳）开源代码：1:1 MuJoCo 仿真 + 真机闭环 + Magic Box 语音

本目录是《TuntunClaw 桌面抓取、低延迟语音与自动补货实战》教程的可直接运行开源代码包，完整包含 **1:1 MuJoCo 物理仿真**、**myCobot 280 Pi 真机闭环抓取与回收**、**地瓜 RDK X5 Magic Box 低延迟语音与灯效守护进程**、**FastAPI + SQLite 补货后台**以及 **Web 小程序 / 微信小程序壳工程**。

无任何机械臂或开发板硬件时，可直接在普通 PC 上以 `simulation` 模式完整运行 1:1 物理仿真、顶视视觉分割、三策略抓取对比、自动回收和网页端补货台。

## 目录结构

```text
code/
├── requirements.txt                # Python 依赖清单
├── .env.example                    # 环境变量模板（请勿提交真实密码/私有 IP）
├── scripts/
│   └── verify_sim_pipeline.py      # 无硬件一键回归验证（扫描/三策略抓放/回收/IK 检查）
├── sim/
│   ├── build_scene.py              # 生成 1:1 展台 MuJoCo XML 场景
│   ├── mycobot_280_scene.xml       # myCobot 280 Pi + 相机法兰 + 自适应夹爪 + 双托盘场景
│   ├── sim_engine.py               # 1:1 物理引擎、三策略抓取、双同步镜像与单线程 GL 渲染
│   ├── tray_vision.py              # 仿真与真机共用顶视红盘分割、短轴二分与 PCA 朝向解算
│   ├── workspace_analysis.py       # 6-DOF 顶视 IK 与可达空间分析
│   └── assets/mycobot_280/stl/     # 16 个官方 myCobot 280 Pi + 夹爪 STL 网格
├── hardware/
│   ├── real_mycobot_bridge.py      # myCobot 280 Pi SSH/串口安全桥（关节限位 + 腕部抓拍）
│   ├── real_pick_controller.py     # 真机闭环抓取/回收控制器（底座防倾倒看门狗 + 防后翻肩 IK）
│   ├── top_camera_grabber.py       # 顶视 USB 相机按需抓拍（空闲自动释放 + 多服务共享）
│   ├── web_align_server.py         # 独立双摄 1:1 摆区校准页（默认端口 8091）
│   ├── magicbox_voice.py           # 主机端 TTS 合成、小喇叭清亮 DSP 增强与热缓存调度
│   └── magicbox_voice_daemon.py    # 运行在 RDK X5 Magic Box 上的常驻 HTTP 语音/灯环服务
├── backend/
│   ├── database.py                 # SQLite 实时库存、低库存自动补货清单与事件流水
│   └── server.py                   # FastAPI + SSE + WebSocket 统一服务（默认端口 8090）
├── web/
│   ├── index.html                  # 移动端优先 Web 小程序（多视角直播 + 点单 + 补货清单）
│   ├── align.html                  # 1:1 实拍/仿真摆区校准视图
│   ├── styles.css
│   └── app.js
└── wechat-miniprogram/             # 微信开发者工具原生封装工程
```

## 快速启动

### 1. 安装依赖

```bash
cd code
python -m pip install -r requirements.txt
```

### 2. 无硬件 1:1 仿真一键验收

```bash
python scripts/verify_sim_pipeline.py
```

该脚本不依赖任何真机或摄像头，自动执行 1:1 顶视视觉扫描、`baseline / oriented / servo_closed_loop` 三策略抓放、核桃木盘自动回收以及真机放料位前屈肩 IK 安全校验。

### 3. 启动 Web 控制台（默认安全运行在 `simulation` 模式）

```bash
cp .env.example .env
python backend/server.py
```

启动后在浏览器打开 `http://127.0.0.1:8090/` 即可体验多视角 3D 孪生、扫描盘点、点单取送、自动回收与补货清单。

### 4. 接入真机与 Magic Box（可选）

1. 在 `.env` 或环境变量中配置 `MYCOBOT_HOST`、`MYCOBOT_USER`（推荐配置 SSH 免密公钥）、`MAGICBOX_HOST` 与 `TOP_CAM_INDEX`。
2. 将 `hardware/magicbox_voice_daemon.py` 上传至 RDK X5 Magic Box 并后台常驻（监听 `8095` 端口）。
3. 在网页控制台右上角将运行模式切换为 `dual_sync`（真实 + 仿真双同步）或 `real_hardware`（真机闭环），先点击「回安全位」确认底座锚点自检通过后再执行抓取。

## 上游资产许可说明

`sim/assets/mycobot_280/stl/` 中的机械臂与夹爪几何网格转换自 Elephant Robotics 官方开源仓库 [mycobot_ros](https://github.com/elephantrobotics/mycobot_ros)（BSD-3-Clause），其余仿真、视觉、控制器与前后端代码均按本仓库主许可开放学习与复用。
