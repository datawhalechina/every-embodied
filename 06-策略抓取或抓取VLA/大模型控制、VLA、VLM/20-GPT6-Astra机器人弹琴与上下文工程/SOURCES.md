# 资源来源与开源范围

本章自有代码与教程沿用仓库根目录 CC BY 4.0 许可证。第三方模型、机器人资源、音色与字体分别遵守其自身许可证。

| 资源 | 来源与处理 |
| --- | --- |
| R1 Pro | [GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim/tree/abe7f5161eeaa150e6eaffdf443af5df7f23f356)，回放包保留 Apache-2.0 LICENSE 与 NOTICE |
| Shadow Hand | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)，Apache-2.0 通知保存在回放包 |
| G1 扩展练习 | [固定来源提交](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/unitree_g1)，保留 Unitree BSD-3-Clause 许可证 |
| 钢琴环境 | [RoboPianist](https://github.com/google-research/robopianist)，按 requirements-sim.txt 安装 |
| G0.5 | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 与 [模型发布页](https://huggingface.co/OpenGalaxea/G05)，用户按上游条款自行下载，权重不纳入本仓库 |
| PianoMime 扩展 | [PianoMime](https://github.com/sNiper-Qian/pianomime)，本章仅提供适配入口，权重与数据从上游获取 |
| 钢琴音色 | RoboPianist 所用 TimGM6mb.sf2，按原项目获取，不重复打包音色文件 |

## 随附资源

- `assets/galaxea-replay.zip`：R1 Pro 双手入门回放，包含场景、模型网格、初始化状态、执行器控制、计划与报告。
- `assets/whole-robot-replay.zip`：G1 扩展回放，也是已有物理回放检查所用样本。
- `assets/relocation-before.png`、`relocation-after.png`、`relocation-recovery.mp4`：本项目 6 厘米移琴就位实验的画面，视频无声。
- `assets/relocation-static-results.json`：历史静态移琴对照结果。
- `assets/relocation-session-report.json`：历史连续移琴续弹结果。
- `assets/public-relocation-demo.mp4`、`public-relocation-final.png`、`public-relocation-report.json`：本次公开练习的连续移琴续弹视频、截图和完整结果。
- `assets/public-physical-report.json`：公开练习固定琴位演奏报告。
- `relocation-exercise.json`：本章公开的和弦与音阶调试练习。

机器人仿真改动包括固定底盘、理想重力补偿、双 Shadow Hand 安装适配、接触控制、琴键阻尼与灯光。回放包内许可证位于 `scene/LICENSE-*.txt`，来源通知位于 `scene/NOTICE*.txt`。

宣传成片中的歌曲录音、完整歌曲曲谱、未确认再分发许可的 Isaac Sim 贴图、第三方视频素材，以及本地密钥均不纳入发布。教程代码可用于自己的授权素材。不要把本仓库许可证套用到这些第三方资源上。

## 脚本导航

| 类别 | 入口 |
| --- | --- |
| 本章主线 | polyphonic_piano.py、g05_relocation_session.py |
| 模型推理与就位 | g05_piano_client.py、g05_piano_worker.py、g05_recovery.py |
| 机器人与回放 | prepare_galaxea.py、galaxea_piano.py、replay_windows.py |
| 曲谱处理 | midi_score.py、bimanual_fingering.py、simplify_performance.py |
| 接触与声音复核 | audit_polyphonic_contacts.py、piano_events.py |
| 可选扩展 | physical_piano.py、whole_robot_piano.py、pianomime_runner.py |
| 回归检查 | test_*.py |

`runs/` 是每台机器自己的输出目录，默认忽略；每次运行使用新的目录保存结果。
