# 04 圆屏、语音与官方 Skill 接口

Aura 适合承担人与执行器之间的交互入口：接收意图、找人转向、显示任务、播放结果、展示机器人相机。本章把这些能力拆成可测试的输出通路，再封装成 Skill。

## 4.1 圆屏显示任务与反馈

建议状态为 `ready → listening → locating_person → delegating → executing → reporting`，每个状态有简短文案和可见图标。执行中嵌入 R1 最新相机缩略图，并附采样时间；相机失联时显示上次采样时间与等待状态。

显示状态由任务事件更新，不能只播放一段固定动画代表任务进度。圆形屏幕要裁剪在有效区域内，避免把长行文字贴到圆边。普通网页使用易读正文字体；本专题视频中的标题使用手写风格并保留圆角，视频画面不靠裁剪来遮住相机边缘。

## 4.2 语音输入与输出分开验收

输入链为麦克风 PCM → 音量与格式检查 → ASR → 规范化意图；输出链为报告文本 → TTS → 音频文件/扬声器 → 实际听觉检查。WebSocket 的 `say` 在本样机协议中可能没有完成确认；工具调用成功不能直接等同于扬声器已播完。

我们在真机麦克风诊断中观察过文件有长度却接近无声的情况，采集和识别要分别排查。家庭仿真案例采用合成的自然人声作为输入，再实际执行 Whisper 解码；Aura 结果报告同样为合成音轨。它验证语音意图到仿真交接，真机麦克风声学验收另记。

自然语音可接合适的本地或云端 TTS；选型需要比较中文清晰度、首包延迟、授权方式与音色许可。代码接口传递 `text / audio_path / duration / timestamp`，避免把某家服务凭据写进 Skill。正文与录像都应说明配音来源。

## 4.3 官方技能包的结构

本课程附带 [`aura-lab-guide`](../code/skills/aura-lab-guide)，与实际设备应用、电脑侧模型分开。包结构如下：

```text
skills/aura-lab-guide/
  MANIFEST.yaml
  skill.md
  main.py
  README.md
```

`MANIFEST.yaml` 声明 ID、版本、权限和 Python 入口；`skill.md` 包含双语标题、描述、分类与说明；`main.py` 提供异步 `run(context, task)`。目录名与 ID 保持一致。

以向导技能为例，只声明 `say`，不申请相机、运动或网络权限。这里 `context.say()` 是官方上下文中的聊天工具语义，不能据其名称推断一定驱动实体扬声器。

```yaml
permissions:
  tools: [say]
  network:
    domains: []
  filesystem:
    scope: skill
  maxRuntimeSec: 5
entry:
  type: python
  main: main.py
  function: run
```

## 4.4 静态校验、入口测试、宿主测试

官方开发规范和校验器：[SPEC](https://github.com/FrontierX-AuraOS/AuraOS-skill-store/blob/main/SPEC.md)、[Developer Guide](https://github.com/FrontierX-AuraOS/AuraOS-skill-store/blob/main/DEVELOPER_GUIDE.md)。本地历史核验对应 `86aa4da7249f75f671e03717942d13c262e2aefb`；仓库主分支会继续变化。

准备该版本的上游源码后，在 `code` 目录执行：

```bash
python vendor/AuraOS-skill-store/scripts/validate-skill.py skills/aura-lab-guide
```

上游不在本课程代码中重复分发，`vendor` 是读者自己的外部源码目录。校验器通过后，用一个只记录 `say` 的测试 context 调用四个入口分支，检查参数和返回值。最后在真实宿主中检查加载、权限、超时和输出。静态检查、测试 context 与真机安装分别记录结果。

## 4.5 把识别结果接入 Skill

将第三章产生的事件压缩为有限结构：人物方向、举手状态、卡片结果、时间和事件 ID。技能只根据这些字段组织交互，模型重计算留在电脑/工作站。网络结果服务应限制允许域名、超时和响应大小；图像中的文字不得覆盖系统指令。

每项任务交接生成唯一 `task_id`，让 Aura 的界面与执行器回执引用同一个 ID。结果报告先查看验收字段再生成文案，例如三罐均在桶内且释放稳定后才播报“清理完成”。执行失败则显示实际失败阶段，而不是继续播放成功动画。

## 4.6 作业

实现任务开始、执行中、失败、成功四种圆屏状态，分别用一个可检查事件触发。提交 Skill 静态校验、入口测试、屏幕截图和一段带声音的过程录像。真机与仿真各用独立列标注，避免把电脑播放音轨当作真机扬声器验收。
