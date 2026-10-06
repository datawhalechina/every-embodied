# Astra：组织上下文与结构化演奏计划

Astra 适合在演奏前整理任务、输出计划、根据运行报告分析问题。实时按键则由本地控制器执行。

## 离线准备

本章的上下文工具使用标准库，先生成请求和示例计划：

```bash
python piano_context.py prepare --out runs/context
python piano_context.py verify --plan example.plan.json
```

输入包括 [task.json](task.json)、[plan.schema.json](plan.schema.json) 和示例计划。可用 `--image` 加入场景截图。`--feedback` 接收 JSON 数组，每条记录可含 `time_seconds`、`expected_midi`、`observed_midi`、`outcome`、`error`；不能直接传入整份对象格式的演奏报告。

## 可选在线调用

`piano_context.py online --run runs/context` 会发起真实 API 请求并产生费用，调用前设置自己的凭据。脚本的请求模型名为 `gpt-6-astra`；请先确认所用端点对该模型开放，不要把桌面客户端订阅当作 API 权限。

在线调用需另装 `openai` Python 包，并设置 `OPENAI_API_KEY`；使用自有兼容端点时按其说明设置 `OPENAI_BASE_URL`。密钥不写入仓库，返回解析见 [piano_context.py](piano_context.py)。本章随附样例可离线运行，模型接入不是启动仿真的前置条件。

生成的计划必须通过字段、时间、指法和可达范围检查，再运行仿真。曲目旋律以用户提供或获授权的曲谱为准，不凭曲名让模型猜音符。

## 三层协作

- Astra：准备上下文与候选计划，阅读运行反馈。
- G0.5：根据实际观测给出双臂调整提案。
- 本地演奏控制器：执行双手指法、和弦、琴键接触与音频生成。

当前开源主流程可以直接读取已准备的 JSON 练习，不需要在线调用 Astra。
