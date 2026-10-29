#!/usr/bin/env python3
"""Add a visible, notebook-native run control cell to the three model notebooks."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = (
    "14_smolvla_end_to_end.ipynb",
    "15_pi0_end_to_end.ipynb",
    "16_act_end_to_end.ipynb",
)

RUN_CONTROL_MARKER = "# RUN_CONTROL_CELL"
RUN_CONTROL_MARKDOWN = (
    "## 运行控制台：先在这里选择本次实验\n\n"
    "下面这个单元格是三个模型通用的开关。长训默认打开，Smoke、评估和 protected 续训默认关闭；这样 Notebook 默认会展示真实长训过程，但不会同时启动多条训练或评估任务。\n\n"
    "建议顺序：\n\n"
    "1. 第一次把 `RUN_SMOKE` 设为 `True`，确认环境、数据、模型和一次反向传播都正常。\n"
    "2. 确认无误后把 `RUN_LONG_TRAIN` 设为 `True`，执行后面的真实训练单元格，Notebook 会显示实时 `tqdm`、loss、耗时和 checkpoint。\n"
    "3. 训练完成后把 `RUN_EVAL` 设为 `True`，执行严格评估单元格，结果会汇总为 `x/y` 成功率，并在有渲染条件时保存视频。\n"
    "4. 如果要复现正式保护结果，再把 `RUN_PROTECTED_TRAIN` 设为 `True`，执行对应 protected recipe 单元格。它和普通教学长训是两条明确标注的训练谱系。\n\n"
    "训练循环和评估循环都在 Notebook Python kernel 内执行，不是 `cat` 静态日志，也不是把训练交给外部 shell 脚本。"
)
RUN_CONTROL_CODE = (
    f"{RUN_CONTROL_MARKER}\n"
    "# 只修改下面四个布尔值，然后按顺序执行后面的单元格。\n"
    "RUN_SMOKE = False\n"
    "RUN_LONG_TRAIN = True\n"
    "RUN_EVAL = False\n"
    "RUN_PROTECTED_TRAIN = False\n\n"
    "print({\n"
    "    'RUN_SMOKE': RUN_SMOKE,\n"
    "    'RUN_LONG_TRAIN': RUN_LONG_TRAIN,\n"
    "    'RUN_EVAL': RUN_EVAL,\n"
    "    'RUN_PROTECTED_TRAIN': RUN_PROTECTED_TRAIN,\n"
    "})\n"
    "print('长训通常需要几十分钟到数小时；评估会逐 episode 闭环运行，时间更长。')\n"
)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


for filename in NOTEBOOKS:
    path = NOTEBOOK_DIR / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    already_patched = any(RUN_CONTROL_MARKER in "".join(cell.get("source", [])) for cell in notebook["cells"])
    if not already_patched:
        notebook["cells"][3:3] = [
            markdown_cell(RUN_CONTROL_MARKDOWN),
            code_cell(RUN_CONTROL_CODE),
        ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if RUN_CONTROL_MARKER in source:
            source = source.replace("RUN_LONG_TRAIN = False", "RUN_LONG_TRAIN = True")
        source = source.replace(
            'RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN")',
            'RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN", True)',
        )
        if "默认全部关闭" in source:
            source = source.replace(
                "默认全部关闭，避免打开 Notebook 后误启动数小时训练。",
                "长训默认打开，Smoke、评估和 protected 续训默认关闭；不会同时启动多条任务。",
            )
        source = source.replace(
            'protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)',
            'protected_train_enabled = globals().get("RUN_PROTECTED_TRAIN", env_flag("RUN_PROTECTED_TRAIN", False))',
        )
        cell["source"] = source
        if "train_lerobot_config_in_notebook(" in source and not source.lstrip().startswith("def train_lerobot_config_in_notebook"):
            # Remove stale "未启动" output. The next real execution must produce
            # the progress bar and metrics in this cell rather than showing a
            # misleading snapshot from the previous default.
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("patched:", path, "(existing control updated)" if already_patched else "")
